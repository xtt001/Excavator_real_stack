"""Near-home feedback controller for real excavator recording sessions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from testbed.backends.real.contracts import REAL_ACTION_DIM, as_real_vector4


def _vector4(value: Any, *, name: str, default: Sequence[float] | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        value = default
    if isinstance(value, (int, float)):
        value = [float(value)] * REAL_ACTION_DIM
    return as_real_vector4(value, name=name)


def _positive_vector4(
    value: Any,
    *,
    name: str,
    default: Sequence[float],
) -> np.ndarray:
    arr = _vector4(value, name=name, default=default)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} values must be positive")
    return arr


@dataclass(frozen=True)
class GoHomeConfig:
    """Configuration for safe near-home qpos feedback control."""

    home_pose_rad: np.ndarray
    near_tolerance_rad: np.ndarray
    success_tolerance_rad: np.ndarray
    p_gain: np.ndarray
    max_action: np.ndarray
    qvel_stable_rad_s: np.ndarray
    dwell_s: float = 0.4
    timeout_s: float = 8.0
    runaway_error_factor: float = 1.5

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "GoHomeConfig | None":
        raw = dict(cfg or {})
        if not bool(raw.get("enabled", False)):
            return None
        if raw.get("home_pose_rad") is None:
            raise ValueError("teleop.recording.go_home.home_pose_rad must be set")
        home = _vector4(raw.get("home_pose_rad"), name="home_pose_rad")
        near = _positive_vector4(
            raw.get("near_tolerance_rad"),
            name="near_tolerance_rad",
            default=[0.20] * REAL_ACTION_DIM,
        )
        success = _positive_vector4(
            raw.get("success_tolerance_rad"),
            name="success_tolerance_rad",
            default=[0.03] * REAL_ACTION_DIM,
        )
        p_gain = _positive_vector4(
            raw.get("p_gain"),
            name="p_gain",
            default=[1.2] * REAL_ACTION_DIM,
        )
        max_action = _positive_vector4(
            raw.get("max_action"),
            name="max_action",
            default=[0.20] * REAL_ACTION_DIM,
        )
        qvel_stable = _positive_vector4(
            raw.get("qvel_stable_rad_s"),
            name="qvel_stable_rad_s",
            default=[0.02] * REAL_ACTION_DIM,
        )
        dwell_s = float(raw.get("dwell_s", 0.4))
        timeout_s = float(raw.get("timeout_s", 8.0))
        if dwell_s < 0.0:
            raise ValueError("dwell_s must be >= 0")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        return cls(
            home_pose_rad=home,
            near_tolerance_rad=near,
            success_tolerance_rad=success,
            p_gain=p_gain,
            max_action=max_action,
            qvel_stable_rad_s=qvel_stable,
            dwell_s=dwell_s,
            timeout_s=timeout_s,
            runaway_error_factor=float(raw.get("runaway_error_factor", 1.5)),
        )


@dataclass(frozen=True)
class GoHomeResult:
    """One controller update result."""

    action: np.ndarray
    done: bool = False
    failed: bool = False
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


class GoHomeController:
    """Run a bounded P controller from near-home qpos to the configured home pose."""

    def __init__(self, config: GoHomeConfig) -> None:
        self.config = config
        self.started = False
        self.start_time_ns = 0
        self.end_time_ns = 0
        self._start_s = 0.0
        self._stable_since_s: float | None = None
        self.start_qpos = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self.final_qpos = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self.final_error = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self.result_code = ""
        self.failed_reason = ""

    def start(self, obs: Mapping[str, Any], *, now_ns: int | None = None) -> None:
        qpos = _obs_qpos(obs)
        error = self.config.home_pose_rad - qpos
        if np.any(np.abs(error) > self.config.near_tolerance_rad):
            raise ValueError(
                "current pose is outside go-home near_tolerance_rad: "
                + _format_vector(error)
            )
        self.started = True
        self.start_time_ns = int(now_ns or time.time_ns())
        self.end_time_ns = 0
        self._start_s = time.monotonic()
        self._stable_since_s = None
        self.start_qpos = qpos.astype(np.float32, copy=True)
        self.final_qpos = qpos.astype(np.float32, copy=True)
        self.final_error = error.astype(np.float32, copy=True)
        self.result_code = "running"
        self.failed_reason = ""

    def update(self, obs: Mapping[str, Any], *, now_s: float | None = None) -> GoHomeResult:
        if not self.started:
            return self._fail("not_started", obs)
        now = time.monotonic() if now_s is None else float(now_s)
        try:
            qpos = _obs_qpos(obs)
            qvel = _obs_qvel(obs)
        except ValueError as exc:
            return self._fail(f"feedback_invalid:{exc}", obs)

        error = self.config.home_pose_rad - qpos
        self.final_qpos = qpos.astype(np.float32, copy=True)
        self.final_error = error.astype(np.float32, copy=True)

        elapsed = max(0.0, now - self._start_s)
        if elapsed > self.config.timeout_s:
            return self._fail("timeout", obs)

        runaway_limit = self.config.near_tolerance_rad * self.config.runaway_error_factor
        if np.any(np.abs(error) > runaway_limit):
            return self._fail("runaway_error", obs)

        in_position = np.all(np.abs(error) <= self.config.success_tolerance_rad)
        stable_velocity = np.all(np.abs(qvel) <= self.config.qvel_stable_rad_s)
        if in_position and stable_velocity:
            if self._stable_since_s is None:
                self._stable_since_s = now
            if now - self._stable_since_s >= self.config.dwell_s:
                return self._finish("succeeded", done=True)
            action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        else:
            self._stable_since_s = None
            action = self.config.p_gain * error
            action = np.clip(action, -self.config.max_action, self.config.max_action)

        return GoHomeResult(
            action=np.asarray(action, dtype=np.float32),
            diagnostics=self._diagnostics(
                qpos=qpos,
                qvel=qvel,
                action=action,
                in_position=in_position,
                stable_velocity=stable_velocity,
                elapsed_s=elapsed,
            ),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "go_home_result": str(self.result_code),
            "go_home_failed_reason": str(self.failed_reason),
            "go_home_start_time_ns": int(self.start_time_ns),
            "go_home_end_time_ns": int(self.end_time_ns),
            "go_home_target_pose_rad": self.config.home_pose_rad.astype(np.float32),
            "go_home_start_qpos": self.start_qpos.astype(np.float32),
            "go_home_final_qpos": self.final_qpos.astype(np.float32),
            "go_home_final_error": self.final_error.astype(np.float32),
        }

    def _finish(self, result_code: str, *, done: bool = False, failed: bool = False) -> GoHomeResult:
        self.result_code = result_code
        self.end_time_ns = int(time.time_ns())
        self.started = False
        return GoHomeResult(
            action=np.zeros(REAL_ACTION_DIM, dtype=np.float32),
            done=done,
            failed=failed,
            reason=self.failed_reason or result_code,
            diagnostics={
                "go_home_running": 0,
                "go_home_result_code": result_code,
                "go_home_error": self.final_error.astype(np.float32),
                "go_home_commanded_action": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
            },
        )

    def _fail(self, reason: str, obs: Mapping[str, Any]) -> GoHomeResult:
        try:
            qpos = _obs_qpos(obs)
            self.final_qpos = qpos.astype(np.float32, copy=True)
            self.final_error = (self.config.home_pose_rad - qpos).astype(np.float32)
        except ValueError:
            pass
        self.failed_reason = str(reason)
        return self._finish("failed", failed=True)

    def _diagnostics(
        self,
        *,
        qpos: np.ndarray,
        qvel: np.ndarray,
        action: np.ndarray,
        in_position: bool,
        stable_velocity: bool,
        elapsed_s: float,
    ) -> dict[str, Any]:
        return {
            "go_home_running": 1,
            "go_home_result_code": "running",
            "go_home_error": (self.config.home_pose_rad - qpos).astype(np.float32),
            "go_home_qvel": np.asarray(qvel, dtype=np.float32),
            "go_home_commanded_action": np.asarray(action, dtype=np.float32),
            "go_home_in_position": int(bool(in_position)),
            "go_home_stable_velocity": int(bool(stable_velocity)),
            "go_home_elapsed_s": float(elapsed_s),
        }


def _obs_qpos(obs: Mapping[str, Any]) -> np.ndarray:
    if "qpos" not in obs:
        raise ValueError("missing qpos")
    return as_real_vector4(obs["qpos"], name="qpos")


def _obs_qvel(obs: Mapping[str, Any]) -> np.ndarray:
    if "qvel" not in obs:
        raise ValueError("missing qvel")
    return as_real_vector4(obs["qvel"], name="qvel")


def _format_vector(value: Any) -> str:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{float(v):+.4f}" for v in arr) + "]"
