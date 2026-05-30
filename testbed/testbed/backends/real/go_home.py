"""Near-home feedback controller for real excavator recording sessions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from testbed.backends.real.contracts import (
    REAL_ACTION_DIM,
    REAL_ACTION_ORDER,
    as_real_vector4,
)


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


def _nonnegative_vector4(
    value: Any,
    *,
    name: str,
    default: Sequence[float],
) -> np.ndarray:
    arr = _vector4(value, name=name, default=default)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} values must be non-negative")
    return arr


def _sign_vector4(
    value: Any,
    *,
    name: str,
    default: Sequence[float],
) -> np.ndarray:
    arr = _vector4(value, name=name, default=default)
    if np.any(arr == 0.0):
        raise ValueError(f"{name} values must be non-zero signs")
    return np.where(arr < 0.0, -1.0, 1.0).astype(np.float32)


def _axis_order(value: Any, *, name: str) -> tuple[int, ...]:
    if value is None:
        items: list[Any] = list(range(REAL_ACTION_DIM))
    elif isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    else:
        items = list(value)
    if not items:
        raise ValueError(f"{name} must include at least one axis")

    by_name = {axis_name: idx for idx, axis_name in enumerate(REAL_ACTION_ORDER)}
    parsed: list[int] = []
    for item in items:
        if isinstance(item, str):
            key = item.strip().lower()
            if key not in by_name:
                raise ValueError(
                    f"{name} unknown axis {item!r}; expected one of {REAL_ACTION_ORDER}"
                )
            idx = by_name[key]
        else:
            idx = int(item)
        if idx < 0 or idx >= REAL_ACTION_DIM:
            raise ValueError(f"{name} axis index out of range: {idx}")
        if idx in parsed:
            raise ValueError(f"{name} contains duplicate axis: {idx}")
        parsed.append(idx)

    parsed.extend(idx for idx in range(REAL_ACTION_DIM) if idx not in parsed)
    return tuple(parsed)


@dataclass(frozen=True)
class GoHomeConfig:
    """Configuration for safe near-home qpos feedback control."""

    home_pose_rad: np.ndarray
    near_tolerance_rad: np.ndarray
    success_tolerance_rad: np.ndarray
    center_tolerance_rad: np.ndarray
    resume_tolerance_rad: np.ndarray
    control_signs: np.ndarray
    p_gain: np.ndarray
    d_gain: np.ndarray
    min_action: np.ndarray
    min_action_positive: np.ndarray
    min_action_negative: np.ndarray
    max_action: np.ndarray
    near_max_action: np.ndarray
    coast_stop_time_s: np.ndarray
    coast_reactivation_delay_s: np.ndarray
    action_slew_rate: np.ndarray
    action_ramp_rate: np.ndarray
    center_approach_action: np.ndarray
    control_decision_period_s: float
    axis_order: tuple[int, ...]
    max_active_axes: int
    sign_reversal_delay_s: np.ndarray
    sign_reversal_min_error_rad: np.ndarray
    qpos_filter_tau_s: np.ndarray
    qvel_filter_tau_s: np.ndarray
    axis_center_hold_s: np.ndarray
    stall_action_step: np.ndarray
    stall_detection_s: float
    stall_boost_interval_s: float
    stall_error_progress_rad: np.ndarray
    stall_qvel_threshold_rad_s: np.ndarray
    wrong_direction_detection_s: float
    wrong_direction_error_increase_rad: np.ndarray
    wrong_direction_cooldown_s: np.ndarray
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
        center = _positive_vector4(
            raw.get("center_tolerance_rad"),
            name="center_tolerance_rad",
            default=success,
        )
        resume = _positive_vector4(
            raw.get("resume_tolerance_rad"),
            name="resume_tolerance_rad",
            default=np.maximum(success, center * 1.5),
        )
        if np.any(center > success):
            raise ValueError("center_tolerance_rad values must be <= success_tolerance_rad")
        if np.any(resume < center):
            raise ValueError("resume_tolerance_rad values must be >= center_tolerance_rad")
        control_signs = _sign_vector4(
            raw.get("control_signs"),
            name="control_signs",
            default=[1.0] * REAL_ACTION_DIM,
        )
        p_gain = _positive_vector4(
            raw.get("p_gain"),
            name="p_gain",
            default=[1.2] * REAL_ACTION_DIM,
        )
        d_gain = _nonnegative_vector4(
            raw.get("d_gain"),
            name="d_gain",
            default=[0.0] * REAL_ACTION_DIM,
        )
        min_action = _nonnegative_vector4(
            raw.get("min_action"),
            name="min_action",
            default=[0.0] * REAL_ACTION_DIM,
        )
        min_action_positive = _nonnegative_vector4(
            raw.get("min_action_positive"),
            name="min_action_positive",
            default=min_action,
        )
        min_action_negative = _nonnegative_vector4(
            raw.get("min_action_negative"),
            name="min_action_negative",
            default=min_action,
        )
        max_action = _positive_vector4(
            raw.get("max_action"),
            name="max_action",
            default=[0.20] * REAL_ACTION_DIM,
        )
        near_max_action = _positive_vector4(
            raw.get("near_max_action"),
            name="near_max_action",
            default=max_action,
        )
        if np.any(min_action_positive > max_action) or np.any(
            min_action_negative > max_action
        ):
            raise ValueError("min_action values must be <= max_action values")
        if np.any(min_action_positive > near_max_action) or np.any(
            min_action_negative > near_max_action
        ):
            raise ValueError("min_action values must be <= near_max_action values")
        if np.any(near_max_action > max_action):
            raise ValueError("near_max_action values must be <= max_action values")
        coast_stop_time_s = _nonnegative_vector4(
            raw.get("coast_stop_time_s"),
            name="coast_stop_time_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        coast_reactivation_delay_s = _nonnegative_vector4(
            raw.get("coast_reactivation_delay_s"),
            name="coast_reactivation_delay_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        action_slew_rate = _nonnegative_vector4(
            raw.get("action_slew_rate"),
            name="action_slew_rate",
            default=[0.0] * REAL_ACTION_DIM,
        )
        action_ramp_rate = _nonnegative_vector4(
            raw.get("action_ramp_rate"),
            name="action_ramp_rate",
            default=[0.0] * REAL_ACTION_DIM,
        )
        center_approach_action = _nonnegative_vector4(
            raw.get("center_approach_action"),
            name="center_approach_action",
            default=[0.0] * REAL_ACTION_DIM,
        )
        control_decision_period_s = float(raw.get("control_decision_period_s", 0.0))
        if control_decision_period_s < 0.0:
            raise ValueError("control_decision_period_s must be >= 0")
        axis_order = _axis_order(raw.get("axis_order"), name="axis_order")
        max_active_axes = int(raw.get("max_active_axes", REAL_ACTION_DIM))
        if max_active_axes <= 0 or max_active_axes > REAL_ACTION_DIM:
            raise ValueError(f"max_active_axes must be in [1,{REAL_ACTION_DIM}]")
        sign_reversal_delay_s = _nonnegative_vector4(
            raw.get("sign_reversal_delay_s"),
            name="sign_reversal_delay_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        sign_reversal_min_error_rad = _nonnegative_vector4(
            raw.get("sign_reversal_min_error_rad"),
            name="sign_reversal_min_error_rad",
            default=[0.0] * REAL_ACTION_DIM,
        )
        qpos_filter_tau_s = _nonnegative_vector4(
            raw.get("qpos_filter_tau_s"),
            name="qpos_filter_tau_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        qvel_filter_tau_s = _nonnegative_vector4(
            raw.get("qvel_filter_tau_s"),
            name="qvel_filter_tau_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        axis_center_hold_s = _nonnegative_vector4(
            raw.get("axis_center_hold_s"),
            name="axis_center_hold_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        stall_action_step = _nonnegative_vector4(
            raw.get("stall_action_step"),
            name="stall_action_step",
            default=[0.0] * REAL_ACTION_DIM,
        )
        stall_detection_s = float(raw.get("stall_detection_s", 1.0))
        stall_boost_interval_s = float(raw.get("stall_boost_interval_s", 0.75))
        if stall_detection_s < 0.0:
            raise ValueError("stall_detection_s must be >= 0")
        if stall_boost_interval_s <= 0.0:
            raise ValueError("stall_boost_interval_s must be positive")
        stall_error_progress = _nonnegative_vector4(
            raw.get("stall_error_progress_rad"),
            name="stall_error_progress_rad",
            default=[0.002] * REAL_ACTION_DIM,
        )
        stall_qvel_threshold = _nonnegative_vector4(
            raw.get("stall_qvel_threshold_rad_s"),
            name="stall_qvel_threshold_rad_s",
            default=[0.01] * REAL_ACTION_DIM,
        )
        wrong_direction_detection_s = float(
            raw.get("wrong_direction_detection_s", stall_detection_s)
        )
        if wrong_direction_detection_s < 0.0:
            raise ValueError("wrong_direction_detection_s must be >= 0")
        wrong_direction_error_increase = _nonnegative_vector4(
            raw.get("wrong_direction_error_increase_rad"),
            name="wrong_direction_error_increase_rad",
            default=[1.0e9] * REAL_ACTION_DIM,
        )
        wrong_direction_cooldown = _nonnegative_vector4(
            raw.get("wrong_direction_cooldown_s"),
            name="wrong_direction_cooldown_s",
            default=[0.0] * REAL_ACTION_DIM,
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
            center_tolerance_rad=center,
            resume_tolerance_rad=resume,
            control_signs=control_signs,
            p_gain=p_gain,
            d_gain=d_gain,
            min_action=min_action,
            min_action_positive=min_action_positive,
            min_action_negative=min_action_negative,
            max_action=max_action,
            near_max_action=near_max_action,
            coast_stop_time_s=coast_stop_time_s,
            coast_reactivation_delay_s=coast_reactivation_delay_s,
            action_slew_rate=action_slew_rate,
            action_ramp_rate=action_ramp_rate,
            center_approach_action=center_approach_action,
            control_decision_period_s=control_decision_period_s,
            axis_order=axis_order,
            max_active_axes=max_active_axes,
            sign_reversal_delay_s=sign_reversal_delay_s,
            sign_reversal_min_error_rad=sign_reversal_min_error_rad,
            qpos_filter_tau_s=qpos_filter_tau_s,
            qvel_filter_tau_s=qvel_filter_tau_s,
            axis_center_hold_s=axis_center_hold_s,
            stall_action_step=stall_action_step,
            stall_detection_s=stall_detection_s,
            stall_boost_interval_s=stall_boost_interval_s,
            stall_error_progress_rad=stall_error_progress,
            stall_qvel_threshold_rad_s=stall_qvel_threshold,
            wrong_direction_detection_s=wrong_direction_detection_s,
            wrong_direction_error_increase_rad=wrong_direction_error_increase,
            wrong_direction_cooldown_s=wrong_direction_cooldown,
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
    """Run a bounded PD controller from near-home qpos to the configured home pose."""

    def __init__(self, config: GoHomeConfig) -> None:
        self.config = config
        self.started = False
        self.start_time_ns = 0
        self.end_time_ns = 0
        self._start_s = 0.0
        self._last_update_s: float | None = None
        self._last_decision_s: float | None = None
        self._filter_last_s: float | None = None
        self._stable_since_s: float | None = None
        self._axis_active = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_scheduled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._center_approach_axis = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_center_since_s = np.full(REAL_ACTION_DIM, np.nan, dtype=np.float64)
        self._axis_coast_until_s = np.zeros(REAL_ACTION_DIM, dtype=np.float64)
        self._axis_ramp_start_s = np.full(REAL_ACTION_DIM, np.nan, dtype=np.float64)
        self._axis_ramp_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._action_ramp_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_command_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_last_sign_change_s = np.full(REAL_ACTION_DIM, -np.inf, dtype=np.float64)
        self._sign_reversal_blocked = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._last_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._held_unsmoothed_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._filtered_qpos = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._filtered_qvel = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._raw_error = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._stall_reference_error = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._stall_reference_s = np.zeros(REAL_ACTION_DIM, dtype=np.float64)
        self._stall_last_boost_s = np.full(REAL_ACTION_DIM, -np.inf, dtype=np.float64)
        self._stall_action_boost = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_stalled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_wrong_direction = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._stall_action_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._effective_min_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._action_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
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
        self._last_update_s = None
        self._last_decision_s = None
        self._filter_last_s = self._start_s
        self._stable_since_s = None
        self._axis_active = np.abs(error) > self.config.center_tolerance_rad
        self._axis_scheduled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._center_approach_axis = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_center_since_s = np.where(
            np.abs(error) <= self.config.center_tolerance_rad,
            self._start_s,
            np.nan,
        ).astype(np.float64)
        self._axis_coast_until_s = np.zeros(REAL_ACTION_DIM, dtype=np.float64)
        self._axis_ramp_start_s = np.full(REAL_ACTION_DIM, np.nan, dtype=np.float64)
        self._axis_ramp_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._action_ramp_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_command_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_last_sign_change_s = np.full(REAL_ACTION_DIM, -np.inf, dtype=np.float64)
        self._sign_reversal_blocked = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._last_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._held_unsmoothed_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._filtered_qpos = qpos.astype(np.float32, copy=True)
        self._filtered_qvel = _obs_qvel(obs).astype(np.float32, copy=True)
        self._raw_error = error.astype(np.float32, copy=True)
        abs_error = np.abs(error).astype(np.float32)
        self._stall_reference_error = abs_error.copy()
        self._stall_reference_s = np.full(REAL_ACTION_DIM, self._start_s, dtype=np.float64)
        self._stall_last_boost_s = np.full(REAL_ACTION_DIM, -np.inf, dtype=np.float64)
        self._stall_action_boost = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_stalled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_wrong_direction = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._stall_action_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._effective_min_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._action_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
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
            raw_qpos = _obs_qpos(obs)
            raw_qvel = _obs_qvel(obs)
        except ValueError as exc:
            return self._fail(f"feedback_invalid:{exc}", obs)
        qpos, qvel = self._filtered_feedback(raw_qpos, raw_qvel, now_s=now)

        error = self.config.home_pose_rad - qpos
        raw_error = self.config.home_pose_rad - raw_qpos
        self._raw_error = raw_error.astype(np.float32, copy=True)
        self.final_qpos = raw_qpos.astype(np.float32, copy=True)
        self.final_error = raw_error.astype(np.float32, copy=True)

        elapsed = max(0.0, now - self._start_s)

        runaway_limit = self.config.near_tolerance_rad * self.config.runaway_error_factor
        if np.any(np.abs(error) > runaway_limit):
            return self._fail("runaway_error", obs)

        abs_error = np.abs(error)
        acceptable_position = np.all(abs_error <= self.config.success_tolerance_rad)
        in_position = np.all(abs_error <= self.config.center_tolerance_rad)
        stable_velocity = np.all(np.abs(qvel) <= self.config.qvel_stable_rad_s)
        if elapsed > self.config.timeout_s and not (
            acceptable_position and stable_velocity
        ):
            return self._fail("timeout", obs)
        decision_updated = False
        if acceptable_position and stable_velocity:
            if self._stable_since_s is None:
                self._stable_since_s = now
            if now - self._stable_since_s >= self.config.dwell_s:
                return self._finish("succeeded", done=True)
        else:
            self._stable_since_s = None

        if in_position:
            self._axis_active = np.zeros(REAL_ACTION_DIM, dtype=bool)
            self._axis_scheduled = np.zeros(REAL_ACTION_DIM, dtype=bool)
            self._center_approach_axis = np.zeros(REAL_ACTION_DIM, dtype=bool)
            self._axis_stalled = np.zeros(REAL_ACTION_DIM, dtype=bool)
            self._axis_wrong_direction = np.zeros(REAL_ACTION_DIM, dtype=bool)
            self._stall_action_boost = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            self._effective_min_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            self._action_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            self._action_ramp_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            self._axis_ramp_start_s = np.full(REAL_ACTION_DIM, np.nan, dtype=np.float64)
            self._axis_ramp_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            self._sign_reversal_blocked = np.zeros(REAL_ACTION_DIM, dtype=bool)
            self._held_unsmoothed_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        else:
            decision_updated = self._decision_due(now)
            if not decision_updated:
                action = self._held_unsmoothed_action.copy()
            else:
                self._last_decision_s = now
                action = self._compute_control_decision(
                    now=now,
                    error=error,
                    qvel=qvel,
                    abs_error=abs_error,
                )

        unsmoothed_action = np.asarray(action, dtype=np.float32)
        action = self._apply_slew_limit(unsmoothed_action, now_s=now)

        return GoHomeResult(
            action=np.asarray(action, dtype=np.float32),
            diagnostics=self._diagnostics(
                qpos=qpos,
                qvel=qvel,
                raw_qpos=raw_qpos,
                raw_qvel=raw_qvel,
                action=action,
                unsmoothed_action=unsmoothed_action,
                in_position=in_position,
                acceptable_position=acceptable_position,
                stable_velocity=stable_velocity,
                elapsed_s=elapsed,
                decision_updated=decision_updated,
            ),
        )

    def _decision_due(self, now_s: float) -> bool:
        period = float(self.config.control_decision_period_s)
        if period <= 0.0 or self._last_decision_s is None:
            return True
        return (now_s - self._last_decision_s) >= period

    def _compute_control_decision(
        self,
        *,
        now: float,
        error: np.ndarray,
        qvel: np.ndarray,
        abs_error: np.ndarray,
    ) -> np.ndarray:
        coast_waiting = now < self._axis_coast_until_s
        resume_axis = (abs_error > self.config.resume_tolerance_rad) & ~coast_waiting
        centered_axis = self._centered_axis_held(abs_error, now_s=now)
        self._axis_active = np.where(
            resume_axis,
            True,
            np.where(centered_axis, False, self._axis_active),
        )
        self._axis_active = np.where(coast_waiting, False, self._axis_active)
        moving_toward_center = (error * qvel) > 0.0
        approach_zone = (
            self._axis_active
            & moving_toward_center
            & (
                abs_error
                <= (
                    self.config.center_tolerance_rad
                    + np.abs(qvel) * self.config.coast_stop_time_s
                )
            )
        )
        low_speed_approach_axis = approach_zone & (
            self.config.center_approach_action > 0.0
        )
        coast_axis = approach_zone & ~low_speed_approach_axis
        if np.any(coast_axis):
            self._axis_coast_until_s = np.where(
                coast_axis,
                now + self.config.coast_reactivation_delay_s,
                self._axis_coast_until_s,
            )
        self._axis_active = np.where(coast_axis, False, self._axis_active)
        if not np.any(self._axis_active):
            outside_center = abs_error > self.config.center_tolerance_rad
            approach_to_center = (
                outside_center
                & moving_toward_center
                & (
                    abs_error
                    <= (
                        self.config.center_tolerance_rad
                        + np.abs(qvel) * self.config.coast_stop_time_s
                    )
                )
            )
            coasting_to_center = approach_to_center & (
                self.config.center_approach_action <= 0.0
            )
            reactivate_axis = outside_center & ~coasting_to_center & ~coast_waiting
            if np.any(reactivate_axis):
                relative_error = np.where(
                    reactivate_axis,
                    abs_error / self.config.center_tolerance_rad,
                    -np.inf,
                )
                self._axis_active[int(np.argmax(relative_error))] = True
        self._apply_axis_scheduler(abs_error=abs_error, coast_waiting=coast_waiting)
        pd_action = self.config.control_signs * (
            self.config.p_gain * error - self.config.d_gain * qvel
        )
        action_limit = np.where(
            abs_error <= self.config.success_tolerance_rad,
            self.config.near_max_action,
            self.config.max_action,
        ).astype(np.float32)
        self._center_approach_axis = (
            self._axis_active
            & moving_toward_center
            & (
                abs_error
                <= (
                    self.config.center_tolerance_rad
                    + np.abs(qvel) * self.config.coast_stop_time_s
                )
            )
            & (self.config.center_approach_action > 0.0)
        )
        action_limit = np.where(
            self._center_approach_axis,
            np.minimum(action_limit, self.config.center_approach_action),
            action_limit,
        ).astype(np.float32)
        action = np.where(self._axis_active, pd_action, 0.0)
        action = np.clip(action, -action_limit, action_limit)
        action_sign = _nonzero_sign(
            action,
            fallback=self.config.control_signs * error,
        )
        action = self._apply_sign_reversal_guard(
            now_s=now,
            action=action,
            action_sign=action_sign,
            abs_error=abs_error,
        )
        action_sign = _nonzero_sign(
            action,
            fallback=self.config.control_signs * error,
        )
        min_action = np.where(
            action_sign >= 0.0,
            self.config.min_action_positive,
            self.config.min_action_negative,
        )
        self._update_stall_boost(
            now_s=now,
            abs_error=abs_error,
            qvel=qvel,
            action_sign=action_sign,
            min_action=min_action,
        )
        action = self._apply_wrong_direction_guard(
            now_s=now,
            abs_error=abs_error,
            action=action,
        )
        action_sign = _nonzero_sign(
            action,
            fallback=self.config.control_signs * error,
        )
        boosted_axis = self._axis_stalled & (self._stall_action_boost > 0.0)
        action_limit = np.where(
            boosted_axis,
            np.maximum(action_limit, self.config.max_action),
            action_limit,
        ).astype(np.float32)
        effective_min_action = np.minimum(
            min_action + self._stall_action_boost,
            action_limit,
        )
        action_ramp_limit = self._update_action_ramp_limit(
            now_s=now,
            action_sign=action_sign,
            effective_min_action=effective_min_action,
            action_limit=action_limit,
        )
        self._effective_min_action = np.where(
            self._axis_active,
            effective_min_action,
            0.0,
        ).astype(np.float32)
        self._action_limit = action_limit.astype(np.float32)
        self._action_ramp_limit = np.where(
            self._axis_active,
            action_ramp_limit,
            0.0,
        ).astype(np.float32)
        action = np.clip(action, -action_ramp_limit, action_ramp_limit)
        small_active = (
            self._axis_active
            & (effective_min_action > 0.0)
            & (np.abs(action) < effective_min_action)
        )
        action = np.where(
            small_active,
            action_sign * effective_min_action,
            action,
        )
        action = np.clip(action, -action_ramp_limit, action_ramp_limit)
        self._held_unsmoothed_action = np.asarray(action, dtype=np.float32)
        return self._held_unsmoothed_action.copy()

    def _apply_axis_scheduler(
        self,
        *,
        abs_error: np.ndarray,
        coast_waiting: np.ndarray,
    ) -> None:
        active = np.asarray(self._axis_active, dtype=bool)
        if self.config.max_active_axes >= REAL_ACTION_DIM:
            self._axis_scheduled = active.copy()
            return

        eligible = (
            active
            & (abs_error > self.config.center_tolerance_rad)
            & ~np.asarray(coast_waiting, dtype=bool)
        )
        scheduled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        count = 0
        for idx in self.config.axis_order:
            if not eligible[int(idx)]:
                continue
            scheduled[int(idx)] = True
            count += 1
            if count >= self.config.max_active_axes:
                break
        self._axis_scheduled = scheduled
        self._axis_active = active & scheduled

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
            "go_home_final_filtered_error": (
                self.config.home_pose_rad - self._filtered_qpos
            ).astype(np.float32),
        }

    def _finish(self, result_code: str, *, done: bool = False, failed: bool = False) -> GoHomeResult:
        self.result_code = result_code
        self.end_time_ns = int(time.time_ns())
        self.started = False
        self._last_decision_s = None
        self._last_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._held_unsmoothed_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_scheduled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._center_approach_axis = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_command_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_last_sign_change_s = np.full(REAL_ACTION_DIM, -np.inf, dtype=np.float64)
        self._sign_reversal_blocked = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_stalled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_wrong_direction = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._stall_action_boost = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._effective_min_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._action_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._action_ramp_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_ramp_start_s = np.full(REAL_ACTION_DIM, np.nan, dtype=np.float64)
        self._axis_ramp_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        filtered_error = (self.config.home_pose_rad - self._filtered_qpos).astype(np.float32)
        return GoHomeResult(
            action=np.zeros(REAL_ACTION_DIM, dtype=np.float32),
            done=done,
            failed=failed,
            reason=self.failed_reason or result_code,
            diagnostics={
                "go_home_running": 0,
                "go_home_result_code": result_code,
                "go_home_error": filtered_error,
                "go_home_raw_error": self.final_error.astype(np.float32),
                "go_home_qvel": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_raw_qvel": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_filtered_qpos": self._filtered_qpos.astype(np.float32),
                "go_home_filtered_qvel": self._filtered_qvel.astype(np.float32),
                "go_home_commanded_action": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_unsmoothed_action": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_axis_active": np.zeros(REAL_ACTION_DIM, dtype=np.int32),
                "go_home_axis_scheduled": np.zeros(REAL_ACTION_DIM, dtype=np.int32),
                "go_home_center_approach_axis": np.zeros(
                    REAL_ACTION_DIM, dtype=np.int32
                ),
                "go_home_axis_stalled": np.zeros(REAL_ACTION_DIM, dtype=np.int32),
                "go_home_axis_wrong_direction": np.zeros(REAL_ACTION_DIM, dtype=np.int32),
                "go_home_stall_action_boost": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_effective_min_action": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_action_limit": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_action_ramp_limit": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_control_signs": self.config.control_signs.astype(np.float32),
                "go_home_command_sign": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_sign_reversal_blocked": np.zeros(REAL_ACTION_DIM, dtype=np.int32),
                "go_home_decision_updated": 0,
                "go_home_control_decision_period_s": float(
                    self.config.control_decision_period_s
                ),
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
        raw_qpos: np.ndarray,
        raw_qvel: np.ndarray,
        action: np.ndarray,
        unsmoothed_action: np.ndarray,
        in_position: bool,
        acceptable_position: bool,
        stable_velocity: bool,
        elapsed_s: float,
        decision_updated: bool,
    ) -> dict[str, Any]:
        return {
            "go_home_running": 1,
            "go_home_result_code": "running",
            "go_home_error": (self.config.home_pose_rad - qpos).astype(np.float32),
            "go_home_raw_error": (self.config.home_pose_rad - raw_qpos).astype(np.float32),
            "go_home_qvel": np.asarray(qvel, dtype=np.float32),
            "go_home_raw_qvel": np.asarray(raw_qvel, dtype=np.float32),
            "go_home_filtered_qpos": np.asarray(qpos, dtype=np.float32),
            "go_home_filtered_qvel": np.asarray(qvel, dtype=np.float32),
            "go_home_commanded_action": np.asarray(action, dtype=np.float32),
            "go_home_unsmoothed_action": np.asarray(unsmoothed_action, dtype=np.float32),
            "go_home_axis_active": self._axis_active.astype(np.int32),
            "go_home_axis_scheduled": self._axis_scheduled.astype(np.int32),
            "go_home_center_approach_axis": self._center_approach_axis.astype(np.int32),
            "go_home_axis_stalled": self._axis_stalled.astype(np.int32),
            "go_home_axis_wrong_direction": self._axis_wrong_direction.astype(np.int32),
            "go_home_stall_action_boost": self._stall_action_boost.astype(np.float32),
            "go_home_effective_min_action": self._effective_min_action.astype(np.float32),
            "go_home_action_limit": self._action_limit.astype(np.float32),
            "go_home_action_ramp_limit": self._action_ramp_limit.astype(np.float32),
            "go_home_control_signs": self.config.control_signs.astype(np.float32),
            "go_home_command_sign": self._axis_command_sign.astype(np.float32),
            "go_home_sign_reversal_blocked": self._sign_reversal_blocked.astype(np.int32),
            "go_home_decision_updated": int(bool(decision_updated)),
            "go_home_control_decision_period_s": float(
                self.config.control_decision_period_s
            ),
            "go_home_in_position": int(bool(in_position)),
            "go_home_acceptable_position": int(bool(acceptable_position)),
            "go_home_stable_velocity": int(bool(stable_velocity)),
            "go_home_elapsed_s": float(elapsed_s),
        }

    def _filtered_feedback(
        self,
        raw_qpos: np.ndarray,
        raw_qvel: np.ndarray,
        *,
        now_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = 0.02 if self._filter_last_s is None else max(0.0, now_s - self._filter_last_s)
        self._filter_last_s = now_s
        self._filtered_qpos = _lowpass_vector(
            raw_qpos,
            self._filtered_qpos,
            tau_s=self.config.qpos_filter_tau_s,
            dt_s=dt,
        )
        self._filtered_qvel = _lowpass_vector(
            raw_qvel,
            self._filtered_qvel,
            tau_s=self.config.qvel_filter_tau_s,
            dt_s=dt,
        )
        return self._filtered_qpos.copy(), self._filtered_qvel.copy()

    def _centered_axis_held(self, abs_error: np.ndarray, *, now_s: float) -> np.ndarray:
        inside = abs_error <= self.config.center_tolerance_rad
        newly_inside = inside & np.isnan(self._axis_center_since_s)
        if np.any(newly_inside):
            self._axis_center_since_s = np.where(
                newly_inside,
                now_s,
                self._axis_center_since_s,
            )
        self._axis_center_since_s = np.where(
            inside,
            self._axis_center_since_s,
            np.nan,
        )
        held_s = np.where(
            np.isnan(self._axis_center_since_s),
            0.0,
            now_s - self._axis_center_since_s,
        )
        return inside & (held_s >= self.config.axis_center_hold_s)

    def _update_action_ramp_limit(
        self,
        *,
        now_s: float,
        action_sign: np.ndarray,
        effective_min_action: np.ndarray,
        action_limit: np.ndarray,
    ) -> np.ndarray:
        rate = self.config.action_ramp_rate.astype(np.float32)
        if np.all(rate <= 0.0):
            self._axis_ramp_start_s = np.where(
                self._axis_active,
                self._axis_ramp_start_s,
                np.nan,
            )
            self._axis_ramp_sign = np.where(
                self._axis_active,
                action_sign,
                0.0,
            ).astype(np.float32)
            return action_limit.astype(np.float32)

        active = np.asarray(self._axis_active, dtype=bool)
        sign = np.asarray(action_sign, dtype=np.float32)
        sign_changed = active & (self._axis_ramp_sign != 0.0) & (
            sign != self._axis_ramp_sign
        )
        needs_start = active & (np.isnan(self._axis_ramp_start_s) | sign_changed)
        if np.any(needs_start):
            self._axis_ramp_start_s = np.where(
                needs_start,
                now_s,
                self._axis_ramp_start_s,
            )
        inactive = ~active
        if np.any(inactive):
            self._axis_ramp_start_s[inactive] = np.nan
        self._axis_ramp_sign = np.where(active, sign, 0.0).astype(np.float32)

        age_s = np.where(
            np.isnan(self._axis_ramp_start_s),
            0.0,
            now_s - self._axis_ramp_start_s,
        )
        ramp_limit = effective_min_action + rate * age_s.astype(np.float32)
        ramp_limit = np.where(rate <= 0.0, action_limit, ramp_limit)
        return np.minimum(ramp_limit, action_limit).astype(np.float32)

    def _apply_sign_reversal_guard(
        self,
        *,
        now_s: float,
        action: np.ndarray,
        action_sign: np.ndarray,
        abs_error: np.ndarray,
    ) -> np.ndarray:
        active = np.asarray(self._axis_active, dtype=bool)
        sign = np.where(active, np.asarray(action_sign, dtype=np.float32), 0.0)
        previous = self._axis_command_sign.astype(np.float32)
        reversal = active & (previous != 0.0) & (sign != 0.0) & (sign != previous)
        enough_time = (
            now_s - self._axis_last_sign_change_s
        ) >= self.config.sign_reversal_delay_s
        enough_error = abs_error >= self.config.sign_reversal_min_error_rad
        blocked = reversal & ~(enough_time & enough_error)
        self._sign_reversal_blocked = blocked.astype(bool)
        if np.any(blocked):
            active = active.copy()
            active[blocked] = False
            self._axis_active = active
            self._axis_coast_until_s = np.where(
                blocked,
                np.maximum(
                    self._axis_coast_until_s,
                    now_s + self.config.sign_reversal_delay_s,
                ),
                self._axis_coast_until_s,
            )
            action = np.asarray(action, dtype=np.float32).copy()
            action[blocked] = 0.0

        allowed = active & (sign != 0.0) & ~blocked
        changed = allowed & ((previous == 0.0) | (sign != previous))
        if np.any(changed):
            self._axis_last_sign_change_s[changed] = float(now_s)
        self._axis_command_sign = np.where(allowed, sign, previous).astype(np.float32)
        return np.asarray(action, dtype=np.float32)

    def _apply_wrong_direction_guard(
        self,
        *,
        now_s: float,
        abs_error: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        active = np.asarray(self._axis_active, dtype=bool)
        age_s = np.asarray(now_s - self._stall_reference_s, dtype=np.float32)
        growing_error = (
            abs_error
            >= (
                self._stall_reference_error
                + self.config.wrong_direction_error_increase_rad
            )
        )
        wrong = active & growing_error & (
            age_s >= float(self.config.wrong_direction_detection_s)
        )
        self._axis_wrong_direction = wrong.astype(bool)
        if not np.any(wrong):
            return np.asarray(action, dtype=np.float32)

        self._axis_active[wrong] = False
        self._axis_stalled[wrong] = False
        self._stall_action_boost[wrong] = 0.0
        self._effective_min_action[wrong] = 0.0
        self._action_limit[wrong] = 0.0
        self._action_ramp_limit[wrong] = 0.0
        self._axis_coast_until_s = np.where(
            wrong,
            np.maximum(
                self._axis_coast_until_s,
                now_s + self.config.wrong_direction_cooldown_s,
            ),
            self._axis_coast_until_s,
        )
        guarded = np.asarray(action, dtype=np.float32).copy()
        guarded[wrong] = 0.0
        return guarded

    def _apply_slew_limit(self, action: np.ndarray, *, now_s: float) -> np.ndarray:
        rate = self.config.action_slew_rate.astype(np.float32)
        if np.all(rate <= 0.0):
            self._last_update_s = now_s
            self._last_action = np.asarray(action, dtype=np.float32)
            return np.asarray(action, dtype=np.float32)
        dt = 0.02 if self._last_update_s is None else max(0.0, now_s - self._last_update_s)
        self._last_update_s = now_s
        unlimited = rate <= 0.0
        max_delta = rate * float(dt)
        delta = np.asarray(action, dtype=np.float32) - self._last_action
        limited_delta = np.where(unlimited, delta, np.clip(delta, -max_delta, max_delta))
        limited = self._last_action + limited_delta
        self._last_action = limited.astype(np.float32)
        return self._last_action.copy()

    def _update_stall_boost(
        self,
        *,
        now_s: float,
        abs_error: np.ndarray,
        qvel: np.ndarray,
        action_sign: np.ndarray,
        min_action: np.ndarray,
    ) -> None:
        active = np.asarray(self._axis_active, dtype=bool)
        sign = np.asarray(action_sign, dtype=np.float32)
        sign_changed = active & (self._stall_action_sign != 0.0) & (
            sign != self._stall_action_sign
        )
        reset = (~active) | sign_changed
        if np.any(reset):
            self._stall_reference_error[reset] = abs_error[reset].astype(np.float32)
            self._stall_reference_s[reset] = float(now_s)
            self._stall_last_boost_s[reset] = -np.inf
            self._stall_action_boost[reset] = 0.0
            self._axis_stalled[reset] = False
        self._stall_action_sign = np.where(active, sign, 0.0).astype(np.float32)

        improved = active & (
            abs_error
            <= (self._stall_reference_error - self.config.stall_error_progress_rad)
        )
        if np.any(improved):
            self._stall_reference_error[improved] = abs_error[improved].astype(np.float32)
            self._stall_reference_s[improved] = float(now_s)
            self._axis_stalled[improved] = False

        age_s = np.asarray(now_s - self._stall_reference_s, dtype=np.float32)
        no_progress = active & ~improved & (
            age_s >= float(self.config.stall_detection_s)
        )
        low_velocity = np.abs(qvel) <= self.config.stall_qvel_threshold_rad_s
        stalled = no_progress & low_velocity
        self._axis_stalled = stalled.astype(bool)
        boost_due = stalled & (
            (now_s - self._stall_last_boost_s)
            >= float(self.config.stall_boost_interval_s)
        )
        step = self.config.stall_action_step.astype(np.float32)
        boost_cap = np.maximum(
            self.config.max_action.astype(np.float32) - min_action.astype(np.float32),
            0.0,
        )
        if np.any(boost_due & (step > 0.0)):
            idx = boost_due & (step > 0.0)
            self._stall_action_boost[idx] = np.minimum(
                self._stall_action_boost[idx] + step[idx],
                boost_cap[idx],
            ).astype(np.float32)
            self._stall_last_boost_s[idx] = float(now_s)


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


def _nonzero_sign(value: Any, *, fallback: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    fallback_arr = np.asarray(fallback, dtype=np.float32)
    sign = np.sign(arr)
    return np.where(sign == 0.0, np.sign(fallback_arr), sign)


def _lowpass_vector(
    raw: np.ndarray,
    previous: np.ndarray,
    *,
    tau_s: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    raw_arr = np.asarray(raw, dtype=np.float32)
    prev_arr = np.asarray(previous, dtype=np.float32)
    tau = np.asarray(tau_s, dtype=np.float32)
    if dt_s <= 0.0 or np.all(tau <= 0.0):
        return raw_arr.astype(np.float32)
    alpha = np.where(tau <= 0.0, 1.0, float(dt_s) / (tau + float(dt_s)))
    return (prev_arr + alpha * (raw_arr - prev_arr)).astype(np.float32)
