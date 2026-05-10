"""Safety guard for real-world action commands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_DIM = 4


@dataclass(frozen=True)
class GuardInfo:
    """Diagnostic information for the latest guard decision."""

    triggered: bool
    reasons: tuple[str, ...]
    raw_action: np.ndarray
    safe_action: np.ndarray


class ActionGuard:
    """
    Intercepts actions before env.step().

    Parameters
    ----------
    joint_limits    (2, N) array — [[lower], [upper]] joint position limits.
    vel_limit       Maximum absolute joint velocity (rad/s).
    action_clip     Maximum absolute normalized command per axis.
    max_delta       Maximum per-step normalized command change per axis.
    sensor_timeout_s Maximum allowed sensor age before forcing zero command.
    """

    def __init__(
        self,
        joint_limits: np.ndarray | None = None,
        vel_limit: float = float("inf"),
        action_clip: float | np.ndarray | list[float] | tuple[float, ...] = 1.0,
        max_delta: float | np.ndarray | list[float] | tuple[float, ...] | None = None,
        sensor_timeout_s: float | None = None,
    ):
        self.joint_limits  = joint_limits
        self.vel_limit     = vel_limit
        self.action_clip = _broadcast(action_clip, name="action_clip")
        self.max_delta = None if max_delta is None else _broadcast(max_delta, name="max_delta")
        self.sensor_timeout_s = None if sensor_timeout_s is None else float(sensor_timeout_s)
        self.trigger_count = 0
        self.total_count = 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_info = GuardInfo(
            triggered=False,
            reasons=(),
            raw_action=np.zeros(ACTION_DIM, dtype=np.float32),
            safe_action=np.zeros(ACTION_DIM, dtype=np.float32),
        )

    def check(
        self,
        action: np.ndarray,
        current_qpos: np.ndarray | None = None,
        *,
        deadman_pressed: bool = True,
        estop_active: bool = False,
        manual_override_active: bool = False,
        sensor_age_s: float | None = None,
        sensor_stale: bool = False,
    ) -> tuple[np.ndarray, bool]:
        """
        Validate and (optionally) clip an action.

        Parameters
        ----------
        action       Raw action from policy.
        current_qpos Current joint positions (for velocity limiting).

        Returns
        -------
        (safe_action, triggered)
          safe_action : (possibly clipped) action array.
          triggered   : True if any guard rule fired.
        """
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw_action.shape != (ACTION_DIM,):
            raise ValueError(f"action must have shape ({ACTION_DIM},), got {raw_action.shape}")

        self.total_count += 1
        reasons: list[str] = []

        safe_action = np.clip(raw_action, -self.action_clip, self.action_clip)
        if not np.allclose(safe_action, raw_action):
            reasons.append("action_clip")

        if self.max_delta is not None:
            lower = self._last_action - self.max_delta
            upper = self._last_action + self.max_delta
            rate_limited = np.clip(safe_action, lower, upper)
            if not np.allclose(rate_limited, safe_action):
                reasons.append("rate_limit")
            safe_action = rate_limited

        if not bool(deadman_pressed):
            safe_action = np.zeros_like(safe_action)
            reasons.append("deadman_released")
        if bool(estop_active):
            safe_action = np.zeros_like(safe_action)
            reasons.append("estop_active")
        if bool(manual_override_active):
            safe_action = np.zeros_like(safe_action)
            reasons.append("manual_override_active")
        if bool(sensor_stale):
            safe_action = np.zeros_like(safe_action)
            reasons.append("sensor_stale")
        if (
            sensor_age_s is not None
            and self.sensor_timeout_s is not None
            and float(sensor_age_s) > self.sensor_timeout_s
        ):
            safe_action = np.zeros_like(safe_action)
            reasons.append("sensor_timeout")

        triggered = bool(reasons)
        if triggered:
            self.trigger_count += 1
        self._last_action = safe_action.astype(np.float32, copy=True)
        self.last_info = GuardInfo(
            triggered=triggered,
            reasons=tuple(reasons),
            raw_action=raw_action.astype(np.float32, copy=True),
            safe_action=self._last_action.copy(),
        )
        return self._last_action.copy(), triggered

    @property
    def trigger_rate(self) -> float:
        """Guard trigger rate (requires caller to track total steps)."""
        if self.total_count <= 0:
            return 0.0
        return float(self.trigger_count) / float(self.total_count)

    def reset(self) -> None:
        self.trigger_count = 0
        self.total_count = 0
        self._last_action.fill(0.0)
        self.last_info = GuardInfo(
            triggered=False,
            reasons=(),
            raw_action=np.zeros(ACTION_DIM, dtype=np.float32),
            safe_action=np.zeros(ACTION_DIM, dtype=np.float32),
        )


def _broadcast(
    value: float | np.ndarray | list[float] | tuple[float, ...],
    *,
    name: str,
) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full(ACTION_DIM, float(value), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (ACTION_DIM,):
        raise ValueError(f"{name} must be scalar or shape ({ACTION_DIM},), got {arr.shape}")
    return arr
