"""Axis response smoothing for teleop action sources."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np


ACTION_DIM = 4


def _move_towards(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + np.sign(delta) * max_delta


def _broadcast(value: float | Sequence[float], *, name: str) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full(ACTION_DIM, float(value), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (ACTION_DIM,):
        raise ValueError(f"{name} must be a scalar or length-{ACTION_DIM} sequence")
    return arr


@dataclass
class AxisResponseProfile:
    deadzone: float = 0.05
    attack_rate: float = 4.0
    release_rate: float = 6.0
    recenter_rate: float = 7.0
    exponent: float = 1.0

    def apply(self, raw_value: float, current_value: float, delta_time: float) -> float:
        target_value = self.remap(raw_value)
        if abs(target_value) < 1.0e-4:
            rate = self.recenter_rate
        elif abs(target_value) > abs(current_value):
            rate = self.attack_rate
        else:
            rate = self.release_rate

        max_delta = max(float(rate), 0.0) * max(float(delta_time), 0.0)
        return float(_move_towards(current_value, target_value, max_delta))

    def remap(self, raw_value: float) -> float:
        magnitude = abs(float(raw_value))
        if magnitude <= float(self.deadzone):
            return 0.0

        normalized = (magnitude - float(self.deadzone)) / max(
            1.0 - float(self.deadzone), 1.0e-6
        )
        normalized = float(np.clip(normalized, 0.0, 1.0))
        curved = normalized ** max(float(self.exponent), 0.01)
        return float(np.sign(raw_value) * curved)


class ActionResponseSmoother:
    """
    Stateful 4-axis smoother for actuator-speed command vectors.

    The smoothing profile is applied independently per output axis.
    """

    def __init__(
        self,
        *,
        deadzone: float | Sequence[float] = 0.05,
        attack_rate: float | Sequence[float] = 4.0,
        release_rate: float | Sequence[float] = 6.0,
        recenter_rate: float | Sequence[float] = 7.0,
        exponent: float | Sequence[float] = 1.0,
        default_dt: float = 0.02,
    ) -> None:
        deadzone_arr = _broadcast(deadzone, name="deadzone")
        attack_arr = _broadcast(attack_rate, name="attack_rate")
        release_arr = _broadcast(release_rate, name="release_rate")
        recenter_arr = _broadcast(recenter_rate, name="recenter_rate")
        exponent_arr = _broadcast(exponent, name="exponent")

        self._profiles = [
            AxisResponseProfile(
                deadzone=float(deadzone_arr[i]),
                attack_rate=float(attack_arr[i]),
                release_rate=float(release_arr[i]),
                recenter_rate=float(recenter_arr[i]),
                exponent=float(exponent_arr[i]),
            )
            for i in range(ACTION_DIM)
        ]
        self._default_dt = float(default_dt)
        self._current = np.zeros(ACTION_DIM, dtype=np.float32)
        self._last_step_time: float | None = None

    def reset(self) -> None:
        self._current.fill(0.0)
        self._last_step_time = None

    def apply(self, raw_action: np.ndarray, *, delta_time: float | None = None) -> np.ndarray:
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if raw_action.shape != (ACTION_DIM,):
            raise ValueError(f"raw_action must have shape ({ACTION_DIM},), got {raw_action.shape}")

        dt = self._resolve_dt(delta_time)
        for axis_idx, profile in enumerate(self._profiles):
            self._current[axis_idx] = profile.apply(
                raw_value=float(raw_action[axis_idx]),
                current_value=float(self._current[axis_idx]),
                delta_time=dt,
            )
        return self._current.copy()

    def _resolve_dt(self, delta_time: float | None) -> float:
        if delta_time is not None:
            dt = float(delta_time)
        else:
            now = time.perf_counter()
            if self._last_step_time is None:
                dt = self._default_dt
            else:
                dt = max(now - self._last_step_time, 0.0)
            self._last_step_time = now
        return max(dt, 0.0)
