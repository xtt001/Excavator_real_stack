"""Short-horizon transition features for trajectory-support evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from testbed.policies.offline_eval import AXIS_NAMES
from testbed.policies.trajectory_support_eval import effective_action_channels


@dataclass(frozen=True)
class TransitionSamples:
    start_steps: np.ndarray
    target_qpos_delta: np.ndarray
    initial_qvel_displacement: np.ndarray
    action_impulse: np.ndarray


def build_transition_samples(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
    dt: float,
    horizon_steps: int,
    stride: int,
    qvel_to_qpos_sign: np.ndarray,
    action_to_qpos_sign: np.ndarray,
) -> TransitionSamples:
    """Build aligned state targets and causal command features."""

    qpos_values = _validate_matrix(qpos, name="qpos")
    qvel_values = _validate_matrix(qvel, name="qvel")
    action_values = _validate_matrix(action, name="action")
    if qpos_values.shape != qvel_values.shape or qpos_values.shape != action_values.shape:
        raise ValueError(
            f"qpos, qvel, and action must share shape, got {qpos_values.shape}, "
            f"{qvel_values.shape}, {action_values.shape}"
        )
    step_s = float(dt)
    if not np.isfinite(step_s) or step_s <= 0.0:
        raise ValueError("dt must be finite and positive")
    horizon = int(horizon_steps)
    if horizon <= 0 or horizon >= qpos_values.shape[0]:
        raise ValueError(f"horizon_steps must satisfy 0 < horizon < {qpos_values.shape[0]}")
    step_stride = int(stride)
    if step_stride <= 0:
        raise ValueError("stride must be positive")
    qvel_sign = _validate_signs(qvel_to_qpos_sign, name="qvel_to_qpos_sign")
    action_sign = _validate_signs(action_to_qpos_sign, name="action_to_qpos_sign")

    starts = np.arange(0, qpos_values.shape[0] - horizon, step_stride, dtype=np.int64)
    target = qpos_values[starts + horizon] - qpos_values[starts]
    target[:, 0] = (target[:, 0] + np.pi) % (2.0 * np.pi) - np.pi
    initial_velocity = qvel_values[starts] * qvel_sign.reshape(1, -1) * (horizon * step_s)
    channels = effective_action_channels(action_values, thresholds)
    signed_effective = (channels[:, :, 0] - channels[:, :, 1]) * action_sign.reshape(1, -1)
    impulse = np.stack(
        [signed_effective[start : start + horizon].sum(axis=0) * step_s for start in starts],
        axis=0,
    )
    return TransitionSamples(
        start_steps=starts,
        target_qpos_delta=target,
        initial_qvel_displacement=initial_velocity,
        action_impulse=impulse,
    )


def _validate_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"{name} must have shape (T, {len(AXIS_NAMES)}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _validate_signs(values: np.ndarray, *, name: str) -> np.ndarray:
    signs = np.asarray(values, dtype=np.float64)
    if signs.shape != (len(AXIS_NAMES),) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError(f"{name} must contain four values chosen from -1 and 1")
    return signs
