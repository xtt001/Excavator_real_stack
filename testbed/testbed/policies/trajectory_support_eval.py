"""Effective-action intent integration for trajectory-support evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any

import numpy as np

from testbed.policies.offline_eval import AXIS_NAMES


DIRECTION_NAMES = ("pos", "neg")
_NUMERIC_EPSILON = 1.0e-7


def effective_action_channels(
    actions: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    """Return non-negative post-deadzone magnitudes in axis-major pos/neg order."""

    action = _validate_actions(actions)
    out = np.zeros((action.shape[0], len(AXIS_NAMES), 2), dtype=np.float64)
    for axis_idx, axis in enumerate(AXIS_NAMES):
        axis_thresholds = thresholds.get(axis)
        if axis_thresholds is None:
            raise ValueError(f"thresholds are missing axis {axis!r}")
        pos = _validate_threshold(axis_thresholds.get("pos"), axis=axis, direction="pos")
        neg = _validate_threshold(axis_thresholds.get("neg"), axis=axis, direction="neg")
        pos_excess = action[:, axis_idx] - pos
        neg_excess = -action[:, axis_idx] - neg
        out[:, axis_idx, 0] = np.where(
            pos_excess > _NUMERIC_EPSILON,
            pos_excess / (1.0 - pos),
            0.0,
        )
        out[:, axis_idx, 1] = np.where(
            neg_excess > _NUMERIC_EPSILON,
            neg_excess / (1.0 - neg),
            0.0,
        )
    return out


def cumulative_intent(channels: np.ndarray, *, dt: float) -> np.ndarray:
    """Integrate effective channels with an explicit zero-valued origin row."""

    values = _validate_channels(channels)
    step_s = _validate_dt(dt)
    cumulative = np.cumsum(values, axis=0, dtype=np.float64) * step_s
    origin = np.zeros((1, len(AXIS_NAMES), 2), dtype=np.float64)
    return np.concatenate([origin, cumulative], axis=0)


def impulse_metrics(
    expert_impulse: np.ndarray,
    policy_impulse: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, float | None]:
    """Compare separate positive/negative impulse channels without cancellation."""

    expert = _validate_impulse(expert_impulse, name="expert_impulse")
    policy = _validate_impulse(policy_impulse, name="policy_impulse")
    eps = float(epsilon)
    if not isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    expert_total = float(expert.sum())
    policy_total = float(policy.sum())
    expert_norm = float(np.linalg.norm(expert.reshape(-1)))
    policy_norm = float(np.linalg.norm(policy.reshape(-1)))
    direction_cosine = (
        float(np.dot(expert.reshape(-1), policy.reshape(-1)) / (expert_norm * policy_norm))
        if expert_norm > eps and policy_norm > eps
        else None
    )
    expert_net = expert[:, 0] - expert[:, 1]
    policy_net = policy[:, 0] - policy[:, 1]
    return {
        "expert_total_impulse": expert_total,
        "policy_total_impulse": policy_total,
        "magnitude_ratio": policy_total / expert_total if expert_total > eps else None,
        "direction_cosine": direction_cosine,
        "missing_expert_impulse": float(np.maximum(expert - policy, 0.0).sum()),
        "extra_policy_impulse": float(np.maximum(policy - expert, 0.0).sum()),
        "channel_l1_error": float(np.abs(policy - expert).sum()),
        "net_axis_l1_error": float(np.abs(policy_net - expert_net).sum()),
        "expert_cancellation_ratio": _cancellation_ratio(expert, epsilon=eps),
        "policy_cancellation_ratio": _cancellation_ratio(policy, epsilon=eps),
    }


def compute_intent_horizon_rows(
    expert_action: np.ndarray,
    policy_action: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
    *,
    dt: float,
    horizons: Iterable[int],
    start: int = 0,
    end: int | None = None,
    stride: int = 1,
) -> list[dict[str, Any]]:
    """Compute rolling effective-intent comparisons for explicit horizons."""

    expert = _validate_actions(expert_action)
    policy = _validate_actions(policy_action)
    if expert.shape != policy.shape:
        raise ValueError(f"expert_action and policy_action must share shape, got {expert.shape} vs {policy.shape}")
    step_s = _validate_dt(dt)
    range_start = int(start)
    range_end = expert.shape[0] if end is None else int(end)
    if not 0 <= range_start <= range_end <= expert.shape[0]:
        raise ValueError(
            f"window must satisfy 0 <= start <= end <= {expert.shape[0]}, got {range_start}:{range_end}"
        )
    step_stride = int(stride)
    if step_stride <= 0:
        raise ValueError("stride must be positive")

    expert_channels = effective_action_channels(expert, thresholds)
    policy_channels = effective_action_channels(policy, thresholds)
    rows: list[dict[str, Any]] = []
    for raw_horizon in horizons:
        horizon = int(raw_horizon)
        if horizon <= 0:
            raise ValueError(f"horizons must contain positive integers, got {raw_horizon!r}")
        last_start = range_end - horizon
        for anchor in range(range_start, last_start + 1, step_stride):
            stop = anchor + horizon
            expert_window = expert_channels[anchor:stop]
            policy_window = policy_channels[anchor:stop]
            expert_path = cumulative_intent(expert_window, dt=step_s)
            policy_path = cumulative_intent(policy_window, dt=step_s)
            expert_impulse = expert_path[-1]
            policy_impulse = policy_path[-1]
            row: dict[str, Any] = {
                "start_step": int(anchor),
                "end_step_exclusive": int(stop),
                "horizon_steps": int(horizon),
                "horizon_seconds": float(horizon * step_s),
                **impulse_metrics(expert_impulse, policy_impulse),
                "cumulative_path_mean_channel_l1": float(
                    np.mean(np.abs(policy_path[1:] - expert_path[1:]).sum(axis=(1, 2)))
                ),
                "cumulative_path_max_channel_l1": float(
                    np.max(np.abs(policy_path[1:] - expert_path[1:]).sum(axis=(1, 2)))
                ),
            }
            for axis_idx, axis in enumerate(AXIS_NAMES):
                for direction_idx, direction in enumerate(DIRECTION_NAMES):
                    row[f"expert_{axis}_{direction}_impulse"] = float(expert_impulse[axis_idx, direction_idx])
                    row[f"policy_{axis}_{direction}_impulse"] = float(policy_impulse[axis_idx, direction_idx])
            rows.append(row)
    return rows


def _validate_actions(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    expected = ("T", len(AXIS_NAMES))
    if values.ndim != 2 or values.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"actions must have shape {expected}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("actions must contain only finite values")
    return values


def _validate_channels(channels: np.ndarray) -> np.ndarray:
    values = np.asarray(channels, dtype=np.float64)
    expected = ("T", len(AXIS_NAMES), len(DIRECTION_NAMES))
    if values.ndim != 3 or values.shape[1:] != expected[1:]:
        raise ValueError(f"channels must have shape {expected}, got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("channels must contain finite non-negative values")
    return values


def _validate_impulse(impulse: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(impulse, dtype=np.float64)
    expected = (len(AXIS_NAMES), len(DIRECTION_NAMES))
    if values.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return values


def _validate_threshold(value: Any, *, axis: str, direction: str) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"threshold for {axis}.{direction} must be numeric") from exc
    if not isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold for {axis}.{direction} must satisfy 0 <= value < 1")
    return threshold


def _validate_dt(dt: float) -> float:
    step_s = float(dt)
    if not isfinite(step_s) or step_s <= 0.0:
        raise ValueError("dt must be finite and positive")
    return step_s


def _cancellation_ratio(impulse: np.ndarray, *, epsilon: float) -> float:
    total = float(impulse.sum())
    if total <= epsilon:
        return 0.0
    return float(2.0 * np.minimum(impulse[:, 0], impulse[:, 1]).sum() / total)
