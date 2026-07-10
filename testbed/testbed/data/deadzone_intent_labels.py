"""Action-aligned deadzone intent labels for ACT training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


AXIS_NAMES = ("swing", "boom", "stick", "bucket")


@dataclass(frozen=True)
class DeadzoneIntentLabels:
    move_mask: np.ndarray
    stop_mask: np.ndarray
    wrong_mask: np.ndarray
    action_loss_mask: np.ndarray


def compute_deadzone_intent_labels(
    *,
    actions: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    action_loss_mask: np.ndarray | None = None,
    tail_idle_mask: np.ndarray | None = None,
    owner_automation: np.ndarray | None = None,
) -> DeadzoneIntentLabels:
    action_arr = _validate_actions(actions)
    n_steps = action_arr.shape[0]
    pos, neg = _threshold_arrays(thresholds)

    expert_pos = action_arr >= pos.reshape(1, -1)
    expert_neg = action_arr <= -neg.reshape(1, -1)
    expert_axis_dir = np.stack([expert_pos, expert_neg], axis=-1)
    expert_effective = np.any(expert_axis_dir, axis=(1, 2))

    action_valid = _optional_bool_mask(action_loss_mask, n_steps, default=True)
    tail_idle = _optional_bool_mask(tail_idle_mask, n_steps, default=False)
    automation = _optional_bool_mask(owner_automation, n_steps, default=False)
    forced_stop = (~action_valid) | tail_idle | automation

    move_mask = expert_axis_dir & (~forced_stop[:, None, None])
    stop_mask = (~expert_effective) | forced_stop

    wrong_mask = np.ones((n_steps, len(AXIS_NAMES), 2), dtype=bool)
    wrong_mask[expert_axis_dir & (~forced_stop[:, None, None])] = False

    return DeadzoneIntentLabels(
        move_mask=move_mask.astype(bool),
        stop_mask=stop_mask.astype(bool),
        wrong_mask=wrong_mask.astype(bool),
        action_loss_mask=action_valid.astype(bool),
    )


def masked_action_stats(actions: np.ndarray, action_loss_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action_arr = _validate_actions(actions)
    mask = _optional_bool_mask(action_loss_mask, action_arr.shape[0], default=False)
    selected = action_arr[mask]
    if selected.size == 0:
        raise ValueError("action_loss_mask selects no action rows")
    mean = selected.mean(axis=0).astype(np.float32)
    std = selected.std(axis=0).clip(min=1e-2).astype(np.float32)
    return mean, std


def _validate_actions(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"actions must have shape (T, A), got {arr.shape}")
    if arr.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"actions must have {len(AXIS_NAMES)} axes, got {arr.shape[1]}")
    return arr


def _threshold_arrays(thresholds: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    pos: list[float] = []
    neg: list[float] = []
    for axis in AXIS_NAMES:
        raw = thresholds.get(axis)
        if not isinstance(raw, dict):
            raise ValueError(f"thresholds missing axis {axis!r}")
        pos.append(_threshold_value(raw.get("pos"), axis=axis, direction="pos"))
        neg.append(_threshold_value(raw.get("neg"), axis=axis, direction="neg"))
    return np.asarray(pos, dtype=np.float32), np.asarray(neg, dtype=np.float32)


def _threshold_value(value: Any, *, axis: str, direction: str) -> float:
    if isinstance(value, dict):
        if "threshold_action_abs" in value:
            value = value["threshold_action_abs"]
        elif "value" in value:
            value = value["value"]
    if value is None:
        raise ValueError(f"threshold for {axis}.{direction} is missing")
    result = float(value)
    if result < 0.0:
        raise ValueError(f"threshold for {axis}.{direction} must be >= 0")
    return result


def _optional_bool_mask(mask: np.ndarray | None, n_steps: int, *, default: bool) -> np.ndarray:
    if mask is None:
        return np.full(n_steps, bool(default), dtype=bool)
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size != n_steps:
        raise ValueError(f"mask length must be {n_steps}, got {arr.size}")
    return arr
