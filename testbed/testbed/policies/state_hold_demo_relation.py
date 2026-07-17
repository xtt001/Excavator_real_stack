"""Descriptive relation between a state-hold trace and one demo anchor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from testbed.policies.deadzone_eval import effective_direction_mask
from testbed.policies.offline_eval import AXIS_NAMES

_DIRECTIONS = ("pos", "neg")


def evaluate_state_hold_trace_demo_relation(
    *,
    expert_action: np.ndarray,
    action_trace: Sequence[Sequence[float]],
    thresholds: Mapping[str, Mapping[str, float]],
    target_axis_index: int,
    target_direction: str,
) -> dict[str, Any]:
    """Compare held ticks with one demo anchor without judging correctness."""

    expert = np.asarray(expert_action, dtype=np.float32).reshape(-1)
    if expert.shape != (len(AXIS_NAMES),):
        raise ValueError(
            f"expert_action must have shape ({len(AXIS_NAMES)},), got {expert.shape}"
        )
    trace = np.asarray(action_trace, dtype=np.float32)
    if trace.size == 0:
        trace = np.zeros((0, len(AXIS_NAMES)), dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != len(AXIS_NAMES):
        raise ValueError(
            f"action_trace must have shape (T, {len(AXIS_NAMES)}), got {trace.shape}"
        )
    if not np.all(np.isfinite(expert)) or not np.all(np.isfinite(trace)):
        raise ValueError("expert_action and action_trace must be finite")
    axis_index = int(target_axis_index)
    if not 0 <= axis_index < len(AXIS_NAMES):
        raise ValueError(f"target_axis_index is out of range: {axis_index}")
    direction = str(target_direction)
    if direction not in _DIRECTIONS:
        raise ValueError("target_direction must be 'pos' or 'neg'")

    normalized_thresholds = {
        axis: {side: float(thresholds[axis][side]) for side in _DIRECTIONS}
        for axis in AXIS_NAMES
    }
    expert_effective = effective_direction_mask(
        expert.reshape(1, -1), normalized_thresholds
    )[0]
    trace_effective = effective_direction_mask(trace, normalized_thresholds)
    anchor_extra = trace_effective & ~expert_effective[None, :, :]
    anchor_extra_tick_mask = anchor_extra.any(axis=(1, 2))
    effective_axis_count = trace_effective.any(axis=2).sum(axis=1)

    opposite_index = 1 if direction == "pos" else 0
    target_index = 0 if direction == "pos" else 1
    target_opposite = trace_effective[:, axis_index, opposite_index]
    target_effective = trace_effective[:, axis_index, target_index]
    flips = _direction_flip_count(trace_effective)
    labels = [
        f"{axis}{'+' if side == 'pos' else '-'}"
        for axis_index_value, axis in enumerate(AXIS_NAMES)
        for side_index, side in enumerate(_DIRECTIONS)
        if bool(np.any(anchor_extra[:, axis_index_value, side_index]))
    ]
    return {
        "single_demo_anchor_effective_directions": [
            f"{axis}{'+' if side == 'pos' else '-'}"
            for axis_index_value, axis in enumerate(AXIS_NAMES)
            for side_index, side in enumerate(_DIRECTIONS)
            if bool(expert_effective[axis_index_value, side_index])
        ],
        "state_hold_anchor_extra_effective": bool(np.any(anchor_extra)),
        "state_hold_anchor_extra_effective_tick_count": int(
            np.count_nonzero(anchor_extra_tick_mask)
        ),
        "state_hold_anchor_extra_effective_direction_count": int(
            np.count_nonzero(anchor_extra)
        ),
        "state_hold_anchor_extra_effective_tick_indices": np.flatnonzero(
            anchor_extra_tick_mask
        )
        .astype(int)
        .tolist(),
        "state_hold_anchor_extra_effective_directions": labels,
        "state_hold_opposite_to_demo_target_tick_count": int(
            np.count_nonzero(target_opposite)
        ),
        "state_hold_demo_target_effective_tick_count": int(
            np.count_nonzero(target_effective)
        ),
        "state_hold_direction_flip_count": flips,
        "state_hold_effective_axis_count_trace": effective_axis_count.astype(
            int
        ).tolist(),
        "state_hold_max_effective_axes": (
            int(effective_axis_count.max()) if effective_axis_count.size else 0
        ),
    }


def _direction_flip_count(effective: np.ndarray) -> int:
    signs = np.where(
        effective[:, :, 0],
        1,
        np.where(effective[:, :, 1], -1, 0),
    )
    last = np.zeros(len(AXIS_NAMES), dtype=np.int8)
    flips = 0
    for row in signs:
        active = row != 0
        flips += int(np.count_nonzero(active & (last != 0) & (row != last)))
        last = np.where(active, row, last).astype(np.int8)
    return flips
