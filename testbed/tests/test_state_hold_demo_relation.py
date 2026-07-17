from __future__ import annotations

import numpy as np
import pytest

from testbed.policies.state_hold_demo_relation import (
    evaluate_state_hold_trace_demo_relation,
)


def _thresholds() -> dict[str, dict[str, float]]:
    return {
        axis: {"pos": 0.5, "neg": 0.5} for axis in ("swing", "boom", "stick", "bucket")
    }


def test_trace_safety_uses_full_expert_direction_set_and_counts_flips() -> None:
    expert = np.array([0.0, 0.8, 0.0, -0.8], dtype=np.float32)
    trace = [
        [0.0, 0.8, 0.0, -0.8],
        [0.7, 0.8, 0.0, -0.8],
        [-0.7, -0.8, 0.0, -0.8],
    ]

    result = evaluate_state_hold_trace_demo_relation(
        expert_action=expert,
        action_trace=trace,
        thresholds=_thresholds(),
        target_axis_index=1,
        target_direction="pos",
    )

    assert result["single_demo_anchor_effective_directions"] == [
        "boom+",
        "bucket-",
    ]
    assert result["state_hold_anchor_extra_effective"] is True
    assert result["state_hold_anchor_extra_effective_tick_count"] == 2
    assert result["state_hold_anchor_extra_effective_direction_count"] == 3
    assert result["state_hold_anchor_extra_effective_tick_indices"] == [1, 2]
    assert result["state_hold_anchor_extra_effective_directions"] == [
        "swing+",
        "swing-",
        "boom-",
    ]
    assert result["state_hold_opposite_to_demo_target_tick_count"] == 1
    assert result["state_hold_demo_target_effective_tick_count"] == 2
    assert result["state_hold_direction_flip_count"] == 2
    assert result["state_hold_effective_axis_count_trace"] == [2, 3, 3]
    assert result["state_hold_max_effective_axes"] == 3


def test_trace_safety_accepts_empty_trace_and_rejects_bad_target() -> None:
    result = evaluate_state_hold_trace_demo_relation(
        expert_action=np.zeros(4, dtype=np.float32),
        action_trace=[],
        thresholds=_thresholds(),
        target_axis_index=0,
        target_direction="pos",
    )
    assert result["state_hold_anchor_extra_effective"] is False
    assert result["state_hold_max_effective_axes"] == 0

    with pytest.raises(ValueError, match="target_direction"):
        evaluate_state_hold_trace_demo_relation(
            expert_action=np.zeros(4, dtype=np.float32),
            action_trace=[],
            thresholds=_thresholds(),
            target_axis_index=0,
            target_direction="idle",
        )
