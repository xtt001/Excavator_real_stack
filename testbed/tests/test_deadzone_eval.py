from __future__ import annotations

import numpy as np

from testbed.policies.deadzone_eval import (
    build_window_ranges,
    compute_deadzone_window_rows,
    longest_true_segment_with_gap,
)


def _thresholds(value: float = 0.5) -> dict[str, dict[str, float]]:
    return {
        axis: {"pos": value, "neg": value}
        for axis in ("swing", "boom", "stick", "bucket")
    }


def test_deadzone_window_counts_same_direction_and_extra_actions() -> None:
    expert = np.zeros((6, 4), dtype=np.float32)
    policy = np.zeros((6, 4), dtype=np.float32)
    expert[1:4, 1] = 0.8
    policy[2:4, 1] = 0.7
    policy[4, 3] = 0.9

    rows = compute_deadzone_window_rows(
        model="candidate",
        episode_id="episode_001",
        expert_action=expert,
        policy_action=policy,
        thresholds=_thresholds(),
        windows=("full_available",),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["single_demo_any_effective_frames"] == 3
    assert row["policy_any_effective_frames"] == 3
    assert row["single_demo_same_axis_direction_effective_frames"] == 2
    assert row["policy_outside_single_demo_frame_effective_frames"] == 1
    assert row[
        "single_demo_same_axis_direction_effective_pct_of_demo_effective"
    ] == 100.0 * 2.0 / 3.0
    assert row["policy_outside_single_demo_frame_effective_pct"] == 100.0 / 6.0
    assert row["policy_boom_pos_eff_pct"] == 100.0 * 2.0 / 6.0
    assert row["policy_bucket_pos_eff_pct"] == 100.0 / 6.0


def test_longest_true_segment_allows_short_gaps() -> None:
    mask = np.array([False, True, True, False, False, True, False, True, False, False, False, True])

    assert longest_true_segment_with_gap(mask, gap=2) == (1, 8)
    assert longest_true_segment_with_gap(mask, gap=0) == (1, 3)


def test_build_window_ranges_handles_empty_effective_segment() -> None:
    ranges = build_window_ranges(
        np.zeros(20, dtype=bool),
        total_steps=20,
        windows=("start40", "end80", "longest_single_demo_effective_segment_gap5"),
    )

    assert ranges == [
        ("start40", 0, 20),
        ("end80", 0, 20),
        ("longest_single_demo_effective_segment_gap5", 0, 0),
    ]
