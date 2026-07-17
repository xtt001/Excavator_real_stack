from __future__ import annotations

import numpy as np

from testbed.policies.trajectory_deadzone_liveness import (
    apply_mechanical_deadzone_assist,
    evaluate_trajectory_liveness,
)


def _thresholds(value: float = 0.5) -> dict[str, dict[str, float]]:
    return {
        axis: {"pos": value, "neg": value}
        for axis in ("swing", "boom", "stick", "bucket")
    }


def test_dense_liveness_scores_future_crossing_and_underconfidence() -> None:
    expert = np.zeros((6, 4), dtype=np.float32)
    policy = np.zeros((6, 4), dtype=np.float32)
    expert[1:5, 1] = 0.8
    policy[1:3, 1] = 0.2
    policy[3:5, 1] = 0.7

    report = evaluate_trajectory_liveness(
        episode_id="episode_001",
        expert_action=expert,
        policy_action=policy,
        thresholds=_thresholds(),
        horizons=(1, 3),
        persist_steps=2,
    )

    summary = report["episode_summary"]
    assert summary["opportunities"] == 4
    assert summary["hit_count_h1"] == 2
    assert summary["hit_count_h3"] == 4
    assert summary["underconfidence_count"] == 2
    assert summary["zero_liveness_h1"] == 2
    axis_row = next(
        row
        for row in report["axis_direction_summary"]
        if row["axis"] == "boom" and row["direction"] == "pos"
    )
    assert axis_row["current_margin_min"] < 0.0
    assert axis_row["margin_median"] > -0.5


def test_multi_axis_targets_are_scored_independently() -> None:
    expert = np.zeros((3, 4), dtype=np.float32)
    policy = np.zeros((3, 4), dtype=np.float32)
    expert[0, 1] = 0.8
    expert[0, 3] = 0.8
    policy[0, 1] = 0.8
    policy[0, 3] = 0.2
    policy[1, 3] = 0.8

    report = evaluate_trajectory_liveness(
        episode_id="episode_002",
        expert_action=expert,
        policy_action=policy,
        thresholds=_thresholds(),
        horizons=(1, 2),
    )
    rows = {
        (row["axis"], row["direction"]): row
        for row in report["opportunities"]
        if row["step"] == 0
    }
    assert rows[("boom", "pos")]["hit_h1"] is True
    assert rows[("bucket", "pos")]["hit_h1"] is False
    assert rows[("bucket", "pos")]["hit_h2"] is True
    assert report["episode_summary"]["wrong_extra_active_frames"] == 0


def test_mechanical_assist_is_sequential_and_does_not_change_strong_action() -> None:
    thresholds = _thresholds()
    actions = np.zeros((4, 4), dtype=np.float32)
    actions[:3, 1] = 0.3
    actions[3, 1] = 0.8
    assisted = apply_mechanical_deadzone_assist(
        actions,
        thresholds,
        trigger_fraction=0.5,
        min_consecutive_steps=2,
        margin=0.02,
    )
    assert np.isclose(assisted[0, 1], 0.3)
    assert assisted[1, 1] > thresholds["boom"]["pos"]
    assert assisted[2, 1] > thresholds["boom"]["pos"]
    assert np.isclose(assisted[3, 1], 0.8)
