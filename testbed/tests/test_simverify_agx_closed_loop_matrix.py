from __future__ import annotations

import numpy as np

from testbed.simverify.agx_closed_loop_matrix import (
    classify_swing_sector,
    extract_observable_cycle_entries,
)


def _row(
    tick: int,
    *,
    swing: float,
    bucket_action: float,
) -> dict:
    action = np.zeros(4, dtype=np.float32)
    action[3] = bucket_action
    return {
        "policy_tick": tick,
        "qpos": [swing, 0.0, 0.0, 0.0],
        "actual_sent_action": action.astype(float).tolist(),
        "cycle_index": 0,
        "condition_route": {"route": "current", "route_index": 0},
    }


def test_sector_classification_preserves_frozen_review_margin() -> None:
    boundaries = (0.52, 0.58)
    assert (
        classify_swing_sector(
            0.50,
            boundaries=boundaries,
            review_margin=0.01,
        )
        == "left"
    )
    assert (
        classify_swing_sector(
            0.525,
            boundaries=boundaries,
            review_margin=0.01,
        )
        == "boundary_review"
    )
    assert (
        classify_swing_sector(
            0.55,
            boundaries=boundaries,
            review_margin=0.01,
        )
        == "center"
    )
    assert (
        classify_swing_sector(
            0.60,
            boundaries=boundaries,
            review_margin=0.01,
        )
        == "right"
    )


def test_cycle_entries_require_dump_release_between_positive_bucket_runs() -> None:
    rows = [
        _row(0, swing=0.49, bucket_action=0.2),
        _row(1, swing=0.50, bucket_action=0.2),
        _row(2, swing=0.55, bucket_action=0.0),
        _row(3, swing=0.70, bucket_action=-0.2),
        _row(4, swing=0.69, bucket_action=-0.2),
        _row(5, swing=0.60, bucket_action=0.0),
        _row(6, swing=0.55, bucket_action=0.2),
        _row(7, swing=0.56, bucket_action=0.2),
        # A short gap does not create a third cycle without another release.
        _row(8, swing=0.57, bucket_action=0.0),
        _row(9, swing=0.56, bucket_action=0.2),
        _row(10, swing=0.57, bucket_action=0.2),
        _row(11, swing=0.70, bucket_action=-0.2),
        _row(12, swing=0.69, bucket_action=-0.2),
        _row(13, swing=0.60, bucket_action=0.0),
        _row(14, swing=0.60, bucket_action=0.2),
        _row(15, swing=0.61, bucket_action=0.2),
    ]

    result = extract_observable_cycle_entries(
        rows,
        action_deadzone=0.05,
        dump_swing_threshold=0.63,
        minimum_policy_ticks=2,
        sector_boundaries=(0.52, 0.58),
        sector_review_margin=0.005,
    )

    assert [entry["policy_tick"] for entry in result["dig_entries"]] == [
        0,
        6,
        14,
    ]
    assert [entry["sector"] for entry in result["dig_entries"]] == [
        "left",
        "center",
        "right",
    ]
    assert [
        (event["start_tick"], event["end_tick"]) for event in result["dump_releases"]
    ] == [
        (3, 5),
        (11, 13),
    ]
