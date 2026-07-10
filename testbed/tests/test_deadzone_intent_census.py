import numpy as np

from testbed.policies.deadzone_eval import (
    aggregate_intent_census_rows,
    compute_intent_census_row,
)


def test_intent_census_counts_move_frames_and_axis_direction_events() -> None:
    thresholds = {
        "swing": {"pos": 0.5, "neg": 0.5},
        "boom": {"pos": 0.5, "neg": 0.5},
        "stick": {"pos": 0.5, "neg": 0.5},
        "bucket": {"pos": 0.5, "neg": 0.5},
    }
    actions = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.0],
            [0.0, -0.7, 0.0, 0.8],
            [0.0, 0.0, 0.1, 0.0],
        ],
        dtype=np.float32,
    )

    row = compute_intent_census_row(
        episode_id="episode_test",
        action=actions,
        thresholds=thresholds,
    )

    assert row["steps"] == 4
    assert row["should_move_frames"] == 2
    assert row["should_stop_frames"] == 2
    assert row["effective_axis_dir_events"] == 3
    assert row["multi_dir_move_frames"] == 1
    assert row["swing_pos_frames"] == 1
    assert row["boom_neg_frames"] == 1
    assert row["bucket_pos_frames"] == 1
    assert row["mean_effective_dirs_per_move_frame"] == 1.5

    aggregate = aggregate_intent_census_rows([row])
    assert aggregate[0]["episodes"] == 1
    assert aggregate[0]["total_steps"] == 4
    assert aggregate[0]["should_move_pct"] == 50.0
    assert aggregate[0]["effective_axis_dir_events"] == 3
