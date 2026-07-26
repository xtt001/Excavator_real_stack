from __future__ import annotations

import numpy as np

from testbed.simverify.m3_transition_delta_stitch import (
    aggregate_paired_rollouts,
    derive_delta_stitch_thresholds,
    exact_nearest_unseen_indices,
    run_delta_stitch_rollouts,
)
from testbed.simverify.m3_transition_stitch import ACTION_DIM, STATE_DIM


def test_exact_neighbor_excludes_current_episode_and_consumed_row() -> None:
    bank = np.zeros((4, 2), dtype=np.float32)
    bank[:, 0] = [0.0, 0.1, 0.2, 0.3]
    distances, indices = exact_nearest_unseen_indices(
        bank,
        np.zeros((1, 2), dtype=np.float32),
        bank_episode_ids=np.asarray([1, 2, 3, 4]),
        query_episode_ids=np.asarray([1]),
        seen_indices=[{1}],
        device="cpu",
        batch_size=1,
    )
    assert indices.tolist() == [2]
    assert np.isclose(distances[0], 0.2)


def test_delta_rollout_integrates_local_delta_without_transition_reuse() -> None:
    count = 3
    bank = {
        "retrieval": np.zeros((count, STATE_DIM + ACTION_DIM), dtype=np.float32),
        "episode_id": np.asarray([2, 3, 4], dtype=np.int64),
        "progress": np.asarray([4.0, 0.0, 2.0], dtype=np.float32),
        "next_progress": np.asarray([4.5, 2.5, 4.0], dtype=np.float32),
        "next_state_standardized": np.zeros((count, STATE_DIM), dtype=np.float32),
        "next_action_standardized": np.zeros((count, ACTION_DIM), dtype=np.float32),
    }
    initial = {
        "episode_id": np.asarray([1], dtype=np.int64),
        "cycle_id": np.asarray([7], dtype=np.int64),
        "state_standardized": np.zeros((1, STATE_DIM), dtype=np.float32),
        "action_standardized": np.zeros((1, ACTION_DIM), dtype=np.float32),
    }
    rows = run_delta_stitch_rollouts(
        bank=bank,
        initial=initial,
        initial_indices=[0],
        support_radius=1.0,
        max_steps=3,
        action_mode="recorded_expert",
        device="cpu",
        batch_size=1,
    )
    assert rows[0]["completed"] is True
    assert rows[0]["unique_transition_count"] == 3
    assert np.isclose(rows[0]["accumulated_progress"], 5.0)
    assert rows[0]["absolute_progress_used_for_rollout_state"] is False


def test_train_thresholds_require_action_dependent_completion() -> None:
    candidate = [
        {
            "episode_id": 3,
            "cycle_id": 1,
            "completed": True,
            "steps": 10,
            "max_retrieval_distance": 0.2,
        },
        {
            "episode_id": 4,
            "cycle_id": 2,
            "completed": True,
            "steps": 12,
            "max_retrieval_distance": 0.3,
        },
    ]
    null = [
        {
            "episode_id": 3,
            "cycle_id": 1,
            "completed": False,
            "steps": 20,
            "max_retrieval_distance": 0.2,
        },
        {
            "episode_id": 4,
            "cycle_id": 2,
            "completed": False,
            "steps": 20,
            "max_retrieval_distance": 0.3,
        },
    ]
    episode_rows = aggregate_paired_rollouts(candidate, null)
    thresholds = derive_delta_stitch_thresholds(episode_rows)
    assert thresholds["candidate_completion_rate_lower"] == 1.0
    assert thresholds["paired_completion_delta_lower"] == 1.0
    assert thresholds["median_action_null_completion_rate_upper"] == 0.0
