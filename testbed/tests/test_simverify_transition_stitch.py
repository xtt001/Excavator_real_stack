from __future__ import annotations

import numpy as np

from testbed.simverify.m3_transition_stitch import (
    ACTION_DIM,
    STATE_DIM,
    apply_standardization,
    compose_retrieval_vectors,
    exact_nearest_indices,
    fit_retrieval_normalization,
)


def test_retrieval_vectors_give_state_and_action_equal_group_weight() -> None:
    state = np.zeros((2, STATE_DIM), dtype=np.float32)
    action = np.zeros((2, ACTION_DIM), dtype=np.float32)
    state[1] = 1.0
    action[1] = 1.0
    vectors = compose_retrieval_vectors(state, action)
    squared = float(np.sum((vectors[1] - vectors[0]) ** 2))
    assert np.isclose(squared, 2.0)


def test_robust_normalization_uses_train_values_only() -> None:
    state = np.stack(
        (
            np.zeros(STATE_DIM, dtype=np.float32),
            np.full(STATE_DIM, 2.0, dtype=np.float32),
        )
    )
    action = np.stack(
        (
            np.zeros(ACTION_DIM, dtype=np.float32),
            np.full(ACTION_DIM, 4.0, dtype=np.float32),
        )
    )
    contract = fit_retrieval_normalization(state, action)
    transformed = apply_standardization(state, contract["state"])
    np.testing.assert_allclose(transformed[0], -1.0)
    np.testing.assert_allclose(transformed[1], 1.0)


def test_exact_neighbor_excludes_same_source_episode() -> None:
    bank = np.zeros((3, STATE_DIM + ACTION_DIM), dtype=np.float32)
    bank[1, 0] = 0.1
    bank[2, 0] = 1.0
    query = np.zeros((1, STATE_DIM + ACTION_DIM), dtype=np.float32)
    distances, indices = exact_nearest_indices(
        bank,
        query,
        bank_episode_ids=np.asarray([1, 2, 3]),
        query_episode_ids=np.asarray([1]),
        exclude_same_episode=True,
        device="cpu",
        batch_size=1,
    )
    assert indices.tolist() == [1]
    assert np.isclose(distances[0], 0.1)
