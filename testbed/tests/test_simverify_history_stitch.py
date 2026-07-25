from __future__ import annotations

import numpy as np

from testbed.simverify.m3_transition_stitch import ACTION_DIM, STATE_DIM
from testbed.simverify.m3_transition_stitch_history import (
    compose_history_retrieval_vectors,
    derive_history_deltas,
)


def test_history_retrieval_gives_four_groups_equal_weight() -> None:
    state = np.zeros((2, STATE_DIM), dtype=np.float32)
    state_delta = np.zeros((2, STATE_DIM), dtype=np.float32)
    action = np.zeros((2, ACTION_DIM), dtype=np.float32)
    action_delta = np.zeros((2, ACTION_DIM), dtype=np.float32)
    state[1] = 1.0
    state_delta[1] = 1.0
    action[1] = 1.0
    action_delta[1] = 1.0
    vectors = compose_history_retrieval_vectors(
        state,
        state_delta,
        action,
        action_delta,
    )
    assert np.isclose(np.sum((vectors[1] - vectors[0]) ** 2), 4.0)


def test_history_delta_resets_at_cycle_start() -> None:
    nodes = {
        "episode_id": np.asarray([1, 1, 1]),
        "cycle_id": np.asarray([0, 0, 1]),
        "tick": np.asarray([10, 11, 11]),
        "state": np.stack(
            (
                np.zeros(STATE_DIM),
                np.ones(STATE_DIM),
                np.full(STATE_DIM, 2.0),
            )
        ).astype(np.float32),
        "action": np.stack(
            (
                np.zeros(ACTION_DIM),
                np.ones(ACTION_DIM),
                np.full(ACTION_DIM, 2.0),
            )
        ).astype(np.float32),
    }
    state_delta, action_delta = derive_history_deltas(nodes)
    np.testing.assert_allclose(state_delta[0], 0.0)
    np.testing.assert_allclose(state_delta[1], 1.0)
    np.testing.assert_allclose(state_delta[2], 0.0)
    np.testing.assert_allclose(action_delta[2], 0.0)
