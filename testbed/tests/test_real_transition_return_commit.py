from __future__ import annotations

import numpy as np

from testbed.tasks.real_transition_return_commit import (
    RETURN_COMMIT_KEY,
    build_return_commit_contract,
    derive_return_commit,
    return_commit_chunk_valid_mask,
)


def test_derive_return_commit_uses_final_negative_segment() -> None:
    action = np.zeros((10, 4), dtype=np.float32)
    action[3:5, 0] = -0.08
    action[7:, 0] = -0.1
    excursion = np.asarray([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    phase = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32)

    result = derive_return_commit(
        action=action,
        excursion_observed=excursion,
        return_phase=phase,
        chunk_steps=4,
    )

    assert result.evaluable is True
    assert result.event_row == 7
    np.testing.assert_array_equal(result.state[:, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1])


def test_derive_return_commit_uses_segment_ending_before_return_row() -> None:
    action = np.zeros((9, 4), dtype=np.float32)
    action[6:8, 0] = -0.08
    excursion = np.asarray([0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    phase = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)

    result = derive_return_commit(
        action=action,
        excursion_observed=excursion,
        return_phase=phase,
        chunk_steps=3,
    )

    assert result.evaluable is True
    assert result.event_row == 6


def test_derive_return_commit_is_invalid_without_return_phase() -> None:
    action = np.zeros((6, 4), dtype=np.float32)
    excursion = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.float32)
    phase = np.zeros(6, dtype=np.float32)

    result = derive_return_commit(
        action=action,
        excursion_observed=excursion,
        return_phase=phase,
        chunk_steps=3,
    )

    assert result.evaluable is False
    assert result.event_row is None
    assert result.reason == "missing_return_phase"
    assert not result.state.any()
    assert not result.valid_mask.any()


def test_return_commit_chunk_mask_stops_at_boundary() -> None:
    state = np.asarray([[0], [0], [1], [1]], dtype=np.float32)

    mask = return_commit_chunk_valid_mask(state, chunk_steps=3)

    np.testing.assert_array_equal(
        mask,
        np.asarray(
            [
                [1, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_return_commit_contract_declares_hindsight_and_planner_owner() -> None:
    contract = build_return_commit_contract()

    assert contract["condition_key"] == RETURN_COMMIT_KEY
    assert contract["derivation"]["causal_online"] is False
    assert contract["runtime"]["owner"] == "planner_or_explicit_task_command"
