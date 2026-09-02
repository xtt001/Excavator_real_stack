from __future__ import annotations

import numpy as np
import pytest

from testbed.data.task_state_v2 import (
    build_task_state_sequence,
    task_state_candidate_starts,
    task_state_chunk_valid_mask,
    task_state_vector,
)


def test_task_state_hides_next_target_until_return_commit() -> None:
    digging = task_state_vector(
        current_side="B",
        dig_target="B",
        next_target="A",
        dig_complete=0,
        return_commit=0,
    )
    settled = task_state_vector(
        current_side="B",
        dig_target="B",
        next_target="A",
        dig_complete=1,
        return_commit=0,
    )
    returning = task_state_vector(
        current_side="B",
        dig_target="B",
        next_target="A",
        dig_complete=1,
        return_commit=1,
    )

    np.testing.assert_allclose(digging, [1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(settled, [1.0, 1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(returning, [1.0, 1.0, 1.0, 1.0, -1.0])


def test_task_state_preserves_commit_before_complete_overlap() -> None:
    sequence = build_task_state_sequence(
        total_steps=20,
        current_side="B",
        dig_target="B",
        next_target="A",
        work_complete_row=12,
        return_commit_row=8,
    )

    np.testing.assert_allclose(sequence[7], [1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(sequence[8], [1.0, 1.0, 0.0, 1.0, -1.0])
    np.testing.assert_allclose(sequence[12], [1.0, 1.0, 1.0, 1.0, -1.0])


def test_task_state_rejects_unrepresented_independent_dig_target() -> None:
    with pytest.raises(ValueError, match="must equal dig_target"):
        task_state_vector(
            current_side="A",
            dig_target="B",
            next_target="A",
            dig_complete=0,
            return_commit=0,
        )


@pytest.mark.parametrize(
    ("work_complete_row", "return_commit_row"), [(10, 15), (15, 10)]
)
def test_sampling_balances_start_body_boundary_and_return(
    work_complete_row: int, return_commit_row: int
) -> None:
    candidates = task_state_candidate_starts(
        total_steps=40,
        work_complete_row=work_complete_row,
        return_commit_row=return_commit_row,
        action_window_steps=5,
    ).by_name()

    np.testing.assert_array_equal(candidates["work_start"], [0])
    np.testing.assert_array_equal(candidates["work_body"], np.arange(1, 6))
    np.testing.assert_array_equal(candidates["boundary_state"], [10])
    np.testing.assert_array_equal(candidates["return_body"], np.arange(15, 36))


def test_chunk_mask_stops_at_each_task_state_change_and_episode_end() -> None:
    before_complete = task_state_chunk_valid_mask(
        timestep=7,
        total_steps=20,
        action_chunk_size=8,
        work_complete_row=10,
        return_commit_row=15,
    )
    at_complete = task_state_chunk_valid_mask(
        timestep=10,
        total_steps=20,
        action_chunk_size=8,
        work_complete_row=10,
        return_commit_row=15,
    )
    tail = task_state_chunk_valid_mask(
        timestep=17,
        total_steps=20,
        action_chunk_size=8,
        work_complete_row=10,
        return_commit_row=15,
    )

    np.testing.assert_array_equal(
        before_complete, [True, True, True, False, False, False, False, False]
    )
    np.testing.assert_array_equal(
        at_complete, [True, True, True, True, True, False, False, False]
    )
    np.testing.assert_array_equal(
        tail, [True, True, True, False, False, False, False, False]
    )
