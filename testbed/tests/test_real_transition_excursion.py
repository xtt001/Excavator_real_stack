from __future__ import annotations

import numpy as np

from testbed.tasks.real_transition_excursion import (
    derive_excursion_observed,
    excursion_chunk_valid_mask,
)


def test_excursion_state_latches_only_sustained_positive_displacement() -> None:
    qpos = np.zeros((8, 4), dtype=np.float32)
    qpos[:, 0] = [0.0, 0.09, 0.02, 0.081, 0.09, 0.10, 0.04, -0.2]

    state = derive_excursion_observed(
        qpos=qpos,
        minimum_delta_rad=0.08,
        minimum_consecutive_samples=3,
    )

    assert state[:, 0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_excursion_state_rejects_negative_shortcut() -> None:
    qpos = np.zeros((5, 4), dtype=np.float32)
    qpos[:, 0] = [0.0, -0.09, -0.10, -0.11, -0.12]

    state = derive_excursion_observed(
        qpos=qpos,
        minimum_delta_rad=0.08,
        minimum_consecutive_samples=3,
    )

    assert not bool(state.any())


def test_excursion_chunk_mask_stops_before_latch_boundary() -> None:
    state = np.asarray([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)

    mask = excursion_chunk_valid_mask(state, chunk_steps=3)

    assert mask.tolist() == [
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 0],
    ]
