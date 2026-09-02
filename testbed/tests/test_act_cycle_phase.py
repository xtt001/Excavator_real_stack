from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from testbed.policies.act.cycle_phase import (
    cycle_phase_candidate_indices,
    cycle_phase_loss_terms,
    resolve_cycle_phase_loss_config,
)
from testbed.tasks.real_transition_phase import (
    CYCLE_PHASE_KEY,
    derive_cycle_phase,
    phase_chunk_valid_mask,
)


def _config(tmp_path: Path, *, weight: float = 1.0) -> dict:
    thresholds = tmp_path / "deadzone.json"
    thresholds.write_text(
        json.dumps(
            {
                "deadzone_action": {
                    "swing": {"pos": 0.661, "neg": 0.721}
                }
            }
        ),
        encoding="utf-8",
    )
    return resolve_cycle_phase_loss_config(
        {
            "enabled": True,
            "scope": "train_and_validation",
            "condition_key": CYCLE_PHASE_KEY,
            "threshold_json": str(thresholds),
            "action_window_steps": 2,
            "append_samples_per_episode": 1,
            "weight": weight,
        }
    )


def test_cycle_phase_is_causal_and_latches_only_after_positive_excursion() -> None:
    qpos = np.zeros((9, 4), dtype=np.float32)
    qvel = np.zeros_like(qpos)
    qpos[:, 0] = [0.2, 0.24, 0.29, 0.31, 0.50, 0.48, 0.43, 0.40, 0.38]
    qvel[6:, 0] = -0.2

    phase = derive_cycle_phase(
        qpos=qpos,
        qvel=qvel,
        excursion_min_delta_rad=0.08,
        excursion_min_consecutive_samples=2,
    )

    np.testing.assert_array_equal(phase[:6, 0], np.zeros(6))
    np.testing.assert_array_equal(phase[6:, 0], np.ones(3))
    valid = phase_chunk_valid_mask(phase, chunk_steps=3)
    assert valid[4].tolist() == [1, 1, 0]
    assert valid[6].tolist() == [1, 1, 1]


def test_negative_direct_to_target_motion_never_counts_as_excursion() -> None:
    qpos = np.zeros((8, 4), dtype=np.float32)
    qvel = np.zeros_like(qpos)
    qpos[:, 0] = [0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4]
    qvel[:, 0] = -0.2

    phase = derive_cycle_phase(
        qpos=qpos,
        qvel=qvel,
        excursion_min_delta_rad=0.08,
        excursion_min_consecutive_samples=3,
    )

    np.testing.assert_array_equal(phase, np.zeros((8, 1), dtype=np.float32))


def test_cycle_phase_candidates_and_direct_counterfactual_loss(tmp_path: Path) -> None:
    config = _config(tmp_path)
    actions = np.zeros((8, 4), dtype=np.float32)
    actions[1:3, 0] = 0.8
    actions[5:7, 0] = -0.8
    phase = np.asarray([[0], [0], [0], [0], [1], [1], [1], [1]], np.float32)
    valid = phase_chunk_valid_mask(phase, chunk_steps=2)
    candidates = cycle_phase_candidate_indices(
        actions=actions,
        phase=phase,
        valid_starts=np.arange(8),
        phase_valid_mask=valid,
        config=config,
    )
    np.testing.assert_array_equal(candidates["pre_positive"], [1])
    np.testing.assert_array_equal(candidates["return_negative"], [5])

    primary = torch.tensor([[[0.8], [0.8]], [[-0.8], [-0.8]]])
    counterfactual = torch.tensor([[[0.0], [0.0]], [[0.0], [0.0]]])
    terms = cycle_phase_loss_terms(
        primary_direct=primary,
        counterfactual_direct=counterfactual,
        return_primary=torch.tensor([False, True]),
        valid=torch.tensor([True, True]),
        config=config,
    )
    assert terms["cycle_phase_loss"].item() == 0.0
    assert terms["cycle_phase_pre_return_no_shortcut_rate"].item() == 1.0

    shortcut = counterfactual.clone()
    shortcut[1, :, 0] = -0.9
    failed = cycle_phase_loss_terms(
        primary_direct=primary,
        counterfactual_direct=shortcut,
        return_primary=torch.tensor([False, True]),
        valid=torch.tensor([True, True]),
        config=config,
    )
    assert failed["cycle_phase_loss"].item() > 0.0
    assert failed["cycle_phase_pre_return_no_shortcut_rate"].item() == 0.0
