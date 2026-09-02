from __future__ import annotations

import numpy as np
import torch

from testbed.policies.act.excursion_observed import (
    excursion_observed_candidate_indices,
    excursion_observed_loss_terms,
)


def _config() -> dict:
    return {
        "enabled": True,
        "scope": "train_and_validation",
        "condition_key": "real_transition_excursion_observed_v1",
        "threshold_json": (
            "/home/pingfan/Excavator_real_stack/policy_bundles/"
            "real_transition_target_release_v2/contracts/"
            "direct_policy_output_mechanical_deadzone.json"
        ),
        "append_samples_per_episode": 1,
        "action_window_steps": 2,
        "weight": 1.0,
        "pre_positive_weight": 1.0,
        "post_negative_weight": 1.0,
        "pre_guard_weight": 1.0,
        "contrast_weight": 1.0,
        "contrast_margin_scale": 1.0,
        "guard_margin": 0.0,
        "qvel_stable_abs_max_rad_s": [0.015, 0.015, 0.02, 0.02],
    }


def test_excursion_candidates_separate_pre_apex_and_moving_return() -> None:
    actions = np.zeros((8, 4), dtype=np.float32)
    actions[0:2, 0] = 0.8
    actions[3:5, 0] = -0.8
    actions[6:8, 0] = -0.8
    qvel = np.zeros_like(actions)
    qvel[6:8, 0] = -0.2
    excursion = np.asarray([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    phase = np.asarray([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.float32)

    result = excursion_observed_candidate_indices(
        actions=actions,
        qvel=qvel,
        excursion_observed=excursion,
        return_phase=phase,
        valid_starts=np.arange(8),
        excursion_valid_mask=np.ones((8, 2), dtype=bool),
        config={
            **_config(),
            "positive_deadzone": 0.661,
            "negative_deadzone": 0.721,
            "axis_index": 0,
            "qvel_stable_abs_max_rad_s": np.asarray(
                [0.015, 0.015, 0.02, 0.02], dtype=np.float32
            ),
        },
    )

    assert result["pre_positive"].tolist() == [0]
    assert result["apex_negative"].tolist() == [3]
    assert result["moving_negative"].tolist() == [6]


def test_excursion_loss_preserves_post_action_and_guards_counterfactual() -> None:
    primary = torch.tensor(
        [[[0.8, 0, 0, 0], [0.8, 0, 0, 0]], [[-0.8, 0, 0, 0], [-0.8, 0, 0, 0]]]
    )
    counterfactual = torch.tensor(
        [[[0.0, 0, 0, 0], [0.0, 0, 0, 0]], [[0.0, 0, 0, 0], [0.0, 0, 0, 0]]]
    )

    result = excursion_observed_loss_terms(
        primary_direct=primary,
        counterfactual_direct=counterfactual,
        post_excursion_primary=torch.tensor([False, True]),
        valid=torch.tensor([True, True]),
        config=_config(),
    )

    assert torch.isclose(result["excursion_observed_loss"], torch.tensor(0.0))
    assert torch.isclose(result["excursion_pre_positive_rate"], torch.tensor(1.0))
    assert torch.isclose(result["excursion_post_negative_rate"], torch.tensor(1.0))
    assert torch.isclose(result["excursion_pre_no_shortcut_rate"], torch.tensor(1.0))
