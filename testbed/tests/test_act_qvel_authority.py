from __future__ import annotations

import numpy as np
import torch

from testbed.policies.act.qvel_authority import (
    qvel_authority_candidate_starts,
    qvel_authority_loss_terms,
    stable_tool_direction_mask,
)


def _config() -> dict:
    return {
        "enabled": True,
        "weight": 1.0,
        "action_window_steps": 2,
        "stable_tool_weight": 1.0,
        "stable_guard_weight": 1.0,
        "moving_return_weight": 1.0,
        "contrast_weight": 1.0,
        "contrast_margin": 0.1,
        "stable_swing_qvel_abs_max": 0.015,
        "moving_swing_qvel_max": -0.05,
        "counterfactual_moving_swing_qvel": -0.266,
        "positive_deadzone_by_axis": np.asarray([0.6, 0.2, 0.5, 0.4]),
        "negative_deadzone_by_axis": np.asarray([0.7, 0.3, 0.5, 0.5]),
    }


def test_candidate_starts_filter_segments_by_qvel() -> None:
    qvel = np.zeros((12, 4), dtype=np.float32)
    qvel[7:, 0] = -0.2
    result = qvel_authority_candidate_starts(
        qvel=qvel,
        valid_starts=np.arange(12),
        segments={
            "tool_pre": [{"start": 0, "end": 5}],
            "swing_return": [{"start": 7, "end": 11}],
        },
        chunk_steps=2,
        config=_config(),
    )

    assert result["stable_tool"].tolist() == [0, 1, 2, 3, 4]
    assert result["moving_return"].tolist() == [7, 8, 9, 10]


def test_stable_tool_mask_excludes_swing() -> None:
    mask = stable_tool_direction_mask(
        np.asarray([-0.9, -0.4, 0.0, 0.0]), config=_config()
    )

    assert not mask[0].any()
    assert mask[1, 1]


def test_loss_accepts_stable_tool_and_moving_return_pair() -> None:
    primary = torch.zeros((2, 2, 4))
    counter = torch.zeros_like(primary)
    primary[0, :, 1] = -0.4
    counter[0, :, 0] = -0.8
    primary[1, :, 0] = -0.8
    counter[1, :, 1] = -0.4
    tool_mask = torch.zeros((2, 4, 2), dtype=torch.bool)
    tool_mask[:, 1, 1] = True

    result = qvel_authority_loss_terms(
        primary_direct=primary,
        counterfactual_direct=counter,
        moving_primary=torch.tensor([False, True]),
        stable_tool_mask=tool_mask,
        valid=torch.tensor([True, True]),
        config=_config(),
    )

    assert result["qvel_authority_pair_rate"].item() == 1.0
    assert result["qvel_authority_stable_tool_rate"].item() == 1.0
