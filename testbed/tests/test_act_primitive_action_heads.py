from __future__ import annotations

import pytest
import torch
from torch import nn

from testbed.data.action_primitive_islands import (
    ACTION_PRIMITIVE_KEY,
    PRIMITIVE_NAMES,
)
from testbed.policies.act.primitive_action_heads import (
    mask_hard_routed_proprio,
    resolve_primitive_action_heads_config,
    select_hard_routed_action,
)


def _config() -> dict:
    return resolve_primitive_action_heads_config(
        {
            "enabled": True,
            "condition_key": ACTION_PRIMITIVE_KEY,
            "primitive_names": list(PRIMITIVE_NAMES),
            "one_hot_start_index": 4,
        },
        robot_state_dim=8,
    )


def test_hard_routing_selects_exactly_one_action_head() -> None:
    shared = nn.Linear(2, 1)
    additional = nn.ModuleList([nn.Linear(2, 1) for _ in range(3)])
    for index, head in enumerate([shared, *additional], start=1):
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, float(index))
    decoder = torch.zeros((4, 3, 2))
    proprio = torch.zeros((4, 8))
    proprio[:, 4:] = torch.eye(4)

    output = select_hard_routed_action(
        shared_head=shared,
        additional_heads=additional,
        decoder_state=decoder,
        proprio=proprio,
        config=_config(),
    )

    assert output.shape == (4, 3, 1)
    torch.testing.assert_close(
        output[:, 0, 0], torch.asarray([1.0, 2.0, 3.0, 4.0])
    )


def test_hard_routing_rejects_soft_or_missing_command() -> None:
    shared = nn.Linear(2, 1)
    additional = nn.ModuleList([nn.Linear(2, 1) for _ in range(3)])
    decoder = torch.zeros((1, 3, 2))
    proprio = torch.zeros((1, 8))
    proprio[0, 4:] = torch.asarray([0.0, 0.5, 0.5, 0.0])

    with pytest.raises(ValueError, match="one finite one-hot"):
        select_hard_routed_action(
            shared_head=shared,
            additional_heads=additional,
            decoder_state=decoder,
            proprio=proprio,
            config=_config(),
        )


def test_config_requires_one_hot_at_end_of_proprio() -> None:
    with pytest.raises(ValueError, match="final proprio columns"):
        resolve_primitive_action_heads_config(
            {
                "enabled": True,
                "condition_key": ACTION_PRIMITIVE_KEY,
                "primitive_names": list(PRIMITIVE_NAMES),
                "one_hot_start_index": 3,
            },
            robot_state_dim=8,
        )


def test_work_return_route_masks_branch_irrelevant_task_fields() -> None:
    config = resolve_primitive_action_heads_config(
        {
            "enabled": True,
            "condition_key": "real_transition_work_context_v1",
            "primitive_names": ["work_A", "work_B", "return_A", "return_B"],
            "one_hot_start_index": 8,
            "branch_zero_indices": {
                "work_A": [4],
                "work_B": [4],
                "return_A": [6, 7],
                "return_B": [6, 7],
            },
        },
        robot_state_dim=12,
    )
    proprio = torch.zeros((2, 12))
    proprio[:, 4] = torch.asarray([1.0, -1.0])
    proprio[:, 6] = 1.0
    proprio[:, 7] = 1.0
    proprio[0, 8:] = torch.asarray([0.0, 1.0, 0.0, 0.0])
    proprio[1, 8:] = torch.asarray([0.0, 0.0, 1.0, 0.0])

    masked = mask_hard_routed_proprio(proprio, config=config)

    assert masked[0, 4] == 0.0
    assert masked[0, 6] == 1.0
    assert masked[0, 7] == 1.0
    assert masked[1, 4] == -1.0
    assert masked[1, 6] == 0.0
    assert masked[1, 7] == 0.0
    torch.testing.assert_close(masked[:, 8:], proprio[:, 8:])
