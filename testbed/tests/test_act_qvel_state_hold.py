from __future__ import annotations

import json
from pathlib import Path

import torch

from testbed.policies.act.qvel_state_hold import (
    qvel_zero_state_hold_loss_terms,
    resolve_qvel_zero_state_hold_config,
)


def _config(tmp_path: Path) -> dict:
    path = tmp_path / "deadzone.json"
    path.write_text(
        json.dumps(
            {
                "deadzone_action": {
                    "swing": {"pos": 0.6, "neg": 0.7},
                    "boom": {"pos": 0.2, "neg": 0.3},
                    "stick": {"pos": 0.4, "neg": 0.4},
                    "bucket": {"pos": 0.3, "neg": 0.5},
                }
            }
        ),
        encoding="utf-8",
    )
    return resolve_qvel_zero_state_hold_config(
        {
            "enabled": True,
            "threshold_json": str(path),
            "weight": 0.5,
            "window_steps": 3,
        }
    )


def test_qvel_zero_state_hold_direct_direction_loss(tmp_path: Path) -> None:
    config = _config(tmp_path)
    actions = torch.zeros((2, 3, 4))
    actions[0, 1, 0] = 0.7
    actions[1, 2, 1] = -0.4
    mask = torch.zeros((2, 4, 2), dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True

    passed = qvel_zero_state_hold_loss_terms(
        policy_direct=actions,
        transition_mask=mask,
        config=config,
    )
    assert passed["qvel_zero_state_hold_loss"].item() == 0.0
    assert passed["qvel_zero_state_hold_direction_hit_rate"].item() == 1.0

    failed = qvel_zero_state_hold_loss_terms(
        policy_direct=torch.zeros_like(actions),
        transition_mask=mask,
        config=config,
    )
    assert failed["qvel_zero_state_hold_loss"].item() > 0.0
    assert failed["qvel_zero_state_hold_direction_hit_rate"].item() == 0.0
