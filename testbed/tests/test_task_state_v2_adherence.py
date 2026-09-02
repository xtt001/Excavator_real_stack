from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from testbed.policies.act.task_state_v2_adherence import (
    resolve_task_state_v2_adherence_config,
    task_state_v2_adherence_loss_terms,
)


def _threshold(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "deadzone_action": {
                    "swing": {"pos": 0.661, "neg": 0.721},
                    "boom": {"pos": 0.259, "neg": 0.357},
                    "stick": {"pos": 0.5, "neg": 0.5},
                    "bucket": {"pos": 0.408, "neg": 0.508},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_uncommitted_guard_penalises_only_mechanically_negative_rows(
    tmp_path: Path,
) -> None:
    config = resolve_task_state_v2_adherence_config(
        {
            "enabled": True,
            "threshold_json": str(_threshold(tmp_path / "deadzone.json")),
            "weight": 2.0,
            "action_window_steps": 3,
            "guard_margin": 0.0,
        }
    )
    policy = torch.zeros(2, 3, 4)
    policy[0, :, 0] = torch.tensor([-0.8, -0.7, -0.9])
    policy[1, :, 0] = -1.0

    result = task_state_v2_adherence_loss_terms(
        policy_direct=policy,
        valid_mask=torch.ones(2, 3, 1, dtype=torch.bool),
        uncommitted=torch.tensor([True, False]),
        config=config,
    )

    expected = ((0.8 - 0.721) + 0.0 + (0.9 - 0.721)) / 3.0
    assert float(result["task_state_v2_uncommitted_guard_loss"]) == pytest.approx(
        expected
    )
    assert float(result["task_state_v2_adherence_loss"]) == pytest.approx(
        2.0 * expected
    )
    assert float(result["task_state_v2_uncommitted_valid_count"]) == 3.0
    assert float(result["task_state_v2_uncommitted_no_negative_rate"]) == pytest.approx(
        1.0 / 3.0
    )


def test_disabled_guard_needs_no_labels() -> None:
    result = task_state_v2_adherence_loss_terms(
        policy_direct=torch.zeros(1, 2, 4),
        valid_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        uncommitted=None,
        config={"enabled": False},
    )

    assert float(result["task_state_v2_adherence_loss"]) == 0.0


def test_worst_query_reduction_matches_existential_window_gate(
    tmp_path: Path,
) -> None:
    config = resolve_task_state_v2_adherence_config(
        {
            "enabled": True,
            "threshold_json": str(_threshold(tmp_path / "deadzone.json")),
            "weight": 1.0,
            "action_window_steps": 3,
            "reduction": "worst_query",
        }
    )
    policy = torch.zeros(2, 3, 4)
    policy[0, :, 0] = torch.tensor([-0.8, -0.7, -0.9])
    policy[1, :, 0] = torch.tensor([-0.75, -0.72, -0.70])

    result = task_state_v2_adherence_loss_terms(
        policy_direct=policy,
        valid_mask=torch.ones(2, 3, 1, dtype=torch.bool),
        uncommitted=torch.tensor([True, True]),
        config=config,
    )

    expected = ((0.9 - 0.721) + (0.75 - 0.721)) / 2.0
    assert float(result["task_state_v2_uncommitted_guard_loss"]) == pytest.approx(
        expected
    )
