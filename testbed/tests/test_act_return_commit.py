from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from testbed.policies.act.return_commit import (
    resolve_return_commit_loss_config,
    return_commit_candidate_indices,
    return_commit_loss_terms,
)


def _threshold_file(path: Path) -> Path:
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


def _config(path: Path) -> dict:
    return resolve_return_commit_loss_config(
        {
            "enabled": True,
            "condition_key": "real_transition_return_commit_v1",
            "threshold_json": str(_threshold_file(path)),
            "intent_threshold": 0.05,
            "contrast_margin": 0.1,
            "append_samples_per_episode": 2,
            "action_window_steps": 3,
        }
    )


def _v2_config(path: Path) -> dict:
    return resolve_return_commit_loss_config(
        {
            "enabled": True,
            "condition_key": "real_transition_return_commit_v1",
            "threshold_json": str(_threshold_file(path)),
            "intent_threshold": 0.05,
            "contrast_margin": 0.05,
            "append_samples_per_episode": 3,
            "action_window_steps": 3,
            "dig_counterfactual_mode": "no_negative_effective",
            "negative_deadzone_guard_margin": 0.0,
        }
    )


def test_return_commit_config_requires_declared_condition(tmp_path: Path) -> None:
    threshold = _threshold_file(tmp_path / "deadzone.json")

    with pytest.raises(ValueError, match="condition_key"):
        resolve_return_commit_loss_config(
            {
                "enabled": True,
                "condition_key": "wrong",
                "threshold_json": str(threshold),
            }
        )


def test_return_commit_candidates_include_dig_onset_and_effective_return(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "deadzone.json")
    actions = np.zeros((12, 4), dtype=np.float32)
    actions[1:4, 0] = 0.8
    actions[6:9, 0] = -0.1
    actions[9:12, 0] = -0.8
    state = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    valid_mask = np.ones((12, 3), dtype=np.uint8)

    result = return_commit_candidate_indices(
        actions=actions,
        return_commit=state,
        valid_starts=np.arange(12),
        return_commit_valid_mask=valid_mask,
        config=config,
    )

    np.testing.assert_array_equal(result["dig_positive"], [1])
    np.testing.assert_array_equal(result["return_onset"], [6])
    np.testing.assert_array_equal(result["return_effective"], [9])


def test_return_commit_loss_rewards_same_observation_action_flip(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "deadzone.json")
    primary = torch.tensor(
        [
            [[0.8], [0.8], [0.8]],
            [[-0.1], [-0.1], [-0.1]],
        ],
        dtype=torch.float32,
    )
    counterfactual = torch.tensor(
        [
            [[-0.1], [-0.1], [-0.1]],
            [[0.1], [0.1], [0.1]],
        ],
        dtype=torch.float32,
    )

    result = return_commit_loss_terms(
        primary_direct=primary,
        counterfactual_direct=counterfactual,
        return_primary=torch.tensor([False, True]),
        valid=torch.tensor([True, True]),
        config=config,
    )

    assert result["return_commit_loss"].item() == pytest.approx(0.0)
    assert result["return_commit_pair_hit_rate"].item() == pytest.approx(1.0)
    assert result["return_commit_dig_primary_effective_rate"].item() == pytest.approx(
        1.0
    )
    assert result["return_commit_return_primary_intent_rate"].item() == pytest.approx(
        1.0
    )


def test_return_commit_loss_penalises_ignored_condition(tmp_path: Path) -> None:
    config = _config(tmp_path / "deadzone.json")
    primary = torch.full((2, 3, 1), 0.8)
    counterfactual = torch.full((2, 3, 1), 0.8)

    result = return_commit_loss_terms(
        primary_direct=primary,
        counterfactual_direct=counterfactual,
        return_primary=torch.tensor([False, True]),
        valid=torch.tensor([True, True]),
        config=config,
    )

    assert result["return_commit_loss"].item() > 0.0
    assert result["return_commit_pair_hit_rate"].item() == pytest.approx(0.0)


def test_return_commit_v2_allows_hold_and_preserves_effective_return(
    tmp_path: Path,
) -> None:
    config = _v2_config(tmp_path / "deadzone.json")
    primary = torch.tensor(
        [
            [[0.8], [0.8], [0.8]],
            [[-0.1], [-0.1], [-0.1]],
            [[-0.8], [-0.8], [-0.8]],
        ],
        dtype=torch.float32,
    )
    counterfactual = torch.tensor(
        [
            [[-0.1], [-0.1], [-0.1]],
            [[0.0], [0.0], [0.0]],
            [[0.8], [0.8], [0.8]],
        ],
        dtype=torch.float32,
    )

    result = return_commit_loss_terms(
        primary_direct=primary,
        counterfactual_direct=counterfactual,
        return_primary=torch.tensor([False, True, True]),
        return_effective_primary=torch.tensor([False, False, True]),
        valid=torch.tensor([True, True, True]),
        config=config,
    )

    assert result["return_commit_loss"].item() == pytest.approx(0.0)
    assert result["return_commit_pair_hit_rate"].item() == pytest.approx(1.0)
    assert result["return_commit_return_primary_effective_rate"].item() == (
        pytest.approx(1.0)
    )
    assert result[
        "return_commit_DIG_counterfactual_no_shortcut_rate"
    ].item() == pytest.approx(1.0)


def test_return_commit_v2_penalises_counterfactual_shortcut(
    tmp_path: Path,
) -> None:
    config = _v2_config(tmp_path / "deadzone.json")
    primary = torch.full((1, 3, 1), -0.1)
    counterfactual = torch.full((1, 3, 1), -0.8)

    result = return_commit_loss_terms(
        primary_direct=primary,
        counterfactual_direct=counterfactual,
        return_primary=torch.tensor([True]),
        return_effective_primary=torch.tensor([False]),
        valid=torch.tensor([True]),
        config=config,
    )

    assert result["return_commit_dig_counterfactual_loss"].item() > 0.0
    assert result[
        "return_commit_DIG_counterfactual_no_shortcut_rate"
    ].item() == pytest.approx(0.0)


def test_return_commit_can_sample_non_swing_dig_transition(
    tmp_path: Path,
) -> None:
    config = _v2_config(tmp_path / "deadzone.json")
    config["dig_candidate_mode"] = "non_swing_transition"
    config["dig_candidate_key"] = "dig_non_swing_transition"
    actions = np.zeros((9, 4), dtype=np.float32)
    actions[1:4, 1] = -0.4
    actions[6:9, 0] = -0.8
    state = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.float32)

    result = return_commit_candidate_indices(
        actions=actions,
        return_commit=state,
        valid_starts=np.arange(9),
        return_commit_valid_mask=np.ones((9, 3), dtype=bool),
        config=config,
    )

    np.testing.assert_array_equal(result["dig_positive"], [])
    np.testing.assert_array_equal(result["dig_non_swing_transition"], [1])
    np.testing.assert_array_equal(result["return_effective"], [6])
