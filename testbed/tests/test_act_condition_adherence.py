from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from testbed.data.dataset import EpisodicDataset, load_data
from testbed.policies.act.adapter import ACTAdapter
from testbed.policies.act.condition_adherence import (
    condition_adherence_loss_terms,
    resolve_condition_adherence_config,
)


def _thresholds() -> dict[str, dict[str, float]]:
    return {
        "swing": {"pos": 0.5, "neg": 0.6},
        "boom": {"pos": 0.4, "neg": 0.4},
        "stick": {"pos": 0.3, "neg": 0.3},
        "bucket": {"pos": 0.2, "neg": 0.2},
    }


def _condition_config() -> dict:
    return {
        "enabled": True,
        "scope": "train_only",
        "condition_key": "real_transition_condition_v1",
        "anchor_rule": "terminal_effective_swing_transition",
        "thresholds": _thresholds(),
        "weight": 1.0,
        "recorded_crossing_weight": 1.0,
        "counterfactual_violation_weight": 1.0,
        "contrast_weight": 1.0,
        "counterfactual_ceiling_fraction": 0.5,
        "advantage_margin_scale": 1.0,
    }


def test_condition_adherence_penalizes_old_target_action_after_goal_flip() -> None:
    config = resolve_condition_adherence_config(_condition_config())
    recorded = torch.zeros((1, 2, 4), dtype=torch.float32, requires_grad=True)
    counterfactual = torch.zeros((1, 2, 4), dtype=torch.float32, requires_grad=True)
    recorded.data[0, 0, 0] = 0.6
    counterfactual.data[0, 0, 0] = 0.4
    mask = torch.zeros((1, 2, 4, 2), dtype=torch.bool)
    mask[0, 0, 0, 0] = True

    terms = condition_adherence_loss_terms(
        recorded_direct=recorded,
        counterfactual_direct=counterfactual,
        anchor_mask=mask,
        config=config,
    )

    torch.testing.assert_close(
        terms["condition_counterfactual_violation_loss"], torch.tensor(0.15)
    )
    torch.testing.assert_close(terms["condition_contrast_loss"], torch.tensor(0.3))
    torch.testing.assert_close(terms["condition_adherence_loss"], torch.tensor(0.45))
    terms["condition_adherence_loss"].backward()
    assert recorded.grad is not None and recorded.grad[0, 0, 0] < 0
    assert counterfactual.grad is not None and counterfactual.grad[0, 0, 0] > 0


def test_condition_adherence_accepts_deadzone_scale_separation() -> None:
    config = resolve_condition_adherence_config(_condition_config())
    recorded = torch.zeros((1, 2, 4), dtype=torch.float32)
    counterfactual = torch.zeros((1, 2, 4), dtype=torch.float32)
    recorded[0, 0, 0] = 0.6
    counterfactual[0, 0, 0] = -0.1
    mask = torch.zeros((1, 2, 4, 2), dtype=torch.bool)
    mask[0, 0, 0, 0] = True

    terms = condition_adherence_loss_terms(
        recorded_direct=recorded,
        counterfactual_direct=counterfactual,
        anchor_mask=mask,
        config=config,
    )

    assert terms["condition_adherence_anchor_count"] == 1
    assert terms["condition_adherence_loss"] == 0


def test_adapter_uses_deterministic_same_observation_goal_flip() -> None:
    class _ConditionModel(torch.nn.Module):
        num_queries = 2

        def forward(self, proprio, image, env_state, actions, is_pad):
            assert self.training is False
            assert actions is None and is_pad is None
            swing = 0.5 + 0.1 * proprio[:, -2]
            prediction = torch.zeros((proprio.shape[0], 2, 4), dtype=proprio.dtype)
            prediction[:, :, 0] = swing.reshape(-1, 1)
            latent = (
                torch.zeros((proprio.shape[0], 1), dtype=proprio.dtype),
                torch.zeros((proprio.shape[0], 1), dtype=proprio.dtype),
            )
            return prediction, None, latent

    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter._model = _ConditionModel()
    adapter._condition_adherence = resolve_condition_adherence_config(
        _condition_config()
    )
    adapter.norm_stats = {
        "action_mean": np.zeros(4, dtype=np.float32),
        "action_std": np.ones(4, dtype=np.float32),
    }
    mask = torch.zeros((1, 2, 4, 2), dtype=torch.bool)
    mask[0, 0, 0, 0] = True

    terms = adapter._condition_adherence_terms(
        proprio=torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]),
        image=torch.zeros((1, 1, 3, 2, 2), dtype=torch.float32),
        counterfactual_proprio=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, -1.0, 1.0]]
        ),
        anchor_mask=mask,
        device_like=torch.zeros((1, 2, 4), dtype=torch.float32),
    )

    torch.testing.assert_close(terms["condition_adherence_loss"], torch.tensor(0.45))
    assert adapter._model.training is True


def test_condition_action_loss_pairs_same_observation_counterfactual() -> None:
    class _ActionClassModel(torch.nn.Module):
        num_queries = 2

        def forward(self, proprio, image, env_state, actions, is_pad):
            assert actions is None and is_pad is None
            side = proprio[:, -2]
            logits = torch.stack((-side, side), dim=-1)
            prediction = torch.zeros(
                (proprio.shape[0], self.num_queries, 4), dtype=proprio.dtype
            )
            latent = (
                torch.zeros((proprio.shape[0], 1), dtype=proprio.dtype),
                torch.zeros((proprio.shape[0], 1), dtype=proprio.dtype),
            )
            return prediction, None, latent, None, None, None, None, logits

    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter._model = _ActionClassModel()
    adapter._condition_action_loss = {"enabled": True, "weight": 1.0}
    proprio = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    counterfactual = torch.tensor([[0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
    result = adapter._condition_action_loss_terms(
        action=torch.zeros((1, 2, 4)),
        logits=torch.zeros((1, 2)),
        proprio=proprio,
        image=torch.zeros((1, 1, 3, 2, 2)),
        counterfactual_proprio=counterfactual,
        target=torch.tensor([1]),
        valid=torch.tensor([True]),
        device_like=torch.zeros((1, 2, 4)),
    )

    assert result["condition_action_valid_count"] == 1
    assert float(result["condition_action_class_loss"]) < 0.2
    assert result["condition_action_eval_count"] == 2
    assert result["condition_action_correct_count"] == 2


def _write_condition_episode(path: Path) -> None:
    length = 6
    action = np.zeros((length, 4), dtype=np.float32)
    action[1, 0] = 0.7
    action[3:5, 0] = -0.7
    with h5py.File(path, "w") as output:
        output.attrs["is_real"] = True
        metadata = output.create_group("metadata")
        metadata.attrs["action_prealigned"] = True
        output.create_dataset("action", data=action)
        output.create_dataset(
            "observations/qpos", data=np.zeros((length, 4), dtype=np.float32)
        )
        output.create_dataset(
            "observations/qvel", data=np.zeros((length, 4), dtype=np.float32)
        )
        output.create_dataset(
            "observations/images/video4",
            data=np.zeros((length, 2, 3, 3), dtype=np.uint8),
        )
        output.create_dataset(
            "conditions/real_transition_condition_v1",
            data=np.tile(np.asarray([1.0, 1.0], dtype=np.float32), (length, 1)),
        )
        output.create_dataset(
            "conditions/valid_mask",
            data=np.tile(np.asarray([1, 1], dtype=np.uint8), (length, 1)),
        )


def test_dataset_emits_flipped_goal_and_terminal_swing_anchor(
    tmp_path: Path, monkeypatch
) -> None:
    _write_condition_episode(tmp_path / "episode_0.hdf5")
    monkeypatch.setattr(np.random, "choice", lambda values: int(values[-1]))
    norm_stats = {
        "action_mean": np.zeros(4, dtype=np.float32),
        "action_std": np.ones(4, dtype=np.float32),
        "proprio_mean": np.zeros(6, dtype=np.float32),
        "proprio_std": np.ones(6, dtype=np.float32),
    }
    state_hold = {
        "enabled": True,
        "thresholds": _thresholds(),
        "probability": 1.0,
        "hold_horizon_steps": 2,
    }
    dataset = EpisodicDataset(
        [0],
        tmp_path,
        ["video4"],
        norm_stats,
        episode_len=6,
        low_dim_keys=["qpos", "real_transition_condition_v1"],
        action_chunk_size=2,
        state_hold_transition=state_hold,
        condition_adherence_loss=_condition_config(),
    )

    sample = dataset[0]

    torch.testing.assert_close(sample["proprio"][-2:], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(
        sample["counterfactual_proprio"][-2:], torch.tensor([-1.0, 1.0])
    )
    expected = torch.zeros((2, 4, 2), dtype=torch.bool)
    expected[0, 0, 1] = True
    assert torch.equal(sample["condition_adherence_mask"], expected)


def test_training_fails_closed_when_cycle_has_no_terminal_swing_anchor(
    tmp_path: Path,
) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    for episode_id in (0, 1):
        path = episodes / f"episode_{episode_id}.hdf5"
        _write_condition_episode(path)
        with h5py.File(path, "r+") as output:
            output["action"][:] = 0.0
    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text(
        json.dumps(
            {
                "schema": "real_transition_cycle_split_manifest_v1",
                "split_owner": "source_block_before_cycle_materialization",
                "episodes": [
                    {"episode_id": 0, "split": "train", "source_block_id": "b01"},
                    {
                        "episode_id": 1,
                        "split": "validation",
                        "source_block_id": "b02",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one terminal swing anchor"):
        load_data(
            dataset_dir=episodes,
            num_episodes=2,
            camera_names=["video4"],
            episode_len=None,
            batch_size_train=1,
            batch_size_val=1,
            num_workers=0,
            pin_memory=False,
            split_manifest_path=split_manifest,
            low_dim_keys=["qpos", "real_transition_condition_v1"],
            episode_ids=[0, 1],
            action_chunk_size=2,
            condition_adherence_loss_train=_condition_config(),
        )
