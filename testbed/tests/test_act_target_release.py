from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from testbed.data.dataset import EpisodicDataset
from testbed.policies.act.adapter import ACTAdapter
from testbed.policies.act.target_release import (
    resolve_target_release_config,
    target_release_candidate_indices,
    target_release_loss_terms,
)


def _write_contract(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "real_transition_target_release_contract_v1",
                "condition_schema": "real_transition_condition_v1",
                "decision_region": {
                    "swing_axis_index": 0,
                    "continue_target_side": "A",
                    "stop_target_side": "B",
                    "swing_qpos_range_rad": [0.1, 0.4],
                    "continue_action_target_abs": 0.65,
                },
                "candidate_rule": {
                    "after_swing_apex": True,
                    "action_window_steps": 2,
                    "stable_window_steps": 3,
                    "stable_qvel_abs_max_rad_s": 0.02,
                },
                "mechanical_deadzone": {
                    "swing": {"pos": 0.5, "neg": 0.6}
                },
            }
        ),
        encoding="utf-8",
    )


def _config(path: Path, *, weight: float = 1.0) -> dict:
    return resolve_target_release_config(
        {
            "enabled": True,
            "scope": "train_and_validation",
            "condition_key": "real_transition_condition_v1",
            "contract_json": str(path),
            "append_samples_per_episode": 1,
            "weight": weight,
            "continue_weight": 1.0,
            "stop_weight": 1.0,
            "contrast_weight": 1.0,
            "contrast_margin_scale": 1.0,
        }
    )


def test_target_release_candidates_are_supported_by_recorded_action(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "contract.json"
    _write_contract(contract)
    config = _config(contract)
    qpos = np.zeros((9, 4), dtype=np.float32)
    qpos[:, 0] = [0.0, 1.0, 0.7, 0.35, 0.3, 0.2, 0.05, 0.0, 0.0]
    qvel = np.zeros_like(qpos)
    actions = np.zeros_like(qpos)
    actions[3:6, 0] = -0.7
    valid = np.arange(9, dtype=np.int64)
    condition_valid = np.ones((9, 2), dtype=bool)

    continue_indices = target_release_candidate_indices(
        qpos=qpos,
        qvel=qvel,
        actions=actions,
        condition=np.tile([-1.0, 1.0], (9, 1)),
        valid_starts=valid,
        condition_valid_mask=condition_valid,
        config=config,
    )
    stop_actions = actions.copy()
    stop_actions[:, 0] = 0.0
    stop_indices = target_release_candidate_indices(
        qpos=qpos,
        qvel=qvel,
        actions=stop_actions,
        condition=np.tile([1.0, 1.0], (9, 1)),
        valid_starts=valid,
        condition_valid_mask=condition_valid,
        config=config,
    )

    np.testing.assert_array_equal(continue_indices, [3, 4])
    np.testing.assert_array_equal(stop_indices, [3, 4, 5])


def test_target_release_loss_accepts_negative_A_and_zero_B(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    _write_contract(contract)
    config = _config(contract, weight=2.0)
    primary = torch.zeros((1, 2, 4), dtype=torch.float32)
    counterfactual = torch.zeros_like(primary)
    primary[:, :, 0] = -0.65

    terms = target_release_loss_terms(
        primary_direct=primary,
        counterfactual_direct=counterfactual,
        continue_primary=torch.tensor([True]),
        valid=torch.tensor([True]),
        config=config,
    )

    assert terms["target_release_loss"] == 0
    assert terms["target_release_pair_hit_rate"] == 1
    assert terms["target_release_valid_count"] == 1


def test_target_release_loss_pushes_pair_apart(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    _write_contract(contract)
    config = _config(contract)
    primary = torch.zeros((1, 2, 4), dtype=torch.float32, requires_grad=True)
    counterfactual = torch.zeros_like(primary, requires_grad=True)
    primary.data[:, :, 0] = -0.2
    counterfactual.data[:, :, 0] = -0.3

    terms = target_release_loss_terms(
        primary_direct=primary,
        counterfactual_direct=counterfactual,
        continue_primary=torch.tensor([True]),
        valid=torch.tensor([True]),
        config=config,
    )
    terms["target_release_loss"].backward()

    assert float(terms["target_release_loss"].detach()) > 0.0
    # Gradient descent subtracts these gradients: the continue branch moves
    # more negative and the stop branch moves back towards zero.
    assert primary.grad is not None and torch.all(primary.grad[:, :, 0] > 0)
    assert counterfactual.grad is not None and torch.all(
        counterfactual.grad[:, :, 0] < 0
    )


def test_adapter_supervises_deterministic_action_not_side_logits(
    tmp_path: Path,
) -> None:
    class _ReleaseModel(torch.nn.Module):
        num_queries = 2

        def forward(self, proprio, image, env_state, actions, is_pad):
            assert self.training is False
            side = proprio[:, -2]
            prediction = torch.zeros(
                (proprio.shape[0], self.num_queries, 4), dtype=proprio.dtype
            )
            prediction[:, :, 0] = torch.where(
                side < 0.0, torch.full_like(side, -0.65), torch.zeros_like(side)
            ).reshape(-1, 1)
            latent = (
                torch.zeros((proprio.shape[0], 1), dtype=proprio.dtype),
                torch.zeros((proprio.shape[0], 1), dtype=proprio.dtype),
            )
            return prediction, None, latent

    contract = tmp_path / "contract.json"
    _write_contract(contract)
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter._model = _ReleaseModel()
    adapter._target_release_loss = _config(contract)
    adapter.norm_stats = {
        "action_mean": np.zeros(4, dtype=np.float32),
        "action_std": np.ones(4, dtype=np.float32),
    }
    proprio = torch.tensor([[0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
    counterfactual = proprio.clone()
    counterfactual[:, -2] = 1.0

    terms = adapter._target_release_loss_terms(
        proprio=proprio,
        image=torch.zeros((1, 1, 3, 2, 2)),
        counterfactual_proprio=counterfactual,
        continue_primary=torch.tensor([True]),
        valid=torch.tensor([True]),
        device_like=torch.zeros((1, 2, 4)),
    )

    assert terms["target_release_loss"] == 0
    assert terms["target_release_pair_hit_rate"] == 1
    assert adapter._model.training is True


def _write_episode(path: Path) -> None:
    length = 8
    qpos = np.zeros((length, 4), dtype=np.float32)
    qpos[:, 0] = [0.0, 1.0, 0.7, 0.35, 0.3, 0.2, 0.0, 0.0]
    action = np.zeros_like(qpos)
    action[3:6, 0] = -0.7
    with h5py.File(path, "w") as output:
        output.attrs["is_real"] = True
        metadata = output.create_group("metadata")
        metadata.attrs["action_prealigned"] = True
        output.create_dataset("action", data=action)
        output.create_dataset("observations/qpos", data=qpos)
        output.create_dataset("observations/qvel", data=np.zeros_like(qpos))
        output.create_dataset(
            "observations/images/video4",
            data=np.zeros((length, 2, 3, 3), dtype=np.uint8),
        )
        output.create_dataset(
            "conditions/real_transition_condition_v1",
            data=np.tile(np.asarray([-1.0, 1.0], dtype=np.float32), (length, 1)),
        )
        output.create_dataset(
            "conditions/valid_mask", data=np.ones((length, 2), dtype=np.uint8)
        )


def test_dataset_appends_release_sample_without_replacing_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    contract = tmp_path / "contract.json"
    _write_contract(contract)
    _write_episode(tmp_path / "episode_0.hdf5")
    monkeypatch.setattr(np.random, "choice", lambda values: int(values[0]))
    dataset = EpisodicDataset(
        [0],
        tmp_path,
        ["video4"],
        {
            "action_mean": np.zeros(4, dtype=np.float32),
            "action_std": np.ones(4, dtype=np.float32),
            "proprio_mean": np.zeros(6, dtype=np.float32),
            "proprio_std": np.ones(6, dtype=np.float32),
        },
        episode_len=8,
        low_dim_keys=["qpos", "real_transition_condition_v1"],
        action_chunk_size=2,
        target_release_loss={
            "enabled": True,
            "scope": "train_and_validation",
            "condition_key": "real_transition_condition_v1",
            "contract_json": str(contract),
            "append_samples_per_episode": 1,
        },
    )

    assert len(dataset) == 2
    baseline = dataset[0]
    release = dataset[1]
    assert not bool(baseline["target_release_valid"])
    assert bool(release["target_release_valid"])
    assert bool(release["target_release_continue_primary"])
    torch.testing.assert_close(
        release["counterfactual_proprio"][-2:], torch.tensor([1.0, 1.0])
    )
