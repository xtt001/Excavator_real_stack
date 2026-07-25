from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from testbed.data.dataset import load_data
from testbed.policies.act.trainer import ACTTrainer
from testbed.runtime.run_metadata import _find_git_root
from testbed.simverify.m3_gate import bootstrap_episode_mean
from testbed.simverify.m3_replay import (
    _validate_b0_checkpoint_contract,
    replay_cycle_arrays,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "testbed/testbed/configs"


def _write_episode(
    path: Path,
    *,
    value: float,
    valid_mask: list[int],
) -> None:
    image = np.zeros((4, 8, 12, 3), dtype=np.uint8)
    values = np.full((4, 4), value, dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.attrs["is_real"] = False
        observations = handle.create_group("observations")
        observations.create_dataset("qpos", data=values)
        observations.create_dataset("qvel", data=values)
        images = observations.create_group("images")
        images.create_dataset("video4", data=image)
        handle.create_dataset("action", data=values)
        conditions = handle.create_group("conditions")
        conditions.create_dataset(
            "valid_mask",
            data=np.asarray(valid_mask, dtype=np.uint8),
        )


def test_simverify_mask_and_train_only_stats_exclude_invalid_and_validation(
    tmp_path: Path,
) -> None:
    _write_episode(
        tmp_path / "episode_0.hdf5",
        value=1.0,
        valid_mask=[1, 1, 0, 0],
    )
    _write_episode(
        tmp_path / "episode_1.hdf5",
        value=100.0,
        valid_mask=[1, 1, 1, 1],
    )
    split_path = tmp_path / "split.yaml"
    split_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset_dir": str(tmp_path.resolve()),
                "available_episode_ids": [0, 1],
                "train_ids": [0],
                "val_ids": [1],
            }
        ),
        encoding="utf-8",
    )

    _train, _val, stats, _is_real, split = load_data(
        dataset_dir=tmp_path,
        num_episodes=2,
        camera_names=["video4"],
        episode_len=4,
        batch_size_train=1,
        batch_size_val=1,
        num_workers=0,
        pin_memory=False,
        split_path=split_path,
        reuse_split=True,
        low_dim_keys=["qpos", "qvel"],
        episode_ids=[0, 1],
        action_chunk_size=2,
        sample_valid_mask_path="conditions/valid_mask",
        norm_stats_train_only=True,
    )

    np.testing.assert_allclose(stats["action_mean"], np.ones(4))
    np.testing.assert_allclose(stats["proprio_mean"], np.ones(8))
    assert split["gap_mask_valid_start_count"][0] == 1
    assert split["norm_stats_episode_ids"] == [0]
    assert split["sample_valid_mask_path"] == "conditions/valid_mask"


def test_b0_config_is_unconditioned_sim_domain_and_holds_out_test() -> None:
    config = yaml.safe_load(
        (CONFIG_ROOT / "simverify_b0_unconditioned_v1.yaml").read_text(encoding="utf-8")
    )
    split = yaml.safe_load(
        (CONFIG_ROOT / "simverify_b0_split_v1.yaml").read_text(encoding="utf-8")
    )

    assert config["experiment_contract"]["baseline_id"] == "B0"
    assert config["experiment_contract"]["condition_input"] == "absent"
    assert config["experiment_contract"]["held_out_test"] == "locked_unread"
    assert config["policy"]["low_dim_keys"] == ["qpos", "qvel"]
    assert "cycle_condition_v1" not in config["policy"]["low_dim_keys"]
    assert config["train"]["sample_valid_mask_path"] == "conditions/valid_mask"
    assert config["train"]["norm_stats_train_only"] is True
    assert config["checkpoint_semantics"]["domain"] == "sim"
    assert config["checkpoint_semantics"]["real_control_allowed"] is False
    assert set(config["task"]["episode_ids"]) == set(
        split["train_ids"] + split["val_ids"]
    )
    assert split["zero_valid_start_excluded_ids"] == [19, 23]
    assert {1, 13, 25, 33}.isdisjoint(config["task"]["episode_ids"])


def test_run_metadata_discovers_worktree_git_root() -> None:
    assert _find_git_root(Path(__file__).resolve()) == REPO_ROOT


def test_checkpoint_embeds_sim_domain_prohibition(tmp_path: Path) -> None:
    class _Adapter:
        @staticmethod
        def state_dict() -> dict[str, torch.Tensor]:
            return {"weight": torch.ones(1)}

    class _Optimizer:
        @staticmethod
        def state_dict() -> dict[str, object]:
            return {}

    path = tmp_path / "policy.ckpt"
    semantics = {
        "domain": "sim",
        "real_control_allowed": False,
        "jetson_allowed": False,
    }
    ACTTrainer._save_ckpt(
        path,
        _Adapter(),
        _Optimizer(),
        0,
        1.0,
        {
            "task_name": "simverify_b0",
            "seed": 0,
            "checkpoint_semantics": semantics,
            "experiment_contract": {"baseline_id": "B0"},
        },
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["config"]["checkpoint_semantics"] == semantics
    assert checkpoint["config"]["experiment_contract"]["baseline_id"] == "B0"


def test_b0_replay_rejects_checkpoint_without_embedded_prohibition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "policy.ckpt"
    torch.save(
        {
            "config": {
                "checkpoint_semantics": {
                    "domain": "sim",
                    "real_control_allowed": True,
                    "jetson_allowed": False,
                },
                "experiment_contract": {
                    "baseline_id": "B0",
                    "condition_input": "absent",
                },
            }
        },
        path,
    )

    with pytest.raises(
        ValueError,
        match="embedded sim-domain prohibition",
    ):
        _validate_b0_checkpoint_contract(path)


def test_b0_cycle_replay_materializes_independent_action_stages(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episode_0.hdf5"
    image = np.zeros((3, 8, 12, 3), dtype=np.uint8)
    with h5py.File(path, "w") as handle:
        observations = handle.create_group("observations")
        observations.create_dataset(
            "qpos",
            data=np.zeros((3, 4), dtype=np.float32),
        )
        observations.create_dataset(
            "qvel",
            data=np.zeros((3, 4), dtype=np.float32),
        )
        images = observations.create_group("images")
        images.create_dataset("video4", data=image)
        handle.create_dataset(
            "action",
            data=np.zeros((3, 4), dtype=np.float32),
        )
        diagnostics = handle.create_group("diagnostics")
        diagnostics.create_dataset(
            "source_observation_index",
            data=np.arange(3, dtype=np.int64),
        )
        diagnostics.create_dataset(
            "target_tick",
            data=np.arange(3, dtype=np.int64),
        )

    class _Policy:
        def reset(self) -> None:
            pass

        def predict(self, _observation: dict[str, object]) -> np.ndarray:
            return np.asarray([0.1, 0.0, 0.0, 0.0], dtype=np.float32)

        def last_raw_action_chunk(self) -> np.ndarray:
            return np.zeros((2, 4), dtype=np.float32)

        def last_raw_action_chunk_direct(self) -> np.ndarray:
            return np.ones((2, 4), dtype=np.float32)

    annotation = {
        "cycle_id": 4,
        "target_steps_20hz": [0, 2],
        "policy_condition": {
            "vector": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
    }
    with h5py.File(path, "r") as episode:
        arrays = replay_cycle_arrays(
            policy=_Policy(),
            episode=episode,
            annotation=annotation,
            camera_names=["video4"],
        )

    assert arrays["raw_policy_chunk_normalized"].shape == (3, 2, 4)
    assert arrays["raw_policy_chunk_direct"].shape == (3, 2, 4)
    assert arrays["temporal_aggregation_action"].shape == (3, 4)
    assert not np.shares_memory(
        arrays["temporal_aggregation_action"],
        arrays["future_runtime_safe_action"],
    )
    np.testing.assert_array_equal(arrays["condition_cycle_id"], [4, 4, 4])


def test_g3_bootstrap_resamples_source_episode_means_deterministically() -> None:
    result = bootstrap_episode_mean(
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        repetitions=10_000,
        seed=7,
    )

    assert result["observed_mean"] == pytest.approx(1.0 / 3.0)
    assert result["p02_5"] == 0.0
    assert result == bootstrap_episode_mean(
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        repetitions=10_000,
        seed=7,
    )
