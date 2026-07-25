from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from testbed.data.dataset import EpisodicDataset, load_data
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
    condition_values: np.ndarray | None = None,
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
        condition_group = handle.create_group("conditions")
        condition_group.create_dataset(
            "valid_mask",
            data=np.asarray(valid_mask, dtype=np.uint8),
        )
        if condition_values is None:
            condition_array = np.asarray(
                [
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 1],
                ],
                dtype=np.float32,
            )
        else:
            condition_array = np.asarray(
                condition_values,
                dtype=np.float32,
            )
        condition_group.create_dataset(
            "cycle_condition_v1",
            data=condition_array,
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
        def __init__(self) -> None:
            self.conditions: list[np.ndarray] = []

        def reset(self) -> None:
            pass

        def predict(self, observation: dict[str, object]) -> np.ndarray:
            if "cycle_condition_v1" in observation:
                self.conditions.append(
                    np.asarray(
                        observation["cycle_condition_v1"],
                        dtype=np.float32,
                    )
                )
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
    policy = _Policy()
    target_condition = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    with h5py.File(path, "r") as episode:
        arrays = replay_cycle_arrays(
            policy=policy,
            episode=episode,
            annotation=annotation,
            camera_names=["video4"],
            condition_override=target_condition,
            pass_condition_to_policy=True,
        )

    assert arrays["raw_policy_chunk_normalized"].shape == (3, 2, 4)
    assert arrays["raw_policy_chunk_direct"].shape == (3, 2, 4)
    assert arrays["temporal_aggregation_action"].shape == (3, 4)
    assert not np.shares_memory(
        arrays["temporal_aggregation_action"],
        arrays["future_runtime_safe_action"],
    )
    np.testing.assert_array_equal(arrays["condition_cycle_id"], [4, 4, 4])
    np.testing.assert_array_equal(
        arrays["condition"],
        np.repeat(
            np.asarray(target_condition, dtype=np.float32).reshape(1, 6),
            3,
            axis=0,
        ),
    )
    assert len(policy.conditions) == 3


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


def test_b1_condition_is_appended_to_train_only_normalized_proprio(
    tmp_path: Path,
) -> None:
    _write_episode(
        tmp_path / "episode_0.hdf5",
        value=1.0,
        valid_mask=[1, 1, 1, 1],
    )
    _write_episode(
        tmp_path / "episode_1.hdf5",
        value=2.0,
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

    train, _val, stats, _is_real, split = load_data(
        dataset_dir=tmp_path,
        num_episodes=2,
        camera_names=["video4"],
        episode_len=4,
        batch_size_train=1,
        batch_size_val=1,
        num_workers=0,
        split_path=split_path,
        reuse_split=True,
        low_dim_keys=["qpos", "qvel", "cycle_condition_v1"],
        episode_ids=[0, 1],
        action_chunk_size=2,
        sample_valid_mask_path="conditions/valid_mask",
        norm_stats_train_only=True,
    )

    sample = next(iter(train))
    assert sample[1].shape == (1, 14)
    assert stats["proprio_mean"].shape == (14,)
    assert stats["proprio_keys"].tolist() == [
        "qpos",
        "qvel",
        "cycle_condition_v1",
    ]
    assert split["condition_shuffle_train"]["enabled"] is False


def test_b2_train_condition_shuffle_is_reproducible_and_preserves_marginals(
    tmp_path: Path,
) -> None:
    _write_episode(
        tmp_path / "episode_0.hdf5",
        value=1.0,
        valid_mask=[1, 1, 1, 1],
    )
    _write_episode(
        tmp_path / "episode_1.hdf5",
        value=2.0,
        valid_mask=[1, 1, 1, 1],
    )
    norm_stats = {
        "action_mean": np.zeros(4, dtype=np.float32),
        "action_std": np.ones(4, dtype=np.float32),
        "proprio_mean": np.zeros(14, dtype=np.float32),
        "proprio_std": np.ones(14, dtype=np.float32),
        "proprio_dim": 14,
        "qpos_only_dim": 4,
    }
    kwargs = {
        "episode_ids": [0, 1],
        "dataset_dir": tmp_path,
        "camera_names": ["video4"],
        "norm_stats": norm_stats,
        "low_dim_keys": ["qpos", "qvel", "cycle_condition_v1"],
        "action_chunk_size": 2,
        "sample_valid_mask_path": "conditions/valid_mask",
        "condition_shuffle_seed": 20260725,
    }
    first = EpisodicDataset(**kwargs)
    second = EpisodicDataset(**kwargs)
    validation = EpisodicDataset(
        **{**kwargs, "condition_shuffle_seed": None}
    )

    manifest = first.condition_shuffle_manifest
    assert manifest["row_count"] == 8
    assert manifest["source_token_counts"] == manifest[
        "shuffled_token_counts"
    ]
    assert manifest["mapping_sha256"] == second.condition_shuffle_manifest[
        "mapping_sha256"
    ]
    assert manifest["changed_row_count"] > 0
    assert validation.condition_shuffle_manifest == {
        "enabled": False,
        "scope": "none",
    }


def test_b1_b2_configs_are_matched_except_condition_association() -> None:
    b1 = yaml.safe_load(
        (CONFIG_ROOT / "simverify_b1_conditioned_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    b2 = yaml.safe_load(
        (CONFIG_ROOT / "simverify_b2_shuffled_condition_v1.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert b1["policy"] == b2["policy"]
    assert b1["policy"]["low_dim_keys"] == [
        "qpos",
        "qvel",
        "cycle_condition_v1",
    ]
    assert b1["policy"]["act_params"]["state_dim"] == 14
    assert b1["task"] | {"task_name": "matched"} == (
        b2["task"] | {"task_name": "matched"}
    )
    b1_train = {
        **b1["train"],
        "ckpt_dir": "matched",
        "condition_shuffle": "declared_factor",
    }
    b2_train = {
        **b2["train"],
        "ckpt_dir": "matched",
        "condition_shuffle": "declared_factor",
    }
    assert b1_train == b2_train
    assert b1["train"]["condition_shuffle"]["enabled"] is False
    assert b2["train"]["condition_shuffle"]["enabled"] is True
    assert b1["checkpoint_semantics"] == b2["checkpoint_semantics"]
