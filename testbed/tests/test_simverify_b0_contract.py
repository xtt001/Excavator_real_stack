from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import yaml

from testbed.data.dataset import load_data
from testbed.runtime.run_metadata import _find_git_root

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
        (CONFIG_ROOT / "simverify_b0_unconditioned_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    split = yaml.safe_load(
        (CONFIG_ROOT / "simverify_b0_split_v1.yaml").read_text(
            encoding="utf-8"
        )
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
