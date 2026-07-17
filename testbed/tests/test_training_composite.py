from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from testbed.data.training_composite import (
    build_training_composite,
    validate_training_composite,
)
from testbed.policies.act.training_preflight import preflight_act_training_config


def _episode(path: Path) -> None:
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("action", data=np.zeros((3, 4), dtype=np.float32))
        obs = h5_file.create_group("observations")
        obs.create_dataset("qpos", data=np.zeros((3, 4), dtype=np.float32))
        obs.create_dataset("qvel", data=np.zeros((3, 4), dtype=np.float32))
        images = obs.create_group("images")
        for camera in ("video4", "video5", "video6", "video7"):
            images.create_dataset(
                camera, data=np.zeros((3, 2, 2, 3), dtype=np.uint8)
            )


def _write(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _view(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    _episode(source / "episode_1.hdf5")
    _episode(source / "episode_2.hdf5")
    split = tmp_path / "source_split.yaml"
    _write(split, {"dataset_dir": str(source), "train_ids": [1], "val_ids": [2]})
    output = tmp_path / "view"
    spec = tmp_path / "spec.yaml"
    _write(
        spec,
        {
            "schema_version": 1,
            "output_dir": str(output),
            "forbidden_source_episode_ids": [105, 106, 107, 108, 109],
            "sources": [
                {
                    "name": "new",
                    "dataset_dir": str(source),
                    "split_path": str(split),
                    "include_roles": ["train", "val"],
                    "composite_start": 10000,
                }
            ],
        },
    )
    build_training_composite(spec)
    return output


def test_build_training_composite_is_symlink_only_and_exact_split(tmp_path: Path) -> None:
    view = _view(tmp_path)
    report = validate_training_composite(view, verify_hashes=True)
    assert report["episode_count"] == 2
    assert report["train_count"] == 1
    assert report["val_count"] == 1
    assert (view / "episode_10000.hdf5").is_symlink()
    assert (view / "episode_10001.hdf5").is_symlink()
    manifest = json.loads((view / "train_ready_manifest.json").read_text())
    assert manifest["test_ids"] == []


def test_build_training_composite_rejects_forbidden_source_episode(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _episode(source / "episode_105.hdf5")
    _episode(source / "episode_2.hdf5")
    split = tmp_path / "split.yaml"
    _write(split, {"dataset_dir": str(source), "train_ids": [105], "val_ids": [2]})
    spec = tmp_path / "spec.yaml"
    _write(
        spec,
        {
            "schema_version": 1,
            "output_dir": str(tmp_path / "view"),
            "forbidden_source_episode_ids": [105, 106, 107, 108, 109],
            "sources": [
                {
                    "name": "bad",
                    "dataset_dir": str(source),
                    "split_path": str(split),
                    "include_roles": ["train", "val"],
                    "composite_start": 10000,
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="forbidden episode ids"):
        build_training_composite(spec)


def test_preflight_reports_pending_finetune_without_starting_training(
    tmp_path: Path,
) -> None:
    view = _view(tmp_path)
    threshold = tmp_path / "deadzone.json"
    threshold.write_text(
        json.dumps(
            {
                "deadzone_action": {
                    "swing": {"pos": 0.661, "neg": 0.721},
                    "boom": {"pos": 0.259, "neg": 0.357},
                    "stick": {"pos": 0.5, "neg": 0.5},
                    "bucket": {"pos": 0.408, "neg": 0.508},
                },
                "metadata": {
                    "action_domain": "direct_policy_output",
                    "policy_action_scale": [1.0, 1.0, 1.0, 1.0],
                },
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    _write(
        config,
        {
            "task": {
                "task_name": "preflight_test",
                "dataset_dir": str(view),
                "train_ready_manifest_path": str(view / "train_ready_manifest.json"),
                "camera_names": ["video4", "video5"],
            },
            "policy": {"class": "ACT", "low_dim_keys": ["qpos"]},
            "train": {
                "num_epochs": 2000,
                "split_path": str(view / "train_val_split.yaml"),
                "ckpt_dir": str(tmp_path / "ckpt"),
                "init_ckpt": str(tmp_path / "future.ckpt"),
                "effective_action": {
                    "enabled": True,
                    "threshold_json": str(threshold),
                },
            },
            "experiment_contract": {
                "expected_num_epochs": 2000,
                "policy_action_scale": [1.0, 1.0, 1.0, 1.0],
                "effective_action_enabled": True,
                "allow_pending_init_checkpoint": True,
                "forbidden_source_episode_ids": [105, 106, 107, 108, 109],
                "forbidden_legacy_heldout_dataset": str(tmp_path / "legacy"),
            },
        },
    )
    report = preflight_act_training_config(config, verify_hashes=True)
    assert report["status"] == "pending_dependency"
    assert report["training_started"] is False
    assert not (tmp_path / "ckpt").exists()


def test_preflight_accepts_declared_four_camera_role_experiment(
    tmp_path: Path,
) -> None:
    view = _view(tmp_path)
    config = tmp_path / "fourcam.yaml"
    cameras = ["video4", "video5", "video6", "video7"]
    _write(
        config,
        {
            "task": {
                "task_name": "fourcam_preflight_test",
                "dataset_dir": str(view),
                "train_ready_manifest_path": str(
                    view / "train_ready_manifest.json"
                ),
                "camera_names": cameras,
            },
            "policy": {
                "class": "ACT",
                "low_dim_keys": ["qpos"],
                "act_params": {
                    "camera_role_encoding": {
                        "enabled": True,
                        "roles": {
                            "video4": "eye",
                            "video5": "eye",
                            "video6": "stick",
                            "video7": "stick",
                        },
                    }
                },
            },
            "train": {
                "num_epochs": 2000,
                "split_path": str(view / "train_val_split.yaml"),
                "ckpt_dir": str(tmp_path / "fourcam_ckpt"),
                "effective_action": {"enabled": False},
            },
            "experiment_contract": {
                "expected_num_epochs": 2000,
                "expected_camera_names": cameras,
                "camera_role_encoding_enabled": True,
                "policy_action_scale": [1.0, 1.0, 1.0, 1.0],
                "effective_action_enabled": False,
                "forbidden_source_episode_ids": [105, 106, 107, 108, 109],
                "forbidden_legacy_heldout_dataset": str(tmp_path / "legacy"),
            },
        },
    )
    report = preflight_act_training_config(config, verify_hashes=True)
    assert report["status"] == "ready"
    assert report["camera_names"] == cameras
    assert report["camera_role_encoding_enabled"] is True
    assert report["deadzone_loss_enabled"] is False
    assert report["state_hold_transition_enabled"] is False


def test_preflight_rejects_camera_contract_mismatch(tmp_path: Path) -> None:
    view = _view(tmp_path)
    config = tmp_path / "mismatch.yaml"
    _write(
        config,
        {
            "task": {
                "task_name": "camera_mismatch",
                "dataset_dir": str(view),
                "train_ready_manifest_path": str(
                    view / "train_ready_manifest.json"
                ),
                "camera_names": ["video4", "video5"],
            },
            "policy": {"class": "ACT", "low_dim_keys": ["qpos"]},
            "train": {
                "num_epochs": 2000,
                "split_path": str(view / "train_val_split.yaml"),
                "ckpt_dir": str(tmp_path / "mismatch_ckpt"),
                "effective_action": {"enabled": False},
            },
            "experiment_contract": {
                "expected_num_epochs": 2000,
                "expected_camera_names": [
                    "video4",
                    "video5",
                    "video6",
                    "video7",
                ],
                "policy_action_scale": [1.0, 1.0, 1.0, 1.0],
                "effective_action_enabled": False,
                "forbidden_source_episode_ids": [105, 106, 107, 108, 109],
                "forbidden_legacy_heldout_dataset": str(tmp_path / "legacy"),
            },
        },
    )
    with pytest.raises(ValueError, match="camera_names does not match"):
        preflight_act_training_config(config, verify_hashes=True)
