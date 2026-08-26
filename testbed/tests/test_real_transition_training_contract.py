from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from testbed.data.dataset import load_data
from testbed.policies.act.trainer import _load_conditioned_warm_start


def _write_cycle(path: Path, *, qpos_value: float, target_code: int) -> None:
    length = 4
    with h5py.File(path, "w") as output:
        output.attrs["is_real"] = True
        metadata = output.create_group("metadata")
        metadata.attrs["action_prealigned"] = True
        observations = output.create_group("observations")
        observations.create_dataset(
            "qpos", data=np.full((length, 4), qpos_value, dtype=np.float32)
        )
        observations.create_dataset(
            "qvel", data=np.zeros((length, 4), dtype=np.float32)
        )
        images = observations.create_group("images")
        images.create_dataset(
            "video4", data=np.zeros((length, 8, 8, 3), dtype=np.uint8)
        )
        output.create_dataset("action", data=np.zeros((length, 4), dtype=np.float32))
        conditions = output.create_group("conditions")
        conditions.create_dataset(
            "real_transition_condition_v1",
            data=np.tile(np.asarray([target_code, 1], dtype=np.float32), (length, 1)),
        )
        # Make the explicit contract observable in the returned is_pad mask.
        conditions.create_dataset(
            "valid_mask",
            data=np.tile(np.asarray([1, 0, 0], dtype=np.uint8), (length, 1)),
        )


def _write_split_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "real_transition_cycle_split_manifest_v1",
                "split_owner": "source_block_before_cycle_materialization",
                "episodes": rows,
            }
        ),
        encoding="utf-8",
    )


def test_loader_consumes_condition_mask_and_source_block_split(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    _write_cycle(episodes / "episode_0.hdf5", qpos_value=1.0, target_code=1)
    _write_cycle(episodes / "episode_1.hdf5", qpos_value=100.0, target_code=-1)
    _write_cycle(episodes / "episode_2.hdf5", qpos_value=200.0, target_code=1)
    split_manifest = tmp_path / "split_manifest.json"
    _write_split_manifest(
        split_manifest,
        [
            {"episode_id": 0, "split": "train", "source_block_id": "b01"},
            {"episode_id": 1, "split": "validation", "source_block_id": "b02"},
            {"episode_id": 2, "split": "locked_test", "source_block_id": "b03"},
        ],
    )

    train_loader, _val_loader, stats, is_real, split_info = load_data(
        dataset_dir=episodes,
        num_episodes=10,
        camera_names=["video4"],
        episode_len=None,
        batch_size_train=1,
        batch_size_val=1,
        num_workers=0,
        pin_memory=False,
        split_manifest_path=split_manifest,
        reuse_split=True,
        low_dim_keys=["qpos", "real_transition_condition_v1"],
        episode_ids=[0, 1, 2],
        action_chunk_size=3,
    )

    assert is_real is True
    assert split_info["train_ids"] == [0]
    assert split_info["val_ids"] == [1]
    assert split_info["locked_test_ids"] == [2]
    assert split_info["normalization_episode_ids"] == [0]
    np.testing.assert_allclose(stats["proprio_mean"][:4], 1.0)
    np.testing.assert_allclose(stats["proprio_mean"][4:], [0.0, 0.0])
    np.testing.assert_allclose(stats["proprio_std"][4:], [1.0, 1.0])

    _image, proprio, _action, is_pad = next(iter(train_loader))
    np.testing.assert_allclose(proprio.numpy()[0, 4:], [1.0, 1.0])
    assert is_pad.numpy()[0, :3].tolist() == [False, True, True]


def test_loader_exposes_qvel_in_the_goal_conditioned_proprio_contract(
    tmp_path: Path,
) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    _write_cycle(episodes / "episode_0.hdf5", qpos_value=1.0, target_code=1)
    _write_cycle(episodes / "episode_1.hdf5", qpos_value=2.0, target_code=-1)
    split_manifest = tmp_path / "split_manifest.json"
    _write_split_manifest(
        split_manifest,
        [
            {"episode_id": 0, "split": "train", "source_block_id": "b01"},
            {"episode_id": 1, "split": "validation", "source_block_id": "b02"},
        ],
    )

    train_loader, _val_loader, stats, _is_real, split_info = load_data(
        dataset_dir=episodes,
        num_episodes=10,
        camera_names=["video4"],
        episode_len=None,
        batch_size_train=1,
        batch_size_val=1,
        num_workers=0,
        pin_memory=False,
        split_manifest_path=split_manifest,
        low_dim_keys=["qpos", "qvel", "real_transition_condition_v1"],
        episode_ids=[0, 1],
        action_chunk_size=3,
    )

    assert split_info["low_dim_keys"] == [
        "qpos",
        "qvel",
        "real_transition_condition_v1",
    ]
    assert split_info["low_dim_dim"] == 10
    assert stats["proprio_dim"] == 10
    _image, proprio, _action, _is_pad = next(iter(train_loader))
    assert proprio.shape[-1] == 10


def test_manifest_split_rejects_cross_split_source_block(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    _write_cycle(episodes / "episode_0.hdf5", qpos_value=1.0, target_code=1)
    _write_cycle(episodes / "episode_1.hdf5", qpos_value=2.0, target_code=-1)
    split_manifest = tmp_path / "split_manifest.json"
    _write_split_manifest(
        split_manifest,
        [
            {"episode_id": 0, "split": "train", "source_block_id": "b01"},
            {"episode_id": 1, "split": "validation", "source_block_id": "b01"},
        ],
    )

    with pytest.raises(ValueError, match="source_block_id appears in multiple splits"):
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
            action_chunk_size=3,
        )


class _FakeAdapter:
    def __init__(self) -> None:
        self.loaded: dict[str, torch.Tensor] | None = None
        self._state = {
            "model.input_proj_robot_state.weight": torch.full((3, 6), 9.0),
            "model.input_proj_robot_state.bias": torch.zeros(3),
            "model.encoder_joint_proj.weight": torch.full((3, 6), 9.0),
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state

    def load_state_dict(
        self, state: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        assert strict is True
        self.loaded = state


def test_warm_start_copies_qpos_and_zeros_condition_columns() -> None:
    adapter = _FakeAdapter()
    source = {
        "model.input_proj_robot_state.weight": torch.arange(
            12, dtype=torch.float32
        ).reshape(3, 4),
        "model.input_proj_robot_state.bias": torch.arange(3, dtype=torch.float32),
        "model.encoder_joint_proj.weight": torch.arange(
            12, dtype=torch.float32
        ).reshape(3, 4)
        + 20,
    }

    expanded = _load_conditioned_warm_start(adapter, source)  # type: ignore[arg-type]

    assert len(expanded) == 2
    assert adapter.loaded is not None
    for key in expanded:
        torch.testing.assert_close(adapter.loaded[key][:, :4], source[key])
        torch.testing.assert_close(adapter.loaded[key][:, 4:], torch.zeros(3, 2))
    torch.testing.assert_close(
        adapter.loaded["model.input_proj_robot_state.bias"],
        source["model.input_proj_robot_state.bias"],
    )
