from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from testbed.data.action_primitive_islands import (
    ACTION_PRIMITIVE_KEY,
    ACTION_PRIMITIVE_SCHEMA,
    PRIMITIVE_NAMES,
    derive_action_primitive_islands,
)
from testbed.data.dataset import EpisodicDataset


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    episodes = root / "episodes"
    episodes.mkdir()
    action = np.zeros((90, 4), dtype=np.float32)
    action[5:25, 1] = -0.5
    action[30:50, 0] = 0.8
    action[55:75, 3] = 0.6
    action[75:90, 0] = -0.8
    episode = episodes / "episode_1.hdf5"
    with h5py.File(episode, "w") as handle:
        handle.attrs["is_real"] = True
        metadata = handle.create_group("metadata")
        metadata.attrs["action_prealigned"] = True
        handle.create_dataset("action", data=action)
        handle.create_dataset(
            "observations/qpos", data=np.zeros((90, 4), dtype=np.float32)
        )
        handle.create_dataset(
            "observations/qvel", data=np.zeros((90, 4), dtype=np.float32)
        )
        handle.create_dataset(
            "observations/images/video4",
            data=np.zeros((90, 2, 3, 3), dtype=np.uint8),
        )
        condition = np.zeros((90, 2), dtype=np.float32)
        condition[:, 0] = -1.0
        condition[:, 1] = 1.0
        handle.create_dataset(
            "conditions/real_transition_condition_v1", data=condition
        )
        handle.create_dataset(
            "conditions/valid_mask",
            data=np.ones((90, 10), dtype=np.uint8),
        )
    threshold = root / "deadzone.json"
    threshold.write_text(
        json.dumps(
            {
                "deadzone_action": {
                    "swing": {"pos": 0.661, "neg": 0.721},
                    "boom": {"pos": 0.259, "neg": 0.357},
                    "stick": {"pos": 0.5, "neg": 0.5},
                    "bucket": {"pos": 0.408, "neg": 0.508},
                }
            }
        )
    )
    islands = derive_action_primitive_islands(
        action,
        positive_thresholds=[0.661, 0.259, 0.5, 0.408],
        negative_thresholds=[0.721, 0.357, 0.5, 0.508],
        action_window_steps=10,
    )
    manifest = root / "primitive_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": ACTION_PRIMITIVE_SCHEMA,
                "dataset_root": str(root),
                "chunk_steps": 10,
                "primitive_names": list(PRIMITIVE_NAMES),
                "episodes": [
                    {
                        "episode_id": 1,
                        "candidate_counts": {
                            name: int(islands.candidate_starts[name].size)
                            for name in PRIMITIVE_NAMES
                        },
                    }
                ],
            }
        )
    )
    return episodes, threshold, manifest


def _norm_stats() -> dict[str, np.ndarray]:
    return {
        "action_mean": np.zeros(4, dtype=np.float32),
        "action_std": np.ones(4, dtype=np.float32),
        "proprio_mean": np.zeros(14, dtype=np.float32),
        "proprio_std": np.ones(14, dtype=np.float32),
    }


def test_dataset_balances_four_oracle_primitive_tiers(
    tmp_path: Path, monkeypatch
) -> None:
    episodes, threshold, manifest = _write_fixture(tmp_path)
    monkeypatch.setattr(np.random, "choice", lambda values: int(values[0]))
    dataset = EpisodicDataset(
        [1],
        episodes,
        ["video4"],
        _norm_stats(),
        episode_len=90,
        low_dim_keys=[
            "qpos",
            "real_transition_condition_v1",
            "qvel",
            ACTION_PRIMITIVE_KEY,
        ],
        action_chunk_size=10,
        action_primitive_islands={
            "enabled": True,
            "condition_key": ACTION_PRIMITIVE_KEY,
            "primitive_names": list(PRIMITIVE_NAMES),
            "threshold_json": str(threshold),
            "manifest_path": str(manifest),
            "action_window_steps": 10,
            "append_samples_per_episode": 3,
        },
    )

    assert len(dataset) == 4
    for index in range(4):
        sample = dataset[index]
        assert isinstance(sample, tuple)
        proprio = sample[1]
        assert torch.equal(
            proprio[-4:],
            torch.nn.functional.one_hot(torch.tensor(index), 4).float(),
        )


def test_dataset_rejects_manifest_population_drift(tmp_path: Path) -> None:
    episodes, threshold, manifest = _write_fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["episodes"][0]["candidate_counts"]["swing_out"] += 1
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="population changed"):
        EpisodicDataset(
            [1],
            episodes,
            ["video4"],
            _norm_stats(),
            episode_len=90,
            low_dim_keys=[
                "qpos",
                "real_transition_condition_v1",
                "qvel",
                ACTION_PRIMITIVE_KEY,
            ],
            action_chunk_size=10,
            action_primitive_islands={
                "enabled": True,
                "condition_key": ACTION_PRIMITIVE_KEY,
                "primitive_names": list(PRIMITIVE_NAMES),
                "threshold_json": str(threshold),
                "manifest_path": str(manifest),
                "action_window_steps": 10,
                "append_samples_per_episode": 3,
            },
        )
