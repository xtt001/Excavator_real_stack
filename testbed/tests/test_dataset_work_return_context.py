from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from testbed.data.dataset import EpisodicDataset
from testbed.data.work_return_context import (
    WORK_CONTEXT_KEY,
    WORK_CONTEXT_SCHEMA,
    derive_work_return_context,
)


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    episodes = root / "episodes"
    episodes.mkdir()
    action = np.zeros((100, 4), dtype=np.float32)
    action[5:30, 1] = -0.5
    action[35:55, 0] = 0.8
    action[60:80, 3] = 0.6
    action[85:100, 0] = -0.8
    with h5py.File(episodes / "episode_1.hdf5", "w") as handle:
        handle.attrs["is_real"] = True
        metadata = handle.create_group("metadata")
        metadata.attrs["action_prealigned"] = True
        handle.create_dataset("action", data=action)
        handle.create_dataset(
            "observations/qpos", data=np.zeros((100, 4), dtype=np.float32)
        )
        handle.create_dataset(
            "observations/qvel", data=np.zeros((100, 4), dtype=np.float32)
        )
        handle.create_dataset(
            "observations/images/video4",
            data=np.zeros((100, 2, 3, 3), dtype=np.uint8),
        )
        condition = np.tile(
            np.asarray([-1.0, 1.0], dtype=np.float32), (100, 1)
        )
        handle.create_dataset(
            "conditions/real_transition_condition_v1", data=condition
        )
        handle.create_dataset(
            "conditions/valid_mask", data=np.ones((100, 10), dtype=np.uint8)
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
    context = derive_work_return_context(
        action,
        positive_thresholds=[0.661, 0.259, 0.5, 0.408],
        negative_thresholds=[0.721, 0.357, 0.5, 0.508],
        action_window_steps=10,
    )
    manifest = root / "context.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": WORK_CONTEXT_SCHEMA,
                "dataset_root": str(root),
                "chunk_steps": 10,
                "episodes": [
                    {
                        "episode_id": 1,
                        "current_anchor": "B",
                        "dig_target": "B",
                        "next_target": "A",
                        "work_complete_boundary_row": context.boundary_row,
                        "candidate_counts": {
                            "work": int(context.work_starts.size),
                            "return": int(context.return_starts.size),
                        },
                    }
                ],
            }
        )
    )
    return episodes, threshold, manifest


def test_dataset_balances_full_work_and_return_sequences(
    tmp_path: Path, monkeypatch
) -> None:
    episodes, threshold, manifest = _fixture(tmp_path)
    monkeypatch.setattr(np.random, "choice", lambda values: int(values[0]))
    stats = {
        "action_mean": np.zeros(4, dtype=np.float32),
        "action_std": np.ones(4, dtype=np.float32),
        "proprio_mean": np.zeros(16, dtype=np.float32),
        "proprio_std": np.ones(16, dtype=np.float32),
    }
    dataset = EpisodicDataset(
        [1],
        episodes,
        ["video4"],
        stats,
        episode_len=100,
        low_dim_keys=[
            "qpos",
            "real_transition_condition_v1",
            "qvel",
            WORK_CONTEXT_KEY,
        ],
        action_chunk_size=10,
        work_return_context={
            "enabled": True,
            "condition_key": WORK_CONTEXT_KEY,
            "threshold_json": str(threshold),
            "manifest_path": str(manifest),
            "action_window_steps": 10,
            "append_samples_per_episode": 1,
        },
    )

    assert len(dataset) == 2
    work = dataset[0][1]
    returning = dataset[1][1]
    torch.testing.assert_close(
        work[10:16], torch.asarray([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        returning[10:16], torch.asarray([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    )
    assert work[4] == -1.0
    assert returning[4] == -1.0
