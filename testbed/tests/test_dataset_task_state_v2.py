from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from testbed.data.dataset import EpisodicDataset
from testbed.data.task_state_v2 import (
    TASK_STATE_V2_DIM,
    TASK_STATE_V2_KEY,
    TASK_STATE_V2_SCHEMA,
    TASK_STATE_V2_TIERS,
    task_state_candidate_starts,
)


def _fixture(root: Path) -> tuple[Path, Path]:
    episodes = root / "episodes"
    episodes.mkdir()
    total_steps = 40
    work_complete = 10
    return_commit = 13
    with h5py.File(episodes / "episode_1.hdf5", "w") as handle:
        handle.attrs["is_real"] = True
        metadata = handle.create_group("metadata")
        metadata.attrs["action_prealigned"] = True
        handle.create_dataset(
            "action", data=np.zeros((total_steps, 4), dtype=np.float32)
        )
        handle.create_dataset(
            "observations/qpos",
            data=np.zeros((total_steps, 4), dtype=np.float32),
        )
        handle.create_dataset(
            "observations/qvel",
            data=np.zeros((total_steps, 4), dtype=np.float32),
        )
        handle.create_dataset(
            "observations/images/video4",
            data=np.zeros((total_steps, 2, 3, 3), dtype=np.uint8),
        )
    candidates = task_state_candidate_starts(
        total_steps=total_steps,
        work_complete_row=work_complete,
        return_commit_row=return_commit,
        action_window_steps=5,
    ).by_name()
    manifest = root / "task_state_v2.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": TASK_STATE_V2_SCHEMA,
                "dataset_root": str(root),
                "chunk_steps": 5,
                "task_state_dim": TASK_STATE_V2_DIM,
                "tier_names": list(TASK_STATE_V2_TIERS),
                "episodes": [
                    {
                        "episode_id": 1,
                        "n_rows": total_steps,
                        "current_side": "B",
                        "dig_target": "B",
                        "next_target": "A",
                        "work_complete_row": work_complete,
                        "return_commit_row": return_commit,
                        "candidate_starts": {
                            name: values.astype(int).tolist()
                            for name, values in candidates.items()
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return episodes, manifest


def test_dataset_emits_four_balanced_task_states_and_masks_boundary_chunk(
    tmp_path: Path, monkeypatch
) -> None:
    episodes, manifest = _fixture(tmp_path)
    monkeypatch.setattr(np.random, "choice", lambda values: int(values[0]))
    stats = {
        "action_mean": np.zeros(4, dtype=np.float32),
        "action_std": np.ones(4, dtype=np.float32),
        "proprio_mean": np.zeros(13, dtype=np.float32),
        "proprio_std": np.ones(13, dtype=np.float32),
    }
    dataset = EpisodicDataset(
        [1],
        episodes,
        ["video4"],
        stats,
        episode_len=40,
        low_dim_keys=["qpos", "qvel", TASK_STATE_V2_KEY],
        action_chunk_size=5,
        task_state_v2={
            "enabled": True,
            "condition_key": TASK_STATE_V2_KEY,
            "manifest_path": str(manifest),
            "action_window_steps": 5,
            "append_samples_per_episode": 3,
        },
    )

    assert len(dataset) == 4
    work_start = dataset[0]
    work_body = dataset[1]
    boundary = dataset[2]
    returning = dataset[3]
    torch.testing.assert_close(
        work_start["proprio"][8:13],
        torch.asarray([1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        work_body["proprio"][8:13],
        torch.asarray([1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        boundary["proprio"][8:13],
        torch.asarray([1.0, 1.0, 1.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        returning["proprio"][8:13],
        torch.asarray([1.0, 1.0, 1.0, 1.0, -1.0]),
    )
    torch.testing.assert_close(
        boundary["is_pad"][:5],
        torch.asarray([False, False, False, True, True]),
    )
    assert not bool(returning["is_pad"][:5].any())
    assert not bool(work_start["task_state_v2_uncommitted"])
    assert bool(boundary["task_state_v2_uncommitted"])
    assert not bool(returning["task_state_v2_uncommitted"])
