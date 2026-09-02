from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from testbed.data.dataset import EpisodicDataset


def _thresholds() -> dict[str, dict[str, float]]:
    return {
        "swing": {"pos": 0.5, "neg": 0.5},
        "boom": {"pos": 0.4, "neg": 0.4},
        "stick": {"pos": 0.3, "neg": 0.3},
        "bucket": {"pos": 0.2, "neg": 0.2},
    }


def _write_episode(path: Path, *, action_prealigned: bool = True) -> None:
    action = np.zeros((6, 4), dtype=np.float32)
    action[1:3, 0] = 0.6
    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["is_real"] = True
        metadata = h5_file.create_group("metadata")
        metadata.attrs["action_prealigned"] = action_prealigned
        h5_file.create_dataset("action", data=action)
        h5_file.create_dataset(
            "observations/qpos",
            data=np.arange(24, dtype=np.float32).reshape(6, 4),
        )
        h5_file.create_dataset(
            "observations/qvel",
            data=np.zeros((6, 4), dtype=np.float32),
        )
        h5_file.create_dataset(
            "observations/images/video4",
            data=np.zeros((6, 2, 3, 3), dtype=np.uint8),
        )


def _norm_stats() -> dict[str, np.ndarray]:
    return {
        "action_mean": np.zeros(4, dtype=np.float32),
        "action_std": np.ones(4, dtype=np.float32),
        "proprio_mean": np.zeros(4, dtype=np.float32),
        "proprio_std": np.ones(4, dtype=np.float32),
    }


def _config(*, probability: float, append_samples_per_episode: int = 0) -> dict:
    return {
        "enabled": True,
        "thresholds": _thresholds(),
        "probability": probability,
        "hold_horizon_steps": 2,
        "append_samples_per_episode": append_samples_per_episode,
    }


def test_dataset_can_force_transition_anchor_and_emit_direction_mask(
    tmp_path: Path,
) -> None:
    _write_episode(tmp_path / "episode_1.hdf5")

    dataset = EpisodicDataset(
        [1],
        tmp_path,
        ["video4"],
        _norm_stats(),
        episode_len=6,
        low_dim_keys=["qpos"],
        action_chunk_size=2,
        state_hold_transition=_config(probability=1.0),
    )
    sample = dataset[0]

    assert isinstance(sample, dict)
    assert torch.equal(sample["proprio"], torch.arange(4, 8, dtype=torch.float32))
    expected = torch.zeros((4, 2), dtype=torch.bool)
    expected[0, 0] = True
    assert torch.equal(sample["state_hold_transition_mask"], expected)


def test_uniform_non_transition_sample_emits_an_explicit_zero_mask(
    tmp_path: Path, monkeypatch
) -> None:
    _write_episode(tmp_path / "episode_1.hdf5")
    monkeypatch.setattr(np.random, "choice", lambda values: int(values[0]))

    dataset = EpisodicDataset(
        [1],
        tmp_path,
        ["video4"],
        _norm_stats(),
        episode_len=6,
        action_chunk_size=2,
        state_hold_transition=_config(probability=0.0),
    )
    sample = dataset[0]

    assert isinstance(sample, dict)
    assert not sample["state_hold_transition_mask"].any()


def test_transition_sampling_rejects_unaligned_real_actions(tmp_path: Path) -> None:
    _write_episode(tmp_path / "episode_1.hdf5", action_prealigned=False)

    with pytest.raises(ValueError, match="action_prealigned=true"):
        EpisodicDataset(
            [1],
            tmp_path,
            ["video4"],
            _norm_stats(),
            episode_len=6,
            action_chunk_size=2,
            state_hold_transition=_config(probability=1.0),
        )


def test_appended_state_hold_tier_forces_anchor_without_changing_probability(
    tmp_path: Path, monkeypatch
) -> None:
    _write_episode(tmp_path / "episode_1.hdf5")
    monkeypatch.setattr(np.random, "choice", lambda values: int(values[0]))

    dataset = EpisodicDataset(
        [1],
        tmp_path,
        ["video4"],
        _norm_stats(),
        episode_len=6,
        low_dim_keys=["qpos"],
        action_chunk_size=2,
        state_hold_transition=_config(
            probability=0.0,
            append_samples_per_episode=1,
        ),
    )

    assert len(dataset) == 2
    ordinary = dataset[0]
    forced = dataset[1]
    assert not ordinary["state_hold_transition_mask"].any()
    assert forced["state_hold_transition_mask"][0, 0]
