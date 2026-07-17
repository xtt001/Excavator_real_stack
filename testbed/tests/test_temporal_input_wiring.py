from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from testbed.data.causal_visual_history import (
    CausalVisualHistory,
    resolve_temporal_input_config,
)
from testbed.data.dataset import EpisodicDataset
from testbed.policies.act.adapter import ACTAdapter


def _norm_stats() -> dict[str, np.ndarray | int]:
    return {
        "action_mean": np.zeros(4, dtype=np.float32),
        "action_std": np.ones(4, dtype=np.float32),
        "proprio_mean": np.zeros(4, dtype=np.float32),
        "proprio_std": np.ones(4, dtype=np.float32),
        "proprio_dim": 4,
        "qpos_only_dim": 4,
    }


def _write_episode(path: Path, *, steps: int = 4) -> None:
    image_rows = np.stack(
        [np.full((4, 5, 3), step, dtype=np.uint8) for step in range(steps)],
        axis=0,
    )
    with h5py.File(path, "w") as f:
        observations = f.create_group("observations")
        observations.create_dataset("qpos", data=np.zeros((steps, 4), dtype=np.float32))
        observations.create_dataset("qvel", data=np.zeros((steps, 4), dtype=np.float32))
        observations.create_group("images").create_dataset("fpv", data=image_rows)
        f.create_dataset("action", data=np.zeros((steps, 4), dtype=np.float32))


def test_temporal_dataset_uses_causal_startup_padding_and_frame_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_episode(tmp_path / "episode_0.hdf5")
    dataset = EpisodicDataset(
        [0],
        tmp_path,
        ["fpv"],
        _norm_stats(),
        temporal_input={"enabled": True, "history_steps": 3},
    )

    # Force a deterministic sample at t=2 after the constructor warm-up.
    monkeypatch.setattr(
        np.random,
        "choice",
        lambda values: int(np.asarray(values, dtype=np.int64)[2]),
    )
    sample = dataset[0]

    assert sample[0].shape == (3, 1, 3, 4, 5)
    # Pixel value is step index / 255 after the dataset conversion.
    np.testing.assert_allclose(
        sample[0][:, 0, 0, 0, 0].numpy(),
        np.asarray([0, 1, 2], dtype=np.float32) / 255.0,
    )


def test_temporal_config_is_disabled_by_default_and_validates_history() -> None:
    assert resolve_temporal_input_config(None) == {
        "enabled": False,
        "history_steps": 4,
    }
    assert resolve_temporal_input_config({"enabled": True, "history_length": 2}) == {
        "enabled": True,
        "history_steps": 2,
    }


class _RecordingModel:
    training = False

    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def eval(self) -> None:
        self.training = False

    def __call__(self, _proprio, image, _env_state):
        self.images.append(image.detach().cpu())
        return (
            torch.zeros((1, 1, 4), dtype=torch.float32),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _minimal_temporal_adapter(model: _RecordingModel) -> ACTAdapter:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.device = torch.device("cpu")
    adapter.norm_stats = {
        "action_mean": np.zeros(4, dtype=np.float32),
        "action_std": np.ones(4, dtype=np.float32),
        "proprio_mean": np.zeros(4, dtype=np.float32),
        "proprio_std": np.ones(4, dtype=np.float32),
    }
    adapter.temporal_agg = False
    adapter._camera_names = ["fpv"]
    adapter._low_dim_keys = ["qpos"]
    adapter._temporal_input = {"enabled": True, "history_steps": 3}
    adapter._visual_history = CausalVisualHistory(["fpv"], history_length=3)
    adapter._temporal_last_timestamps = {}
    adapter._temporal_fallback_timestamp = 0
    adapter._last_temporal_input_diagnostics = None
    adapter._model = model
    adapter._num_queries = 1
    adapter._t = 0
    adapter._cached_actions = None
    adapter._factorized_action = {"enabled": False}
    adapter._factorized_aggregator = None
    adapter._normalize = lambda image: image
    adapter._proprio_mean = torch.zeros(4)
    adapter._proprio_std = torch.ones(4)
    adapter._last_goal_effect_diagnostics = None
    return adapter


def test_adapter_history_uses_local_fallback_and_reset() -> None:
    model = _RecordingModel()
    adapter = _minimal_temporal_adapter(model)
    image = np.zeros((3, 4, 5), dtype=np.float32)

    adapter._predict_action_and_optional_intent({"qpos": np.zeros(4), "image_fpv": image})
    adapter._predict_action_and_optional_intent({"qpos": np.zeros(4), "image_fpv": image + 1})

    assert tuple(model.images[-1].shape) == (1, 3, 1, 3, 4, 5)
    assert adapter.temporal_input_diagnostics["timestamp_source"] == "local_step_fallback"
    assert adapter.temporal_input_diagnostics["valid_mask"]["fpv"] == [False, True, True]

    adapter.reset()
    adapter._predict_action_and_optional_intent({"qpos": np.zeros(4), "image_fpv": image + 2})
    assert adapter.temporal_input_diagnostics["valid_mask"]["fpv"] == [False, False, True]
