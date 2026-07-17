from __future__ import annotations

import pytest
import torch
from torch import nn

from testbed.policies.act.detr.models.detr_vae import (
    DETRVAE,
    TemporalFeatureMixer,
)


class _DummyBackbone(nn.Module):
    num_channels = 2

    def forward(self, image: torch.Tensor):
        # Keep the feature map small while retaining distinct values for the
        # newest-frame identity checks.
        features = image[:, :2, ::2, ::2]
        pos = torch.zeros(
            (1, 2, features.shape[-2], features.shape[-1]),
            dtype=image.dtype,
            device=image.device,
        )
        return [features], [pos]


class _PerImagePositionBackbone(_DummyBackbone):
    def forward(self, image: torch.Tensor):
        features, pos = super().forward(image)
        pos = [pos[0].expand(image.shape[0], -1, -1, -1).contiguous()]
        return features, pos


class _DummyTransformer(nn.Module):
    d_model = 2

    def __init__(self, num_queries: int):
        super().__init__()
        self.num_queries = num_queries

    def forward(
        self,
        src: torch.Tensor,
        _mask,
        _query_embed: torch.Tensor,
        _pos_embed: torch.Tensor,
        *_extra,
    ):
        pooled = src.mean(dim=(2, 3))
        hidden = pooled.unsqueeze(1).expand(-1, self.num_queries, -1)
        return hidden.unsqueeze(0)


class _DummyEncoder(nn.Module):
    def forward(self, input_tensor: torch.Tensor, **_kwargs):
        return input_tensor


def _build_model(temporal_input_config: dict | None = None) -> DETRVAE:
    return DETRVAE(
        backbones=[_DummyBackbone()],
        transformer=_DummyTransformer(num_queries=3),
        encoder=_DummyEncoder(),
        robot_state_dim=2,
        action_dim=2,
        num_queries=3,
        camera_names=["eye"],
        temporal_input_config=temporal_input_config,
    ).eval()


def test_disabled_temporal_input_keeps_five_dimensional_path() -> None:
    model = _build_model()
    image = torch.randn(2, 1, 3, 4, 4)
    output = model(torch.zeros(2, 2), image, None)

    assert output[0].shape == (2, 3, 2)
    assert not model.temporal_input_enabled
    assert model.temporal_feature_mixer is None


def test_temporal_input_accepts_causal_six_dimensional_history() -> None:
    model = _build_model({"enabled": True, "history_steps": 3})
    image = torch.randn(2, 3, 1, 3, 4, 4)
    output = model(torch.zeros(2, 2), image, None)

    assert output[0].shape == (2, 3, 2)
    assert model.temporal_input_enabled
    assert model.temporal_history_steps == 3

    with pytest.raises(ValueError, match="causal image shape"):
        model(torch.zeros(2, 2), image[:, -1], None)
    with pytest.raises(ValueError, match="history length mismatch"):
        model(torch.zeros(2, 2), image[:, :2], None)


def test_temporal_input_selects_newest_per_image_position_copy() -> None:
    model = DETRVAE(
        backbones=[_PerImagePositionBackbone()],
        transformer=_DummyTransformer(num_queries=3),
        encoder=_DummyEncoder(),
        robot_state_dim=2,
        action_dim=2,
        num_queries=3,
        camera_names=["eye"],
        temporal_input_config={"enabled": True, "history_steps": 3},
    ).eval()
    image = torch.randn(2, 3, 1, 3, 4, 4)

    output = model(torch.zeros(2, 2), image, None)

    assert output[0].shape == (2, 3, 2)


def test_temporal_feature_mixer_is_newest_frame_identity() -> None:
    mixer = TemporalFeatureMixer(channels=3, history_steps=3).eval()
    features = torch.randn(2, 3, 3, 2, 2)

    mixed = mixer(features)

    torch.testing.assert_close(mixed, features[:, -1])
    repeated = features[:, -1:].expand(-1, 3, -1, -1, -1)
    torch.testing.assert_close(mixer(repeated), features[:, -1])


def test_old_single_frame_state_loads_non_strict_and_temporal_is_exact_rollback() -> None:
    torch.manual_seed(7)
    single_frame = _build_model()
    temporal = _build_model({"enabled": True, "history_steps": 3})

    incompatible = temporal.load_state_dict(single_frame.state_dict(), strict=False)

    assert set(incompatible.missing_keys) == {
        "temporal_feature_mixer.proj.weight",
        "temporal_feature_mixer.proj.bias",
    }
    assert incompatible.unexpected_keys == []

    newest = torch.randn(2, 1, 3, 4, 4)
    history = newest.unsqueeze(1).expand(-1, 3, -1, -1, -1, -1)
    qpos = torch.zeros(2, 2)
    old_output = single_frame(qpos, newest, None)
    temporal_output = temporal(qpos, history, None)

    torch.testing.assert_close(temporal_output[0], old_output[0])
    torch.testing.assert_close(temporal_output[1], old_output[1])
