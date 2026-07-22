from __future__ import annotations

import torch
from torch import nn

from testbed.policies.act.detr.models.detr_vae import DETRVAE


class _RecordingBackbone(nn.Module):
    num_channels = 4

    def __init__(self) -> None:
        super().__init__()
        self.call_shapes: list[tuple[int, ...]] = []

    def forward(self, image: torch.Tensor):
        self.call_shapes.append(tuple(image.shape))
        base = image.mean(dim=1, keepdim=True)
        features = base.repeat(1, self.num_channels, 1, 1)
        position = torch.stack(
            [base[:, 0] + float(index) for index in range(self.num_channels)],
            dim=1,
        )
        return [features], [position]


class _TransformerStub(nn.Module):
    d_model = 4

    def forward(
        self,
        src: torch.Tensor,
        mask,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        latent_input=None,
        proprio_input=None,
        additional_pos_embed=None,
    ) -> torch.Tensor:
        del mask, pos_embed, latent_input, proprio_input, additional_pos_embed
        return torch.zeros(
            1,
            src.shape[0],
            query_embed.shape[0],
            self.d_model,
            dtype=src.dtype,
            device=src.device,
        )


class _EncoderStub(nn.Module):
    def forward(self, src: torch.Tensor, **_kwargs) -> torch.Tensor:
        return src


def _build_model(backbone: _RecordingBackbone) -> DETRVAE:
    return DETRVAE(
        [backbone],
        _TransformerStub(),
        _EncoderStub(),
        robot_state_dim=4,
        action_dim=4,
        num_queries=3,
        camera_names=["video4", "video5", "video6", "video7"],
    )


def test_batched_camera_backbone_matches_sequential_layout() -> None:
    torch.manual_seed(7)
    backbone = _RecordingBackbone()
    model = _build_model(backbone).eval()
    image = torch.randn(2, 4, 3, 5, 7)

    sequential_features, sequential_positions = model._extract_camera_features(
        image,
        batch_cameras=False,
    )
    assert backbone.call_shapes == [(2, 3, 5, 7)] * 4

    backbone.call_shapes.clear()
    batched_features, batched_positions = model._extract_camera_features(
        image,
        batch_cameras=True,
    )

    assert backbone.call_shapes == [(8, 3, 5, 7)]
    assert len(batched_features) == len(sequential_features) == 4
    assert len(batched_positions) == len(sequential_positions) == 4
    for actual, expected in zip(batched_features, sequential_features, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    for actual, expected in zip(batched_positions, sequential_positions, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_inference_forward_calls_shared_backbone_once() -> None:
    backbone = _RecordingBackbone()
    model = _build_model(backbone).eval()
    image = torch.randn(1, 4, 3, 5, 7)
    qpos = torch.zeros(1, 4)

    with torch.inference_mode():
        output = model(qpos, image, None)
        action, is_pad, latent, intent = output[:4]

    assert backbone.call_shapes == [(4, 3, 5, 7)]
    assert action.shape == (1, 3, 4)
    assert is_pad.shape == (1, 3, 1)
    assert latent == [None, None]
    assert intent is None


def test_training_forward_keeps_historical_per_camera_backbone_calls() -> None:
    backbone = _RecordingBackbone()
    model = _build_model(backbone).train()
    image = torch.randn(2, 4, 3, 5, 7)
    qpos = torch.zeros(2, 4)
    actions = torch.zeros(2, 3, 4)
    is_pad = torch.zeros(2, 3, dtype=torch.bool)

    output = model(qpos, image, None, actions=actions, is_pad=is_pad)
    action, _, _, _ = output[:4]

    assert backbone.call_shapes == [(2, 3, 5, 7)] * 4
    assert action.shape == (2, 3, 4)
