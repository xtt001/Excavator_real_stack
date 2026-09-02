from __future__ import annotations

import torchvision

from testbed.policies.act.detr.models.backbone import Backbone


def test_checkpoint_backbone_can_be_built_without_pretrained_download(
    monkeypatch,
) -> None:
    original = torchvision.models.resnet18
    calls: list[dict] = []

    def recording_factory(**kwargs):
        calls.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(torchvision.models, "resnet18", recording_factory)

    Backbone(
        "resnet18",
        train_backbone=True,
        return_interm_layers=False,
        dilation=False,
        pretrained=False,
    )

    assert len(calls) == 1
    assert calls[0]["weights"] is None
    assert "pretrained" not in calls[0]
