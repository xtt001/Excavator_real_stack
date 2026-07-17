from __future__ import annotations

from pathlib import Path

import pytest
import torch

from testbed.policies.act.trainer import ACTTrainer


def test_load_model_state_dict_accepts_wrapped_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "wrapped.ckpt"
    expected = {"layer.weight": torch.tensor([1.0])}
    torch.save({"model_state_dict": expected, "optimizer_state_dict": {}}, path)

    loaded = ACTTrainer._load_model_state_dict(path)

    assert set(loaded) == {"layer.weight"}
    assert torch.equal(loaded["layer.weight"], expected["layer.weight"])


def test_load_model_state_dict_accepts_raw_mapping(tmp_path: Path) -> None:
    path = tmp_path / "raw.ckpt"
    expected = {"layer.bias": torch.tensor([2.0])}
    torch.save(expected, path)

    loaded = ACTTrainer._load_model_state_dict(path)

    assert torch.equal(loaded["layer.bias"], expected["layer.bias"])


@pytest.mark.parametrize("payload", [[], {}, {"model_state_dict": {}}])
def test_load_model_state_dict_rejects_invalid_payload(
    tmp_path: Path, payload: object
) -> None:
    path = tmp_path / "invalid.ckpt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="model state"):
        ACTTrainer._load_model_state_dict(path)
