from __future__ import annotations

import pytest
import torch

from testbed.policies.act.checkpoint_init import expand_proprio_state_dict


def _state(width: int) -> dict[str, torch.Tensor]:
    return {
        "input_proj_robot_state.weight": torch.arange(
            2 * width, dtype=torch.float32
        ).reshape(2, width),
        "input_proj_robot_state.bias": torch.tensor([1.0, 2.0]),
        "encoder_joint_proj.weight": torch.arange(
            2 * width, dtype=torch.float32
        ).reshape(2, width)
        + 10.0,
        "encoder_joint_proj.bias": torch.tensor([3.0, 4.0]),
        "action_head.weight": torch.ones((4, 2)),
    }


def test_expansion_copies_qpos_columns_and_zeroes_new_features() -> None:
    source = _state(4)
    target = _state(12)

    expanded, report = expand_proprio_state_dict(source=source, target=target)

    for key in ("input_proj_robot_state.weight", "encoder_joint_proj.weight"):
        assert torch.equal(expanded[key][:, :4], source[key])
        assert torch.equal(expanded[key][:, 4:], torch.zeros((2, 8)))
    assert torch.equal(
        expanded["input_proj_robot_state.bias"],
        source["input_proj_robot_state.bias"],
    )
    assert len(report["expanded"]) == 2


def test_zero_new_features_preserve_projection_exactly() -> None:
    source = _state(4)
    target = _state(12)
    expanded, _ = expand_proprio_state_dict(source=source, target=target)
    qpos = torch.tensor([0.2, -0.3, 0.4, -0.5])
    original = source["input_proj_robot_state.weight"] @ qpos
    new_input = torch.cat([qpos, torch.zeros(8)])

    actual = expanded["input_proj_robot_state.weight"] @ new_input

    assert torch.equal(actual, original)


def test_expansion_rejects_any_unapproved_shape_change() -> None:
    source = _state(4)
    target = _state(12)
    target["action_head.weight"] = torch.ones((4, 3))

    with pytest.raises(ValueError, match="unsupported shape mismatch"):
        expand_proprio_state_dict(source=source, target=target)


def test_expansion_rejects_key_drift() -> None:
    source = _state(4)
    target = _state(12)
    source.pop("encoder_joint_proj.bias")

    with pytest.raises(ValueError, match="keys do not match"):
        expand_proprio_state_dict(source=source, target=target)


def test_expansion_allows_missing_optional_auxiliary_head() -> None:
    source = _state(4)
    target = _state(12)
    target["intent_head.weight"] = torch.ones((8, 2))
    target["intent_head.bias"] = torch.zeros(8)

    expanded, report = expand_proprio_state_dict(source=source, target=target)

    assert "intent_head.weight" in report["missing_optional_keys"]
    assert "intent_head.bias" in report["missing_optional_keys"]
    assert "intent_head.weight" not in expanded
