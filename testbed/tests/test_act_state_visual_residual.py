from __future__ import annotations

import pytest
import torch

from testbed.actions.policy import _act_policy_config_from_resolved
from testbed.policies.act.adapter import ACTAdapter
from testbed.policies.act.state_visual_residual import (
    resolve_state_visual_residual_config,
    stage_for_epoch,
)
from testbed.policies.act.trainer import (
    _load_state_visual_residual_warm_start,
    _load_state_visual_task_state_v2_warm_start,
    _load_vision_backbone_warm_start,
)


def _config() -> dict:
    return resolve_state_visual_residual_config(
        {
            "enabled": True,
            "low_hidden_dim": 128,
            "residual_keep_indices": [0, 1, 2, 3],
            "stages": [
                {
                    "name": "low",
                    "start_epoch": 0,
                    "end_epoch": 2,
                    "train_low": True,
                    "train_residual": False,
                    "residual_scale": 0.0,
                },
                {
                    "name": "joint",
                    "start_epoch": 2,
                    "end_epoch": 4,
                    "train_low": True,
                    "train_residual": True,
                    "residual_scale": 1.0,
                },
            ],
        },
        robot_state_dim=10,
        num_queries=20,
        action_dim=4,
    )


def test_stage_schedule_is_half_open() -> None:
    config = _config()

    assert stage_for_epoch(config, 0)["name"] == "low"
    assert stage_for_epoch(config, 1)["name"] == "low"
    assert stage_for_epoch(config, 2)["name"] == "joint"
    with pytest.raises(ValueError, match="outside"):
        stage_for_epoch(config, 4)


def test_stage_schedule_rejects_gaps() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        resolve_state_visual_residual_config(
            {
                "enabled": True,
                "stages": [
                    {
                        "name": "a",
                        "start_epoch": 0,
                        "end_epoch": 2,
                        "train_low": True,
                        "train_residual": False,
                    },
                    {
                        "name": "b",
                        "start_epoch": 3,
                        "end_epoch": 4,
                        "train_low": True,
                        "train_residual": True,
                    },
                ],
            },
            robot_state_dim=10,
            num_queries=20,
            action_dim=4,
        )


def test_vision_warm_start_does_not_copy_control_weights() -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.values = {
                "backbones.0.0.body.conv1.weight": torch.zeros(2, 2),
                "input_proj_robot_state.weight": torch.zeros(2, 2),
                "action_head.weight": torch.zeros(2, 2),
            }

        def state_dict(self):
            return self.values

        def load_state_dict(self, values, strict=True):
            assert strict
            self.values = values

    adapter = FakeAdapter()
    copied = _load_vision_backbone_warm_start(
        adapter,
        {
            "backbones.0.0.body.conv1.weight": torch.ones(2, 2),
            "input_proj_robot_state.weight": torch.ones(2, 2),
            "action_head.weight": torch.ones(2, 2),
        },
    )

    assert copied == ["backbones.0.0.body.conv1.weight"]
    assert torch.all(adapter.values["backbones.0.0.body.conv1.weight"] == 1)
    assert torch.all(adapter.values["input_proj_robot_state.weight"] == 0)
    assert torch.all(adapter.values["action_head.weight"] == 0)


def test_resolved_bundle_loader_preserves_state_visual_residual_config() -> None:
    config = _act_policy_config_from_resolved(
        {
            "task": {
                "equipment_model": "real_excavator",
                "camera_names": ["video4"],
            },
            "policy": {
                "low_dim_keys": [
                    "qpos",
                    "real_transition_condition_v1",
                    "qvel",
                ],
                "act_params": {"chunk_size": 20, "state_dim": 10},
            },
            "train": {
                "state_visual_residual": {
                    "enabled": True,
                    "low_hidden_dim": 256,
                    "residual_keep_indices": [0, 1, 2, 3],
                    "stages": [
                        {
                            "name": "joint",
                            "start_epoch": 0,
                            "end_epoch": 1,
                            "train_low": True,
                            "train_residual": True,
                            "residual_scale": 1.0,
                        }
                    ],
                }
            },
        }
    )

    assert config["state_visual_residual"]["enabled"] is True
    assert config["state_visual_residual"]["residual_keep_indices"] == [0, 1, 2, 3]


def test_state_visual_warm_start_resets_low_head_and_expands_proprio() -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.values = {
                "low_dim_action_head.0.weight": torch.zeros(2, 4),
                "low_dim_action_head.0.bias": torch.zeros(2),
                "state_visual_residual_proprio_mask": torch.tensor(
                    [1.0, 1.0, 0.0, 0.0]
                ),
                "input_proj_robot_state.weight": torch.zeros(2, 4),
                "encoder_joint_proj.weight": torch.zeros(2, 4),
                "action_head.weight": torch.zeros(2, 2),
            }

        def state_dict(self):
            return self.values

        def load_state_dict(self, values, strict=True):
            assert strict
            self.values = values

    adapter = FakeAdapter()
    report = _load_state_visual_residual_warm_start(
        adapter,
        {
            "low_dim_action_head.0.weight": torch.ones(2, 2),
            "low_dim_action_head.0.bias": torch.ones(2),
            "state_visual_residual_proprio_mask": torch.ones(2),
            "input_proj_robot_state.weight": torch.ones(2, 2),
            "encoder_joint_proj.weight": torch.ones(2, 2),
            "action_head.weight": torch.ones(2, 2),
        },
    )

    assert report["expanded"] == [
        "encoder_joint_proj.weight",
        "input_proj_robot_state.weight",
    ]
    assert torch.all(adapter.values["low_dim_action_head.0.weight"] == 0)
    assert torch.all(adapter.values["low_dim_action_head.0.bias"] == 0)
    assert torch.equal(
        adapter.values["state_visual_residual_proprio_mask"],
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )
    assert torch.all(adapter.values["input_proj_robot_state.weight"][:, :2] == 1)
    assert torch.all(adapter.values["input_proj_robot_state.weight"][:, 2:] == 0)
    assert torch.all(adapter.values["encoder_joint_proj.weight"][:, :2] == 1)
    assert torch.all(adapter.values["encoder_joint_proj.weight"][:, 2:] == 0)
    assert torch.all(adapter.values["action_head.weight"] == 1)


def test_legacy_checkpoint_gets_disabled_residual_buffer_defaults() -> None:
    class FakeModel:
        state_visual_residual_config = {"enabled": False}

        def __init__(self) -> None:
            self.loaded = None

        def state_dict(self):
            return {
                "weight": torch.zeros(1),
                "state_visual_residual_proprio_mask": torch.ones(4),
                "state_visual_residual_scale": torch.ones(1),
            }

        def load_state_dict(self, values, strict=True):
            assert strict
            self.loaded = values
            return "loaded"

    adapter = object.__new__(ACTAdapter)
    adapter._model = FakeModel()

    assert adapter.load_state_dict({"weight": torch.ones(1)}) == "loaded"
    assert set(adapter._model.loaded) == {
        "weight",
        "state_visual_residual_proprio_mask",
        "state_visual_residual_scale",
    }


def test_task_state_v2_warm_start_semantically_remaps_qvel_and_target() -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.values = {
                "low_dim_action_head.0.weight": torch.zeros(2, 13),
                "low_dim_action_head.0.bias": torch.zeros(2),
                "low_dim_action_head.1.weight": torch.zeros(2),
                "input_proj_robot_state.weight": torch.zeros(2, 13),
                "encoder_joint_proj.weight": torch.zeros(2, 13),
                "state_visual_residual_proprio_mask": torch.tensor(
                    [1.0] * 4 + [0.0] * 9
                ),
                "state_visual_residual_scale": torch.ones(1),
                "action_head.weight": torch.zeros(2, 2),
            }

        def state_dict(self):
            return self.values

        def load_state_dict(self, values, strict=True):
            assert strict
            self.values = values

    source_first = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    source_projection = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    source = {
        "low_dim_action_head.0.weight": source_first,
        "low_dim_action_head.0.bias": torch.tensor([100.0, 200.0]),
        "low_dim_action_head.1.weight": torch.ones(2),
        "input_proj_robot_state.weight": source_projection,
        "encoder_joint_proj.weight": source_projection + 20.0,
        "state_visual_residual_proprio_mask": torch.tensor(
            [1.0] * 4 + [0.0] * 6
        ),
        "state_visual_residual_scale": torch.ones(1),
        "action_head.weight": torch.ones(2, 2),
    }
    adapter = FakeAdapter()

    report = _load_state_visual_task_state_v2_warm_start(adapter, source)

    target_first = adapter.values["low_dim_action_head.0.weight"]
    torch.testing.assert_close(target_first[:, 0:4], source_first[:, 0:4])
    torch.testing.assert_close(target_first[:, 4:8], source_first[:, 6:10])
    torch.testing.assert_close(target_first[:, 12], source_first[:, 4])
    assert torch.count_nonzero(target_first[:, 8:12]).item() == 0
    torch.testing.assert_close(
        adapter.values["low_dim_action_head.0.bias"],
        source["low_dim_action_head.0.bias"] + source_first[:, 5],
    )
    torch.testing.assert_close(
        adapter.values["input_proj_robot_state.weight"][:, :4],
        source_projection[:, :4],
    )
    assert (
        torch.count_nonzero(
            adapter.values["input_proj_robot_state.weight"][:, 4:]
        ).item()
        == 0
    )
    assert report["remapped"] == [
        "encoder_joint_proj.weight",
        "input_proj_robot_state.weight",
        "low_dim_action_head.0.bias",
        "low_dim_action_head.0.weight",
    ]
