from __future__ import annotations

import pytest
import torch

from testbed.actions.policy import _act_policy_config_from_resolved
from testbed.policies.act.camera_role_encoding import (
    CameraRoleEncoding,
    resolve_camera_role_encoding_config,
)

CAMERAS = ["video4", "video5", "video6", "video7"]
CONFIG = {
    "enabled": True,
    "roles": {
        "video4": "eye",
        "video5": "eye",
        "video6": "stick",
        "video7": "stick",
    },
}


def test_camera_role_config_requires_an_explicit_role_for_every_camera() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        resolve_camera_role_encoding_config(
            {"enabled": True, "roles": {"video4": "eye"}},
            camera_names=CAMERAS,
        )


def test_role_encoding_starts_as_exact_zero_rollback() -> None:
    encoding = CameraRoleEncoding(
        hidden_dim=8,
        camera_names=CAMERAS,
        config=CONFIG,
    )

    for camera_index in range(len(CAMERAS)):
        torch.testing.assert_close(
            encoding(camera_index),
            torch.zeros((1, 8, 1, 1)),
        )


def test_camera_identity_is_distinct_while_role_identity_is_shared() -> None:
    encoding = CameraRoleEncoding(
        hidden_dim=2,
        camera_names=CAMERAS,
        config=CONFIG,
    )
    with torch.no_grad():
        encoding.camera_embedding.weight.copy_(
            torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        )
        encoding.role_embedding.weight.copy_(
            torch.tensor([[0.0, 10.0], [0.0, 20.0]])
        )

    torch.testing.assert_close(encoding(0).flatten(), torch.tensor([1.0, 10.0]))
    torch.testing.assert_close(encoding(1).flatten(), torch.tensor([2.0, 10.0]))
    torch.testing.assert_close(encoding(2).flatten(), torch.tensor([3.0, 20.0]))
    torch.testing.assert_close(encoding(3).flatten(), torch.tensor([4.0, 20.0]))


def test_resolved_bundle_reconstructs_camera_role_encoding() -> None:
    resolved = {
        "task": {
            "camera_names": CAMERAS,
            "equipment_model": "real_excavator",
        },
        "policy": {
            "low_dim_keys": ["qpos"],
            "act_params": {
                "chunk_size": 20,
                "state_dim": 4,
                "camera_role_encoding": CONFIG,
            },
        },
        "train": {},
    }

    policy_config = _act_policy_config_from_resolved(resolved)

    assert policy_config["camera_role_encoding"] == CONFIG
