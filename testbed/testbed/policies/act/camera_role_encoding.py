"""Learned camera and physical-role identity for multi-view ACT inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


def resolve_camera_role_encoding_config(
    config: Mapping[str, Any] | None,
    *,
    camera_names: Sequence[str],
) -> dict[str, Any]:
    """Validate and normalize the opt-in camera-role encoding contract."""

    raw = dict(config or {})
    enabled = bool(raw.get("enabled", False))
    names = [str(name) for name in camera_names]
    if len(set(names)) != len(names):
        raise ValueError("camera_names must not contain duplicates")
    if not enabled:
        return {"enabled": False, "roles": {}, "role_names": []}

    roles_raw = raw.get("roles")
    if not isinstance(roles_raw, Mapping):
        raise ValueError(
            "camera_role_encoding.roles must map every camera name to a physical role"
        )
    roles = {str(name): str(role) for name, role in roles_raw.items()}
    missing = [name for name in names if name not in roles]
    extra = [name for name in roles if name not in names]
    if missing or extra:
        raise ValueError(
            "camera_role_encoding.roles must exactly match camera_names: "
            f"missing={missing}, extra={extra}"
        )
    if any(not roles[name] for name in names):
        raise ValueError("camera_role_encoding role names must not be empty")

    role_names = list(dict.fromkeys(roles[name] for name in names))
    return {
        "enabled": True,
        "roles": {name: roles[name] for name in names},
        "role_names": role_names,
    }


class CameraRoleEncoding(nn.Module):
    """Add learned per-camera and shared physical-role identity to spatial tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        camera_names: Sequence[str],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        resolved = resolve_camera_role_encoding_config(
            config,
            camera_names=camera_names,
        )
        if not resolved["enabled"]:
            raise ValueError("CameraRoleEncoding requires enabled configuration")

        self.camera_names = tuple(str(name) for name in camera_names)
        self.role_names = tuple(resolved["role_names"])
        role_index = {name: index for index, name in enumerate(self.role_names)}
        self.register_buffer(
            "camera_role_indices",
            torch.tensor(
                [role_index[resolved["roles"][name]] for name in self.camera_names],
                dtype=torch.long,
            ),
        )
        self.camera_embedding = nn.Embedding(len(self.camera_names), int(hidden_dim))
        self.role_embedding = nn.Embedding(len(self.role_names), int(hidden_dim))
        nn.init.zeros_(self.camera_embedding.weight)
        nn.init.zeros_(self.role_embedding.weight)

    def forward(self, camera_index: int) -> torch.Tensor:
        if not 0 <= int(camera_index) < len(self.camera_names):
            raise IndexError(f"camera index out of range: {camera_index}")
        index = torch.tensor(
            int(camera_index),
            dtype=torch.long,
            device=self.camera_embedding.weight.device,
        )
        role_index = self.camera_role_indices[index]
        identity = self.camera_embedding(index) + self.role_embedding(role_index)
        return identity.view(1, -1, 1, 1)
