"""Structured observation container for real-excavator policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StructuredState:
    """Canonical observation container used by policy adapters."""

    qpos: np.ndarray
    qvel: np.ndarray
    images: dict[str, np.ndarray] = field(default_factory=dict)
    env_state: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_joints(self) -> int:
        return len(self.qpos)

    @property
    def has_images(self) -> bool:
        return bool(self.images)

    @property
    def camera_names(self) -> list[str]:
        return list(self.images.keys())

    def as_policy_input(self) -> dict[str, np.ndarray]:
        """Flatten state into the dict format expected by Policy.predict()."""

        data: dict[str, np.ndarray] = {
            "qpos": self.qpos.copy(),
            "qvel": self.qvel.copy(),
        }
        for camera_name, image in self.images.items():
            data[f"image_{camera_name}"] = image.copy()
        if self.env_state.size > 0:
            data["env_state"] = self.env_state.copy()
        return data


class StateConverter:
    """Convert raw backend observations into StructuredState."""

    def convert(self, obs: dict[str, Any]) -> StructuredState:
        return StructuredState(
            qpos=np.array(obs.get("qpos", []), dtype=np.float32),
            qvel=np.array(obs.get("qvel", []), dtype=np.float32),
            images={str(k): np.array(v) for k, v in obs.get("images", {}).items()},
            env_state=np.array(obs.get("env_state", []), dtype=np.float32),
            metadata={
                key: value
                for key, value in obs.items()
                if key not in {"qpos", "qvel", "images", "env_state"}
            },
        )

    @staticmethod
    def stack(states: list[StructuredState]) -> dict[str, np.ndarray]:
        """Stack a list of StructuredStates into batched numpy arrays."""

        return {
            "qpos": np.stack([state.qpos for state in states]),
            "qvel": np.stack([state.qvel for state in states]),
            "env_state": np.stack([state.env_state for state in states]),
        }
