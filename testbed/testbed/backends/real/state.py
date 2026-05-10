"""State-reader interfaces for real-machine backends.

State readers provide timestamped joint and camera samples. The backend then
uses the sync builder to turn those samples into the standard testbed
observation dict.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from testbed.backends.real.contracts import REAL_ACTION_DIM, as_real_action
from testbed.backends.real.control import ControlResult
from testbed.backends.real.sync import TimestampedSample


@dataclass(frozen=True)
class RealStateSamples:
    """Raw timestamped samples read from the machine-side state streams."""

    joint: TimestampedSample
    images: Mapping[str, TimestampedSample] = field(default_factory=dict)


class RealStateReader(ABC):
    """Abstract source of real-machine joint/status/camera samples."""

    def reset(self, seed: int | None = None) -> None:
        """Mark an episode boundary for reader-owned buffers."""

    def apply_control_result(self, result: ControlResult, *, dt: float) -> None:
        """Let mock readers update local state after a command acknowledgement."""

    @abstractmethod
    def read(
        self,
        *,
        step_id: int,
        action_timestamp_ns: int | None = None,
    ) -> RealStateSamples:
        """Return the latest timestamped joint sample and camera samples."""

    def close(self) -> None:
        """Release reader resources."""


class MockStateReader(RealStateReader):
    """Synthetic state reader for local recorder/QC/policy development."""

    def __init__(
        self,
        *,
        image_width: int = 160,
        image_height: int = 120,
        velocity_scale_rad_s: float = 0.5,
        camera_names: Sequence[str] = ("fpv",),
        joint_latency_ns: int = 0,
        image_latency_ns: int = 0,
    ) -> None:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")
        self._image_width = int(image_width)
        self._image_height = int(image_height)
        self._velocity_scale = float(velocity_scale_rad_s)
        self._camera_names = tuple(str(name) for name in camera_names)
        if not self._camera_names:
            raise ValueError("at least one camera name is required")
        self._joint_latency_ns = int(joint_latency_ns)
        self._image_latency_ns = int(image_latency_ns)
        self._qpos = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._qvel = np.zeros(REAL_ACTION_DIM, dtype=np.float32)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(int(seed))
        self._qpos.fill(0.0)
        self._qvel.fill(0.0)

    def apply_control_result(self, result: ControlResult, *, dt: float) -> None:
        commanded = as_real_action(result.commanded_action, clip=True)
        self._qvel = (commanded * self._velocity_scale).astype(np.float32)
        self._qpos = (self._qpos + self._qvel * float(dt)).astype(np.float32)

    def read(
        self,
        *,
        step_id: int,
        action_timestamp_ns: int | None = None,
    ) -> RealStateSamples:
        receive_ns = time.time_ns()
        joint_ts_ns = max(1, receive_ns - self._joint_latency_ns)
        image_ts_ns = max(1, receive_ns - self._image_latency_ns)
        joint = TimestampedSample(
            timestamp_ns=joint_ts_ns,
            payload={
                "qpos": self._qpos.copy(),
                "qvel": self._qvel.copy(),
                "status": np.zeros(16, dtype=np.int32),
            },
            source="joint_state",
            receive_time_ns=receive_ns,
        )
        images = {
            name: TimestampedSample(
                timestamp_ns=image_ts_ns,
                payload=self._mock_image(step_id=step_id, camera_name=name),
                source=name,
                receive_time_ns=receive_ns,
            )
            for name in self._camera_names
        }
        return RealStateSamples(joint=joint, images=images)

    def _mock_image(self, *, step_id: int, camera_name: str) -> np.ndarray:
        h, w = self._image_height, self._image_width
        image = np.zeros((h, w, 3), dtype=np.uint8)
        name_offset = sum(ord(ch) for ch in camera_name) % 255
        image[..., 0] = (int(step_id) * 3 + name_offset) % 255
        image[..., 1] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        image[..., 2] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        return image
