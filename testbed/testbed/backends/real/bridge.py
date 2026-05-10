"""Shared bridge-client adapters for real-machine integrations.

The bridge shape models the future hardware process without requiring ROS, CAN,
or a machine in the development environment. A bridge client can be shared by a
LowLevelController and a RealStateReader so commands and state come from the
same boundary.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np

from testbed.backends.real.contracts import (
    REAL_ACTION_DIM,
    action4_to_speed_scalar8,
    as_real_action,
)
from testbed.backends.real.control import ControlResult, LowLevelController
from testbed.backends.real.state import RealStateReader, RealStateSamples
from testbed.backends.real.sync import TimestampedSample


class RealBridgeClient(ABC):
    """Abstract client for a process that owns machine commands and state."""

    def reset(self, seed: int | None = None) -> None:
        """Mark an episode boundary for bridge-owned buffers."""

    @abstractmethod
    def send_action(
        self,
        action: np.ndarray,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> ControlResult:
        """Send one normalized four-axis command through the bridge."""

    def apply_control_result(self, result: ControlResult, *, dt: float) -> None:
        """Let local/mock bridge clients update state after an ack."""

    @abstractmethod
    def read_state(
        self,
        *,
        step_id: int,
        action_timestamp_ns: int | None = None,
    ) -> RealStateSamples:
        """Read timestamped joint/status/camera samples from the bridge."""

    def close(self) -> None:
        """Release bridge resources."""


class InProcessMockBridgeClient(RealBridgeClient):
    """Local bridge client with deterministic command/state behavior."""

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
        self._last_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._closed = False
        self.send_count = 0
        self.read_count = 0

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action.copy()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(int(seed))
        self._qpos.fill(0.0)
        self._qvel.fill(0.0)
        self._last_action.fill(0.0)
        self.send_count = 0
        self.read_count = 0

    def send_action(
        self,
        action: np.ndarray,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> ControlResult:
        self._ensure_open()
        commanded = as_real_action(action, clip=True)
        raw_low_level = action4_to_speed_scalar8(commanded, clip=True)
        self._last_action = commanded.copy()
        self.send_count += 1
        return ControlResult(
            ack=True,
            fault_code="",
            controller_timestamp_ns=time.time_ns(),
            commanded_action=commanded,
            raw_low_level_command=raw_low_level,
        )

    def apply_control_result(self, result: ControlResult, *, dt: float) -> None:
        self._ensure_open()
        commanded = as_real_action(result.commanded_action, clip=True)
        self._qvel = (commanded * self._velocity_scale).astype(np.float32)
        self._qpos = (self._qpos + self._qvel * float(dt)).astype(np.float32)

    def read_state(
        self,
        *,
        step_id: int,
        action_timestamp_ns: int | None = None,
    ) -> RealStateSamples:
        self._ensure_open()
        self.read_count += 1
        receive_ns = time.time_ns()
        joint_ts_ns = max(1, receive_ns - self._joint_latency_ns)
        image_ts_ns = max(1, receive_ns - self._image_latency_ns)
        joint = TimestampedSample(
            timestamp_ns=joint_ts_ns,
            payload={
                "qpos": self._qpos.copy(),
                "qvel": self._qvel.copy(),
                "status": np.zeros(16, dtype=np.int32),
                "env_state": np.concatenate([self._qpos, self._qvel]).astype(np.float32),
            },
            source="bridge_joint_state",
            receive_time_ns=receive_ns,
        )
        images = {
            name: TimestampedSample(
                timestamp_ns=image_ts_ns,
                payload=self._mock_image(step_id=step_id, camera_name=name),
                source=f"bridge_camera:{name}",
                receive_time_ns=receive_ns,
            )
            for name in self._camera_names
        }
        return RealStateSamples(joint=joint, images=images)

    def close(self) -> None:
        self._closed = True

    def _mock_image(self, *, step_id: int, camera_name: str) -> np.ndarray:
        h, w = self._image_height, self._image_width
        image = np.zeros((h, w, 3), dtype=np.uint8)
        name_offset = sum(ord(ch) for ch in camera_name) % 255
        image[..., 0] = (int(step_id) * 5 + name_offset) % 255
        image[..., 1] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        image[..., 2] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
        return image

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("bridge client is closed")


class BridgeLowLevelController(LowLevelController):
    """LowLevelController adapter backed by a shared bridge client."""

    def __init__(self, client: RealBridgeClient) -> None:
        self.client = client

    def send(self, action: np.ndarray, state: dict[str, Any] | None = None) -> ControlResult:
        return self.client.send_action(action, state=state)

    def close(self) -> None:
        self.client.close()


class BridgeStateReader(RealStateReader):
    """RealStateReader adapter backed by a shared bridge client."""

    def __init__(self, client: RealBridgeClient) -> None:
        self.client = client

    def reset(self, seed: int | None = None) -> None:
        self.client.reset(seed=seed)

    def apply_control_result(self, result: ControlResult, *, dt: float) -> None:
        self.client.apply_control_result(result, dt=dt)

    def read(
        self,
        *,
        step_id: int,
        action_timestamp_ns: int | None = None,
    ) -> RealStateSamples:
        return self.client.read_state(
            step_id=step_id,
            action_timestamp_ns=action_timestamp_ns,
        )

    def close(self) -> None:
        self.client.close()
