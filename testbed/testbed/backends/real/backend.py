"""Real excavator backend shell with safe mock/noop implementations.

The real-world branch uses this backend for data collection. The mock mode
lets the HDF5/QC/training path run without machine hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from testbed.backends.base import Backend
from testbed.backends.real.bridge import (
    BridgeLowLevelController,
    BridgeStateReader,
    InProcessMockBridgeClient,
    RealBridgeClient,
)
from testbed.backends.real.contracts import (
    REAL_ACTION_ORDER,
    REAL_QPOS_ORDER,
    REAL_QVEL_ORDER,
)
from testbed.backends.real.control import (
    ACTION_DIM,
    ControlResult,
    LowLevelController,
    MockLowLevelController,
    NoopLowLevelController,
)
from testbed.backends.real.state import MockStateReader, RealStateReader
from testbed.backends.real.sync import DEFAULT_SYNC_SLOP_NS, SynchronizedObservationBuilder


@dataclass
class RealExcavatorTimeStep:
    observation: dict[str, Any]
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class RealExcavatorBackend(Backend):
    """
    Backend facade for real excavator data collection.

    The current implementation intentionally supports mock/noop controller
    modes and a mock state reader. Future machine-specific integrations should
    provide LowLevelController and RealStateReader implementations while
    preserving the public observation/action contract.
    """

    def __init__(
        self,
        *,
        controller: LowLevelController | None = None,
        controller_mode: str = "mock",
        state_reader: RealStateReader | None = None,
        state_reader_mode: str = "mock",
        bridge_client: RealBridgeClient | None = None,
        sync_builder: SynchronizedObservationBuilder | None = None,
        sync_max_slop_ns: int = DEFAULT_SYNC_SLOP_NS,
        control_hz: float = 50.0,
        image_width: int = 160,
        image_height: int = 120,
        mock_velocity_scale_rad_s: float = 0.5,
    ) -> None:
        self._control_hz = float(control_hz)
        self._dt = 1.0 / self._control_hz if self._control_hz > 0 else 0.02
        self._step_id = 0
        self._last_obs: dict[str, Any] | None = None
        self._sync_builder = sync_builder or SynchronizedObservationBuilder(
            max_slop_ns=int(sync_max_slop_ns)
        )
        self._last_control_result = ControlResult(
            ack=True,
            fault_code="init",
            controller_timestamp_ns=time.time_ns(),
            commanded_action=np.zeros(ACTION_DIM, dtype=np.float32),
            raw_low_level_command=np.zeros(ACTION_DIM, dtype=np.float32),
        )
        self._bridge_client = bridge_client
        if self._bridge_client is None and (
            controller_mode == "bridge_mock" or state_reader_mode == "bridge_mock"
        ):
            self._bridge_client = InProcessMockBridgeClient(
                image_width=image_width,
                image_height=image_height,
                velocity_scale_rad_s=mock_velocity_scale_rad_s,
            )

        if controller is not None:
            self._controller = controller
        elif controller_mode == "mock":
            self._controller = MockLowLevelController()
        elif controller_mode == "noop":
            self._controller = NoopLowLevelController()
        elif controller_mode == "bridge_mock":
            assert self._bridge_client is not None
            self._controller = BridgeLowLevelController(self._bridge_client)
        elif controller_mode == "bridge_tcp":
            if self._bridge_client is None:
                raise ValueError("controller_mode='bridge_tcp' requires bridge_client.")
            self._controller = BridgeLowLevelController(self._bridge_client)
        else:
            raise ValueError(
                f"Unsupported real controller_mode={controller_mode!r}. "
                "Expected 'mock', 'noop', 'bridge_mock', or 'bridge_tcp'."
            )

        if state_reader is not None:
            self._state_reader = state_reader
        elif state_reader_mode == "mock":
            self._state_reader = MockStateReader(
                image_width=image_width,
                image_height=image_height,
                velocity_scale_rad_s=mock_velocity_scale_rad_s,
            )
        elif state_reader_mode == "bridge_mock":
            assert self._bridge_client is not None
            self._state_reader = BridgeStateReader(self._bridge_client)
        elif state_reader_mode == "bridge_tcp":
            if self._bridge_client is None:
                raise ValueError("state_reader_mode='bridge_tcp' requires bridge_client.")
            self._state_reader = BridgeStateReader(self._bridge_client)
        else:
            raise ValueError(
                f"Unsupported real state_reader_mode={state_reader_mode!r}. "
                "Expected 'mock', 'bridge_mock', 'bridge_tcp', or provide a state_reader."
            )

    def reset(self, seed: int | None = None) -> RealExcavatorTimeStep:
        """Backward-compatible alias for start_episode()."""
        return self.start_episode(seed=seed)

    def start_episode(self, seed: int | None = None) -> RealExcavatorTimeStep:
        """Mark a new episode boundary; this does not reset real hardware."""
        self._step_id = 0
        self._state_reader.reset(seed=seed)
        obs = self.read_state()
        return self._timestep_from_obs(obs)

    def apply_status_toggle_mask(self, toggle_mask: int) -> bool:
        return self._controller.apply_status_toggle_mask(toggle_mask)

    def step(self, action: np.ndarray) -> RealExcavatorTimeStep:
        state = self._last_obs or self.read_state()
        result = self._controller.send(action, state=state)
        self._last_control_result = result
        self._state_reader.apply_control_result(result, dt=self._dt)
        self._step_id += 1
        obs = self.read_state(action_timestamp_ns=result.controller_timestamp_ns)
        return self._timestep_from_obs(obs, result=result)

    def read_state(self, *, action_timestamp_ns: int | None = None) -> dict[str, Any]:
        samples = self._state_reader.read(
            step_id=int(self._step_id),
            action_timestamp_ns=action_timestamp_ns,
        )
        sync_result = self._sync_builder.build(
            joint_sample=samples.joint,
            image_samples=samples.images,
            step_id=int(self._step_id),
            action_timestamp_ns=action_timestamp_ns,
        )
        obs = sync_result.observation
        obs["control_result"] = _control_result_to_dict(self._last_control_result)
        self._last_obs = obs
        return obs

    def render(self, camera_id: str, height: int = 480, width: int = 640) -> np.ndarray:
        if self._last_obs is None:
            self.read_state()
        assert self._last_obs is not None
        image = self._last_obs.get("images", {}).get(camera_id)
        if image is None:
            raise KeyError(f"Camera {camera_id!r} not found in latest observation.")
        if image.shape[:2] == (height, width):
            return image.copy()
        import cv2

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    @property
    def dt(self) -> float:
        return self._dt

    def close(self) -> None:
        self._controller.close()
        self._state_reader.close()

    @property
    def controller(self) -> LowLevelController:
        return self._controller

    @property
    def state_reader(self) -> RealStateReader:
        return self._state_reader

    def _timestep_from_obs(
        self,
        obs: dict[str, Any],
        *,
        result: ControlResult | None = None,
    ) -> RealExcavatorTimeStep:
        result = result or self._last_control_result
        info = {
            "step_id": int(obs.get("step_id", 0)),
            "sensor_timestamp_ns": int(obs.get("sensor_timestamp_ns", 0)),
            "control_result": _control_result_to_dict(result),
            "task_success": False,
            "reward_phase": "real_recording",
            "task_step_successes": [],
            "task_step_failures": [],
            "task_metrics": {},
        }
        return RealExcavatorTimeStep(observation=obs, reward=0.0, done=False, info=info)


def _control_result_to_dict(result: ControlResult) -> dict[str, Any]:
    return {
        "ack": bool(result.ack),
        "fault_code": str(result.fault_code),
        "controller_timestamp_ns": int(result.controller_timestamp_ns),
        "commanded_action": np.asarray(result.commanded_action, dtype=np.float32).copy(),
        "raw_low_level_command": (
            None
            if result.raw_low_level_command is None
            else np.asarray(result.raw_low_level_command, dtype=np.float32).copy()
        ),
    }
