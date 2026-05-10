"""Optional ROS/CAN controller adapter shell.

This file is deliberately importable on laptops without ROS.  The real ROS
node wiring belongs behind this adapter and should be activated only on a
machine that has the ROS workspace, CAN devices, and hardware safety process.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from testbed.backends.real.bridge import RealBridgeClient
from testbed.backends.real.contracts import action4_to_speed_scalar8
from testbed.backends.real.control import ControlResult, LowLevelController
from testbed.backends.real.state import RealStateReader, RealStateSamples


class RosCanUnavailableError(RuntimeError):
    """Raised when the ROS/CAN adapter is used without ROS dependencies."""


def import_ros_client_library() -> Any:
    """
    Import ROS lazily so the rest of testbed stays usable without ROS.

    ROS 2 uses ``rclpy`` and ROS 1 uses ``rospy``.  The concrete integration can
    choose one later; this helper only proves optional dependency handling.
    """

    try:
        import rclpy  # type: ignore

        return rclpy
    except ImportError:
        pass

    try:
        import rospy  # type: ignore

        return rospy
    except ImportError as exc:
        raise RosCanUnavailableError(
            "ROS Python client is not installed. Use mock/noop adapters on "
            "development machines without ROS/CAN hardware."
        ) from exc


class RosCanLowLevelController(LowLevelController):
    """
    Placeholder LowLevelController for a future ROS/CAN bridge.

    The constructor performs the lazy ROS import only when this adapter is
    explicitly instantiated.  Topic names, message types, and node lifecycle
    should be filled in during hardware-side integration.
    """

    def __init__(self, *, node_name: str = "excavator_testbed_bridge") -> None:
        self._ros = import_ros_client_library()
        self.node_name = str(node_name)
        raise RosCanUnavailableError(
            "RosCanLowLevelController is a stub boundary. Implement ROS topic "
            "publish/subscribe wiring on the machine that owns the ROS/CAN stack."
        )

    def send(self, action: np.ndarray, state: dict[str, Any] | None = None) -> ControlResult:
        speed_scalar8 = action4_to_speed_scalar8(action, clip=True)
        return ControlResult(
            ack=False,
            fault_code="ros_can_stub",
            controller_timestamp_ns=time.time_ns(),
            commanded_action=speed_scalar8[:4].astype(np.float32, copy=True),
            raw_low_level_command=speed_scalar8.astype(np.float32, copy=True),
        )


class RosCanBridgeClient(RealBridgeClient):
    """
    Placeholder shared bridge client for the future ROS/CAN hardware process.

    Production code should implement this shape when command acknowledgements,
    joint/status samples, and camera timestamps are owned by one ROS-side
    process.  Keeping this as a RealBridgeClient avoids making the whole
    testbed a ROS package.
    """

    def __init__(
        self,
        *,
        node_name: str = "excavator_testbed_real_bridge",
        command_topic: str = "/excavator/testbed/command",
        state_topic: str = "/excavator/state",
        camera_topic: str = "/excavator/fpv/image",
    ) -> None:
        self._ros = import_ros_client_library()
        self.node_name = str(node_name)
        self.command_topic = str(command_topic)
        self.state_topic = str(state_topic)
        self.camera_topic = str(camera_topic)
        raise RosCanUnavailableError(
            "RosCanBridgeClient is a stub boundary. Implement ROS command, "
            "state, and camera wiring in the real-machine bridge process."
        )

    def send_action(
        self,
        action: np.ndarray,
        *,
        state: dict[str, Any] | None = None,
    ) -> ControlResult:
        speed_scalar8 = action4_to_speed_scalar8(action, clip=True)
        return ControlResult(
            ack=False,
            fault_code="ros_can_bridge_stub",
            controller_timestamp_ns=time.time_ns(),
            commanded_action=speed_scalar8[:4].astype(np.float32, copy=True),
            raw_low_level_command=speed_scalar8.astype(np.float32, copy=True),
        )

    def read_state(
        self,
        *,
        step_id: int,
        action_timestamp_ns: int | None = None,
    ) -> RealStateSamples:
        raise RosCanUnavailableError("RosCanBridgeClient is not implemented yet.")


class RosCanStateReader(RealStateReader):
    """
    Placeholder RealStateReader for a future ROS/CAN state bridge.

    It should subscribe to timestamped joint/status/camera streams and return
    RealStateSamples for the backend sync builder.  It is intentionally only a
    boundary in this no-hardware development branch.
    """

    def __init__(self, *, node_name: str = "excavator_testbed_state_bridge") -> None:
        self._ros = import_ros_client_library()
        self.node_name = str(node_name)
        raise RosCanUnavailableError(
            "RosCanStateReader is a stub boundary. Implement ROS subscriptions "
            "on the machine that owns the ROS/CAN stack."
        )

    def read(
        self,
        *,
        step_id: int,
        action_timestamp_ns: int | None = None,
    ) -> RealStateSamples:
        raise RosCanUnavailableError("RosCanStateReader is not implemented yet.")
