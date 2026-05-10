"""Real excavator backend and low-level control interfaces."""

from testbed.backends.real.backend import RealExcavatorBackend, RealExcavatorTimeStep
from testbed.backends.real.bridge import (
    BridgeLowLevelController,
    BridgeStateReader,
    InProcessMockBridgeClient,
    RealBridgeClient,
)
from testbed.backends.real.bridge_protocol import (
    BRIDGE_PROTOCOL_VERSION,
    BridgeProtocolError,
    control_result_from_payload,
    control_result_to_payload,
    decode_frame,
    encode_frame,
    state_samples_from_payload,
    state_samples_to_payload,
)
from testbed.backends.real.bridge_server import JsonTcpBridgeMockServer
from testbed.backends.real.bridge_socket import JsonTcpBridgeClient
from testbed.backends.real.contracts import (
    EXCAVATOR_API_AXIS_ORDER,
    REAL_ACTION_ORDER,
    REAL_QPOS_ORDER,
    REAL_QVEL_ORDER,
    action4_to_speed_scalar8,
)
from testbed.backends.real.control import (
    ControlResult,
    LowLevelController,
    MockLowLevelController,
    NoopLowLevelController,
)
from testbed.backends.real.excavator_api import (
    ExcavatorApiPacketAdapter,
    ServoPacketV3,
    SocketExcavatorApiController,
    StatusPacketV5,
)
from testbed.backends.real.ros_can import (
    RosCanBridgeClient,
    RosCanLowLevelController,
    RosCanStateReader,
    RosCanUnavailableError,
)
from testbed.backends.real.state import MockStateReader, RealStateReader, RealStateSamples
from testbed.backends.real.sync import (
    DEFAULT_SYNC_SLOP_NS,
    SyncResult,
    SynchronizedObservationBuilder,
    TimestampedBuffer,
    TimestampedSample,
)

__all__ = [
    "DEFAULT_SYNC_SLOP_NS",
    "BRIDGE_PROTOCOL_VERSION",
    "EXCAVATOR_API_AXIS_ORDER",
    "REAL_ACTION_ORDER",
    "REAL_QPOS_ORDER",
    "REAL_QVEL_ORDER",
    "ControlResult",
    "BridgeLowLevelController",
    "BridgeProtocolError",
    "BridgeStateReader",
    "ExcavatorApiPacketAdapter",
    "InProcessMockBridgeClient",
    "JsonTcpBridgeClient",
    "JsonTcpBridgeMockServer",
    "LowLevelController",
    "MockLowLevelController",
    "MockStateReader",
    "NoopLowLevelController",
    "RealBackend",
    "RealBridgeClient",
    "RealExcavatorBackend",
    "RealExcavatorTimeStep",
    "RealStateReader",
    "RealStateSamples",
    "RosCanBridgeClient",
    "RosCanLowLevelController",
    "RosCanStateReader",
    "RosCanUnavailableError",
    "ServoPacketV3",
    "SocketExcavatorApiController",
    "SyncResult",
    "StatusPacketV5",
    "SynchronizedObservationBuilder",
    "TimestampedBuffer",
    "TimestampedSample",
    "action4_to_speed_scalar8",
    "control_result_from_payload",
    "control_result_to_payload",
    "decode_frame",
    "encode_frame",
    "state_samples_from_payload",
    "state_samples_to_payload",
]

RealBackend = RealExcavatorBackend
