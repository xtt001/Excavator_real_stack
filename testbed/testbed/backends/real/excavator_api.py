"""Adapters for the lower excavator C++ control API boundary.

The lower repository currently exposes a C++ ``excavator_api`` library and a
TCP demo protocol.  This module keeps the Python testbed boundary small: map a
4-axis normalized testbed action into the 8-axis speed scalar command expected
by that lower layer.  It does not import ROS and does not require CAN hardware.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from testbed.backends.real.contracts import action4_to_speed_scalar8
from testbed.backends.real.control import ControlResult, LowLevelController


SERVO_MAGIC = 0x56524553
SERVO_PACKET_VERSION = 3
STATUS_PACKET_VERSION = 5
SERVO_PACKET_STRUCT = struct.Struct("<II9d")
STATUS_PACKET_STRUCT = struct.Struct("<IIdHH")


@dataclass(frozen=True)
class ServoPacketV3:
    """TCP servo packet compatible with the lower repo demo client."""

    speed_scalar8: np.ndarray
    motor_speed_normalized: float = 0.0

    def to_bytes(self) -> bytes:
        speed = np.asarray(self.speed_scalar8, dtype=np.float64).reshape(-1)
        if speed.shape != (8,):
            raise ValueError(f"speed_scalar8 must have shape (8,), got {speed.shape}")
        if not np.all(np.isfinite(speed)):
            raise ValueError("speed_scalar8 contains NaN or Inf")
        speed = np.clip(speed, -1.0, 1.0)
        motor = float(np.clip(self.motor_speed_normalized, -1.0, 1.0))
        return SERVO_PACKET_STRUCT.pack(
            SERVO_MAGIC,
            SERVO_PACKET_VERSION,
            *(float(v) for v in speed),
            motor,
        )


@dataclass(frozen=True)
class StatusPacketV5:
    """TCP status-toggle packet compatible with the lower repo demo client."""

    motor_speed_normalized: float = 0.0
    toggle_mask: int = 0

    def to_bytes(self) -> bytes:
        motor = float(np.clip(self.motor_speed_normalized, -1.0, 1.0))
        return STATUS_PACKET_STRUCT.pack(
            SERVO_MAGIC,
            STATUS_PACKET_VERSION,
            motor,
            int(self.toggle_mask) & 0x07FF,
            0,
        )


class ExcavatorApiPacketAdapter:
    """Pure adapter from testbed action to lower-library packet bytes."""

    def __init__(
        self,
        *,
        axis_signs: Sequence[float] | None = None,
        motor_speed_normalized: float = 0.0,
    ) -> None:
        self.axis_signs = axis_signs
        self.motor_speed_normalized = float(motor_speed_normalized)

    def speed_scalar8_from_action(self, action: np.ndarray) -> np.ndarray:
        return action4_to_speed_scalar8(action, axis_signs=self.axis_signs, clip=True)

    def servo_packet(self, action: np.ndarray) -> ServoPacketV3:
        return ServoPacketV3(
            speed_scalar8=self.speed_scalar8_from_action(action),
            motor_speed_normalized=self.motor_speed_normalized,
        )

    def servo_bytes(self, action: np.ndarray) -> bytes:
        return self.servo_packet(action).to_bytes()


class SocketExcavatorApiController(LowLevelController):
    """
    LowLevelController that writes lower-repo TCP demo packets to a socket.

    The caller owns process supervision and hardware safety.  In this branch it
    is mainly a narrow adapter boundary; mock/noop remain the default recorder
    modes for machines without ROS/CAN/hardware.
    """

    def __init__(
        self,
        sock: socket.socket,
        *,
        axis_signs: Sequence[float] | None = None,
        motor_speed_normalized: float = 0.0,
        send_status_each_step: bool = False,
    ) -> None:
        self._sock = sock
        self._adapter = ExcavatorApiPacketAdapter(
            axis_signs=axis_signs,
            motor_speed_normalized=motor_speed_normalized,
        )
        self._send_status_each_step = bool(send_status_each_step)
        self.send_count = 0

    def send(self, action: np.ndarray, state: dict[str, Any] | None = None) -> ControlResult:
        speed_scalar8 = self._adapter.speed_scalar8_from_action(action)
        raw = self._adapter.servo_packet(action).to_bytes()
        try:
            self._sock.sendall(raw)
            if self._send_status_each_step:
                self._sock.sendall(
                    StatusPacketV5(
                        motor_speed_normalized=self._adapter.motor_speed_normalized,
                    ).to_bytes()
                )
        except OSError as exc:
            return ControlResult(
                ack=False,
                fault_code=f"socket_error:{exc.__class__.__name__}",
                controller_timestamp_ns=time.time_ns(),
                commanded_action=speed_scalar8[:4].astype(np.float32, copy=True),
                raw_low_level_command=speed_scalar8.astype(np.float32, copy=True),
            )

        self.send_count += 1
        return ControlResult(
            ack=True,
            fault_code="",
            controller_timestamp_ns=time.time_ns(),
            commanded_action=speed_scalar8[:4].astype(np.float32, copy=True),
            raw_low_level_command=speed_scalar8.astype(np.float32, copy=True),
        )

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
