"""Socket client for the external real-machine bridge protocol.

The client is inert until explicitly instantiated and used. It uses standard
library sockets only, so the package remains importable without ROS/CAN.
"""

from __future__ import annotations

import socket
import time
from typing import Any, Mapping

import numpy as np

from testbed.backends.real.bridge import RealBridgeClient
from testbed.backends.real.bridge_protocol import (
    BridgeProtocolError,
    control_result_from_payload,
    encode_frame,
    decode_frame,
    request_message,
    state_samples_from_payload,
)
from testbed.backends.real.contracts import as_real_action
from testbed.backends.real.control import ControlResult
from testbed.backends.real.state import RealStateSamples


class JsonTcpBridgeClient(RealBridgeClient):
    """Newline-delimited JSON/TCP implementation of RealBridgeClient."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int,
        timeout_s: float = 1.0,
        connect_on_init: bool = False,
    ) -> None:
        if int(port) <= 0:
            raise ValueError("bridge TCP port must be a positive integer")
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._sock: socket.socket | None = None
        self._file: Any | None = None
        if connect_on_init:
            self._connect()

    def reset(self, seed: int | None = None) -> None:
        self._request("reset", {"seed": seed})

    def send_action(
        self,
        action: np.ndarray,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> ControlResult:
        action4 = as_real_action(action, clip=True)
        response = self._request(
            "send_action",
            {
                "action": action4.tolist(),
                "state": _state_summary(state),
                "send_time_ns": time.time_ns(),
            },
        )
        return control_result_from_payload(response)

    def apply_status_toggle_mask(self, toggle_mask: int) -> bool:
        mask = int(toggle_mask) & 0x07FF
        if mask == 0:
            return True
        response = self._request("send_status", {"toggle_mask": mask})
        return bool(response.get("ack", False))

    def read_state(
        self,
        *,
        step_id: int,
        action_timestamp_ns: int | None = None,
    ) -> RealStateSamples:
        response = self._request(
            "read_state",
            {
                "step_id": int(step_id),
                "action_timestamp_ns": action_timestamp_ns,
                "request_time_ns": time.time_ns(),
            },
        )
        return state_samples_from_payload(response)

    def close(self) -> None:
        try:
            if self._sock is not None:
                try:
                    self._request("close", {})
                except Exception:
                    pass
        finally:
            if self._file is not None:
                self._file.close()
            if self._sock is not None:
                self._sock.close()
            self._file = None
            self._sock = None

    def _connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        self._sock = sock
        self._file = sock.makefile("rwb")

    def _request(self, request_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._connect()
        assert self._file is not None
        message = request_message(f"{request_type}.request", payload)
        self._file.write(encode_frame(message))
        self._file.flush()
        frame = self._file.readline()
        if not frame:
            raise BridgeProtocolError("bridge socket closed before response")
        response = decode_frame(frame)
        expected_type = f"{request_type}.response"
        if response.get("type") != expected_type:
            raise BridgeProtocolError(
                f"unexpected bridge response {response.get('type')!r}, expected {expected_type!r}"
            )
        if not bool(response.get("ok", True)):
            raise BridgeProtocolError(str(response.get("error", "bridge request failed")))
        payload_raw = response.get("payload", {})
        if not isinstance(payload_raw, Mapping):
            raise BridgeProtocolError("bridge response payload must be a mapping")
        return dict(payload_raw)


def _state_summary(state: Mapping[str, Any] | None) -> dict[str, Any]:
    if not state:
        return {}
    summary: dict[str, Any] = {}
    for key in ("step_id", "timestamp_ns", "sensor_timestamp_ns", "sync_timestamp_ns"):
        if key in state:
            summary[key] = int(state[key])
    for key in ("qpos", "qvel"):
        if key in state:
            summary[key] = np.asarray(state[key], dtype=np.float32).reshape(-1).tolist()
    return summary
