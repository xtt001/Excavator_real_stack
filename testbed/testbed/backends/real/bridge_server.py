"""Local JSON/TCP bridge mock server.

This server uses the same protocol as JsonTcpBridgeClient and the same
InProcessMockBridgeClient state model as bridge_mock mode. It is for local
bring-up only; it does not touch ROS, CAN, or hardware.
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Any, Mapping

import numpy as np

from testbed.backends.real.bridge import InProcessMockBridgeClient, RealBridgeClient
from testbed.backends.real.bridge_protocol import (
    BridgeProtocolError,
    control_result_to_payload,
    decode_frame,
    encode_frame,
    response_message,
    state_samples_to_payload,
)

log = logging.getLogger(__name__)


class JsonTcpBridgeMockServer:
    """Small loopback server implementing the external bridge JSON/TCP protocol."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        dt: float = 0.02,
        image_width: int = 160,
        image_height: int = 120,
        velocity_scale_rad_s: float = 0.5,
        bridge_client: RealBridgeClient | None = None,
        one_shot: bool = False,
    ) -> None:
        if int(port) < 0:
            raise ValueError("port must be non-negative")
        self.host = str(host)
        self.port = int(port)
        self.bound_port: int | None = None
        self.dt = float(dt)
        self.one_shot = bool(one_shot)
        self.client = bridge_client or InProcessMockBridgeClient(
            image_width=image_width,
            image_height=image_height,
            velocity_scale_rad_s=velocity_scale_rad_s,
        )
        self._stop = threading.Event()
        self._ready = threading.Event()

    def serve_forever(self) -> None:
        """Serve bridge protocol requests until shutdown or one-shot completion."""

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            server.settimeout(0.2)
            self.bound_port = int(server.getsockname()[1])
            self._ready.set()
            log.info("Bridge mock server listening on %s:%d", self.host, self.bound_port)

            while not self._stop.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                log.info("Bridge mock client connected from %s", addr)
                with conn:
                    self._handle_connection(conn)
                if self.one_shot:
                    break
        self.client.close()
        log.info("Bridge mock server stopped.")

    def shutdown(self) -> None:
        self._stop.set()

    def wait_until_ready(self, timeout_s: float = 2.0) -> bool:
        return self._ready.wait(timeout=float(timeout_s))

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn.makefile("rwb") as stream:
            while not self._stop.is_set():
                line = stream.readline()
                if not line:
                    break
                response = self._handle_frame(line)
                stream.write(encode_frame(response))
                stream.flush()
                if response.get("type") in {"close.response", "shutdown.response"}:
                    break

    def _handle_frame(self, frame: bytes) -> dict[str, Any]:
        try:
            message = decode_frame(frame)
            message_type = str(message.get("type", ""))
            payload = message.get("payload", {})
            if not isinstance(payload, Mapping):
                raise BridgeProtocolError("request payload must be a mapping")

            if message_type == "reset.request":
                self.client.reset(seed=_optional_int(payload.get("seed")))
                return response_message("reset.response", {"reset": True})

            if message_type == "send_action.request":
                action = np.asarray(payload.get("action", []), dtype=np.float32)
                state_raw = payload.get("state", {})
                state = state_raw if isinstance(state_raw, Mapping) else {}
                result = self.client.send_action(action, state=state)
                self.client.apply_control_result(result, dt=self.dt)
                return response_message(
                    "send_action.response",
                    control_result_to_payload(result),
                )

            if message_type == "send_status.request":
                toggle_mask = int(payload.get("toggle_mask", 0)) & 0x07FF
                ack = self.client.apply_status_toggle_mask(toggle_mask)
                return response_message(
                    "send_status.response",
                    {"ack": bool(ack), "toggle_mask": toggle_mask},
                )

            if message_type == "read_state.request":
                samples = self.client.read_state(
                    step_id=int(payload.get("step_id", 0)),
                    action_timestamp_ns=_optional_int(payload.get("action_timestamp_ns")),
                )
                return response_message(
                    "read_state.response",
                    state_samples_to_payload(samples),
                )

            if message_type == "close.request":
                return response_message("close.response", {"closed": True})

            if message_type == "shutdown.request":
                self.shutdown()
                return response_message("shutdown.response", {"shutdown": True})

            return response_message(
                message_type.replace(".request", ".response") or "unknown.response",
                {},
                ok=False,
                error=f"unsupported request type {message_type!r}",
            )
        except Exception as exc:
            log.exception("Bridge mock request failed.")
            return response_message(
                "error.response",
                {},
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
