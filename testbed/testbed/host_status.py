"""Latest-only localhost telemetry for the read-only host dashboard."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Mapping
from typing import Any

HOST_STATUS_PROTOCOL_VERSION = 1
HOST_STATUS_MESSAGE_TYPE = "excavator.host_status"
DEFAULT_HOST_STATUS_HOST = "127.0.0.1"
DEFAULT_HOST_STATUS_PORT = 8781
MAX_HOST_STATUS_DATAGRAM_BYTES = 60_000


class HostStatusProtocolError(ValueError):
    pass


def encode_host_status(payload: Mapping[str, Any]) -> bytes:
    frame = json.dumps(
        {
            "version": HOST_STATUS_PROTOCOL_VERSION,
            "type": HOST_STATUS_MESSAGE_TYPE,
            "payload": dict(payload),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(frame) > MAX_HOST_STATUS_DATAGRAM_BYTES:
        raise HostStatusProtocolError(
            f"host status datagram too large: {len(frame)} bytes"
        )
    return frame


def decode_host_status(frame: bytes | str) -> dict[str, Any]:
    text = frame.decode("utf-8") if isinstance(frame, bytes) else str(frame)
    try:
        message = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HostStatusProtocolError(f"invalid host status JSON: {exc}") from exc
    if not isinstance(message, Mapping):
        raise HostStatusProtocolError("host status must be a JSON object")
    if int(message.get("version", -1)) != HOST_STATUS_PROTOCOL_VERSION:
        raise HostStatusProtocolError(
            f"unsupported host status version {message.get('version')!r}"
        )
    if message.get("type") != HOST_STATUS_MESSAGE_TYPE:
        raise HostStatusProtocolError(
            f"unexpected host status type {message.get('type')!r}"
        )
    payload = message.get("payload")
    if not isinstance(payload, Mapping):
        raise HostStatusProtocolError("host status payload must be a JSON object")
    return dict(payload)


def build_host_status_snapshot(
    *,
    seq: int,
    target: str,
    configured_hz: float,
    input_device: str,
    source_id: str,
    action: Any,
    action_latency_ms: Any,
    extras: Mapping[str, Any] | None,
    event_flags: Mapping[str, Any] | None,
    receiver_status: Any | None,
) -> dict[str, Any]:
    extras_dict = dict(extras or {})
    receiver_payload = (
        dict(getattr(receiver_status, "payload", {}) or {})
        if receiver_status is not None
        else {}
    )
    receiver_receive_time_ns = int(
        getattr(receiver_status, "receive_time_ns", 0) or 0
    )
    try:
        action_values = [float(item) for item in action]
    except TypeError:
        action_values = []
    status11 = extras_dict.get("status11")
    try:
        status_values = [int(item) for item in status11]
    except TypeError:
        status_values = []
    try:
        latency = None if action_latency_ms is None else float(action_latency_ms)
    except (TypeError, ValueError):
        latency = None
    return {
        "timestamp_ns": int(time.time_ns()),
        "sender": {
            "running": 1,
            "seq": int(seq),
            "target": str(target),
            "configured_hz": float(configured_hz),
            "input_device": str(input_device),
            "source_id": str(source_id),
            "action": action_values,
            "action_latency_ms": latency,
            "status11": status_values,
            "toggle_mask": int(extras_dict.get("toggle_mask", 0) or 0),
            "events": {
                str(key): value for key, value in dict(event_flags or {}).items()
            },
        },
        "receiver_available": int(receiver_status is not None),
        "receiver_receive_time_ns": receiver_receive_time_ns,
        "receiver": receiver_payload,
    }


class LocalHostStatusPublisher:
    """Non-blocking UDP publisher; dashboard absence never affects control."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST_STATUS_HOST,
        port: int = DEFAULT_HOST_STATUS_PORT,
        max_hz: float = 10.0,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.address = (str(host), int(port))
        self.min_period_s = 0.0 if max_hz <= 0 else 1.0 / float(max_hz)
        self._last_publish_s = 0.0
        self._socket: socket.socket | None = None
        if self.enabled:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            self._socket = sock

    def publish(self, payload: Mapping[str, Any], *, force: bool = False) -> bool:
        if self._socket is None:
            return False
        now_s = time.monotonic()
        if (
            not force
            and self.min_period_s > 0.0
            and now_s - self._last_publish_s < self.min_period_s
        ):
            return False
        try:
            self._socket.sendto(encode_host_status(payload), self.address)
        except (BlockingIOError, OSError, HostStatusProtocolError):
            return False
        self._last_publish_s = now_s
        return True

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
