"""Dependency-light client for the real-transition event server."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Mapping
from typing import Any


TRANSITION_CONTROL_PROTOCOL_VERSION = 1
TRANSITION_CONTROL_COMMAND = "real_transition.command"
TRANSITION_CONTROL_RESPONSE = "real_transition.response"
DEFAULT_TRANSITION_CONTROL_PORT = 8771
_MAX_FRAME_BYTES = 64 * 1024


class TransitionControlClientError(ValueError):
    """Raised when the v2 event server rejects or malforms a response."""


def send_transition_command(
    *,
    host: str,
    port: int = DEFAULT_TRANSITION_CONTROL_PORT,
    command: str,
    payload: Mapping[str, Any] | None = None,
    timeout_s: float = 3.0,
) -> dict[str, Any]:
    with socket.create_connection(
        (str(host), int(port)), timeout=float(timeout_s)
    ) as sock:
        sock.settimeout(float(timeout_s))
        sock.sendall(_encode_command(command, payload))
        return _decode_response(_receive_frame(sock))


def _encode_command(
    command: str, payload: Mapping[str, Any] | None = None
) -> bytes:
    message = {
        "version": TRANSITION_CONTROL_PROTOCOL_VERSION,
        "type": TRANSITION_CONTROL_COMMAND,
        "command": str(command),
        "payload": dict(payload or {}),
        "client_time_ns": time.time_ns(),
    }
    return json.dumps(
        message, separators=(",", ":"), sort_keys=True
    ).encode("utf-8") + b"\n"


def _decode_response(frame: bytes | str) -> dict[str, Any]:
    text = frame.decode("utf-8") if isinstance(frame, bytes) else str(frame)
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise TransitionControlClientError(
            f"invalid transition protocol JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TransitionControlClientError(
            "transition protocol frame must be an object"
        )
    if int(value.get("version", -1)) != TRANSITION_CONTROL_PROTOCOL_VERSION:
        raise TransitionControlClientError("unsupported transition protocol version")
    if value.get("type") != TRANSITION_CONTROL_RESPONSE:
        raise TransitionControlClientError(
            f"unexpected transition protocol type {value.get('type')!r}"
        )
    if not bool(value.get("ok", False)):
        raise TransitionControlClientError(
            str(value.get("error", "transition command failed"))
        )
    result = value.get("result", {})
    if not isinstance(result, Mapping):
        raise TransitionControlClientError(
            "transition response result must be an object"
        )
    return dict(result)


def _receive_frame(sock: socket.socket) -> bytes:
    payload = bytearray()
    while len(payload) < _MAX_FRAME_BYTES:
        chunk = sock.recv(min(4096, _MAX_FRAME_BYTES - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
        newline = payload.find(b"\n")
        if newline >= 0:
            return bytes(payload[:newline])
    if len(payload) >= _MAX_FRAME_BYTES:
        raise TransitionControlClientError(
            "transition protocol frame exceeds 64 KiB"
        )
    if not payload:
        raise TransitionControlClientError("empty transition protocol frame")
    return bytes(payload)
