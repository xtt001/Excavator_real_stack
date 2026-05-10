#!/usr/bin/env python3
"""Protocol checks for the C++ excavator_real_bridge smoke test."""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any


def _read_line(sock: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(1)
        if not chunk:
            raise RuntimeError("bridge closed before sending a response")
        if chunk == b"\n":
            break
        chunks.append(chunk)
    return json.loads(b"".join(chunks).decode("utf-8"))


def _send_raw(sock: socket.socket, raw: str) -> dict[str, Any]:
    sock.sendall(raw.encode("utf-8") + b"\n")
    return _read_line(sock)


def _request(sock: socket.socket, msg_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return _send_raw(
        sock,
        json.dumps(
            {
                "version": 1,
                "type": msg_type,
                "payload": {} if payload is None else payload,
            },
            separators=(",", ":"),
        ),
    )


def _expect(condition: bool, message: str, response: dict[str, Any]) -> None:
    if not condition:
        raise RuntimeError(f"{message}; response={response}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--heartbeat-timeout-ms", type=int, default=500)
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()

    with socket.create_connection((args.host, args.port), timeout=5.0) as sock:
        sock.settimeout(5.0)

        invalid_json = _send_raw(sock, "{not json")
        _expect(invalid_json.get("ok") is False, "invalid JSON should fail", invalid_json)

        missing_action = _request(sock, "send_action.request", {})
        _expect(missing_action.get("ok") is False, "missing action should fail", missing_action)

        wrong_dim = _request(sock, "send_action.request", {"action": [0.0, 0.0, 0.0]})
        _expect(wrong_dim.get("ok") is False, "wrong action dimension should fail", wrong_dim)

        valid_action = _request(sock, "send_action.request", {"action": [0.1, 0.0, 0.0, 0.0]})
        payload = valid_action.get("payload", {})
        _expect(valid_action.get("ok") is True, "valid action should succeed", valid_action)
        _expect(payload.get("ack") is True, "valid action should be acknowledged", valid_action)
        _expect(
            payload.get("commanded_action") == [0.1, 0.0, 0.0, 0.0],
            "commanded_action should round-trip",
            valid_action,
        )
        _expect(
            payload.get("raw_low_level_command") == [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "raw_low_level_command should be 8D with fixed trailing zeros",
            valid_action,
        )

        time.sleep(max(args.heartbeat_timeout_ms / 1000.0 + 0.25, 0.3))
        state = _request(sock, "read_state.request", {"step_id": 0})
        _expect(state.get("ok") is True, "read_state should succeed after watchdog", state)
        _expect("joint" in state.get("payload", {}), "read_state should include joint sample", state)
        _expect("fpv" in state.get("payload", {}).get("images", {}), "read_state should include fpv image", state)

        if args.shutdown:
            shutdown = _request(sock, "shutdown.request")
            _expect(shutdown.get("ok") is True, "shutdown should succeed", shutdown)
        else:
            close = _request(sock, "close.request")
            _expect(close.get("ok") is True, "close should succeed", close)

    print("bridge protocol checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
