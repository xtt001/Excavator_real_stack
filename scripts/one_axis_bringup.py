#!/usr/bin/env python3
"""Low-speed one-axis bridge command tool for supervised hardware bring-up."""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any


AXIS_INDEX = {
    "swing": 0,
    "boom": 1,
    "stick": 2,
    "bucket": 3,
}


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


def _request(sock: socket.socket, msg_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    message = {
        "version": 1,
        "type": msg_type,
        "payload": {} if payload is None else payload,
    }
    sock.sendall(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n")
    response = _read_line(sock)
    if response.get("type") != msg_type.replace(".request", ".response"):
        raise RuntimeError(f"unexpected response type: {response}")
    if response.get("ok") is not True:
        raise RuntimeError(f"bridge request failed: {response}")
    return response


def _action(axis: str, value: float) -> list[float]:
    out = [0.0, 0.0, 0.0, 0.0]
    out[AXIS_INDEX[axis]] = float(value)
    return out


def _send_action(sock: socket.socket, action: list[float], label: str) -> None:
    response = _request(sock, "send_action.request", {"action": action, "send_time_ns": time.time_ns()})
    payload = response.get("payload", {})
    print(
        f"{label}: ack={payload.get('ack')} fault={payload.get('fault_code')!r} "
        f"raw={payload.get('raw_low_level_command')}"
    )


def _read_state(sock: socket.socket, step_id: int) -> None:
    response = _request(
        sock,
        "read_state.request",
        {"step_id": int(step_id), "request_time_ns": time.time_ns()},
    )
    payload = response.get("payload", {})
    joint = payload.get("joint", {}).get("payload", {})
    print(
        f"state step={step_id}: qpos={joint.get('qpos')} qvel={joint.get('qvel')} "
        f"status={joint.get('status')}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--axis", choices=sorted(AXIS_INDEX), required=True)
    parser.add_argument("--direction", choices=("positive", "negative", "both"), default="positive")
    parser.add_argument("--amplitude", type=float, default=0.03)
    parser.add_argument("--max-amplitude", type=float, default=0.08)
    parser.add_argument("--duration-s", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument(
        "--confirm-hardware-motion",
        action="store_true",
        help="Required for any non-zero command, including simulated rehearsals.",
    )
    parser.add_argument("--shutdown", action="store_true", help="Ask the bridge process to exit at the end.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.port <= 0:
        raise ValueError("--port must be positive")
    if not (0.0 < args.amplitude <= args.max_amplitude <= 0.2):
        raise ValueError("--amplitude must be positive, <= --max-amplitude, and --max-amplitude <= 0.2")
    if args.duration_s <= 0 or args.settle_s < 0 or args.rate_hz <= 0:
        raise ValueError("--duration-s and --rate-hz must be positive; --settle-s must be non-negative")
    if not args.confirm_hardware_motion:
        raise RuntimeError("--confirm-hardware-motion is required before sending a non-zero axis command")

    period_s = 1.0 / args.rate_hz
    directions = [1.0, -1.0] if args.direction == "both" else [1.0 if args.direction == "positive" else -1.0]
    step_id = 0

    with socket.create_connection((args.host, args.port), timeout=args.timeout_s) as sock:
        sock.settimeout(args.timeout_s)
        _send_action(sock, [0.0, 0.0, 0.0, 0.0], "pre-zero")
        _read_state(sock, step_id)
        step_id += 1

        for sign in directions:
            value = sign * args.amplitude
            print(f"axis={args.axis} command={value:+.4f} duration_s={args.duration_s:.3f}")
            deadline = time.monotonic() + args.duration_s
            while time.monotonic() < deadline:
                _send_action(sock, _action(args.axis, value), "axis")
                time.sleep(period_s)
            _send_action(sock, [0.0, 0.0, 0.0, 0.0], "post-axis-zero")
            _read_state(sock, step_id)
            step_id += 1
            if args.settle_s > 0:
                settle_deadline = time.monotonic() + args.settle_s
                while time.monotonic() < settle_deadline:
                    _send_action(sock, [0.0, 0.0, 0.0, 0.0], "settle-zero")
                    time.sleep(period_s)

        if args.shutdown:
            _request(sock, "shutdown.request")
            print("bridge shutdown requested")
        else:
            _request(sock, "close.request")
    print("one-axis bring-up sequence complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"one_axis_bringup failed: {exc}")
        raise SystemExit(1)
