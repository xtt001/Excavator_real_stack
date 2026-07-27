#!/usr/bin/env python3
"""Read-only PACT/Unity environment worker for the Real Stack probe.

The parent process launches this script with ``PYTHONPATH`` pointing at the
external PACT checkout.  Keeping it in a subprocess prevents the two
repositories' same-named ``testbed`` packages from sharing one interpreter.
"""

from __future__ import annotations

import argparse
import pickle
import struct
import sys
import traceback
from collections.abc import Mapping
from typing import Any, BinaryIO

import numpy as np

from testbed.backends.agx.backend import AgxSimBackend

FRAME_HEADER = struct.Struct("!Q")
MAX_FRAME_BYTES = 16 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5057)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    backend = AgxSimBackend(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout,
        reset_terrain=True,
        reset_pose=True,
    )
    try:
        while True:
            request = _read_frame(sys.stdin.buffer)
            try:
                result, should_close = _dispatch(backend, request)
                _write_frame(
                    sys.stdout.buffer,
                    {"ok": True, "result": result},
                )
                if should_close:
                    break
            except Exception as exc:
                _write_frame(
                    sys.stdout.buffer,
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
    finally:
        backend.close()


def _dispatch(
    backend: AgxSimBackend,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    operation = str(request.get("op", ""))
    if operation == "get_info":
        info = backend.get_info()
        return (
            {
                "protocol_version": info.protocol_version,
                "runtime_build_id": info.runtime_build_id,
                "dt": float(info.dt),
                "control_hz": float(info.control_hz),
                "action_order": list(info.action_order),
                "qpos_order": list(info.qpos_order),
                "qvel_order": list(info.qvel_order),
                "camera_names": list(info.camera_names),
                "supports_images": bool(info.supports_images),
                "warnings": list(info.warnings),
            },
            False,
        )
    if operation == "reset":
        timestep = backend.reset(seed=int(request.get("seed", 0)))
        return _observable_result(timestep), False
    if operation == "step":
        action = np.asarray(request["action"], dtype=np.float32).reshape(-1)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("step action must be finite shape (4,)")
        return _observable_result(backend.step(action)), False
    if operation == "close":
        return {"closed": True}, True
    raise ValueError(f"unknown worker operation: {operation!r}")


def _observable_result(timestep: Any) -> dict[str, Any]:
    observation = timestep.observation
    encoded = observation.get("encoded_images")
    if not isinstance(encoded, Mapping):
        raise ValueError("PACT AGX backend returned no encoded_images mapping")
    return {
        "qpos": np.asarray(observation["qpos"], dtype=np.float32),
        "qvel": np.asarray(observation["qvel"], dtype=np.float32),
        "encoded_images": dict(encoded),
        "step_id": int(observation["step_id"]),
        "sim_time_ns": int(observation["sim_time_ns"]),
        "warnings": list(observation.get("warnings", [])),
    }


def _write_frame(stream: BinaryIO, payload: Any) -> None:
    encoded = pickle.dumps(payload, protocol=5)
    if len(encoded) > MAX_FRAME_BYTES:
        raise ValueError("worker frame exceeds bounded size")
    stream.write(FRAME_HEADER.pack(len(encoded)))
    stream.write(encoded)
    stream.flush()


def _read_frame(stream: BinaryIO) -> Any:
    header = _read_exact(stream, FRAME_HEADER.size)
    (length,) = FRAME_HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ValueError("worker frame exceeds bounded size")
    return pickle.loads(_read_exact(stream, int(length)))


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("parent pipe closed during framed message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


if __name__ == "__main__":
    main()
