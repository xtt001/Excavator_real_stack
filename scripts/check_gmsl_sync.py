#!/usr/bin/env python3
"""Read-only startup gate for the four-camera GMSL timestamp group."""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


EXPECTED_CAMERAS = ("video4", "video5", "video6", "video7")


def _request_state(
    stream: Any,
    *,
    step_id: int,
) -> Mapping[str, Any]:
    request = {
        "version": 1,
        "type": "read_state.request",
        "payload": {
            "step_id": int(step_id),
            "request_time_ns": time.time_ns(),
        },
    }
    stream.write(
        json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
    )
    stream.flush()
    line = stream.readline()
    if not line:
        raise RuntimeError("gateway closed before returning read_state.response")
    response = json.loads(line)
    if response.get("type") != "read_state.response" or response.get("ok") is not True:
        raise RuntimeError(f"gateway read_state failed: {response}")
    payload = response.get("payload")
    if not isinstance(payload, Mapping):
        raise RuntimeError("gateway read_state payload is not an object")
    return payload


def camera_group_sample(
    payload: Mapping[str, Any],
    *,
    expected_cameras: Sequence[str] = EXPECTED_CAMERAS,
) -> dict[str, Any]:
    images = payload.get("images")
    if not isinstance(images, Mapping):
        raise RuntimeError("read_state payload has no image map")
    missing = [camera for camera in expected_cameras if camera not in images]
    if missing:
        raise RuntimeError("missing GMSL cameras: " + ",".join(missing))

    metadata: list[Mapping[str, Any]] = []
    for camera in expected_cameras:
        sample = images[camera]
        value = sample.get("metadata") if isinstance(sample, Mapping) else None
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{camera} has no timestamp-group metadata")
        metadata.append(value)

    group_ids = {int(value.get("group_id", 0) or 0) for value in metadata}
    group_counts = {
        int(value.get("group_camera_count", 0) or 0) for value in metadata
    }
    group_valid_values = {
        int(value.get("group_valid", 0) or 0) for value in metadata
    }
    skew_values = [float(value.get("group_skew_ms", float("inf"))) for value in metadata]
    v4l2_errors = [int(value.get("v4l2_error", 0) or 0) for value in metadata]
    coherent = (
        len(group_ids) == 1
        and next(iter(group_ids), 0) > 0
        and group_counts == {len(expected_cameras)}
        and group_valid_values == {1}
        and not any(v4l2_errors)
    )
    return {
        "group_id": next(iter(group_ids), 0) if len(group_ids) == 1 else 0,
        "valid": bool(coherent),
        "skew_ms": max(skew_values),
        "v4l2_timestamps_ns": {
            camera: int(value.get("v4l2_timestamp_ns", 0) or 0)
            for camera, value in zip(expected_cameras, metadata)
        },
    }


def evaluate_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    min_valid_fraction: float,
    max_skew_ms: float,
    min_distinct_groups: int,
) -> dict[str, Any]:
    rows = [dict(sample) for sample in samples]
    if not rows:
        raise RuntimeError("camera sync gate collected no samples")
    valid_count = sum(
        bool(row.get("valid"))
        and float(row.get("skew_ms", float("inf"))) <= float(max_skew_ms)
        for row in rows
    )
    distinct_groups = {
        int(row.get("group_id", 0) or 0)
        for row in rows
        if int(row.get("group_id", 0) or 0) > 0
    }
    skews = [float(row.get("skew_ms", float("inf"))) for row in rows]
    valid_fraction = valid_count / len(rows)
    passed = (
        valid_fraction >= float(min_valid_fraction)
        and len(distinct_groups) >= int(min_distinct_groups)
        and max(skews) <= float(max_skew_ms)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "sample_count": len(rows),
        "valid_count": int(valid_count),
        "valid_fraction": float(valid_fraction),
        "distinct_group_count": len(distinct_groups),
        "skew_ms": {
            "median": float(statistics.median(skews)),
            "max": float(max(skews)),
        },
        "requirements": {
            "min_valid_fraction": float(min_valid_fraction),
            "max_skew_ms": float(max_skew_ms),
            "min_distinct_groups": int(min_distinct_groups),
        },
    }


def collect_samples(
    *,
    host: str,
    port: int,
    duration_s: float,
    rate_hz: float,
    timeout_s: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + float(duration_s)
    period_s = 1.0 / float(rate_hz)
    rows: list[dict[str, Any]] = []
    with socket.create_connection((str(host), int(port)), timeout=float(timeout_s)) as sock:
        sock.settimeout(float(timeout_s))
        stream = sock.makefile("rwb")
        step_id = 0
        while time.monotonic() < deadline:
            started = time.monotonic()
            rows.append(camera_group_sample(_request_state(stream, step_id=step_id)))
            step_id += 1
            remaining = period_s - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.98)
    parser.add_argument("--max-skew-ms", type=float, default=5.0)
    parser.add_argument("--min-distinct-groups", type=int, default=30)
    args = parser.parse_args()
    if args.duration_s <= 0 or args.rate_hz <= 0 or args.timeout_s <= 0:
        parser.error("duration, rate, and timeout must be positive")
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        parser.error("--min-valid-fraction must be in [0,1]")
    try:
        report = evaluate_samples(
            collect_samples(
                host=args.host,
                port=args.port,
                duration_s=args.duration_s,
                rate_hz=args.rate_hz,
                timeout_s=args.timeout_s,
            ),
            min_valid_fraction=args.min_valid_fraction,
            max_skew_ms=args.max_skew_ms,
            min_distinct_groups=args.min_distinct_groups,
        )
    except Exception as exc:
        report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
