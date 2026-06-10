#!/usr/bin/env python3
"""Read-only IMU/qvel logger for the slave Jetson bridge."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
AXES = ("swing", "boom", "stick", "bucket")
BUCKET_QUATERNION_POLICY_OFFSET_ENV = "EXCAVATOR_BUCKET_QUATERNION_OFFSET_RAD"
BUCKET_QUATERNION_POLICY_OFFSET_DEFAULT_RAD = -0.4060066694119653


def bucket_quaternion_policy_offset_rad() -> float:
    raw = os.environ.get(BUCKET_QUATERNION_POLICY_OFFSET_ENV)
    if raw is None or raw == "":
        return BUCKET_QUATERNION_POLICY_OFFSET_DEFAULT_RAD
    try:
        value = float(raw)
    except ValueError:
        return BUCKET_QUATERNION_POLICY_OFFSET_DEFAULT_RAD
    if not math.isfinite(value):
        return BUCKET_QUATERNION_POLICY_OFFSET_DEFAULT_RAD
    return value


class BridgeClient:
    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._sock: socket.socket | None = None
        self._file: Any | None = None

    def close(self) -> None:
        try:
            if self._file is not None:
                self._file.close()
            if self._sock is not None:
                self._sock.close()
        finally:
            self._file = None
            self._sock = None

    def _connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        self._sock = sock
        self._file = sock.makefile("rwb")

    def request(self, request_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._connect()
        assert self._file is not None
        message = {
            "version": PROTOCOL_VERSION,
            "type": f"{request_type}.request",
            "payload": payload,
        }
        frame = json.dumps(message, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
        try:
            self._file.write(frame)
            self._file.flush()
            response_frame = self._file.readline()
        except OSError:
            self.close()
            raise
        if not response_frame:
            self.close()
            raise RuntimeError("bridge socket closed before response")
        response = json.loads(response_frame.decode("utf-8"))
        expected_type = f"{request_type}.response"
        if int(response.get("version", -1)) != PROTOCOL_VERSION:
            raise RuntimeError(f"bad protocol version: {response.get('version')}")
        if response.get("type") != expected_type:
            raise RuntimeError(f"bad response type: {response.get('type')}, expected {expected_type}")
        if not bool(response.get("ok", True)):
            raise RuntimeError(str(response.get("error", "bridge request failed")))
        payload_raw = response.get("payload", {})
        if not isinstance(payload_raw, dict):
            raise RuntimeError("bridge response payload is not an object")
        return payload_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read qpos/qvel/IMU from the slave bridge, print qvel quality, "
            "and write a JSONL log. This script never sends actions."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="bridge/gateway host, default 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="gateway 8765 or C++ bridge 8766")
    parser.add_argument("--timeout-s", type=float, default=1.0, help="socket timeout")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="read_state polling rate")
    parser.add_argument("--duration-s", type=float, default=60.0, help="0 means run until Ctrl+C")
    parser.add_argument("--output", type=Path, default=None, help="JSONL output path")
    parser.add_argument("--print-every-s", type=float, default=1.0, help="terminal print interval")
    parser.add_argument("--window-s", type=float, default=1.0, help="quality check window")
    parser.add_argument(
        "--stationary-qpos-span-rad",
        type=float,
        default=0.005,
        help="axis is treated as stationary if qpos span is below this in the window",
    )
    parser.add_argument(
        "--stationary-qvel-rad-s",
        type=float,
        default=0.03,
        help="stationary qvel above this is flagged",
    )
    parser.add_argument(
        "--residual-threshold-rad-s",
        type=float,
        default=0.05,
        help="abs(qvel - qpos_diff_qvel) window mean above this is flagged",
    )
    parser.add_argument("--verbose-imu", action="store_true", help="print every IMU gyro/rpy line")
    return parser.parse_args()


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    filename = f"imu_qvel_{stamp}.jsonl"
    candidates = [
        Path("/media/mundane/EXTERNAL_USB/imu_qvel_tests"),
        Path("/media/mundane/D/Excavator_real_stack/logs/imu_qvel_tests"),
        Path.cwd() / "logs" / "imu_qvel_tests",
    ]
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write_test"
            probe.write_text("", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return directory / filename
        except OSError:
            continue
    fallback = Path.cwd() / filename
    return fallback


def to_float_list(value: Any, n: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) < n:
        return None
    try:
        return [float(value[i]) for i in range(n)]
    except (TypeError, ValueError):
        return None


def finite_list(values: list[float] | None) -> bool:
    return values is not None and all(math.isfinite(v) for v in values)


def safe_float(value: Any, default: float = -1.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def angle_delta_rad(current: float, previous: float) -> float:
    delta = current - previous
    if abs(delta) <= math.pi:
        return delta
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def qvel_from_qpos(current: list[float], previous: list[float] | None, dt_s: float | None) -> list[float] | None:
    if previous is None or dt_s is None or dt_s <= 1e-6:
        return None
    return [angle_delta_rad(c, p) / dt_s for c, p in zip(current, previous)]


def gyro_joint_qvel_rad_s(imu_debug: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(imu_debug, dict):
        return None
    devices = imu_debug.get("devices")
    if not isinstance(devices, list) or len(devices) < 4:
        return None

    def gyro_dps(device_index: int, axis_index: int) -> float | None:
        device = devices[device_index]
        if not isinstance(device, dict):
            return None
        gyro = to_float_list(device.get("gyro_dps"), 3)
        if not finite_list(gyro):
            return None
        return gyro[axis_index]

    imu1_y = gyro_dps(0, 1)
    imu2_y = gyro_dps(1, 1)
    imu3_y = gyro_dps(2, 1)
    imu4_z = gyro_dps(3, 2)
    if None in (imu1_y, imu2_y, imu3_y, imu4_z):
        return None
    deg_to_rad = math.pi / 180.0
    return [
        -float(imu4_z) * deg_to_rad,
        float(imu3_y) * deg_to_rad,
        (float(imu2_y) - float(imu3_y)) * deg_to_rad,
        (float(imu1_y) - float(imu2_y)) * deg_to_rad,
    ]


def imu_joint_qpos_from_rpy_rad(imu_debug: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(imu_debug, dict):
        return None
    devices = imu_debug.get("devices")
    if not isinstance(devices, list) or len(devices) < 4:
        return None

    def rpy_rad(device_index: int, axis_index: int) -> float | None:
        device = devices[device_index]
        if not isinstance(device, dict):
            return None
        rpy = to_float_list(device.get("rpy_rad"), 3)
        if not finite_list(rpy):
            return None
        return rpy[axis_index]

    imu1_y = rpy_rad(0, 1)
    imu2_y = rpy_rad(1, 1)
    imu3_y = rpy_rad(2, 1)
    imu4_z = rpy_rad(3, 2)
    if None in (imu1_y, imu2_y, imu3_y, imu4_z):
        return None
    return [
        float(imu4_z),
        float(imu3_y),
        float(imu2_y) - float(imu3_y),
        float(imu1_y) - float(imu2_y),
    ]


def quaternion_wxyz(device: dict[str, Any]) -> list[float] | None:
    try:
        if int(device.get("online", 1)) == 0 or int(device.get("valid_quaternion", 0)) == 0:
            return None
    except (TypeError, ValueError):
        return None
    quat = to_float_list(device.get("quaternion_wxyz"), 4)
    if not finite_list(quat):
        return None
    norm = math.sqrt(sum(float(v) * float(v) for v in quat))
    if not math.isfinite(norm) or norm <= 0.5 or norm >= 1.5:
        return None
    return [float(v) / norm for v in quat]


def quaternion_multiply(lhs: list[float], rhs: list[float]) -> list[float]:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def quaternion_conjugate(quat: list[float]) -> list[float]:
    return [quat[0], -quat[1], -quat[2], -quat[3]]


def bucket_qpos_from_quaternion_rad(devices: list[Any]) -> float | None:
    if len(devices) < 2 or not isinstance(devices[0], dict) or not isinstance(devices[1], dict):
        return None
    imu1_q = quaternion_wxyz(devices[0])
    imu2_q = quaternion_wxyz(devices[1])
    if imu1_q is None or imu2_q is None:
        return None
    relative = quaternion_multiply(quaternion_conjugate(imu2_q), imu1_q)
    return (
        2.0 * math.atan2(relative[2], relative[0])
        + bucket_quaternion_policy_offset_rad()
        + math.pi
    ) % (2.0 * math.pi) - math.pi


def imu_joint_qpos_raw_deg(imu_debug: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(imu_debug, dict):
        return None
    devices = imu_debug.get("devices")
    if not isinstance(devices, list) or len(devices) < 4:
        return None

    def rpy_raw_deg(device_index: int, axis_index: int) -> float | None:
        device = devices[device_index]
        if not isinstance(device, dict):
            return None
        rpy = to_float_list(device.get("rpy_raw_deg"), 3)
        if not finite_list(rpy):
            return None
        return rpy[axis_index]

    imu1_y = rpy_raw_deg(0, 1)
    imu2_y = rpy_raw_deg(1, 1)
    imu3_y = rpy_raw_deg(2, 1)
    imu4_z = rpy_raw_deg(3, 2)
    if None in (imu1_y, imu2_y, imu3_y, imu4_z):
        return None
    bucket_qpos_rad = bucket_qpos_from_quaternion_rad(devices)
    if bucket_qpos_rad is None:
        bucket_qpos_deg = float(imu1_y) - float(imu2_y)
    else:
        bucket_qpos_deg = bucket_qpos_rad * 180.0 / math.pi
    return [
        float(imu4_z),
        float(imu3_y),
        float(imu2_y) - float(imu3_y),
        bucket_qpos_deg,
    ]


def qpos_delta_rad(current: list[float], reference: list[float] | None) -> list[float] | None:
    if reference is None:
        return None
    return [angle_delta_rad(c, r) for c, r in zip(current, reference)]


def vec_minus(current: list[float] | None, reference: list[float] | None) -> list[float] | None:
    if current is None or reference is None:
        return None
    return [c - r for c, r in zip(current, reference)]


def bits(values: Any) -> str:
    if not isinstance(values, list):
        return "----"
    out = []
    for value in values[:4]:
        try:
            out.append("1" if int(value) else "0")
        except (TypeError, ValueError):
            out.append("?")
    while len(out) < 4:
        out.append("-")
    return "".join(out)


def fmt_vec(values: list[float] | None, width: int = 6, precision: int = 3) -> str:
    if values is None:
        return "n/a"
    return "[" + " ".join(f"{v:{width}.{precision}f}" for v in values) + "]"


def fmt_axis_header(width: int = 10, label_width: int = 22) -> str:
    return "  " + f"{'axes':<{label_width}}" + "".join(f"{name:>{width}}" for name in AXES)


def fmt_axis_row(
    label: str,
    values: list[float] | None,
    *,
    width: int = 10,
    precision: int = 2,
    label_width: int = 22,
) -> str:
    prefix = "  " + f"{label:<{label_width}}"
    if values is None:
        return prefix + "n/a"
    return prefix + "".join(f"{v:{width}.{precision}f}" for v in values)


def rad_to_deg(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [v * 180.0 / math.pi for v in values]


def deg_to_rad(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [v * math.pi / 180.0 for v in values]


def mean_abs(columns: list[list[float | None]], axis: int) -> float | None:
    vals = [row[axis] for row in columns if row[axis] is not None and math.isfinite(float(row[axis]))]
    if not vals:
        return None
    return statistics.fmean(abs(float(v)) for v in vals)


def p95_abs(values: list[float]) -> float | None:
    finite = sorted(abs(v) for v in values if math.isfinite(v))
    if not finite:
        return None
    idx = min(len(finite) - 1, int(math.ceil(0.95 * len(finite))) - 1)
    return finite[idx]


def summarize_matrix(samples: list[list[float | None]]) -> dict[str, list[float | None]]:
    summary: dict[str, list[float | None]] = {"mean_abs": [], "p95_abs": [], "max_abs": []}
    for axis in range(len(AXES)):
        vals = [
            float(row[axis])
            for row in samples
            if row[axis] is not None and math.isfinite(float(row[axis]))
        ]
        if not vals:
            summary["mean_abs"].append(None)
            summary["p95_abs"].append(None)
            summary["max_abs"].append(None)
            continue
        summary["mean_abs"].append(statistics.fmean(abs(v) for v in vals))
        summary["p95_abs"].append(p95_abs(vals))
        summary["max_abs"].append(max(abs(v) for v in vals))
    return summary


def summarize_qpos(samples: list[list[float]]) -> dict[str, list[float | None]]:
    summary: dict[str, list[float | None]] = {
        "min": [],
        "max": [],
        "span": [],
        "min_deg": [],
        "max_deg": [],
        "span_deg": [],
    }
    for axis in range(len(AXES)):
        vals = [
            float(row[axis])
            for row in samples
            if row[axis] is not None and math.isfinite(float(row[axis]))
        ]
        if not vals:
            for key in summary:
                summary[key].append(None)
            continue
        vmin = min(vals)
        vmax = max(vals)
        span = vmax - vmin
        summary["min"].append(vmin)
        summary["max"].append(vmax)
        summary["span"].append(span)
        summary["min_deg"].append(vmin * 180.0 / math.pi)
        summary["max_deg"].append(vmax * 180.0 / math.pi)
        summary["span_deg"].append(span * 180.0 / math.pi)
    return summary


def quality_from_window(
    window: deque[dict[str, Any]],
    stationary_qpos_span_rad: float,
    stationary_qvel_rad_s: float,
    residual_threshold_rad_s: float,
) -> dict[str, Any]:
    if not window:
        return {
            "qpos_span": [None] * len(AXES),
            "qvel_abs_mean": [None] * len(AXES),
            "qpos_diff_abs_mean": [None] * len(AXES),
            "residual_abs_mean": [None] * len(AXES),
            "stationary_qvel_bad_axes": [],
            "residual_bad_axes": [],
        }
    qpos_span: list[float | None] = []
    qvel_abs_mean: list[float | None] = []
    diff_abs_mean: list[float | None] = []
    residual_abs_mean: list[float | None] = []
    qpos_columns = [[row["qpos"][axis] for row in window] for axis in range(len(AXES))]
    qvel_rows = [row["qvel_bridge"] for row in window]
    diff_rows = [row["qvel_qpos_diff"] for row in window if row["qvel_qpos_diff"] is not None]
    residual_rows = [
        row["qvel_residual_bridge_minus_qpos_diff"]
        for row in window
        if row["qvel_residual_bridge_minus_qpos_diff"] is not None
    ]

    stationary_bad_axes: list[str] = []
    residual_bad_axes: list[str] = []
    for axis, name in enumerate(AXES):
        qpos_vals = qpos_columns[axis]
        span = max(qpos_vals) - min(qpos_vals)
        qpos_span.append(span)
        qvel_mean = mean_abs(qvel_rows, axis)
        diff_mean = mean_abs(diff_rows, axis)
        residual_mean = mean_abs(residual_rows, axis)
        qvel_abs_mean.append(qvel_mean)
        diff_abs_mean.append(diff_mean)
        residual_abs_mean.append(residual_mean)
        if (
            span <= stationary_qpos_span_rad
            and qvel_mean is not None
            and qvel_mean >= stationary_qvel_rad_s
        ):
            stationary_bad_axes.append(name)
        if residual_mean is not None and residual_mean >= residual_threshold_rad_s:
            residual_bad_axes.append(name)
    return {
        "qpos_span": qpos_span,
        "qvel_abs_mean": qvel_abs_mean,
        "qpos_diff_abs_mean": diff_abs_mean,
        "residual_abs_mean": residual_abs_mean,
        "stationary_qvel_bad_axes": stationary_bad_axes,
        "residual_bad_axes": residual_bad_axes,
    }


def print_imu_debug(imu_debug: dict[str, Any] | None) -> None:
    if not isinstance(imu_debug, dict):
        print("  imu_debug: unavailable; rebuild/restart bridge if raw gyro is needed", flush=True)
        return
    devices = imu_debug.get("devices")
    if not isinstance(devices, list):
        print("  imu_debug.devices: unavailable", flush=True)
        return
    print(
        "  imu  addr on quat gyro    age_ms loss | gyro_dps[x y z]        rpy_deg[x y z]        raw_deg[x y z]",
        flush=True,
    )
    for index, device in enumerate(devices[:4]):
        if not isinstance(device, dict):
            continue
        rpy_rad = to_float_list(device.get("rpy_rad"), 3)
        print(
            f"  imu{index:<1} {str(device.get('device_addr')):>4} "
            f"{str(device.get('online')):>2} {str(device.get('valid_quaternion')):>4} "
            f"{str(device.get('valid_gyro')):>4} "
            f"{safe_float(device.get('host_rx_age_ms')):9.1f} "
            f"{str(device.get('packet_loss_count')):>4} | "
            f"{fmt_vec(to_float_list(device.get('gyro_dps'), 3), width=7, precision=2)} "
            f"{fmt_vec(rad_to_deg(rpy_rad), width=8, precision=2)} "
            f"{fmt_vec(to_float_list(device.get('rpy_raw_deg'), 3), width=8, precision=2)}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    if args.rate_hz <= 0.0:
        raise SystemExit("--rate-hz must be positive")
    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_name(output_path.stem + "_summary.json")

    client = BridgeClient(args.host, args.port, args.timeout_s)
    interval_s = 1.0 / args.rate_hz
    started_wall_ns = time.time_ns()
    started_mono_ns = time.monotonic_ns()
    deadline_mono_ns = None if args.duration_s <= 0.0 else started_mono_ns + int(args.duration_s * 1e9)
    next_print_s = 0.0
    step_id = 0
    read_errors = 0
    last_qpos: list[float] | None = None
    last_joint_ts_ns: int | None = None
    last_read_wall_ns: int | None = None
    window: deque[dict[str, Any]] = deque()
    qvel_samples: list[list[float | None]] = []
    qpos_samples: list[list[float]] = []
    diff_samples: list[list[float | None]] = []
    residual_samples: list[list[float | None]] = []
    raw_imu_samples: list[list[float | None]] = []
    raw_minus_bridge_samples: list[list[float | None]] = []
    rows_written = 0
    missing_imu_debug = 0

    print(f"[imu-qvel] reading {args.host}:{args.port} at target {args.rate_hz:.1f} Hz", flush=True)
    print(f"[imu-qvel] writing {output_path}", flush=True)
    print("[imu-qvel] qvel axes: swing, boom, stick, bucket", flush=True)

    try:
        with output_path.open("w", encoding="utf-8") as fout:
            while True:
                loop_mono_ns = time.monotonic_ns()
                if deadline_mono_ns is not None and loop_mono_ns >= deadline_mono_ns:
                    break
                elapsed_s = (loop_mono_ns - started_mono_ns) / 1e9
                try:
                    response_payload = client.request(
                        "read_state",
                        {
                            "step_id": step_id,
                            "request_time_ns": time.time_ns(),
                        },
                    )
                except Exception as exc:
                    read_errors += 1
                    client.close()
                    if elapsed_s >= next_print_s:
                        print(f"[imu-qvel] read_state error: {exc}", flush=True)
                        next_print_s = elapsed_s + args.print_every_s
                    time.sleep(min(0.2, interval_s))
                    continue

                now_wall_ns = time.time_ns()
                joint = response_payload.get("joint", {})
                if not isinstance(joint, dict):
                    read_errors += 1
                    continue
                payload = joint.get("payload", {})
                if not isinstance(payload, dict):
                    read_errors += 1
                    continue
                qpos = to_float_list(payload.get("qpos"), 4)
                qvel_bridge = to_float_list(payload.get("qvel"), 4)
                if not finite_list(qpos) or not finite_list(qvel_bridge):
                    read_errors += 1
                    continue
                assert qpos is not None
                assert qvel_bridge is not None
                joint_ts_ns = int(joint.get("timestamp_ns") or now_wall_ns)
                if last_joint_ts_ns is None:
                    dt_s = None
                else:
                    dt_s = max(0.0, (joint_ts_ns - last_joint_ts_ns) / 1e9)
                    if dt_s <= 1e-6 and last_read_wall_ns is not None:
                        dt_s = max(0.0, (now_wall_ns - last_read_wall_ns) / 1e9)
                qvel_diff = qvel_from_qpos(qpos, last_qpos, dt_s)
                residual = (
                    [a - b for a, b in zip(qvel_bridge, qvel_diff)]
                    if qvel_diff is not None
                    else None
                )
                imu_health = payload.get("imu_health")
                imu_debug = payload.get("imu_debug")
                if not isinstance(imu_debug, dict):
                    missing_imu_debug += 1
                    imu_debug = None
                raw_imu_qvel = gyro_joint_qvel_rad_s(imu_debug)
                qpos_folded_imu = imu_joint_qpos_from_rpy_rad(imu_debug)
                qpos_raw_imu_deg = imu_joint_qpos_raw_deg(imu_debug)
                qpos_raw_imu = deg_to_rad(qpos_raw_imu_deg)
                qpos_policy_minus_raw_imu = qpos_delta_rad(qpos, qpos_raw_imu)
                qpos_policy_minus_raw_imu_deg_direct = vec_minus(rad_to_deg(qpos), qpos_raw_imu_deg)
                raw_minus_bridge = (
                    [a - b for a, b in zip(raw_imu_qvel, qvel_bridge)]
                    if raw_imu_qvel is not None
                    else None
                )

                window_row = {
                    "elapsed_s": elapsed_s,
                    "qpos": qpos,
                    "qvel_bridge": qvel_bridge,
                    "qvel_qpos_diff": qvel_diff,
                    "qvel_residual_bridge_minus_qpos_diff": residual,
                }
                window.append(window_row)
                while window and elapsed_s - float(window[0]["elapsed_s"]) > args.window_s:
                    window.popleft()
                quality = quality_from_window(
                    window,
                    args.stationary_qpos_span_rad,
                    args.stationary_qvel_rad_s,
                    args.residual_threshold_rad_s,
                )

                qvel_samples.append(qvel_bridge)
                qpos_samples.append(qpos)
                if qvel_diff is not None:
                    diff_samples.append(qvel_diff)
                if residual is not None:
                    residual_samples.append(residual)
                if raw_imu_qvel is not None:
                    raw_imu_samples.append(raw_imu_qvel)
                if raw_minus_bridge is not None:
                    raw_minus_bridge_samples.append(raw_minus_bridge)

                row = {
                    "schema_version": 1,
                    "sample_index": rows_written,
                    "step_id": step_id,
                    "wall_time_ns": now_wall_ns,
                    "monotonic_ns": loop_mono_ns,
                    "elapsed_s": elapsed_s,
                    "bridge": {"host": args.host, "port": args.port},
                    "joint_timestamp_ns": joint_ts_ns,
                    "joint_source": joint.get("source"),
                    "joint_receive_time_ns": joint.get("receive_time_ns"),
                    "dt_s": dt_s,
                    "qpos": qpos,
                    "qpos_deg": rad_to_deg(qpos),
                    "qpos_raw_imu": qpos_raw_imu,
                    "qpos_raw_imu_deg": qpos_raw_imu_deg,
                    "qpos_folded_imu": qpos_folded_imu,
                    "qpos_folded_imu_deg": rad_to_deg(qpos_folded_imu),
                    "qpos_policy_minus_raw_imu": qpos_policy_minus_raw_imu,
                    "qpos_policy_minus_raw_imu_deg": rad_to_deg(qpos_policy_minus_raw_imu),
                    "qpos_policy_minus_raw_imu_deg_direct": qpos_policy_minus_raw_imu_deg_direct,
                    "qvel_bridge": qvel_bridge,
                    "qvel_qpos_diff": qvel_diff,
                    "qvel_residual_bridge_minus_qpos_diff": residual,
                    "qvel_raw_imu_rad_s": raw_imu_qvel,
                    "qvel_raw_imu_minus_bridge": raw_minus_bridge,
                    "status": payload.get("status"),
                    "snapshot_age_ms": payload.get("snapshot_age_ms"),
                    "state_loop_tick": payload.get("state_loop_tick"),
                    "imu_health": imu_health,
                    "imu_debug": imu_debug,
                    "quality": quality,
                }
                fout.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                rows_written += 1

                if elapsed_s >= next_print_s:
                    actual_hz = rows_written / elapsed_s if elapsed_s >= 0.5 else 0.0
                    stationary_bad = quality["stationary_qvel_bad_axes"]
                    residual_bad = quality["residual_bad_axes"]
                    bad_parts = []
                    if stationary_bad:
                        bad_parts.append("stationary_qvel=" + ",".join(stationary_bad))
                    if residual_bad:
                        bad_parts.append("residual=" + ",".join(residual_bad))
                    bad = "ok" if not bad_parts else ";".join(bad_parts)
                    health = imu_health if isinstance(imu_health, dict) else {}
                    print(
                        f"[imu-qvel] t={elapsed_s:7.1f}s hz={actual_hz:5.1f} bad={bad} "
                        f"imu={bits(health.get('online'))} gyro={bits(health.get('valid_gyro'))}",
                        flush=True,
                    )
                    print(fmt_axis_header(width=10, label_width=22), flush=True)
                    print(
                        fmt_axis_row(
                            "qpos raw_imu_deg",
                            qpos_raw_imu_deg,
                            width=10,
                            precision=2,
                            label_width=22,
                        ),
                        flush=True,
                    )
                    print(
                        fmt_axis_row(
                            "qpos policy_deg",
                            rad_to_deg(qpos),
                            width=10,
                            precision=2,
                            label_width=22,
                        ),
                        flush=True,
                    )
                    print(
                        fmt_axis_row(
                            "qpos policy_rad",
                            qpos,
                            width=10,
                            precision=4,
                            label_width=22,
                        ),
                        flush=True,
                    )
                    print(
                        fmt_axis_row(
                            "qpos policy-raw_deg",
                            qpos_policy_minus_raw_imu_deg_direct,
                            width=10,
                            precision=2,
                            label_width=22,
                        ),
                        flush=True,
                    )
                    print(
                        fmt_axis_row(
                            "qpos physical_delta",
                            rad_to_deg(qpos_policy_minus_raw_imu),
                            width=10,
                            precision=2,
                            label_width=22,
                        ),
                        flush=True,
                    )
                    print(
                        fmt_axis_row(
                            "qvel policy_rad_s",
                            qvel_bridge,
                            width=10,
                            precision=4,
                            label_width=22,
                        ),
                        flush=True,
                    )
                    print(
                        fmt_axis_row(
                            "qvel diff_rad_s",
                            qvel_diff,
                            width=10,
                            precision=4,
                            label_width=22,
                        ),
                        flush=True,
                    )
                    print(
                        fmt_axis_row(
                            "qvel raw_gyro_rad_s",
                            raw_imu_qvel,
                            width=10,
                            precision=4,
                            label_width=22,
                        ),
                        flush=True,
                    )
                    print(
                        fmt_axis_row(
                            "qvel resid_rad_s",
                            residual,
                            width=10,
                            precision=4,
                            label_width=22,
                        ),
                        flush=True,
                    )
                    if args.verbose_imu:
                        print_imu_debug(imu_debug)
                    next_print_s = elapsed_s + args.print_every_s

                last_qpos = qpos
                last_joint_ts_ns = joint_ts_ns
                last_read_wall_ns = now_wall_ns
                step_id += 1
                sleep_s = interval_s - (time.monotonic_ns() - loop_mono_ns) / 1e9
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n[imu-qvel] stopped by Ctrl+C", flush=True)
    finally:
        client.close()

    summary = {
        "schema_version": 1,
        "output": str(output_path),
        "rows": rows_written,
        "read_errors": read_errors,
        "missing_imu_debug_rows": missing_imu_debug,
        "target_rate_hz": args.rate_hz,
        "duration_s": (time.monotonic_ns() - started_mono_ns) / 1e9,
        "axes": list(AXES),
        "qpos": summarize_qpos(qpos_samples),
        "qvel_bridge": summarize_matrix(qvel_samples),
        "qvel_qpos_diff": summarize_matrix(diff_samples),
        "qvel_residual_bridge_minus_qpos_diff": summarize_matrix(residual_samples),
        "qvel_raw_imu_rad_s": summarize_matrix(raw_imu_samples),
        "qvel_raw_imu_minus_bridge": summarize_matrix(raw_minus_bridge_samples),
        "notes": [
            "qvel_bridge is the qvel currently returned by read_state.",
            "qvel_qpos_diff is finite-difference qpos using bridge timestamps.",
            "qpos/qpos_deg are the current joint pose in radians/degrees.",
            "qpos_raw_imu/qpos_raw_imu_deg are reconstructed before qpos continuity filtering; bucket uses relative quaternion when valid and falls back to imu_debug.rpy_raw_deg.",
            "qpos_folded_imu/qpos_folded_imu_deg are reconstructed from imu_debug.rpy_rad after per-axis angle folding.",
            "qpos_policy_minus_raw_imu is the shortest-angle delta from raw IMU joint pose to policy qpos.",
            "qpos_policy_minus_raw_imu_deg_direct is policy_deg - raw_imu_deg without branch wrapping.",
            "qvel_raw_imu_rad_s is derived from raw gyro before converter startup bias subtraction.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[imu-qvel] summary {summary_path}", flush=True)
    if rows_written == 0:
        print("[imu-qvel] no rows written", file=sys.stderr, flush=True)
        return 2
    if missing_imu_debug > 0:
        print(
            "[imu-qvel] warning: imu_debug missing in some rows; rebuild/restart bridge for raw gyro fields",
            file=sys.stderr,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
