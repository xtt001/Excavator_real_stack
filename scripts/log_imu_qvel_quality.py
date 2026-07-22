#!/usr/bin/env python3
"""Read-only IMU/qvel logger for the slave Jetson bridge."""

from __future__ import annotations

import argparse
import csv
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
IMU_AXIS_NAMES = ("x", "y", "z")
IMU_RPY_NAMES = ("roll", "pitch", "yaw")
IMU_QUATERNION_NAMES = ("w", "x", "y", "z")
# imu_health/imu_debug use physical-chain order, not the policy AXES order.
# Keep this mapping explicit anywhere a health bit or device index is shown.
IMU_LAYOUT = (
    {"index": 0, "name": "bucket", "can_id": "0x122"},
    {"index": 1, "name": "stick", "can_id": "0x124"},
    {"index": 2, "name": "boom", "can_id": "0x121"},
    {"index": 3, "name": "swing", "can_id": "0x123"},
)
BUCKET_QUATERNION_POLICY_OFFSET_RAD = -0.4060066694119653
BUCKET_PRIMARY_CHART_MIN_STRENGTH = 0.35
BUCKET_TILT_OUTER_REFERENCE_RAD = -0.012059366757032714
BUCKET_TILT_POLICY_OFFSET_RAD = 0.07040270164837034
BUCKET_GRAVITY_HINGE_OUTER_REFERENCE_RAD = 2.0839045979023254
BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD = -2.025561263010988
BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW = 21
DAOYUAN_CHAIN_STICK_POLICY_OFFSET_RAD = 0.19801020488135143
DAOYUAN_CHAIN_BUCKET_POLICY_OFFSET_RAD = -2.006833804661174


def _normalize_joint_rpy_profile(raw: Any) -> str:
    raw = str(raw or "legacy_diff").strip().lower()
    if raw in {"daoyuan_chain", "daoyuan", "chain_rpy"}:
        return "daoyuan_chain"
    return "legacy_diff"


def joint_rpy_profile(imu_debug: dict[str, Any] | None = None) -> str:
    if isinstance(imu_debug, dict):
        raw_profile = imu_debug.get("joint_rpy_profile")
        if raw_profile:
            return _normalize_joint_rpy_profile(raw_profile)
        mapping = imu_debug.get("joint_velocity_mapping")
        if isinstance(mapping, dict):
            stick = mapping.get("stick")
            bucket = mapping.get("bucket")
            if isinstance(stick, dict) and "+" in str(stick.get("gyro_axis", "")):
                return "daoyuan_chain"
            if isinstance(bucket, dict) and "+" in str(bucket.get("position_axis", "")):
                return "daoyuan_chain"
    return _normalize_joint_rpy_profile(os.environ.get("EXCAVATOR_JOINT_RPY_PROFILE"))


def _normalize_bucket_imu0_profile(raw: Any) -> str:
    raw = str(raw or "legacy_y").strip().lower()
    if raw in {"roll_ccw90", "rotated_ccw90", "imu0_roll", "roll"}:
        return "roll_ccw90"
    return "legacy_y"


def bucket_imu0_profile(imu_debug: dict[str, Any] | None = None) -> str:
    if isinstance(imu_debug, dict):
        raw_profile = imu_debug.get("bucket_imu0_profile")
        if raw_profile:
            return _normalize_bucket_imu0_profile(raw_profile)
        mapping = imu_debug.get("joint_velocity_mapping")
        if isinstance(mapping, dict):
            bucket = mapping.get("bucket")
            if isinstance(bucket, dict):
                raw_profile = bucket.get("position_profile")
                if raw_profile:
                    return _normalize_bucket_imu0_profile(raw_profile)
                gyro_axis = str(bucket.get("gyro_axis", "")).lower()
                if "imu0-x" in gyro_axis:
                    return "roll_ccw90"
    return _normalize_bucket_imu0_profile(os.environ.get("EXCAVATOR_BUCKET_IMU0_PROFILE", "legacy_y"))


def bucket_imu0_roll_profile_enabled(imu_debug: dict[str, Any] | None = None) -> bool:
    return bucket_imu0_profile(imu_debug) == "roll_ccw90"


def _safe_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _health_bits4(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) < len(IMU_LAYOUT):
        return None
    out: list[int] = []
    for raw in value[: len(IMU_LAYOUT)]:
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            return None
        if parsed not in (0, 1):
            return None
        out.append(parsed)
    return out


def _health_ages4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < len(IMU_LAYOUT):
        return None
    out: list[float] = []
    for raw in value[: len(IMU_LAYOUT)]:
        parsed = _safe_finite_float(raw)
        if parsed is None:
            return None
        out.append(parsed)
    return out


def _debug_snapshot_initialized(imu_debug: Any) -> bool | None:
    if not isinstance(imu_debug, dict):
        return None
    devices = imu_debug.get("devices")
    if not isinstance(devices, list) or len(devices) < len(IMU_LAYOUT):
        return None
    initialized = False
    for raw_device in devices[: len(IMU_LAYOUT)]:
        if not isinstance(raw_device, dict):
            return None
        try:
            device_addr = int(raw_device.get("device_addr", 0) or 0)
            host_rx_time_ns = int(raw_device.get("host_rx_time_ns", 0) or 0)
        except (TypeError, ValueError):
            return None
        initialized = initialized or device_addr > 0 or host_rx_time_ns > 0
    return initialized


def _imu_labels(indices: list[int]) -> list[str]:
    return [
        f"imu{index}:{IMU_LAYOUT[index]['name']}@{IMU_LAYOUT[index]['can_id']}"
        for index in indices
    ]


def classify_imu_status(
    imu_health: Any,
    imu_debug: Any = None,
    *,
    max_stale_ms: float = 100.0,
) -> dict[str, Any]:
    """Classify per-device IMU health without double-counting one device.

    Offline devices are not also labelled stale/invalid. A default bridge
    snapshot (all device_addr/host_rx_time fields still zero) is reported as
    unavailable rather than as four physical IMUs missing.
    """

    unavailable = {
        "state": "imu_health_unavailable",
        "fault_device_count": None,
        "fault_indices": [],
        "fault_devices": [],
        "missing_indices": [],
        "stale_indices": [],
        "invalid_attitude_indices": [],
        "invalid_gyro_indices": [],
        "invalid_accel_indices": [],
        "error_codes": ["imu_health_unavailable"],
        "online_bits": "----",
        "valid_attitude_bits": "----",
        "valid_quaternion_bits": "----",
        "valid_gyro_bits": "----",
        "valid_accel_bits": "----",
    }
    if not isinstance(imu_health, dict):
        return unavailable

    online = _health_bits4(imu_health.get("online"))
    if online is None:
        return unavailable

    snapshot_initialized = _debug_snapshot_initialized(imu_debug)
    if not any(online) and snapshot_initialized is not True:
        reason = (
            "imu_snapshot_uninitialized"
            if snapshot_initialized is False
            else "imu_health_unavailable"
        )
        return {
            **unavailable,
            "state": reason,
            "error_codes": [reason],
            "online_bits": bits(online),
        }

    valid_attitude = _health_bits4(imu_health.get("valid_attitude"))
    # Daoyuan CAN packets provide native RPY but no quaternion. Keep the bit
    # visible for diagnosis without treating it as a required runtime signal.
    valid_quaternion = _health_bits4(imu_health.get("valid_quaternion"))
    valid_gyro = _health_bits4(imu_health.get("valid_gyro"))
    valid_accel = _health_bits4(imu_health.get("valid_accel"))
    ages_ms = _health_ages4(imu_health.get("host_rx_age_ms"))
    missing_fields = [
        name
        for name, value in (
            ("valid_attitude", valid_attitude),
            ("valid_gyro", valid_gyro),
            ("valid_accel", valid_accel),
            ("host_rx_age_ms", ages_ms),
        )
        if value is None
    ]
    if missing_fields:
        return {
            **unavailable,
            "state": "imu_health_unavailable",
            "error_codes": ["imu_health_unavailable:" + ",".join(missing_fields)],
            "online_bits": bits(online),
            "missing_fields": missing_fields,
        }

    assert valid_attitude is not None
    assert valid_gyro is not None
    assert valid_accel is not None
    assert ages_ms is not None
    missing = [index for index, value in enumerate(online) if value == 0]
    stale = [
        index
        for index, (is_online, age_ms) in enumerate(zip(online, ages_ms))
        if is_online == 1 and (age_ms < 0.0 or age_ms > float(max_stale_ms))
    ]
    invalid_attitude = [
        index
        for index, (is_online, value) in enumerate(zip(online, valid_attitude))
        if is_online == 1 and index not in stale and value == 0
    ]
    invalid_gyro = [
        index
        for index, (is_online, value) in enumerate(zip(online, valid_gyro))
        if is_online == 1 and index not in stale and value == 0
    ]
    invalid_accel = [
        index
        for index, (is_online, value) in enumerate(zip(online, valid_accel))
        if is_online == 1 and index not in stale and value == 0
    ]
    fault_indices = sorted(
        set(missing + stale + invalid_attitude + invalid_gyro + invalid_accel)
    )
    reasons_by_index: dict[int, list[str]] = {index: [] for index in fault_indices}
    for reason, indices in (
        ("offline", missing),
        ("stale", stale),
        ("attitude_invalid", invalid_attitude),
        ("gyro_invalid", invalid_gyro),
        ("accel_invalid", invalid_accel),
    ):
        for index in indices:
            reasons_by_index[index].append(reason)
    fault_devices = [
        {
            **IMU_LAYOUT[index],
            "label": _imu_labels([index])[0],
            "reasons": reasons_by_index[index],
        }
        for index in fault_indices
    ]
    error_codes = []
    for prefix, indices in (
        ("imu_missing", missing),
        ("imu_stale", stale),
        ("imu_attitude_invalid", invalid_attitude),
        ("imu_gyro_invalid", invalid_gyro),
        ("imu_accel_invalid", invalid_accel),
    ):
        if indices:
            error_codes.append(prefix + ":" + ",".join(str(index) for index in indices))
    state = (
        "ok"
        if not fault_indices
        else "single_imu_error"
        if len(fault_indices) == 1
        else "multiple_imu_errors"
    )
    return {
        "state": state,
        "fault_device_count": len(fault_indices),
        "fault_indices": fault_indices,
        "fault_devices": fault_devices,
        "missing_indices": missing,
        "stale_indices": stale,
        "invalid_attitude_indices": invalid_attitude,
        "invalid_gyro_indices": invalid_gyro,
        "invalid_accel_indices": invalid_accel,
        "error_codes": error_codes,
        "online_bits": bits(online),
        "valid_attitude_bits": bits(valid_attitude),
        "valid_quaternion_bits": (
            "----" if valid_quaternion is None else bits(valid_quaternion)
        ),
        "valid_gyro_bits": bits(valid_gyro),
        "valid_accel_bits": bits(valid_accel),
    }


def format_imu_status(status: dict[str, Any]) -> str:
    parts = [
        f"imu_state={status.get('state', 'imu_health_unavailable')}",
        f"fault_devices={status.get('fault_device_count', '?')}",
    ]
    for key, label in (
        ("missing_indices", "offline"),
        ("stale_indices", "stale"),
        ("invalid_attitude_indices", "attitude_invalid"),
        ("invalid_gyro_indices", "gyro_invalid"),
        ("invalid_accel_indices", "accel_invalid"),
    ):
        indices = status.get(key)
        if isinstance(indices, list) and indices:
            parts.append(label + "=" + ",".join(_imu_labels(indices)))
    errors = status.get("error_codes")
    if status.get("state") in {"imu_health_unavailable", "imu_snapshot_uninitialized"}:
        if isinstance(errors, list) and errors:
            parts.append("reason=" + ",".join(str(value) for value in errors))
    return " ".join(parts)


def bucket_imu0_reference_rad(imu_debug: dict[str, Any] | None = None) -> float:
    if not bucket_imu0_roll_profile_enabled(imu_debug):
        return 0.0
    if isinstance(imu_debug, dict):
        value = _safe_finite_float(imu_debug.get("bucket_imu0_reference_rad"))
        if value is not None:
            return value
    value = _safe_finite_float(os.environ.get("EXCAVATOR_BUCKET_IMU0_REFERENCE_RAD", "0"))
    return 0.0 if value is None else value


def bucket_imu0_axis_sign(imu_debug: dict[str, Any] | None = None) -> float:
    if not bucket_imu0_roll_profile_enabled(imu_debug):
        return 1.0
    if isinstance(imu_debug, dict):
        value = _safe_finite_float(imu_debug.get("bucket_imu0_sign"))
        if value is not None:
            return -1.0 if value < 0.0 else 1.0
        value = _safe_finite_float(imu_debug.get("bucket_imu0_gyro_sign"))
        if value is not None:
            return -1.0 if value < 0.0 else 1.0
    value = _safe_finite_float(
        os.environ.get(
            "EXCAVATOR_BUCKET_IMU0_SIGN",
            os.environ.get("EXCAVATOR_BUCKET_IMU0_GYRO_SIGN", "1"),
        )
    )
    return -1.0 if value is not None and value < 0.0 else 1.0


def daoyuan_chain_stick_policy_offset_rad(imu_debug: dict[str, Any] | None = None) -> float:
    if isinstance(imu_debug, dict):
        value = _safe_finite_float(imu_debug.get("daoyuan_stick_policy_offset_rad"))
        if value is not None:
            return value
    value = _safe_finite_float(
        os.environ.get(
            "EXCAVATOR_DAOYUAN_STICK_POLICY_OFFSET_RAD",
            str(DAOYUAN_CHAIN_STICK_POLICY_OFFSET_RAD),
        )
    )
    return DAOYUAN_CHAIN_STICK_POLICY_OFFSET_RAD if value is None else value


def daoyuan_chain_bucket_policy_offset_rad(imu_debug: dict[str, Any] | None = None) -> float:
    if isinstance(imu_debug, dict):
        value = _safe_finite_float(imu_debug.get("daoyuan_bucket_policy_offset_rad"))
        if value is not None:
            return value
    value = _safe_finite_float(
        os.environ.get(
            "EXCAVATOR_DAOYUAN_BUCKET_POLICY_OFFSET_RAD",
            str(DAOYUAN_CHAIN_BUCKET_POLICY_OFFSET_RAD),
        )
    )
    return DAOYUAN_CHAIN_BUCKET_POLICY_OFFSET_RAD if value is None else value


def bucket_tilt_outer_reference_rad() -> float:
    value = _safe_finite_float(
        os.environ.get(
            "EXCAVATOR_BUCKET_TILT_REFERENCE_RAD",
            str(BUCKET_TILT_OUTER_REFERENCE_RAD),
        )
    )
    return BUCKET_TILT_OUTER_REFERENCE_RAD if value is None else value


def bucket_tilt_policy_offset_rad() -> float:
    value = _safe_finite_float(
        os.environ.get(
            "EXCAVATOR_BUCKET_TILT_POLICY_OFFSET_RAD",
            str(BUCKET_TILT_POLICY_OFFSET_RAD),
        )
    )
    return BUCKET_TILT_POLICY_OFFSET_RAD if value is None else value


def bucket_gravity_hinge_outer_reference_rad() -> float:
    value = _safe_finite_float(
        os.environ.get(
            "EXCAVATOR_BUCKET_GRAVITY_HINGE_REFERENCE_RAD",
            str(BUCKET_GRAVITY_HINGE_OUTER_REFERENCE_RAD),
        )
    )
    return BUCKET_GRAVITY_HINGE_OUTER_REFERENCE_RAD if value is None else value


def bucket_gravity_hinge_policy_offset_rad() -> float:
    value = _safe_finite_float(
        os.environ.get(
            "EXCAVATOR_BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD",
            str(BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD),
        )
    )
    return BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD if value is None else value


def bucket_gravity_hinge_median_window() -> int:
    value = _safe_finite_float(
        os.environ.get(
            "EXCAVATOR_BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW",
            str(BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW),
        )
    )
    if value is None:
        return BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW
    return max(1, int(round(value)))


def bucket_initial_position_rad(
    primary_phase_rad: float,
    imu_debug: dict[str, Any] | None = None,
) -> float:
    return primary_phase_rad


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
    parser.add_argument(
        "--vendor-csv-output",
        type=Path,
        default=None,
        help="flat CSV output for IMU vendor analysis; default is <jsonl stem>_vendor.csv",
    )
    parser.add_argument(
        "--no-vendor-csv",
        action="store_true",
        help="do not write the flat IMU vendor CSV",
    )
    parser.add_argument("--print-every-s", type=float, default=1.0, help="terminal print interval")
    parser.add_argument(
        "--imu-max-stale-ms",
        type=float,
        default=100.0,
        help="per-IMU host receive age above this is reported stale",
    )
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


def to_optional_float_list(value: Any, n: int) -> list[float | None] | None:
    if not isinstance(value, list) or len(value) < n:
        return None
    out: list[float | None] = []
    for raw in value[:n]:
        if raw is None:
            out.append(None)
            continue
        parsed = _safe_finite_float(raw)
        if parsed is None:
            return None
        out.append(parsed)
    return out


def finite_list(values: list[float] | None) -> bool:
    return values is not None and all(math.isfinite(v) for v in values)


def _device_signal(
    devices: list[Any],
    device_index: int,
    field: str,
    size: int,
    *,
    valid_field: str,
) -> list[float] | None:
    if device_index >= len(devices):
        return None
    device = devices[device_index]
    if not isinstance(device, dict):
        return None
    try:
        if int(device.get("online", 1)) == 0:
            return None
        # Keep older logs usable when a validity field was not yet emitted,
        # but never consume a value explicitly marked invalid.
        if valid_field in device and int(device.get(valid_field, 0)) == 0:
            return None
    except (TypeError, ValueError):
        return None
    values = to_float_list(device.get(field), size)
    return values if finite_list(values) else None


def angle_delta_rad(current: float, previous: float) -> float:
    delta = current - previous
    if abs(delta) <= math.pi:
        return delta
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def unwrap_angle_nearest(previous: float, current: float) -> float:
    return previous + angle_delta_rad(current, previous)


def qvel_from_qpos(current: list[float], previous: list[float] | None, dt_s: float | None) -> list[float] | None:
    if previous is None or dt_s is None or dt_s <= 1e-6:
        return None
    return [angle_delta_rad(c, p) / dt_s for c, p in zip(current, previous)]


def gyro_joint_qvel_rad_s(
    imu_debug: dict[str, Any] | None,
) -> list[float | None] | None:
    if not isinstance(imu_debug, dict):
        return None
    devices = imu_debug.get("devices")
    if not isinstance(devices, list) or len(devices) < 4:
        return None

    def gyro_dps(device_index: int, axis_index: int) -> float | None:
        gyro = _device_signal(
            devices,
            device_index,
            "gyro_dps",
            3,
            valid_field="valid_gyro",
        )
        if gyro is None:
            return None
        return gyro[axis_index]

    imu1_bucket = gyro_dps(0, 0 if bucket_imu0_roll_profile_enabled(imu_debug) else 1)
    imu2_y = gyro_dps(1, 1)
    imu3_y = gyro_dps(2, 1)
    imu4_z = gyro_dps(3, 2)
    deg_to_rad = math.pi / 180.0
    if joint_rpy_profile(imu_debug) == "daoyuan_chain":
        return [
            None if imu4_z is None else -float(imu4_z) * deg_to_rad,
            None if imu3_y is None else float(imu3_y) * deg_to_rad,
            (
                None
                if imu2_y is None or imu3_y is None
                else (float(imu2_y) + float(imu3_y)) * deg_to_rad
            ),
            (
                None
                if imu1_bucket is None or imu2_y is None
                else -(float(imu1_bucket) + float(imu2_y)) * deg_to_rad
            ),
        ]
    return [
        None if imu4_z is None else -float(imu4_z) * deg_to_rad,
        None if imu3_y is None else float(imu3_y) * deg_to_rad,
        (
            None
            if imu2_y is None or imu3_y is None
            else (float(imu2_y) - float(imu3_y)) * deg_to_rad
        ),
        (
            None
            if imu1_bucket is None or imu2_y is None
            else bucket_imu0_axis_sign(imu_debug)
            * (float(imu2_y) - float(imu1_bucket))
            * deg_to_rad
        ),
    ]


def imu_joint_qpos_from_rpy_rad(
    imu_debug: dict[str, Any] | None,
) -> list[float | None] | None:
    if not isinstance(imu_debug, dict):
        return None
    devices = imu_debug.get("devices")
    if not isinstance(devices, list) or len(devices) < 4:
        return None

    def rpy_rad(device_index: int, axis_index: int) -> float | None:
        rpy = _device_signal(
            devices,
            device_index,
            "rpy_rad",
            3,
            valid_field="valid_attitude",
        )
        if rpy is None:
            return None
        return rpy[axis_index]

    imu1_bucket = rpy_rad(0, 0 if bucket_imu0_roll_profile_enabled(imu_debug) else 1)
    imu2_y = rpy_rad(1, 1)
    imu3_y = rpy_rad(2, 1)
    imu4_z = rpy_rad(3, 2)
    if joint_rpy_profile(imu_debug) == "daoyuan_chain":
        return [
            None if imu4_z is None else float(imu4_z),
            None if imu3_y is None else float(imu3_y),
            (
                None
                if imu2_y is None or imu3_y is None
                else float(imu2_y)
                + float(imu3_y)
                + daoyuan_chain_stick_policy_offset_rad(imu_debug)
            ),
            (
                None
                if imu1_bucket is None or imu2_y is None
                else -(float(imu1_bucket) + float(imu2_y))
                + daoyuan_chain_bucket_policy_offset_rad(imu_debug)
            ),
        ]
    return [
        None if imu4_z is None else float(imu4_z),
        None if imu3_y is None else float(imu3_y),
        (
            None
            if imu2_y is None or imu3_y is None
            else float(imu2_y) - float(imu3_y)
        ),
        (
            None
            if imu1_bucket is None or imu2_y is None
            else bucket_imu0_axis_sign(imu_debug)
            * (
                float(imu1_bucket)
                - float(imu2_y)
                - bucket_imu0_reference_rad(imu_debug)
            )
        ),
    ]


def _bucket_tilt_raw_from_accel_rad(
    imu_debug: dict[str, Any] | None,
) -> tuple[float, float, float, float, float] | None:
    if not isinstance(imu_debug, dict):
        return None
    devices = imu_debug.get("devices")
    if not isinstance(devices, list) or len(devices) < 2:
        return None

    def accel(device_index: int) -> list[float] | None:
        return _device_signal(
            devices,
            device_index,
            "accel_mps2",
            3,
            valid_field="valid_accel",
        )

    imu0_accel = accel(0)
    imu1_accel = accel(1)
    if imu0_accel is None or imu1_accel is None:
        return None
    imu0_xz = math.atan2(float(imu0_accel[0]), float(imu0_accel[2]))
    imu1_yz = math.atan2(float(imu1_accel[1]), float(imu1_accel[2]))
    bucket = imu0_xz - imu1_yz
    imu0_norm = math.sqrt(sum(float(v) * float(v) for v in imu0_accel))
    imu1_norm = math.sqrt(sum(float(v) * float(v) for v in imu1_accel))
    if not all(math.isfinite(v) for v in (bucket, imu0_xz, imu1_yz, imu0_norm, imu1_norm)):
        return None
    return bucket, imu0_xz, imu1_yz, imu0_norm, imu1_norm


class BucketTiltAccelTracker:
    def __init__(self) -> None:
        self._ready = False
        self._bucket_rad = 0.0
        self._imu0_xz_rad = 0.0
        self._imu1_yz_rad = 0.0

    def update(self, imu_debug: dict[str, Any] | None) -> dict[str, float] | None:
        raw = _bucket_tilt_raw_from_accel_rad(imu_debug)
        if raw is None:
            return None
        bucket, imu0_xz, imu1_yz, imu0_norm, imu1_norm = raw
        if not self._ready:
            self._bucket_rad = bucket
            self._imu0_xz_rad = imu0_xz
            self._imu1_yz_rad = imu1_yz
            self._ready = True
        else:
            self._imu0_xz_rad = unwrap_angle_nearest(self._imu0_xz_rad, imu0_xz)
            self._imu1_yz_rad = unwrap_angle_nearest(self._imu1_yz_rad, imu1_yz)
            self._bucket_rad = self._imu0_xz_rad - self._imu1_yz_rad
        return {
            "bucket_rad": self._bucket_rad,
            "bucket_deg": math.degrees(self._bucket_rad),
            "imu0_xz_rad": self._imu0_xz_rad,
            "imu0_xz_deg": math.degrees(self._imu0_xz_rad),
            "imu1_yz_rad": self._imu1_yz_rad,
            "imu1_yz_deg": math.degrees(self._imu1_yz_rad),
            "imu0_accel_norm_mps2": imu0_norm,
            "imu1_accel_norm_mps2": imu1_norm,
            "formula": "atan2(imu0.accel_x,imu0.accel_z)-atan2(imu1.accel_y,imu1.accel_z)",
        }


def _bucket_gravity_hinge_raw_from_accel_rad(
    imu_debug: dict[str, Any] | None,
) -> tuple[float, float, float, float, float] | None:
    if not isinstance(imu_debug, dict):
        return None
    devices = imu_debug.get("devices")
    if not isinstance(devices, list) or len(devices) < 2:
        return None

    def accel(device_index: int) -> list[float] | None:
        return _device_signal(
            devices,
            device_index,
            "accel_mps2",
            3,
            valid_field="valid_accel",
        )

    imu0_accel = accel(0)
    imu1_accel = accel(1)
    if imu0_accel is None or imu1_accel is None:
        return None
    # Offline-selected mechanical axes: IMU0 hinge axis +X, IMU1 hinge axis +Y.
    # For +X, use the Y/Z plane. For +Y, use X/-Z to keep the calibrated sign.
    imu0_phase = math.atan2(float(imu0_accel[2]), float(imu0_accel[1]))
    imu1_phase = math.atan2(-float(imu1_accel[2]), float(imu1_accel[0]))
    bucket = imu0_phase - imu1_phase
    imu0_norm = math.sqrt(sum(float(v) * float(v) for v in imu0_accel))
    imu1_norm = math.sqrt(sum(float(v) * float(v) for v in imu1_accel))
    if not all(math.isfinite(v) for v in (bucket, imu0_phase, imu1_phase, imu0_norm, imu1_norm)):
        return None
    return bucket, imu0_phase, imu1_phase, imu0_norm, imu1_norm


class BucketGravityHingeTracker:
    def __init__(
        self,
        *,
        reference_rad: float,
        policy_offset_rad: float,
        median_window: int,
    ) -> None:
        self.reference_rad = reference_rad
        self.policy_offset_rad = policy_offset_rad
        self.median_window = max(1, int(median_window))
        self._ready = False
        self._bucket_rad = 0.0
        self._imu0_phase_rad = 0.0
        self._imu1_phase_rad = 0.0
        self._outer_zero_window: deque[float] = deque(maxlen=self.median_window)

    def update(self, imu_debug: dict[str, Any] | None) -> dict[str, float] | None:
        raw = _bucket_gravity_hinge_raw_from_accel_rad(imu_debug)
        if raw is None:
            return None
        bucket, imu0_phase, imu1_phase, imu0_norm, imu1_norm = raw
        if not self._ready:
            self._bucket_rad = bucket
            self._imu0_phase_rad = imu0_phase
            self._imu1_phase_rad = imu1_phase
            self._ready = True
        else:
            self._imu0_phase_rad = unwrap_angle_nearest(self._imu0_phase_rad, imu0_phase)
            self._imu1_phase_rad = unwrap_angle_nearest(self._imu1_phase_rad, imu1_phase)
            self._bucket_rad = self._imu0_phase_rad - self._imu1_phase_rad

        outer_zero_rad = self._bucket_rad - self.reference_rad
        policy_aligned_rad = self._bucket_rad + self.policy_offset_rad
        self._outer_zero_window.append(outer_zero_rad)
        median_outer_zero_rad = statistics.median(self._outer_zero_window)
        median_policy_aligned_rad = median_outer_zero_rad + self.reference_rad + self.policy_offset_rad
        return {
            "bucket_rad": self._bucket_rad,
            "bucket_deg": math.degrees(self._bucket_rad),
            "imu0_phase_rad": self._imu0_phase_rad,
            "imu0_phase_deg": math.degrees(self._imu0_phase_rad),
            "imu1_phase_rad": self._imu1_phase_rad,
            "imu1_phase_deg": math.degrees(self._imu1_phase_rad),
            "imu0_accel_norm_mps2": imu0_norm,
            "imu1_accel_norm_mps2": imu1_norm,
            "reference_rad": self.reference_rad,
            "reference_deg": math.degrees(self.reference_rad),
            "policy_offset_rad": self.policy_offset_rad,
            "policy_offset_deg": math.degrees(self.policy_offset_rad),
            "outer_zero_rad": outer_zero_rad,
            "outer_zero_deg": math.degrees(outer_zero_rad),
            "policy_aligned_rad": policy_aligned_rad,
            "policy_aligned_deg": math.degrees(policy_aligned_rad),
            "median_window": self.median_window,
            "median_sample_count": len(self._outer_zero_window),
            "median_outer_zero_rad": median_outer_zero_rad,
            "median_outer_zero_deg": math.degrees(median_outer_zero_rad),
            "median_policy_aligned_rad": median_policy_aligned_rad,
            "median_policy_aligned_deg": math.degrees(median_policy_aligned_rad),
            "formula": (
                "unwrap(atan2(imu0.accel_z,imu0.accel_y)"
                "-atan2(-imu1.accel_z,imu1.accel_x)); hinge axes imu0+X imu1+Y"
            ),
        }


def _quaternion_from_devices(
    devices: list[Any], device_index: int
) -> tuple[float, float, float, float] | None:
    device = devices[device_index]
    if not isinstance(device, dict):
        return None
    try:
        if int(device.get("online", 1)) == 0 or int(device.get("valid_quaternion", 1)) == 0:
            return None
    except (TypeError, ValueError):
        return None
    q = to_float_list(device.get("quaternion_wxyz"), 4)
    if not finite_list(q):
        return None
    assert q is not None
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    return tuple(float(v) / norm for v in q)


def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return w, -x, -y, -z


def _quat_multiply(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _normalize_quaternion(
    q: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    return tuple(float(v) / norm for v in q)


def _bucket_relative_quaternion_wxyz(devices: list[Any]) -> tuple[float, float, float, float] | None:
    imu1 = _quaternion_from_devices(devices, 0)
    imu2 = _quaternion_from_devices(devices, 1)
    if imu1 is None or imu2 is None:
        return None
    return _normalize_quaternion(_quat_multiply(_quat_conjugate(imu2), imu1))


def _bucket_quaternion_charts_from_relative_rad(
    relative: tuple[float, float, float, float],
    *,
    profile: str = "legacy_y",
    sign: float = 1.0,
) -> tuple[float, float, float, float] | None:
    normalized = _normalize_quaternion(relative)
    if normalized is None:
        return None
    rel_w, rel_x, rel_y, rel_z = normalized
    if profile == "roll_ccw90":
        primary = sign * math.remainder(2.0 * math.atan2(rel_x, rel_w), 2.0 * math.pi)
        secondary = sign * math.remainder(2.0 * math.atan2(rel_y, rel_z), 2.0 * math.pi)
        primary_strength = math.hypot(rel_w, rel_x)
        secondary_strength = math.hypot(rel_y, rel_z)
    else:
        primary = (
            math.remainder(2.0 * math.atan2(rel_y, rel_w), 2.0 * math.pi)
            + BUCKET_QUATERNION_POLICY_OFFSET_RAD
        )
        secondary = math.remainder(-2.0 * math.atan2(rel_x, rel_z), 2.0 * math.pi)
        primary_strength = math.hypot(rel_w, rel_y)
        secondary_strength = math.hypot(rel_x, rel_z)
    if not all(
        math.isfinite(v)
        for v in (primary, secondary, primary_strength, secondary_strength)
    ):
        return None
    return primary, secondary, primary_strength, secondary_strength


def _bucket_quaternion_charts_rad(
    devices: list[Any],
    *,
    profile: str = "legacy_y",
    sign: float = 1.0,
    relative_reference: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float] | None:
    relative = _bucket_relative_quaternion_wxyz(devices)
    if relative is None:
        return None
    if profile == "roll_ccw90" and relative_reference is not None:
        relative = _quat_multiply(_quat_conjugate(relative_reference), relative)
    return _bucket_quaternion_charts_from_relative_rad(relative, profile=profile, sign=sign)


class BucketQuaternionPhaseTracker:
    def __init__(self) -> None:
        self._ready = False
        self._primary_phase_rad = 0.0
        self._secondary_phase_rad = 0.0
        self._bucket_rad = 0.0
        self._profile = "legacy_y"
        self._sign = 1.0
        self._reference_rad = 0.0
        self._relative_reference: tuple[float, float, float, float] | None = None

    @staticmethod
    def _raw_roll_minus_pitch_rad(devices: list[Any]) -> float | None:
        roll0 = _device_signal(
            devices, 0, "rpy_raw_deg", 3, valid_field="valid_attitude"
        )
        pitch1 = _device_signal(
            devices, 1, "rpy_raw_deg", 3, valid_field="valid_attitude"
        )
        if roll0 is None or pitch1 is None:
            return None
        return math.radians(float(roll0[0]) - float(pitch1[1]))

    @staticmethod
    def _raw_daoyuan_chain_bucket_rad(devices: list[Any]) -> float | None:
        roll0 = _device_signal(
            devices, 0, "rpy_raw_deg", 3, valid_field="valid_attitude"
        )
        pitch1 = _device_signal(
            devices, 1, "rpy_raw_deg", 3, valid_field="valid_attitude"
        )
        if roll0 is None or pitch1 is None:
            return None
        return -math.radians(float(roll0[0]) + float(pitch1[1]))

    def update_from_devices(
        self,
        devices: list[Any],
        *,
        profile: str = "legacy_y",
        sign: float = 1.0,
        reference_rad: float = 0.0,
    ) -> float | None:
        raw_profile = str(profile or "legacy_y").strip().lower()
        profile = "daoyuan_chain" if raw_profile == "daoyuan_chain" else _normalize_bucket_imu0_profile(raw_profile)
        sign = -1.0 if sign < 0.0 else 1.0
        if (
            self._ready
            and (
                profile != self._profile
                or sign != self._sign
                or abs(reference_rad - self._reference_rad) > 1e-12
            )
        ):
            self._ready = False
            self._relative_reference = None
        self._profile = profile
        self._sign = sign
        self._reference_rad = reference_rad
        if profile == "daoyuan_chain":
            phase = self._raw_daoyuan_chain_bucket_rad(devices)
            if phase is None:
                return None
            phase = phase + reference_rad
            if not self._ready:
                self._primary_phase_rad = phase
                self._secondary_phase_rad = phase
                self._bucket_rad = phase
                self._ready = True
                return self._bucket_rad
            phase = unwrap_angle_nearest(self._primary_phase_rad, phase)
            self._bucket_rad += math.remainder(phase - self._primary_phase_rad, 2.0 * math.pi)
            self._primary_phase_rad = phase
            self._secondary_phase_rad = phase
            return self._bucket_rad
        if profile == "roll_ccw90":
            phase = self._raw_roll_minus_pitch_rad(devices)
            if phase is None:
                return None
            phase = sign * (phase - reference_rad)
            if not self._ready:
                self._primary_phase_rad = phase
                self._secondary_phase_rad = phase
                self._bucket_rad = phase
                self._ready = True
                return self._bucket_rad
            phase = unwrap_angle_nearest(self._primary_phase_rad, phase)
            self._bucket_rad += math.remainder(phase - self._primary_phase_rad, 2.0 * math.pi)
            self._primary_phase_rad = phase
            self._secondary_phase_rad = phase
            return self._bucket_rad
        charts = _bucket_quaternion_charts_rad(
            devices,
            profile=profile,
            sign=sign,
            relative_reference=self._relative_reference,
        )
        if charts is None:
            return None
        primary, secondary, primary_strength, secondary_strength = charts
        if not self._ready:
            self._primary_phase_rad = primary
            self._secondary_phase_rad = secondary
            self._bucket_rad = primary
            self._ready = True
            return self._bucket_rad
        use_secondary = (
            primary_strength < BUCKET_PRIMARY_CHART_MIN_STRENGTH
            and secondary_strength > primary_strength
        )
        primary_delta = math.remainder(primary - self._primary_phase_rad, 2.0 * math.pi)
        secondary_delta = math.remainder(
            secondary - self._secondary_phase_rad, 2.0 * math.pi
        )
        self._bucket_rad += secondary_delta if use_secondary else primary_delta
        self._primary_phase_rad = primary
        self._secondary_phase_rad = secondary
        return self._bucket_rad


def imu_joint_qpos_raw_deg(
    imu_debug: dict[str, Any] | None,
    *,
    bucket_tracker: BucketQuaternionPhaseTracker | None = None,
) -> list[float | None] | None:
    if not isinstance(imu_debug, dict):
        return None
    devices = imu_debug.get("devices")
    if not isinstance(devices, list) or len(devices) < 4:
        return None

    def rpy_raw_deg(device_index: int, axis_index: int) -> float | None:
        rpy = _device_signal(
            devices,
            device_index,
            "rpy_raw_deg",
            3,
            valid_field="valid_attitude",
        )
        if rpy is None:
            return None
        return rpy[axis_index]

    imu0_roll = rpy_raw_deg(0, 0)
    imu1_y = rpy_raw_deg(0, 1)
    imu2_y = rpy_raw_deg(1, 1)
    imu3_y = rpy_raw_deg(2, 1)
    imu4_z = rpy_raw_deg(3, 2)
    joint_profile = joint_rpy_profile(imu_debug)
    profile = bucket_imu0_profile(imu_debug)
    sign = bucket_imu0_axis_sign(imu_debug)
    reference_rad = bucket_imu0_reference_rad(imu_debug)
    if joint_profile == "daoyuan_chain":
        if bucket_tracker is not None:
            bucket_rad = bucket_tracker.update_from_devices(
                devices,
                profile="daoyuan_chain",
                sign=1.0,
                reference_rad=daoyuan_chain_bucket_policy_offset_rad(imu_debug),
            )
        elif imu0_roll is not None and imu2_y is not None:
            bucket_rad = (
                -math.radians(float(imu0_roll) + float(imu2_y))
                + daoyuan_chain_bucket_policy_offset_rad(imu_debug)
            )
        else:
            bucket_rad = None
    elif bucket_tracker is not None:
        bucket_rad = bucket_tracker.update_from_devices(
            devices,
            profile=profile,
            sign=sign,
            reference_rad=reference_rad,
        )
    elif profile == "roll_ccw90":
        if imu1_y is None or imu2_y is None:
            bucket_rad = None
        else:
            bucket_rad = sign * (math.radians(float(imu1_y) - float(imu2_y)) - reference_rad)
    else:
        charts = _bucket_quaternion_charts_rad(devices, profile=profile, sign=sign)
        bucket_rad = None if charts is None else charts[0]
    bucket_deg = None if bucket_rad is None else math.degrees(bucket_rad)
    if joint_profile == "daoyuan_chain":
        return [
            None if imu4_z is None else float(imu4_z),
            None if imu3_y is None else float(imu3_y),
            (
                None
                if imu2_y is None or imu3_y is None
                else float(imu2_y)
                + float(imu3_y)
                + math.degrees(daoyuan_chain_stick_policy_offset_rad(imu_debug))
            ),
            None if bucket_deg is None else float(bucket_deg),
        ]
    return [
        None if imu4_z is None else float(imu4_z),
        None if imu3_y is None else float(imu3_y),
        (
            None
            if imu2_y is None or imu3_y is None
            else float(imu2_y) - float(imu3_y)
        ),
        None if bucket_deg is None else float(bucket_deg),
    ]


def qpos_delta_rad(
    current: list[float],
    reference: list[float | None] | None,
) -> list[float | None] | None:
    if reference is None:
        return None
    return [None if r is None else angle_delta_rad(c, r) for c, r in zip(current, reference)]


def vec_minus(
    current: list[float | None] | None,
    reference: list[float | None] | None,
) -> list[float | None] | None:
    if current is None or reference is None:
        return None
    return [None if c is None or r is None else c - r for c, r in zip(current, reference)]


def branch_alias_axes_from_direct_delta(
    direct_delta_deg: list[float | None] | None,
    *,
    alias_period_deg: float = 360.0,
    tolerance_deg: float = 20.0,
) -> list[str]:
    if direct_delta_deg is None:
        return []
    axes: list[str] = []
    for name, delta in zip(AXES, direct_delta_deg):
        if delta is None:
            continue
        if not math.isfinite(delta):
            continue
        branch_count = round(delta / alias_period_deg)
        if branch_count == 0:
            continue
        if abs(delta - branch_count * alias_period_deg) <= tolerance_deg:
            axes.append(name)
    return axes


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


def fmt_vec(values: list[float | None] | None, width: int = 6, precision: int = 3) -> str:
    if values is None:
        return "n/a"
    return "[" + " ".join(
        f"{'n/a':>{width}}" if value is None else f"{value:{width}.{precision}f}"
        for value in values
    ) + "]"


def fmt_axis_header(width: int = 10, label_width: int = 22) -> str:
    return "  " + f"{'axes':<{label_width}}" + "".join(f"{name:>{width}}" for name in AXES)


def fmt_axis_row(
    label: str,
    values: list[float | None] | None,
    *,
    width: int = 10,
    precision: int = 2,
    label_width: int = 22,
) -> str:
    prefix = "  " + f"{label:<{label_width}}"
    if values is None:
        return prefix + "n/a"
    return prefix + "".join(
        f"{'n/a':>{width}}" if value is None else f"{value:{width}.{precision}f}"
        for value in values
    )


def rad_to_deg(values: list[float | None] | None) -> list[float | None] | None:
    if values is None:
        return None
    return [None if value is None else value * 180.0 / math.pi for value in values]


def _vendor_csv_fieldnames() -> list[str]:
    fields = [
        "schema_version",
        "sample_index",
        "step_id",
        "wall_time_ns",
        "monotonic_ns",
        "elapsed_s",
        "joint_timestamp_ns",
        "joint_receive_time_ns",
        "dt_s",
        "snapshot_age_ms",
        "state_loop_tick",
        "imu_state",
        "imu_fault_device_count",
        "imu_error_codes",
        "imu_fault_devices",
        "bucket_imu0_profile",
        "bucket_imu0_reference_rad",
        "bucket_imu0_sign",
        "bucket_tilt_accel_formula",
        "bucket_tilt_accel_rad",
        "bucket_tilt_accel_deg",
        "bucket_tilt_accel_reference_rad",
        "bucket_tilt_accel_reference_deg",
        "bucket_tilt_accel_policy_offset_rad",
        "bucket_tilt_accel_policy_offset_deg",
        "bucket_tilt_accel_outer_zero_rad",
        "bucket_tilt_accel_outer_zero_deg",
        "bucket_tilt_accel_policy_aligned_rad",
        "bucket_tilt_accel_policy_aligned_deg",
        "bucket_tilt_accel_policy_aligned_minus_bridge_rad",
        "bucket_tilt_accel_policy_aligned_minus_bridge_deg",
        "bucket_tilt_accel_first_frame_offset_rad",
        "bucket_tilt_accel_first_frame_offset_deg",
        "bucket_tilt_accel_first_aligned_rad",
        "bucket_tilt_accel_first_aligned_deg",
        "bucket_tilt_accel_first_aligned_minus_bridge_rad",
        "bucket_tilt_accel_first_aligned_minus_bridge_deg",
        "bucket_tilt_imu0_xz_rad",
        "bucket_tilt_imu0_xz_deg",
        "bucket_tilt_imu1_yz_rad",
        "bucket_tilt_imu1_yz_deg",
        "bucket_tilt_imu0_accel_norm_mps2",
        "bucket_tilt_imu1_accel_norm_mps2",
        "bucket_gravity_hinge_formula",
        "bucket_gravity_hinge_rad",
        "bucket_gravity_hinge_deg",
        "bucket_gravity_hinge_reference_rad",
        "bucket_gravity_hinge_reference_deg",
        "bucket_gravity_hinge_policy_offset_rad",
        "bucket_gravity_hinge_policy_offset_deg",
        "bucket_gravity_hinge_outer_zero_rad",
        "bucket_gravity_hinge_outer_zero_deg",
        "bucket_gravity_hinge_policy_aligned_rad",
        "bucket_gravity_hinge_policy_aligned_deg",
        "bucket_gravity_hinge_policy_aligned_minus_bridge_rad",
        "bucket_gravity_hinge_policy_aligned_minus_bridge_deg",
        "bucket_gravity_hinge_median_window",
        "bucket_gravity_hinge_median_sample_count",
        "bucket_gravity_hinge_median_outer_zero_rad",
        "bucket_gravity_hinge_median_outer_zero_deg",
        "bucket_gravity_hinge_median_policy_aligned_rad",
        "bucket_gravity_hinge_median_policy_aligned_deg",
        "bucket_gravity_hinge_median_policy_aligned_minus_bridge_rad",
        "bucket_gravity_hinge_median_policy_aligned_minus_bridge_deg",
        "bucket_gravity_hinge_imu0_phase_rad",
        "bucket_gravity_hinge_imu0_phase_deg",
        "bucket_gravity_hinge_imu1_phase_rad",
        "bucket_gravity_hinge_imu1_phase_deg",
        "bucket_gravity_hinge_imu0_accel_norm_mps2",
        "bucket_gravity_hinge_imu1_accel_norm_mps2",
    ]
    for name in AXES:
        fields.extend(
            [
                f"qpos_{name}_rad",
                f"qpos_{name}_deg",
                f"qvel_bridge_{name}_rad_s",
                f"qvel_qpos_diff_{name}_rad_s",
                f"qvel_raw_imu_{name}_rad_s",
                f"qvel_residual_{name}_rad_s",
                f"qpos_raw_imu_{name}_rad",
                f"qpos_raw_imu_{name}_deg",
                f"qpos_folded_imu_{name}_rad",
                f"qpos_folded_imu_{name}_deg",
            ]
        )
    for index in range(4):
        prefix = f"imu{index}"
        fields.extend(
            [
                f"{prefix}_device_addr",
                f"{prefix}_online",
                f"{prefix}_valid_attitude",
                f"{prefix}_valid_quaternion",
                f"{prefix}_valid_gyro",
                f"{prefix}_valid_accel",
                f"{prefix}_packet_loss_count",
                f"{prefix}_imu_timestamp_ms",
                f"{prefix}_host_rx_time_ns",
                f"{prefix}_host_rx_age_ms",
            ]
        )
        for axis in IMU_RPY_NAMES:
            fields.append(f"{prefix}_rpy_{axis}_rad")
        for axis in IMU_RPY_NAMES:
            fields.append(f"{prefix}_rpy_raw_{axis}_deg")
        for axis in IMU_AXIS_NAMES:
            fields.append(f"{prefix}_gyro_{axis}_dps")
        for axis in IMU_AXIS_NAMES:
            fields.append(f"{prefix}_accel_{axis}_mps2")
        for axis in IMU_QUATERNION_NAMES:
            fields.append(f"{prefix}_quaternion_{axis}")
    return fields


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return value if math.isfinite(value) else ""
    return value


def _put_axis_values(
    out: dict[str, Any],
    prefix: str,
    suffix: str,
    values: list[float | None] | None,
) -> None:
    for axis, name in enumerate(AXES):
        value = None if values is None or axis >= len(values) else values[axis]
        out[f"{prefix}_{name}_{suffix}"] = _csv_value(value)


def vendor_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    imu_debug = row.get("imu_debug")
    if not isinstance(imu_debug, dict):
        imu_debug = None
    profile = str(row.get("bucket_imu0_profile") or bucket_imu0_profile(imu_debug))
    reference_rad = row.get("bucket_imu0_reference_rad")
    if reference_rad is None:
        reference_rad = bucket_imu0_reference_rad(imu_debug)
    sign = row.get("bucket_imu0_sign")
    if sign is None:
        sign = bucket_imu0_axis_sign(imu_debug)
    imu_status = row.get("imu_status")
    if not isinstance(imu_status, dict):
        imu_status = {}
    error_codes = imu_status.get("error_codes")
    fault_devices = imu_status.get("fault_devices")
    out: dict[str, Any] = {
        "schema_version": row.get("schema_version"),
        "sample_index": row.get("sample_index"),
        "step_id": row.get("step_id"),
        "wall_time_ns": row.get("wall_time_ns"),
        "monotonic_ns": row.get("monotonic_ns"),
        "elapsed_s": row.get("elapsed_s"),
        "joint_timestamp_ns": row.get("joint_timestamp_ns"),
        "joint_receive_time_ns": row.get("joint_receive_time_ns"),
        "dt_s": _csv_value(row.get("dt_s")),
        "snapshot_age_ms": _csv_value(row.get("snapshot_age_ms")),
        "state_loop_tick": row.get("state_loop_tick"),
        "imu_state": _csv_value(imu_status.get("state")),
        "imu_fault_device_count": _csv_value(imu_status.get("fault_device_count")),
        "imu_error_codes": (
            ";".join(str(value) for value in error_codes)
            if isinstance(error_codes, list)
            else ""
        ),
        "imu_fault_devices": (
            ";".join(
                str(device.get("label"))
                for device in fault_devices
                if isinstance(device, dict) and device.get("label")
            )
            if isinstance(fault_devices, list)
            else ""
        ),
        "bucket_imu0_profile": profile,
        "bucket_imu0_reference_rad": _csv_value(reference_rad),
        "bucket_imu0_sign": _csv_value(sign),
    }
    bucket_tilt = row.get("bucket_tilt_accel")
    if not isinstance(bucket_tilt, dict):
        bucket_tilt = {}
    out["bucket_tilt_accel_formula"] = _csv_value(bucket_tilt.get("formula"))
    tilt_csv_map = {
        "bucket_tilt_accel_rad": "bucket_rad",
        "bucket_tilt_accel_deg": "bucket_deg",
        "bucket_tilt_accel_reference_rad": "reference_rad",
        "bucket_tilt_accel_reference_deg": "reference_deg",
        "bucket_tilt_accel_policy_offset_rad": "policy_offset_rad",
        "bucket_tilt_accel_policy_offset_deg": "policy_offset_deg",
        "bucket_tilt_accel_outer_zero_rad": "outer_zero_rad",
        "bucket_tilt_accel_outer_zero_deg": "outer_zero_deg",
        "bucket_tilt_accel_policy_aligned_rad": "policy_aligned_rad",
        "bucket_tilt_accel_policy_aligned_deg": "policy_aligned_deg",
        "bucket_tilt_accel_policy_aligned_minus_bridge_rad": "policy_aligned_minus_bridge_rad",
        "bucket_tilt_accel_policy_aligned_minus_bridge_deg": "policy_aligned_minus_bridge_deg",
        "bucket_tilt_accel_first_frame_offset_rad": "first_frame_offset_rad",
        "bucket_tilt_accel_first_frame_offset_deg": "first_frame_offset_deg",
        "bucket_tilt_accel_first_aligned_rad": "first_aligned_rad",
        "bucket_tilt_accel_first_aligned_deg": "first_aligned_deg",
        "bucket_tilt_accel_first_aligned_minus_bridge_rad": "first_aligned_minus_bridge_rad",
        "bucket_tilt_accel_first_aligned_minus_bridge_deg": "first_aligned_minus_bridge_deg",
        "bucket_tilt_imu0_xz_rad": "imu0_xz_rad",
        "bucket_tilt_imu0_xz_deg": "imu0_xz_deg",
        "bucket_tilt_imu1_yz_rad": "imu1_yz_rad",
        "bucket_tilt_imu1_yz_deg": "imu1_yz_deg",
        "bucket_tilt_imu0_accel_norm_mps2": "imu0_accel_norm_mps2",
        "bucket_tilt_imu1_accel_norm_mps2": "imu1_accel_norm_mps2",
    }
    for out_name, field in tilt_csv_map.items():
        out[out_name] = _csv_value(bucket_tilt.get(field))
    bucket_gravity_hinge = row.get("bucket_gravity_hinge_accel")
    if not isinstance(bucket_gravity_hinge, dict):
        bucket_gravity_hinge = {}
    out["bucket_gravity_hinge_formula"] = _csv_value(bucket_gravity_hinge.get("formula"))
    gravity_csv_map = {
        "bucket_gravity_hinge_rad": "bucket_rad",
        "bucket_gravity_hinge_deg": "bucket_deg",
        "bucket_gravity_hinge_reference_rad": "reference_rad",
        "bucket_gravity_hinge_reference_deg": "reference_deg",
        "bucket_gravity_hinge_policy_offset_rad": "policy_offset_rad",
        "bucket_gravity_hinge_policy_offset_deg": "policy_offset_deg",
        "bucket_gravity_hinge_outer_zero_rad": "outer_zero_rad",
        "bucket_gravity_hinge_outer_zero_deg": "outer_zero_deg",
        "bucket_gravity_hinge_policy_aligned_rad": "policy_aligned_rad",
        "bucket_gravity_hinge_policy_aligned_deg": "policy_aligned_deg",
        "bucket_gravity_hinge_policy_aligned_minus_bridge_rad": "policy_aligned_minus_bridge_rad",
        "bucket_gravity_hinge_policy_aligned_minus_bridge_deg": "policy_aligned_minus_bridge_deg",
        "bucket_gravity_hinge_median_window": "median_window",
        "bucket_gravity_hinge_median_sample_count": "median_sample_count",
        "bucket_gravity_hinge_median_outer_zero_rad": "median_outer_zero_rad",
        "bucket_gravity_hinge_median_outer_zero_deg": "median_outer_zero_deg",
        "bucket_gravity_hinge_median_policy_aligned_rad": "median_policy_aligned_rad",
        "bucket_gravity_hinge_median_policy_aligned_deg": "median_policy_aligned_deg",
        "bucket_gravity_hinge_median_policy_aligned_minus_bridge_rad": (
            "median_policy_aligned_minus_bridge_rad"
        ),
        "bucket_gravity_hinge_median_policy_aligned_minus_bridge_deg": (
            "median_policy_aligned_minus_bridge_deg"
        ),
        "bucket_gravity_hinge_imu0_phase_rad": "imu0_phase_rad",
        "bucket_gravity_hinge_imu0_phase_deg": "imu0_phase_deg",
        "bucket_gravity_hinge_imu1_phase_rad": "imu1_phase_rad",
        "bucket_gravity_hinge_imu1_phase_deg": "imu1_phase_deg",
        "bucket_gravity_hinge_imu0_accel_norm_mps2": "imu0_accel_norm_mps2",
        "bucket_gravity_hinge_imu1_accel_norm_mps2": "imu1_accel_norm_mps2",
    }
    for out_name, field in gravity_csv_map.items():
        out[out_name] = _csv_value(bucket_gravity_hinge.get(field))
    _put_axis_values(out, "qpos", "rad", to_float_list(row.get("qpos"), 4))
    _put_axis_values(out, "qpos", "deg", to_float_list(row.get("qpos_deg"), 4))
    _put_axis_values(out, "qvel_bridge", "rad_s", to_float_list(row.get("qvel_bridge"), 4))
    _put_axis_values(out, "qvel_qpos_diff", "rad_s", to_float_list(row.get("qvel_qpos_diff"), 4))
    _put_axis_values(
        out,
        "qvel_raw_imu",
        "rad_s",
        to_optional_float_list(row.get("qvel_raw_imu_rad_s"), 4),
    )
    _put_axis_values(
        out,
        "qvel_residual",
        "rad_s",
        to_float_list(row.get("qvel_residual_bridge_minus_qpos_diff"), 4),
    )
    _put_axis_values(
        out,
        "qpos_raw_imu",
        "rad",
        to_optional_float_list(row.get("qpos_raw_imu"), 4),
    )
    _put_axis_values(
        out,
        "qpos_raw_imu",
        "deg",
        to_optional_float_list(row.get("qpos_raw_imu_deg"), 4),
    )
    _put_axis_values(
        out,
        "qpos_folded_imu",
        "rad",
        to_optional_float_list(row.get("qpos_folded_imu"), 4),
    )
    _put_axis_values(
        out,
        "qpos_folded_imu",
        "deg",
        to_optional_float_list(row.get("qpos_folded_imu_deg"), 4),
    )

    devices = imu_debug.get("devices") if isinstance(imu_debug, dict) else None
    if not isinstance(devices, list):
        devices = []
    for index in range(4):
        prefix = f"imu{index}"
        raw_device = devices[index] if index < len(devices) else {}
        device = raw_device if isinstance(raw_device, dict) else {}
        for field in (
            "device_addr",
            "online",
            "valid_attitude",
            "valid_quaternion",
            "valid_gyro",
            "valid_accel",
            "packet_loss_count",
            "imu_timestamp_ms",
            "host_rx_time_ns",
            "host_rx_age_ms",
        ):
            out[f"{prefix}_{field}"] = _csv_value(device.get(field))
        rpy_rad = to_float_list(device.get("rpy_rad"), 3)
        rpy_raw_deg = to_float_list(device.get("rpy_raw_deg"), 3)
        gyro = to_float_list(device.get("gyro_dps"), 3)
        accel = to_float_list(device.get("accel_mps2"), 3)
        quat = to_float_list(device.get("quaternion_wxyz"), 4)
        for axis, name in enumerate(IMU_RPY_NAMES):
            out[f"{prefix}_rpy_{name}_rad"] = _csv_value(None if rpy_rad is None else rpy_rad[axis])
            out[f"{prefix}_rpy_raw_{name}_deg"] = _csv_value(
                None if rpy_raw_deg is None else rpy_raw_deg[axis]
            )
        for axis, name in enumerate(IMU_AXIS_NAMES):
            out[f"{prefix}_gyro_{name}_dps"] = _csv_value(None if gyro is None else gyro[axis])
            out[f"{prefix}_accel_{name}_mps2"] = _csv_value(None if accel is None else accel[axis])
        for axis, name in enumerate(IMU_QUATERNION_NAMES):
            out[f"{prefix}_quaternion_{name}"] = _csv_value(None if quat is None else quat[axis])
    return out


def deg_to_rad(values: list[float | None] | None) -> list[float | None] | None:
    if values is None:
        return None
    return [None if value is None else value * math.pi / 180.0 for value in values]


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


def summarize_scalar(samples: list[float]) -> dict[str, float | None]:
    vals = [float(v) for v in samples if math.isfinite(float(v))]
    if not vals:
        return {
            "min": None,
            "max": None,
            "span": None,
            "min_deg": None,
            "max_deg": None,
            "span_deg": None,
        }
    vmin = min(vals)
    vmax = max(vals)
    span = vmax - vmin
    return {
        "min": vmin,
        "max": vmax,
        "span": span,
        "min_deg": math.degrees(vmin),
        "max_deg": math.degrees(vmax),
        "span_deg": math.degrees(span),
    }


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


def print_imu_debug(
    imu_debug: dict[str, Any] | None,
    *,
    imu_status: dict[str, Any] | None = None,
) -> None:
    if not isinstance(imu_debug, dict):
        print(
            "  imu_debug: unavailable; per-device samples cannot be inspected",
            flush=True,
        )
        return
    devices = imu_debug.get("devices")
    if not isinstance(devices, list):
        print("  imu_debug.devices: unavailable", flush=True)
        return
    fault_reasons: dict[int, str] = {}
    if isinstance(imu_status, dict):
        raw_fault_devices = imu_status.get("fault_devices")
        if isinstance(raw_fault_devices, list):
            for raw_fault in raw_fault_devices:
                if not isinstance(raw_fault, dict):
                    continue
                try:
                    index = int(raw_fault.get("index"))
                except (TypeError, ValueError):
                    continue
                reasons = raw_fault.get("reasons")
                if isinstance(reasons, list):
                    fault_reasons[index] = ",".join(str(reason) for reason in reasons)
    print(
        "  slot physical-id       addr on att quat(opt) gyro acc    age_ms loss fault",
        flush=True,
    )
    for layout in IMU_LAYOUT:
        index = int(layout["index"])
        raw_device = devices[index] if index < len(devices) else None
        device = raw_device if isinstance(raw_device, dict) else None
        physical_id = f"{layout['name']}@{layout['can_id']}"
        if device is None:
            print(
                f"  imu{index} {physical_id:<17} {'-':>4}  -   -    -    -   - {'n/a':>9} "
                f"{'-':>4} unavailable",
                flush=True,
            )
            continue
        age_ms = _safe_finite_float(device.get("host_rx_age_ms"))
        age_text = "n/a" if age_ms is None else f"{age_ms:.1f}"
        reason_text = fault_reasons.get(index, "ok")
        print(
            f"  imu{index} {physical_id:<17} {str(device.get('device_addr', '-')):>4} "
            f"{str(device.get('online', '-')):>2} "
            f"{str(device.get('valid_attitude', '-')):>3} "
            f"{str(device.get('valid_quaternion', '-')):>4} "
            f"{str(device.get('valid_gyro', '-')):>4} "
            f"{str(device.get('valid_accel', '-')):>3} "
            f"{age_text:>9} "
            f"{str(device.get('packet_loss_count', '-')):>4} {reason_text}",
            flush=True,
        )
        rpy_rad = to_float_list(device.get("rpy_rad"), 3)
        print(
            "       "
            f"gyro_dps={fmt_vec(to_float_list(device.get('gyro_dps'), 3), width=7, precision=2)} "
            f"rpy_deg={fmt_vec(rad_to_deg(rpy_rad), width=8, precision=2)} "
            f"raw_deg={fmt_vec(to_float_list(device.get('rpy_raw_deg'), 3), width=8, precision=2)}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    if args.rate_hz <= 0.0:
        raise SystemExit("--rate-hz must be positive")
    if args.imu_max_stale_ms < 0.0:
        raise SystemExit("--imu-max-stale-ms must be non-negative")
    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    vendor_csv_path = None
    if not args.no_vendor_csv:
        vendor_csv_path = args.vendor_csv_output or output_path.with_name(
            output_path.stem + "_vendor.csv"
        )
        vendor_csv_path.parent.mkdir(parents=True, exist_ok=True)

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
    bucket_qpos_tracker = BucketQuaternionPhaseTracker()
    bucket_tilt_tracker = BucketTiltAccelTracker()
    bucket_gravity_hinge_tracker = BucketGravityHingeTracker(
        reference_rad=bucket_gravity_hinge_outer_reference_rad(),
        policy_offset_rad=bucket_gravity_hinge_policy_offset_rad(),
        median_window=bucket_gravity_hinge_median_window(),
    )
    bucket_tilt_reference_rad = bucket_tilt_outer_reference_rad()
    bucket_tilt_policy_offset_value_rad = bucket_tilt_policy_offset_rad()
    bucket_tilt_first_frame_offset_rad: float | None = None
    window: deque[dict[str, Any]] = deque()
    qvel_samples: list[list[float | None]] = []
    qpos_samples: list[list[float]] = []
    bucket_tilt_samples: list[float] = []
    bucket_tilt_outer_zero_samples: list[float] = []
    bucket_tilt_policy_aligned_samples: list[float] = []
    bucket_tilt_policy_aligned_minus_bridge_samples: list[float] = []
    bucket_tilt_first_aligned_minus_bridge_samples: list[float] = []
    bucket_gravity_hinge_outer_zero_samples: list[float] = []
    bucket_gravity_hinge_policy_aligned_samples: list[float] = []
    bucket_gravity_hinge_policy_aligned_minus_bridge_samples: list[float] = []
    bucket_gravity_hinge_median_outer_zero_samples: list[float] = []
    bucket_gravity_hinge_median_policy_aligned_samples: list[float] = []
    bucket_gravity_hinge_median_policy_aligned_minus_bridge_samples: list[float] = []
    diff_samples: list[list[float | None]] = []
    residual_samples: list[list[float | None]] = []
    raw_imu_samples: list[list[float | None]] = []
    raw_minus_bridge_samples: list[list[float | None]] = []
    branch_alias_bad_rows = 0
    branch_alias_bad_axis_counts = {name: 0 for name in AXES}
    rows_written = 0
    missing_imu_debug = 0
    bucket_profile_counts: dict[str, int] = {}
    bucket_reference_counts: dict[str, int] = {}
    bucket_sign_counts: dict[str, int] = {}
    imu_state_counts: dict[str, int] = {}
    imu_error_code_counts: dict[str, int] = {}
    imu_fault_device_row_counts = {
        str(layout["name"]): 0 for layout in IMU_LAYOUT
    }
    imu_fault_reason_row_counts: dict[str, int] = {}
    max_imu_fault_devices = 0

    print(f"[imu-qvel] reading {args.host}:{args.port} at target {args.rate_hz:.1f} Hz", flush=True)
    print(f"[imu-qvel] writing {output_path}", flush=True)
    if vendor_csv_path is not None:
        print(f"[imu-qvel] vendor CSV {vendor_csv_path}", flush=True)
    print("[imu-qvel] qvel axes: swing, boom, stick, bucket", flush=True)
    print(
        "[imu-qvel] IMU health order: "
        + ", ".join(
            f"imu{layout['index']}={layout['name']}@{layout['can_id']}"
            for layout in IMU_LAYOUT
        ),
        flush=True,
    )

    csv_fh = None
    csv_writer: csv.DictWriter | None = None
    try:
        if vendor_csv_path is not None:
            csv_fh = vendor_csv_path.open("w", encoding="utf-8", newline="")
            csv_writer = csv.DictWriter(csv_fh, fieldnames=_vendor_csv_fieldnames())
            csv_writer.writeheader()
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
                imu_status = classify_imu_status(
                    imu_health,
                    imu_debug,
                    max_stale_ms=args.imu_max_stale_ms,
                )
                imu_state = str(imu_status["state"])
                imu_state_counts[imu_state] = imu_state_counts.get(imu_state, 0) + 1
                for error_code in imu_status.get("error_codes", []):
                    key = str(error_code)
                    imu_error_code_counts[key] = imu_error_code_counts.get(key, 0) + 1
                fault_device_count = imu_status.get("fault_device_count")
                if isinstance(fault_device_count, int):
                    max_imu_fault_devices = max(max_imu_fault_devices, fault_device_count)
                for fault_device in imu_status.get("fault_devices", []):
                    if not isinstance(fault_device, dict):
                        continue
                    device_name = str(fault_device.get("name", "unknown"))
                    imu_fault_device_row_counts[device_name] = (
                        imu_fault_device_row_counts.get(device_name, 0) + 1
                    )
                    for reason in fault_device.get("reasons", []):
                        reason_key = str(reason)
                        imu_fault_reason_row_counts[reason_key] = (
                            imu_fault_reason_row_counts.get(reason_key, 0) + 1
                        )
                actual_bucket_profile = bucket_imu0_profile(imu_debug)
                actual_bucket_reference_rad = bucket_imu0_reference_rad(imu_debug)
                actual_bucket_sign = bucket_imu0_axis_sign(imu_debug)
                bucket_profile_counts[actual_bucket_profile] = (
                    bucket_profile_counts.get(actual_bucket_profile, 0) + 1
                )
                reference_key = f"{actual_bucket_reference_rad:.12g}"
                sign_key = f"{actual_bucket_sign:.12g}"
                bucket_reference_counts[reference_key] = (
                    bucket_reference_counts.get(reference_key, 0) + 1
                )
                bucket_sign_counts[sign_key] = bucket_sign_counts.get(sign_key, 0) + 1
                raw_imu_qvel = gyro_joint_qvel_rad_s(imu_debug)
                qpos_folded_imu = imu_joint_qpos_from_rpy_rad(imu_debug)
                qpos_raw_imu_deg = imu_joint_qpos_raw_deg(
                    imu_debug, bucket_tracker=bucket_qpos_tracker
                )
                qpos_raw_imu = deg_to_rad(qpos_raw_imu_deg)
                bucket_tilt = bucket_tilt_tracker.update(imu_debug)
                bucket_gravity_hinge = bucket_gravity_hinge_tracker.update(imu_debug)
                if bucket_tilt is not None:
                    if bucket_tilt_first_frame_offset_rad is None:
                        bucket_tilt_first_frame_offset_rad = qpos[3] - bucket_tilt["bucket_rad"]
                    outer_zero_rad = bucket_tilt["bucket_rad"] - bucket_tilt_reference_rad
                    policy_aligned_rad = (
                        bucket_tilt["bucket_rad"] + bucket_tilt_policy_offset_value_rad
                    )
                    policy_aligned_minus_bridge_rad = angle_delta_rad(
                        policy_aligned_rad, qpos[3]
                    )
                    first_aligned_rad = (
                        bucket_tilt["bucket_rad"] + bucket_tilt_first_frame_offset_rad
                    )
                    first_aligned_minus_bridge_rad = angle_delta_rad(first_aligned_rad, qpos[3])
                    bucket_tilt = {
                        **bucket_tilt,
                        "reference_rad": bucket_tilt_reference_rad,
                        "reference_deg": math.degrees(bucket_tilt_reference_rad),
                        "policy_offset_rad": bucket_tilt_policy_offset_value_rad,
                        "policy_offset_deg": math.degrees(bucket_tilt_policy_offset_value_rad),
                        "outer_zero_rad": outer_zero_rad,
                        "outer_zero_deg": math.degrees(outer_zero_rad),
                        "policy_aligned_rad": policy_aligned_rad,
                        "policy_aligned_deg": math.degrees(policy_aligned_rad),
                        "policy_aligned_minus_bridge_rad": policy_aligned_minus_bridge_rad,
                        "policy_aligned_minus_bridge_deg": math.degrees(
                            policy_aligned_minus_bridge_rad
                        ),
                        "first_frame_offset_rad": bucket_tilt_first_frame_offset_rad,
                        "first_frame_offset_deg": math.degrees(
                            bucket_tilt_first_frame_offset_rad
                        ),
                        "first_aligned_rad": first_aligned_rad,
                        "first_aligned_deg": math.degrees(first_aligned_rad),
                        "first_aligned_minus_bridge_rad": first_aligned_minus_bridge_rad,
                        "first_aligned_minus_bridge_deg": math.degrees(
                            first_aligned_minus_bridge_rad
                        ),
                    }
                if bucket_gravity_hinge is not None:
                    policy_minus_bridge_rad = angle_delta_rad(
                        bucket_gravity_hinge["policy_aligned_rad"], qpos[3]
                    )
                    median_policy_minus_bridge_rad = angle_delta_rad(
                        bucket_gravity_hinge["median_policy_aligned_rad"], qpos[3]
                    )
                    bucket_gravity_hinge = {
                        **bucket_gravity_hinge,
                        "policy_aligned_minus_bridge_rad": policy_minus_bridge_rad,
                        "policy_aligned_minus_bridge_deg": math.degrees(policy_minus_bridge_rad),
                        "median_policy_aligned_minus_bridge_rad": median_policy_minus_bridge_rad,
                        "median_policy_aligned_minus_bridge_deg": math.degrees(
                            median_policy_minus_bridge_rad
                        ),
                    }
                qpos_policy_minus_raw_imu = qpos_delta_rad(qpos, qpos_raw_imu)
                qpos_policy_minus_raw_imu_deg_direct = vec_minus(rad_to_deg(qpos), qpos_raw_imu_deg)
                branch_alias_bad_axes = branch_alias_axes_from_direct_delta(
                    qpos_policy_minus_raw_imu_deg_direct
                )
                raw_minus_bridge = (
                    [
                        None if raw is None else raw - bridge
                        for raw, bridge in zip(raw_imu_qvel, qvel_bridge)
                    ]
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
                quality["qpos_branch_alias_axes"] = branch_alias_bad_axes

                qvel_samples.append(qvel_bridge)
                qpos_samples.append(qpos)
                if bucket_tilt is not None:
                    bucket_tilt_samples.append(float(bucket_tilt["bucket_rad"]))
                    bucket_tilt_outer_zero_samples.append(float(bucket_tilt["outer_zero_rad"]))
                    bucket_tilt_policy_aligned_samples.append(
                        float(bucket_tilt["policy_aligned_rad"])
                    )
                    bucket_tilt_policy_aligned_minus_bridge_samples.append(
                        float(bucket_tilt["policy_aligned_minus_bridge_rad"])
                    )
                    bucket_tilt_first_aligned_minus_bridge_samples.append(
                        float(bucket_tilt["first_aligned_minus_bridge_rad"])
                    )
                if bucket_gravity_hinge is not None:
                    bucket_gravity_hinge_outer_zero_samples.append(
                        float(bucket_gravity_hinge["outer_zero_rad"])
                    )
                    bucket_gravity_hinge_policy_aligned_samples.append(
                        float(bucket_gravity_hinge["policy_aligned_rad"])
                    )
                    bucket_gravity_hinge_policy_aligned_minus_bridge_samples.append(
                        float(bucket_gravity_hinge["policy_aligned_minus_bridge_rad"])
                    )
                    bucket_gravity_hinge_median_outer_zero_samples.append(
                        float(bucket_gravity_hinge["median_outer_zero_rad"])
                    )
                    bucket_gravity_hinge_median_policy_aligned_samples.append(
                        float(bucket_gravity_hinge["median_policy_aligned_rad"])
                    )
                    bucket_gravity_hinge_median_policy_aligned_minus_bridge_samples.append(
                        float(bucket_gravity_hinge["median_policy_aligned_minus_bridge_rad"])
                    )
                if qvel_diff is not None:
                    diff_samples.append(qvel_diff)
                if residual is not None:
                    residual_samples.append(residual)
                if raw_imu_qvel is not None:
                    raw_imu_samples.append(raw_imu_qvel)
                if raw_minus_bridge is not None:
                    raw_minus_bridge_samples.append(raw_minus_bridge)
                if branch_alias_bad_axes:
                    branch_alias_bad_rows += 1
                    for axis in branch_alias_bad_axes:
                        branch_alias_bad_axis_counts[axis] += 1

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
                    "bucket_imu0_profile": actual_bucket_profile,
                    "bucket_imu0_reference_rad": actual_bucket_reference_rad,
                    "bucket_imu0_sign": actual_bucket_sign,
                    "bucket_tilt_accel": bucket_tilt,
                    "bucket_gravity_hinge_accel": bucket_gravity_hinge,
                    "status": payload.get("status"),
                    "snapshot_age_ms": payload.get("snapshot_age_ms"),
                    "state_loop_tick": payload.get("state_loop_tick"),
                    "imu_health": imu_health,
                    "imu_debug": imu_debug,
                    "imu_status": imu_status,
                    "quality": quality,
                }
                fout.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                if csv_writer is not None:
                    csv_writer.writerow(vendor_csv_row(row))
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
                    if branch_alias_bad_axes:
                        bad_parts.append("branch_alias=" + ",".join(branch_alias_bad_axes))
                    bad = "ok" if not bad_parts else ";".join(bad_parts)
                    print(
                        f"[imu-qvel] t={elapsed_s:7.1f}s hz={actual_hz:5.1f} bad={bad} "
                        f"online={imu_status['online_bits']} "
                        f"att={imu_status['valid_attitude_bits']} "
                        f"quat={imu_status['valid_quaternion_bits']}(opt) "
                        f"gyro={imu_status['valid_gyro_bits']} "
                        f"acc={imu_status['valid_accel_bits']} "
                        f"bucket_profile={actual_bucket_profile} sign={actual_bucket_sign:g}",
                        flush=True,
                    )
                    print("  " + format_imu_status(imu_status), flush=True)
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
                    if bucket_tilt is not None:
                        print(
                            "  "
                            f"{'bucket tilt_deg':<22}"
                            f"raw={bucket_tilt['bucket_deg']:9.2f} "
                            f"outer0={bucket_tilt['outer_zero_deg']:9.2f} "
                            f"policy={bucket_tilt['policy_aligned_deg']:9.2f} "
                            f"policy-bridge={bucket_tilt['policy_aligned_minus_bridge_deg']:9.2f}",
                            flush=True,
                        )
                    if bucket_gravity_hinge is not None:
                        print(
                            "  "
                            f"{'bucket hinge_deg':<22}"
                            f"raw0={bucket_gravity_hinge['outer_zero_deg']:9.2f} "
                            f"med0={bucket_gravity_hinge['median_outer_zero_deg']:9.2f} "
                            f"policy={bucket_gravity_hinge['median_policy_aligned_deg']:9.2f} "
                            f"med-bridge={bucket_gravity_hinge['median_policy_aligned_minus_bridge_deg']:9.2f}",
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
                    if args.verbose_imu or imu_status["state"] != "ok":
                        print_imu_debug(imu_debug, imu_status=imu_status)
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
        if csv_fh is not None:
            csv_fh.flush()
            csv_fh.close()

    summary = {
        "schema_version": 1,
        "output": str(output_path),
        "vendor_csv_output": "" if vendor_csv_path is None else str(vendor_csv_path),
        "rows": rows_written,
        "read_errors": read_errors,
        "missing_imu_debug_rows": missing_imu_debug,
        "target_rate_hz": args.rate_hz,
        "duration_s": (time.monotonic_ns() - started_mono_ns) / 1e9,
        "axes": list(AXES),
        "imu_status": {
            "layout": [dict(layout) for layout in IMU_LAYOUT],
            "max_stale_ms": args.imu_max_stale_ms,
            "state_counts": imu_state_counts,
            "error_code_counts": imu_error_code_counts,
            "fault_device_row_counts": imu_fault_device_row_counts,
            "fault_reason_row_counts": imu_fault_reason_row_counts,
            "max_fault_devices_in_one_row": max_imu_fault_devices,
        },
        "qpos": summarize_qpos(qpos_samples),
        "qvel_bridge": summarize_matrix(qvel_samples),
        "qvel_qpos_diff": summarize_matrix(diff_samples),
        "qvel_residual_bridge_minus_qpos_diff": summarize_matrix(residual_samples),
        "qvel_raw_imu_rad_s": summarize_matrix(raw_imu_samples),
        "qvel_raw_imu_minus_bridge": summarize_matrix(raw_minus_bridge_samples),
        "bucket_tilt_accel_reference_rad": bucket_tilt_reference_rad,
        "bucket_tilt_accel_reference_deg": math.degrees(bucket_tilt_reference_rad),
        "bucket_tilt_accel_policy_offset_rad": bucket_tilt_policy_offset_value_rad,
        "bucket_tilt_accel_policy_offset_deg": math.degrees(
            bucket_tilt_policy_offset_value_rad
        ),
        "bucket_tilt_accel": summarize_scalar(bucket_tilt_samples),
        "bucket_tilt_accel_outer_zero": summarize_scalar(bucket_tilt_outer_zero_samples),
        "bucket_tilt_accel_policy_aligned": summarize_scalar(
            bucket_tilt_policy_aligned_samples
        ),
        "bucket_tilt_accel_policy_aligned_minus_bridge": summarize_scalar(
            bucket_tilt_policy_aligned_minus_bridge_samples
        ),
        "bucket_tilt_accel_first_aligned_minus_bridge": summarize_scalar(
            bucket_tilt_first_aligned_minus_bridge_samples
        ),
        "bucket_gravity_hinge_reference_rad": bucket_gravity_hinge_tracker.reference_rad,
        "bucket_gravity_hinge_reference_deg": math.degrees(
            bucket_gravity_hinge_tracker.reference_rad
        ),
        "bucket_gravity_hinge_policy_offset_rad": bucket_gravity_hinge_tracker.policy_offset_rad,
        "bucket_gravity_hinge_policy_offset_deg": math.degrees(
            bucket_gravity_hinge_tracker.policy_offset_rad
        ),
        "bucket_gravity_hinge_median_window": bucket_gravity_hinge_tracker.median_window,
        "bucket_gravity_hinge_outer_zero": summarize_scalar(
            bucket_gravity_hinge_outer_zero_samples
        ),
        "bucket_gravity_hinge_policy_aligned": summarize_scalar(
            bucket_gravity_hinge_policy_aligned_samples
        ),
        "bucket_gravity_hinge_policy_aligned_minus_bridge": summarize_scalar(
            bucket_gravity_hinge_policy_aligned_minus_bridge_samples
        ),
        "bucket_gravity_hinge_median_outer_zero": summarize_scalar(
            bucket_gravity_hinge_median_outer_zero_samples
        ),
        "bucket_gravity_hinge_median_policy_aligned": summarize_scalar(
            bucket_gravity_hinge_median_policy_aligned_samples
        ),
        "bucket_gravity_hinge_median_policy_aligned_minus_bridge": summarize_scalar(
            bucket_gravity_hinge_median_policy_aligned_minus_bridge_samples
        ),
        "qpos_branch_alias_bad_rows": branch_alias_bad_rows,
        "qpos_branch_alias_bad_axis_counts": branch_alias_bad_axis_counts,
        "bucket_imu0_profile_counts": bucket_profile_counts,
        "bucket_imu0_reference_rad_counts": bucket_reference_counts,
        "bucket_imu0_sign_counts": bucket_sign_counts,
        "bucket_imu0_profile": (
            max(bucket_profile_counts, key=bucket_profile_counts.get)
            if bucket_profile_counts
            else bucket_imu0_profile()
        ),
        "bucket_imu0_reference_rad": (
            float(max(bucket_reference_counts, key=bucket_reference_counts.get))
            if bucket_reference_counts
            else bucket_imu0_reference_rad()
        ),
        "bucket_imu0_sign": (
            float(max(bucket_sign_counts, key=bucket_sign_counts.get))
            if bucket_sign_counts
            else bucket_imu0_axis_sign()
        ),
        "notes": [
            "imu_health and imu_debug use physical order: imu0=bucket@0x122, imu1=stick@0x124, imu2=boom@0x121, imu3=swing@0x123; this differs from policy AXES order.",
            "imu_status counts unique faulty devices per row. Multiple invalid flags on one IMU remain single_imu_error; offline devices are not double-counted as stale or invalid.",
            "valid_quaternion is reported but is optional for the current Daoyuan native-RPY chain, so quaternion=0 alone is not counted as an IMU fault.",
            "imu_snapshot_uninitialized means the bridge has not published a real per-device diagnostic snapshot; it must not be interpreted as four physical IMUs offline.",
            "qvel_bridge is the qvel currently returned by read_state.",
            "qvel_qpos_diff is finite-difference qpos using bridge timestamps.",
            "qpos/qpos_deg are the current joint pose in radians/degrees.",
            "qpos_raw_imu/qpos_raw_imu_deg reconstruct qpos from imu_debug.rpy_raw_deg. With EXCAVATOR_JOINT_RPY_PROFILE=daoyuan_chain: stick=imu1.pitch+imu2.pitch+offset and bucket=-(imu0.roll+imu1.pitch)+offset. Otherwise bucket follows EXCAVATOR_BUCKET_IMU0_PROFILE: legacy_y uses IMU0/IMU1 quaternion Y twist plus the fixed policy offset, roll_ccw90 uses unwrap(imu0.roll - imu1.pitch) - EXCAVATOR_BUCKET_IMU0_REFERENCE_RAD.",
            "qpos_folded_imu/qpos_folded_imu_deg are reconstructed from imu_debug.rpy_rad after per-axis angle folding.",
            "qpos_policy_minus_raw_imu is the shortest-angle delta from raw IMU joint pose to policy qpos.",
            "qpos_policy_minus_raw_imu_deg_direct is policy_deg - raw_imu_deg without branch wrapping.",
            "qpos_branch_alias_* flags policy qpos that differs from raw IMU by an integer 360deg branch.",
            "qvel_raw_imu_rad_s is derived from raw gyro before converter startup bias subtraction.",
            "bucket_tilt_accel is diagnostic only: atan2(imu0.accel_x, imu0.accel_z) - atan2(imu1.accel_y, imu1.accel_z).",
            "bucket_tilt_accel_outer_zero subtracts EXCAVATOR_BUCKET_TILT_REFERENCE_RAD; default is the 2026-07-07 bucket_outer_side calibration.",
            "bucket_tilt_accel_policy_aligned adds EXCAVATOR_BUCKET_TILT_POLICY_OFFSET_RAD to map the tilt signal into the bridge/policy bucket qpos coordinate at that calibration pose; it does not affect qpos.",
            "bucket_tilt_accel_first_aligned_minus_bridge is only a comparison against first-frame alignment and should not be used as the calibrated zero.",
            "bucket_gravity_hinge_* is diagnostic only. It uses accelerometer gravity projection with hinge axes imu0+X and imu1+Y, anchored by EXCAVATOR_BUCKET_GRAVITY_HINGE_REFERENCE_RAD / EXCAVATOR_BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD.",
            "bucket_gravity_hinge_median_* uses a trailing median window, default 21 samples. It is the current best offline candidate, but it does not affect qpos.",
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
