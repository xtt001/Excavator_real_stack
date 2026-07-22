from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "log_imu_qvel_quality.py"
)
SPEC = importlib.util.spec_from_file_location("log_imu_qvel_quality_status", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
LOGGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOGGER)


def _health(
    *,
    online: list[int] | None = None,
    attitude: list[int] | None = None,
    gyro: list[int] | None = None,
    accel: list[int] | None = None,
    ages_ms: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "online": [1, 1, 1, 1] if online is None else online,
        "valid_attitude": [1, 1, 1, 1] if attitude is None else attitude,
        "valid_quaternion": [0, 0, 0, 0],
        "valid_gyro": [1, 1, 1, 1] if gyro is None else gyro,
        "valid_accel": [1, 1, 1, 1] if accel is None else accel,
        "host_rx_age_ms": [1.0, 1.0, 1.0, 1.0] if ages_ms is None else ages_ms,
    }


def _debug(*, initialized: bool = True) -> dict[str, Any]:
    return {
        "joint_rpy_profile": "daoyuan_chain",
        "devices": [
            {
                "device_addr": index + 1 if initialized else 0,
                "host_rx_time_ns": 1_000_000 + index if initialized else 0,
                "host_rx_age_ms": 1.0,
                "online": 1,
                "valid_attitude": 1,
                "valid_quaternion": 0,
                "valid_gyro": 1,
                "valid_accel": 1,
                "packet_loss_count": 0,
                "rpy_rad": [0.1 + index, 0.2 + index, 0.3 + index],
                "rpy_raw_deg": [10.0 + index, 20.0 + index, 30.0 + index],
                "gyro_dps": [1.0 + index, 2.0 + index, 3.0 + index],
                "accel_mps2": [1.0 + index, 2.0 + index, 9.0 + index],
            }
            for index in range(4)
        ],
    }


def test_healthy_imu_status_is_ok() -> None:
    status = LOGGER.classify_imu_status(_health(), _debug())

    assert status["state"] == "ok"
    assert status["fault_device_count"] == 0
    assert status["error_codes"] == []
    assert status["online_bits"] == "1111"
    assert status["valid_quaternion_bits"] == "0000"


def test_imu_status_reports_one_device_once_for_multiple_invalid_flags() -> None:
    status = LOGGER.classify_imu_status(
        _health(
            attitude=[0, 1, 1, 1],
            gyro=[0, 1, 1, 1],
            accel=[0, 1, 1, 1],
        ),
        _debug(),
    )

    assert status["state"] == "single_imu_error"
    assert status["fault_device_count"] == 1
    assert status["fault_indices"] == [0]
    assert status["fault_devices"][0]["label"] == "imu0:bucket@0x122"
    assert status["fault_devices"][0]["reasons"] == [
        "attitude_invalid",
        "gyro_invalid",
        "accel_invalid",
    ]


def test_imu_status_reports_multiple_unique_devices_without_double_counting() -> None:
    status = LOGGER.classify_imu_status(
        _health(
            online=[0, 1, 1, 1],
            attitude=[0, 1, 0, 1],
            gyro=[0, 1, 1, 1],
            accel=[0, 1, 1, 1],
            ages_ms=[-1.0, 1.0, 150.0, 1.0],
        ),
        _debug(),
        max_stale_ms=100.0,
    )

    assert status["state"] == "multiple_imu_errors"
    assert status["fault_device_count"] == 2
    assert status["fault_indices"] == [0, 2]
    assert status["missing_indices"] == [0]
    assert status["stale_indices"] == [2]
    assert status["invalid_attitude_indices"] == []
    assert status["error_codes"] == ["imu_missing:0", "imu_stale:2"]


def test_default_bridge_snapshot_is_not_reported_as_four_missing_imus() -> None:
    status = LOGGER.classify_imu_status(
        _health(
            online=[0, 0, 0, 0],
            attitude=[0, 0, 0, 0],
            gyro=[0, 0, 0, 0],
            accel=[0, 0, 0, 0],
            ages_ms=[-1.0, -1.0, -1.0, -1.0],
        ),
        _debug(initialized=False),
    )

    assert status["state"] == "imu_snapshot_uninitialized"
    assert status["fault_device_count"] is None
    assert status["missing_indices"] == []
    assert status["error_codes"] == ["imu_snapshot_uninitialized"]


def test_missing_health_payload_is_reported_unavailable_not_multi_imu_error() -> None:
    status = LOGGER.classify_imu_status(None, None)

    assert status["state"] == "imu_health_unavailable"
    assert status["fault_device_count"] is None
    assert status["fault_indices"] == []


def test_offline_imu_values_are_not_used_for_derived_axes() -> None:
    debug = _debug()
    bucket = debug["devices"][0]
    bucket.update(
        {
            "online": 0,
            "valid_attitude": 0,
            "valid_gyro": 0,
            "valid_accel": 0,
        }
    )

    qvel = LOGGER.gyro_joint_qvel_rad_s(debug)
    qpos_folded = LOGGER.imu_joint_qpos_from_rpy_rad(debug)
    qpos_raw_deg = LOGGER.imu_joint_qpos_raw_deg(
        debug,
        bucket_tracker=LOGGER.BucketQuaternionPhaseTracker(),
    )

    assert qvel is not None and all(value is not None for value in qvel[:3])
    assert qvel[3] is None
    assert qpos_folded is not None and all(
        value is not None for value in qpos_folded[:3]
    )
    assert qpos_folded[3] is None
    assert qpos_raw_deg is not None and all(
        value is not None for value in qpos_raw_deg[:3]
    )
    assert qpos_raw_deg[3] is None
    assert LOGGER.BucketTiltAccelTracker().update(debug) is None
    assert (
        LOGGER.BucketGravityHingeTracker(
            reference_rad=0.0,
            policy_offset_rad=0.0,
            median_window=3,
        ).update(debug)
        is None
    )
    assert "n/a" in LOGGER.fmt_axis_row("qvel", qvel)


def test_vendor_csv_keeps_healthy_derived_axes_when_bucket_imu_is_missing() -> None:
    row = {
        "schema_version": 1,
        "imu_status": LOGGER.classify_imu_status(
            _health(
                online=[0, 1, 1, 1],
                attitude=[0, 1, 1, 1],
                gyro=[0, 1, 1, 1],
                accel=[0, 1, 1, 1],
                ages_ms=[-1.0, 1.0, 1.0, 1.0],
            ),
            _debug(),
        ),
        "qvel_raw_imu_rad_s": [0.1, 0.2, 0.3, None],
        "qpos_raw_imu": [1.0, 2.0, 3.0, None],
        "qpos_raw_imu_deg": [10.0, 20.0, 30.0, None],
        "qpos_folded_imu": [1.0, 2.0, 3.0, None],
        "qpos_folded_imu_deg": [10.0, 20.0, 30.0, None],
    }

    csv_row = LOGGER.vendor_csv_row(row)

    assert csv_row["imu_state"] == "single_imu_error"
    assert csv_row["imu_fault_device_count"] == 1
    assert csv_row["imu_fault_devices"] == "imu0:bucket@0x122"
    assert csv_row["qvel_raw_imu_swing_rad_s"] == pytest.approx(0.1)
    assert csv_row["qvel_raw_imu_bucket_rad_s"] == ""
