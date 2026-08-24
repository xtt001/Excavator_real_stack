from __future__ import annotations

import socket
import time
from types import SimpleNamespace

import numpy as np

from testbed.cli.record_real import (
    _go_home_config_status_payload,
    _online_qc_status_payload,
    _publish_remote_receiver_status,
    _receiver_health_status_payload,
    _receiver_latency_status_payload,
    _save_status_payload,
    _storage_status_payload,
)
from testbed.actions.remote import encode_remote_receiver_status
from testbed.host_status import (
    LocalHostStatusPublisher,
    build_host_status_snapshot,
    decode_host_status,
    encode_host_status,
)


def test_host_status_protocol_round_trip() -> None:
    payload = {
        "timestamp_ns": 123,
        "sender": {"seq": 9, "action": [0.1, 0.2, 0.3, 0.4]},
        "receiver": {"receiver_mode": "recording", "record_steps": 42},
    }
    assert decode_host_status(encode_host_status(payload)) == payload


def test_local_status_udp_publisher_is_latest_only_and_non_blocking() -> None:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    port = int(receiver.getsockname()[1])
    publisher = LocalHostStatusPublisher(port=port, max_hz=10.0)
    try:
        assert publisher.publish({"seq": 1}, force=True)
        frame, _addr = receiver.recvfrom(65_535)
        assert decode_host_status(frame)["seq"] == 1
        assert not publisher.publish({"seq": 2})
        time.sleep(0.11)
        assert publisher.publish({"seq": 3})
    finally:
        publisher.close()
        receiver.close()


def test_build_host_status_snapshot_carries_sender_and_receiver() -> None:
    receiver_status = SimpleNamespace(
        payload={"receiver_mode": "armed", "saved": 3},
        receive_time_ns=456,
    )
    snapshot = build_host_status_snapshot(
        seq=7,
        target="192.168.100.1:8770",
        configured_hz=50.0,
        input_device="joystick",
        source_id="host",
        action=np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
        action_latency_ms=1.2,
        extras={"status11": [1, 0, 1], "toggle_mask": 4},
        event_flags={"record_start": False},
        receiver_status=receiver_status,
    )
    assert snapshot["sender"]["seq"] == 7
    assert snapshot["sender"]["status11"] == [1, 0, 1]
    assert snapshot["receiver_available"] == 1
    assert snapshot["receiver"]["receiver_mode"] == "armed"


def test_receiver_latency_payload_reports_only_current_action_age() -> None:
    action_info = SimpleNamespace(
        extras={
            "policy_remote_mode": "policy",
            "model_control": 1,
            "policy_inference_latency_ms": 18.5,
            "action_timestamp_ns": 1_000_000_000,
        }
    )
    payload = _receiver_latency_status_payload(
        action_info=action_info,
        control_result={
            "action_sample_timestamp_ns": 1_000_000_000,
            "controller_timestamp_ns": 1_030_000_000,
        },
        status_time_ns=1_040_000_000,
    )
    assert payload == {
        "action_age_ms": 40.0,
        "action_source": "policy",
        "definition": "receiver status time minus current action sample time",
    }


def test_receiver_health_payload_has_per_imu_fields() -> None:
    health = SimpleNamespace(
        ok=True,
        error_code="",
        errors=(),
        imu_summary="1111",
        diagnostics={
            "imu_online": np.ones(4, dtype=np.int32),
            "imu_valid_attitude": np.ones(4, dtype=np.int32),
            "imu_packet_loss_count": np.asarray([0, 1, 2, 3], dtype=np.int64),
            "imu_host_rx_age_ms": np.asarray([1.0, 2.0, 3.0, 4.0]),
            "bridge_snapshot_age_ms": 0.0,
            "camera_primary": "video4",
            "camera_age_ms": 6.0,
            "remote_action_connected": 1,
            "controller_ack": 1,
        },
    )
    payload = _receiver_health_status_payload(health)
    assert payload["ok"] == 1
    assert payload["imu"]["online"] == [1, 1, 1, 1]
    assert payload["imu"]["packet_loss_count"] == [0, 1, 2, 3]
    assert payload["camera_primary"] == "video4"
    assert payload["bridge_snapshot_age_ms"] == 0.0


def test_online_qc_payload_is_compact_and_jsonable() -> None:
    snapshot = SimpleNamespace(
        status="WARN_MASK",
        error_code="",
        warning_codes=("fpv_drift",),
        train_exclude=True,
        diagnostics={
            "online_qc_total_steps": 20,
            "online_qc_healthy_steps": 18,
            "online_qc_healthy_fraction": np.float32(0.9),
            "unrelated_large_field": np.zeros((100, 100)),
        },
    )
    payload = _online_qc_status_payload(snapshot)
    assert payload["status"] == "WARN_MASK"
    assert payload["warning_codes"] == ["fpv_drift"]
    assert payload["healthy_steps"] == 18
    assert "unrelated_large_field" not in payload


def test_save_status_reports_completed_file_size(tmp_path) -> None:
    saved = tmp_path / "episode_1.hdf5"
    saved.write_bytes(b"123456")
    payload = _save_status_payload(
        {"state": "success", "path": str(saved), "success": 1},
        saved_path=None,
    )
    assert payload["state"] == "success"
    assert payload["file_size_bytes"] == 6


def test_storage_status_counts_completed_and_failed_records(tmp_path) -> None:
    (tmp_path / "episode_0.hdf5").write_bytes(b"0")
    (tmp_path / "episode_3.hdf5").write_bytes(b"3")
    (tmp_path / "episode_bad.hdf5").write_bytes(b"bad")
    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "episode_2_failed_20260713.hdf5").write_bytes(b"failed")

    payload = _storage_status_payload(tmp_path)

    assert payload["recorded_count"] == 2
    assert payload["last_recorded_episode_idx"] == 3
    assert payload["failed_count"] == 1


def test_home_status_reports_runtime_calibration_contract(tmp_path) -> None:
    config = SimpleNamespace(
        home_pose_rad=np.asarray([0.1, -0.2, 0.3, -0.4]),
        near_tolerance_rad=np.asarray([0.5, 0.5, 0.5, 0.5]),
        success_tolerance_rad=np.asarray([0.05, 0.06, 0.07, 0.08]),
        center_tolerance_rad=np.asarray([0.02, 0.03, 0.04, 0.05]),
        timeout_s=8.0,
        dwell_s=0.4,
    )
    payload = _go_home_config_status_payload(
        config,
        source_path=tmp_path / "runtime.yaml",
        phase_home_pose=[0.1, -0.2, 0.3, -0.4],
    )

    assert payload["enabled"] == 1
    assert payload["home_pose_rad"] == [0.1, -0.2, 0.3, -0.4]
    assert payload["success_tolerance_rad"] == [0.05, 0.06, 0.07, 0.08]
    assert payload["runtime_phase_consistent"] == 1


def test_structured_receiver_status_is_remote_protocol_jsonable(tmp_path) -> None:
    class Sink:
        payload = None

        def publish_status(self, payload):
            self.payload = payload

    sink = Sink()
    health = SimpleNamespace(
        ok=True,
        error_code="",
        errors=(),
        imu_summary="1111",
        diagnostics={
            "imu_online": np.ones(4, dtype=np.int32),
            "imu_valid_attitude": np.ones(4, dtype=np.int32),
            "imu_packet_loss_count": np.zeros(4, dtype=np.int64),
            "imu_host_rx_age_ms": np.ones(4, dtype=np.float32),
            "bridge_snapshot_age_ms": 2.0,
            "camera_primary": "video4",
            "camera_age_ms": 3.0,
            "remote_action_connected": 1,
            "controller_ack": 1,
        },
    )
    _publish_remote_receiver_status(
        sink,
        receiver_mode="recording",
        episode_idx=2,
        saved=1,
        record_session=None,
        receiver_health=health,
        record_steps=25,
        storage_path=tmp_path,
        save_status={"state": "idle"},
        home_status={"enabled": 1, "home_pose_rad": [0.1, 0.2, 0.3, 0.4]},
        real_transition_status={
            "session_id": "rt_test",
            "phase": "ready",
            "next_target_side": "B",
        },
    )
    assert sink.payload is not None
    assert sink.payload["health"]["imu"]["online"] == [1, 1, 1, 1]
    assert sink.payload["home"]["enabled"] == 1
    assert sink.payload["real_transition"] == {
        "session_id": "rt_test",
        "phase": "ready",
        "next_target_side": "B",
    }
    assert encode_remote_receiver_status(sink.payload).endswith(b"\n")


def test_saving_status_marks_recording_as_finished(tmp_path) -> None:
    class Sink:
        payload = None

        def publish_status(self, payload):
            self.payload = payload

    sink = Sink()
    _publish_remote_receiver_status(
        sink,
        receiver_mode="saving",
        episode_idx=4,
        saved=3,
        record_session=object(),
        record_steps=8230,
        storage_path=tmp_path,
        save_status={
            "state": "writing",
            "episode_idx": 4,
            "steps": 8230,
            "started_ns": time.time_ns(),
        },
    )

    assert sink.payload is not None
    assert sink.payload["receiver_mode"] == "saving"
    assert sink.payload["recording"] == 0
    assert sink.payload["save"]["state"] == "writing"
