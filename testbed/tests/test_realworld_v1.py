from __future__ import annotations

import importlib.util
import io
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from testbed.backends.real import (
    BridgeLowLevelController,
    BridgeStateReader,
    ControlResult,
    ExcavatorApiPacketAdapter,
    GoHomeConfig,
    GoHomeController,
    InProcessMockBridgeClient,
    JsonTcpBridgeClient,
    JsonTcpBridgeMockServer,
    LowLevelController,
    MockLowLevelController,
    MockStateReader,
    NoopLowLevelController,
    RealActionPump,
    RealExcavatorBackend,
    RealStateSamples,
    SynchronizedObservationBuilder,
    TimestampedBuffer,
    TimestampedSample,
    action4_to_speed_scalar8,
)
from testbed.actions.oem_remote import OemRemoteActionSource, OemRemoteUnavailableError
from testbed.actions.remote import (
    RemoteActionClient,
    RemoteActionProtocolError,
    RemoteActionSource,
    decode_remote_action_update,
    decode_remote_receiver_status,
    encode_remote_action_update,
    encode_remote_receiver_status,
)
from testbed.backends.real.bridge_protocol import (
    control_result_from_payload,
    control_result_to_payload,
    decode_frame,
    encode_frame,
    response_message,
    state_samples_from_payload,
    state_samples_to_payload,
)
from testbed.backends.real.contracts import (
    STATUS_TOGGLE_BIT_COUNT,
    apply_status_toggle_mask_to_status11,
    real_qpos_error_rad,
)
from testbed.backends.real.excavator_api import SERVO_MAGIC, SERVO_PACKET_STRUCT
from testbed.backends.real.ros_can import RosCanLowLevelController, RosCanStateReader
from testbed.runtime.guard import ActionGuard


HAS_H5PY = importlib.util.find_spec("h5py") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_CV2 = importlib.util.find_spec("cv2") is not None


def _can_bind_loopback_socket() -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        return False
    try:
        sock.bind(("127.0.0.1", 0))
        return True
    except PermissionError:
        return False
    finally:
        sock.close()


def _wait_for_remote_seq(
    source: RemoteActionSource,
    seq: int,
    *,
    timeout_s: float = 1.0,
) -> tuple[np.ndarray, object]:
    deadline = time.monotonic() + timeout_s
    last_action: np.ndarray | None = None
    last_info: object | None = None
    while time.monotonic() < deadline:
        last_action, last_info = source.next_action({})
        extras = getattr(last_info, "extras", {}) or {}
        if int(extras.get("remote_action_seq", -1)) == int(seq):
            return last_action, last_info
        time.sleep(0.01)
    raise AssertionError(
        f"timed out waiting for remote_action_seq={seq}; "
        f"last={getattr(last_info, 'extras', {}) if last_info is not None else None}"
    )


def _wait_until_remote_seq_stored(
    source: RemoteActionSource,
    seq: int,
    *,
    timeout_s: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with source._lock:  # noqa: SLF001 - white-box race-free protocol test.
            latest = source._latest  # noqa: SLF001
        if latest is not None and int(latest.packet.seq) == int(seq):
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for stored remote_action_seq={seq}")


class RealworldV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self._bucket_offset_patch = patch.dict(
            os.environ,
            {"EXCAVATOR_BUCKET_QUATERNION_OFFSET_RAD": "0.0"},
        )
        self._bucket_offset_patch.start()

    def tearDown(self) -> None:
        self._bucket_offset_patch.stop()

    def test_remote_action_protocol_round_trip(self) -> None:
        frame = encode_remote_action_update(
            seq=7,
            action=np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
            host_sample_time_ns=123456,
            source_id="host_pad",
            toggle_mask=3,
            reset_requested=True,
            record_start_requested=True,
            policy_start_requested=True,
            go_home_requested=True,
        )

        packet = decode_remote_action_update(frame)
        self.assertEqual(packet.seq, 7)
        np.testing.assert_allclose(
            packet.action,
            np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
        )
        self.assertEqual(packet.host_sample_time_ns, 123456)
        self.assertEqual(packet.source_id, "host_pad")
        self.assertEqual(packet.toggle_mask, 3)
        self.assertTrue(packet.reset_requested)
        self.assertTrue(packet.record_start_requested)
        self.assertTrue(packet.policy_start_requested)
        self.assertTrue(packet.go_home_requested)

    def test_remote_action_protocol_rejects_bad_action_shape(self) -> None:
        with self.assertRaises(RemoteActionProtocolError):
            decode_remote_action_update(
                b'{"version":1,"type":"remote_action.update",'
                b'"payload":{"seq":0,"action":[1,2],"host_sample_time_ns":1}}'
            )

    def test_remote_action_protocol_defaults_go_home_false_for_old_payloads(self) -> None:
        packet = decode_remote_action_update(
            b'{"version":1,"type":"remote_action.update",'
            b'"payload":{"seq":0,"action":[0,0,0,0],'
            b'"host_sample_time_ns":1,"source_id":"old"}}'
        )
        self.assertFalse(packet.go_home_requested)
        self.assertFalse(packet.policy_start_requested)

    def test_remote_receiver_status_protocol_round_trip(self) -> None:
        frame = encode_remote_receiver_status(
            {
                "receiver_mode": "armed",
                "recording": 0,
                "episode_idx": 3,
                "saved": 2,
                "go_home_result": "done",
            }
        )

        payload = decode_remote_receiver_status(frame)
        self.assertEqual(payload["receiver_mode"], "armed")
        self.assertEqual(payload["recording"], 0)
        self.assertEqual(payload["episode_idx"], 3)
        self.assertEqual(payload["saved"], 2)
        self.assertEqual(payload["go_home_result"], "done")

    def test_remote_action_source_receives_latest_and_edges_once(self) -> None:
        if not _can_bind_loopback_socket():
            self.skipTest("loopback socket bind is blocked in this environment")
        source = RemoteActionSource(bind_host="127.0.0.1", port=0, timeout_ms=1000)
        client = RemoteActionClient(host="127.0.0.1", port=source.port, timeout_s=1.0)
        try:
            client.send_update(
                seq=0,
                action=np.array([0.2, 0.0, -0.3, 0.4], dtype=np.float32),
                host_sample_time_ns=111,
                source_id="unit",
                toggle_mask=5,
                reset_requested=True,
                record_start_requested=True,
                go_home_requested=True,
            )
            action, info = _wait_for_remote_seq(source, 0)
            np.testing.assert_allclose(action, [0.2, 0.0, -0.3, 0.4])
            extras = info.extras
            self.assertEqual(extras["toggle_mask"], 5)
            self.assertTrue(extras["reset_requested"])
            self.assertTrue(extras["record_start_requested"])
            self.assertTrue(extras["go_home_requested"])
            self.assertEqual(extras["remote_action_host_sample_ns"], 111)
            self.assertEqual(extras["remote_action_stale"], 0)

            _action_again, info_again = source.next_action({})
            self.assertEqual(info_again.extras["toggle_mask"], 0)
            self.assertFalse(info_again.extras["reset_requested"])
            self.assertFalse(info_again.extras["record_start_requested"])
            self.assertFalse(info_again.extras["go_home_requested"])
        finally:
            client.close()
            source.close()

    def test_remote_receiver_status_feedback_reaches_client(self) -> None:
        if not _can_bind_loopback_socket():
            self.skipTest("loopback socket bind is blocked in this environment")
        source = RemoteActionSource(bind_host="127.0.0.1", port=0, timeout_ms=1000)
        client = RemoteActionClient(host="127.0.0.1", port=source.port, timeout_s=1.0)
        try:
            client.connect()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                with source._lock:  # noqa: SLF001 - white-box connection test.
                    connected = source._connected  # noqa: SLF001
                if connected:
                    break
                time.sleep(0.01)
            source.publish_status(
                {
                    "receiver_mode": "armed",
                    "recording": 0,
                    "episode_idx": 4,
                    "saved": 3,
                    "go_home_result": "done",
                }
            )
            deadline = time.monotonic() + 1.0
            status = None
            while time.monotonic() < deadline:
                status = client.latest_status()
                if status is not None:
                    break
                time.sleep(0.01)
            self.assertIsNotNone(status)
            assert status is not None
            self.assertEqual(status.payload["receiver_mode"], "armed")
            self.assertEqual(status.payload["recording"], 0)
            self.assertEqual(status.payload["episode_idx"], 4)
            self.assertEqual(status.payload["saved"], 3)
            self.assertEqual(status.payload["go_home_result"], "done")
        finally:
            client.close()
            source.close()

    def test_remote_action_source_preserves_edge_when_action_is_overwritten(self) -> None:
        if not _can_bind_loopback_socket():
            self.skipTest("loopback socket bind is blocked in this environment")
        source = RemoteActionSource(bind_host="127.0.0.1", port=0, timeout_ms=1000)
        client = RemoteActionClient(host="127.0.0.1", port=source.port, timeout_s=1.0)
        try:
            client.send_update(
                seq=0,
                action=np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
                host_sample_time_ns=111,
                source_id="unit",
                toggle_mask=1 << 4,
            )
            client.send_update(
                seq=1,
                action=np.array([0.7, 0.2, -0.1, 0.3], dtype=np.float32),
                host_sample_time_ns=112,
                source_id="unit",
            )
            _wait_until_remote_seq_stored(source, 1)

            action, info = source.next_action({})
            np.testing.assert_allclose(action, [0.7, 0.2, -0.1, 0.3])
            self.assertEqual(info.extras["remote_action_seq"], 1)
            self.assertEqual(info.extras["toggle_mask"], 1 << 4)

            _action_again, info_again = source.next_action({})
            self.assertEqual(info_again.extras["toggle_mask"], 0)
        finally:
            client.close()
            source.close()

    def test_remote_action_source_drops_old_seq_and_times_out_to_zero(self) -> None:
        if not _can_bind_loopback_socket():
            self.skipTest("loopback socket bind is blocked in this environment")
        source = RemoteActionSource(bind_host="127.0.0.1", port=0, timeout_ms=20)
        client = RemoteActionClient(host="127.0.0.1", port=source.port, timeout_s=1.0)
        try:
            client.send_update(
                seq=1,
                action=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                host_sample_time_ns=1,
                source_id="unit",
            )
            _wait_for_remote_seq(source, 1)
            client.send_update(
                seq=1,
                action=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                host_sample_time_ns=2,
                source_id="unit",
            )
            time.sleep(0.05)
            action, info = source.next_action({})
            np.testing.assert_allclose(action, np.zeros(4, dtype=np.float32))
            self.assertEqual(info.extras["remote_action_seq"], 1)
            self.assertEqual(info.extras["remote_action_stale"], 1)
            self.assertGreaterEqual(info.extras["remote_action_drop_count"], 1)
        finally:
            client.close()
            source.close()

    def test_remote_action_diagnostics_are_recorded(self) -> None:
        from testbed.cli.record_real import (
            ReceiverHealthSnapshot,
            _build_step_diagnostics,
        )

        guard = SimpleNamespace(
            last_info=SimpleNamespace(triggered=False, reasons=()),
        )
        receiver_health = ReceiverHealthSnapshot(
            ok=True,
            error_code="",
            errors=(),
            imu_summary="1111",
            diagnostics={
                "receiver_health_ok": 1,
                "receiver_health_error_code": "",
                "imu_online": np.ones(4, dtype=np.int32),
                "imu_valid_attitude": np.ones(4, dtype=np.int32),
                "fpv_age_ms": 12.0,
                "bridge_snapshot_age_ms": 8.0,
                "remote_action_connected": 1,
                "controller_ack": 1,
            },
        )
        action_info = SimpleNamespace(
            latency_ms=12.5,
            extras={
                "remote_action_seq": 4,
                "remote_action_host_sample_ns": 100,
                "remote_action_receive_ns": 150,
                "remote_action_age_ms": 2.5,
                "remote_action_stale": 0,
                "remote_action_drop_count": 1,
                "remote_action_connected": 1,
            },
        )
        diagnostics = _build_step_diagnostics(
            obs={
                "timestamp_ns": 1,
                "sensor_timestamp_ns": 2,
                "joint_timestamp_ns": 3,
                "image_timestamp_ns": {"fpv": 4},
                "sync_timestamp_ns": 5,
                "status": np.arange(12, dtype=np.int32),
                "motor_rpm": np.arange(8, dtype=np.float32),
                "plan_rpm": np.arange(8, dtype=np.float32) + 10,
            },
            raw_action=np.zeros(4, dtype=np.float32),
            safe_action=np.zeros(4, dtype=np.float32),
            action_info=action_info,
            action_sample_timestamp_ns=150,
            action_send_timestamp_ns=200,
            guard=guard,
            control_result={"ack": True, "commanded_action": np.zeros(4)},
            receiver_health=receiver_health,
        )

        self.assertEqual(diagnostics["remote_action_seq"], 4)
        self.assertEqual(diagnostics["remote_action_host_sample_ns"], 100)
        self.assertEqual(diagnostics["remote_action_receive_ns"], 150)
        self.assertAlmostEqual(diagnostics["remote_action_age_ms"], 2.5)
        self.assertEqual(diagnostics["remote_action_stale"], 0)
        self.assertEqual(diagnostics["remote_action_drop_count"], 1)
        self.assertEqual(diagnostics["receiver_health_ok"], 1)
        self.assertEqual(diagnostics["receiver_health_error_code"], "")
        np.testing.assert_array_equal(diagnostics["imu_online"], np.ones(4, dtype=np.int32))
        np.testing.assert_array_equal(
            diagnostics["machine_status"], np.arange(12, dtype=np.int32)
        )
        np.testing.assert_allclose(diagnostics["motor_rpm"], np.arange(8, dtype=np.float32))
        np.testing.assert_allclose(
            diagnostics["plan_rpm"], np.arange(8, dtype=np.float32) + 10
        )

        diagnostics_with_raw = _build_step_diagnostics(
            obs={},
            raw_action=np.zeros(4, dtype=np.float32),
            safe_action=np.zeros(4, dtype=np.float32),
            action_info=SimpleNamespace(extras={}),
            action_sample_timestamp_ns=0,
            action_send_timestamp_ns=0,
            guard=guard,
            control_result={
                "ack": True,
                "commanded_action": np.zeros(4),
                "raw_low_level_command": np.arange(8, dtype=np.float32),
            },
        )
        np.testing.assert_allclose(
            diagnostics_with_raw["raw_low_level_command"],
            np.arange(8, dtype=np.float32),
        )

    def test_receiver_health_evaluator_strict_ok_and_errors(self) -> None:
        from testbed.cli.record_real import ReceiverHealthEvaluator

        now_ns = 2_000_000_000
        evaluator = ReceiverHealthEvaluator(
            require_machine_health=True,
            require_remote_action=True,
            bridge_snapshot_timeout_ms=200,
            fpv_max_stale_ms=1000,
        )

        def healthy_obs() -> dict:
            return {
                "image_timestamp_ns": {"fpv": now_ns - 20_000_000},
                "sensor_health": {
                    "bridge_snapshot_age_ms": 10.0,
                    "imu": {
                        "online": [1, 1, 1, 1],
                        "valid_attitude": [1, 1, 1, 1],
                    },
                },
            }

        healthy_info = SimpleNamespace(
            extras={"remote_action_connected": 1, "remote_action_stale": 0}
        )
        healthy_control = {"ack": True, "fault_code": ""}
        snapshot = evaluator.evaluate(
            obs=healthy_obs(),
            action_info=healthy_info,
            control_result=healthy_control,
            now_ns=now_ns,
        )
        self.assertTrue(snapshot.ok)
        self.assertEqual(snapshot.error_code, "")
        self.assertEqual(snapshot.imu_summary, "1111")

        obs = healthy_obs()
        obs["sensor_health"]["imu"]["online"] = [1, 0, 1, 1]
        snapshot = evaluator.evaluate(
            obs=obs,
            action_info=healthy_info,
            control_result=healthy_control,
            now_ns=now_ns,
        )
        self.assertFalse(snapshot.ok)
        self.assertEqual(snapshot.error_code, "imu_missing:1")
        self.assertEqual(snapshot.imu_summary, "1011")

        cases = [
            (
                "fpv_stale",
                {
                    **healthy_obs(),
                    "image_timestamp_ns": {"fpv": now_ns - 2_000_000_000},
                },
                healthy_info,
                healthy_control,
            ),
            (
                "bridge_stale",
                {
                    **healthy_obs(),
                    "sensor_health": {
                        **healthy_obs()["sensor_health"],
                        "bridge_snapshot_age_ms": 250.0,
                    },
                },
                healthy_info,
                healthy_control,
            ),
            (
                "remote_stale",
                healthy_obs(),
                SimpleNamespace(
                    extras={"remote_action_connected": 1, "remote_action_stale": 1}
                ),
                healthy_control,
            ),
            (
                "control_fault",
                healthy_obs(),
                healthy_info,
                {"ack": False, "fault_code": "write_failed"},
            ),
        ]
        for expected, obs_case, info_case, control_case in cases:
            with self.subTest(expected=expected):
                snapshot = evaluator.evaluate(
                    obs=obs_case,
                    action_info=info_case,
                    control_result=control_case,
                    now_ns=now_ns,
                )
                self.assertFalse(snapshot.ok)
                self.assertEqual(snapshot.error_code, expected)

    def test_online_qc_warns_when_qpos_stays_outside_p5_p95(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=_online_qc_reference(),
                qpos_warn_consecutive_steps=5,
                qpos_fail_consecutive_steps=25,
                fpv_sample_interval_steps=999,
            )
        )

        snapshot = None
        for _ in range(5):
            snapshot = evaluator.evaluate(
                obs=_online_qc_obs(qpos=[1.2, 0.0, 0.0, 0.0]),
                now_ns=2_000_000_000,
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.status, "WARN_MASK")
        self.assertEqual(snapshot.error_code, "")
        self.assertIn("qpos_outside_p5_p95", snapshot.warning_codes)
        self.assertTrue(snapshot.train_exclude)
        self.assertEqual(snapshot.diagnostics["online_qc_status"], "WARN_MASK")

    def test_online_qc_masks_when_qpos_stays_outside_p1_p99_by_default(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=_online_qc_reference(),
                qpos_warn_consecutive_steps=5,
                qpos_fail_consecutive_steps=25,
                fpv_sample_interval_steps=999,
            )
        )

        snapshot = None
        for _ in range(25):
            snapshot = evaluator.evaluate(
                obs=_online_qc_obs(qpos=[2.2, 0.0, 0.0, 0.0]),
                now_ns=2_000_000_000,
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.status, "WARN_MASK")
        self.assertEqual(snapshot.error_code, "")
        self.assertIn("qpos_outside_p1_p99", snapshot.warning_codes)
        self.assertTrue(snapshot.train_exclude)

    def test_online_qc_can_hard_fail_persistent_qpos_distribution_outlier(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=_online_qc_reference(),
                qpos_distribution_hard_fail=True,
                qpos_fail_consecutive_steps=25,
                fpv_sample_interval_steps=999,
            )
        )

        snapshot = None
        for _ in range(25):
            snapshot = evaluator.evaluate(
                obs=_online_qc_obs(qpos=[2.2, 0.0, 0.0, 0.0]),
                now_ns=2_000_000_000,
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.status, "FAIL_EPISODE")
        self.assertEqual(snapshot.error_code, "qpos_outside_p1_p99")
        self.assertTrue(snapshot.train_exclude)

    def test_online_qc_final_fails_short_episode(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=_online_qc_reference(),
                min_episode_steps=100,
                min_healthy_steps=80,
                fpv_sample_interval_steps=999,
            )
        )

        for _ in range(65):
            evaluator.evaluate(
                obs=_online_qc_obs(qpos=[0.0, 0.0, 0.0, 0.0]),
                now_ns=2_000_000_000,
            )

        snapshot = evaluator.finalize_episode()

        self.assertEqual(snapshot.status, "FAIL_EPISODE")
        self.assertEqual(snapshot.error_code, "episode_too_short")
        self.assertTrue(snapshot.train_exclude)
        self.assertEqual(snapshot.diagnostics["online_qc_train_ready_candidate"], 0)

    def test_online_qc_final_fails_when_healthy_window_is_insufficient(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=_online_qc_reference(),
                min_episode_steps=50,
                min_healthy_steps=80,
                min_healthy_fraction=0.60,
                fpv_sample_interval_steps=999,
            )
        )

        for _ in range(129):
            evaluator.evaluate(
                obs=_online_qc_obs(qpos=[1.2, 0.0, 0.0, 0.0]),
                now_ns=2_000_000_000,
            )

        snapshot = evaluator.finalize_episode()

        self.assertEqual(snapshot.status, "FAIL_EPISODE")
        self.assertEqual(snapshot.error_code, "insufficient_healthy_steps")
        self.assertEqual(snapshot.diagnostics["online_qc_healthy_steps"], 4)
        self.assertEqual(snapshot.diagnostics["online_qc_train_ready_candidate"], 0)

    def test_online_qc_final_default_allows_enough_masked_episode_windows(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(reference=_online_qc_reference(), fpv_sample_interval_steps=999)
        )

        for _ in range(333):
            evaluator.evaluate(
                obs=_online_qc_obs(qpos=[0.0, 0.0, 0.0, 0.0]),
                now_ns=2_000_000_000,
            )
        for _ in range(767):
            evaluator.evaluate(
                obs=_online_qc_obs(qpos=[1.2, 0.0, 0.0, 0.0]),
                now_ns=2_000_000_000,
            )

        snapshot = evaluator.finalize_episode()

        self.assertEqual(snapshot.status, "PASS")
        self.assertEqual(snapshot.error_code, "")
        self.assertGreaterEqual(snapshot.diagnostics["online_qc_healthy_steps"], 333)
        self.assertEqual(snapshot.diagnostics["online_qc_train_ready_candidate"], 1)

    def test_online_qc_final_fails_bucket_reference_outlier(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        reference = _online_qc_reference()
        reference["bucket_qpos"] = {"p1": -1.0, "p99": 1.0}
        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=reference,
                fpv_sample_interval_steps=999,
                min_episode_steps=1,
                min_healthy_steps=1,
                min_healthy_fraction=0.0,
            )
        )

        for step in range(10):
            evaluator.evaluate(
                obs=_online_qc_obs(
                    qpos=[0.0, 0.0, 0.0, 1.4],
                    qpos_raw_imu=[0.0, 0.0, 0.0, 1.4],
                    image_timestamp_ns=2_000_000_000 + step * 20_000_000,
                ),
                now_ns=2_000_000_000 + step * 20_000_000,
            )

        snapshot = evaluator.finalize_episode(recorded_steps=10)

        self.assertEqual(snapshot.status, "FAIL_EPISODE")
        self.assertEqual(snapshot.error_code, "bucket_reference_outlier")
        self.assertEqual(snapshot.diagnostics["online_qc_train_ready_candidate"], 0)
        self.assertEqual(
            snapshot.diagnostics["online_qc_bucket_reference_status"],
            "FAIL",
        )
        self.assertLess(
            snapshot.diagnostics["online_qc_bucket_ref_high_margin"],
            -0.2,
        )

    def test_online_qc_final_fails_bucket_semantic_drop(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        reference = _online_qc_reference()
        reference["bucket_qpos"] = {"p1": -2.0, "p99": 1.0}
        reference["bucket_semantic"] = _online_qc_bucket_semantic_reference(
            end_p5=0.8,
            max_p5=1.2,
            late_max_p5=1.2,
            min_p99=-1.0,
        )
        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=reference,
                fpv_sample_interval_steps=999,
                min_episode_steps=1,
                min_healthy_steps=1,
                min_healthy_fraction=0.0,
            )
        )
        bucket = np.concatenate(
            [
                np.linspace(0.0, 1.1, 20),
                np.linspace(1.1, 0.0, 40),
                np.zeros(20),
            ]
        )
        _run_online_qc_bucket_series(evaluator, bucket)

        snapshot = evaluator.finalize_episode(recorded_steps=int(bucket.size))

        self.assertEqual(snapshot.status, "FAIL_EPISODE")
        self.assertEqual(snapshot.error_code, "bucket_semantic_outlier")
        self.assertEqual(snapshot.diagnostics["online_qc_train_ready_candidate"], 0)
        self.assertEqual(
            snapshot.diagnostics["online_qc_bucket_reference_status"],
            "WARN",
        )
        self.assertEqual(
            snapshot.diagnostics["online_qc_bucket_semantic_decision"],
            "drop",
        )
        self.assertIn(
            "bucket_end_or_late_recovery_too_low",
            snapshot.diagnostics["online_qc_bucket_semantic_notes"],
        )

    def test_online_qc_final_warns_for_bucket_semantic_review(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        reference = _online_qc_reference()
        reference["bucket_qpos"] = {"p1": -2.0, "p99": 1.0}
        reference["bucket_semantic"] = _online_qc_bucket_semantic_reference(
            end_p5=0.8,
            max_p5=0.8,
            late_max_p5=0.8,
            min_p99=0.0,
        )
        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=reference,
                fpv_sample_interval_steps=999,
                min_episode_steps=1,
                min_healthy_steps=1,
                min_healthy_fraction=0.0,
            )
        )
        bucket = np.full(80, 1.1, dtype=np.float64)
        _run_online_qc_bucket_series(evaluator, bucket)

        snapshot = evaluator.finalize_episode(recorded_steps=int(bucket.size))

        self.assertEqual(snapshot.status, "WARN_MASK")
        self.assertEqual(snapshot.error_code, "")
        self.assertIn("bucket_semantic_review", snapshot.warning_codes)
        self.assertEqual(snapshot.diagnostics["online_qc_train_ready_candidate"], 1)
        self.assertEqual(
            snapshot.diagnostics["online_qc_bucket_semantic_decision"],
            "review",
        )
        self.assertIn(
            "bucket_min_too_shallow",
            snapshot.diagnostics["online_qc_bucket_semantic_notes"],
        )

    def test_online_qc_final_keeps_bucket_reference_warn_when_semantics_pass(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        reference = _online_qc_reference()
        reference["bucket_qpos"] = {"p1": -2.0, "p99": 1.0}
        reference["bucket_semantic"] = _online_qc_bucket_semantic_reference(
            end_p5=0.8,
            max_p5=0.8,
            late_max_p5=0.8,
            min_p99=1.2,
        )
        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=reference,
                fpv_sample_interval_steps=999,
                min_episode_steps=1,
                min_healthy_steps=1,
                min_healthy_fraction=0.0,
            )
        )
        bucket = np.full(80, 1.1, dtype=np.float64)
        _run_online_qc_bucket_series(evaluator, bucket)

        snapshot = evaluator.finalize_episode(recorded_steps=int(bucket.size))

        self.assertEqual(snapshot.status, "PASS")
        self.assertEqual(snapshot.error_code, "")
        self.assertEqual(snapshot.warning_codes, ())
        self.assertEqual(snapshot.diagnostics["online_qc_train_ready_candidate"], 1)
        self.assertEqual(
            snapshot.diagnostics["online_qc_bucket_semantic_decision"],
            "keep",
        )
        self.assertEqual(
            snapshot.diagnostics["online_qc_bucket_semantic_info"],
            "bucket_reference_semantic_keep",
        )

    def test_online_qc_fails_immediately_on_qpos_jump(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=_online_qc_reference(),
                qpos_jump_fail_rad=0.20,
                fpv_sample_interval_steps=999,
            )
        )
        first = evaluator.evaluate(
            obs=_online_qc_obs(qpos=[0.0, 0.0, 0.0, 0.0]),
            now_ns=2_000_000_000,
        )
        second = evaluator.evaluate(
            obs=_online_qc_obs(qpos=[0.21, 0.0, 0.0, 0.0]),
            now_ns=2_020_000_000,
        )

        self.assertEqual(first.status, "PASS")
        self.assertEqual(second.status, "FAIL_EPISODE")
        self.assertEqual(second.error_code, "qpos_jump")

    def test_online_qc_masks_then_fails_when_raw_imu_qpos_diverges(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=_online_qc_reference(),
                imu_qpos_delta_warn_consecutive_steps=5,
                imu_qpos_delta_fail_consecutive_steps=25,
                max_policy_raw_qpos_delta_rad=[0.08, 0.08, 0.08, 0.08],
                fpv_sample_interval_steps=999,
            )
        )

        snapshot = None
        for _ in range(5):
            snapshot = evaluator.evaluate(
                obs=_online_qc_obs(
                    qpos=[0.0, 0.0, 0.0, 0.0],
                    qpos_raw_imu=[0.10, 0.0, 0.0, 0.0],
                ),
                now_ns=2_000_000_000,
            )
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.status, "WARN_MASK")
        self.assertIn("imu_qpos_delta_high", snapshot.warning_codes)
        self.assertTrue(snapshot.train_exclude)

        for _ in range(20):
            snapshot = evaluator.evaluate(
                obs=_online_qc_obs(
                    qpos=[0.0, 0.0, 0.0, 0.0],
                    qpos_raw_imu=[0.10, 0.0, 0.0, 0.0],
                ),
                now_ns=2_000_000_000,
            )
        self.assertEqual(snapshot.status, "FAIL_EPISODE")
        self.assertEqual(snapshot.error_code, "imu_qpos_delta_high")

    def test_online_qc_warns_but_does_not_fail_when_raw_imu_qpos_missing(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(reference=_online_qc_reference(), fpv_sample_interval_steps=999)
        )

        snapshot = evaluator.evaluate(
            obs=_online_qc_obs(qpos=[0.0, 0.0, 0.0, 0.0]),
            now_ns=2_000_000_000,
        )

        self.assertEqual(snapshot.status, "PASS")
        self.assertEqual(snapshot.error_code, "")
        self.assertIn("imu_qpos_reference_missing", snapshot.warning_codes)
        self.assertFalse(snapshot.train_exclude)

    def test_online_qc_config_loads_reference_path_and_missing_path_degrades(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "online_qc_reference.json"
            path.write_text(json.dumps(_online_qc_reference()), encoding="utf-8")

            loaded = OnlineQcConfig.from_mapping(
                {
                    "enabled": True,
                    "reference_path": str(path),
                    "qpos_warn_consecutive_steps": 7,
                }
            )
            missing = OnlineQcConfig.from_mapping(
                {
                    "enabled": True,
                    "reference_path": str(Path(tmpdir) / "missing.json"),
                }
            )

        self.assertTrue(loaded.enabled)
        self.assertEqual(loaded.reference["reference_id"], "unit-reference")
        self.assertEqual(loaded.qpos_warn_consecutive_steps, 7)
        self.assertTrue(missing.enabled)
        self.assertIsNone(missing.reference)

    def test_step_diagnostics_include_online_qc_fields(self) -> None:
        from testbed.cli.record_real import _build_step_diagnostics
        from testbed.data.online_qc import OnlineQcSnapshot

        diagnostics = _build_step_diagnostics(
            obs={},
            raw_action=np.zeros(4, dtype=np.float32),
            safe_action=np.zeros(4, dtype=np.float32),
            action_info=SimpleNamespace(extras={}),
            action_sample_timestamp_ns=0,
            action_send_timestamp_ns=0,
            guard=SimpleNamespace(last_info=SimpleNamespace(triggered=False, reasons=())),
            control_result={"ack": True, "commanded_action": np.zeros(4)},
            online_qc=OnlineQcSnapshot(
                status="WARN_MASK",
                error_code="",
                warning_codes=("fpv_drift",),
                train_exclude=True,
                diagnostics={
                    "online_qc_status": "WARN_MASK",
                    "online_qc_error_code": "",
                    "online_qc_warning_codes": "fpv_drift",
                    "train_exclude_mask": 1,
                },
            ),
        )

        self.assertEqual(diagnostics["online_qc_status"], "WARN_MASK")
        self.assertEqual(diagnostics["online_qc_warning_codes"], "fpv_drift")
        self.assertEqual(diagnostics["train_exclude_mask"], 1)

    def test_online_qc_fails_immediately_on_bad_fpv_frame(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(reference=_online_qc_reference(), fpv_sample_interval_steps=1)
        )

        black = evaluator.evaluate(
            obs=_online_qc_obs(
                qpos=[0.0, 0.0, 0.0, 0.0],
                qpos_raw_imu=[0.0, 0.0, 0.0, 0.0],
                image=np.zeros((8, 8, 3), dtype=np.uint8),
            ),
            now_ns=2_000_000_000,
        )
        self.assertEqual(black.status, "FAIL_EPISODE")
        self.assertEqual(black.error_code, "fpv_black")

        bad_jpeg = OnlineTrainingQcEvaluator(
            OnlineQcConfig(reference=_online_qc_reference(), fpv_sample_interval_steps=1)
        ).evaluate(
            obs=_online_qc_obs(
                qpos=[0.0, 0.0, 0.0, 0.0],
                qpos_raw_imu=[0.0, 0.0, 0.0, 0.0],
                encoded_image=np.asarray([1, 2, 3, 4], dtype=np.uint8),
            ),
            now_ns=2_000_000_000,
        )
        self.assertEqual(bad_jpeg.status, "FAIL_EPISODE")
        self.assertEqual(bad_jpeg.error_code, "fpv_decode_failed")

    def test_online_qc_fails_immediately_on_repeated_fpv_frame(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(reference=_online_qc_reference(), fpv_sample_interval_steps=1)
        )
        image = _online_qc_pattern_image()

        first = evaluator.evaluate(
            obs=_online_qc_obs(
                qpos=[0.0, 0.0, 0.0, 0.0],
                qpos_raw_imu=[0.0, 0.0, 0.0, 0.0],
                image=image,
                image_timestamp_ns=2_000_000_000,
            ),
            now_ns=2_000_000_000,
        )
        second = evaluator.evaluate(
            obs=_online_qc_obs(
                qpos=[0.0, 0.0, 0.0, 0.0],
                qpos_raw_imu=[0.0, 0.0, 0.0, 0.0],
                image=image,
                image_timestamp_ns=2_020_000_000,
            ),
            now_ns=2_020_000_000,
        )

        self.assertEqual(first.status, "PASS")
        self.assertEqual(second.status, "FAIL_EPISODE")
        self.assertEqual(second.error_code, "fpv_duplicate")

    def test_online_qc_masks_persistent_fpv_drift_by_default(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=_online_qc_reference(),
                fpv_sample_interval_steps=1,
                fpv_drift_warn_consecutive_samples=5,
                fpv_drift_fail_consecutive_samples=25,
            )
        )

        snapshot = None
        for step in range(5):
            snapshot = evaluator.evaluate(
                obs=_online_qc_obs(
                    qpos=[0.0, 0.0, 0.0, 0.0],
                    qpos_raw_imu=[0.0, 0.0, 0.0, 0.0],
                    image=_online_qc_bright_drift_image(step),
                    image_timestamp_ns=2_000_000_000 + step * 20_000_000,
                ),
                now_ns=2_000_000_000 + step * 20_000_000,
            )
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.status, "WARN_MASK")
        self.assertIn("fpv_drift", snapshot.warning_codes)
        self.assertTrue(snapshot.train_exclude)

        for step in range(5, 25):
            snapshot = evaluator.evaluate(
                obs=_online_qc_obs(
                    qpos=[0.0, 0.0, 0.0, 0.0],
                    qpos_raw_imu=[0.0, 0.0, 0.0, 0.0],
                    image=_online_qc_bright_drift_image(step),
                    image_timestamp_ns=2_000_000_000 + step * 20_000_000,
                ),
                now_ns=2_000_000_000 + step * 20_000_000,
            )
        self.assertEqual(snapshot.status, "WARN_MASK")
        self.assertEqual(snapshot.error_code, "")
        self.assertIn("fpv_drift", snapshot.warning_codes)

    def test_online_qc_can_hard_fail_persistent_fpv_drift(self) -> None:
        from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator

        evaluator = OnlineTrainingQcEvaluator(
            OnlineQcConfig(
                reference=_online_qc_reference(),
                fpv_sample_interval_steps=1,
                fpv_drift_hard_fail=True,
                fpv_drift_warn_consecutive_samples=5,
                fpv_drift_fail_consecutive_samples=25,
            )
        )

        snapshot = None
        for step in range(25):
            snapshot = evaluator.evaluate(
                obs=_online_qc_obs(
                    qpos=[0.0, 0.0, 0.0, 0.0],
                    qpos_raw_imu=[0.0, 0.0, 0.0, 0.0],
                    image=_online_qc_bright_drift_image(step),
                    image_timestamp_ns=2_000_000_000 + step * 20_000_000,
                ),
                now_ns=2_000_000_000 + step * 20_000_000,
            )
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.status, "FAIL_EPISODE")
        self.assertEqual(snapshot.error_code, "fpv_drift")

    @unittest.skipUnless(HAS_H5PY, "h5py is required for online QC reference tests")
    def test_build_online_qc_reference_writes_qpos_and_fpv_stats(self) -> None:
        from testbed.cli.build_online_qc_reference import build_online_qc_reference

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            _write_online_qc_reference_episode(dataset_dir / "episode_0.hdf5", offset=0.0)
            _write_online_qc_reference_episode(dataset_dir / "episode_1.hdf5", offset=0.2)
            manifest_path = root / "train_ready_manifest.json"
            manifest_path.write_text(
                json.dumps({"train_ready_episode_ids": [0, 1]}),
                encoding="utf-8",
            )
            summary_path = root / "training_qc_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "reference": {
                            "qpos": {
                                "p1": [-9.0, -9.0, -9.0, -9.0],
                                "p5": [-8.0, -8.0, -8.0, -8.0],
                                "p95": [8.0, 8.0, 8.0, 8.0],
                                "p99": [9.0, 9.0, 9.0, 9.0],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "online_qc_reference.json"

            reference = build_online_qc_reference(
                dataset_dir=dataset_dir,
                manifest_path=manifest_path,
                training_qc_summary_path=summary_path,
                output_path=output_path,
            )

            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], 1)
            self.assertEqual(written["episode_ids"], [0, 1])
            np.testing.assert_allclose(
                written["qpos"]["p5"],
                [0.0, 0.1, 0.2, 0.3],
            )
            np.testing.assert_allclose(
                written["qpos"]["p95"],
                [0.2, 0.3, 0.4, 0.5],
            )
            self.assertIn("reference_id", written)
            self.assertGreater(written["fpv"]["brightness"]["count"], 0)
            self.assertEqual(len(written["fpv"]["fingerprint"]), 64)
            self.assertEqual(reference["reference_id"], written["reference_id"])

    @unittest.skipUnless(HAS_H5PY, "h5py is required for online QC reference tests")
    def test_build_online_qc_reference_writes_bucket_semantic_reference(self) -> None:
        from testbed.cli.build_online_qc_reference import build_online_qc_reference

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            for episode_id in range(5):
                _write_online_qc_reference_episode(
                    dataset_dir / f"episode_{episode_id}.hdf5",
                    offset=float(episode_id) * 0.05,
                )
            manifest_path = root / "train_ready_manifest.json"
            manifest_path.write_text(
                json.dumps({"train_ready_episode_ids": [0, 1, 2, 3, 4]}),
                encoding="utf-8",
            )
            bucket_qpos_ref = {
                "count": 123,
                "p1": -1.5,
                "p5": -1.2,
                "median": 0.0,
                "p95": 1.2,
                "p99": 1.5,
            }
            summary_path = root / "training_qc_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "strict_pass_episode_ids": [0, 1, 2, 3, 4],
                        "train_ready_episode_ids": [0, 1, 2, 3, 4],
                        "reference": {"bucket_qpos": bucket_qpos_ref},
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "online_qc_reference.json"

            reference = build_online_qc_reference(
                dataset_dir=dataset_dir,
                manifest_path=manifest_path,
                training_qc_summary_path=summary_path,
                output_path=output_path,
            )

            self.assertEqual(reference["bucket_qpos"], bucket_qpos_ref)
            self.assertEqual(reference["bucket_semantic"]["count"], 5)
            self.assertIn("end", reference["bucket_semantic"])
            self.assertIn("late_max", reference["bucket_semantic"])
            self.assertEqual(
                reference["source_reference"]["strict_pass_episode_ids"],
                [0, 1, 2, 3, 4],
            )
            self.assertEqual(
                reference["source_reference"]["train_ready_episode_ids"],
                [0, 1, 2, 3, 4],
            )
            self.assertEqual(
                reference["source_reference"]["manifest_path"],
                str(manifest_path),
            )
            self.assertEqual(
                reference["source_reference"]["training_qc_summary_path"],
                str(summary_path),
            )
            self.assertEqual(len(reference["reference_id"]), 16)

    def test_online_qc_failure_blocks_record_start(self) -> None:
        from testbed.cli.record_real import _online_qc_blocks_record_start
        from testbed.data.online_qc import OnlineQcSnapshot

        self.assertTrue(
            _online_qc_blocks_record_start(
                OnlineQcSnapshot(
                    status="FAIL_EPISODE",
                    error_code="fpv_drift",
                    train_exclude=True,
                )
            )
        )
        self.assertFalse(
            _online_qc_blocks_record_start(
                OnlineQcSnapshot(
                    status="WARN_MASK",
                    error_code="",
                    warning_codes=("fpv_drift",),
                    train_exclude=True,
                )
            )
        )

    @unittest.skipUnless(HAS_H5PY, "h5py is required for record mask tests")
    def test_record_session_backfills_recent_online_qc_mask(self) -> None:
        from testbed.cli.record_real import RecordSession
        from testbed.data.recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session = RecordSession(
                recorder_cls=EpisodeRecorder,
                dataset_dir=root / "dataset",
                failed_dir=root / "failed",
                episode_idx=0,
                metadata={"is_real": True},
                camera_names=[],
            )
            for step in range(3):
                session.record_step(
                    obs={
                        "qpos": np.zeros(4, dtype=np.float32),
                        "qvel": np.zeros(4, dtype=np.float32),
                    },
                    action=np.zeros(4, dtype=np.float32),
                    diagnostics={"train_exclude_mask": 0},
                    step_id=step,
                )

            session.mark_recent_train_exclude(window_steps=2)
            path = session.save_success()

            import h5py

            with h5py.File(path, "r") as f:
                np.testing.assert_array_equal(
                    f["diagnostics/train_exclude_mask"][()],
                    np.asarray([0, 1, 1]),
                )

    @unittest.skipUnless(HAS_H5PY, "h5py is required for record mask tests")
    def test_record_session_final_online_qc_failure_saves_failed_episode(self) -> None:
        from testbed.cli.record_real import (
            RecordSession,
            _save_record_session_with_online_qc_final,
        )
        from testbed.data.online_qc import OnlineQcSnapshot
        from testbed.data.recorder import EpisodeRecorder

        class FinalFailQc:
            def finalize_episode(self, *, recorded_steps: int) -> OnlineQcSnapshot:
                self.recorded_steps = int(recorded_steps)
                return OnlineQcSnapshot(
                    status="FAIL_EPISODE",
                    error_code="bucket_semantic_outlier",
                    train_exclude=True,
                    diagnostics={
                        "online_qc_reference_id": "ref-bucket-123",
                        "online_qc_train_ready_candidate": 0,
                        "online_qc_bucket_reference_status": "WARN",
                        "online_qc_bucket_semantic_decision": "drop",
                        "online_qc_bucket_semantic_notes": (
                            "bucket_end_or_late_recovery_too_low"
                        ),
                    },
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session = RecordSession(
                recorder_cls=EpisodeRecorder,
                dataset_dir=root / "dataset",
                failed_dir=root / "failed",
                episode_idx=0,
                metadata={"is_real": True},
                camera_names=[],
            )
            for step in range(3):
                session.record_step(
                    obs={
                        "qpos": np.zeros(4, dtype=np.float32),
                        "qvel": np.zeros(4, dtype=np.float32),
                    },
                    action=np.zeros(4, dtype=np.float32),
                    diagnostics={"train_exclude_mask": 0},
                    step_id=step,
                )
            evaluator = FinalFailQc()

            path, saved_success, snapshot = _save_record_session_with_online_qc_final(
                session,
                online_qc_evaluator=evaluator,
                error_time_ns=123,
            )

            self.assertFalse(saved_success)
            self.assertEqual(snapshot.error_code, "bucket_semantic_outlier")
            self.assertEqual(evaluator.recorded_steps, 3)
            self.assertIn("/failed/", str(path))

            import h5py

            with h5py.File(path, "r") as f:
                meta = f["metadata"].attrs
                self.assertFalse(bool(meta["success"]))
                self.assertEqual(meta["record_stop_reason"], "online_qc_failed")
                self.assertEqual(meta["record_error_code"], "bucket_semantic_outlier")
                self.assertEqual(int(meta["record_error_time_ns"]), 123)
                self.assertEqual(meta["online_qc_final_status"], "FAIL_EPISODE")
                self.assertEqual(int(meta["online_qc_train_ready_candidate"]), 0)
                self.assertEqual(meta["online_qc_reference_id"], "ref-bucket-123")
                self.assertEqual(meta["online_qc_bucket_reference_status"], "WARN")
                self.assertEqual(meta["online_qc_bucket_semantic_decision"], "drop")
                self.assertEqual(
                    meta["online_qc_bucket_semantic_notes"],
                    "bucket_end_or_late_recovery_too_low",
                )

    def test_action_pump_repeats_latest_action(self) -> None:
        class RecordingController(LowLevelController):
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.actions: list[np.ndarray] = []

            def send(self, action: np.ndarray, state: dict | None = None) -> ControlResult:
                commanded = np.asarray(action, dtype=np.float32).copy()
                with self.lock:
                    self.actions.append(commanded)
                return ControlResult(
                    ack=True,
                    fault_code="",
                    controller_timestamp_ns=time.time_ns(),
                    commanded_action=commanded,
                    raw_low_level_command=commanded.copy(),
                )

        controller = RecordingController()
        pump = RealActionPump(
            controller,
            hz=40,
            send_immediately_on_update=False,
            zero_on_stop=False,
        )
        target = np.array([0.4, 0.0, -0.2, 0.1], dtype=np.float32)
        pump.update_action(target)
        pump.start()
        try:
            time.sleep(0.09)
        finally:
            pump.stop(close_controller=False)

        with controller.lock:
            actions = list(controller.actions)
        self.assertGreaterEqual(len(actions), 2)
        np.testing.assert_allclose(actions[-1], target)

    def test_go_home_controller_converges_and_rejects_far_start(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "max_action": [0.2, 0.2, 0.2, 0.2],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.0,
                "timeout_s": 1.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        controller.start(
            {
                "qpos": np.array([0.1, -0.1, 0.05, -0.05], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        result = controller.update(
            {
                "qpos": np.array([0.1, -0.1, 0.05, -0.05], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        np.testing.assert_allclose(result.action, [-0.1, 0.1, -0.05, 0.05])
        done = controller.update(
            {
                "qpos": np.zeros(4, dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        self.assertTrue(done.done)
        self.assertEqual(controller.metadata()["go_home_result"], "succeeded")

        far_controller = GoHomeController(cfg)
        with self.assertRaises(ValueError):
            far_controller.start(
                {
                    "qpos": np.ones(4, dtype=np.float32),
                    "qvel": np.zeros(4, dtype=np.float32),
                }
            )

    def test_go_home_controller_uses_shortest_swing_home_error(self) -> None:
        home_swing = np.deg2rad(216.46)
        wrapped_feedback = home_swing - 2.0 * np.pi + 0.002
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [home_swing, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.05, 0.05, 0.05, 0.05],
                "success_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "center_tolerance_rad": [0.005, 0.005, 0.005, 0.005],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "max_action": [0.2, 0.2, 0.2, 0.2],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.0,
                "timeout_s": 1.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        obs = {
            "qpos": np.array([wrapped_feedback, 0.0, 0.0, 0.0], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }

        controller.start(obs)
        result = controller.update(obs)

        self.assertTrue(result.done)
        self.assertLess(abs(result.diagnostics["go_home_error"][0]), 0.003)
        self.assertLess(abs(controller.metadata()["go_home_final_error"][0]), 0.003)
        self.assertGreater(abs(home_swing - wrapped_feedback), 6.0)

    def test_go_home_controller_rejects_policy_raw_qpos_divergence_on_start(
        self,
    ) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [999.0, 999.0, 999.0, 999.0],
                "success_tolerance_rad": [0.05, 0.05, 0.05, 0.05],
                "max_policy_raw_qpos_delta_rad": [0.08, 0.08, 0.08, 0.08],
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)

        with self.assertRaisesRegex(ValueError, "feedback_inconsistent"):
            controller.start(
                {
                    "qpos": np.zeros(4, dtype=np.float32),
                    "qpos_raw_imu": np.array([0.0, 0.0, 0.0, 0.40], dtype=np.float32),
                    "qvel": np.zeros(4, dtype=np.float32),
                }
            )

    def test_go_home_controller_uses_bucket_quaternion_for_raw_feedback(
        self,
    ) -> None:
        bucket_rad = float(np.deg2rad(59.40))

        def pitch_quat_wxyz(pitch_deg: float) -> list[float]:
            half = float(np.deg2rad(pitch_deg)) * 0.5
            return [float(np.cos(half)), 0.0, float(np.sin(half)), 0.0]

        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, bucket_rad],
                "near_tolerance_rad": [999.0, 999.0, 999.0, 999.0],
                "success_tolerance_rad": [0.05, 0.05, 0.05, 0.05],
                "max_policy_raw_qpos_delta_rad": [999.0, 999.0, 999.0, 0.08],
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        obs = {
            "qpos": np.array([0.0, 0.0, 0.0, bucket_rad], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
            "imu_debug": {
                "devices": [
                    {
                        "online": 1,
                        "valid_attitude": 1,
                        "valid_quaternion": 1,
                        "rpy_raw_deg": [0.0, -56.47, 0.0],
                        "quaternion_wxyz": pitch_quat_wxyz(25.26),
                    },
                    {
                        "online": 1,
                        "valid_attitude": 1,
                        "valid_quaternion": 1,
                        "rpy_raw_deg": [0.0, -34.14, 0.0],
                        "quaternion_wxyz": pitch_quat_wxyz(-34.14),
                    },
                    {"online": 1, "valid_attitude": 1, "rpy_raw_deg": [0.0, 0.0, 0.0]},
                    {"online": 1, "valid_attitude": 1, "rpy_raw_deg": [0.0, 0.0, 0.0]},
                ],
            },
        }

        controller.start(obs)
        result = controller.update(obs)

        self.assertEqual(result.diagnostics["go_home_feedback_consistent"], 1)
        self.assertAlmostEqual(
            float(result.diagnostics["go_home_raw_imu_qpos"][3]),
            bucket_rad,
            places=5,
        )

    def test_go_home_controller_applies_bucket_quaternion_policy_offset(
        self,
    ) -> None:
        legacy_offset = -0.4060066694119653

        def pitch_quat_wxyz(pitch_deg: float) -> list[float]:
            half = float(np.deg2rad(pitch_deg)) * 0.5
            return [float(np.cos(half)), 0.0, float(np.sin(half)), 0.0]

        obs = {
            "qpos": np.array([0.0, 0.0, 0.0, legacy_offset], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
            "imu_debug": {
                "devices": [
                    {
                        "online": 1,
                        "valid_attitude": 1,
                        "valid_quaternion": 1,
                        "rpy_raw_deg": [0.0, 0.0, 0.0],
                        "quaternion_wxyz": pitch_quat_wxyz(0.0),
                    },
                    {
                        "online": 1,
                        "valid_attitude": 1,
                        "valid_quaternion": 1,
                        "rpy_raw_deg": [0.0, 0.0, 0.0],
                        "quaternion_wxyz": pitch_quat_wxyz(0.0),
                    },
                    {"online": 1, "valid_attitude": 1, "rpy_raw_deg": [0.0, 0.0, 0.0]},
                    {"online": 1, "valid_attitude": 1, "rpy_raw_deg": [0.0, 0.0, 0.0]},
                ],
            },
        }
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, legacy_offset],
                "near_tolerance_rad": [999.0, 999.0, 999.0, 999.0],
                "success_tolerance_rad": [0.05, 0.05, 0.05, 0.05],
                "max_policy_raw_qpos_delta_rad": [0.08, 0.08, 0.08, 0.08],
            }
        )
        assert cfg is not None
        with patch.dict(
            os.environ,
            {"EXCAVATOR_BUCKET_QUATERNION_OFFSET_RAD": str(legacy_offset)},
        ):
            controller = GoHomeController(cfg)
            controller.start(obs)
            result = controller.update(obs)

        self.assertEqual(result.diagnostics["go_home_feedback_consistent"], 1)
        self.assertAlmostEqual(
            float(result.diagnostics["go_home_raw_imu_qpos"][3]),
            legacy_offset,
            places=5,
        )

    def test_go_home_controller_limits_bucket_quaternion_branch_jump(
        self,
    ) -> None:
        def pitch_quat_wxyz(pitch_deg: float) -> list[float]:
            half = float(np.deg2rad(pitch_deg)) * 0.5
            return [float(np.cos(half)), 0.0, float(np.sin(half)), 0.0]

        def obs_for_bucket(bucket_deg: float) -> dict[str, object]:
            return {
                "qpos": np.zeros(4, dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
                "imu_debug": {
                    "devices": [
                        {
                            "online": 1,
                            "valid_attitude": 1,
                            "valid_quaternion": 1,
                            "rpy_raw_deg": [0.0, bucket_deg, 0.0],
                            "quaternion_wxyz": pitch_quat_wxyz(bucket_deg),
                        },
                        {
                            "online": 1,
                            "valid_attitude": 1,
                            "valid_quaternion": 1,
                            "rpy_raw_deg": [0.0, 0.0, 0.0],
                            "quaternion_wxyz": pitch_quat_wxyz(0.0),
                        },
                        {"online": 1, "valid_attitude": 1, "rpy_raw_deg": [0.0, 0.0, 0.0]},
                        {"online": 1, "valid_attitude": 1, "rpy_raw_deg": [0.0, 0.0, 0.0]},
                    ],
                },
            }

        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [999.0, 999.0, 999.0, 999.0],
                "success_tolerance_rad": [0.05, 0.05, 0.05, 0.05],
                "max_policy_raw_qpos_delta_rad": [0.08, 0.08, 0.08, 0.08],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)

        controller.start(obs_for_bucket(0.0))
        result = controller.update(obs_for_bucket(120.0), now_s=controller._start_s)

        self.assertEqual(result.diagnostics["go_home_feedback_consistent"], 1)
        self.assertAlmostEqual(
            float(result.diagnostics["go_home_raw_imu_qpos"][3]),
            float(np.deg2rad(2.5)),
            places=5,
        )
        self.assertAlmostEqual(
            float(result.diagnostics["go_home_policy_raw_delta"][3]),
            -float(np.deg2rad(2.5)),
            places=5,
        )

    def test_go_home_controller_allows_imu_debug_bucket_raw_branch_for_done(
        self,
    ) -> None:
        def pitch_quat_wxyz(pitch_deg: float) -> list[float]:
            half = float(np.deg2rad(pitch_deg)) * 0.5
            return [float(np.cos(half)), 0.0, float(np.sin(half)), 0.0]

        obs = {
            "qpos": np.zeros(4, dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
            "imu_debug": {
                "devices": [
                    {
                        "online": 1,
                        "valid_attitude": 1,
                        "valid_quaternion": 1,
                        "rpy_raw_deg": [0.0, 23.0, 0.0],
                        "quaternion_wxyz": pitch_quat_wxyz(23.0),
                    },
                    {
                        "online": 1,
                        "valid_attitude": 1,
                        "valid_quaternion": 1,
                        "rpy_raw_deg": [0.0, 0.0, 0.0],
                        "quaternion_wxyz": pitch_quat_wxyz(0.0),
                    },
                    {"online": 1, "valid_attitude": 1, "rpy_raw_deg": [0.0, 0.0, 0.0]},
                    {"online": 1, "valid_attitude": 1, "rpy_raw_deg": [0.0, 0.0, 0.0]},
                ],
            },
        }
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [999.0, 999.0, 999.0, 999.0],
                "success_tolerance_rad": [0.05, 0.05, 0.05, 0.05],
                "center_tolerance_rad": [0.02, 0.02, 0.02, 0.02],
                "max_policy_raw_qpos_delta_rad": [0.08, 0.08, 0.08, 0.08],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)

        controller.start(obs)
        result = controller.update(obs, now_s=controller._start_s)

        self.assertTrue(result.done)
        self.assertEqual(result.diagnostics["go_home_feedback_consistent"], 1)
        self.assertAlmostEqual(
            float(result.diagnostics["go_home_policy_raw_delta"][3]),
            -float(np.deg2rad(23.0)),
            places=5,
        )

    def test_go_home_controller_does_not_finish_when_policy_raw_qpos_diverges(
        self,
    ) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [999.0, 999.0, 999.0, 999.0],
                "success_tolerance_rad": [0.05, 0.05, 0.05, 0.05],
                "center_tolerance_rad": [0.02, 0.02, 0.02, 0.02],
                "max_policy_raw_qpos_delta_rad": [0.08, 0.08, 0.08, 0.08],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.0,
                "timeout_s": 1.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        controller.start(
            {
                "qpos": np.array([0.20, 0.0, 0.0, 0.0], dtype=np.float32),
                "qpos_raw_imu": np.array([0.20, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )

        result = controller.update(
            {
                "qpos": np.zeros(4, dtype=np.float32),
                "qpos_raw_imu": np.array([0.0, 0.0, 0.0, 0.40], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            },
            now_s=controller._start_s,
        )

        self.assertFalse(result.done)
        self.assertEqual(result.diagnostics["go_home_feedback_consistent"], 0)
        self.assertEqual(result.diagnostics["go_home_acceptable_position"], 0)
        np.testing.assert_allclose(
            result.diagnostics["go_home_policy_raw_delta"],
            [0.0, 0.0, 0.0, -0.40],
            atol=1e-6,
        )

    def test_real_qpos_error_wraps_only_swing_axis(self) -> None:
        home = np.array([np.deg2rad(216.46), 0.20, -0.30, 0.40], dtype=np.float32)
        current = np.array(
            [home[0] - 2.0 * np.pi + 0.01, -0.10, 0.10, -0.20],
            dtype=np.float32,
        )

        err = real_qpos_error_rad(home, current)

        np.testing.assert_allclose(err, [-0.01, 0.30, -0.40, 0.60], atol=1e-6)

    def test_go_home_controller_times_out(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.001, 0.001, 0.001, 0.001],
                "timeout_s": 0.001,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        obs = {
            "qpos": np.ones(4, dtype=np.float32) * 0.1,
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(obs)
        time.sleep(0.01)
        result = controller.update(obs)
        self.assertTrue(result.failed)
        self.assertEqual(result.reason, "timeout")

    def test_go_home_controller_allows_dwell_to_complete_after_timeout_if_centered(
        self,
    ) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "center_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.2,
                "timeout_s": 0.1,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        start_obs = {
            "qpos": np.ones(4, dtype=np.float32) * 0.1,
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(start_obs)
        centered_obs = {
            "qpos": np.zeros(4, dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }

        settling = controller.update(centered_obs, now_s=controller._start_s + 0.11)
        self.assertFalse(settling.failed)
        self.assertFalse(settling.done)
        np.testing.assert_allclose(settling.action, np.zeros(4, dtype=np.float32))

        done = controller.update(centered_obs, now_s=controller._start_s + 0.32)
        self.assertTrue(done.done)
        self.assertEqual(controller.metadata()["go_home_result"], "succeeded")

    def test_go_home_controller_accepts_stable_success_band_outside_center(
        self,
    ) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.08, 0.08, 0.08, 0.08],
                "center_tolerance_rad": [0.02, 0.02, 0.02, 0.02],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.2,
                "timeout_s": 0.1,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        start_obs = {
            "qpos": np.ones(4, dtype=np.float32) * 0.1,
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(start_obs)
        acceptable_obs = {
            "qpos": np.ones(4, dtype=np.float32) * 0.05,
            "qvel": np.zeros(4, dtype=np.float32),
        }

        settling = controller.update(acceptable_obs, now_s=controller._start_s + 0.11)
        self.assertFalse(settling.failed)
        self.assertFalse(settling.done)
        self.assertEqual(settling.diagnostics["go_home_acceptable_position"], 1)
        self.assertEqual(settling.diagnostics["go_home_in_position"], 0)

        done = controller.update(acceptable_obs, now_s=controller._start_s + 0.32)
        self.assertTrue(done.done)
        self.assertEqual(controller.metadata()["go_home_result"], "succeeded")

    def test_go_home_controller_applies_min_action_per_axis(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "min_action": [0.05, 0.05, 0.05, 0.05],
                "max_action": [0.2, 0.2, 0.2, 0.2],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.0,
                "timeout_s": 1.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        obs = {
            "qpos": np.array([0.02, 0.005, -0.02, 0.0], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(obs)
        result = controller.update(obs)
        np.testing.assert_allclose(result.action, [-0.05, 0.0, 0.05, 0.0])

    def test_go_home_controller_applies_directional_min_action(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "min_action": [0.05, 0.05, 0.05, 0.05],
                "min_action_positive": [0.08, 0.08, 0.08, 0.08],
                "min_action_negative": [0.04, 0.04, 0.04, 0.04],
                "max_action": [0.2, 0.2, 0.2, 0.2],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.0,
                "timeout_s": 1.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        obs = {
            "qpos": np.array([-0.02, 0.02, 0.0, 0.0], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(obs)
        result = controller.update(obs)
        np.testing.assert_allclose(result.action[:2], [0.08, -0.04])

    def test_go_home_controller_serializes_axes_by_configured_order(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "min_action": [0.05, 0.05, 0.05, 0.05],
                "max_action": [0.2, 0.2, 0.2, 0.2],
                "axis_order": ["boom", "stick", "bucket", "swing"],
                "max_active_axes": 1,
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 10.0,
                "timeout_s": 5.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        obs = {
            "qpos": np.array([0.10, 0.10, 0.10, 0.10], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(obs)

        first = controller.update(obs)
        np.testing.assert_allclose(first.action, [0.0, -0.10, 0.0, 0.0])
        np.testing.assert_allclose(
            first.diagnostics["go_home_axis_scheduled"],
            [0, 1, 0, 0],
        )

        boom_centered = controller.update(
            {
                "qpos": np.array([0.10, 0.0, 0.10, 0.10], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        np.testing.assert_allclose(boom_centered.action, [0.0, 0.0, -0.10, 0.0])
        np.testing.assert_allclose(
            boom_centered.diagnostics["go_home_axis_scheduled"],
            [0, 0, 1, 0],
        )

    def test_go_home_controller_gradually_boosts_stalled_axis(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.05, 0.05, 0.05, 0.05],
                "center_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "resume_tolerance_rad": [0.02, 0.02, 0.02, 0.02],
                "p_gain": [0.1, 0.1, 0.1, 0.1],
                "min_action": [0.10, 0.10, 0.10, 0.10],
                "max_action": [0.30, 0.30, 0.30, 0.30],
                "near_max_action": [0.12, 0.12, 0.12, 0.12],
                "stall_detection_s": 0.5,
                "stall_boost_interval_s": 0.5,
                "stall_action_step": [0.05, 0.0, 0.0, 0.0],
                "stall_error_progress_rad": [0.001, 0.001, 0.001, 0.001],
                "stall_qvel_threshold_rad_s": [0.01, 0.01, 0.01, 0.01],
                "qvel_stable_rad_s": [0.001, 0.001, 0.001, 0.001],
                "dwell_s": 0.0,
                "timeout_s": 5.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        obs = {
            "qpos": np.array([0.08, 0.0, 0.0, 0.0], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(obs)
        start = controller._start_s

        first = controller.update(obs, now_s=start)
        np.testing.assert_allclose(first.action, [-0.10, 0.0, 0.0, 0.0])
        self.assertEqual(first.diagnostics["go_home_axis_stalled"][0], 0)

        boosted_once = controller.update(obs, now_s=start + 0.6)
        np.testing.assert_allclose(boosted_once.action, [-0.15, 0.0, 0.0, 0.0])
        self.assertEqual(boosted_once.diagnostics["go_home_axis_stalled"][0], 1)
        np.testing.assert_allclose(
            boosted_once.diagnostics["go_home_stall_action_boost"],
            [0.05, 0.0, 0.0, 0.0],
        )

        boosted_twice = controller.update(obs, now_s=start + 1.2)
        np.testing.assert_allclose(boosted_twice.action, [-0.20, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            boosted_twice.diagnostics["go_home_stall_action_boost"],
            [0.10, 0.0, 0.0, 0.0],
        )

    def test_go_home_controller_cools_down_wrong_direction_axis(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "max_action": [0.3, 0.3, 0.3, 0.3],
                "wrong_direction_detection_s": 0.2,
                "wrong_direction_error_increase_rad": [0.02, 999.0, 999.0, 999.0],
                "wrong_direction_cooldown_s": [0.5, 0.0, 0.0, 0.0],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 10.0,
                "timeout_s": 5.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        start_obs = {
            "qpos": np.array([0.10, 0.0, 0.0, 0.0], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(start_obs)
        start = controller._start_s

        first = controller.update(start_obs, now_s=start)
        self.assertLess(first.action[0], 0.0)

        worsening = controller.update(
            {
                "qpos": np.array([0.14, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            },
            now_s=start + 0.3,
        )
        np.testing.assert_allclose(worsening.action, np.zeros(4, dtype=np.float32))
        self.assertEqual(worsening.diagnostics["go_home_axis_wrong_direction"][0], 1)

        cooling_down = controller.update(
            {
                "qpos": np.array([0.20, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            },
            now_s=start + 0.6,
        )
        np.testing.assert_allclose(cooling_down.action, np.zeros(4, dtype=np.float32))

    def test_go_home_controller_ramps_from_min_action_not_zero(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "p_gain": [10.0, 10.0, 10.0, 10.0],
                "min_action": [0.10, 0.10, 0.10, 0.10],
                "max_action": [0.30, 0.30, 0.30, 0.30],
                "action_ramp_rate": [0.10, 0.10, 0.10, 0.10],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.0,
                "timeout_s": 5.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        obs = {
            "qpos": np.array([0.20, 0.0, 0.0, 0.0], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(obs)
        start = controller._start_s

        first = controller.update(obs, now_s=start)
        np.testing.assert_allclose(first.action, [-0.10, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            first.diagnostics["go_home_action_ramp_limit"],
            [0.10, 0.0, 0.0, 0.0],
        )

        ramped = controller.update(obs, now_s=start + 0.5)
        np.testing.assert_allclose(ramped.action, [-0.15, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            ramped.diagnostics["go_home_action_ramp_limit"],
            [0.15, 0.0, 0.0, 0.0],
            atol=1e-6,
        )

        capped = controller.update(obs, now_s=start + 3.0)
        np.testing.assert_allclose(capped.action, [-0.30, 0.0, 0.0, 0.0])

    def test_go_home_controller_coasts_when_moving_toward_center(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.08, 0.08, 0.08, 0.08],
                "center_tolerance_rad": [0.02, 0.02, 0.02, 0.02],
                "resume_tolerance_rad": [0.06, 0.06, 0.06, 0.06],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "max_action": [0.2, 0.2, 0.2, 0.2],
                "coast_stop_time_s": [0.5, 0.5, 0.5, 0.5],
                "qvel_stable_rad_s": [0.2, 0.2, 0.2, 0.2],
                "dwell_s": 10.0,
                "timeout_s": 20.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        controller.start(
            {
                "qpos": np.array([0.10, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        start = controller._start_s
        result = controller.update(
            {
                "qpos": np.array([0.04, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.array([-0.05, 0.0, 0.0, 0.0], dtype=np.float32),
            },
            now_s=start + 0.1,
        )
        np.testing.assert_allclose(result.action, np.zeros(4, dtype=np.float32))
        self.assertEqual(result.diagnostics["go_home_axis_active"][0], 0)

        stalled = controller.update(
            {
                "qpos": np.array([0.04, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            },
            now_s=start + 0.2,
        )
        self.assertLess(stalled.action[0], 0.0)

    def test_go_home_controller_uses_low_speed_center_approach(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.08, 0.08, 0.08, 0.08],
                "center_tolerance_rad": [0.02, 0.02, 0.02, 0.02],
                "resume_tolerance_rad": [0.06, 0.06, 0.06, 0.06],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "min_action": [0.10, 0.10, 0.10, 0.10],
                "max_action": [0.2, 0.2, 0.2, 0.2],
                "coast_stop_time_s": [0.5, 0.5, 0.5, 0.5],
                "center_approach_action": [0.03, 0.0, 0.0, 0.0],
                "qvel_stable_rad_s": [0.2, 0.2, 0.2, 0.2],
                "dwell_s": 10.0,
                "timeout_s": 20.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        controller.start(
            {
                "qpos": np.array([0.10, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        result = controller.update(
            {
                "qpos": np.array([0.04, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.array([-0.05, 0.0, 0.0, 0.0], dtype=np.float32),
            },
            now_s=controller._start_s + 0.1,
        )
        np.testing.assert_allclose(result.action, [-0.03, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            result.diagnostics["go_home_center_approach_axis"],
            [1, 0, 0, 0],
        )

    def test_go_home_controller_filters_noisy_feedback_for_decisions(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.1, 0.1, 0.1, 0.1],
                "center_tolerance_rad": [0.05, 0.05, 0.05, 0.05],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "max_action": [0.2, 0.2, 0.2, 0.2],
                "qpos_filter_tau_s": [0.2, 0.2, 0.2, 0.2],
                "qvel_filter_tau_s": [0.2, 0.2, 0.2, 0.2],
                "qvel_stable_rad_s": [0.1, 0.1, 0.1, 0.1],
                "dwell_s": 10.0,
                "timeout_s": 20.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        start_obs = {
            "qpos": np.array([0.04, 0.0, 0.0, 0.0], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(start_obs)
        result = controller.update(
            {
                "qpos": np.array([0.20, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            },
            now_s=controller._start_s + 0.02,
        )

        self.assertLess(abs(result.diagnostics["go_home_error"][0]), 0.06)
        self.assertGreater(abs(result.diagnostics["go_home_raw_error"][0]), 0.19)

    def test_go_home_controller_delays_reactivation_after_coast_stop(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.08, 0.08, 0.08, 0.08],
                "center_tolerance_rad": [0.02, 0.02, 0.02, 0.02],
                "resume_tolerance_rad": [0.06, 0.06, 0.06, 0.06],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "max_action": [0.2, 0.2, 0.2, 0.2],
                "coast_stop_time_s": [0.5, 0.5, 0.5, 0.5],
                "coast_reactivation_delay_s": [0.3, 0.3, 0.3, 0.3],
                "qvel_stable_rad_s": [0.01, 0.01, 0.01, 0.01],
                "dwell_s": 10.0,
                "timeout_s": 20.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        controller.start(
            {
                "qpos": np.array([0.10, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        start = controller._start_s
        coasting = controller.update(
            {
                "qpos": np.array([0.04, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.array([-0.05, 0.0, 0.0, 0.0], dtype=np.float32),
            },
            now_s=start + 0.1,
        )
        np.testing.assert_allclose(coasting.action, np.zeros(4, dtype=np.float32))
        waiting = controller.update(
            {
                "qpos": np.array([0.04, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            },
            now_s=start + 0.2,
        )
        np.testing.assert_allclose(waiting.action, np.zeros(4, dtype=np.float32))

        reactivated = controller.update(
            {
                "qpos": np.array([0.04, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            },
            now_s=start + 0.5,
        )
        self.assertLess(reactivated.action[0], 0.0)

    def test_go_home_controller_pd_damps_motion_toward_home(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "d_gain": [0.5, 0.5, 0.5, 0.5],
                "max_action": [0.2, 0.2, 0.2, 0.2],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.0,
                "timeout_s": 1.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        obs = {
            "qpos": np.array([-0.1, 0.1, 0.0, 0.0], dtype=np.float32),
            "qvel": np.array([0.1, -0.1, 0.0, 0.0], dtype=np.float32),
        }
        controller.start(obs)
        result = controller.update(obs)
        np.testing.assert_allclose(result.action[:2], [0.05, -0.05], atol=1e-6)

    def test_go_home_controller_applies_control_signs(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "control_signs": [1.0, 1.0, -1.0, 1.0],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "min_action": [0.05, 0.05, 0.05, 0.05],
                "max_action": [0.2, 0.2, 0.2, 0.2],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.0,
                "timeout_s": 1.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        obs = {
            "qpos": np.array([0.0, 0.0, 0.2, 0.0], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(obs)
        result = controller.update(obs)
        self.assertGreater(result.action[2], 0.0)
        np.testing.assert_allclose(
            result.diagnostics["go_home_control_signs"],
            [1.0, 1.0, -1.0, 1.0],
        )

    def test_go_home_controller_uses_axis_hysteresis(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.08, 0.08, 0.08, 0.08],
                "center_tolerance_rad": [0.02, 0.02, 0.02, 0.02],
                "resume_tolerance_rad": [0.06, 0.06, 0.06, 0.06],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "max_action": [0.2, 0.2, 0.2, 0.2],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 10.0,
                "timeout_s": 20.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        controller.start(
            {
                "qpos": np.array([0.05, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        still_active = controller.update(
            {
                "qpos": np.array([0.03, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        self.assertLess(still_active.action[0], 0.0)
        centered = controller.update(
            {
                "qpos": np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        np.testing.assert_allclose(centered.action, np.zeros(4, dtype=np.float32))
        held_while_other_axis_active = controller.update(
            {
                "qpos": np.array([0.04, 0.07, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        self.assertEqual(
            held_while_other_axis_active.diagnostics["go_home_axis_active"][0], 0
        )
        self.assertLess(held_while_other_axis_active.action[1], 0.0)
        reactivated_to_avoid_deadlock = controller.update(
            {
                "qpos": np.array([0.04, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        self.assertLess(reactivated_to_avoid_deadlock.action[0], 0.0)
        resumed = controller.update(
            {
                "qpos": np.array([0.07, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            }
        )
        self.assertLess(resumed.action[0], 0.0)

    def test_go_home_controller_slew_limits_action_changes(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "p_gain": [10.0, 10.0, 10.0, 10.0],
                "max_action": [1.0, 1.0, 1.0, 1.0],
                "action_slew_rate": [1.0, 1.0, 1.0, 1.0],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 0.0,
                "timeout_s": 1.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        obs = {
            "qpos": np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(obs)
        result = controller.update(obs, now_s=controller._start_s)
        np.testing.assert_allclose(result.action, [-0.02, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            result.diagnostics["go_home_unsmoothed_action"],
            [-1.0, 0.0, 0.0, 0.0],
        )

    def test_go_home_controller_holds_control_decision_between_periods(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "p_gain": [1.0, 1.0, 1.0, 1.0],
                "max_action": [0.3, 0.3, 0.3, 0.3],
                "control_decision_period_s": 0.1,
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 10.0,
                "timeout_s": 5.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        start_obs = {
            "qpos": np.array([0.20, 0.0, 0.0, 0.0], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(start_obs)
        start = controller._start_s

        first = controller.update(start_obs, now_s=start)
        np.testing.assert_allclose(first.action, [-0.20, 0.0, 0.0, 0.0])
        self.assertEqual(first.diagnostics["go_home_decision_updated"], 1)

        closer_obs = {
            "qpos": np.array([0.05, 0.0, 0.0, 0.0], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        held = controller.update(closer_obs, now_s=start + 0.05)
        np.testing.assert_allclose(held.action, [-0.20, 0.0, 0.0, 0.0])
        self.assertEqual(held.diagnostics["go_home_decision_updated"], 0)

        refreshed = controller.update(closer_obs, now_s=start + 0.11)
        np.testing.assert_allclose(refreshed.action, [-0.05, 0.0, 0.0, 0.0])
        self.assertEqual(refreshed.diagnostics["go_home_decision_updated"], 1)

    def test_go_home_controller_blocks_small_rapid_sign_reversals(self) -> None:
        cfg = GoHomeConfig.from_mapping(
            {
                "enabled": True,
                "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
                "near_tolerance_rad": [0.5, 0.5, 0.5, 0.5],
                "success_tolerance_rad": [0.2, 0.2, 0.2, 0.2],
                "center_tolerance_rad": [0.01, 0.01, 0.01, 0.01],
                "resume_tolerance_rad": [0.02, 0.02, 0.02, 0.02],
                "p_gain": [10.0, 10.0, 10.0, 10.0],
                "max_action": [0.3, 0.3, 0.3, 0.3],
                "sign_reversal_delay_s": [0.5, 0.0, 0.0, 0.0],
                "sign_reversal_min_error_rad": [0.08, 0.0, 0.0, 0.0],
                "qvel_stable_rad_s": [0.02, 0.02, 0.02, 0.02],
                "dwell_s": 10.0,
                "timeout_s": 5.0,
            }
        )
        assert cfg is not None
        controller = GoHomeController(cfg)
        start_obs = {
            "qpos": np.array([0.10, 0.0, 0.0, 0.0], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        controller.start(start_obs)
        start = controller._start_s

        lowering = controller.update(start_obs, now_s=start)
        self.assertLess(lowering.action[0], 0.0)
        self.assertEqual(lowering.diagnostics["go_home_command_sign"][0], -1.0)

        small_rapid_reverse = controller.update(
            {
                "qpos": np.array([-0.05, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            },
            now_s=start + 0.1,
        )
        np.testing.assert_allclose(
            small_rapid_reverse.action,
            np.zeros(4, dtype=np.float32),
        )
        self.assertEqual(
            small_rapid_reverse.diagnostics["go_home_sign_reversal_blocked"][0],
            1,
        )

        same_direction_during_valve_settle = controller.update(
            {
                "qpos": np.array([0.10, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            },
            now_s=start + 0.4,
        )
        np.testing.assert_allclose(
            same_direction_during_valve_settle.action,
            np.zeros(4, dtype=np.float32),
        )

        small_late_reverse = controller.update(
            {
                "qpos": np.array([-0.05, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            },
            now_s=start + 0.6,
        )
        np.testing.assert_allclose(
            small_late_reverse.action,
            np.zeros(4, dtype=np.float32),
        )

        large_late_reverse = controller.update(
            {
                "qpos": np.array([-0.10, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
            },
            now_s=start + 1.2,
        )
        self.assertGreater(large_late_reverse.action[0], 0.0)
        self.assertEqual(
            large_late_reverse.diagnostics["go_home_sign_reversal_blocked"][0],
            0,
        )

    def test_go_home_may_start_from_armed_or_recording(self) -> None:
        from testbed.cli.record_real import _go_home_start_context

        self.assertEqual(_go_home_start_context("armed", False), "armed")
        self.assertEqual(_go_home_start_context("recording", True), "recording")
        self.assertIsNone(_go_home_start_context("armed", True))
        self.assertIsNone(_go_home_start_context("recording", False))
        self.assertIsNone(_go_home_start_context("go_home", True))
        self.assertIsNone(_go_home_start_context("fault", False))

    def test_go_home_step_diagnostics_have_full_episode_defaults(self) -> None:
        from testbed.cli.record_real import _ensure_go_home_step_diagnostics

        diagnostics: dict[str, object] = {}
        _ensure_go_home_step_diagnostics(diagnostics)

        self.assertEqual(diagnostics["go_home_running"], 0)
        self.assertEqual(diagnostics["go_home_result_code"], "")
        np.testing.assert_allclose(
            diagnostics["go_home_error"], np.zeros(4, dtype=np.float32)
        )
        np.testing.assert_allclose(
            diagnostics["go_home_commanded_action"], np.zeros(4, dtype=np.float32)
        )
        np.testing.assert_allclose(
            diagnostics["go_home_stall_action_boost"], np.zeros(4, dtype=np.float32)
        )
        np.testing.assert_allclose(
            diagnostics["go_home_axis_stalled"], np.zeros(4, dtype=np.int32)
        )
        np.testing.assert_allclose(
            diagnostics["go_home_axis_wrong_direction"],
            np.zeros(4, dtype=np.int32),
        )
        np.testing.assert_allclose(
            diagnostics["go_home_sign_reversal_blocked"],
            np.zeros(4, dtype=np.int32),
        )
        self.assertEqual(diagnostics["go_home_decision_updated"], 0)
        self.assertEqual(diagnostics["go_home_control_decision_period_s"], 0.0)

        diagnostics["go_home_running"] = 1
        _ensure_go_home_step_diagnostics(diagnostics)
        self.assertEqual(diagnostics["go_home_running"], 1)

    def test_remote_control_loop_polls_action_source_at_control_rate(self) -> None:
        from testbed.actions.base import ActionInfo
        from testbed.cli.record_real import _RemoteControlLoop
        from testbed.runtime.guard import ActionGuard

        class ConstantActionSource:
            def next_action(self, obs: dict) -> tuple[np.ndarray, ActionInfo]:
                return (
                    np.array([0.25, 0.0, 0.0, 0.0], dtype=np.float32),
                    ActionInfo(
                        source_type="teleop",
                        source_id="unit",
                        extras={"action_timestamp_ns": time.time_ns()},
                    ),
                )

        class RecordingPump:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.times: list[float] = []
                self.latest_result = ControlResult(
                    ack=True,
                    fault_code="init",
                    controller_timestamp_ns=time.time_ns(),
                    commanded_action=np.zeros(4, dtype=np.float32),
                    raw_low_level_command=np.zeros(4, dtype=np.float32),
                )

            def update_action(
                self, action: np.ndarray, *, state: dict | None = None
            ) -> ControlResult:
                now_ns = time.time_ns()
                with self.lock:
                    self.times.append(time.monotonic())
                self.latest_result = ControlResult(
                    ack=True,
                    fault_code="",
                    controller_timestamp_ns=now_ns,
                    commanded_action=np.asarray(action, dtype=np.float32).copy(),
                    raw_low_level_command=np.asarray(action, dtype=np.float32).copy(),
                )
                return self.latest_result

            def apply_status_toggle_mask(self, toggle_mask: int) -> bool:
                return True

        pump = RecordingPump()
        loop = _RemoteControlLoop(
            action_source=ConstantActionSource(),
            guard=ActionGuard(action_clip=1.0, max_delta=1.0, sensor_timeout_s=1.0),
            control_pump=pump,
            rate_hz=50.0,
            initial_obs={
                "qpos": np.zeros(4, dtype=np.float32),
                "safety_state": {},
                "sensor_timestamp_ns": time.time_ns(),
            },
        )
        loop.start()
        try:
            time.sleep(0.16)
        finally:
            loop.stop()
        with pump.lock:
            times = list(pump.times)
        self.assertGreaterEqual(len(times), 5)

    def test_remote_control_loop_fault_hold_forces_zero_output(self) -> None:
        from testbed.actions.base import ActionInfo
        from testbed.cli.record_real import _RemoteControlLoop
        from testbed.runtime.guard import ActionGuard

        class ConstantActionSource:
            def next_action(self, obs: dict) -> tuple[np.ndarray, ActionInfo]:
                return (
                    np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32),
                    ActionInfo(
                        source_type="teleop",
                        source_id="unit",
                        extras={
                            "action_timestamp_ns": time.time_ns(),
                            "toggle_mask": 1 << 5,
                        },
                    ),
                )

        class RecordingPump:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.actions: list[np.ndarray] = []
                self.toggle_masks: list[int] = []
                self.latest_result = ControlResult(
                    ack=True,
                    fault_code="",
                    controller_timestamp_ns=time.time_ns(),
                    commanded_action=np.zeros(4, dtype=np.float32),
                    raw_low_level_command=np.zeros(4, dtype=np.float32),
                )

            def update_action(
                self, action: np.ndarray, *, state: dict | None = None
            ) -> ControlResult:
                commanded = np.asarray(action, dtype=np.float32).copy()
                with self.lock:
                    self.actions.append(commanded)
                self.latest_result = ControlResult(
                    ack=True,
                    fault_code="",
                    controller_timestamp_ns=time.time_ns(),
                    commanded_action=commanded,
                    raw_low_level_command=commanded.copy(),
                )
                return self.latest_result

            def apply_status_toggle_mask(self, toggle_mask: int) -> bool:
                with self.lock:
                    self.toggle_masks.append(int(toggle_mask))
                return True

        pump = RecordingPump()
        loop = _RemoteControlLoop(
            action_source=ConstantActionSource(),
            guard=ActionGuard(action_clip=1.0, max_delta=1.0, sensor_timeout_s=1.0),
            control_pump=pump,
            rate_hz=50.0,
            initial_obs={
                "qpos": np.zeros(4, dtype=np.float32),
                "safety_state": {},
                "sensor_timestamp_ns": time.time_ns(),
            },
        )
        loop.set_fault_hold(True)
        loop.start()
        try:
            time.sleep(0.08)
        finally:
            loop.stop()

        with pump.lock:
            actions = list(pump.actions)
            toggle_masks = list(pump.toggle_masks)
        self.assertGreaterEqual(len(actions), 2)
        for action in actions:
            np.testing.assert_allclose(action, np.zeros(4, dtype=np.float32))
        self.assertGreaterEqual(len(toggle_masks), 1)
        self.assertTrue(all(mask == (1 << 5) for mask in toggle_masks))

    def test_remote_control_loop_releases_scripted_zero_hold(self) -> None:
        from testbed.actions.base import ActionInfo
        from testbed.cli.record_real import _RemoteControlLoop
        from testbed.runtime.guard import ActionGuard

        class ConstantActionSource:
            def next_action(self, obs: dict) -> tuple[np.ndarray, ActionInfo]:
                return (
                    np.array([0.5, -0.25, 0.0, 0.0], dtype=np.float32),
                    ActionInfo(
                        source_type="teleop",
                        source_id="unit",
                        extras={"action_timestamp_ns": time.time_ns()},
                    ),
                )

        class RecordingPump:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.actions: list[np.ndarray] = []
                self.latest_result = ControlResult(
                    ack=True,
                    fault_code="",
                    controller_timestamp_ns=time.time_ns(),
                    commanded_action=np.zeros(4, dtype=np.float32),
                    raw_low_level_command=np.zeros(4, dtype=np.float32),
                )

            def update_action(
                self, action: np.ndarray, *, state: dict | None = None
            ) -> ControlResult:
                commanded = np.asarray(action, dtype=np.float32).copy()
                with self.lock:
                    self.actions.append(commanded)
                self.latest_result = ControlResult(
                    ack=True,
                    fault_code="",
                    controller_timestamp_ns=time.time_ns(),
                    commanded_action=commanded,
                    raw_low_level_command=commanded.copy(),
                )
                return self.latest_result

            def apply_status_toggle_mask(self, toggle_mask: int) -> bool:
                return True

        pump = RecordingPump()
        loop = _RemoteControlLoop(
            action_source=ConstantActionSource(),
            guard=ActionGuard(action_clip=1.0, max_delta=1.0, sensor_timeout_s=1.0),
            control_pump=pump,
            rate_hz=50.0,
            initial_obs={
                "qpos": np.zeros(4, dtype=np.float32),
                "safety_state": {},
                "sensor_timestamp_ns": time.time_ns(),
            },
        )
        loop.set_scripted_action(np.zeros(4, dtype=np.float32))
        loop.set_fault_hold(True)
        loop.start()
        try:
            time.sleep(0.06)
            loop.set_scripted_action(None)
            loop.set_fault_hold(False)
            time.sleep(0.08)
        finally:
            loop.stop()

        with pump.lock:
            actions = list(pump.actions)
        self.assertTrue(any(np.allclose(action, 0.0) for action in actions))
        self.assertTrue(
            any(np.allclose(action, [0.5, -0.25, 0.0, 0.0]) for action in actions)
        )

    def test_record_real_remote_control_hold_helpers_release_zero_hold(self) -> None:
        from testbed.cli.record_real import (
            _hold_remote_control_zero,
            _release_remote_control,
        )

        class FakeRemoteLoop:
            def __init__(self) -> None:
                self.scripted_action: np.ndarray | None = None
                self.fault_hold = False

            def set_scripted_action(self, action: np.ndarray | None) -> None:
                self.scripted_action = None if action is None else action.copy()

            def set_fault_hold(self, enabled: bool) -> None:
                self.fault_hold = bool(enabled)

        loop = FakeRemoteLoop()
        _hold_remote_control_zero(loop)
        self.assertTrue(loop.fault_hold)
        np.testing.assert_allclose(
            loop.scripted_action, np.zeros(4, dtype=np.float32)
        )

        _release_remote_control(loop)
        self.assertFalse(loop.fault_hold)
        self.assertIsNone(loop.scripted_action)

    def test_receiver_health_fpv_stale_does_not_block_armed_control(self) -> None:
        from testbed.cli.record_real import _receiver_health_blocks_control

        fpv_stale = SimpleNamespace(ok=False, errors=("fpv_stale",))
        remote_stale = SimpleNamespace(ok=False, errors=("remote_stale",))

        self.assertFalse(
            _receiver_health_blocks_control(
                fpv_stale,
                receiver_mode="armed",
                has_record_session=False,
            )
        )
        self.assertFalse(
            _receiver_health_blocks_control(
                fpv_stale,
                receiver_mode="go_home",
                has_record_session=False,
            )
        )
        self.assertTrue(
            _receiver_health_blocks_control(
                fpv_stale,
                receiver_mode="recording",
                has_record_session=True,
            )
        )
        self.assertTrue(
            _receiver_health_blocks_control(
                remote_stale,
                receiver_mode="armed",
                has_record_session=False,
            )
        )

    def test_status_toggle_mask_semantics(self) -> None:
        status11 = [0] * STATUS_TOGGLE_BIT_COUNT
        apply_status_toggle_mask_to_status11(status11, 1 << 0)
        self.assertEqual(status11[0], 1)
        apply_status_toggle_mask_to_status11(status11, 1 << 0)
        self.assertEqual(status11[0], 0)
        apply_status_toggle_mask_to_status11(status11, 1 << 9)
        self.assertEqual(status11[9], 1)
        apply_status_toggle_mask_to_status11(status11, 1 << 9)
        self.assertEqual(status11[9], 2)

    def test_mock_controller_applies_status_toggle_mask(self) -> None:
        mock = MockLowLevelController()
        self.assertTrue(mock.apply_status_toggle_mask(1 << 2))
        self.assertEqual(mock.status11[2], 1)
        self.assertEqual(mock.last_toggle_mask, 1 << 2)

    def test_low_level_controllers_return_control_result(self) -> None:
        action = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)

        mock = MockLowLevelController()
        mock_result = mock.send(action, state={"qpos": np.zeros(4, dtype=np.float32)})
        self.assertTrue(mock_result.ack)
        self.assertEqual(mock_result.fault_code, "")
        np.testing.assert_allclose(mock_result.commanded_action, action)
        self.assertEqual(mock.send_count, 1)

        noop = NoopLowLevelController()
        noop_result = noop.send(action)
        self.assertTrue(noop_result.ack)
        self.assertEqual(noop_result.fault_code, "noop")
        np.testing.assert_allclose(noop_result.commanded_action, np.zeros(4, dtype=np.float32))
        self.assertEqual(noop.send_count, 1)

    def test_real_backend_mock_step_outputs_real_contract_fields(self) -> None:
        backend = RealExcavatorBackend(
            controller_mode="mock",
            control_hz=50.0,
            image_width=12,
            image_height=8,
        )
        try:
            ts = backend.start_episode(seed=0)
            np.testing.assert_allclose(ts.observation["qpos"], np.zeros(4, dtype=np.float32))
            action = np.array([0.1, 0.0, -0.1, 0.2], dtype=np.float32)
            ts_next = backend.step(action)
            obs = ts_next.observation
            self.assertEqual(obs["qpos"].shape, (4,))
            self.assertEqual(obs["qvel"].shape, (4,))
            self.assertEqual(obs["images"]["fpv"].shape, (8, 12, 3))
            self.assertIn("joint_timestamp_ns", obs)
            self.assertIn("image_timestamp_ns", obs)
            self.assertIn("sync_max_skew_ns", obs)
            self.assertIn("safety_state", obs)
            self.assertIn("control_result", ts_next.info)
            np.testing.assert_allclose(
                ts_next.info["control_result"]["commanded_action"],
                action,
            )
        finally:
            backend.close()

    def test_real_backend_accepts_injected_state_reader(self) -> None:
        state_reader = MockStateReader(
            image_width=5,
            image_height=4,
            velocity_scale_rad_s=1.0,
            image_latency_ns=100,
        )
        backend = RealExcavatorBackend(
            controller_mode="mock",
            state_reader=state_reader,
            sync_max_slop_ns=20,
            control_hz=10.0,
        )
        try:
            ts = backend.start_episode(seed=0)
            self.assertIs(backend.state_reader, state_reader)
            self.assertIn("fpv_skew_exceeds_slop", ts.observation["sync_warnings"])

            action = np.array([0.1, -0.1, 0.0, 0.2], dtype=np.float32)
            ts_next = backend.step(action)

            np.testing.assert_allclose(
                ts_next.observation["qvel"],
                action,
                rtol=1e-6,
                atol=1e-6,
            )
            self.assertEqual(ts_next.observation["images"]["fpv"].shape, (4, 5, 3))
        finally:
            backend.close()

    def test_bridge_mock_backend_uses_shared_command_state_boundary(self) -> None:
        backend = RealExcavatorBackend(
            controller_mode="bridge_mock",
            state_reader_mode="bridge_mock",
            control_hz=10.0,
            image_width=6,
            image_height=4,
            mock_velocity_scale_rad_s=1.0,
        )
        try:
            ts = backend.start_episode(seed=0)
            self.assertIsInstance(backend.controller, BridgeLowLevelController)
            self.assertIsInstance(backend.state_reader, BridgeStateReader)
            np.testing.assert_allclose(ts.observation["qpos"], np.zeros(4, dtype=np.float32))

            action = np.array([0.2, -0.1, 0.3, -0.4], dtype=np.float32)
            ts_next = backend.step(action)

            np.testing.assert_allclose(ts_next.observation["qvel"], action)
            np.testing.assert_allclose(ts_next.observation["qpos"], action * 0.1)
            self.assertEqual(ts_next.observation["env_state"].shape, (8,))
            self.assertEqual(ts_next.observation["images"]["fpv"].shape, (4, 6, 3))
            raw_low_level = ts_next.info["control_result"]["raw_low_level_command"]
            self.assertEqual(raw_low_level.shape, (8,))
            np.testing.assert_allclose(raw_low_level[:4], action)
            np.testing.assert_allclose(raw_low_level[4:], np.zeros(4, dtype=np.float32))
        finally:
            backend.close()

    def test_bridge_client_can_be_shared_explicitly(self) -> None:
        client = InProcessMockBridgeClient(
            image_width=5,
            image_height=3,
            velocity_scale_rad_s=0.5,
        )
        backend = RealExcavatorBackend(
            controller_mode="bridge_mock",
            state_reader_mode="bridge_mock",
            bridge_client=client,
            control_hz=20.0,
        )
        try:
            backend.start_episode(seed=0)
            backend.step(np.ones(4, dtype=np.float32) * 0.2)

            self.assertEqual(client.send_count, 1)
            self.assertGreaterEqual(client.read_count, 2)
            np.testing.assert_allclose(
                client.last_action,
                np.ones(4, dtype=np.float32) * 0.2,
            )
        finally:
            backend.close()

    def test_start_episode_skips_remote_reset_on_bridge_tcp(self) -> None:
        client = InProcessMockBridgeClient(
            image_width=4,
            image_height=3,
            velocity_scale_rad_s=0.5,
        )
        client.send_count = 3
        backend = RealExcavatorBackend(
            controller_mode="bridge_tcp",
            state_reader_mode="bridge_tcp",
            bridge_client=client,
            control_hz=20.0,
        )
        try:
            backend.start_episode(seed=0)
            self.assertEqual(client.send_count, 3)
        finally:
            backend.close()

    def test_bridge_protocol_round_trips_control_and_state_samples(self) -> None:
        result = ControlResult(
            ack=True,
            fault_code="",
            controller_timestamp_ns=123,
            commanded_action=np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
            raw_low_level_command=np.arange(8, dtype=np.float32),
        )
        payload = control_result_to_payload(result)
        decoded_result = control_result_from_payload(payload)

        self.assertTrue(decoded_result.ack)
        self.assertEqual(decoded_result.controller_timestamp_ns, 123)
        np.testing.assert_allclose(decoded_result.commanded_action, result.commanded_action)
        np.testing.assert_allclose(
            decoded_result.raw_low_level_command,
            result.raw_low_level_command,
        )

        samples = RealStateSamples(
            joint=TimestampedSample(
                timestamp_ns=1_000,
                payload={
                    "qpos": np.arange(4, dtype=np.float32),
                    "qvel": np.arange(4, dtype=np.float32) * 0.1,
                    "status": np.arange(6, dtype=np.int32),
                    "env_state": np.arange(8, dtype=np.float32),
                    "imu_health": {
                        "online": [1, 0, 1, 1],
                        "valid_attitude": [1, 1, 1, 1],
                        "valid_gyro": [1, 1, 1, 1],
                        "valid_accel": [1, 1, 1, 1],
                        "packet_loss_count": [0, 3, 0, 0],
                        "host_rx_age_ms": [5.0, 6.0, 5.5, 4.0],
                    },
                },
                source="joint",
                receive_time_ns=1_010,
            ),
            images={
                "fpv": TimestampedSample(
                    timestamp_ns=1_020,
                    payload=np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
                    source="fpv",
                    receive_time_ns=1_030,
                )
            },
        )
        decoded_samples = state_samples_from_payload(state_samples_to_payload(samples))

        np.testing.assert_allclose(decoded_samples.joint.payload["qpos"], np.arange(4))
        self.assertEqual(
            decoded_samples.joint.payload["imu_health"]["online"],
            [1, 0, 1, 1],
        )
        np.testing.assert_array_equal(
            decoded_samples.images["fpv"].payload,
            np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        )

        if HAS_CV2:
            import cv2

            rgb = np.zeros((4, 5, 3), dtype=np.uint8)
            rgb[..., 0] = 200
            ok, jpeg = cv2.imencode(
                ".jpg",
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            self.assertTrue(ok)
            jpeg_samples = RealStateSamples(
                joint=samples.joint,
                images={
                    "fpv": TimestampedSample(
                        timestamp_ns=1_020,
                        payload={
                            "encoding": "jpeg",
                            "shape": rgb.shape,
                            "data": jpeg.reshape(-1),
                        },
                        source="fpv",
                        receive_time_ns=1_030,
                    )
                },
            )
            decoded_jpeg = state_samples_from_payload(state_samples_to_payload(jpeg_samples))
            jpeg_payload = decoded_jpeg.images["fpv"].payload
            self.assertEqual(jpeg_payload["encoding"], "jpeg")
            self.assertEqual(jpeg_payload["shape"], (4, 5, 3))
            self.assertGreater(len(jpeg_payload["data"]), 0)

        frame = encode_frame({"type": "ping.request", "payload": {"value": np.int64(7)}})
        self.assertEqual(decode_frame(frame)["payload"]["value"], 7)

    def test_json_tcp_bridge_client_talks_to_loopback_server(self) -> None:
        if not _can_bind_loopback_socket():
            self.skipTest("loopback socket bind is blocked in this environment")

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = int(server.getsockname()[1])
        errors: list[BaseException] = []

        def serve_once() -> None:
            try:
                conn, _addr = server.accept()
                with conn, conn.makefile("rwb") as stream:
                    while True:
                        line = stream.readline()
                        if not line:
                            break
                        message = decode_frame(line)
                        message_type = message["type"]
                        if message_type == "reset.request":
                            payload = {}
                        elif message_type == "send_action.request":
                            action = np.asarray(message["payload"]["action"], dtype=np.float32)
                            payload = control_result_to_payload(
                                ControlResult(
                                    ack=True,
                                    fault_code="",
                                    controller_timestamp_ns=2_000,
                                    commanded_action=action,
                                    raw_low_level_command=action4_to_speed_scalar8(action),
                                )
                            )
                        elif message_type == "send_status.request":
                            payload = {
                                "ack": True,
                                "toggle_mask": int(message["payload"].get("toggle_mask", 0)),
                            }
                        elif message_type == "read_state.request":
                            payload = state_samples_to_payload(
                                RealStateSamples(
                                    joint=TimestampedSample(
                                        timestamp_ns=3_000,
                                        payload={
                                            "qpos": np.ones(4, dtype=np.float32),
                                            "qvel": np.zeros(4, dtype=np.float32),
                                            "status": np.zeros(4, dtype=np.int32),
                                        },
                                        source="joint",
                                    ),
                                    images={
                                        "fpv": TimestampedSample(
                                            timestamp_ns=3_010,
                                            payload=np.zeros((2, 2, 3), dtype=np.uint8),
                                            source="fpv",
                                        )
                                    },
                                )
                            )
                        elif message_type == "close.request":
                            response = response_message("close.response", {})
                            stream.write(encode_frame(response))
                            stream.flush()
                            break
                        else:
                            response = response_message(
                                message_type.replace(".request", ".response"),
                                {},
                                ok=False,
                                error=f"unexpected {message_type}",
                            )
                            stream.write(encode_frame(response))
                            stream.flush()
                            continue

                        response = response_message(
                            message_type.replace(".request", ".response"),
                            payload,
                        )
                        stream.write(encode_frame(response))
                        stream.flush()
            except BaseException as exc:
                errors.append(exc)
            finally:
                server.close()

        thread = threading.Thread(target=serve_once, daemon=True)
        thread.start()
        client = JsonTcpBridgeClient(port=port, timeout_s=1.0)
        try:
            client.reset(seed=0)
            result = client.send_action(np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32))
            status = client.apply_status_toggle_mask(1 << 1)
            samples = client.read_state(step_id=1, action_timestamp_ns=2_000)
            self.assertTrue(result.ack)
            self.assertTrue(status)
            np.testing.assert_allclose(result.commanded_action, [0.1, 0.2, -0.3, 0.4])
            np.testing.assert_allclose(samples.joint.payload["qpos"], np.ones(4))
            self.assertEqual(samples.images["fpv"].payload.shape, (2, 2, 3))
        finally:
            client.close()
            thread.join(timeout=2.0)
        self.assertFalse(errors)

    def test_json_tcp_bridge_client_can_preserve_application_error_connection(self) -> None:
        if not _can_bind_loopback_socket():
            self.skipTest("loopback socket bind is blocked in this environment")

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = int(server.getsockname()[1])
        errors: list[BaseException] = []

        def serve_once() -> None:
            try:
                conn, _addr = server.accept()
                with conn, conn.makefile("rwb") as stream:
                    line = stream.readline()
                    self.assertTrue(line)
                    message = decode_frame(line)
                    self.assertEqual(message["type"], "read_state.request")
                    stream.write(
                        encode_frame(
                            response_message(
                                "read_state.response",
                                {},
                                ok=False,
                                error="temporary state timeout",
                            )
                        )
                    )
                    stream.flush()

                    line = stream.readline()
                    self.assertTrue(line)
                    message = decode_frame(line)
                    self.assertEqual(message["type"], "send_action.request")
                    action = np.asarray(message["payload"]["action"], dtype=np.float32)
                    payload = control_result_to_payload(
                        ControlResult(
                            ack=True,
                            fault_code="",
                            controller_timestamp_ns=4_000,
                            commanded_action=action,
                            raw_low_level_command=action4_to_speed_scalar8(action),
                        )
                    )
                    stream.write(encode_frame(response_message("send_action.response", payload)))
                    stream.flush()
            except BaseException as exc:
                errors.append(exc)
            finally:
                server.close()

        thread = threading.Thread(target=serve_once, daemon=True)
        thread.start()
        client = JsonTcpBridgeClient(port=port, timeout_s=1.0)
        try:
            response = client._request_response("read_state", {"step_id": 1})
            self.assertFalse(response["ok"])
            result = client.send_action(np.zeros(4, dtype=np.float32))
            self.assertTrue(result.ack)
        finally:
            client.force_close()
            thread.join(timeout=2.0)
        self.assertFalse(errors)

    def test_apply_data_side_slave_defaults(self) -> None:
        from testbed.cli.data_side import apply_data_side_config

        cfg = {"real": {"bridge": {"host": "10.0.0.2", "port": 0}}, "task": {}}
        side = apply_data_side_config(cfg, data_side="slave")
        self.assertEqual(side, "slave")
        self.assertEqual(cfg["task"]["dataset_dir"], "/data/real_teleop_v1")
        self.assertEqual(cfg["real"]["bridge"]["host"], "127.0.0.1")
        self.assertEqual(cfg["real"]["bridge"]["port"], 8765)

    def test_apply_data_side_host_respects_cli_overrides(self) -> None:
        from testbed.cli.data_side import apply_data_side_config

        cfg = {"real": {"bridge": {"port": 0}}, "task": {"dataset_dir": "keep"}}
        side = apply_data_side_config(
            cfg,
            data_side="host",
            cli_output_dir="/custom/out",
            cli_bridge_host="192.168.1.50",
        )
        self.assertEqual(side, "host")
        self.assertEqual(cfg["task"]["dataset_dir"], "keep")
        self.assertNotIn("host", cfg["real"]["bridge"])

        cfg2 = {"real": {"bridge": {"port": 0}}, "task": {}}
        apply_data_side_config(cfg2, data_side="host")
        self.assertEqual(cfg2["task"]["dataset_dir"], "data/real_teleop_v1")

    def test_apply_data_side_defaults_to_slave_when_unset(self) -> None:
        from testbed.cli.data_side import apply_data_side_config

        cfg: dict = {"real": {"bridge": {"port": 0}}, "task": {}}
        with patch.dict(os.environ, {}, clear=True):
            side = apply_data_side_config(cfg)
        self.assertEqual(side, "slave")
        self.assertEqual(cfg["task"]["dataset_dir"], "/data/real_teleop_v1")
        self.assertEqual(cfg["real"]["data_side"], "slave")

    def test_apply_data_side_uses_yaml_when_cli_unset(self) -> None:
        from testbed.cli.data_side import apply_data_side_config

        cfg: dict = {"real": {"data_side": "host", "bridge": {"port": 0}}, "task": {}}
        with patch.dict(os.environ, {}, clear=True):
            side = apply_data_side_config(cfg, data_side=None)
        self.assertEqual(side, "host")
        self.assertEqual(cfg["task"]["dataset_dir"], "data/real_teleop_v1")
        self.assertEqual(cfg["real"]["bridge"]["host"], "192.168.100.1")

    def test_apply_data_side_uses_env_when_cli_and_yaml_unset(self) -> None:
        from testbed.cli.data_side import apply_data_side_config

        cfg: dict = {"real": {"bridge": {"port": 0}}, "task": {}}
        with patch.dict(os.environ, {"EXCAVATOR_DATA_SIDE": "host"}, clear=True):
            side = apply_data_side_config(cfg, data_side=None)
        self.assertEqual(side, "host")
        self.assertEqual(cfg["real"]["data_side"], "host")

    def test_apply_data_side_cli_overrides_env_and_yaml(self) -> None:
        from testbed.cli.data_side import apply_data_side_config

        cfg: dict = {"real": {"data_side": "host", "bridge": {"port": 0}}, "task": {}}
        with patch.dict(os.environ, {"EXCAVATOR_DATA_SIDE": "host"}, clear=True):
            side = apply_data_side_config(cfg, data_side="slave")
        self.assertEqual(side, "slave")
        self.assertEqual(cfg["task"]["dataset_dir"], "/data/real_teleop_v1")

    def test_apply_data_side_invalid_raises(self) -> None:
        from testbed.cli.data_side import apply_data_side_config

        with self.assertRaises(ValueError):
            apply_data_side_config({}, data_side="edge")

    def test_ssh_qc_watcher_parses_remote_find_output_and_waits_stable(self) -> None:
        from testbed.cli import dataset_qc_watch_ssh as watcher

        episodes = watcher.parse_remote_find_output(
            ".episode_0.hdf5.tmp.1\t10\t1.0\n"
            "episode_2.hdf5\t20\t2.0\n"
            "episode_1.hdf5\t10\t1.0\n"
            "failed_episode_3.hdf5\t30\t3.0\n"
        )
        self.assertEqual([episode.name for episode in episodes], ["episode_1.hdf5", "episode_2.hdf5"])

        samples = iter(
            [
                watcher.RemoteEpisode("episode_1.hdf5", 10, 1.0),
                watcher.RemoteEpisode("episode_1.hdf5", 10, 1.0),
            ]
        )
        with patch.object(watcher, "stat_remote_episode", side_effect=lambda **_kwargs: next(samples)):
            stable = watcher.wait_until_remote_stable(
                target="host",
                remote_dir="/data",
                name="episode_1.hdf5",
                checks=2,
                interval_s=0.0,
            )
        assert stable is not None
        self.assertEqual(stable.size, 10)

    def test_record_real_builds_bridge_client_from_tcp_config(self) -> None:
        from testbed.cli.record_real import _build_bridge_client, _build_control_pump_client

        self.assertIsNone(_build_bridge_client({}, "mock", "mock"))

        client = _build_bridge_client(
            {
                "bridge": {
                    "host": "127.0.0.2",
                    "port": 12345,
                    "timeout_s": 0.25,
                }
            },
            "bridge_tcp",
            "mock",
        )
        try:
            self.assertIsInstance(client, JsonTcpBridgeClient)
            self.assertEqual(client.host, "127.0.0.2")
            self.assertEqual(client.port, 12345)
            self.assertEqual(client.timeout_s, 0.25)
        finally:
            client.close()

        with self.assertRaises(ValueError):
            _build_bridge_client({"bridge": {"port": 0}}, "bridge_tcp", "bridge_tcp")

        pump_client = _build_control_pump_client(
            {
                "bridge": {
                    "host": "127.0.0.1",
                    "port": 8765,
                    "timeout_s": 1.0,
                }
            },
            {
                "bridge": {
                    "host": "127.0.0.1",
                    "port": 8766,
                    "timeout_s": 0.2,
                }
            },
        )
        try:
            self.assertIsInstance(pump_client, JsonTcpBridgeClient)
            self.assertEqual(pump_client.host, "127.0.0.1")
            self.assertEqual(pump_client.port, 8766)
            self.assertEqual(pump_client.timeout_s, 0.2)
        finally:
            pump_client.close()

    def test_record_real_live_action_line_shows_raw_and_send(self) -> None:
        from testbed.cli.record_real import _LiveActionLine

        stream = io.StringIO()
        live_line = _LiveActionLine(enabled=True)
        with patch("sys.stdout", stream):
            live_line.update(
                step=3,
                raw_action=np.array([0.1, -0.2, 0.0, 0.4], dtype=np.float32),
                safe_action=np.zeros(4, dtype=np.float32),
                action_info=SimpleNamespace(extras={}),
                sensor_age_s=0.0123,
                control_result={
                    "ack": True,
                    "fault_code": "",
                    "commanded_action": np.array([0.1, -0.1, 0.0, 0.2], dtype=np.float32),
                },
                guard_reasons=("action_clip",),
            )

        output = stream.getvalue()
        self.assertIn("raw=[+0.100,-0.200,+0.000,+0.400]", output)
        self.assertIn("send=[+0.100,-0.100,+0.000,+0.200]", output)
        self.assertIn("hz=", output)
        self.assertIn("ctl_ms=", output)
        self.assertNotIn("host_now_ns", output)
        self.assertNotIn("sensor_timestamp_ns", output)

    def test_record_real_live_action_line_shows_go_home_status(self) -> None:
        from testbed.cli.record_real import _LiveActionLine

        stream = io.StringIO()
        live_line = _LiveActionLine(enabled=True)
        go_home_update = SimpleNamespace(
            done=False,
            failed=False,
            reason="",
            diagnostics={
                "go_home_error": np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
                "go_home_qvel": np.array([0.0, 0.01, 0.0, -0.01], dtype=np.float32),
                "go_home_commanded_action": np.array(
                    [0.12, -0.2, 0.2, -0.2], dtype=np.float32
                ),
                "go_home_elapsed_s": 2.5,
                "go_home_in_position": 0,
                "go_home_stable_velocity": 1,
            },
        )
        with patch("sys.stdout", stream):
            live_line.update(
                step=4,
                mode="go_home",
                raw_action=np.zeros(4, dtype=np.float32),
                safe_action=np.zeros(4, dtype=np.float32),
                action_info=SimpleNamespace(extras={}),
                sensor_age_s=0.01,
                control_result={"ack": True, "fault_code": ""},
                guard_reasons=(),
                go_home_update=go_home_update,
            )

        output = stream.getvalue()
        self.assertIn("home=going_home", output)
        self.assertIn("t=2.5s", output)
        self.assertIn("maxerr=0.400", output)
        self.assertIn("maxaxis=bucket", output)
        self.assertIn("inpos=0", output)
        self.assertIn("stable=1", output)

    def test_record_real_qc_dashboard_shows_fixed_rows_and_dynamic_qc_fields(
        self,
    ) -> None:
        from testbed.cli.record_real import ReceiverHealthSnapshot, _QcDashboard
        from testbed.data.online_qc import OnlineQcSnapshot

        receiver_health = ReceiverHealthSnapshot(
            ok=True,
            error_code="",
            errors=(),
            imu_summary="1111",
            diagnostics={
                "fpv_age_ms": 12.5,
                "bridge_snapshot_age_ms": 8.0,
            },
        )
        online_qc = OnlineQcSnapshot(
            status="WARN_MASK",
            error_code="",
            warning_codes=("qpos_outside_p5_p95", "fpv_drift"),
            train_exclude=True,
            diagnostics={
                "online_qc_qpos_warn_count": 37,
                "online_qc_qpos_fail_count": 4,
                "online_qc_fpv_drift_score": 1.16,
                "online_qc_fpv_drift_count": 12,
                "online_qc_healthy_fraction": 0.42,
                "online_qc_healthy_steps": 425,
                "online_qc_train_exclude_steps": 609,
                "online_qc_train_ready_candidate": 0,
                "online_qc_bucket_reference_status": "WARN",
                "online_qc_bucket_ref_high_margin": -0.10,
                "online_qc_bucket_semantic_decision": "review",
                "online_qc_bucket_semantic_notes": "bucket_min_too_shallow",
                "train_exclude_mask": 1,
            },
        )

        stream = io.StringIO()
        dashboard = _QcDashboard(enabled=True, max_events=3)
        with patch("sys.stdout", stream):
            dashboard.update(
                mode="recording",
                episode_idx=7,
                saved=2,
                record_steps=123,
                receiver_health=receiver_health,
                online_qc=online_qc,
                action_info=SimpleNamespace(
                    extras={"remote_action_seq": 41, "remote_action_stale": 0}
                ),
                message="recording",
            )

        output = stream.getvalue()
        self.assertIn("mode=recording", output)
        self.assertIn("episode=7", output)
        self.assertIn("steps=123", output)
        self.assertIn("saved=2", output)
        self.assertIn("qc=WARN_MASK", output)
        self.assertIn("CHECK", output)
        self.assertIn("STATE", output)
        self.assertIn("CODE/WARN", output)
        self.assertIn("VALUE", output)
        self.assertIn("COUNT", output)
        self.assertIn("ACTION", output)
        self.assertIn("receiver_health", output)
        self.assertIn("qpos_distribution", output)
        self.assertIn("bucket_reference", output)
        self.assertIn("bucket_semantic", output)
        self.assertIn("fpv_drift", output)
        self.assertIn("episode_final", output)
        self.assertIn("qpos_outside_p5_p95", output)
        self.assertIn("fpv_drift", output)
        self.assertIn("bucket_min_too_shallow", output)
        self.assertIn("candidate=0", output)
        self.assertIn("decision=review", output)
        self.assertIn("warn=37", output)
        self.assertIn("fail=4", output)
        self.assertIn("score=1.16", output)
        self.assertIn("healthy=425", output)
        self.assertIn("mask=609", output)

    def test_record_real_qc_dashboard_keeps_recent_warning_events(self) -> None:
        from testbed.cli.record_real import ReceiverHealthSnapshot, _QcDashboard
        from testbed.data.online_qc import OnlineQcSnapshot

        dashboard = _QcDashboard(enabled=True, max_events=2)
        dashboard.add_event("INFO", "first")
        dashboard.add_event("WARN", "second")
        dashboard.add_event("FAIL", "third")
        receiver_health = ReceiverHealthSnapshot(
            ok=True,
            error_code="",
            errors=(),
            imu_summary="1111",
            diagnostics={},
        )
        online_qc = OnlineQcSnapshot(
            status="PASS",
            error_code="",
            warning_codes=(),
            train_exclude=False,
            diagnostics={},
        )

        stream = io.StringIO()
        with patch("sys.stdout", stream):
            dashboard.update(
                mode="armed",
                episode_idx=3,
                saved=1,
                record_steps=0,
                receiver_health=receiver_health,
                online_qc=online_qc,
                action_info=SimpleNamespace(extras={}),
                message="ready",
            )

        output = stream.getvalue()
        self.assertNotIn("first", output)
        self.assertIn("second", output)
        self.assertIn("third", output)

    def test_record_real_qc_event_log_writes_only_warn_fail_events(self) -> None:
        from testbed.cli.record_real import ReceiverHealthSnapshot, _QcDashboard, _QcEventLogger
        from testbed.data.online_qc import OnlineQcSnapshot

        with tempfile.TemporaryDirectory() as tmpdir:
            event_path = Path(tmpdir) / "events.jsonl"
            logger = _QcEventLogger(event_path)
            dashboard = _QcDashboard(
                enabled=True,
                max_events=3,
                event_logger=logger,
            )
            receiver_health = ReceiverHealthSnapshot(
                ok=True,
                error_code="",
                errors=(),
                imu_summary="1111",
                diagnostics={},
            )
            warn_snapshot = OnlineQcSnapshot(
                status="WARN_MASK",
                error_code="",
                warning_codes=("bucket_semantic_review",),
                train_exclude=True,
                diagnostics={
                    "online_qc_reference_id": "ref-events",
                    "online_qc_bucket_semantic_decision": "review",
                },
            )
            pass_snapshot = OnlineQcSnapshot(
                status="PASS",
                error_code="",
                warning_codes=(),
                train_exclude=False,
                diagnostics={"online_qc_reference_id": "ref-events"},
            )
            stream = io.StringIO()
            with patch("sys.stdout", stream):
                dashboard.update(
                    mode="recording",
                    episode_idx=9,
                    saved=1,
                    record_steps=22,
                    receiver_health=receiver_health,
                    online_qc=warn_snapshot,
                    action_info=SimpleNamespace(extras={}),
                    message="warn",
                )
                dashboard.update(
                    mode="recording",
                    episode_idx=9,
                    saved=1,
                    record_steps=23,
                    receiver_health=receiver_health,
                    online_qc=pass_snapshot,
                    action_info=SimpleNamespace(extras={}),
                    message="pass",
                )
            logger.close()

            events = [
                json.loads(line)
                for line in event_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event["episode_idx"], 9)
            self.assertEqual(event["record_steps"], 22)
            self.assertEqual(event["mode"], "recording")
            self.assertEqual(event["level"], "WARN")
            self.assertEqual(event["code"], "bucket_semantic_review")
            self.assertEqual(event["online_qc_status"], "WARN_MASK")
            self.assertEqual(event["online_qc_reference_id"], "ref-events")
            self.assertEqual(
                event["diagnostics"]["online_qc_bucket_semantic_decision"],
                "review",
            )

    def test_record_real_qc_dashboard_marks_fpv_frame_hard_fail(self) -> None:
        from testbed.cli.record_real import ReceiverHealthSnapshot, _QcDashboard
        from testbed.data.online_qc import OnlineQcSnapshot

        receiver_health = ReceiverHealthSnapshot(
            ok=True,
            error_code="",
            errors=(),
            imu_summary="1111",
            diagnostics={},
        )
        online_qc = OnlineQcSnapshot(
            status="FAIL_EPISODE",
            error_code="fpv_duplicate",
            warning_codes=(),
            train_exclude=True,
            diagnostics={},
        )
        dashboard = _QcDashboard(enabled=True)

        output = dashboard.render(
            mode="recording",
            episode_idx=4,
            saved=0,
            record_steps=12,
            receiver_health=receiver_health,
            online_qc=online_qc,
            action_info=SimpleNamespace(extras={}),
            message="",
        )

        fpv_frame_line = next(
            line for line in output.splitlines() if line.startswith("fpv_frame")
        )
        self.assertIn("FAIL", fpv_frame_line)
        self.assertIn("fpv_duplicate", fpv_frame_line)
        self.assertIn("fail", fpv_frame_line)

    def test_json_tcp_bridge_mock_server_updates_state(self) -> None:
        if not _can_bind_loopback_socket():
            self.skipTest("loopback socket bind is blocked in this environment")

        server = JsonTcpBridgeMockServer(
            port=0,
            dt=0.1,
            image_width=3,
            image_height=2,
            velocity_scale_rad_s=1.0,
            one_shot=True,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.assertTrue(server.wait_until_ready(timeout_s=2.0))
        assert server.bound_port is not None

        client = JsonTcpBridgeClient(port=server.bound_port, timeout_s=1.0)
        try:
            client.reset(seed=0)
            client.apply_status_toggle_mask(1 << 3)
            result = client.send_action(np.array([0.4, -0.2, 0.0, 0.1], dtype=np.float32))
            samples = client.read_state(step_id=1, action_timestamp_ns=result.controller_timestamp_ns)

            self.assertTrue(result.ack)
            self.assertEqual(server.client.status_toggle_count, 1)
            self.assertEqual(server.client._status11[3], 1)
            np.testing.assert_allclose(samples.joint.payload["qvel"], [0.4, -0.2, 0.0, 0.1])
            np.testing.assert_allclose(samples.joint.payload["qpos"], [0.04, -0.02, 0.0, 0.01])
            self.assertEqual(samples.images["fpv"].payload.shape, (2, 3, 3))
        finally:
            client.close()
            thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())

    def test_gateway_reads_fpv_shm_once_when_fresh(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        ros2_bridge_path = str(repo_root / "ros2_bridge")
        sys.path.insert(0, ros2_bridge_path)
        try:
            from excavator_bridge_gateway.gateway_server import _fpv_sample_from_shm
        finally:
            sys.path.remove(ros2_bridge_path)

        class Reader:
            def __init__(self) -> None:
                self.calls = 0

            def read_latest(self):
                self.calls += 1
                return SimpleNamespace(
                    timestamp_ns=time.time_ns(),
                    receive_time_ns=time.time_ns(),
                    height=2,
                    width=3,
                    rgb=np.arange(18, dtype=np.uint8).tobytes(),
                )

        reader = Reader()
        sample = _fpv_sample_from_shm(
            reader,
            max_stale_ms=1000,
            placeholder_width=8,
            placeholder_height=6,
            frame_id=1,
            fpv_source="auto",
            fpv_encoding="raw",
            jpeg_quality=95,
        )

        self.assertEqual(reader.calls, 1)
        self.assertEqual(sample["source"], "ros2_compressed_fpv")
        self.assertEqual(sample["payload"]["shape"], [2, 3, 3])

    @unittest.skipUnless(HAS_CV2, "cv2 is required for JPEG gateway tests")
    def test_gateway_can_return_jpeg_fpv_payload(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        ros2_bridge_path = str(repo_root / "ros2_bridge")
        sys.path.insert(0, ros2_bridge_path)
        try:
            from excavator_bridge_gateway.gateway_server import _fpv_sample_from_shm
        finally:
            sys.path.remove(ros2_bridge_path)

        class Reader:
            def read_latest(self):
                return SimpleNamespace(
                    timestamp_ns=time.time_ns(),
                    receive_time_ns=time.time_ns(),
                    height=32,
                    width=32,
                    rgb=np.arange(32 * 32 * 3, dtype=np.uint8).tobytes(),
                )

        sample = _fpv_sample_from_shm(
            Reader(),
            max_stale_ms=1000,
            placeholder_width=8,
            placeholder_height=6,
            frame_id=1,
            fpv_source="auto",
            fpv_encoding="jpeg",
            jpeg_quality=95,
        )

        self.assertEqual(sample["payload"]["encoding"], "jpeg")
        self.assertEqual(sample["payload"]["shape"], [32, 32, 3])
        self.assertLess(len(sample["payload"]["data_b64"]), 32 * 32 * 3 * 4 // 3)

    @unittest.skipUnless(HAS_CV2, "cv2 is required for JPEG gateway tests")
    def test_gateway_jpeg_cache_encodes_each_sequence_once(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        ros2_bridge_path = str(repo_root / "ros2_bridge")
        sys.path.insert(0, ros2_bridge_path)
        try:
            import excavator_bridge_gateway.gateway_server as gateway_server
        finally:
            sys.path.remove(ros2_bridge_path)

        class Reader:
            def __init__(self) -> None:
                self.sequence = 1
                self.calls = 0

            def read_latest(self):
                self.calls += 1
                return SimpleNamespace(
                    timestamp_ns=time.time_ns(),
                    receive_time_ns=time.time_ns(),
                    sequence=self.sequence,
                    height=4,
                    width=4,
                    rgb=np.zeros(4 * 4 * 3, dtype=np.uint8).tobytes(),
                )

        reader = Reader()
        encode_count = 0
        original_fpv_payload = gateway_server._fpv_payload

        def fake_fpv_payload(**kwargs):
            nonlocal encode_count
            encode_count += 1
            return {
                "encoding": "jpeg",
                "shape": [kwargs["height"], kwargs["width"], 3],
                "data_b64": "abcd",
            }

        cache = gateway_server.FpvPayloadCache(
            reader,
            fpv_source="auto",
            fpv_encoding="jpeg",
            jpeg_quality=95,
            max_encode_hz=1000.0,
        )
        gateway_server._fpv_payload = fake_fpv_payload
        try:
            cache.start()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and encode_count < 1:
                time.sleep(0.01)
            self.assertEqual(encode_count, 1)
            time.sleep(0.05)
            self.assertEqual(encode_count, 1)
            reader.sequence = 2
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and encode_count < 2:
                time.sleep(0.01)
            self.assertEqual(encode_count, 2)
            cached = cache.latest(max_stale_ms=1000)
            self.assertIsNotNone(cached)
            self.assertEqual(cached["payload"]["encoding"], "jpeg")
        finally:
            cache.stop()
            gateway_server._fpv_payload = original_fpv_payload

    @unittest.skipUnless(HAS_H5PY, "h5py is required for HDF5 round-trip tests")
    def test_hdf5_real_metadata_and_diagnostics_round_trip(self) -> None:
        from testbed.data.hdf5_io import read_episode, write_episode

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episode_0.hdf5"
            write_episode(
                path,
                qpos=np.zeros((2, 4), dtype=np.float32),
                qvel=np.zeros((2, 4), dtype=np.float32),
                actions=np.zeros((2, 4), dtype=np.float32),
                images={"fpv": np.zeros((2, 4, 4, 3), dtype=np.uint8)},
                rewards=np.zeros(2, dtype=np.float32),
                metadata={
                    "is_real": True,
                    "platform": "real_excavator",
                    "qpos_units": "rad",
                    "qvel_units": "rad/s",
                    "hydraulic_cylinder_available": False,
                },
                diagnostics=_real_diagnostics(2),
            )

            episode = read_episode(path)
            self.assertTrue(episode["is_real"])
            self.assertEqual(episode["metadata"]["platform"], "real_excavator")
            self.assertEqual(set(episode["diagnostics"]), set(_real_diagnostics(2)))
            self.assertEqual(episode["diagnostics"]["guard_reason"], ["", "action_clip"])

    @unittest.skipUnless(HAS_H5PY and HAS_CV2, "h5py and cv2 are required for JPEG HDF5 tests")
    def test_hdf5_encoded_jpeg_images_round_trip_as_rgb_images(self) -> None:
        import cv2
        import h5py

        from testbed.data.hdf5_io import read_episode, write_episode

        rgb = np.zeros((2, 6, 8, 3), dtype=np.uint8)
        rgb[0, ..., 0] = 220
        rgb[1, ..., 1] = 180
        encoded_frames = []
        for frame in rgb:
            ok, jpeg = cv2.imencode(
                ".jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            self.assertTrue(ok)
            encoded_frames.append(jpeg.reshape(-1))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episode_0.hdf5"
            write_episode(
                path,
                qpos=np.zeros((2, 4), dtype=np.float32),
                qvel=np.zeros((2, 4), dtype=np.float32),
                actions=np.zeros((2, 4), dtype=np.float32),
                encoded_images={"fpv": encoded_frames},
                rewards=np.zeros(2, dtype=np.float32),
                metadata={"is_real": True, "image_format": "jpeg", "camera_names": "fpv"},
                diagnostics=_real_diagnostics(2),
            )

            with h5py.File(path, "r") as f:
                self.assertIn("observations/encoded_images/fpv", f)
                self.assertEqual(f["observations/encoded_images/fpv"].attrs["encoding"], "jpeg")
                self.assertNotIn("observations/images/fpv", f)

            episode = read_episode(path)
            self.assertIn("fpv", episode["encoded_images"])
            self.assertEqual(episode["images"]["fpv"].shape, (2, 6, 8, 3))
            self.assertGreater(float(episode["images"]["fpv"][0, ..., 0].mean()), 180.0)

    def test_bucket_qpos_repair_is_noop_for_clean_series(self) -> None:
        from testbed.data.bucket_repair import repair_bucket_series

        qpos = np.linspace(-0.6, -0.2, 40, dtype=np.float32)
        qvel = np.gradient(qpos, 0.02).astype(np.float32)
        repaired, repair_mask, bad_mask, degraded = repair_bucket_series(
            bucket_qpos=qpos,
            bucket_qvel=qvel,
            timestamps_ns=np.arange(40, dtype=np.int64) * 20_000_000,
        )
        np.testing.assert_allclose(repaired, qpos, atol=1e-6)
        self.assertFalse(bool(np.any(repair_mask)))
        self.assertFalse(bool(np.any(bad_mask)))
        self.assertFalse(degraded)

    def test_bucket_qpos_repair_reconstructs_branch_jump_and_env_state(self) -> None:
        from testbed.data.bucket_repair import repair_episode
        from testbed.data.hdf5_io import write_episode

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "episode_47.hdf5"
            dst = Path(tmpdir) / "repaired" / "episode_47.hdf5"
            n = 80
            t = np.arange(n, dtype=np.int64) * 20_000_000
            bucket = np.linspace(-0.6, -1.0, n, dtype=np.float32)
            corrupted = bucket.copy()
            corrupted[35:45] += 3.0
            qpos = np.zeros((n, 4), dtype=np.float32)
            qpos[:, 3] = corrupted
            qvel = np.zeros((n, 4), dtype=np.float32)
            qvel[:, 3] = np.gradient(bucket, 0.02)
            env_state = np.concatenate([qpos, qvel], axis=1)
            diagnostics = _real_diagnostics(n)
            diagnostics["joint_timestamp_ns"] = t
            write_episode(
                src,
                qpos=qpos,
                qvel=qvel,
                actions=np.zeros((n, 4), dtype=np.float32),
                images={"fpv": np.zeros((n, 4, 4, 3), dtype=np.uint8)},
                env_state=env_state,
                metadata={"is_real": True, "success": 1},
                step_ns=t,
                diagnostics=diagnostics,
            )

            result = repair_episode(src, dst, imu_log_dir=None)
            self.assertTrue(result.repaired)
            import h5py

            with h5py.File(dst, "r") as f:
                repaired_bucket = f["observations/qpos"][:, 3]
                self.assertLess(float(np.max(np.abs(np.diff(repaired_bucket)))), 0.20)
                np.testing.assert_allclose(
                    f["observations/env_state"][:, 3],
                    repaired_bucket,
                    atol=1e-6,
                )
                self.assertIn("repairs/bucket_qpos_v1/repair_mask", f)

    @unittest.skipUnless(HAS_H5PY, "h5py is required for 20Hz builder tests")
    def test_20hz_builder_excludes_go_home_and_uses_unique_fpv_timestamps(self) -> None:
        from testbed.data.hdf5_io import write_episode
        from testbed.data.resample_20hz import build_20hz_episode

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "episode_1.hdf5"
            dst = Path(tmpdir) / "out" / "episode_1.hdf5"
            n = 100
            step_ns = np.arange(n, dtype=np.int64) * 20_000_000
            image_ts = (np.arange(n, dtype=np.int64) // 2) * 40_000_000 + 1_000_000_000
            image_ts[40:] += 400_000_000
            diagnostics = _real_diagnostics(n)
            diagnostics.update(
                {
                    "image_timestamp_ns_fpv": image_ts,
                    "image_timestamp_ns": image_ts,
                    "joint_timestamp_ns": image_ts,
                    "action_sample_timestamp_ns": image_ts,
                    "go_home_requested": np.r_[np.zeros(70, dtype=np.int8), np.ones(30, dtype=np.int8)],
                    "go_home_running": np.r_[np.zeros(71, dtype=np.int8), np.ones(29, dtype=np.int8)],
                }
            )
            actions = np.repeat(np.arange(n, dtype=np.float32).reshape(n, 1), 4, axis=1)
            write_episode(
                src,
                qpos=np.zeros((n, 4), dtype=np.float32),
                qvel=np.zeros((n, 4), dtype=np.float32),
                actions=actions,
                images={"fpv": np.zeros((n, 4, 4, 3), dtype=np.uint8)},
                metadata={"is_real": True, "success": 1},
                step_ns=step_ns,
                diagnostics=diagnostics,
            )
            row = build_20hz_episode(input_path=src, output_path=dst)
            self.assertLess(row["last_source_index"], 70)
            self.assertGreater(row["source_time_gap_event_count"], 0)
            import h5py

            with h5py.File(dst, "r") as f:
                source_idx = f["diagnostics/source_observation_index"][()]
                source_action_idx = f["diagnostics/source_action_index"][()]
                image_out = f["diagnostics/image_timestamp_ns_fpv"][()]
                source_gap_ms = f["diagnostics/source_time_gap_ms"][()]
                train_exclude_mask = f["diagnostics/train_exclude_mask"][()].astype(bool)
                self.assertTrue(np.all(np.diff(source_idx) > 0))
                self.assertEqual(len(np.unique(image_out)), len(image_out))
                np.testing.assert_array_less(source_action_idx, source_idx + 1)
                self.assertEqual(source_gap_ms.shape, source_idx.shape)
                self.assertGreater(float(np.max(source_gap_ms)), 250.0)
                self.assertTrue(np.any(train_exclude_mask))
                self.assertTrue(bool(f["metadata"].attrs["action_prealigned"]))

    @unittest.skipUnless(HAS_H5PY, "h5py is required for training QC tests")
    def test_training_qc_flags_bucket_jump_and_fpv_gap(self) -> None:
        from testbed.data.hdf5_io import write_episode
        from testbed.data.training_qc import episode_training_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episode_1.hdf5"
            n = 20
            qpos = np.zeros((n, 4), dtype=np.float32)
            qpos[10:, 3] = 3.0
            diagnostics = _real_diagnostics(n)
            image_ts = np.arange(n, dtype=np.int64) * 50_000_000 + 1_000_000_000
            image_ts[10:] += 300_000_000
            diagnostics.update(
                {
                    "image_timestamp_ns_fpv": image_ts,
                    "image_timestamp_ns": image_ts,
                    "joint_timestamp_ns": np.arange(n, dtype=np.int64) * 20_000_000,
                    "fpv_age_ms": np.full(n, 10.0),
                    "sync_max_skew_ns": np.zeros(n, dtype=np.int64),
                    "receiver_health_ok": np.ones(n, dtype=np.int8),
                    "imu_online": np.ones((n, 4), dtype=np.int32),
                    "imu_valid_attitude": np.ones((n, 4), dtype=np.int32),
                }
            )
            write_episode(
                path,
                qpos=qpos,
                qvel=np.zeros((n, 4), dtype=np.float32),
                actions=np.zeros((n, 4), dtype=np.float32),
                images={"fpv": np.zeros((n, 4, 4, 3), dtype=np.uint8) + 30},
                metadata={"is_real": True, "success": 1},
                diagnostics=diagnostics,
            )
            metrics = episode_training_metrics(
                path,
                reference_stats=None,
                make_plot=False,
                plot_dir=None,
            )
            self.assertEqual(metrics["training_status"], "FAIL")
            self.assertIn("qpos_jump", metrics["training_warnings"])
            self.assertIn("fpv_gap_fail", metrics["training_warnings"])

    @unittest.skipUnless(HAS_H5PY, "h5py is required for training QC tests")
    def test_training_qc_treats_masked_fpv_gap_as_info(self) -> None:
        from testbed.data.hdf5_io import write_episode
        from testbed.data.training_qc import episode_training_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episode_1.hdf5"
            n = 80
            diagnostics = _real_diagnostics(n)
            image_ts = np.arange(n, dtype=np.int64) * 40_000_000 + 1_000_000_000
            image_ts[30:] += 350_000_000
            source_gap_ms = np.zeros(n, dtype=np.float32)
            source_gap_ms[1:] = np.diff(image_ts).astype(np.float64) * 1e-6
            train_exclude_mask = np.zeros(n, dtype=np.uint8)
            train_exclude_mask[20:41] = 1
            diagnostics.update(
                {
                    "image_timestamp_ns_fpv": image_ts,
                    "image_timestamp_ns": image_ts,
                    "joint_timestamp_ns": np.arange(n, dtype=np.int64) * 40_000_000,
                    "fpv_age_ms": np.full(n, 10.0),
                    "sync_max_skew_ns": np.zeros(n, dtype=np.int64),
                    "receiver_health_ok": np.ones(n, dtype=np.int8),
                    "imu_online": np.ones((n, 4), dtype=np.int32),
                    "imu_valid_attitude": np.ones((n, 4), dtype=np.int32),
                    "source_time_gap_ms": source_gap_ms,
                    "train_exclude_mask": train_exclude_mask,
                }
            )
            write_episode(
                path,
                qpos=np.zeros((n, 4), dtype=np.float32),
                qvel=np.zeros((n, 4), dtype=np.float32),
                actions=np.zeros((n, 4), dtype=np.float32),
                images={"fpv": np.zeros((n, 4, 4, 3), dtype=np.uint8) + 30},
                metadata={"is_real": True, "success": 1},
                diagnostics=diagnostics,
            )
            metrics = episode_training_metrics(
                path,
                reference_stats=None,
                make_plot=False,
                plot_dir=None,
            )
            self.assertEqual(metrics["training_status"], "PASS")
            self.assertNotIn("fpv_gap_fail", metrics["training_warnings"])
            self.assertIn("usable_with_gap_mask", metrics["training_info"])
            self.assertEqual(metrics["fpv_gap_mask_status"], "usable_with_gap_mask")

    @unittest.skipUnless(HAS_TORCH, "torch is required for ACT data loader tests")
    def test_training_sampler_excludes_windows_crossing_gap_mask(self) -> None:
        from testbed.data.dataset import _valid_start_indices

        mask = np.zeros(12, dtype=bool)
        mask[5] = True
        starts = _valid_start_indices(
            total_steps=12,
            train_exclude_mask=mask,
            action_chunk_size=4,
        )
        self.assertNotIn(2, starts.tolist())
        self.assertNotIn(5, starts.tolist())
        self.assertIn(1, starts.tolist())
        self.assertIn(6, starts.tolist())

    def test_bucket_semantic_decision_drops_bad_recovery_and_reviews_shallow_min(self) -> None:
        from testbed.data.training_qc import _bucket_semantic_decision

        reference = {
            "end": {"p5": 0.20},
            "max": {"p5": 0.60},
            "late_max": {"p5": 0.48},
            "min": {"p99": -1.54},
            "max_jump": {"p99": 0.17},
        }
        drop_features = {
            "end": -0.25,
            "max": 0.24,
            "late_max": 0.05,
            "min": -2.1,
            "max_jump": 0.09,
        }
        review_features = {
            "end": 0.92,
            "max": 1.08,
            "late_max": 1.08,
            "min": -1.43,
            "max_jump": 0.10,
        }
        keep_features = {
            "end": 0.70,
            "max": 0.71,
            "late_max": 0.71,
            "min": -2.1,
            "max_jump": 0.16,
        }

        self.assertEqual(_bucket_semantic_decision(drop_features, reference)[0], "drop")
        self.assertEqual(_bucket_semantic_decision(review_features, reference)[0], "review")
        self.assertEqual(_bucket_semantic_decision(keep_features, reference)[0], "keep")

    @unittest.skipUnless(HAS_H5PY, "h5py is required for recorder metadata tests")
    def test_recorder_metadata_uses_recorded_image_shape_and_timestamps(self) -> None:
        import h5py

        from testbed.data.recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = EpisodeRecorder(
                output_dir=Path(tmpdir),
                episode_idx=0,
                metadata={
                    "camera_width": 160,
                    "camera_height": 120,
                    "camera_fps": 50.0,
                    "camera_names": "fpv",
                },
                camera_names=["fpv"],
            )
            for step, timestamp_ns in enumerate(
                [1_000_000_000, 1_100_000_000, 1_200_000_000]
            ):
                recorder.record(
                    {
                        "qpos": np.zeros(4, dtype=np.float32),
                        "qvel": np.zeros(4, dtype=np.float32),
                        "images": {"fpv": np.zeros((4, 6, 3), dtype=np.uint8)},
                    },
                    np.zeros(4, dtype=np.float32),
                    step_id=step,
                    diagnostics={"image_timestamp_ns": timestamp_ns},
                )
            path = recorder.save(success=True)

            with h5py.File(path, "r") as f:
                metadata = f["metadata"].attrs
                self.assertEqual(metadata["camera_width"], 6)
                self.assertEqual(metadata["camera_height"], 4)
                self.assertAlmostEqual(float(metadata["camera_fps"]), 10.0)
            self.assertFalse(list(Path(tmpdir).glob("*.tmp.*")))

    @unittest.skipUnless(HAS_H5PY, "h5py is required for recorder metadata tests")
    def test_recorder_saves_encoded_images_and_uses_encoded_shape(self) -> None:
        import h5py

        from testbed.data.recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = EpisodeRecorder(
                output_dir=Path(tmpdir),
                episode_idx=0,
                metadata={
                    "camera_width": 160,
                    "camera_height": 120,
                    "camera_names": "fpv",
                },
                camera_names=["fpv"],
            )
            recorder.record(
                {
                    "qpos": np.zeros(4, dtype=np.float32),
                    "qvel": np.zeros(4, dtype=np.float32),
                    "encoded_images": {
                        "fpv": {
                            "encoding": "jpeg",
                            "shape": (480, 640, 3),
                            "data": np.arange(16, dtype=np.uint8),
                        }
                    },
                },
                np.zeros(4, dtype=np.float32),
                diagnostics={"image_timestamp_ns": 1_000_000_000},
            )
            path = recorder.save(success=True)

            with h5py.File(path, "r") as f:
                metadata = f["metadata"].attrs
                self.assertEqual(metadata["camera_width"], 640)
                self.assertEqual(metadata["camera_height"], 480)
                self.assertEqual(metadata["image_format"], "jpeg")
                self.assertIn("observations/encoded_images/fpv", f)

    @unittest.skipUnless(HAS_H5PY, "h5py is required for recorder save tests")
    def test_recorder_save_does_not_publish_partial_episode_on_write_error(self) -> None:
        from testbed.data.recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = EpisodeRecorder(output_dir=Path(tmpdir), episode_idx=0)
            recorder.record(
                {
                    "qpos": np.zeros(4, dtype=np.float32),
                    "qvel": np.zeros(4, dtype=np.float32),
                },
                np.zeros(4, dtype=np.float32),
            )

            def _fail_after_touch(path, **_kwargs):
                Path(path).write_bytes(b"partial")
                raise RuntimeError("write failed")

            with patch("testbed.data.recorder.write_episode", side_effect=_fail_after_touch):
                with self.assertRaises(RuntimeError):
                    recorder.save(success=False)

            self.assertFalse((Path(tmpdir) / "episode_0.hdf5").exists())
            self.assertTrue(list(Path(tmpdir).glob(".episode_0.hdf5.tmp.*")))

    @unittest.skipUnless(HAS_H5PY, "h5py is required for record failure tests")
    def test_record_session_saves_failed_episode_under_failed_dir(self) -> None:
        import h5py

        from testbed.cli.record_real import RecordSession
        from testbed.data.recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session = RecordSession(
                recorder_cls=EpisodeRecorder,
                dataset_dir=root,
                failed_dir=root / "failed",
                episode_idx=7,
                metadata={"task_name": "unit"},
                camera_names=[],
            )
            session.record_step(
                obs={
                    "qpos": np.zeros(4, dtype=np.float32),
                    "qvel": np.zeros(4, dtype=np.float32),
                },
                action=np.zeros(4, dtype=np.float32),
                diagnostics={
                    "receiver_health_ok": 0,
                    "receiver_health_error_code": "imu_missing:1",
                    "imu_online": np.array([1, 0, 1, 1], dtype=np.int32),
                    "imu_valid_attitude": np.ones(4, dtype=np.int32),
                    "fpv_age_ms": 12.0,
                    "bridge_snapshot_age_ms": 10.0,
                    "remote_action_connected": 1,
                    "controller_ack": 1,
                },
            )

            path = session.save_failed(
                error_code="imu_missing:1",
                error_time_ns=123456,
            )

            assert path is not None
            self.assertEqual(path.parent.name, "failed")
            self.assertTrue(path.name.startswith("episode_7_failed_"))
            self.assertFalse((root / "episode_7.hdf5").exists())
            with h5py.File(path, "r") as f:
                metadata = f["metadata"].attrs
                self.assertEqual(int(metadata["success"]), 0)
                self.assertEqual(metadata["record_stop_reason"], "sensor_error")
                self.assertEqual(metadata["record_error_code"], "imu_missing:1")
                self.assertEqual(int(metadata["record_error_time_ns"]), 123456)
                self.assertIn("receiver_health_ok", f["diagnostics"])

    def test_real_action_maps_to_lower_speed_scalar8(self) -> None:
        action = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        speed = action4_to_speed_scalar8(action)
        self.assertEqual(speed.shape, (8,))
        np.testing.assert_allclose(speed[:4], action)
        np.testing.assert_allclose(speed[4:], np.zeros(4, dtype=np.float32))

        adapter = ExcavatorApiPacketAdapter()
        packet = adapter.servo_bytes(action)
        self.assertEqual(len(packet), SERVO_PACKET_STRUCT.size)
        unpacked = struct.unpack("<II9d", packet)
        self.assertEqual(unpacked[0], SERVO_MAGIC)
        self.assertEqual(unpacked[1], 3)
        np.testing.assert_allclose(np.asarray(unpacked[2:6], dtype=np.float32), action)

    def test_ros_can_stub_is_import_safe_without_ros(self) -> None:
        self.assertTrue(issubclass(RosCanLowLevelController, object))
        self.assertTrue(issubclass(RosCanStateReader, object))

    def test_sync_builder_aligns_joint_and_image_timestamps(self) -> None:
        builder = SynchronizedObservationBuilder(max_slop_ns=50)
        joint = TimestampedSample(
            timestamp_ns=1_000,
            payload={
                "qpos": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                "qvel": np.array([0.0, 0.1, 0.0, -0.1], dtype=np.float32),
            },
            source="joint",
        )
        image = TimestampedSample(
            timestamp_ns=1_030,
            payload=np.zeros((4, 5, 3), dtype=np.uint8),
            source="fpv",
        )

        result = builder.build(
            joint_sample=joint,
            image_samples={"fpv": image},
            step_id=7,
            action_timestamp_ns=990,
        )

        obs = result.observation
        self.assertEqual(obs["step_id"], 7)
        self.assertEqual(obs["joint_timestamp_ns"], 1_000)
        self.assertEqual(obs["image_timestamp_ns"]["fpv"], 1_030)
        self.assertEqual(obs["sync_timestamp_ns"], 1_000)
        self.assertEqual(obs["sync_max_skew_ns"], 30)
        self.assertEqual(result.max_skew_ns, 30)
        self.assertEqual(obs["images"]["fpv"].shape, (4, 5, 3))

    def test_sync_builder_preserves_imu_health_payload(self) -> None:
        builder = SynchronizedObservationBuilder(max_slop_ns=50)
        joint = TimestampedSample(
            timestamp_ns=1_000,
            payload={
                "qpos": np.zeros(4, dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
                "imu_health": {
                    "online": [1, 0, 1, 1],
                    "valid_attitude": [1, 1, 1, 1],
                    "valid_gyro": [1, 1, 1, 1],
                    "valid_accel": [1, 1, 1, 1],
                    "packet_loss_count": [0, 2, 0, 0],
                    "host_rx_age_ms": [10.0, 11.0, 10.5, 9.5],
                },
                "snapshot_age_ms": 15.0,
                "qpos_raw_imu": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                "imu_debug": {
                    "devices": [
                        {"online": 1, "valid_attitude": 1, "rpy_raw_deg": [0.0, 1.0, 2.0]},
                        {"online": 1, "valid_attitude": 1, "rpy_raw_deg": [0.0, 3.0, 4.0]},
                        {"online": 1, "valid_attitude": 1, "rpy_raw_deg": [0.0, 5.0, 6.0]},
                        {"online": 1, "valid_attitude": 1, "rpy_raw_deg": [0.0, 7.0, 8.0]},
                    ],
                },
            },
            source="joint",
        )

        result = builder.build(
            joint_sample=joint,
            image_samples={
                "fpv": TimestampedSample(
                    timestamp_ns=1_010,
                    payload=np.zeros((2, 2, 3), dtype=np.uint8),
                    source="fpv",
                )
            },
            step_id=1,
            action_timestamp_ns=900,
        )

        sensor_health = result.observation["sensor_health"]
        self.assertEqual(sensor_health["imu"]["online"], [1, 0, 1, 1])
        self.assertEqual(sensor_health["imu"]["packet_loss_count"], [0, 2, 0, 0])
        self.assertAlmostEqual(sensor_health["bridge_snapshot_age_ms"], 15.0)
        self.assertIn("imu_debug", result.observation)
        np.testing.assert_allclose(result.observation["qpos_raw_imu"], [0.1, 0.2, 0.3, 0.4])

    def test_timestamped_buffer_selects_nearest_sample(self) -> None:
        buffer = TimestampedBuffer(maxlen=4)
        buffer.add({"v": 1}, timestamp_ns=100)
        buffer.add({"v": 2}, timestamp_ns=140)
        buffer.add({"v": 3}, timestamp_ns=220)

        sample = buffer.nearest(150, max_slop_ns=20)

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.payload["v"], 2)
        self.assertIsNone(buffer.nearest(180, max_slop_ns=20))

    def test_oem_remote_stub_is_import_safe_and_timestamped(self) -> None:
        source = OemRemoteActionSource(allow_stub=True)
        action, info = source.next_action({})

        np.testing.assert_allclose(action, np.zeros(4, dtype=np.float32))
        self.assertEqual(info.source_type, "teleop")
        self.assertEqual(info.source_id, "oem_remote_stub")
        self.assertFalse(info.extras["remote_available"])
        self.assertGreater(int(info.extras["action_timestamp_ns"]), 0)

    def test_oem_remote_requires_reader_unless_stub_is_enabled(self) -> None:
        with self.assertRaises(OemRemoteUnavailableError):
            OemRemoteActionSource()

    def test_action_guard_covers_real_safety_rules(self) -> None:
        guard = ActionGuard(action_clip=0.20, max_delta=0.02, sensor_timeout_s=0.20)

        safe, triggered = guard.check(np.array([0.5, -0.5, 0.1, -0.1], dtype=np.float32))
        self.assertTrue(triggered)
        np.testing.assert_allclose(
            safe,
            np.array([0.02, -0.02, 0.02, -0.02], dtype=np.float32),
        )
        self.assertIn("action_clip", guard.last_info.reasons)
        self.assertIn("rate_limit", guard.last_info.reasons)

        safe, triggered = guard.check(np.ones(4, dtype=np.float32) * 0.1, deadman_pressed=False)
        self.assertTrue(triggered)
        np.testing.assert_allclose(safe, np.zeros(4, dtype=np.float32))
        self.assertIn("deadman_released", guard.last_info.reasons)

        guard.reset()
        safe, triggered = guard.check(np.ones(4, dtype=np.float32) * 0.1, estop_active=True)
        self.assertTrue(triggered)
        np.testing.assert_allclose(safe, np.zeros(4, dtype=np.float32))
        self.assertIn("estop_active", guard.last_info.reasons)

        guard.reset()
        safe, triggered = guard.check(
            np.ones(4, dtype=np.float32) * 0.1,
            sensor_age_s=0.21,
        )
        self.assertTrue(triggered)
        np.testing.assert_allclose(safe, np.zeros(4, dtype=np.float32))
        self.assertIn("sensor_timeout", guard.last_info.reasons)

    @unittest.skipUnless(HAS_H5PY, "h5py is required for dataset QC tests")
    def test_real_qc_profile_allows_missing_env_state_and_reports_diagnostics(self) -> None:
        from testbed.data.qc import run_dataset_qc

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True)
            _write_real_episode(dataset_dir / "episode_0.hdf5", length=3)

            result = run_dataset_qc(
                dataset_dir=dataset_dir,
                profile="real",
                short_episode_threshold=0,
            )

            with open(result["summary_path"]) as f:
                summary = json.load(f)
            self.assertEqual(summary["profile"], "real")
            self.assertEqual(summary["warnings"]["missing_env_state_ids"], [])
            self.assertEqual(summary["warnings"]["real_diagnostic_missing_ids"], [])

    @unittest.skipUnless(HAS_H5PY, "h5py is required for episode QC tests")
    def test_episode_qc_detects_black_fpv_frames(self) -> None:
        from testbed.data.episode_qc import run_episode_qc
        from testbed.data.hdf5_io import write_episode

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episode_0.hdf5"
            write_episode(
                path,
                qpos=np.zeros((3, 4), dtype=np.float32),
                qvel=np.zeros((3, 4), dtype=np.float32),
                actions=np.zeros((3, 4), dtype=np.float32),
                images={"fpv": np.zeros((3, 4, 4, 3), dtype=np.uint8)},
                rewards=np.zeros(3, dtype=np.float32),
                metadata={
                    "is_real": True,
                    "platform": "real_excavator",
                    "success": 1,
                    "qpos_units": "rad",
                    "qvel_units": "rad/s",
                    "hydraulic_cylinder_available": False,
                },
                step_ids=np.arange(3, dtype=np.int64),
                step_ns=np.arange(1, 4, dtype=np.int64),
                diagnostics=_real_diagnostics(3),
            )

            result = run_episode_qc(path, output_dir=Path(tmpdir) / "qc")
            self.assertFalse(result["ok"])
            self.assertIn("fpv_black_frames", result["errors"])
            self.assertTrue((Path(tmpdir) / "qc" / "episode_0.json").exists())

    @unittest.skipUnless(HAS_H5PY, "h5py is required for phase label tests")
    def test_phase_labeler_uses_shortest_swing_home_error(self) -> None:
        from testbed.data.phase_labeler import PhaseLabelConfig, _home_distance, _near_home

        home_swing = np.deg2rad(216.46)
        cfg = PhaseLabelConfig.from_mapping(
            {
                "home_pose_rad": [home_swing, 0.0, 0.0, 0.0],
                "near_home_tolerance_rad": [0.05, 0.05, 0.05, 0.05],
            }
        )
        qpos = np.array([home_swing - 2.0 * np.pi + 0.01, 0.0, 0.0, 0.0])

        self.assertTrue(_near_home(qpos, cfg))
        self.assertLess(_home_distance(qpos, cfg), 0.02)

    @unittest.skipUnless(HAS_H5PY, "h5py is required for phase label tests")
    def test_phase_labeler_generates_monotonic_real_workflow_labels(self) -> None:
        from testbed.data.hdf5_io import write_episode
        from testbed.data.phase_labeler import (
            PhaseLabelConfig,
            label_episode_phases,
            write_phase_labels,
        )

        length = 40
        qpos = np.zeros((length, 4), dtype=np.float32)
        qvel = np.zeros((length, 4), dtype=np.float32)
        qpos[:10, 0] = 0.0
        qpos[10:20, 0] = np.linspace(0.2, 0.9, 10)
        qvel[10:20, 0] = 0.1
        qpos[20:25, 0] = 1.0
        qvel[20:25, 3] = 0.1
        qpos[25:, 0] = np.linspace(0.8, 0.0, length - 25)
        qvel[25:, 0] = -0.1
        go_home_running = np.zeros(length, dtype=np.int32)
        go_home_running[35:] = 1
        diagnostics = _real_diagnostics(length)
        diagnostics["go_home_running"] = go_home_running

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episode_0.hdf5"
            write_episode(
                path,
                qpos=qpos,
                qvel=qvel,
                actions=np.zeros((length, 4), dtype=np.float32),
                images={"fpv": np.ones((length, 2, 2, 3), dtype=np.uint8) * 40},
                metadata={"is_real": True, "success": 1},
                diagnostics=diagnostics,
            )
            result = label_episode_phases(
                path,
                config=PhaseLabelConfig.from_mapping(
                    {
                        "home_pose_rad": [0, 0, 0, 0],
                        "near_home_tolerance_rad": [0.1, 0.1, 0.1, 0.1],
                        "dig_swing_range": [-0.1, 0.1],
                        "dump_swing_range": [0.9, 1.1],
                        "swing_velocity_threshold_rad_s": 0.02,
                        "bucket_velocity_threshold_rad_s": 0.02,
                        "dwell_steps": 1,
                    }
                ),
            )
            self.assertIn("SWING_TO_DUMP", result["phases"])
            self.assertIn("DUMP", result["phases"])
            self.assertIn("RETURN_NEAR_HOME", result["phases"])
            self.assertIn("GO_HOME", result["phases"])
            self.assertEqual(result["phases"][-1], "END")
            json_path, csv_path = write_phase_labels(result, Path(tmpdir) / "labels")
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

    @unittest.skipUnless(HAS_H5PY and HAS_TORCH, "h5py and torch are required for ACT data loader tests")
    def test_act_loader_reads_real_style_qpos_plus_qvel_episode(self) -> None:
        from testbed.data.dataset import load_data

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True)
            _write_real_episode(dataset_dir / "episode_0.hdf5", length=4)
            _write_real_episode(dataset_dir / "episode_1.hdf5", length=5)

            train_loader, _val_loader, norm_stats, is_real, split_info = load_data(
                dataset_dir=dataset_dir,
                num_episodes=2,
                camera_names=["fpv"],
                episode_len=6,
                batch_size_train=1,
                batch_size_val=1,
                num_workers=0,
                prefetch_factor=1,
                persistent_workers=False,
                pin_memory=False,
                split_seed=0,
                train_split_ratio=0.5,
                reuse_split=False,
                low_dim_keys=["qpos", "qvel"],
            )

            self.assertTrue(is_real)
            self.assertEqual(norm_stats["proprio_mean"].shape, (8,))
            self.assertEqual(split_info["low_dim_keys"], ["qpos", "qvel"])
            self.assertEqual(split_info["low_dim_dim"], 8)

            image_data, proprio_data, action_data, is_pad = next(iter(train_loader))
            self.assertEqual(image_data.shape[1:], (1, 3, 8, 8))
            self.assertEqual(proprio_data.shape[1:], (8,))
            self.assertEqual(action_data.shape[1:], (6, 4))
            self.assertEqual(is_pad.shape[1:], (6,))

    @unittest.skipUnless(
        HAS_H5PY and HAS_TORCH and HAS_CV2,
        "h5py, torch, and cv2 are required for JPEG ACT data loader tests",
    )
    def test_act_loader_reads_encoded_jpeg_episode(self) -> None:
        from testbed.data.dataset import load_data

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True)
            _write_real_episode(dataset_dir / "episode_0.hdf5", length=4, encoded_jpeg=True)

            train_loader, _val_loader, _norm_stats, is_real, _split_info = load_data(
                dataset_dir=dataset_dir,
                num_episodes=1,
                camera_names=["fpv"],
                episode_len=6,
                batch_size_train=1,
                batch_size_val=1,
                num_workers=0,
                prefetch_factor=1,
                persistent_workers=False,
                pin_memory=False,
                split_seed=0,
                train_split_ratio=0.5,
                reuse_split=False,
                low_dim_keys=["qpos", "qvel"],
            )

            image_data, _proprio, _action, _is_pad = next(iter(train_loader))
            self.assertTrue(is_real)
            self.assertEqual(image_data.shape[1:], (1, 3, 8, 8))


def _write_real_episode(path: Path, *, length: int, encoded_jpeg: bool = False) -> None:
    from testbed.data.hdf5_io import write_episode

    images = None
    encoded_images = None
    image_format = "raw_rgb"
    if encoded_jpeg:
        import cv2

        frames = []
        for step in range(length):
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[..., step % 3] = 180
            ok, jpeg = cv2.imencode(
                ".jpg",
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            if not ok:
                raise RuntimeError("failed to encode test JPEG")
            frames.append(jpeg.reshape(-1))
        encoded_images = {"fpv": frames}
        image_format = "jpeg"
    else:
        images = {"fpv": np.zeros((length, 8, 8, 3), dtype=np.uint8)}

    write_episode(
        path,
        qpos=np.linspace(0.0, 0.1, length * 4, dtype=np.float32).reshape(length, 4),
        qvel=np.linspace(0.0, 0.2, length * 4, dtype=np.float32).reshape(length, 4),
        actions=np.zeros((length, 4), dtype=np.float32),
        images=images,
        encoded_images=encoded_images,
        rewards=np.zeros(length, dtype=np.float32),
        metadata={
            "is_real": True,
            "platform": "real_excavator",
            "image_format": image_format,
            "success": 0,
            "qpos_units": "rad",
            "qvel_units": "rad/s",
            "hydraulic_cylinder_available": False,
            "action_semantics": "normalized_teleop_cmd_v1",
        },
        step_ids=np.arange(length, dtype=np.int64),
        step_ns=np.arange(1, length + 1, dtype=np.int64),
        diagnostics=_real_diagnostics(length),
    )


def _real_diagnostics(length: int) -> dict[str, np.ndarray | list[str]]:
    guard_reason = [""] * length
    if length > 1:
        guard_reason[1] = "action_clip"
    return {
        "raw_action": np.zeros((length, 4), dtype=np.float32),
        "guard_triggered": np.zeros(length, dtype=np.int8),
        "guard_reason": guard_reason,
        "controller_ack": np.ones(length, dtype=np.int8),
        "controller_fault_code": [""] * length,
        "controller_timestamp_ns": np.arange(length, dtype=np.int64),
        "commanded_action": np.zeros((length, 4), dtype=np.float32),
    }


def _online_qc_reference() -> dict:
    return {
        "reference_id": "unit-reference",
        "qpos": {
            "p1": [-1.5, -1.5, -1.5, -1.5],
            "p5": [-1.0, -1.0, -1.0, -1.0],
            "p95": [1.0, 1.0, 1.0, 1.0],
            "p99": [1.5, 1.5, 1.5, 1.5],
        },
        "fpv": {
            "brightness": {"median": 80.0, "mad": 5.0, "p1": 50.0, "p99": 110.0},
            "contrast": {"median": 12.0, "mad": 3.0, "p1": 4.0, "p99": 30.0},
            "jpeg_size": {"median": 256.0, "mad": 64.0, "p1": 64.0, "p99": 1024.0},
            "fingerprint": [80.0] * 64,
        },
    }


def _online_qc_bucket_semantic_reference(
    *,
    end_p5: float,
    max_p5: float,
    late_max_p5: float,
    min_p99: float,
) -> dict:
    reference: dict[str, object] = {"count": 5}
    defaults = {
        "p1": 0.0,
        "p5": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
    }
    for key in (
        "start",
        "end",
        "min",
        "max",
        "range",
        "argmin",
        "argmax",
        "early_max",
        "late_max",
        "max_jump",
    ):
        reference[key] = dict(defaults)
    reference["end"]["p5"] = float(end_p5)
    reference["max"]["p5"] = float(max_p5)
    reference["late_max"]["p5"] = float(late_max_p5)
    reference["min"]["p99"] = float(min_p99)
    reference["max_jump"]["p99"] = 0.20
    return reference


def _run_online_qc_bucket_series(evaluator, bucket: np.ndarray) -> None:
    values = np.asarray(bucket, dtype=np.float64).reshape(-1)
    for step, bucket_qpos in enumerate(values):
        qpos = [0.0, 0.0, 0.0, float(bucket_qpos)]
        evaluator.evaluate(
            obs=_online_qc_obs(
                qpos=qpos,
                qpos_raw_imu=qpos,
                image_timestamp_ns=2_000_000_000 + step * 20_000_000,
            ),
            now_ns=2_000_000_000 + step * 20_000_000,
        )


def _online_qc_obs(
    *,
    qpos: list[float],
    qpos_raw_imu: list[float] | None = None,
    image: np.ndarray | None = None,
    encoded_image: np.ndarray | bytes | None = None,
    image_timestamp_ns: int = 2_000_000_000,
) -> dict:
    obs: dict[str, object] = {
        "qpos": np.asarray(qpos, dtype=np.float32),
        "qvel": np.zeros(4, dtype=np.float32),
        "image_timestamp_ns": {"fpv": int(image_timestamp_ns)},
    }
    if qpos_raw_imu is not None:
        obs["qpos_raw_imu"] = np.asarray(qpos_raw_imu, dtype=np.float32)
    if image is None and encoded_image is None:
        image = _online_qc_pattern_image()
    if image is not None:
        obs["images"] = {"fpv": np.asarray(image, dtype=np.uint8)}
    if encoded_image is not None:
        obs["encoded_images"] = {
            "fpv": {
                "encoding": "jpeg",
                "data": encoded_image,
                "shape": [4, 4, 3],
            }
        }
    return obs


def _online_qc_pattern_image() -> np.ndarray:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[::2, :, :] = 92
    image[1::2, :, :] = 68
    return image


def _online_qc_bright_drift_image(step: int) -> np.ndarray:
    image = np.full((8, 8, 3), 130 + (step % 3), dtype=np.uint8)
    image[step % 8, (step * 3) % 8, :] = 150
    return image


def _write_online_qc_reference_episode(path: Path, *, offset: float) -> None:
    from testbed.data.hdf5_io import write_episode

    length = 4
    qpos = np.tile(
        np.asarray([offset, offset + 0.1, offset + 0.2, offset + 0.3], dtype=np.float32),
        (length, 1),
    )
    images = np.stack(
        [
            np.clip(_online_qc_pattern_image().astype(np.int16) + step, 0, 255).astype(
                np.uint8
            )
            for step in range(length)
        ]
    )
    write_episode(
        path,
        qpos=qpos,
        qvel=np.zeros((length, 4), dtype=np.float32),
        actions=np.zeros((length, 4), dtype=np.float32),
        images={"fpv": images},
        metadata={"is_real": True, "success": 1},
        diagnostics=_real_diagnostics(length),
    )


if __name__ == "__main__":
    unittest.main()
