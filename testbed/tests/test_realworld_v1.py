from __future__ import annotations

import importlib.util
import json
import socket
import struct
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from testbed.backends.real import (
    BridgeLowLevelController,
    BridgeStateReader,
    ControlResult,
    ExcavatorApiPacketAdapter,
    InProcessMockBridgeClient,
    JsonTcpBridgeClient,
    JsonTcpBridgeMockServer,
    MockLowLevelController,
    MockStateReader,
    NoopLowLevelController,
    RealExcavatorBackend,
    RealStateSamples,
    SynchronizedObservationBuilder,
    TimestampedBuffer,
    TimestampedSample,
    action4_to_speed_scalar8,
)
from testbed.actions.oem_remote import OemRemoteActionSource, OemRemoteUnavailableError
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
)
from testbed.backends.real.excavator_api import SERVO_MAGIC, SERVO_PACKET_STRUCT
from testbed.backends.real.ros_can import RosCanLowLevelController, RosCanStateReader
from testbed.runtime.guard import ActionGuard


HAS_H5PY = importlib.util.find_spec("h5py") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _can_bind_loopback_socket() -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return True
    except PermissionError:
        return False
    finally:
        sock.close()


class RealworldV1Tests(unittest.TestCase):
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
        np.testing.assert_array_equal(
            decoded_samples.images["fpv"].payload,
            np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        )

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
        side = apply_data_side_config(cfg)
        self.assertEqual(side, "slave")
        self.assertEqual(cfg["task"]["dataset_dir"], "/data/real_teleop_v1")
        self.assertEqual(cfg["real"]["data_side"], "slave")

    def test_apply_data_side_invalid_raises(self) -> None:
        from testbed.cli.data_side import apply_data_side_config

        with self.assertRaises(ValueError):
            apply_data_side_config({}, data_side="edge")

    def test_record_real_builds_bridge_client_from_tcp_config(self) -> None:
        from testbed.cli.record_real import _build_bridge_client

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


def _write_real_episode(path: Path, *, length: int) -> None:
    from testbed.data.hdf5_io import write_episode

    write_episode(
        path,
        qpos=np.linspace(0.0, 0.1, length * 4, dtype=np.float32).reshape(length, 4),
        qvel=np.linspace(0.0, 0.2, length * 4, dtype=np.float32).reshape(length, 4),
        actions=np.zeros((length, 4), dtype=np.float32),
        images={"fpv": np.zeros((length, 8, 8, 3), dtype=np.uint8)},
        rewards=np.zeros(length, dtype=np.float32),
        metadata={
            "is_real": True,
            "platform": "real_excavator",
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


if __name__ == "__main__":
    unittest.main()
