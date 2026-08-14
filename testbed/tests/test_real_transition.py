from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from testbed.cli.record_real import RecordSession, _load_yaml_config
from testbed.data.recorder import EpisodeRecorder
from testbed.tasks.home_side_contract import (
    build_home_side_contract,
    validate_home_side_contract,
)
from testbed.tasks.real_transition import (
    REQUIRED_GOAL_ACK_SOURCES,
    TransitionContractError,
    TransitionRunPackage,
    build_session_manifests,
    find_run_spec,
    prepare_session_directory,
    sha256_file,
    summarize_sequence_manifest,
    verify_run_package,
)
from testbed.tasks.real_transition_runtime import (
    TransitionTaskRuntime,
    TransitionTaskServer,
    send_transition_command,
)


class RealTransitionPlanTest(unittest.TestCase):
    def test_manual_profile_inherits_field_contract_and_disables_policy_helpers(
        self,
    ) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "testbed"
            / "configs"
            / "teleop_real_transition_v2_0_1.yaml"
        )
        config = _load_yaml_config(config_path)
        self.assertEqual(config["teleop"]["input"], "remote")
        self.assertIsNone(config["teleop"]["joystick"]["policy_start_button"])
        self.assertIsNone(config["teleop"]["joystick"]["record_start_button"])
        self.assertFalse(config["teleop"]["recording"]["go_home"]["enabled"])
        self.assertEqual(
            config["task"]["camera_names"], ["video4", "video5", "video6", "video7"]
        )
        self.assertEqual(config["task"]["max_steps"], 12000)
        self.assertEqual(config["receiver"]["health"]["mode"], "strict")

    def test_plan_is_deterministic_balanced_and_split_by_block(self) -> None:
        first_sequence, first_split = build_session_manifests(
            session_id="field_20260813",
            seed=20260813,
            created_at_utc="2026-08-13T00:00:00Z",
        )
        second_sequence, second_split = build_session_manifests(
            session_id="field_20260813",
            seed=20260813,
            created_at_utc="2026-08-13T00:00:00Z",
        )
        self.assertEqual(first_sequence, second_sequence)
        self.assertEqual(first_split, second_split)
        self.assertEqual(
            summarize_sequence_manifest(first_sequence),
            {
                "blocks": {"locked_test": 1, "train": 4, "validation": 1},
                "runs": {"locked_test": 4, "train": 16, "validation": 4},
                "cycles": {"locked_test": 16, "train": 64, "validation": 16},
                "transitions": {
                    "A->A": 24,
                    "A->B": 24,
                    "B->A": 24,
                    "B->B": 24,
                },
            },
        )
        core_blocks = [
            block
            for block in first_sequence["blocks"]
            if block["priority_tier"] == "minimum_64_cycle"
        ]
        self.assertEqual(len(core_blocks), 4)
        self.assertEqual(
            sorted(block["split"] for block in core_blocks),
            ["locked_test", "train", "train", "validation"],
        )

    def test_prepare_is_idempotent_but_rejects_changed_frozen_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = prepare_session_directory(
                output_root=root,
                session_id="s01",
                seed=7,
                created_at_utc="2026-08-13T00:00:00Z",
            )
            second = prepare_session_directory(
                output_root=root,
                session_id="s01",
                seed=7,
            )
            self.assertEqual(first, second)
            with self.assertRaisesRegex(
                TransitionContractError, "refusing to overwrite immutable artifact"
            ):
                prepare_session_directory(
                    output_root=root,
                    session_id="s01",
                    seed=8,
                    created_at_utc="2026-08-13T00:00:00Z",
                )


class RealTransitionRunPackageTest(unittest.TestCase):
    def test_record_session_writes_success_and_failure_to_run_raw_path(self) -> None:
        for success in (True, False):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                raw_path = root / "block_b01" / "run_b01_r01" / "raw.hdf5"
                session = RecordSession(
                    recorder_cls=EpisodeRecorder,
                    dataset_dir=root / "ordinary",
                    failed_dir=root / "ordinary" / "failed",
                    episode_idx=0,
                    metadata={"task_name": "test"},
                    camera_names=[],
                    save_path=raw_path,
                )
                session.record_step(
                    obs={
                        "qpos": np.zeros(4, dtype=np.float32),
                        "qvel": np.zeros(4, dtype=np.float32),
                        "images": {},
                    },
                    action=np.zeros(4, dtype=np.float32),
                    step_id=1,
                    step_ns=1_000_000_000,
                )
                if success:
                    saved_path = session.save_success()
                else:
                    saved_path = session.save_failed(
                        error_code="test",
                        error_time_ns=1_000_000_000,
                        stop_reason="test",
                    )
                self.assertEqual(saved_path, raw_path)
                self.assertTrue(raw_path.is_file())

    def test_four_cycle_round_trip_and_checksum_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence, _ = build_session_manifests(
                session_id="s01",
                seed=11,
                created_at_utc="2026-08-13T00:00:00Z",
            )
            spec = next(
                find_run_spec(sequence, run["run_id"])
                for block in sequence["blocks"]
                for run in block["runs"]
                if run["template_id"] == "P0"
            )
            run_dir = root / f"block_{spec.block_id}" / f"run_{spec.run_id}"
            package = TransitionRunPackage(run_dir=run_dir, run_spec=spec)
            self._write_raw(run_dir / "raw.hdf5", n_rows=32)
            package.start_run(step_id=0, step_ns=self._step_ns(0))
            package.mark_initial_ready(step_id=1, step_ns=self._step_ns(1))
            for cycle_index in range(4):
                base = 2 + cycle_index * 7
                package.commit_next_goal(
                    step_id=base,
                    step_ns=self._step_ns(base),
                    commit_ack_sources=REQUIRED_GOAL_ACK_SOURCES,
                    expected_return_swing_sign=(-1 if cycle_index % 2 else 1),
                )
                package.mark_dump_end(
                    step_id=base + 3,
                    step_ns=self._step_ns(base + 3),
                )
                package.mark_target_ready(
                    step_id=base + 6,
                    step_ns=self._step_ns(base + 6),
                )
            package.complete_run(step_id=30, step_ns=self._step_ns(30))

            sequence_path = root / "sequence_manifest.json"
            sequence_path.write_text(
                json.dumps(sequence, sort_keys=True) + "\n", encoding="utf-8"
            )
            home_contract = root / "home_side_contract.json"
            home_contract.write_text(
                '{"schema":"home_side_contract_v1"}\n', encoding="utf-8"
            )
            resolved = root / "resolved_record_config.yaml"
            resolved.write_text("task: real_transition\n", encoding="utf-8")
            manifest = package.seal(
                git_commit="a" * 40,
                resolved_config_sha256=sha256_file(resolved),
                owner_artifacts={
                    "sequence_manifest": sequence_path,
                    "home_side_contract": home_contract,
                    "resolved_record_config": resolved,
                },
                field_context={"workface_reset_id": "wf01"},
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed_cycles"], 4)

            report = verify_run_package(run_dir)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["event_summary"]["goal_commits"], 4)
            self.assertEqual(report["alignment"]["n_events"], 15)

            with (run_dir / "task_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(TransitionContractError, "checksum mismatch"):
                verify_run_package(run_dir)

    def test_seal_rejects_event_time_not_present_in_hdf5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence, _ = build_session_manifests(
                session_id="s02",
                seed=3,
                created_at_utc="2026-08-13T00:00:00Z",
            )
            first_run_id = sequence["blocks"][0]["runs"][0]["run_id"]
            spec = find_run_spec(sequence, first_run_id)
            package = TransitionRunPackage(run_dir=root / "run", run_spec=spec)
            self._write_raw(package.raw_path, n_rows=4)
            package.start_run(step_id=0, step_ns=self._step_ns(0))
            package.abort_run(
                step_id=1,
                step_ns=self._step_ns(1) + 1,
                reason="test",
            )
            owner = root / "owner.json"
            owner.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                TransitionContractError, "step/time pair does not match"
            ):
                package.seal(
                    git_commit="b" * 40,
                    resolved_config_sha256="c" * 64,
                    owner_artifacts={"owner": owner},
                    stop_reason="test",
                )

    def test_goal_commit_requires_three_acknowledgements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sequence, _ = build_session_manifests(
                session_id="s03",
                seed=5,
                created_at_utc="2026-08-13T00:00:00Z",
            )
            run_id = sequence["blocks"][0]["runs"][0]["run_id"]
            package = TransitionRunPackage(
                run_dir=Path(tmp) / "run",
                run_spec=find_run_spec(sequence, run_id),
            )
            package.start_run(step_id=0, step_ns=self._step_ns(0))
            package.mark_initial_ready(step_id=1, step_ns=self._step_ns(1))
            with self.assertRaisesRegex(
                TransitionContractError, "missing acknowledgements: display"
            ):
                package.commit_next_goal(
                    step_id=2,
                    step_ns=self._step_ns(2),
                    commit_ack_sources=("recorder", "router"),
                )

    @staticmethod
    def _step_ns(step: int) -> int:
        return 1_000_000_000 + step * 20_000_000

    @classmethod
    def _write_raw(cls, path: Path, *, n_rows: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            observations = handle.create_group("observations")
            observations.create_dataset(
                "qpos", data=np.zeros((n_rows, 4), dtype=np.float32)
            )
            observations.create_dataset(
                "qvel", data=np.zeros((n_rows, 4), dtype=np.float32)
            )
            handle.create_dataset(
                "action", data=np.zeros((n_rows, 4), dtype=np.float32)
            )
            timestamps = handle.create_group("timestamps")
            timestamps.create_dataset("step_id", data=np.arange(n_rows, dtype=np.int64))
            timestamps.create_dataset(
                "step_ns",
                data=np.asarray([cls._step_ns(step) for step in range(n_rows)]),
            )


class RealTransitionRuntimeTest(unittest.TestCase):
    def test_cycle_timeout_is_fail_closed_on_a_recorded_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = prepare_session_directory(
                output_root=root,
                session_id="timeout01",
                seed=17,
                created_at_utc="2026-08-13T00:00:00Z",
            )
            session_dir = Path(prepared["session_dir"])
            home_contract = session_dir / "home_side_contract.json"
            _write_valid_home_contract(home_contract)
            runtime = TransitionTaskRuntime(
                session_dir=session_dir,
                sequence_manifest_path=prepared["sequence_manifest"],
                split_manifest_path=prepared["split_manifest"],
                home_side_contract_path=home_contract,
                resolved_record_config_yaml="task: real_transition\n",
                git_commit="e" * 40,
                cycle_review_s=0.02,
                cycle_stop_s=0.04,
                run_stop_s=1.0,
            )
            runtime.update_receiver_state(mode="armed", health_ok=True)
            runtime.handle_command(
                "start-run",
                {"field_context": {"workface_reset_id": "wf01"}},
            )
            runtime.handle_command("initial-ready")
            self.assertTrue(runtime.consume_record_start_request())
            runtime.attach_recording(episode_idx=0)
            runtime.update_recorded_step(
                step_id=0,
                step_ns=RealTransitionRunPackageTest._step_ns(0),
            )
            runtime.handle_command("commit-goal", {"display_ack": True})
            runtime.update_recorded_step(
                step_id=1,
                step_ns=RealTransitionRunPackageTest._step_ns(1),
            )
            self.assertEqual(runtime.status()["timing_warning"], "cycle_review")
            runtime.update_recorded_step(
                step_id=2,
                step_ns=RealTransitionRunPackageTest._step_ns(2),
            )
            stop = runtime.consume_stop_request()
            self.assertIsNotNone(stop)
            assert stop is not None
            self.assertFalse(stop.success)
            self.assertEqual(stop.stop_reason, "cycle_timeout")

    def test_task_server_drives_manual_markers_without_sending_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = prepare_session_directory(
                output_root=root,
                session_id="runtime01",
                seed=13,
                created_at_utc="2026-08-13T00:00:00Z",
            )
            session_dir = Path(prepared["session_dir"])
            home_contract = session_dir / "home_side_contract.json"
            _write_valid_home_contract(home_contract)
            runtime = TransitionTaskRuntime(
                session_dir=session_dir,
                sequence_manifest_path=prepared["sequence_manifest"],
                split_manifest_path=prepared["split_manifest"],
                home_side_contract_path=home_contract,
                resolved_record_config_yaml="task: real_transition\n",
                git_commit="d" * 40,
            )
            runtime.update_receiver_state(mode="armed", health_ok=True)
            server = TransitionTaskServer(
                runtime=runtime, bind_host="127.0.0.1", port=0
            )
            server.start()
            try:
                started = send_transition_command(
                    host="127.0.0.1",
                    port=server.port,
                    command="start-run",
                    payload={
                        "field_context": {
                            "workface_reset_id": "wf01",
                            "workface_action": "restore",
                        }
                    },
                )
                self.assertTrue(started["active"])
                self.assertFalse(runtime.consume_record_start_request())
                send_transition_command(
                    host="127.0.0.1",
                    port=server.port,
                    command="initial-ready",
                )
                self.assertTrue(runtime.consume_record_start_request())
                runtime.attach_recording(episode_idx=0)
                for step in range(32):
                    runtime.update_recorded_step(
                        step_id=step,
                        step_ns=RealTransitionRunPackageTest._step_ns(step),
                    )
                    cycle_offset = step - 2
                    if cycle_offset >= 0:
                        within = cycle_offset % 7
                        cycle = cycle_offset // 7
                        if cycle < 4 and within == 0:
                            send_transition_command(
                                host="127.0.0.1",
                                port=server.port,
                                command="commit-goal",
                                payload={"display_ack": True},
                            )
                        elif cycle < 4 and within == 3:
                            send_transition_command(
                                host="127.0.0.1",
                                port=server.port,
                                command="dump-end",
                            )
                        elif cycle < 4 and within == 6:
                            send_transition_command(
                                host="127.0.0.1",
                                port=server.port,
                                command="target-ready",
                            )
                stop = runtime.consume_stop_request()
                self.assertIsNotNone(stop)
                assert stop is not None
                self.assertTrue(stop.success)
                RealTransitionRunPackageTest._write_raw(
                    runtime.active_raw_path,
                    n_rows=32,
                )
                manifest = runtime.seal_saved_run(
                    raw_path=runtime.active_raw_path,
                    stop_reason=stop.stop_reason,
                )
                self.assertEqual(manifest["status"], "complete")
                run_dir = Path(started["raw_path"]).parent
                self.assertEqual(verify_run_package(run_dir)["status"], "PASS")
            finally:
                server.close()


class HomeSideContractTest(unittest.TestCase):
    def test_field_windows_resolve_historical_floor_and_clean_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = _valid_home_calibration(root)
            contract = build_home_side_contract(
                calibration,
                source_base_dir=root,
            )
            validate_home_side_contract(contract)
            home = contract["home_reference"]
            self.assertEqual(home["classification_deadband_rad"], 0.05)
            self.assertAlmostEqual(
                home["clean_endpoint_min_abs_side_coordinate_rad"],
                0.08,
            )
            sides = {side["side_id"]: side for side in contract["sides"]}
            self.assertLessEqual(
                sides["A"]["demonstrated_side_coordinate_support_rad"][1],
                -0.14,
            )
            self.assertGreaterEqual(
                sides["B"]["demonstrated_side_coordinate_support_rad"][0],
                0.14,
            )

    def test_home_contract_rejects_fewer_than_ten_independent_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = _valid_home_calibration(root)
            calibration["samples"] = [
                sample
                for sample in calibration["samples"]
                if not (
                    sample["side"] == "home" and sample["reference_id"] == "home_09"
                )
            ]
            with self.assertRaisesRegex(
                TransitionContractError, "at least 10 accepted home windows"
            ):
                build_home_side_contract(calibration, source_base_dir=root)


def _write_valid_home_contract(path: Path) -> None:
    contract = build_home_side_contract(
        _valid_home_calibration(path.parent),
        source_base_dir=path.parent,
    )
    path.write_text(
        json.dumps(contract, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _valid_home_calibration(root: Path) -> dict[str, object]:
    source_config = root / "field_home_source.yaml"
    source_config.write_text("home_pose_rad: [0, 0, 0, 0]\n", encoding="utf-8")
    samples = []
    home_offsets = np.linspace(-0.004, 0.005, 10)
    for side, centres in (
        ("home", home_offsets),
        ("A", np.linspace(0.15, 0.24, 10)),
        ("B", np.linspace(-0.24, -0.15, 10)),
    ):
        for index, centre in enumerate(centres):
            qpos = np.asarray(
                [
                    [centre - 0.0001, 0.10, -0.20, 0.30],
                    [centre + 0.0001, 0.1001, -0.1999, 0.2999],
                ],
                dtype=np.float64,
            )
            samples.append(
                {
                    "reference_id": f"{side}_{index:02d}",
                    "side": side,
                    "accepted": True,
                    "stable_duration_s": 0.5,
                    "visual_confirmed": True,
                    "visual_reference_ids": [f"frame_{side}_{index:02d}"],
                    "commanded_action_abs_max": 0.0,
                    "qpos_samples_rad": qpos.tolist(),
                    "qvel_samples_rad_s": np.zeros((2, 4), dtype=np.float64).tolist(),
                }
            )
    return {
        "schema": "real_transition_home_calibration_samples_v1",
        "context_version": "ctx01",
        "physical_left_qpos_sign": 1,
        "resolved_by": "field_engineer",
        "resolved_at": "2026-08-13T00:00:00Z",
        "home_reference": {
            "source_config": str(source_config),
            "source_value_path": "home_pose_rad",
            "home_pose_rad": [0.0, 0.0, 0.0, 0.0],
        },
        "samples": samples,
    }


if __name__ == "__main__":
    unittest.main()
