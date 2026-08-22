from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from testbed.backends.real.bridge import InProcessMockBridgeClient
from testbed.cli.record_real import RecordSession, _load_yaml_config
from testbed.cli.teleop_remote import _load_yaml_config as _load_sender_yaml_config
from testbed.data.recorder import EpisodeRecorder
from testbed.tasks.home_side_calibration import (
    _camera_image_from_payload,
    capture_home_calibration_window,
    initialise_home_calibration,
)
from testbed.tasks.home_side_contract import (
    build_rule_ready_contract,
    build_home_side_contract,
    classify_ready_swing_qpos,
    validate_rule_ready_contract,
    validate_home_side_contract,
)
from testbed.tasks.real_transition import (
    ATOMIC_TRANSITIONS,
    DATA_CONTRACT_VERSION,
    LEGACY_DATA_CONTRACT_VERSION,
    LEGACY_RUN_MANIFEST_SCHEMA,
    LEGACY_SEQUENCE_MANIFEST_SCHEMA,
    LEGACY_TASK_EVENT_SCHEMA,
    REQUIRED_GOAL_ACK_SOURCES,
    RUN_MANIFEST_SCHEMA,
    SEQUENCE_MANIFEST_SCHEMA,
    TransitionContractError,
    TransitionRunPackage,
    TransitionRunSpec,
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
from testbed.transition_control_client import (
    send_transition_command as lightweight_send_transition_command,
)


READY_CONFIRMATIONS = {
    "bucket_clear_confirmed": True,
    "operator_confirmed": True,
}


def _ready_event_evidence(side: str) -> dict[str, object]:
    return {
        "actual_side": side,
        "expected_side": side,
        "window_complete": True,
        "sample_gap_ok": True,
        "swing_stable": True,
        "clean_side_window": True,
        "bucket_clear_confirmed": True,
        "operator_confirmed": True,
        "non_swing_axes_gate_ready": False,
    }


def _feed_ready_window(
    runtime: TransitionTaskRuntime,
    *,
    side: str,
    end_ns: int,
    swing_qvel: float = 0.0,
) -> None:
    swing_qpos = {
        "A": -0.10,
        "B": 0.10,
        "home": 0.000690,
        "transition": 0.06,
        "outside": 0.50,
    }[side]
    for offset in range(10, -1, -1):
        runtime.update_ready_observation(
            step_ns=end_ns - offset * 50_000_000,
            qpos=[swing_qpos, 1.7, -2.4, 0.9],
            qvel=[swing_qvel, 3.0, -4.0, 5.0],
        )


def _feed_camera_sync_window(
    runtime: TransitionTaskRuntime,
    *,
    start_ns: int,
    count: int,
    skew_ms: float,
    error_flag_cameras: tuple[str, ...] = (),
) -> None:
    for index in range(count):
        group_id = index + 1
        runtime.update_camera_sync_observation(
            step_ns=start_ns + index * 20_000_000,
            image_metadata={
                camera: {
                    "group_id": group_id,
                    "group_camera_count": 4,
                    "group_valid": 1,
                    "group_skew_ms": skew_ms,
                    "v4l2_error": int(camera in error_flag_cameras),
                }
                for camera in ("video4", "video5", "video6", "video7")
            },
        )


class RealTransitionPlanTest(unittest.TestCase):
    def test_manual_profile_inherits_field_contract_and_preserves_field_joysticks(
        self,
    ) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "testbed"
            / "configs"
            / "teleop_real_transition_v2_0_1.yaml"
        )
        config = _load_yaml_config(config_path)
        self.assertEqual(config["teleop"]["input"], "joystick")
        self.assertIsNone(config["teleop"]["joystick"]["policy_start_button"])
        self.assertIsNone(config["teleop"]["joystick"]["record_start_button"])
        self.assertEqual(config["teleop"]["joystick"]["go_home_button"], 2)
        self.assertEqual(
            config["teleop"]["joystick"]["status_reserved_buttons"], [1, 3]
        )
        self.assertTrue(config["teleop"]["recording"]["go_home"]["enabled"])
        self.assertEqual(
            config["teleop"]["joystick"]["joystick_ids"], [0, 1, 0, 1]
        )
        self.assertEqual(config["teleop"]["joystick"]["axis_map"], [1, 1, 0, 0])
        self.assertEqual(
            config["teleop"]["joystick"]["invert"], [True, False, True, False]
        )
        self.assertEqual(
            config["task"]["camera_names"], ["video4", "video5", "video6", "video7"]
        )
        self.assertEqual(config["task"]["max_steps"], 15000)
        self.assertIn("ready_contract", config["real_transition"])
        self.assertNotIn("home_side_contract", config["real_transition"])
        self.assertEqual(
            config["real_transition"]["time_limits"]["run_stop_s_per_cycle"],
            60.0,
        )
        self.assertEqual(config["receiver"]["health"]["mode"], "strict")

        sender_config = _load_sender_yaml_config(config_path)
        self.assertEqual(
            sender_config["teleop"]["joystick"], config["teleop"]["joystick"]
        )

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
        self.assertEqual(first_sequence["schema"], SEQUENCE_MANIFEST_SCHEMA)
        self.assertEqual(first_sequence["data_contract_version"], DATA_CONTRACT_VERSION)
        summary = summarize_sequence_manifest(first_sequence)
        self.assertEqual(
            summary["blocks"],
            {"locked_test": 1, "train": 4, "validation": 1},
        )
        self.assertEqual(
            summary["runs"],
            {"locked_test": 4, "train": 16, "validation": 4},
        )
        self.assertEqual(
            summary["cycles"],
            {"locked_test": 16, "train": 64, "validation": 16},
        )
        self.assertEqual(summary["cycle_lengths"], {"3": 8, "4": 8, "5": 8})
        self.assertEqual(
            summary["transitions"],
            {transition: 24 for transition in ATOMIC_TRANSITIONS},
        )
        self.assertEqual(
            summary["transitions_by_priority_tier"],
            {
                "minimum_64_cycle": {
                    transition: 16 for transition in ATOMIC_TRANSITIONS
                },
                "train_expansion_96_cycle": {
                    transition: 8 for transition in ATOMIC_TRANSITIONS
                },
            },
        )
        self.assertEqual(summary["unique_sequence_count"], 24)
        self.assertEqual(
            summary["matched_start_pairs_by_split"],
            {"locked_test": 2, "train": 8, "validation": 2},
        )
        self.assertEqual(
            summary["pair_first_targets_by_balance_group"],
            {
                group: {
                    "A": {"A": 1, "B": 1},
                    "B": {"A": 1, "B": 1},
                }
                for group in ("evaluation", "minimum_train", "train_expansion")
            },
        )
        all_sequences = [
            tuple(run["sequence"])
            for block in first_sequence["blocks"]
            for run in block["runs"]
        ]
        self.assertEqual(len(set(all_sequences)), 24)
        self.assertNotIn(("A", "B", "B", "A", "A"), all_sequences)
        self.assertNotIn(("B", "A", "A", "B", "B"), all_sequences)
        for block in first_sequence["blocks"]:
            runs = block["runs"]
            self.assertEqual(
                {
                    f"{run['initial_side']}->{run['scripted_targets'][0]}"
                    for run in runs
                },
                set(ATOMIC_TRANSITIONS),
            )
            for cycle_index in range(3):
                self.assertEqual(
                    sorted(run["scripted_targets"][cycle_index] for run in runs),
                    ["A", "A", "B", "B"],
                )
            pair_ids = {run["matched_start_pair_id"] for run in runs}
            self.assertEqual(len(pair_ids), 2)
            for pair_id in pair_ids:
                members = [
                    run for run in runs if run["matched_start_pair_id"] == pair_id
                ]
                self.assertEqual(
                    {run["matched_start_pair_member_rank"] for run in members},
                    {0, 1},
                )
                self.assertEqual(
                    abs(
                        members[0]["run_rank_in_block"]
                        - members[1]["run_rank_in_block"]
                    ),
                    1,
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

        other_sequence, _ = build_session_manifests(
            session_id="field_20260813",
            seed=20260814,
            created_at_utc="2026-08-13T00:00:00Z",
        )
        self.assertNotEqual(
            [
                run["sequence_id"]
                for block in first_sequence["blocks"]
                for run in block["runs"]
            ],
            [
                run["sequence_id"]
                for block in other_sequence["blocks"]
                for run in block["runs"]
            ],
        )

    def test_core_run_contract_accepts_seven_cycles(self) -> None:
        spec = TransitionRunSpec(
            session_id="free01",
            block_id="b01",
            run_id="b01_r01",
            split="train",
            sequence_id="L7_ABAABBAB",
            collection_rank=0,
            run_rank_in_block=0,
            sequence=tuple("ABAABBAB"),
        )
        spec.validate()
        self.assertEqual(spec.cycle_count, 7)

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
            ready_contract = Path(first["ready_contract"])
            self.assertTrue(ready_contract.is_file())
            validate_rule_ready_contract(
                json.loads(ready_contract.read_text(encoding="utf-8"))
            )
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

    def test_variable_cycle_round_trip_and_checksum_verification(self) -> None:
        sequence, _ = build_session_manifests(
            session_id="s01",
            seed=11,
            created_at_utc="2026-08-13T00:00:00Z",
        )
        specs_by_length = {}
        for block in sequence["blocks"]:
            for run in block["runs"]:
                spec = find_run_spec(sequence, run["run_id"])
                specs_by_length.setdefault(spec.cycle_count, spec)

        for cycle_count in (3, 4, 5):
            with self.subTest(
                cycle_count=cycle_count
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                spec = specs_by_length[cycle_count]
                run_dir = root / f"block_{spec.block_id}" / f"run_{spec.run_id}"
                package = TransitionRunPackage(run_dir=run_dir, run_spec=spec)
                n_rows = 2 + cycle_count * 7
                self._write_raw(run_dir / "raw.hdf5", n_rows=n_rows)
                package.start_run(step_id=0, step_ns=self._step_ns(0))
                package.mark_initial_ready(
                    step_id=1,
                    step_ns=self._step_ns(1),
                    ready_evidence=_ready_event_evidence(spec.initial_side),
                )
                last_step = 1
                for cycle_index in range(cycle_count):
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
                        realized_target_side=spec.targets[cycle_index],
                        ready_evidence=_ready_event_evidence(
                            spec.targets[cycle_index]
                        ),
                    )
                    last_step = base + 6
                package.complete_run(
                    step_id=last_step,
                    step_ns=self._step_ns(last_step),
                )

                sequence_path = root / "sequence_manifest.json"
                sequence_path.write_text(
                    json.dumps(sequence, sort_keys=True) + "\n", encoding="utf-8"
                )
                ready_contract = root / "ready_contract.json"
                ready_contract.write_text(
                    json.dumps(build_rule_ready_contract(), sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                resolved = root / "resolved_record_config.yaml"
                resolved.write_text("task: real_transition\n", encoding="utf-8")
                manifest = package.seal(
                    git_commit="a" * 40,
                    resolved_config_sha256=sha256_file(resolved),
                    owner_artifacts={
                        "sequence_manifest": sequence_path,
                        "ready_contract": ready_contract,
                        "resolved_record_config": resolved,
                    },
                    field_context={"workface_reset_id": "wf01"},
                )
                self.assertEqual(manifest["schema"], RUN_MANIFEST_SCHEMA)
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["completed_cycles"], cycle_count)
                self.assertEqual(manifest["realized_targets"], list(spec.targets))

                report = verify_run_package(run_dir)
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(
                    report["event_summary"]["goal_commits"], cycle_count
                )
                self.assertEqual(
                    report["alignment"]["n_events"], 3 * cycle_count + 3
                )

                with (run_dir / "task_events.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write("{}\n")
                with self.assertRaisesRegex(
                    TransitionContractError, "checksum mismatch"
                ):
                    verify_run_package(run_dir)

    def test_legacy_four_cycle_package_remains_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = TransitionRunSpec(
                session_id="legacy01",
                block_id="b01",
                run_id="b01_r01",
                split="train",
                sequence_id="legacy_P0",
                collection_rank=0,
                run_rank_in_block=0,
                sequence=("A", "B", "B", "A", "A"),
                manifest_schema=LEGACY_SEQUENCE_MANIFEST_SCHEMA,
                data_contract_version=LEGACY_DATA_CONTRACT_VERSION,
                task_event_schema=LEGACY_TASK_EVENT_SCHEMA,
                run_manifest_schema=LEGACY_RUN_MANIFEST_SCHEMA,
                legacy_template_id="P0",
            )
            package = TransitionRunPackage(run_dir=root / "run", run_spec=spec)
            self._write_raw(package.raw_path, n_rows=30)
            package.start_run(step_id=0, step_ns=self._step_ns(0))
            package.mark_initial_ready(step_id=1, step_ns=self._step_ns(1))
            last_step = 1
            for cycle_index in range(4):
                base = 2 + cycle_index * 7
                package.commit_next_goal(
                    step_id=base,
                    step_ns=self._step_ns(base),
                    commit_ack_sources=REQUIRED_GOAL_ACK_SOURCES,
                )
                package.mark_dump_end(
                    step_id=base + 3,
                    step_ns=self._step_ns(base + 3),
                )
                package.mark_target_ready(
                    step_id=base + 6,
                    step_ns=self._step_ns(base + 6),
                )
                last_step = base + 6
            package.complete_run(
                step_id=last_step,
                step_ns=self._step_ns(last_step),
            )
            owner = root / "owner.json"
            owner.write_text("{}\n", encoding="utf-8")
            package.seal(
                git_commit="b" * 40,
                resolved_config_sha256="c" * 64,
                owner_artifacts={"owner": owner},
            )
            self.assertEqual(verify_run_package(package.run_dir)["status"], "PASS")

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
            package.mark_initial_ready(
                step_id=1,
                step_ns=self._step_ns(1),
                ready_evidence=_ready_event_evidence(package.run_spec.initial_side),
            )
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
        return 1_000_000_000 + step * 100_000_000

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
    def test_camera_sync_gate_blocks_start_until_a_coherent_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepared = prepare_session_directory(
                output_root=Path(tmp),
                session_id="camera_gate01",
                seed=23,
                created_at_utc="2026-08-21T00:00:00Z",
            )
            runtime = TransitionTaskRuntime(
                session_dir=prepared["session_dir"],
                sequence_manifest_path=prepared["sequence_manifest"],
                split_manifest_path=prepared["split_manifest"],
                ready_contract_path=prepared["ready_contract"],
                resolved_record_config_yaml="task: real_transition\n",
                git_commit="a" * 40,
                camera_sync={
                    "enabled": True,
                    "expected_cameras": [
                        "video4",
                        "video5",
                        "video6",
                        "video7",
                    ],
                    "stable_window_s": 0.2,
                    "max_group_skew_ms": 5.0,
                    "min_valid_fraction": 0.98,
                    "min_distinct_groups": 10,
                },
            )
            runtime.update_receiver_state(mode="armed", health_ok=True)
            with self.assertRaisesRegex(
                TransitionContractError, "camera_sync_no_samples"
            ):
                runtime.handle_mark()

            _feed_camera_sync_window(
                runtime,
                start_ns=1_000_000_000,
                count=11,
                skew_ms=8.0,
            )
            self.assertIn(
                "camera_sync_valid_fraction",
                runtime.status()["camera_sync_state"]["blockers"],
            )
            with self.assertRaisesRegex(
                TransitionContractError, "camera_sync_valid_fraction"
            ):
                runtime.handle_mark()

            passing = TransitionTaskRuntime(
                session_dir=prepared["session_dir"],
                sequence_manifest_path=prepared["sequence_manifest"],
                split_manifest_path=prepared["split_manifest"],
                ready_contract_path=prepared["ready_contract"],
                resolved_record_config_yaml="task: real_transition\n",
                git_commit="a" * 40,
                camera_sync={
                    "enabled": True,
                    "stable_window_s": 0.2,
                    "max_group_skew_ms": 5.0,
                    "min_valid_fraction": 0.98,
                    "min_distinct_groups": 10,
                },
            )
            passing.update_receiver_state(mode="armed", health_ok=True)
            _feed_camera_sync_window(
                passing,
                start_ns=2_000_000_000,
                count=11,
                skew_ms=0.04,
                error_flag_cameras=("video4", "video5"),
            )
            camera_state = passing.status()["camera_sync_state"]
            self.assertTrue(camera_state["ready"])
            self.assertEqual(camera_state["blockers"], [])
            self.assertEqual(
                camera_state["v4l2_error_flag_cameras"],
                ["video4", "video5"],
            )
            started = passing.handle_mark()
            self.assertEqual(started["mark_action"], "start-run")

    def test_one_state_aware_mark_advances_run_and_goals_are_automatic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepared = prepare_session_directory(
                output_root=Path(tmp),
                session_id="mark01",
                seed=23,
                created_at_utc="2026-08-21T00:00:00Z",
            )
            runtime = TransitionTaskRuntime(
                session_dir=prepared["session_dir"],
                sequence_manifest_path=prepared["sequence_manifest"],
                split_manifest_path=prepared["split_manifest"],
                ready_contract_path=prepared["ready_contract"],
                resolved_record_config_yaml="task: real_transition\n",
                git_commit="a" * 40,
                field_context_defaults={
                    "workface_reset_id_prefix": "wf_",
                    "workface_action": "fresh_strip",
                },
            )
            runtime.update_receiver_state(mode="armed", health_ok=True)
            started = runtime.handle_mark()
            self.assertEqual(started["mark_action"], "start-run")
            self.assertEqual(
                started["field_context"]["workface_reset_id"], "wf_001"
            )
            initial = str(started["initial_side"])
            _feed_ready_window(runtime, side=initial, end_ns=1_000_000_000)
            initial_mark = runtime.handle_mark()
            self.assertEqual(initial_mark["mark_action"], "initial-ready")
            self.assertTrue(runtime.consume_record_start_request())
            runtime.attach_recording(episode_idx=0)
            runtime.update_recorded_step(step_id=0, step_ns=1_000_000_000)
            self.assertEqual(runtime.status()["phase"], "goal_committed")

            cycle_count = int(runtime.status()["planned_cycle_count"])
            step_id = 0
            for cycle_index in range(cycle_count):
                step_id += 1
                runtime.update_recorded_step(
                    step_id=step_id,
                    step_ns=1_000_000_000 + step_id * 100_000_000,
                )
                dump = runtime.handle_mark()
                self.assertEqual(dump["mark_action"], "dump-end")

                step_id += 1
                target = str(runtime.status()["next_target_side"])
                target_ns = 1_000_000_000 + step_id * 100_000_000
                _feed_ready_window(runtime, side=target, end_ns=target_ns)
                runtime.update_recorded_step(step_id=step_id, step_ns=target_ns)
                ready = runtime.handle_mark()
                self.assertEqual(ready["mark_action"], "target-ready")
                expected_phase = (
                    "complete"
                    if cycle_index + 1 == cycle_count
                    else "goal_committed"
                )
                self.assertEqual(runtime.status()["phase"], expected_phase)

            stop = runtime.consume_stop_request()
            self.assertIsNotNone(stop)
            assert stop is not None
            self.assertTrue(stop.success)
            package = runtime._active_package  # noqa: SLF001 - event ownership check.
            assert package is not None
            operator_events = [
                event
                for event in package._events  # noqa: SLF001
                if event["event_type"]
                in {"initial_ready_mark", "dump_end_mark", "target_ready_mark"}
            ]
            self.assertTrue(operator_events)
            self.assertTrue(
                all(event["event_source"] == "operator" for event in operator_events)
            )
            goals = [
                event
                for event in package._events  # noqa: SLF001
                if event["event_type"] == "goal_commit"
            ]
            self.assertEqual(len(goals), cycle_count)
            self.assertTrue(
                all(event["event_source"] == "sequencer" for event in goals)
            )

    def test_one_session_arm_runs_all_cycle_boundaries_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepared = prepare_session_directory(
                output_root=Path(tmp),
                session_id="auto_arm01",
                seed=31,
                created_at_utc="2026-08-21T00:00:00Z",
            )
            runtime = TransitionTaskRuntime(
                session_dir=prepared["session_dir"],
                sequence_manifest_path=prepared["sequence_manifest"],
                split_manifest_path=prepared["split_manifest"],
                ready_contract_path=prepared["ready_contract"],
                resolved_record_config_yaml="task: real_transition\n",
                git_commit="a" * 40,
                field_context_defaults={
                    "workface_reset_id_prefix": "wf_",
                    "workface_action": "fresh_strip",
                },
                automatic_annotation={
                    "enabled": True,
                    "activity_action_abs_min": 0.05,
                    "require_inter_run_activity": True,
                },
            )
            runtime.update_receiver_state(mode="armed", health_ok=True)
            next_spec = runtime._next_run_spec_hint  # noqa: SLF001
            assert next_spec is not None
            _feed_ready_window(
                runtime,
                side=next_spec.initial_side,
                end_ns=1_000_000_000,
            )
            runtime.update_operator_action(action=[0.0, 0.0, 0.0, 0.0])

            armed = runtime.handle_mark()
            self.assertEqual(armed["mark_action"], "arm-session")
            self.assertTrue(armed["session_armed"])
            started = runtime.advance_automatic_workflow(
                allow_recorded_events=False
            )
            self.assertIsNotNone(started)
            self.assertTrue(runtime.consume_record_start_request())
            runtime.attach_recording(episode_idx=0)
            runtime.update_recorded_step(step_id=0, step_ns=1_000_000_000)
            runtime.advance_automatic_workflow(allow_recorded_events=True)
            self.assertEqual(runtime.status()["phase"], "goal_committed")

            cycle_count = int(runtime.status()["planned_cycle_count"])
            step_id = 0
            step_ns = 1_000_000_000
            for cycle_index in range(cycle_count):
                for excursion_sample in range(3):
                    step_id += 1
                    step_ns += 20_000_000
                    runtime.update_ready_observation(
                        step_ns=step_ns,
                        qpos=[1.60, 0.0, 0.0, 0.0],
                        qvel=[0.2, 0.0, 0.0, 0.0],
                    )
                    runtime.update_recorded_step(
                        step_id=step_id, step_ns=step_ns
                    )
                    runtime.advance_automatic_workflow(
                        allow_recorded_events=True
                    )
                    if excursion_sample < 2:
                        self.assertFalse(
                            runtime.status()["cycle_excursion_observed"]
                        )
                self.assertTrue(runtime.status()["cycle_excursion_observed"])

                target = str(runtime.status()["next_target_side"])
                step_id += 1
                step_ns += 600_000_000
                _feed_ready_window(runtime, side=target, end_ns=step_ns)
                runtime.update_recorded_step(step_id=step_id, step_ns=step_ns)
                completed = runtime.advance_automatic_workflow(
                    allow_recorded_events=True
                )
                self.assertIsNotNone(completed)
                expected_phase = (
                    "complete"
                    if cycle_index + 1 == cycle_count
                    else "goal_committed"
                )
                self.assertEqual(runtime.status()["phase"], expected_phase)

            stop = runtime.consume_stop_request()
            self.assertIsNotNone(stop)
            assert stop is not None
            self.assertTrue(stop.success)
            package = runtime._active_package  # noqa: SLF001
            assert package is not None
            event_types = [
                event["event_type"]
                for event in package._events  # noqa: SLF001
            ]
            self.assertEqual(event_types.count("dump_end_mark"), 0)
            self.assertEqual(
                event_types.count("cycle_excursion_observed"), cycle_count
            )
            automatic_ready = [
                event
                for event in package._events  # noqa: SLF001
                if event["event_type"]
                in {"initial_ready_mark", "target_ready_mark"}
            ]
            self.assertTrue(
                all(event["event_source"] == "automatic" for event in automatic_ready)
            )

    def test_ready_gate_uses_only_stable_clean_swing_plus_confirmations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepared = prepare_session_directory(
                output_root=Path(tmp),
                session_id="ready01",
                seed=23,
                created_at_utc="2026-08-17T00:00:00Z",
            )
            runtime = TransitionTaskRuntime(
                session_dir=prepared["session_dir"],
                sequence_manifest_path=prepared["sequence_manifest"],
                split_manifest_path=prepared["split_manifest"],
                ready_contract_path=prepared["ready_contract"],
                resolved_record_config_yaml="task: real_transition\n",
                git_commit="a" * 40,
            )
            runtime.update_receiver_state(mode="armed", health_ok=True)
            idle = runtime.status()
            self.assertEqual(idle["sealed_run_count"], 0)
            self.assertEqual(idle["next_run_id"], "b01_r01")
            self.assertEqual(idle["next_run_ordinal"], 1)
            runtime.handle_command(
                "start-run",
                {
                    "field_context": {
                        "workface_reset_id": "wf01",
                        "workface_action": "restore",
                    }
                },
            )
            expected = runtime.status()["initial_side"]
            for index, (side, error) in enumerate(
                (
                    ("home", "swing_side_home"),
                    ("transition", "swing_side_transition"),
                    ("outside", "swing_side_outside_safe_range"),
                )
            ):
                _feed_ready_window(
                    runtime,
                    side=side,
                    end_ns=1_000_000_000 + index * 600_000_000,
                )
                with self.assertRaisesRegex(TransitionContractError, error):
                    runtime.handle_command("initial-ready", READY_CONFIRMATIONS)

            _feed_ready_window(
                runtime,
                side=expected,
                end_ns=2_800_000_000,
                swing_qvel=0.0151,
            )
            with self.assertRaisesRegex(TransitionContractError, "swing_not_stable"):
                runtime.handle_command("initial-ready", READY_CONFIRMATIONS)

            _feed_ready_window(
                runtime,
                side=expected,
                end_ns=3_400_000_000,
            )
            state = runtime.status()["ready_state"]
            self.assertEqual(state["actual_side"], expected)
            self.assertEqual(state["blockers"], [])
            self.assertEqual(
                state["non_swing_qvel_abs_max_rad_s"],
                [3.0, 4.0, 5.0],
            )
            self.assertFalse(state["non_swing_axes_gate_ready"])
            with self.assertRaisesRegex(
                TransitionContractError, "bucket_clear_confirmed=true"
            ):
                runtime.handle_command(
                    "initial-ready", {"operator_confirmed": True}
                )
            runtime.handle_command("initial-ready", READY_CONFIRMATIONS)
            self.assertTrue(runtime.consume_record_start_request())

    def test_ready_window_accepts_real_50hz_timestamps_without_exact_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepared = prepare_session_directory(
                output_root=Path(tmp),
                session_id="ready_real_cadence01",
                seed=23,
                created_at_utc="2026-08-17T00:00:00Z",
            )
            runtime = TransitionTaskRuntime(
                session_dir=prepared["session_dir"],
                sequence_manifest_path=prepared["sequence_manifest"],
                split_manifest_path=prepared["split_manifest"],
                ready_contract_path=prepared["ready_contract"],
                resolved_record_config_yaml="task: real_transition\n",
                git_commit="a" * 40,
            )
            runtime.update_receiver_state(mode="armed", health_ok=True)
            started = runtime.handle_command(
                "start-run",
                {
                    "field_context": {
                        "workface_reset_id": "wf01",
                        "workface_action": "restore",
                    }
                },
            )
            swing_qpos = -0.10 if started["initial_side"] == "A" else 0.10
            end_ns = 2_000_000_000
            for offset in range(30, -1, -1):
                runtime.update_ready_observation(
                    step_ns=end_ns - offset * 20_500_000,
                    qpos=[swing_qpos, 1.7, -2.4, 0.9],
                    qvel=[0.0, 3.0, -4.0, 5.0],
                )

            state = runtime.status()["ready_state"]
            self.assertGreaterEqual(state["window_duration_s"], 0.5)
            self.assertTrue(state["window_complete"])
            self.assertEqual(state["blockers"], [])

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
            ready_contract = Path(prepared["ready_contract"])
            runtime = TransitionTaskRuntime(
                session_dir=session_dir,
                sequence_manifest_path=prepared["sequence_manifest"],
                split_manifest_path=prepared["split_manifest"],
                ready_contract_path=ready_contract,
                resolved_record_config_yaml="task: real_transition\n",
                git_commit="e" * 40,
                cycle_review_s=0.05,
                cycle_stop_s=0.15,
                run_stop_s=1.0,
            )
            runtime.update_receiver_state(mode="armed", health_ok=True)
            with self.assertRaisesRegex(
                TransitionContractError, "requires workface_action"
            ):
                runtime.handle_command(
                    "start-run",
                    {"field_context": {"workface_reset_id": "wf01"}},
                )
            self.assertFalse(any(session_dir.glob("block_*")))
            runtime.handle_command(
                "start-run",
                {
                    "field_context": {
                        "workface_reset_id": "wf01",
                        "workface_action": "restore",
                    }
                },
            )
            _feed_ready_window(
                runtime,
                side=runtime.status()["initial_side"],
                end_ns=RealTransitionRunPackageTest._step_ns(0),
            )
            runtime.handle_command("initial-ready", READY_CONFIRMATIONS)
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
            ready_contract = Path(prepared["ready_contract"])
            runtime = TransitionTaskRuntime(
                session_dir=session_dir,
                sequence_manifest_path=prepared["sequence_manifest"],
                split_manifest_path=prepared["split_manifest"],
                ready_contract_path=ready_contract,
                resolved_record_config_yaml="task: real_transition\n",
                git_commit="d" * 40,
            )
            runtime.update_receiver_state(mode="armed", health_ok=True)
            server = TransitionTaskServer(
                runtime=runtime, bind_host="127.0.0.1", port=0
            )
            server.start()
            try:
                started = lightweight_send_transition_command(
                    host="127.0.0.1",
                    port=server.port,
                    command="start-run",
                    payload={
                        "field_context": {
                            "workface_reset_id": "wf01",
                            "workface_action": "restore",
                            "planned_sequence_id": "must_not_override_manifest",
                        }
                    },
                )
                self.assertTrue(started["active"])
                cycle_count = int(started["planned_cycle_count"])
                self.assertIn(cycle_count, {3, 4, 5})
                self.assertEqual(
                    started["time_limits_s"]["run_stop"], 60.0 * cycle_count
                )
                self.assertFalse(runtime.consume_record_start_request())
                _feed_ready_window(
                    runtime,
                    side=started["initial_side"],
                    end_ns=RealTransitionRunPackageTest._step_ns(0),
                )
                send_transition_command(
                    host="127.0.0.1",
                    port=server.port,
                    command="initial-ready",
                    payload=READY_CONFIRMATIONS,
                )
                self.assertTrue(runtime.consume_record_start_request())
                runtime.attach_recording(episode_idx=0)
                n_rows = 2 + cycle_count * 7
                for step in range(n_rows):
                    runtime.update_recorded_step(
                        step_id=step,
                        step_ns=RealTransitionRunPackageTest._step_ns(step),
                    )
                    cycle_offset = step - 2
                    if cycle_offset >= 0:
                        within = cycle_offset % 7
                        cycle = cycle_offset // 7
                        if cycle < cycle_count and within == 0:
                            send_transition_command(
                                host="127.0.0.1",
                                port=server.port,
                                command="commit-goal",
                                payload={"display_ack": True},
                            )
                        elif cycle < cycle_count and within == 3:
                            send_transition_command(
                                host="127.0.0.1",
                                port=server.port,
                                command="dump-end",
                            )
                        elif cycle < cycle_count and within == 6:
                            target = runtime.status()["next_target_side"]
                            _feed_ready_window(
                                runtime,
                                side=target,
                                end_ns=RealTransitionRunPackageTest._step_ns(step),
                            )
                            send_transition_command(
                                host="127.0.0.1",
                                port=server.port,
                                command="target-ready",
                                payload=READY_CONFIRMATIONS,
                            )
                stop = runtime.consume_stop_request()
                self.assertIsNotNone(stop)
                assert stop is not None
                self.assertTrue(stop.success)
                RealTransitionRunPackageTest._write_raw(
                    runtime.active_raw_path,
                    n_rows=n_rows,
                )
                manifest = runtime.seal_saved_run(
                    raw_path=runtime.active_raw_path,
                    stop_reason=stop.stop_reason,
                )
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(
                    manifest["field_context"]["planned_sequence_id"],
                    manifest["sequence_id"],
                )
                run_dir = Path(started["raw_path"]).parent
                self.assertEqual(verify_run_package(run_dir)["status"], "PASS")
            finally:
                server.close()

    def test_realized_target_mismatch_aborts_without_consuming_next_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = prepare_session_directory(
                output_root=root,
                session_id="mismatch01",
                seed=19,
                created_at_utc="2026-08-13T00:00:00Z",
            )
            session_dir = Path(prepared["session_dir"])
            ready_contract = Path(prepared["ready_contract"])
            runtime = TransitionTaskRuntime(
                session_dir=session_dir,
                sequence_manifest_path=prepared["sequence_manifest"],
                split_manifest_path=prepared["split_manifest"],
                ready_contract_path=ready_contract,
                resolved_record_config_yaml="task: real_transition\n",
                git_commit="f" * 40,
            )
            runtime.update_receiver_state(mode="armed", health_ok=True)
            runtime.handle_command(
                "start-run",
                {
                    "field_context": {
                        "workface_reset_id": "wf01",
                        "workface_action": "restore",
                    }
                },
            )
            _feed_ready_window(
                runtime,
                side=runtime.status()["initial_side"],
                end_ns=RealTransitionRunPackageTest._step_ns(0),
            )
            runtime.handle_command("initial-ready", READY_CONFIRMATIONS)
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
            runtime.handle_command("dump-end")
            expected = runtime.status()["next_target_side"]
            realized = "B" if expected == "A" else "A"
            _feed_ready_window(
                runtime,
                side=realized,
                end_ns=RealTransitionRunPackageTest._step_ns(1),
            )
            result = runtime.handle_command(
                "target-ready",
                READY_CONFIRMATIONS,
            )
            self.assertEqual(result["phase"], "aborted")
            self.assertEqual(result["completed_cycles"], 0)
            stop = runtime.consume_stop_request()
            self.assertIsNotNone(stop)
            assert stop is not None
            self.assertFalse(stop.success)
            self.assertEqual(stop.stop_reason, "realized_target_mismatch")


class HomeSideContractTest(unittest.TestCase):
    def test_rule_contract_classifies_home_transition_sides_and_safe_limits(
        self,
    ) -> None:
        contract = build_rule_ready_contract()
        validate_rule_ready_contract(contract)
        self.assertEqual(contract["schema"], "real_transition_ready_rule_contract_v3")
        self.assertEqual(
            contract["ready_requirements"]["operator_authorization_policy"],
            "single_session_arm",
        )
        self.assertEqual(
            contract["ready_requirements"]["bucket_clear_policy"],
            "posthoc_visual_qc",
        )
        self.assertEqual(classify_ready_swing_qpos(contract, 0.000690), "home")
        self.assertEqual(classify_ready_swing_qpos(contract, -0.049), "home")
        self.assertEqual(classify_ready_swing_qpos(contract, 0.061), "transition")
        self.assertEqual(classify_ready_swing_qpos(contract, -0.080), "A")
        self.assertEqual(classify_ready_swing_qpos(contract, 0.081), "B")
        self.assertEqual(
            classify_ready_swing_qpos(contract, -0.3893),
            "outside_safe_range",
        )
        self.assertEqual(
            classify_ready_swing_qpos(contract, 0.4190),
            "outside_safe_range",
        )

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


class HomeSideCalibrationCaptureTest(unittest.TestCase):
    def test_gateway_jpeg_payload_is_decoded_to_rgb(self) -> None:
        import cv2

        rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        rgb[..., 0] = 220
        rgb[..., 1] = 80
        rgb[..., 2] = 20
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        self.assertTrue(ok)

        decoded = _camera_image_from_payload(
            {"encoding": "jpeg", "data": encoded}, camera="video4"
        )

        self.assertEqual(decoded.shape, rgb.shape)
        self.assertGreater(float(decoded[..., 0].mean()), 200.0)
        self.assertLess(float(decoded[..., 2].mean()), 40.0)

    def test_initialise_and_capture_read_only_four_camera_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.yaml"
            source.write_text(
                "teleop:\n"
                "  recording:\n"
                "    go_home:\n"
                "      home_pose_rad: [0.0, 0.1, -0.2, 0.3]\n",
                encoding="utf-8",
            )
            calibration = root / "home_calibration_samples.json"
            initial = initialise_home_calibration(
                output_path=calibration,
                context_version="ctx01",
                resolved_by="field_engineer",
                physical_left_qpos_sign=1,
                source_config=source,
                source_value_path="teleop.recording.go_home.home_pose_rad",
                created_at_utc="2026-08-17T00:00:00Z",
            )
            self.assertEqual(initial["accepted_window_counts"]["home"], 0)
            self.assertTrue(Path(initial["home_source_snapshot"]).is_file())

            result = capture_home_calibration_window(
                calibration_path=calibration,
                side="home",
                reference_id="home_00",
                confirm_visual=True,
                confirm_no_software_action_source=True,
                receiver_port=_unused_tcp_port(),
                client_factory=lambda: InProcessMockBridgeClient(
                    camera_names=("video4", "video5", "video6", "video7"),
                ),
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["accepted_window_counts"]["home"], 1)
            payload = json.loads(calibration.read_text(encoding="utf-8"))
            self.assertEqual(payload["samples"][0]["reference_id"], "home_00")
            self.assertEqual(payload["samples"][0]["commanded_action_abs_max"], 0.0)
            for camera in ("video4", "video5", "video6", "video7"):
                self.assertTrue(
                    (root / "calibration_visuals" / "home_00" / f"{camera}.jpg").is_file()
                )

    def test_capture_refuses_when_receiver_port_is_listening(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.yaml"
            source.write_text("home_pose_rad: [0, 0, 0, 0]\n", encoding="utf-8")
            calibration = root / "calibration.json"
            initialise_home_calibration(
                output_path=calibration,
                context_version="ctx01",
                resolved_by="field_engineer",
                physical_left_qpos_sign=1,
                source_config=source,
                source_value_path="home_pose_rad",
            )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                receiver_port = int(listener.getsockname()[1])
                with self.assertRaisesRegex(
                    TransitionContractError,
                    "receiver port .* is listening",
                ):
                    capture_home_calibration_window(
                        calibration_path=calibration,
                        side="home",
                        reference_id="home_00",
                        confirm_visual=True,
                        confirm_no_software_action_source=True,
                        receiver_port=receiver_port,
                        client_factory=lambda: InProcessMockBridgeClient(
                            camera_names=(
                                "video4",
                                "video5",
                                "video6",
                                "video7",
                            ),
                        ),
                    )


def _write_valid_home_contract(path: Path) -> None:
    contract = build_home_side_contract(
        _valid_home_calibration(path.parent),
        source_base_dir=path.parent,
    )
    path.write_text(
        json.dumps(contract, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
