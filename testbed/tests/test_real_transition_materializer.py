from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from testbed.data.hdf5_io import write_episode
from testbed.tasks.real_transition import (
    REQUIRED_GOAL_ACK_SOURCES,
    TransitionRunPackage,
    TransitionRunSpec,
    _AsyncEventJournal,
    sha256_file,
)
from testbed.tasks.real_transition_materializer import (
    ACTION_LABEL_OFFSET_S,
    ANNOTATION_SCHEMA,
    CONDITION_SCHEMA,
    _read_indexed,
    materialize_transition_run,
)


def _ready_evidence(side: str) -> dict[str, object]:
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


def _automatic_ready_evidence(side: str, *, excursion: bool) -> dict[str, object]:
    return {
        **_ready_evidence(side),
        "boundary_mode": "automatic_session_arm",
        "session_armed": True,
        "operator_confirmation_source": "session_arm",
        "bucket_clear_confirmed": False,
        "bucket_clear_policy": "posthoc_visual_qc",
        "cycle_excursion_observed": excursion,
    }


def _build_sealed_run(
    root: Path,
    *,
    invalid_camera_row: int | None = None,
    automatic_boundaries: bool = False,
) -> Path:
    spec = TransitionRunSpec(
        session_id="session01",
        block_id="b01",
        run_id="b01_r01",
        split="train",
        sequence_id="sequence01",
        collection_rank=0,
        run_rank_in_block=0,
        sequence=("A", "B", "B", "A"),
    )
    run_dir = root / "block_b01" / "run_b01_r01"
    package = TransitionRunPackage(run_dir=run_dir, run_spec=spec)
    n_rows = 106
    step_ids = np.arange(n_rows, dtype=np.int64)
    step_ns = 1_000_000_000 + step_ids * 20_000_000
    qpos = np.zeros((n_rows, 4), dtype=np.float32)
    side_qpos = {"A": -0.20, "B": 0.20}
    qpos[:, 0] = side_qpos[spec.initial_side]
    actions = np.zeros((n_rows, 4), dtype=np.float32)
    cycle_bounds = ((0, 20, 35), (35, 55, 70), (70, 90, 105))
    for index, (goal, dump, target) in enumerate(cycle_bounds):
        current_qpos = side_qpos[spec.sequence[index]]
        target_qpos = side_qpos[spec.targets[index]]
        outbound_end = goal + 12
        qpos[goal : goal + 5, 0] = current_qpos
        qpos[goal + 5 : outbound_end + 1, 0] = np.linspace(
            current_qpos, 1.60, outbound_end - (goal + 5) + 1
        )
        qpos[outbound_end : dump + 1, 0] = 1.60
        qpos[dump : target - 5 + 1, 0] = np.linspace(
            1.60, target_qpos, (target - 5) - dump + 1
        )
        qpos[target - 5 : target + 1, 0] = target_qpos
        actions[goal + 6 : target - 1, 0] = 0.2
    qvel = np.zeros((n_rows, 4), dtype=np.float32)
    qvel[1:, 0] = np.diff(qpos[:, 0]) / 0.02
    frames = {
        camera: [
            np.asarray([0xFF, 0xD8, index % 255, 0xFF, 0xD9], dtype=np.uint8)
            for index in range(n_rows)
        ]
        for camera in ("video4", "video5", "video6", "video7")
    }
    diagnostics: dict[str, np.ndarray] = {
        "raw_action": actions.copy(),
        "commanded_action": actions.copy(),
        "action_sample_timestamp_ns": step_ns.copy(),
    }
    for camera in frames:
        valid = np.ones(n_rows, dtype=np.int64)
        if invalid_camera_row is not None and camera == "video7":
            valid[int(invalid_camera_row)] = 0
        diagnostics[f"image_group_valid_{camera}"] = valid
        diagnostics[f"image_group_skew_ms_{camera}"] = np.full(
            n_rows, 0.04, dtype=np.float32
        )
        diagnostics[f"image_group_id_{camera}"] = np.arange(
            1, n_rows + 1, dtype=np.int64
        )
    write_episode(
        package.raw_path,
        qpos=qpos,
        qvel=qvel,
        actions=actions,
        encoded_images=frames,
        step_ids=step_ids,
        step_ns=step_ns,
        action_src_types=["teleop"] * n_rows,
        action_src_ids=["remote:joystick"] * n_rows,
        diagnostics=diagnostics,
        metadata={"is_real": True, "record_hz": 50.0},
    )

    package.start_run(step_id=0, step_ns=int(step_ns[0]))
    package.mark_initial_ready(
        step_id=0,
        step_ns=int(step_ns[0]),
        ready_evidence=(
            _automatic_ready_evidence("A", excursion=False)
            if automatic_boundaries
            else _ready_evidence("A")
        ),
        event_source="automatic" if automatic_boundaries else "operator",
    )
    for index, (goal, dump, target) in enumerate(cycle_bounds):
        package.commit_next_goal(
            step_id=goal,
            step_ns=int(step_ns[goal]),
            commit_ack_sources=REQUIRED_GOAL_ACK_SOURCES,
            notes="automatic frozen-sequence goal commit",
        )
        if automatic_boundaries:
            anchor = float(qpos[goal, 0])
            current = float(qpos[goal + 8, 0])
            package.record_cycle_excursion(
                step_id=goal + 8,
                step_ns=int(step_ns[goal + 8]),
                anchor_swing_qpos_rad=anchor,
                swing_qpos_rad=current,
                swing_delta_rad=current - anchor,
                threshold_rad=0.08,
                consecutive_samples=3,
            )
        else:
            package.mark_dump_end(
                step_id=dump,
                step_ns=int(step_ns[dump]),
                event_source="operator",
            )
        package.mark_target_ready(
            step_id=target,
            step_ns=int(step_ns[target]),
            realized_target_side=spec.targets[index],
            ready_evidence=(
                _automatic_ready_evidence(spec.targets[index], excursion=True)
                if automatic_boundaries
                else _ready_evidence(spec.targets[index])
            ),
            event_source="automatic" if automatic_boundaries else "operator",
        )
    package.complete_run(step_id=105, step_ns=int(step_ns[105]))

    owner = root / "owner.json"
    owner.write_text("{}\n", encoding="utf-8")
    resolved = root / "resolved.yaml"
    resolved.write_text("task: real_transition\n", encoding="utf-8")
    package.seal(
        git_commit="a" * 40,
        resolved_config_sha256=sha256_file(resolved),
        owner_artifacts={"owner": owner, "resolved": resolved},
        field_context={
            "workface_reset_id": "wf_001",
            "workface_action": "fresh_strip",
        },
    )
    return run_dir


def test_materializer_builds_conditioned_ready_to_ready_cycles(tmp_path: Path) -> None:
    run_dir = _build_sealed_run(tmp_path / "raw")
    output = tmp_path / "cycles"

    result = materialize_transition_run(run_dir=run_dir, output_dir=output)

    assert result["status"] == "PASS"
    assert result["cycle_count"] == 3
    assert result["clean_cycle_count"] == 3
    assert result["train_ready_episode_ids"] == [0, 1, 2]
    ready = json.loads((output / "train_ready_manifest.json").read_text())
    assert ready["dataset_dir"] == str(output / "episodes")
    assert ready["train_ready_episode_ids"] == [0, 1, 2]

    with h5py.File(output / "episodes" / "episode_0.hdf5", "r") as episode:
        condition = np.asarray(episode[f"conditions/{CONDITION_SCHEMA}"][()])
        assert condition.ndim == 2 and condition.shape[1] == 2
        assert np.all(condition[:, 0] == 1.0)
        assert np.all(condition[:, 1] == 1.0)
        valid = np.asarray(episode["conditions/valid_mask"][()])
        assert valid.shape == (condition.shape[0], 20)
        assert np.count_nonzero(valid[0]) == min(20, condition.shape[0])
        assert np.count_nonzero(valid[-1]) == 1
        source_rows = np.asarray(episode["provenance/source_row_index"][()])
        action_rows = np.asarray(
            episode["provenance/source_action_row_index"][()]
        )
        assert source_rows[0] == 1
        assert action_rows[0] == 0
        assert episode["metadata"].attrs["action_label_offset_s"] == (
            ACTION_LABEL_OFFSET_S
        )
        assert set(episode["observations/encoded_images"].keys()) == {
            "video4",
            "video5",
            "video6",
            "video7",
        }

    annotations = [
        json.loads(line)
        for line in (output / "annotations" / "cycle_annotations_v2.jsonl")
        .read_text()
        .splitlines()
    ]
    assert annotations[0]["schema"] == ANNOTATION_SCHEMA
    assert annotations[0]["boundaries"]["goal_commit_confirmed"][
        "event_source"
    ] == "sequencer"
    assert annotations[0]["boundaries"]["target_ready_confirmed"][
        "event_source"
    ] == "operator"
    assert (output / "SHA256SUMS.txt").is_file()


def test_materializer_accepts_session_arm_automatic_cycles_without_dump_mark(
    tmp_path: Path,
) -> None:
    run_dir = _build_sealed_run(
        tmp_path / "raw",
        automatic_boundaries=True,
    )
    output = tmp_path / "cycles"

    result = materialize_transition_run(run_dir=run_dir, output_dir=output)

    assert result["clean_cycle_count"] == 3
    annotations = [
        json.loads(line)
        for line in (output / "annotations" / "cycle_annotations_v2.jsonl")
        .read_text()
        .splitlines()
    ]
    assert annotations[0]["boundaries"]["dump_end_confirmed"] is None
    assert annotations[0]["boundaries"]["cycle_excursion_observed"][
        "event_source"
    ] == "automatic"
    assert annotations[0]["boundaries"]["target_ready_confirmed"][
        "event_source"
    ] == "automatic"


def test_materializer_excludes_a_cycle_with_invalid_camera_group(tmp_path: Path) -> None:
    run_dir = _build_sealed_run(tmp_path / "raw", invalid_camera_row=6)
    output = tmp_path / "cycles"

    result = materialize_transition_run(run_dir=run_dir, output_dir=output)

    assert result["excluded_cycle_count"] == 1
    assert result["train_ready_episode_ids"] == [1, 2]
    rows = [
        json.loads(line)
        for line in (output / "cycle_manifest.jsonl").read_text().splitlines()
    ]
    assert rows[0]["training_tier"] == "excluded"
    assert "camera_group_invalid" in rows[0]["qc_reasons"]


def test_event_journal_only_fsyncs_when_closed_durably(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        "testbed.tasks.real_transition.os.fsync", lambda fd: calls.append(int(fd))
    )
    journal = _AsyncEventJournal(tmp_path / "events.jsonl")
    journal.append('{"event":1}\n')
    journal._queue.join()  # noqa: SLF001 - verifies the critical-path contract.
    assert calls == []

    journal.close_durable()
    assert len(calls) == 1


def test_materializer_hdf5_reader_preserves_repeated_action_indices(
    tmp_path: Path,
) -> None:
    path = tmp_path / "indices.hdf5"
    with h5py.File(path, "w") as output:
        output.create_dataset("value", data=np.asarray([10, 20, 30], dtype=np.int64))
    with h5py.File(path, "r") as source:
        values = _read_indexed(
            source["value"],
            np.asarray([0, 0, 2, 2, 2], dtype=np.int64),
        )
    np.testing.assert_array_equal(values, [10, 10, 30, 30, 30])


def test_materializer_hdf5_reader_supports_zero_width_diagnostics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "zero_width.hdf5"
    with h5py.File(path, "w") as output:
        output.create_dataset("status11", shape=(4, 0), dtype=np.int32)
    with h5py.File(path, "r") as source:
        values = _read_indexed(
            source["status11"],
            np.asarray([3, 1, 1], dtype=np.int64),
        )
    assert values.shape == (3, 0)
    assert values.dtype == np.dtype(np.int32)
