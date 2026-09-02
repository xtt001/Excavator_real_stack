"""Deterministically materialize sealed transition runs into 20 Hz cycles.

The raw run package remains immutable.  A session-level ARM authorizes
automatic ready boundaries, while the frozen sequencer ``goal_commit`` owns the
cycle condition.  Manual MARK runs remain readable.  Every derived row retains
its source row/step provenance.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.tasks.home_side_contract import READY_RULE_DEFAULTS
from testbed.tasks.real_transition import (
    CONDITION_SCHEMA,
    SIDE_CODES,
    TransitionContractError,
    sha256_file,
    verify_run_package,
)
from testbed.tasks.real_transition_excursion import (
    EXCURSION_OBSERVED_KEY,
    build_excursion_contract,
    derive_excursion_observed,
    excursion_chunk_valid_mask,
)
from testbed.tasks.real_transition_phase import (
    CYCLE_PHASE_KEY,
    build_cycle_phase_contract,
    derive_cycle_phase,
    phase_chunk_valid_mask,
)
from testbed.tasks.real_transition_return_commit import (
    RETURN_COMMIT_ACTION_INTENT_THRESHOLD,
    RETURN_COMMIT_KEY,
    ReturnCommitDerivation,
    build_return_commit_contract,
    derive_return_commit,
)

ANNOTATION_SCHEMA = "real_transition_cycle_annotation_v2"
ANNOTATION_VERSION = "session_arm_auto_materializer_v5_return_commit"
CYCLE_MANIFEST_SCHEMA = "real_transition_cycle_manifest_v1"
MATERIALIZER_SCHEMA = "real_transition_cycle_materializer_v1"
EXPECTED_CAMERAS = ("video4", "video5", "video6", "video7")
TARGET_HZ = 20.0
ACTION_LABEL_OFFSET_S = -0.02
ACTION_INTENT_THRESHOLD = 0.05
GOAL_LEAD_CLEAN_MS = 100.0
GOAL_LEAD_EXCLUDE_MS = 50.0
LOCAL_SOURCE_GAP_MAX_MS = 100.0
DERIVED_GAP_MAX_MS = 120.0
STRUCTURAL_GAP_MAX_MS = 250.0
CAMERA_GROUP_SKEW_MAX_MS = 5.0
ACT_CHUNK_STEPS = 20
HOME_SWING_RAD = float(READY_RULE_DEFAULTS["home_swing_qpos_rad"])
CLEAN_READY_MIN_DELTA_RAD = float(
    READY_RULE_DEFAULTS["clean_ready_min_abs_delta_rad"]
)
LEFT_SIGN = int(READY_RULE_DEFAULTS["physical_left_qpos_sign"])
EXCURSION_MIN_DELTA_RAD = float(
    READY_RULE_DEFAULTS["cycle_excursion_min_abs_delta_rad"]
)
EXCURSION_MIN_CONSECUTIVE_SAMPLES = int(
    READY_RULE_DEFAULTS["cycle_excursion_min_consecutive_samples"]
)


def materialize_transition_run(
    *, run_dir: Path | str, output_dir: Path | str
) -> dict[str, Any]:
    """Materialize all completed cycles from one sealed raw run."""

    return materialize_transition_dataset(
        run_dirs=[Path(run_dir)], output_dir=output_dir
    )


def materialize_transition_session(
    *, session_dir: Path | str, output_dir: Path | str
) -> dict[str, Any]:
    """Materialize every sealed run in a session in frozen collection order."""

    session = Path(session_dir).resolve()
    candidates = [path.parent for path in session.glob("block_*/run_*/run_manifest.json")]
    if not candidates:
        raise TransitionContractError(
            f"transition session has no sealed run packages: {session}"
        )
    return materialize_transition_dataset(
        run_dirs=candidates,
        output_dir=output_dir,
        source_session_dir=session,
    )


def materialize_transition_dataset(
    *,
    run_dirs: Iterable[Path | str],
    output_dir: Path | str,
    source_session_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build one immutable cycle dataset from one or more sealed runs."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise TransitionContractError(
            f"refusing to overwrite materialized output: {destination}"
        )
    source_runs = _sorted_verified_runs(run_dirs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.incomplete-", dir=destination.parent
        )
    )
    try:
        result = _materialize_into(
            source_runs=source_runs,
            output_dir=temporary,
            final_output_dir=destination,
            source_session_dir=(
                None
                if source_session_dir is None
                else Path(source_session_dir).resolve()
            ),
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    result["output_dir"] = str(destination)
    result["episodes_dir"] = str(destination / "episodes")
    result["train_ready_manifest"] = str(destination / "train_ready_manifest.json")
    return result


def _sorted_verified_runs(run_dirs: Iterable[Path | str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for value in run_dirs:
        run_dir = Path(value).resolve()
        if run_dir in seen:
            continue
        seen.add(run_dir)
        verification = verify_run_package(run_dir)
        manifest = _read_json_object(run_dir / "run_manifest.json")
        rows.append(
            {
                "run_dir": run_dir,
                "verification": verification,
                "manifest": manifest,
            }
        )
    if not rows:
        raise TransitionContractError("no transition run directories were provided")
    rows.sort(
        key=lambda row: (
            int(row["manifest"].get("collection_rank", 1_000_000)),
            int(row["manifest"].get("run_rank_in_block", 1_000_000)),
            str(row["manifest"].get("run_id", "")),
        )
    )
    return rows


def _materialize_into(
    *,
    source_runs: list[dict[str, Any]],
    output_dir: Path,
    final_output_dir: Path,
    source_session_dir: Path | None,
) -> dict[str, Any]:
    annotations_dir = output_dir / "annotations"
    episodes_dir = output_dir / "episodes"
    annotations_dir.mkdir(parents=True)
    episodes_dir.mkdir(parents=True)

    cycle_records: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for run_row in source_runs:
        records, run_annotations = _inspect_run_cycles(run_row)
        cycle_records.extend(records)
        annotations.extend(run_annotations)

    annotation_path = annotations_dir / "cycle_annotations_v2.jsonl"
    _write_jsonl(annotation_path, annotations)
    annotation_sha256 = sha256_file(annotation_path)
    phase_contract_path = output_dir / "cycle_phase_contract.json"
    _write_json(
        phase_contract_path,
        build_cycle_phase_contract(
            excursion_min_delta_rad=EXCURSION_MIN_DELTA_RAD,
            excursion_min_consecutive_samples=EXCURSION_MIN_CONSECUTIVE_SAMPLES,
        ),
    )
    phase_contract_sha256 = sha256_file(phase_contract_path)
    excursion_contract_path = output_dir / "excursion_observed_contract.json"
    _write_json(
        excursion_contract_path,
        build_excursion_contract(
            minimum_delta_rad=EXCURSION_MIN_DELTA_RAD,
            minimum_consecutive_samples=EXCURSION_MIN_CONSECUTIVE_SAMPLES,
        ),
    )
    excursion_contract_sha256 = sha256_file(excursion_contract_path)
    return_commit_contract_path = output_dir / "return_commit_contract.json"
    _write_json(
        return_commit_contract_path,
        build_return_commit_contract(
            action_intent_threshold=RETURN_COMMIT_ACTION_INTENT_THRESHOLD
        ),
    )
    return_commit_contract_sha256 = sha256_file(return_commit_contract_path)

    manifest_rows: list[dict[str, Any]] = []
    train_ready_ids: list[int] = []
    for episode_id, record in enumerate(cycle_records):
        record["episode_id"] = int(episode_id)
        episode_name = f"episode_{episode_id}.hdf5"
        episode_path = episodes_dir / episode_name
        return_commit = _write_cycle_episode(
            record=record,
            output_path=episode_path,
            annotation_sha256=annotation_sha256,
            phase_contract_sha256=phase_contract_sha256,
            excursion_contract_sha256=excursion_contract_sha256,
            return_commit_contract_sha256=return_commit_contract_sha256,
        )
        record["return_commit_evaluable"] = bool(return_commit.evaluable)
        record["return_commit_event_row"] = return_commit.event_row
        record["return_commit_reason"] = return_commit.reason
        episode_sha256 = sha256_file(episode_path)
        row = _cycle_manifest_row(
            record,
            episode_name=episode_name,
            episode_sha256=episode_sha256,
            annotation_sha256=annotation_sha256,
            phase_contract_sha256=phase_contract_sha256,
            excursion_contract_sha256=excursion_contract_sha256,
            return_commit_contract_sha256=return_commit_contract_sha256,
        )
        manifest_rows.append(row)
        if row["training_tier"] == "clean":
            train_ready_ids.append(int(episode_id))

    cycle_manifest_path = output_dir / "cycle_manifest.jsonl"
    _write_jsonl(cycle_manifest_path, manifest_rows)
    split_payload = {
        "schema": "real_transition_cycle_split_manifest_v1",
        "split_owner": "source_block_before_cycle_materialization",
        "episodes": [
            {
                "episode_id": int(row["episode_id"]),
                "cycle_id": row["cycle_id"],
                "split": row["split"],
                "source_block_id": row["source_block_id"],
                "source_run_id": row["source_run_id"],
            }
            for row in manifest_rows
        ],
    }
    _write_json(output_dir / "split_manifest.json", split_payload)

    excluded = [
        {
            "episode_id": int(row["episode_id"]),
            "cycle_id": row["cycle_id"],
            "status": row["training_tier"],
            "reasons": list(row["qc_reasons"]),
        }
        for row in manifest_rows
        if row["training_tier"] != "clean"
    ]
    train_ready_payload = {
        "schema": "real_transition_train_ready_manifest_v1",
        "schema_version": 1,
        "generated_at": _utc_now(),
        # Use the final path, not the temporary build path.
        "dataset_dir": str(final_output_dir / "episodes"),
        "condition_schema": CONDITION_SCHEMA,
        "cycle_phase_schema": CYCLE_PHASE_KEY,
        "cycle_phase_contract_sha256": phase_contract_sha256,
        "excursion_observed_schema": EXCURSION_OBSERVED_KEY,
        "excursion_observed_contract_sha256": excursion_contract_sha256,
        "return_commit_schema": RETURN_COMMIT_KEY,
        "return_commit_contract_sha256": return_commit_contract_sha256,
        "train_ready_episode_ids": train_ready_ids,
        "strict_pass_episode_ids": train_ready_ids,
        "warn_episode_ids": [
            int(row["episode_id"])
            for row in manifest_rows
            if row["training_tier"] == "review"
        ],
        "failed_episode_ids": [
            int(row["episode_id"])
            for row in manifest_rows
            if row["training_tier"] == "excluded"
        ],
        "excluded_episode_ids": [int(item["episode_id"]) for item in excluded],
        "excluded": excluded,
    }
    _write_json(output_dir / "train_ready_manifest.json", train_ready_payload)

    config_payload = {
        "schema": MATERIALIZER_SCHEMA,
        "annotation_schema": ANNOTATION_SCHEMA,
        "annotation_version": ANNOTATION_VERSION,
        "condition_schema": CONDITION_SCHEMA,
        "source_session_dir": (
            None if source_session_dir is None else str(source_session_dir)
        ),
        "source_run_dirs": [str(row["run_dir"]) for row in source_runs],
        "expected_cameras": list(EXPECTED_CAMERAS),
        "target_hz": TARGET_HZ,
        "selection_rule": "first_source_row_not_earlier_than_50ms_grid",
        "action_label_offset_s": ACTION_LABEL_OFFSET_S,
        "action_index_rule": "last_action_sample_not_later_than_observation_plus_offset",
        "action_intent_threshold": ACTION_INTENT_THRESHOLD,
        "goal_lead_clean_ms": GOAL_LEAD_CLEAN_MS,
        "goal_lead_exclude_ms": GOAL_LEAD_EXCLUDE_MS,
        "camera_group_skew_max_ms": CAMERA_GROUP_SKEW_MAX_MS,
        "automatic_cycle_excursion": {
            "detector": "positive_swing_displacement_from_goal_anchor",
            "min_abs_delta_rad": EXCURSION_MIN_DELTA_RAD,
            "min_consecutive_samples": EXCURSION_MIN_CONSECUTIVE_SAMPLES,
        },
        "cycle_phase_contract": build_cycle_phase_contract(
            excursion_min_delta_rad=EXCURSION_MIN_DELTA_RAD,
            excursion_min_consecutive_samples=EXCURSION_MIN_CONSECUTIVE_SAMPLES,
        ),
        "excursion_observed_contract": build_excursion_contract(
            minimum_delta_rad=EXCURSION_MIN_DELTA_RAD,
            minimum_consecutive_samples=EXCURSION_MIN_CONSECUTIVE_SAMPLES,
        ),
        "return_commit_contract": build_return_commit_contract(
            action_intent_threshold=RETURN_COMMIT_ACTION_INTENT_THRESHOLD
        ),
        "dump_boundary_policy": "optional_manual_event_else_return_proxy_only",
        "local_source_gap_max_ms": LOCAL_SOURCE_GAP_MAX_MS,
        "derived_gap_max_ms": DERIVED_GAP_MAX_MS,
        "structural_gap_max_ms": STRUCTURAL_GAP_MAX_MS,
        "act_chunk_steps": ACT_CHUNK_STEPS,
    }
    _write_json(output_dir / "resolved_materializer_config.json", config_payload)
    _write_checksums(output_dir)
    return {
        "status": "PASS",
        "source_run_count": len(source_runs),
        "cycle_count": len(manifest_rows),
        "clean_cycle_count": len(train_ready_ids),
        "review_cycle_count": sum(
            row["training_tier"] == "review" for row in manifest_rows
        ),
        "excluded_cycle_count": sum(
            row["training_tier"] == "excluded" for row in manifest_rows
        ),
        "train_ready_episode_ids": train_ready_ids,
        "annotation_sha256": annotation_sha256,
        "cycle_phase_contract_sha256": phase_contract_sha256,
        "excursion_observed_contract_sha256": excursion_contract_sha256,
        "return_commit_contract_sha256": return_commit_contract_sha256,
        "return_commit_evaluable_count": sum(
            bool(row.get("return_commit_evaluable", False))
            for row in cycle_records
        ),
    }


def _inspect_run_cycles(
    run_row: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = Path(run_row["run_dir"])
    manifest = dict(run_row["manifest"])
    events = _read_jsonl(run_dir / "task_events.jsonl")
    events_by_cycle: dict[int, dict[str, dict[str, Any]]] = {}
    initial_ready = None
    passive_events: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("event_type", ""))
        if event_type == "initial_ready_mark":
            initial_ready = event
        if event_type in {"manual_intervention", "safety_stop", "run_abort"}:
            passive_events.append(event)
        index = event.get("cycle_index")
        if index is None:
            continue
        events_by_cycle.setdefault(int(index), {})[event_type] = event
    if initial_ready is None:
        raise TransitionContractError(f"run has no initial_ready_mark: {run_dir}")

    raw_path = run_dir / "raw.hdf5"
    records: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    planned_sequence = [str(value) for value in manifest["planned_sequence"]]
    with h5py.File(raw_path, "r") as source:
        step_ids = np.asarray(source["timestamps/step_id"][()], dtype=np.int64)
        step_ns = np.asarray(source["timestamps/step_ns"][()], dtype=np.int64)
        row_by_step = {int(step): index for index, step in enumerate(step_ids)}
        action = np.asarray(source["action"][()], dtype=np.float32)
        raw_action = (
            np.asarray(source["diagnostics/raw_action"][()], dtype=np.float32)
            if "diagnostics/raw_action" in source
            else action
        )
        commanded_action = (
            np.asarray(source["diagnostics/commanded_action"][()], dtype=np.float32)
            if "diagnostics/commanded_action" in source
            else action
        )
        intent_amplitude = np.maximum(
            np.max(np.abs(action), axis=1), np.max(np.abs(raw_action), axis=1)
        )
        effective_amplitude = np.max(np.abs(commanded_action), axis=1)
        for cycle_index in range(int(manifest["completed_cycles"])):
            cycle_events = events_by_cycle.get(cycle_index, {})
            required = ("goal_commit", "target_ready_mark")
            missing = [name for name in required if name not in cycle_events]
            if missing:
                raise TransitionContractError(
                    f"run {manifest['run_id']} cycle {cycle_index} missing events: "
                    + ", ".join(missing)
                )
            goal = cycle_events["goal_commit"]
            dump = cycle_events.get("dump_end_mark")
            excursion = cycle_events.get("cycle_excursion_observed")
            target_ready = cycle_events["target_ready_mark"]
            current_ready = (
                initial_ready
                if cycle_index == 0
                else events_by_cycle[cycle_index - 1]["target_ready_mark"]
            )
            goal_row = _event_row(goal, row_by_step, step_ns)
            target_row = _event_row(target_ready, row_by_step, step_ns)
            current_ready_row = _event_row(current_ready, row_by_step, step_ns)
            dump_row = (
                None if dump is None else _event_row(dump, row_by_step, step_ns)
            )
            excursion_event_row = (
                None
                if excursion is None
                else _event_row(excursion, row_by_step, step_ns)
            )
            if not (current_ready_row <= goal_row <= target_row):
                raise TransitionContractError(
                    f"run {manifest['run_id']} cycle {cycle_index} event rows are invalid"
                )
            for label, row in (
                ("dump", dump_row),
                ("excursion", excursion_event_row),
            ):
                if row is not None and not goal_row <= row <= target_row:
                    raise TransitionContractError(
                        f"run {manifest['run_id']} cycle {cycle_index} {label} row is invalid"
                    )
            obs_indices = _select_cycle_observation_indices(
                step_ns=step_ns,
                first_row=goal_row + 1,
                last_row=target_row,
            )
            if not len(obs_indices):
                raise TransitionContractError(
                    f"run {manifest['run_id']} cycle {cycle_index} has no rows after goal"
                )
            obs_indices, camera_filter_metrics = _filter_camera_observation_indices(
                source, obs_indices
            )
            action_indices = _select_action_indices(
                source=source,
                observation_indices=obs_indices,
                minimum_index=goal_row,
            )
            intent_row = _first_threshold_row(
                intent_amplitude,
                start=goal_row,
                stop=target_row,
                threshold=ACTION_INTENT_THRESHOLD,
            )
            effective_row = _first_threshold_row(
                effective_amplitude,
                start=goal_row,
                stop=target_row,
                threshold=ACTION_INTENT_THRESHOLD,
            )
            swing_qpos = np.asarray(
                source["observations/qpos"][goal_row : target_row + 1, 0],
                dtype=np.float64,
            )
            goal_anchor_qpos = float(swing_qpos[0])
            excursion_mask = (
                _shortest_angle_array(swing_qpos - goal_anchor_qpos)
                >= EXCURSION_MIN_DELTA_RAD
            )
            excursion_end = _first_consecutive_true_end(
                excursion_mask,
                EXCURSION_MIN_CONSECUTIVE_SAMPLES,
            )
            excursion_data_row = (
                None if excursion_end is None else goal_row + excursion_end
            )
            return_ready_proxy_row = _last_target_side_entry(
                swing_qpos,
                target_side=str(target_ready["realized_target_side"]),
            )
            if return_ready_proxy_row is not None:
                return_ready_proxy_row += goal_row
            goal_lead_ms = (
                None
                if intent_row is None
                else float((step_ns[intent_row] - int(goal["event_step_ns"])) * 1e-6)
            )
            target = str(goal["scripted_target_side"])
            current = planned_sequence[cycle_index]
            record = {
                "run_dir": run_dir,
                "raw_path": raw_path,
                "manifest": manifest,
                "cycle_index": cycle_index,
                "cycle_id": str(goal["cycle_id"]),
                "goal_epoch": int(goal["goal_epoch"]),
                "current_ready_side": current,
                "target_side": target,
                "target_side_code": int(SIDE_CODES[target]),
                "realized_target_side": str(target_ready["realized_target_side"]),
                "expected_return_swing_sign": goal.get(
                    "expected_return_swing_sign"
                ),
                "transition_type": f"{current}->{target}",
                "goal_event": goal,
                "dump_event": dump,
                "excursion_event": excursion,
                "target_ready_event": target_ready,
                "current_ready_event": current_ready,
                "source_indices": obs_indices,
                "source_action_indices": action_indices,
                "source_step_ids": step_ids[obs_indices],
                "source_step_ns": step_ns[obs_indices],
                "source_step_ids_all": step_ids,
                "source_step_ns_all": step_ns,
                "source_raw_sha256": str(manifest["artifacts"]["raw.hdf5"]),
                "source_events_sha256": str(
                    manifest["artifacts"]["task_events.jsonl"]
                ),
                "goal_row": goal_row,
                "dump_row": dump_row,
                "excursion_event_row": excursion_event_row,
                "excursion_data_row": excursion_data_row,
                "return_ready_proxy_row": return_ready_proxy_row,
                "target_ready_row": target_row,
                "first_intent_row": intent_row,
                "first_effective_action_row": effective_row,
                "goal_lead_ms": goal_lead_ms,
                "camera_filter_metrics": camera_filter_metrics,
                "passive_events": passive_events,
            }
            tier, reasons, metrics = _classify_cycle(source, record)
            record["training_tier"] = tier
            record["qc_reasons"] = reasons
            record["qc_metrics"] = metrics
            records.append(record)
            annotations.append(_annotation_record(record))
    return records, annotations


def _classify_cycle(
    source: h5py.File, record: Mapping[str, Any]
) -> tuple[str, list[str], dict[str, Any]]:
    obs_idx = np.asarray(record["source_indices"], dtype=np.int64)
    action_idx = np.asarray(record["source_action_indices"], dtype=np.int64)
    reasons: list[str] = []
    severity = "clean"

    def flag(reason: str, level: str) -> None:
        nonlocal severity
        reasons.append(reason)
        if level == "excluded" or (level == "review" and severity == "clean"):
            severity = level

    for path in ("observations/qpos", "observations/qvel", "action"):
        if path not in source:
            flag(f"missing_{path.replace('/', '_')}", "excluded")
            continue
        indices = action_idx if path == "action" else obs_idx
        values = _read_indexed(source[path], indices)
        if not np.all(np.isfinite(values)):
            flag(f"nonfinite_{path.replace('/', '_')}", "excluded")

    image_group_path = "observations/encoded_images"
    if image_group_path not in source:
        flag("missing_encoded_images", "excluded")
    else:
        group = source[image_group_path]
        for camera in EXPECTED_CAMERAS:
            if camera not in group:
                flag(f"missing_camera_{camera}", "excluded")
                continue
            dataset = group[camera]
            if dataset.shape[0] <= int(obs_idx[-1]):
                flag(f"short_camera_{camera}", "excluded")
                continue
            if any(np.asarray(dataset[int(index)]).size == 0 for index in obs_idx):
                flag(f"empty_camera_frame_{camera}", "excluded")

    camera_valid, camera_metrics = _camera_sync_qc(source, obs_idx)
    if camera_valid is None:
        flag("camera_group_evidence_missing", "review")
    elif not camera_valid:
        flag("camera_group_invalid", "excluded")

    selected_ns = np.asarray(record["source_step_ns"], dtype=np.int64)
    derived_gaps = np.diff(selected_ns).astype(np.float64) * 1e-6
    first = int(record["goal_row"])
    last = int(record["target_ready_row"])
    raw_ns = np.asarray(source["timestamps/step_ns"][first : last + 1], dtype=np.int64)
    raw_gaps = np.diff(raw_ns).astype(np.float64) * 1e-6
    raw_gap_max = float(np.max(raw_gaps)) if raw_gaps.size else 0.0
    derived_gap_max = float(np.max(derived_gaps)) if derived_gaps.size else 0.0
    if raw_gap_max > STRUCTURAL_GAP_MAX_MS or derived_gap_max > STRUCTURAL_GAP_MAX_MS:
        flag("structural_time_gap", "excluded")
    elif raw_gap_max > LOCAL_SOURCE_GAP_MAX_MS or derived_gap_max > DERIVED_GAP_MAX_MS:
        flag("local_marker_window_gap", "review")

    lead = record.get("goal_lead_ms")
    automatic_boundary = str(
        record["current_ready_event"].get("event_source", "")
    ) == "automatic"
    if not automatic_boundary:
        if lead is None:
            flag("no_cycle_action_intent", "review")
        elif float(lead) < GOAL_LEAD_EXCLUDE_MS:
            flag("late_goal_commit", "excluded")
        elif float(lead) < GOAL_LEAD_CLEAN_MS:
            flag("short_goal_lead", "review")

    if str(record["realized_target_side"]) != str(record["target_side"]):
        flag("realized_target_mismatch", "excluded")
    if record.get("excursion_data_row") is None:
        flag("cycle_ready_range_excursion_missing", "excluded")
    if str(record["target_ready_event"].get("event_source", "")) == "automatic":
        excursion_event = record.get("excursion_event")
        if not isinstance(excursion_event, Mapping):
            flag("automatic_excursion_event_missing", "excluded")
        elif dict(excursion_event.get("detector_evidence", {}) or {}).get(
            "detector"
        ) != "swing_displacement_from_goal_anchor":
            flag("automatic_excursion_evidence_invalid", "excluded")
    if str(record["manifest"].get("status", "")) != "complete":
        flag("source_run_not_complete", "excluded")
    if record["passive_events"]:
        flag("manual_or_safety_event", "excluded")

    if "action_source/type" not in source:
        flag("action_source_missing", "review")
    else:
        source_types = {
            _decode_text(value)
            for value in _read_indexed(source["action_source/type"], action_idx)
        }
        if source_types != {"teleop"}:
            flag("non_teleop_action_source", "excluded")

    metrics = {
        "goal_lead_ms": lead,
        "goal_lead_gate_applied": not automatic_boundary,
        "raw_source_gap_max_ms": raw_gap_max,
        "derived_gap_max_ms": derived_gap_max,
        **dict(record.get("camera_filter_metrics", {})),
        **camera_metrics,
    }
    return severity, reasons, metrics


def _camera_sync_qc(
    source: h5py.File, indices: np.ndarray
) -> tuple[bool | None, dict[str, Any]]:
    valid_paths = [f"diagnostics/image_group_valid_{cam}" for cam in EXPECTED_CAMERAS]
    skew_paths = [f"diagnostics/image_group_skew_ms_{cam}" for cam in EXPECTED_CAMERAS]
    group_paths = [f"diagnostics/image_group_id_{cam}" for cam in EXPECTED_CAMERAS]
    if any(path not in source for path in (*valid_paths, *skew_paths, *group_paths)):
        return None, {
            "camera_group_valid_fraction": None,
            "camera_group_skew_max_ms": None,
            "camera_distinct_group_count": None,
        }
    valid = np.stack(
        [np.asarray(source[path][indices], dtype=np.int64) for path in valid_paths],
        axis=1,
    )
    skews = np.stack(
        [np.asarray(source[path][indices], dtype=np.float64) for path in skew_paths],
        axis=1,
    )
    groups = np.stack(
        [np.asarray(source[path][indices], dtype=np.int64) for path in group_paths],
        axis=1,
    )
    same_group = np.all(groups == groups[:, :1], axis=1)
    positive_group = groups[:, 0] > 0
    row_valid = (
        np.all(valid == 1, axis=1)
        & same_group
        & positive_group
        & np.all(np.isfinite(skews), axis=1)
        & (np.max(skews, axis=1) <= CAMERA_GROUP_SKEW_MAX_MS)
    )
    distinct_count = int(np.unique(groups[:, 0]).size)
    all_valid = bool(
        np.all(row_valid) and distinct_count == int(indices.size)
    )
    return all_valid, {
        "camera_group_valid_fraction": float(np.mean(row_valid)),
        "camera_group_skew_max_ms": float(np.max(skews)),
        "camera_distinct_group_count": distinct_count,
    }


def _filter_camera_observation_indices(
    source: h5py.File, indices: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Drop locally unusable or repeated camera groups before materialization.

    A 20 Hz timestamp grid can occasionally select the same 30 Hz GMSL group
    twice near scheduler jitter.  Keeping both rows creates duplicate visual
    observations; rejecting the whole cycle discards hundreds of otherwise
    valid samples.  Invalid rows and later occurrences of a repeated group are
    removed here.  The existing derived-gap checks remain responsible for
    rejecting a cycle when filtering creates a material time discontinuity.
    """

    selected = np.asarray(indices, dtype=np.int64)
    valid_paths = [f"diagnostics/image_group_valid_{cam}" for cam in EXPECTED_CAMERAS]
    skew_paths = [f"diagnostics/image_group_skew_ms_{cam}" for cam in EXPECTED_CAMERAS]
    group_paths = [f"diagnostics/image_group_id_{cam}" for cam in EXPECTED_CAMERAS]
    if any(path not in source for path in (*valid_paths, *skew_paths, *group_paths)):
        return selected, {
            "camera_candidate_row_count": int(selected.size),
            "camera_dropped_invalid_row_count": 0,
            "camera_dropped_repeated_group_count": 0,
        }

    valid = np.stack(
        [np.asarray(source[path][selected], dtype=np.int64) for path in valid_paths],
        axis=1,
    )
    skews = np.stack(
        [np.asarray(source[path][selected], dtype=np.float64) for path in skew_paths],
        axis=1,
    )
    groups = np.stack(
        [np.asarray(source[path][selected], dtype=np.int64) for path in group_paths],
        axis=1,
    )
    same_group = np.all(groups == groups[:, :1], axis=1)
    row_valid = (
        np.all(valid == 1, axis=1)
        & same_group
        & (groups[:, 0] > 0)
        & np.all(np.isfinite(skews), axis=1)
        & (np.max(skews, axis=1) <= CAMERA_GROUP_SKEW_MAX_MS)
    )
    keep = np.zeros(selected.size, dtype=bool)
    seen_groups: set[int] = set()
    repeated_count = 0
    for row, is_valid in enumerate(row_valid):
        if not bool(is_valid):
            continue
        group_id = int(groups[row, 0])
        if group_id in seen_groups:
            repeated_count += 1
            continue
        seen_groups.add(group_id)
        keep[row] = True

    filtered = selected[keep]
    # Preserve an auditable excluded episode instead of failing the entire
    # session build when a cycle contains no usable camera observation.
    if not filtered.size:
        filtered = selected
    return filtered, {
        "camera_candidate_row_count": int(selected.size),
        "camera_dropped_invalid_row_count": int(np.count_nonzero(~row_valid)),
        "camera_dropped_repeated_group_count": int(repeated_count),
    }


def _annotation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    automatic = record["target_ready_event"].get("event_source") == "automatic"
    return {
        "schema": ANNOTATION_SCHEMA,
        "annotation_version": ANNOTATION_VERSION,
        "annotation_source": (
            "session_ARM_auto_boundary_plus_frozen_sequencer"
            if automatic
            else "operator_MARK_plus_frozen_sequencer"
        ),
        "source_raw_sha256": record["source_raw_sha256"],
        "source_events_sha256": record["source_events_sha256"],
        "source_session_id": record["manifest"]["session_id"],
        "source_block_id": record["manifest"]["block_id"],
        "source_run_id": record["manifest"]["run_id"],
        "cycle_id": record["cycle_id"],
        "cycle_index": int(record["cycle_index"]),
        "boundaries": {
            "initial_ready_confirmed": _boundary(record["current_ready_event"]),
            "goal_commit_confirmed": _boundary(record["goal_event"]),
            "first_cycle_intent_confirmed": _row_boundary(record, "first_intent_row"),
            "first_effective_action_confirmed": _row_boundary(
                record, "first_effective_action_row"
            ),
            "dump_end_confirmed": (
                None
                if record.get("dump_event") is None
                else _boundary(record["dump_event"])
            ),
            "cycle_excursion_observed": (
                _boundary(record["excursion_event"])
                if record.get("excursion_event") is not None
                else _row_boundary(record, "excursion_data_row")
            ),
            "return_to_ready_entry_proxy": _row_boundary(
                record, "return_ready_proxy_row"
            ),
            "target_ready_confirmed": _boundary(record["target_ready_event"]),
        },
        "cycle_validity_confirmed": {
            "training_tier": record["training_tier"],
            "reasons": list(record["qc_reasons"]),
            "metrics": dict(record["qc_metrics"]),
        },
        "evidence": {
            "goal_owner": "frozen_sequence_goal_commit",
            "ready_owner": (
                "automatic_session_ARM_detector"
                if automatic
                else "operator_state_aware_MARK"
            ),
            "dump_owner": (
                "operator_state_aware_MARK"
                if record.get("dump_event") is not None
                else "not_required_return_proxy_only"
            ),
            "camera_owner": "recorded_GMSL_group_metadata",
        },
    }


def _write_cycle_episode(
    *,
    record: Mapping[str, Any],
    output_path: Path,
    annotation_sha256: str,
    phase_contract_sha256: str,
    excursion_contract_sha256: str,
    return_commit_contract_sha256: str,
) -> ReturnCommitDerivation:
    obs_idx = np.asarray(record["source_indices"], dtype=np.int64)
    action_idx = np.asarray(record["source_action_indices"], dtype=np.int64)
    n_rows = int(obs_idx.size)
    with h5py.File(Path(record["raw_path"]), "r") as source, h5py.File(
        output_path, "x"
    ) as output:
        output.attrs["is_real"] = bool(source.attrs.get("is_real", True))
        metadata = output.create_group("metadata")
        if "metadata" in source:
            for key, value in source["metadata"].attrs.items():
                try:
                    metadata.attrs[key] = value
                except (TypeError, ValueError):
                    metadata.attrs[key] = str(value)
        metadata.attrs["schema"] = "real_transition_cycle_hdf5_v1"
        metadata.attrs["condition_schema"] = CONDITION_SCHEMA
        metadata.attrs["record_hz"] = TARGET_HZ
        metadata.attrs["control_hz"] = TARGET_HZ
        metadata.attrs["dt"] = 1.0 / TARGET_HZ
        metadata.attrs["n_steps"] = n_rows
        metadata.attrs["source_dataset_path"] = str(record["raw_path"])
        metadata.attrs["action_label_offset_s"] = ACTION_LABEL_OFFSET_S
        metadata.attrs["action_prealigned"] = True
        metadata.attrs["sampling_hz"] = TARGET_HZ
        metadata.attrs["annotation_version"] = ANNOTATION_VERSION
        metadata.attrs["annotation_sha256"] = annotation_sha256
        metadata.attrs["cycle_phase_schema"] = CYCLE_PHASE_KEY
        metadata.attrs["cycle_phase_contract_sha256"] = phase_contract_sha256
        metadata.attrs["excursion_observed_schema"] = EXCURSION_OBSERVED_KEY
        metadata.attrs["excursion_observed_contract_sha256"] = (
            excursion_contract_sha256
        )
        metadata.attrs["return_commit_schema"] = RETURN_COMMIT_KEY
        metadata.attrs["return_commit_contract_sha256"] = (
            return_commit_contract_sha256
        )

        _copy_indexed(source, output, "observations/qpos", obs_idx)
        _copy_indexed(source, output, "observations/qvel", obs_idx)
        _copy_encoded_images(source, output, obs_idx)
        _copy_indexed(source, output, "action", action_idx)
        _copy_indexed(source, output, "timestamps/step_id", obs_idx)
        _copy_indexed(source, output, "timestamps/step_ns", obs_idx)
        for path in ("action_source/type", "action_source/id"):
            if path in source:
                _copy_indexed(source, output, path, action_idx)
        if "diagnostics" in source:
            for name, dataset in source["diagnostics"].items():
                if dataset.shape and dataset.shape[0] == source["action"].shape[0]:
                    _copy_indexed(source, output, f"diagnostics/{name}", obs_idx)

        diagnostics = output.require_group("diagnostics")
        _replace_dataset(diagnostics, "source_observation_index", obs_idx)
        _replace_dataset(diagnostics, "source_action_index", action_idx)
        gaps = np.zeros(n_rows, dtype=np.float32)
        if n_rows > 1:
            gaps[1:] = np.diff(np.asarray(record["source_step_ns"])) * 1e-6
        _replace_dataset(diagnostics, "source_time_gap_ms", gaps)

        conditions = output.create_group("conditions")
        target_code = int(record["target_side_code"])
        condition = np.column_stack(
            (
                np.full(n_rows, target_code, dtype=np.float32),
                np.ones(n_rows, dtype=np.float32),
            )
        )
        conditions.create_dataset(CONDITION_SCHEMA, data=condition)
        conditions.create_dataset(
            "target_side_code", data=np.full(n_rows, target_code, dtype=np.int8)
        )
        conditions.create_dataset(
            "goal_active_mask", data=np.ones(n_rows, dtype=np.uint8)
        )
        conditions.create_dataset(
            "goal_epoch",
            data=np.full(n_rows, int(record["goal_epoch"]), dtype=np.int32),
        )
        _write_string_array(
            conditions,
            "cycle_id",
            [str(record["cycle_id"])] * n_rows,
        )
        goal_valid_mask = np.zeros((n_rows, ACT_CHUNK_STEPS), dtype=np.uint8)
        for row in range(n_rows):
            goal_valid_mask[row, : min(ACT_CHUNK_STEPS, n_rows - row)] = 1
        qpos_values = _read_indexed(source["observations/qpos"], obs_idx)
        qvel_values = _read_indexed(source["observations/qvel"], obs_idx)
        phase = derive_cycle_phase(
            qpos=qpos_values,
            qvel=qvel_values,
            excursion_min_delta_rad=EXCURSION_MIN_DELTA_RAD,
            excursion_min_consecutive_samples=EXCURSION_MIN_CONSECUTIVE_SAMPLES,
        )
        phase_valid_mask = phase_chunk_valid_mask(
            phase, chunk_steps=ACT_CHUNK_STEPS
        )
        excursion_observed = derive_excursion_observed(
            qpos=qpos_values,
            minimum_delta_rad=EXCURSION_MIN_DELTA_RAD,
            minimum_consecutive_samples=EXCURSION_MIN_CONSECUTIVE_SAMPLES,
        )
        excursion_valid_mask = excursion_chunk_valid_mask(
            excursion_observed,
            chunk_steps=ACT_CHUNK_STEPS,
        )
        action_values = _read_indexed(source["action"], action_idx)
        return_commit = derive_return_commit(
            action=action_values,
            excursion_observed=excursion_observed,
            return_phase=phase,
            chunk_steps=ACT_CHUNK_STEPS,
            action_intent_threshold=RETURN_COMMIT_ACTION_INTENT_THRESHOLD,
        )
        if record["training_tier"] == "clean" and not return_commit.evaluable:
            raise TransitionContractError(
                f"clean cycle {record['cycle_id']} has no evaluable return commit: "
                f"{return_commit.reason}"
            )
        conditions.create_dataset(CYCLE_PHASE_KEY, data=phase)
        conditions.create_dataset(EXCURSION_OBSERVED_KEY, data=excursion_observed)
        conditions.create_dataset(RETURN_COMMIT_KEY, data=return_commit.state)
        conditions.create_dataset("goal_valid_mask", data=goal_valid_mask)
        conditions.create_dataset(
            "cycle_phase_valid_mask", data=phase_valid_mask
        )
        conditions.create_dataset(
            "excursion_observed_valid_mask", data=excursion_valid_mask
        )
        conditions.create_dataset(
            "return_commit_valid_mask", data=return_commit.valid_mask
        )
        conditions.create_dataset(
            "valid_mask",
            data=(
                goal_valid_mask
                & phase_valid_mask
                & excursion_valid_mask
                & return_commit.valid_mask
            ),
        )

        labels = output.create_group("labels")
        for key, value in {
            "current_ready_side": record["current_ready_side"],
            "scripted_target_side": record["target_side"],
            "realized_target_side": record["realized_target_side"],
            "home_side_coordinate_rad": 0.000690,
            "goal_source": "frozen_sequence_automatic_commit",
            "transition_type": record["transition_type"],
            "transition_success": int(
                record["realized_target_side"] == record["target_side"]
            ),
            "physical_effect": (
                "automatic_ready_cycle_complete"
                if record["target_ready_event"].get("event_source") == "automatic"
                else "operator_marked_cycle_complete"
            ),
            "failure_reason": ",".join(record["qc_reasons"]),
            "expected_return_swing_sign": (
                0
                if record["expected_return_swing_sign"] is None
                else int(record["expected_return_swing_sign"])
            ),
            "training_tier": record["training_tier"],
            "return_commit_evaluable": int(return_commit.evaluable),
            "return_commit_event_row": (
                -1 if return_commit.event_row is None else return_commit.event_row
            ),
            "return_commit_reason": str(return_commit.reason or ""),
        }.items():
            labels.attrs[key] = value

        provenance = output.create_group("provenance")
        provenance.attrs["source_session_id"] = str(record["manifest"]["session_id"])
        provenance.attrs["source_block_id"] = str(record["manifest"]["block_id"])
        provenance.attrs["source_run_id"] = str(record["manifest"]["run_id"])
        provenance.attrs["annotation_version"] = ANNOTATION_VERSION
        provenance.attrs["annotation_sha256"] = annotation_sha256
        provenance.create_dataset("source_row_index", data=obs_idx)
        provenance.create_dataset("source_action_row_index", data=action_idx)
        provenance.create_dataset(
            "source_step_id", data=np.asarray(record["source_step_ids"], dtype=np.int64)
        )
    return return_commit


def _cycle_manifest_row(
    record: Mapping[str, Any],
    *,
    episode_name: str,
    episode_sha256: str,
    annotation_sha256: str,
    phase_contract_sha256: str,
    excursion_contract_sha256: str,
    return_commit_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": CYCLE_MANIFEST_SCHEMA,
        "episode_id": int(record["episode_id"]),
        "episode_path": f"episodes/{episode_name}",
        "episode_sha256": episode_sha256,
        "annotation_version": ANNOTATION_VERSION,
        "annotation_sha256": annotation_sha256,
        "cycle_phase_schema": CYCLE_PHASE_KEY,
        "cycle_phase_contract_sha256": phase_contract_sha256,
        "excursion_observed_schema": EXCURSION_OBSERVED_KEY,
        "excursion_observed_contract_sha256": excursion_contract_sha256,
        "return_commit_schema": RETURN_COMMIT_KEY,
        "return_commit_contract_sha256": return_commit_contract_sha256,
        "return_commit_evaluable": bool(record["return_commit_evaluable"]),
        "return_commit_event_row": record["return_commit_event_row"],
        "return_commit_source_action_row": (
            None
            if record["return_commit_event_row"] is None
            else int(
                np.asarray(record["source_action_indices"], dtype=np.int64)[
                    int(record["return_commit_event_row"])
                ]
            )
        ),
        "return_commit_reason": record["return_commit_reason"],
        "source_session_id": record["manifest"]["session_id"],
        "source_block_id": record["manifest"]["block_id"],
        "source_run_id": record["manifest"]["run_id"],
        "split": record["manifest"]["split"],
        "cycle_id": record["cycle_id"],
        "cycle_index": int(record["cycle_index"]),
        "goal_epoch": int(record["goal_epoch"]),
        "current_ready_side": record["current_ready_side"],
        "scripted_target_side": record["target_side"],
        "target_side_code": int(record["target_side_code"]),
        "realized_target_side": record["realized_target_side"],
        "transition_type": record["transition_type"],
        "training_tier": record["training_tier"],
        "training_ready": record["training_tier"] == "clean",
        "qc_reasons": list(record["qc_reasons"]),
        "qc_metrics": dict(record["qc_metrics"]),
        "n_rows": int(len(record["source_indices"])),
        "first_source_row": int(record["source_indices"][0]),
        "last_source_row": int(record["source_indices"][-1]),
    }


def _select_cycle_observation_indices(
    *, step_ns: np.ndarray, first_row: int, last_row: int
) -> np.ndarray:
    if first_row > last_row:
        return np.zeros(0, dtype=np.int64)
    timestamps = np.asarray(step_ns, dtype=np.int64)
    start_ns = int(timestamps[first_row])
    end_ns = int(timestamps[last_row])
    period_ns = int(round(1_000_000_000.0 / TARGET_HZ))
    targets = np.arange(start_ns, end_ns + 1, period_ns, dtype=np.int64)
    indices = np.searchsorted(timestamps, targets, side="left")
    indices = indices[(indices >= first_row) & (indices <= last_row)]
    if indices.size and int(indices[-1]) != last_row:
        indices = np.append(indices, np.int64(last_row))
    return np.unique(indices).astype(np.int64)


def _select_action_indices(
    *, source: h5py.File, observation_indices: np.ndarray, minimum_index: int
) -> np.ndarray:
    if "diagnostics/action_sample_timestamp_ns" in source:
        action_ns = np.asarray(
            source["diagnostics/action_sample_timestamp_ns"][()], dtype=np.int64
        )
        if np.any(action_ns <= 0) or np.any(np.diff(action_ns) < 0):
            action_ns = np.asarray(source["timestamps/step_ns"][()], dtype=np.int64)
    else:
        action_ns = np.asarray(source["timestamps/step_ns"][()], dtype=np.int64)
    observation_ns = np.asarray(source["timestamps/step_ns"][observation_indices], dtype=np.int64)
    offset_ns = int(round(ACTION_LABEL_OFFSET_S * 1_000_000_000.0))
    selected = np.searchsorted(action_ns, observation_ns + offset_ns, side="right") - 1
    return np.clip(selected, int(minimum_index), len(action_ns) - 1).astype(np.int64)


def _first_threshold_row(
    values: np.ndarray, *, start: int, stop: int, threshold: float
) -> int | None:
    selected = np.flatnonzero(np.asarray(values[start : stop + 1]) > float(threshold))
    return None if not selected.size else int(start + selected[0])


def _event_row(
    event: Mapping[str, Any], row_by_step: Mapping[int, int], step_ns: np.ndarray
) -> int:
    step = int(event["event_step_id"])
    if step not in row_by_step:
        raise TransitionContractError(f"event step_id {step} is absent from raw HDF5")
    row = int(row_by_step[step])
    if int(step_ns[row]) != int(event["event_step_ns"]):
        raise TransitionContractError(f"event step/time mismatch for step_id {step}")
    return row


def _boundary(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_step_id": int(event["event_step_id"]),
        "source_step_ns": int(event["event_step_ns"]),
        "event_id": str(event["event_id"]),
        "event_source": str(event["event_source"]),
    }


def _row_boundary(record: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    row = record.get(key)
    if row is None:
        return None
    detector = {
        "excursion_data_row": "positive_swing_displacement_from_goal_anchor",
        "return_ready_proxy_row": "final_entry_into_target_clean_side",
    }.get(key, "action_amplitude_threshold")
    result = {
        "source_row_index": int(row),
        "source_step_id": int(record["source_step_ids_all"][int(row)]),
        "source_step_ns": int(record["source_step_ns_all"][int(row)]),
        "detector": detector,
    }
    if detector == "action_amplitude_threshold":
        result["threshold"] = ACTION_INTENT_THRESHOLD
    elif detector == "positive_swing_displacement_from_goal_anchor":
        result["min_abs_delta_rad"] = EXCURSION_MIN_DELTA_RAD
        result["min_consecutive_samples"] = EXCURSION_MIN_CONSECUTIVE_SAMPLES
    return result


def _shortest_angle_array(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def _first_consecutive_true_end(
    values: np.ndarray, required: int
) -> int | None:
    count = 0
    for index, value in enumerate(np.asarray(values, dtype=bool)):
        count = count + 1 if bool(value) else 0
        if count >= required:
            return int(index)
    return None


def _last_target_side_entry(
    swing_qpos: np.ndarray, *, target_side: str
) -> int | None:
    delta = _shortest_angle_array(
        np.asarray(swing_qpos, dtype=np.float64) - HOME_SWING_RAD
    )
    target_sign = LEFT_SIGN if target_side == "A" else -LEFT_SIGN
    in_target = delta * target_sign >= CLEAN_READY_MIN_DELTA_RAD
    entries = np.flatnonzero(in_target & np.r_[False, ~in_target[:-1]])
    return None if not entries.size else int(entries[-1])


def _copy_indexed(
    source: h5py.File, output: h5py.File, path: str, indices: np.ndarray
) -> None:
    dataset = source[path]
    parent_path, name = path.rsplit("/", 1) if "/" in path else ("", path)
    parent = output.require_group(parent_path) if parent_path else output
    values = _read_indexed(dataset, indices)
    copied = parent.create_dataset(name, data=values, dtype=dataset.dtype)
    for key, value in dataset.attrs.items():
        copied.attrs[key] = value


def _read_indexed(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read ordered indices while preserving duplicates rejected by h5py."""

    requested = np.asarray(indices, dtype=np.int64).reshape(-1)
    if any(size == 0 for size in dataset.shape[1:]):
        return np.empty((requested.size, *dataset.shape[1:]), dtype=dataset.dtype)
    if not requested.size:
        return np.asarray(dataset[0:0])
    unique, inverse = np.unique(requested, return_inverse=True)
    return np.asarray(dataset[unique])[inverse]


def _copy_encoded_images(
    source: h5py.File, output: h5py.File, indices: np.ndarray
) -> None:
    source_group = source["observations/encoded_images"]
    output_group = output.require_group("observations/encoded_images")
    dtype = h5py.vlen_dtype(np.dtype("uint8"))
    for camera in EXPECTED_CAMERAS:
        source_dataset = source_group[camera]
        target = output_group.create_dataset(camera, (len(indices),), dtype=dtype)
        for key, value in source_dataset.attrs.items():
            target.attrs[key] = value
        target.attrs.setdefault("encoding", "jpeg")
        for output_index, source_index in enumerate(indices):
            target[output_index] = np.asarray(
                source_dataset[int(source_index)], dtype=np.uint8
            ).reshape(-1)


def _replace_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=np.asarray(values))


def _write_string_array(group: h5py.Group, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    dataset = group.create_dataset(name, (len(values),), dtype=dtype)
    dataset[:] = values


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    dict(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
                )
                + "\n"
            )


def _write_checksums(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    with (output_dir / "SHA256SUMS.txt").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.relative_to(output_dir)}\n")


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TransitionContractError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TransitionContractError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            rows.append(value)
    return rows


def _decode_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
