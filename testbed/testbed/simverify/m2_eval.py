"""M2 offline-evaluation contracts for observable-only SimVerify.

The builder consumes only the immutable M0 package's train and validation
splits.  It does not load a policy, read held-out observations, or fabricate
model metrics.  Its artifacts freeze the replay questions, anchor sets,
expert event envelopes, runtime scheduling semantics, and the three-stage
action trace contract required before B0/B1/B2 runs may begin.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.simverify.annotations import SECTORS, condition_vector
from testbed.simverify.artifacts import (
    artifact_identity,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance

EVENT_ORDER = (
    "ready_start",
    "dig_entry_proxy",
    "carry_transition_proxy",
    "dump_start_proxy",
    "dump_end_proxy",
    "ready_end",
)
AXES = ("swing", "boom", "stick", "bucket")
M2_SCHEMA = "simverify_m2_offline_eval_contract_v1"
TRACE_SCHEMA = "simverify_policy_replay_trace_v1"
DEFAULT_M0_ROOT = Path(
    "/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3"
)
DEFAULT_M2_ROOT = Path(
    "/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3_m2_contract_v1"
)


def test_intent_registry() -> dict[str, Any]:
    """Return the frozen E00-E07 question and evidence-boundary registry."""

    common = {
        "evidence_scope": "recorded-observation/offline",
        "closed_loop_claim_allowed": False,
        "privilege_inputs_allowed": False,
        "held_out_access_before_gate_thresholds_sha": False,
    }
    intents = {
        "E00": {
            "question": "Can the bundle and dataset be consumed under one canonical observable-only contract?",
            "observable_inputs": [
                "dataset_manifest",
                "split_manifest",
                "camera_mapping",
                "state_action_time_contract",
                "condition_schema",
                "checkpoint_bundle_manifest",
                "dataset_stats",
            ],
            "intervention": "none_contract_validation_only",
            "metrics": ["contract_violation_count"],
            "can_prove": "artifact compatibility and privilege isolation",
            "cannot_prove": "policy task performance",
            "stop_conditions": ["any_contract_violation"],
        },
        "E01": {
            "question": "What does the policy emit along the complete recorded observation path?",
            "observable_inputs": [
                "four_camera_observation",
                "qpos",
                "qvel",
                "cycle_condition_v1_when_valid",
            ],
            "intervention": "none_teacher_forced_recorded_order",
            "metrics": [
                "raw_policy_chunk",
                "temporal_aggregation_action",
                "future_runtime_safe_action",
                "effective_axis_class",
            ],
            "can_prove": "open-loop policy outputs on recorded observations",
            "cannot_prove": "environment response or closed-loop success",
            "stop_conditions": [
                "missing_action_stage",
                "condition_sidecar_mismatch",
            ],
        },
        "E02": {
            "question": "Do replayed actions contain the observable task events in a compatible order?",
            "observable_inputs": [
                "recorded_observation_replay_trace",
                "expert_event_envelope",
            ],
            "intervention": "task_event_extraction",
            "metrics": [
                "required_event_coverage",
                "event_order_violation_rate",
                "missing_phase_rate",
                "opposite_direction_rate",
                "unexpected_effective_axis_rate",
                "deadzone_effective_recall",
            ],
            "can_prove": "recorded-path action-event compatibility",
            "cannot_prove": "physical digging or payload release",
            "stop_conditions": ["event_extractor_contract_mismatch"],
        },
        "E03": {
            "question": "Does a supported condition swap change actions in a stable target-related way?",
            "observable_inputs": [
                "identical_recorded_observation_history",
                "condition_support_index",
                "base_and_swapped_condition",
            ],
            "intervention": "change_exactly_one_condition_field",
            "metrics": [
                "token_swap_action_effect",
                "token_swap_direction_accuracy",
                "token_response_latency_ticks",
                "same_token_repeat_consistency",
                "current_sector_sensitivity",
                "next_sector_sensitivity",
                "condition_ignored_rate",
            ],
            "can_prove": "condition sensitivity on supported recorded anchors",
            "cannot_prove": "counterfactual physical outcome",
            "stop_conditions": [
                "unsupported_counterfactual_in_success_denominator",
                "observation_history_changed",
            ],
        },
        "E04": {
            "question": "Does condition/task response depend on a camera role or fixed image shortcut?",
            "observable_inputs": [
                "same_recorded_state_and_condition",
                "four_camera_observation",
            ],
            "intervention": (
                "eye_only_or_stick_only_or_single_dropout_or_role_swap_or_hold"
            ),
            "metrics": [
                "eye_only_retention",
                "stick_only_retention",
                "single_camera_dropout_retention",
                "pair_swap_failure_rate",
                "cross_role_swap_failure_rate",
            ],
            "can_prove": "offline camera-role sensitivity",
            "cannot_prove": "real-camera generalization",
            "stop_conditions": ["more_than_one_primary_factor_changed"],
        },
        "E05": {
            "question": "How do policy state and temporal aggregation evolve under a held recorded observation?",
            "observable_inputs": [
                "one_recorded_observation_anchor",
                "fixed_condition",
                "policy_state_snapshot",
            ],
            "intervention": "repeat_identical_observation_for_declared_ticks",
            "metrics": [
                "state_hold_direction_flip_rate",
                "active_hold_ticks",
                "unexpected_effective_axis_rate",
                "temporal_aggregation_population",
                "snapshot_restore_max_abs_delta",
            ],
            "can_prove": "internal policy and aggregation evolution under state hold",
            "cannot_prove": "hydraulic soil contact or qpos response",
            "stop_conditions": ["generated_physical_state_used"],
        },
        "E06": {
            "question": "Does recorded-observation replay obey deployed delay, stale, timeout, and latest-wins semantics?",
            "observable_inputs": [
                "recorded_observation_sequence",
                "raw_policy_chunks",
                "declared_runtime_schedule",
            ],
            "intervention": (
                "delay_completion_or_skip_observation_or_repeat_latest_or_timeout"
            ),
            "metrics": [
                "stale_action_age_ticks",
                "delay_event_order_violation_rate",
                "effective_direction_disagreement",
                "timeout_zero_count",
            ],
            "can_prove": "offline runtime scheduling semantics",
            "cannot_prove": "real-time hardware performance",
            "stop_conditions": ["unrecorded_observation_generated"],
        },
        "E07": {
            "question": "Does a two-cycle recorded path preserve both cycle semantics across the ready boundary?",
            "observable_inputs": [
                "two_consecutive_accepted_cycles",
                "recorded_observations",
                "both_cycle_conditions",
            ],
            "intervention": "none_two_cycle_recorded_path",
            "metrics": [
                "two_cycle_phase_coverage",
                "ready_boundary_discontinuity",
                "second_cycle_condition_start",
            ],
            "can_prove": "two-cycle recorded-path semantic continuity",
            "cannot_prove": "closed-loop repeated excavation",
            "stop_conditions": ["single_trajectory_used_for_verdict"],
        },
    }
    return {
        "schema": "simverify_test_intent_registry_v1",
        "hard_rule": "HR-12",
        "intents": {key: {**common, **value} for key, value in sorted(intents.items())},
    }


def replay_trace_schema() -> dict[str, Any]:
    """Return the three-stage action trace contract."""

    return {
        "schema": TRACE_SCHEMA,
        "evidence_scope": "recorded-observation/offline",
        "required_arrays": {
            "raw_policy_chunk_normalized": {
                "shape": ["T", "Q", 4],
                "dtype": "float32",
                "meaning": "direct neural output before action unnormalization",
            },
            "raw_policy_chunk_direct": {
                "shape": ["T", "Q", 4],
                "dtype": "float32",
                "meaning": "raw chunk after frozen action unnormalization",
            },
            "temporal_aggregation_action": {
                "shape": ["T", 4],
                "dtype": "float32",
                "meaning": "ACT temporal aggregation output before runtime safety",
            },
            "future_runtime_safe_action": {
                "shape": ["T", 4],
                "dtype": "float32",
                "meaning": (
                    "future runtime-safe action; M2/B0-B2 offline replay must "
                    "preserve it separately even when equal to aggregation"
                ),
            },
            "expert_action": {
                "shape": ["T", 4],
                "dtype": "float32",
                "meaning": "recorded source-domain actuator_speed_cmd",
            },
            "condition": {
                "shape": ["T", 6],
                "dtype": "float32",
                "meaning": "explicit cycle_condition_v1",
            },
        },
        "required_index_arrays": [
            "target_tick",
            "source_observation_index",
            "condition_cycle_id",
            "condition_valid_mask",
            "observation_age_ticks",
            "action_age_ticks",
        ],
        "required_provenance": [
            "git_sha",
            "git_dirty",
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "camera_mapping_sha256",
            "condition_schema_sha256",
            "checkpoint_sha256",
            "dataset_stats_sha256",
            "resolved_config_sha256",
            "test_intent_registry_sha256",
            "episode_ids",
            "seed",
            "inference_precision",
            "temporal_aggregation_config",
        ],
        "stage_aliasing_forbidden": True,
        "teacher_forced_open_loop": True,
        "closed_loop_execution": False,
    }


def validate_replay_trace_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    chunk_size: int,
) -> None:
    """Fail closed when raw, aggregated, and runtime-safe stages are aliased."""

    schema = replay_trace_schema()
    required_float = set(schema["required_arrays"])
    required_index = set(schema["required_index_arrays"])
    missing = (required_float | required_index) - set(arrays)
    if missing:
        raise ValueError(f"replay trace missing arrays: {sorted(missing)}")
    raw_normalized = np.asarray(arrays["raw_policy_chunk_normalized"])
    raw_direct = np.asarray(arrays["raw_policy_chunk_direct"])
    if raw_normalized.ndim != 3 or raw_normalized.shape[1:] != (chunk_size, 4):
        raise ValueError("raw normalized chunk must have shape (T, Q, 4)")
    if raw_direct.shape != raw_normalized.shape:
        raise ValueError("raw direct chunk shape mismatch")
    step_count = raw_direct.shape[0]
    for key in (
        "temporal_aggregation_action",
        "future_runtime_safe_action",
        "expert_action",
    ):
        value = np.asarray(arrays[key])
        if value.shape != (step_count, 4):
            raise ValueError(f"{key} must have shape (T, 4)")
    condition = np.asarray(arrays["condition"])
    if condition.shape != (step_count, 6):
        raise ValueError("condition must have shape (T, 6)")
    for key in required_float:
        value = np.asarray(arrays[key])
        if value.dtype != np.float32:
            raise ValueError(f"{key} must be float32")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values")
    for key in required_index:
        value = np.asarray(arrays[key])
        if value.shape != (step_count,):
            raise ValueError(f"{key} must have shape (T,)")
        if key == "condition_valid_mask":
            if value.dtype not in (np.dtype("bool"), np.dtype("uint8")):
                raise ValueError("condition_valid_mask must be bool or uint8")
        elif not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{key} must use an integer dtype")
    action_stages = (
        "raw_policy_chunk_normalized",
        "raw_policy_chunk_direct",
        "temporal_aggregation_action",
        "future_runtime_safe_action",
    )
    for index, left in enumerate(action_stages):
        for right in action_stages[index + 1 :]:
            if np.shares_memory(
                np.asarray(arrays[left]),
                np.asarray(arrays[right]),
            ):
                raise ValueError(f"action stages must not alias: {left}, {right}")


def effective_signature(
    action: np.ndarray,
    deadzone: Sequence[float],
) -> tuple[int, int, int, int]:
    values = np.asarray(action, dtype=np.float64)
    threshold = np.asarray(deadzone, dtype=np.float64)
    if values.shape != (4,) or threshold.shape != (4,):
        raise ValueError("action and deadzone must both have shape (4,)")
    if not np.isfinite(values).all() or np.any(threshold < 0):
        raise ValueError("invalid action or deadzone")
    return tuple(
        int(1 if value > limit else -1 if value < -limit else 0)
        for value, limit in zip(values, threshold)
    )


def extract_ordered_task_events(
    action: np.ndarray,
    templates: Mapping[str, Mapping[str, Any]],
    *,
    deadzone: Sequence[float],
) -> dict[str, Any]:
    """Greedily match data-generated effective signatures in timing envelopes."""

    values = np.asarray(action, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] <= 0:
        raise ValueError("action must have shape (T, 4) with T > 0")
    signatures = [effective_signature(row, deadzone) for row in values]
    found: dict[str, int | None] = {}
    match_source: dict[str, str] = {}
    previous = 0
    for event_name in EVENT_ORDER:
        if event_name == "ready_start":
            found[event_name] = 0
            match_source[event_name] = "observable_cycle_boundary"
            continue
        if event_name == "ready_end":
            found[event_name] = values.shape[0] - 1
            match_source[event_name] = "observable_cycle_boundary"
            continue
        template = templates[event_name]
        required_axis_signs = template["required_axis_signs"]
        if len(required_axis_signs) != 4:
            raise ValueError("required_axis_signs must have four entries")
        low_fraction = float(template["relative_position"]["p02_5"])
        high_fraction = float(template["relative_position"]["p97_5"])
        final_index = values.shape[0] - 1
        low = max(previous + 1, int(np.floor(low_fraction * final_index)))
        high = min(
            final_index - 1,
            int(np.ceil(high_fraction * final_index)),
        )
        match = next(
            (
                tick
                for tick in range(low, high + 1)
                if all(
                    expected is None or signatures[tick][axis] == int(expected)
                    for axis, expected in enumerate(required_axis_signs)
                )
            ),
            None,
        )
        found[event_name] = match
        match_source[event_name] = "train_generated_effective_axis_rule"
        if match is not None:
            previous = int(match)
    matched = [found[name] is not None for name in EVENT_ORDER]
    ordered_ticks = [
        int(found[name]) for name in EVENT_ORDER if found[name] is not None
    ]
    return {
        "schema": "simverify_action_event_extraction_v1",
        "event_ticks": found,
        "event_match_source": match_source,
        "required_event_coverage": float(np.mean(matched)),
        "missing_events": [name for name in EVENT_ORDER if found[name] is None],
        "event_order_valid": all(
            left < right for left, right in zip(ordered_ticks[:-1], ordered_ticks[1:])
        ),
        "teacher_forced_recorded_observation": True,
        "physical_event_claimed": False,
    }


def simulate_latest_wins(
    chunks: Sequence[Mapping[str, Any]],
    *,
    control_ticks: int,
    timeout_ticks: int,
) -> dict[str, np.ndarray]:
    """Apply explicit latest-wins, stale-offset, repeat-last, and timeout rules."""

    if control_ticks <= 0 or timeout_ticks < 0:
        raise ValueError("invalid control or timeout ticks")
    prepared = []
    for row in chunks:
        issue = int(row["issue_tick"])
        ready = int(row["ready_tick"])
        values = np.asarray(row["raw_chunk_direct"], dtype=np.float32)
        if issue < 0 or ready < issue or values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("invalid scheduled chunk")
        prepared.append((issue, ready, values))
    prepared.sort(key=lambda item: (item[1], item[0]))
    runtime = np.zeros((control_ticks, 4), dtype=np.float32)
    source_issue = np.full(control_ticks, -1, dtype=np.int64)
    action_age = np.full(control_ticks, -1, dtype=np.int64)
    timed_out = np.zeros(control_ticks, dtype=bool)
    active: tuple[int, int, np.ndarray] | None = None
    for tick in range(control_ticks):
        ready_rows = [row for row in prepared if row[1] <= tick]
        if ready_rows:
            newest = max(ready_rows, key=lambda item: item[0])
            if active is None or newest[0] > active[0]:
                active = newest
        if active is None:
            timed_out[tick] = True
            continue
        issue, _ready, values = active
        age = tick - issue
        source_issue[tick] = issue
        action_age[tick] = age
        if age > timeout_ticks:
            timed_out[tick] = True
            continue
        offset = min(max(age, 0), values.shape[0] - 1)
        runtime[tick] = values[offset]
    return {
        "future_runtime_safe_action": runtime,
        "source_issue_tick": source_issue,
        "action_age_ticks": action_age,
        "timed_out": timed_out,
    }


def run_m2_contract_builder(
    *,
    m0_root: str | Path = DEFAULT_M0_ROOT,
    output_root: str | Path = DEFAULT_M2_ROOT,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build the immutable M2 pre-model offline-evaluation contract package."""

    m0 = Path(m0_root).resolve(strict=True)
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable M2 output already exists: {destination}")
    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        not git.get("git_available")
        or git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
    ):
        raise ValueError("M2 builder requires a clean v2.0.0-simVerify worktree")
    manifest = _read_json(m0 / "dataset_manifest.json")
    if manifest.get("status") != "m0_artifacts_frozen_m1_import_smoke_pending":
        raise ValueError("M2 requires a frozen passing M0 package")
    split = _read_json(m0 / "split_groups.json")
    train_ids = list(map(int, split["splits"]["train"]))
    validation_ids = list(map(int, split["splits"]["validation"]))
    held_out_ids = list(map(int, split["splits"]["held_out_test"]))
    if set(train_ids) & set(validation_ids):
        raise ValueError("train/validation split overlap")

    input_paths = {
        "dataset_manifest": m0 / "dataset_manifest.json",
        "split_manifest": m0 / "split_groups.json",
        "camera_mapping": m0 / "camera_mapping.json",
        "state_action_contract": m0 / "state_action_contract.json",
        "condition_schema": m0 / "cycle_condition_v1.schema.json",
        "annotation_sidecar": m0 / "cycle_annotations.jsonl",
        "condition_support_index": m0 / "condition_support_index.json",
        "transition_inventory": m0 / "transition_inventory.json",
        "gate_thresholds_contract": m0 / "gate_thresholds_contract_v1.json",
        "m1_report": m0.parent / "sim_observable_cycle_v3_m1_import_smoke.json",
    }
    input_identities = {
        name: artifact_identity(path) for name, path in input_paths.items()
    }
    m1_report = _read_json(input_paths["m1_report"])
    if (
        m1_report.get("status") != "passed"
        or m1_report.get("passed") is not True
        or m1_report.get("held_out_test_read") is not False
        or m1_report.get("training_started") is not False
        or m1_report.get("closed_loop_execution") is not False
    ):
        raise ValueError("M2 requires a passing offline-only M1 import smoke")
    episode_manifest = {int(row["episode_id"]): row for row in manifest["episodes"]}
    episode_actions: dict[int, np.ndarray] = {}
    episode_input_identities: list[dict[str, Any]] = []
    for episode_id in sorted(train_ids + validation_ids):
        relative = f"episodes/episode_{episode_id}.hdf5"
        path = m0 / relative
        identity = artifact_identity(path)
        expected = episode_manifest[episode_id]["sha256"]
        if identity["sha256"] != expected:
            raise ValueError(f"M0 episode checksum mismatch: {episode_id}")
        episode_input_identities.append(
            {
                "episode_id": episode_id,
                "split": "train" if episode_id in train_ids else "validation",
                "path": relative,
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )
        with h5py.File(path, "r") as handle:
            episode_actions[episode_id] = np.asarray(
                handle["action"],
                dtype=np.float32,
            )

    annotations = _read_jsonl(m0 / "cycle_annotations.jsonl")
    if any(int(row["episode_id"]) in held_out_ids for row in annotations):
        raise ValueError("M2 annotation sidecar unexpectedly contains held-out rows")
    calibration_ids = set(train_ids + validation_ids)
    accepted = [
        row
        for row in annotations
        if row["quality"]["status"] == "accepted"
        and int(row["episode_id"]) in calibration_ids
    ]
    state_contract = _read_json(m0 / "state_action_contract.json")
    deadzone = list(
        map(float, state_contract["action"]["source_generation"]["deadzone"])
    )
    event_envelope = _fit_expert_event_envelope(
        accepted,
        episode_actions,
        deadzone=deadzone,
        train_ids=set(train_ids),
        validation_ids=set(validation_ids),
    )
    support = _read_json(m0 / "condition_support_index.json")
    counterfactual_rows = _counterfactual_anchors(
        accepted,
        support,
    )
    state_hold_rows = _state_hold_anchors(accepted)
    two_cycle_rows = _two_cycle_anchors(accepted)

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"temporary M2 output exists: {temporary}")
    temporary.mkdir(parents=True)
    artifacts: list[dict[str, Any]] = []
    try:
        provenance = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git": git,
            "m0_root": str(m0),
            "m0_dataset_manifest_sha256": input_identities["dataset_manifest"][
                "sha256"
            ],
            "m1_report_sha256": input_identities["m1_report"]["sha256"],
            "split_manifest_sha256": input_identities["split_manifest"]["sha256"],
            "evidence_scope": "recorded-observation/offline",
            "held_out_episode_access_count": 0,
        }
        registry_identity = write_json(
            temporary / "test_intent_registry_v1.json",
            {**test_intent_registry(), "provenance": provenance},
        )
        artifacts.append(registry_identity)
        trace_identity = write_json(
            temporary / "replay_trace_schema_v1.json",
            {**replay_trace_schema(), "provenance": provenance},
        )
        artifacts.append(trace_identity)
        event_identity = write_json(
            temporary / "expert_event_envelope_v1.json",
            {**event_envelope, "provenance": provenance},
        )
        artifacts.append(event_identity)
        counterfactual_identity = write_jsonl(
            temporary / "condition_counterfactual_anchors_v1.jsonl",
            counterfactual_rows,
        )
        artifacts.append(counterfactual_identity)
        hold_identity = write_jsonl(
            temporary / "state_hold_anchors_v1.jsonl",
            state_hold_rows,
        )
        artifacts.append(hold_identity)
        two_cycle_identity = write_jsonl(
            temporary / "two_cycle_anchors_v1.jsonl",
            two_cycle_rows,
        )
        artifacts.append(two_cycle_identity)
        runtime_identity = write_json(
            temporary / "delay_latest_wins_contract_v1.json",
            {
                "schema": "simverify_delay_latest_wins_contract_v1",
                "control_hz": 20.0,
                "parameters_required_per_run": [
                    "inference_issue_ticks",
                    "inference_ready_ticks",
                    "observation_skip_ticks",
                    "timeout_ticks",
                ],
                "latest_wins": (
                    "at each control tick select the ready chunk with greatest "
                    "issue_tick; discard older ready chunks"
                ),
                "stale_offset": (
                    "execute chunk offset control_tick-minus-issue_tick; clamp "
                    "to last chunk row"
                ),
                "repeat_recent_action": (
                    "when no newer chunk is ready, continue the active chunk"
                ),
                "timeout": "strictly older than timeout_ticks yields zero",
                "observation_source": "recorded_only_no_generated_physical_state",
                "parameters_are_scenario_variables_not_gate_thresholds": True,
                "provenance": provenance,
            },
        )
        artifacts.append(runtime_identity)
        authorization_identity = write_json(
            temporary / "m2_authorization_report_v1.json",
            {
                "schema": "simverify_m2_authorization_report_v1",
                "stage": "M2",
                "status": "offline_eval_skeleton_frozen_model_replays_pending",
                "passed": True,
                "test_intent_count": 8,
                "expert_event_template_count": len(EVENT_ORDER),
                "counterfactual_anchor_count": len(counterfactual_rows),
                "supported_counterfactual_anchor_count": sum(
                    bool(row["supported"]) for row in counterfactual_rows
                ),
                "state_hold_anchor_count": len(state_hold_rows),
                "two_cycle_anchor_count": len(two_cycle_rows),
                "held_out_test_read": False,
                "gate_thresholds_v1_generated": False,
                "b0_replay_artifact_exists": False,
                "b1_replay_artifact_exists": False,
                "b2_replay_artifact_exists": False,
                "m3_unconditioned_baseline_authorized": True,
                "m4_conditioned_candidate_authorized": False,
                "training_started": False,
                "closed_loop_execution": False,
                "provenance": provenance,
            },
        )
        artifacts.append(authorization_identity)
        package_manifest = {
            "schema": M2_SCHEMA,
            "stage": "M2",
            "status": "completed_model_replays_pending",
            "evidence_scope": "recorded-observation/offline",
            "training_started": False,
            "held_out_test_authorized": False,
            "gate_thresholds_v1_status": "not_generated",
            "source_splits": ["train", "validation"],
            "source_episode_ids": {
                "train": train_ids,
                "validation": validation_ids,
                "held_out_test": "locked_unread",
            },
            "input_artifacts": input_identities,
            "input_episodes": episode_input_identities,
            "artifacts": {
                Path(identity["path"]).name: {
                    "sha256": identity["sha256"],
                    "size_bytes": identity["size_bytes"],
                }
                for identity in artifacts
            },
            "provenance": provenance,
        }
        manifest_identity = write_json(
            temporary / "m2_manifest.json",
            package_manifest,
        )
        artifacts.append(manifest_identity)
        checksums_identity = write_checksums(
            temporary,
            artifacts,
            path=temporary / "checksums.sha256",
        )
        os.rename(temporary, destination)
        return {
            "schema": "simverify_m2_completion_v1",
            "status": "completed",
            "output_root": str(destination),
            "m2_manifest_sha256": manifest_identity["sha256"],
            "checksums_sha256": checksums_identity["sha256"],
            "counterfactual_anchor_count": len(counterfactual_rows),
            "state_hold_anchor_count": len(state_hold_rows),
            "two_cycle_anchor_count": len(two_cycle_rows),
            "held_out_test_read": False,
            "training_started": False,
            "closed_loop_execution": False,
        }
    except BaseException as exc:
        if temporary.exists():
            failure = temporary / "BUILD_FAILED.json"
            if not failure.exists():
                failure.write_text(
                    json.dumps(
                        {
                            "schema": "simverify_m2_build_failure_v1",
                            "status": "failed",
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "training_started": False,
                            "held_out_test_read": False,
                            "git": git,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        raise


def _fit_expert_event_envelope(
    accepted: Sequence[Mapping[str, Any]],
    episode_actions: Mapping[int, np.ndarray],
    *,
    deadzone: Sequence[float],
    train_ids: set[int],
    validation_ids: set[int],
) -> dict[str, Any]:
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {event: [] for event in EVENT_ORDER} for split in ("train", "validation")
    }
    cycles_by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in accepted:
        episode_id = int(record["episode_id"])
        split = "train" if episode_id in train_ids else "validation"
        if episode_id not in train_ids | validation_ids:
            raise ValueError("expert envelope received an episode outside calibration")
        start, end = map(int, record["target_steps_20hz"])
        length = end - start
        if length <= 0:
            raise ValueError("accepted cycle has empty target interval")
        cycles_by_split[split].append(record)
        action = episode_actions[episode_id]
        for event_name in EVENT_ORDER:
            event = record["observable_events"][event_name]
            tick = int(event["representative_target_tick"])
            inclusive_end_allowed = event_name == "ready_end" and tick == end
            if (
                not (start <= tick < end or inclusive_end_allowed)
                or tick >= action.shape[0]
            ):
                raise ValueError("event target tick lies outside accepted cycle")
            rows[split][event_name].append(
                {
                    "episode_id": episode_id,
                    "cycle_id": int(record["cycle_id"]),
                    "relative_position": float((tick - start) / length),
                    "target_tick": tick,
                    "action": action[tick].astype(np.float64),
                    "effective_signature": effective_signature(
                        action[tick],
                        deadzone,
                    ),
                }
            )
    templates: dict[str, Any] = {}
    for event_name in EVENT_ORDER:
        train_rows = rows["train"][event_name]
        if not train_rows:
            raise ValueError(f"no train expert rows for {event_name}")
        signatures = Counter(row["effective_signature"] for row in train_rows)
        mode_signature, mode_count = sorted(
            signatures.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        positions = np.asarray(
            [row["relative_position"] for row in train_rows],
            dtype=np.float64,
        )
        actions = np.stack([row["action"] for row in train_rows])
        axis_sign_rules: list[dict[str, Any]] = []
        required_axis_signs: list[int | None] = []
        for axis_index, axis_name in enumerate(AXES):
            counts = Counter(
                int(row["effective_signature"][axis_index]) for row in train_rows
            )
            mode_sign, sign_count = sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]
            wilson_lower = _wilson_lower_bound(
                sign_count,
                len(train_rows),
            )
            required = bool(wilson_lower > 0.5)
            required_axis_signs.append(int(mode_sign) if required else None)
            axis_sign_rules.append(
                {
                    "axis": axis_name,
                    "mode_sign": int(mode_sign),
                    "mode_count": int(sign_count),
                    "sample_count": len(train_rows),
                    "one_sided_95pct_wilson_lower_bound": wilson_lower,
                    "required": required,
                    "rule": "required_only_when_lower_bound_exceeds_majority",
                }
            )
        templates[event_name] = {
            "mode_effective_signature": list(mode_signature),
            "mode_count": int(mode_count),
            "train_count": len(train_rows),
            "event_match_mode": (
                "observable_cycle_boundary"
                if event_name in {"ready_start", "ready_end"}
                else "train_generated_axis_majority"
            ),
            "required_axis_signs": (
                [None, None, None, None]
                if event_name in {"ready_start", "ready_end"}
                else required_axis_signs
            ),
            "axis_sign_rules": axis_sign_rules,
            "signature_counts": {
                ",".join(map(str, signature)): int(count)
                for signature, count in sorted(signatures.items())
            },
            "relative_position": _quantile_summary(positions),
            "action_at_observable_anchor": {
                "axis_median": np.median(actions, axis=0).tolist(),
                "axis_p02_5": np.quantile(actions, 0.025, axis=0).tolist(),
                "axis_p97_5": np.quantile(actions, 0.975, axis=0).tolist(),
            },
        }
    validation_results: list[dict[str, Any]] = []
    for record in cycles_by_split["validation"]:
        episode_id = int(record["episode_id"])
        start, end = map(int, record["target_steps_20hz"])
        extracted = extract_ordered_task_events(
            episode_actions[episode_id][start : end + 1],
            templates,
            deadzone=deadzone,
        )
        validation_results.append(
            {
                "episode_id": episode_id,
                "cycle_id": int(record["cycle_id"]),
                "required_event_coverage": extracted["required_event_coverage"],
                "event_order_valid": extracted["event_order_valid"],
                "missing_events": extracted["missing_events"],
            }
        )
    return {
        "schema": "simverify_expert_event_envelope_v1",
        "event_order": list(EVENT_ORDER),
        "axis_order": list(AXES),
        "effective_deadzone": list(map(float, deadzone)),
        "cycle_replay_window": (
            "target_steps_20hz is half-open for condition ownership; event "
            "extraction includes the shared ready_end boundary tick"
        ),
        "template_fit_split": "train",
        "calibration_split": "validation",
        "templates": templates,
        "validation_expert_distribution": {
            "cycle_count": len(validation_results),
            "required_event_coverage": _quantile_summary(
                np.asarray(
                    [row["required_event_coverage"] for row in validation_results],
                    dtype=np.float64,
                )
            ),
            "event_order_violation_rate": float(
                np.mean([not row["event_order_valid"] for row in validation_results])
            ),
            "cycles": validation_results,
        },
        "threshold_status": (
            "expert_distribution_only_B0_repeat_noise_and_B2_null_pending"
        ),
        "gate_thresholds_v1_generated": False,
    }


def _counterfactual_anchors(
    accepted: Sequence[Mapping[str, Any]],
    support: Mapping[str, Any],
) -> list[dict[str, Any]]:
    support_by_key = {
        (int(row["episode_id"]), int(row["cycle_id"])): row
        for row in support["entries"]
    }
    result: list[dict[str, Any]] = []
    for record in accepted:
        key = (int(record["episode_id"]), int(record["cycle_id"]))
        row = support_by_key[key]
        original_current = str(record["policy_condition"]["current_sector"])
        original_next = str(record["policy_condition"]["next_ready_sector"])
        interventions = [
            ("current_sector", target)
            for target in SECTORS
            if target != original_current
        ] + [("next_sector", target) for target in SECTORS if target != original_next]
        for changed_factor, target in interventions:
            target_current = (
                target if changed_factor == "current_sector" else original_current
            )
            target_next = target if changed_factor == "next_sector" else original_next
            support_key = "current" if changed_factor == "current_sector" else "next"
            support_evidence = row["counterfactuals"][support_key][target]
            supported = bool(support_evidence["supported"])
            result.append(
                {
                    "schema": "simverify_condition_counterfactual_anchor_v1",
                    "episode_id": key[0],
                    "cycle_id": key[1],
                    "split": record["split"],
                    "target_steps_20hz": record["target_steps_20hz"],
                    "base_condition": {
                        "current_sector": original_current,
                        "next_sector": original_next,
                        "vector": condition_vector(
                            original_current,
                            original_next,
                        ),
                    },
                    "target_condition": {
                        "current_sector": target_current,
                        "next_sector": target_next,
                        "vector": condition_vector(
                            target_current,
                            target_next,
                        ),
                    },
                    "changed_factors": [changed_factor],
                    "primary_factor_count": 1,
                    "supported": supported,
                    "status": (
                        "supported_counterfactual"
                        if supported
                        else "unsupported_counterfactual"
                    ),
                    "support_evidence": {
                        "changed_factor": changed_factor,
                        "target_sector": target,
                        "nearest_neighbor_support": copy.deepcopy(support_evidence),
                    },
                    "observation_history_intervention_allowed": False,
                    "included_in_success_denominator": supported,
                }
            )
    return result


def _state_hold_anchors(
    accepted: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for record in accepted:
        start, end = map(int, record["target_steps_20hz"])
        for event_name in EVENT_ORDER:
            tick = int(
                record["observable_events"][event_name]["representative_target_tick"]
            )
            result.append(
                {
                    "schema": "simverify_state_hold_anchor_v1",
                    "episode_id": int(record["episode_id"]),
                    "cycle_id": int(record["cycle_id"]),
                    "split": record["split"],
                    "event_name": event_name,
                    "observation_target_tick": tick,
                    "cycle_target_steps_20hz": [start, end],
                    "remaining_recorded_ticks_in_cycle": int(end - tick),
                    "condition": copy.deepcopy(record["policy_condition"]),
                    "hold_ticks": None,
                    "hold_ticks_status": ("must_be_declared_in_future_replay_config"),
                    "policy_state_snapshot_required": True,
                    "generated_physical_state_allowed": False,
                }
            )
    return result


def _two_cycle_anchors(
    accepted: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in accepted:
        grouped[int(record["episode_id"])][int(record["cycle_id"])] = record
    result = []
    for episode_id, cycles in sorted(grouped.items()):
        for cycle_id, first in sorted(cycles.items()):
            second = cycles.get(cycle_id + 1)
            if second is None:
                continue
            if (
                first["policy_condition"]["next_ready_sector"]
                != second["policy_condition"]["current_sector"]
            ):
                raise ValueError("accepted two-cycle condition continuity mismatch")
            result.append(
                {
                    "schema": "simverify_two_cycle_anchor_v1",
                    "episode_id": episode_id,
                    "split": first["split"],
                    "first_cycle_id": cycle_id,
                    "second_cycle_id": cycle_id + 1,
                    "target_steps_20hz": [
                        int(first["target_steps_20hz"][0]),
                        int(second["target_steps_20hz"][1]),
                    ],
                    "shared_ready_boundary_tick": int(first["target_steps_20hz"][1]),
                    "first_condition": copy.deepcopy(first["policy_condition"]),
                    "second_condition": copy.deepcopy(second["policy_condition"]),
                    "closed_loop_claim_allowed": False,
                }
            )
    return result


def _quantile_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "count": int(array.size),
        "p02_5": float(np.quantile(array, 0.025)),
        "p50": float(np.quantile(array, 0.5)),
        "p97_5": float(np.quantile(array, 0.975)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _wilson_lower_bound(
    successes: int,
    attempts: int,
    *,
    z: float = 1.6448536269514722,
) -> float:
    """Return the one-sided 95% Wilson lower bound for a binomial rate."""

    if attempts <= 0 or not 0 <= successes <= attempts:
        raise ValueError("invalid Wilson count")
    proportion = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = proportion + z * z / (2.0 * attempts)
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / attempts + z * z / (4.0 * attempts * attempts)
    )
    return float((center - spread) / denominator)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
