"""Checksum-bound evaluation of paired gated-condition AGX branches."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from testbed.simverify.agx_closed_loop_matrix import classify_swing_sector
from testbed.simverify.artifacts import (
    artifact_identity,
    verify_checksums,
    write_checksums,
    write_json,
)
from testbed.simverify.contracts import (
    SOURCE_ACTION_ORDER,
    SOURCE_QPOS_ORDER,
    sha256_file,
)

BRANCH_EVAL_SCHEMA = "simverify_habit_agx_shared_prefix_branch_diagnostic_v1"


def compute_branch_effects(
    reference: np.ndarray,
    repeat: np.ndarray,
    treatment: np.ndarray,
    *,
    takeover_tick: int,
) -> dict[str, Any]:
    """Compare a condition intervention against same-condition repeat noise."""

    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (reference, repeat, treatment)
    ]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("branch arrays must share one shape")
    if arrays[0].ndim != 2 or arrays[0].shape[1] != 4:
        raise ValueError("branch arrays must have shape (T,4)")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("branch arrays must be finite")
    tick = int(takeover_tick)
    if not 0 <= tick < arrays[0].shape[0]:
        raise ValueError("takeover_tick lies outside branch arrays")
    reference_values, repeat_values, treatment_values = arrays
    repeat_delta = repeat_values[tick:] - reference_values[tick:]
    treatment_delta = treatment_values[tick:] - reference_values[tick:]
    repeat_mean = np.mean(np.abs(repeat_delta), axis=0)
    treatment_mean = np.mean(np.abs(treatment_delta), axis=0)
    ratio = treatment_mean / np.maximum(repeat_mean, 1.0e-12)
    return {
        "takeover_tick": tick,
        "repeat_mean_abs_delta": repeat_mean.tolist(),
        "repeat_max_abs_delta": np.max(np.abs(repeat_delta), axis=0).tolist(),
        "treatment_mean_abs_delta": treatment_mean.tolist(),
        "treatment_max_abs_delta": np.max(
            np.abs(treatment_delta),
            axis=0,
        ).tolist(),
        "treatment_to_repeat_mean_abs_ratio": ratio.tolist(),
        "treatment_exceeds_repeat_variability_per_axis": (
            treatment_mean > repeat_mean
        ).tolist(),
        "treatment_exceeds_repeat_variability_all_axes": bool(
            np.all(treatment_mean > repeat_mean)
        ),
        "final_repeat_minus_reference": repeat_delta[-1].tolist(),
        "final_treatment_minus_reference": treatment_delta[-1].tolist(),
    }


def build_habit_agx_branch_diagnostic(
    *,
    reference_root: str | Path,
    repeat_root: str | Path,
    treatment_root: str | Path,
    definition_root: str | Path,
    output_root: str | Path,
    current_git: Mapping[str, Any],
    terminal_window_policy_ticks: int = 20,
) -> dict[str, Any]:
    """Build an immutable, non-promotable paired branch diagnostic."""

    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable branch output exists: {destination}")
    records = {
        "reference": _load_run(reference_root),
        "repeat": _load_run(repeat_root),
        "treatment": _load_run(treatment_root),
    }
    pairing = _validate_pairing(records)
    definition = Path(definition_root).resolve(strict=True)
    boundary_path = definition / "dig_ready_boundary_audit_v1.json"
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    sector = boundary["sector_thresholds"]
    numeric = boundary["numeric_thresholds"]
    deadzone = float(numeric["action_deadzone"])
    takeover_tick = int(pairing["takeover_tick"])
    qpos_effects = compute_branch_effects(
        records["reference"]["qpos"],
        records["repeat"]["qpos"],
        records["treatment"]["qpos"],
        takeover_tick=takeover_tick,
    )
    action_effects = compute_branch_effects(
        records["reference"]["action"],
        records["repeat"]["action"],
        records["treatment"]["action"],
        takeover_tick=takeover_tick,
    )
    terminal = {
        role: _terminal_diagnostic(
            record,
            target_sector=str(record["target_sector"]),
            boundaries=sector["boundaries_low_to_high"],
            review_margin=float(sector["boundary_review_margin"]),
            action_deadzone=deadzone,
            window=int(terminal_window_policy_ticks),
        )
        for role, record in records.items()
    }
    signal_detected = bool(
        qpos_effects["treatment_exceeds_repeat_variability_all_axes"]
        and action_effects["treatment_exceeds_repeat_variability_all_axes"]
    )
    endpoint_success = bool(
        terminal["reference"]["numeric_endpoint_held"]
        and terminal["repeat"]["numeric_endpoint_held"]
        and terminal["treatment"]["numeric_endpoint_held"]
    )
    result = {
        "schema": BRANCH_EVAL_SCHEMA,
        "status": "completed_paired_branch_diagnostic",
        "evidence_scope": "sim_closed_loop_diagnostic_non_promotable",
        "closed_loop_execution_after_shared_prefix": True,
        "formal_gate_result": False,
        "task_success_claimed": False,
        "real_control_candidate": False,
        "held_out_test_read": False,
        "pairing_contract": pairing,
        "axis_order": {
            "qpos": list(SOURCE_QPOS_ORDER),
            "action": list(SOURCE_ACTION_ORDER),
        },
        "condition_response": {
            "qpos": qpos_effects,
            "action": action_effects,
            "condition_changes_rollout_above_repeat_variability": signal_detected,
            "interpretation": (
                "condition_signal_detected_but_not_converted_to_target_endpoint"
                if signal_detected and not endpoint_success
                else "condition_signal_and_numeric_endpoint_observed"
                if signal_detected
                else "condition_response_not_separated_from_repeat_variability"
            ),
        },
        "terminal_execution": terminal,
        "numeric_endpoint_success_all_branches": endpoint_success,
        "observable_cycle_completed": False,
        "observable_cycle_completed_reason": (
            "visual_dig_ready_confirmation_not_run_and_numeric_endpoint_not_held"
            if not endpoint_success
            else "visual_dig_ready_confirmation_not_run"
        ),
        "physical_effect_validated": False,
        "interpretation_lock": {
            "can_state": [
                "shared_prefix_reached_one_causal_dump_end_branch_state",
                "condition_changed_post_commit_actions_and_observations",
                "numeric_target_endpoint_hold_was_not_achieved",
            ],
            "cannot_state": [
                "observable_cycle_success",
                "physical_excavation_success",
                "held_out_generalization",
                "real_machine_transfer",
                "deployment_readiness",
            ],
        },
    }
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"branch temporary path exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        identities = [write_json(temporary / "paired_branch_diagnostic.json", result)]
        manifest = {
            "schema": "simverify_habit_agx_branch_manifest_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "current_git": dict(current_git),
            "definition": {
                "root": str(definition),
                "dig_ready_boundary_audit_v1": artifact_identity(boundary_path),
            },
            "inputs": {
                role: record["identity"] for role, record in records.items()
            },
            "parameters": {
                "terminal_window_policy_ticks": int(terminal_window_policy_ticks),
                "action_deadzone": deadzone,
                "sector_boundaries": list(
                    map(float, sector["boundaries_low_to_high"])
                ),
                "sector_review_margin": float(sector["boundary_review_margin"]),
            },
        }
        identities.append(write_json(temporary / "branch_manifest.json", manifest))
        write_checksums(
            temporary,
            identities,
            path=temporary / "checksums.sha256",
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def _load_run(root_value: str | Path) -> dict[str, Any]:
    root = Path(root_value).resolve(strict=True)
    verification = verify_checksums(root, root / "checksums.sha256")
    if not verification["ok"]:
        raise ValueError(f"run checksum verification failed: {root}")
    manifest_path = root / "run_manifest.json"
    ticks_path = root / "policy_ticks.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in ticks_path.read_text(encoding="utf-8").splitlines()
    ]
    if (
        manifest.get("status") != "completed_bounded_diagnostic"
        or manifest.get("closed_loop_execution") is not True
        or manifest.get("task_success_claimed") is not False
        or manifest.get("bundle_contract", {}).get("baseline_id") != "B1"
    ):
        raise ValueError(f"run contract mismatch: {root}")
    prefix = manifest.get("shared_action_prefix_contract", {})
    condition = manifest.get("condition_contract", {})
    if prefix.get("enabled") is not True or condition.get("committed") is not True:
        raise ValueError(f"run lacks committed shared-prefix branch: {root}")
    qpos = np.asarray([row["qpos"] for row in rows], dtype=np.float64)
    action = np.asarray(
        [row["actual_sent_action"] for row in rows],
        dtype=np.float64,
    )
    qvel = np.asarray([row["qvel"] for row in rows], dtype=np.float64)
    if qpos.shape != action.shape or qpos.shape != qvel.shape or qpos.shape[1:] != (4,):
        raise ValueError(f"run arrays require shape (T,4): {root}")
    intervention = manifest["test_intent"]["intervention"]
    return {
        "root": str(root),
        "manifest": manifest,
        "rows": rows,
        "qpos": qpos,
        "qvel": qvel,
        "action": action,
        "current_sector": str(intervention["current_sector"]),
        "target_sector": str(intervention["next_sector"]),
        "seed": int(intervention["seed"]),
        "identity": {
            "root": str(root),
            "run_manifest_sha256": sha256_file(manifest_path),
            "policy_ticks_sha256": sha256_file(ticks_path),
            "checksums_sha256": sha256_file(root / "checksums.sha256"),
            "verified_file_count": int(verification["verified_file_count"]),
        },
    }


def _validate_pairing(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    reference = records["reference"]
    repeat = records["repeat"]
    treatment = records["treatment"]
    manifests = [record["manifest"] for record in records.values()]
    prefix_contracts = [
        manifest["shared_action_prefix_contract"] for manifest in manifests
    ]
    prefix_counts = {
        int(contract["policy_tick_count"]) for contract in prefix_contracts
    }
    prefix_shas = {
        str(contract["provenance"]["source_policy_ticks"]["sha256"])
        for contract in prefix_contracts
    }
    commits = {
        int(manifest["condition_contract"]["commit_policy_tick"])
        for manifest in manifests
    }
    real_stack_commits = {
        str(manifest["provenance"]["current_repo"]["commit"])
        for manifest in manifests
    }
    checkpoints = {
        str(
            manifest["bundle_contract"]["artifacts"]["policy_best.ckpt"]["sha256"]
        )
        for manifest in manifests
    }
    if not (
        len(prefix_counts)
        == len(prefix_shas)
        == len(commits)
        == len(real_stack_commits)
        == len(checkpoints)
        == 1
    ):
        raise ValueError("paired branches do not share one frozen provenance")
    takeover = next(iter(prefix_counts))
    commit = next(iter(commits))
    if takeover != commit:
        raise ValueError("shared prefix must end exactly at condition commit")
    if (
        reference["seed"] != repeat["seed"]
        or reference["seed"] != treatment["seed"]
        or reference["current_sector"] != repeat["current_sector"]
        or reference["current_sector"] != treatment["current_sector"]
        or reference["target_sector"] != repeat["target_sector"]
        or treatment["target_sector"] == reference["target_sector"]
    ):
        raise ValueError("paired branch role contract mismatch")
    lengths = {record["qpos"].shape[0] for record in records.values()}
    if len(lengths) != 1:
        raise ValueError("paired branches must use one bounded horizon")
    return {
        "environment_seed": int(reference["seed"]),
        "policy_seed": int(
            reference["manifest"]["inference_reproducibility"]["policy_seed"]
        ),
        "current_sector": str(reference["current_sector"]),
        "reference_target": str(reference["target_sector"]),
        "repeat_target": str(repeat["target_sector"]),
        "treatment_target": str(treatment["target_sector"]),
        "takeover_tick": int(takeover),
        "condition_commit_tick": int(commit),
        "shared_prefix_policy_ticks_sha256": next(iter(prefix_shas)),
        "real_stack_commit": next(iter(real_stack_commits)),
        "checkpoint_sha256": next(iter(checkpoints)),
        "policy_tick_count": next(iter(lengths)),
    }


def _terminal_diagnostic(
    record: Mapping[str, Any],
    *,
    target_sector: str,
    boundaries: list[float],
    review_margin: float,
    action_deadzone: float,
    window: int,
) -> dict[str, Any]:
    if int(window) <= 0 or int(window) > record["qpos"].shape[0]:
        raise ValueError("invalid terminal window")
    qpos = record["qpos"]
    qvel = record["qvel"]
    action = record["action"]
    final_sector = classify_swing_sector(
        float(qpos[-1, 0]),
        boundaries=boundaries,
        review_margin=review_margin,
    )
    terminal_swing_action = float(np.mean(np.abs(action[-window:, 0])))
    terminal_swing_qvel = float(np.mean(np.abs(qvel[-window:, 0])))
    target_match = final_sector == target_sector
    quiet = (
        terminal_swing_action <= float(action_deadzone)
        and terminal_swing_qvel <= float(action_deadzone)
    )
    return {
        "target_sector": target_sector,
        "final_sector": final_sector,
        "final_qpos": qpos[-1].tolist(),
        "terminal_window_policy_ticks": int(window),
        "terminal_mean_abs_swing_action": terminal_swing_action,
        "terminal_mean_abs_swing_qvel": terminal_swing_qvel,
        "target_sector_terminal_match": bool(target_match),
        "terminal_swing_quiet": bool(quiet),
        "numeric_endpoint_held": bool(target_match and quiet),
        "commanded_motion_not_realized_terminally": bool(
            terminal_swing_action > float(action_deadzone)
            and terminal_swing_qvel <= float(action_deadzone)
        ),
        "visual_dig_ready_confirmed": False,
    }
