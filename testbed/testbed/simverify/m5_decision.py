"""Immutable terminal M5 decision for the frozen SimVerify experiment."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from testbed.simverify.artifacts import (
    verify_checksums,
    write_checksums,
    write_json,
)
from testbed.simverify.contracts import git_provenance, sha256_file

EVIDENCE_SCOPE = "recorded-observation/offline"
HELD_OUT_EPISODES = {1, 13, 25, 33}
EXPECTED_CONDITION_BASELINES = {"B1", "B1.1", "B1.2"}


def build_m5_decision(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    m0_root: str | Path,
    m1_report_path: str | Path,
    m2_root: str | Path,
    g3_root: str | Path,
    condition_gate_roots: Sequence[str | Path],
) -> dict[str, Any]:
    """Validate the complete evidence chain and write a terminal M5 package."""

    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
        or not git.get("git_available")
    ):
        raise ValueError("M5 decision requires a clean SimVerify worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable M5 output exists: {destination}")

    m0 = _validate_m0(Path(m0_root).resolve(strict=True))
    m1 = _validate_m1(
        Path(m1_report_path).resolve(strict=True),
        m0_manifest_sha256=m0["manifest_sha256"],
    )
    m2 = _validate_m2(
        Path(m2_root).resolve(strict=True),
        m0_manifest_sha256=m0["manifest_sha256"],
        m1_report_sha256=m1["report_sha256"],
    )
    replay_cache: dict[tuple[Path, str], dict[str, Any]] = {}
    g3 = _validate_g3(
        Path(g3_root).resolve(strict=True),
        m0_manifest_sha256=m0["manifest_sha256"],
        m2_manifest_sha256=m2["manifest_sha256"],
        m2_root=m2["root"],
        replay_cache=replay_cache,
    )
    condition_gates = [
        _validate_condition_gate(
            Path(path).resolve(strict=True),
            m0_manifest_sha256=m0["manifest_sha256"],
            m2_manifest_sha256=m2["manifest_sha256"],
            replay_cache=replay_cache,
        )
        for path in condition_gate_roots
    ]
    baselines = {item["baseline_id"] for item in condition_gates}
    if baselines != EXPECTED_CONDITION_BASELINES:
        raise ValueError(
            "M5 requires B1, B1.1, and B1.2 condition Gate packages; "
            f"got {sorted(baselines)}"
        )
    condition_gates.sort(key=lambda item: _baseline_order(item["baseline_id"]))
    if any(item["decision"] != "condition_understanding_not_established" for item in condition_gates):
        raise ValueError("M5 revise_condition requires every supplied G4 revision to fail")

    decision = _decision_payload(
        m0=m0,
        m1=m1,
        m2=m2,
        g3=g3,
        condition_gates=condition_gates,
    )
    manifest = {
        "schema": "simverify_m5_manifest_v2",
        "stage": "M5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "decision": "revise_condition",
        "terminal_for_experiment_version": True,
        "evidence_scope": EVIDENCE_SCOPE,
        "closed_loop_execution": False,
        "held_out_test_read": False,
        "held_out_test_authorized": False,
        "control_candidate": False,
        "inputs": {
            "m0": m0["identity"],
            "m1": m1["identity"],
            "m2": m2["identity"],
            "g3": g3["identity"],
            "condition_gates": [item["identity"] for item in condition_gates],
        },
    }

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    try:
        decision_identity = write_json(temporary / "decision.json", decision)
        identities.append(decision_identity)
        manifest_identity = write_json(temporary / "m5_manifest.json", manifest)
        identities.append(manifest_identity)
        checksums_identity = write_checksums(
            temporary,
            identities,
            path=temporary / "checksums.sha256",
        )
        os.rename(temporary, destination)
        return {
            "status": "completed",
            "output_root": str(destination),
            "decision": "revise_condition",
            "terminal_for_experiment_version": True,
            "decision_sha256": decision_identity["sha256"],
            "manifest_sha256": manifest_identity["sha256"],
            "checksums_sha256": checksums_identity["sha256"],
            "held_out_test_read": False,
            "closed_loop_execution": False,
            "control_candidate": False,
        }
    except BaseException as exc:
        if temporary.exists():
            write_json(
                temporary / "BUILD_FAILED.json",
                {
                    "schema": "simverify_m5_build_failure_v2",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                    "git": git,
                },
            )
        raise


def _decision_payload(
    *,
    m0: Mapping[str, Any],
    m1: Mapping[str, Any],
    m2: Mapping[str, Any],
    g3: Mapping[str, Any],
    condition_gates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "simverify_m5_decision_v2",
        "stage": "M5",
        "decision": "revise_condition",
        "terminal_for_experiment_version": True,
        "evidence_scope": EVIDENCE_SCOPE,
        "closed_loop_execution": False,
        "held_out_test_read": False,
        "held_out_test_authorized": False,
        "gate_thresholds_v1_generated": False,
        "control_candidate": False,
        "sim_observable_only": False,
        "real_finetune_candidate": False,
        "real_control_allowed": False,
        "gate_path": [
            {
                "gate": "G0",
                "result": "pass",
                "evidence": m0["identity"],
            },
            {
                "gate": "G1",
                "result": "pass",
                "evidence": m0["identity"],
            },
            {
                "gate": "G2",
                "result": "pass",
                "evidence": m0["identity"],
            },
            {
                "gate": "M1",
                "result": "pass",
                "evidence": m1["identity"],
            },
            {
                "gate": "M2",
                "result": "pass",
                "evidence": m2["identity"],
            },
            {
                "gate": "G3",
                "result": "pass_recorded_observation_baseline",
                "evidence": g3["identity"],
            },
            {
                "gate": "G4",
                "result": "revise_condition",
                "evidence": [item["identity"] for item in condition_gates],
            },
            {
                "gate": "G5",
                "result": "not_entered",
                "reason": "G4 prerequisite did not pass",
            },
            {
                "gate": "G6",
                "result": "not_entered",
                "reason": "no promotable candidate exists",
            },
        ],
        "condition_revision_history": [
            {
                "baseline_id": item["baseline_id"],
                "decision": item["decision"],
                "factor_pass": item["factor_pass"],
                "source_episode_summary": item["source_episode_summary"],
                "identity": item["identity"],
            }
            for item in condition_gates
        ],
        "terminal_reason": (
            "B1, B1.1, and B1.2 all failed the frozen fixed-observation "
            "condition-understanding criteria; B1.2 improved next-sector phase "
            "specificity but nearly erased its semantic response"
        ),
        "next_authorized_action": (
            "freeze a new one-factor condition representation or routing contract"
        ),
        "forbidden_next_actions": [
            "read held-out test",
            "enter G5 or G6",
            "claim simulator or real closed-loop success",
            "promote any sim checkpoint to shadow or control",
            "tune B1.2 coefficient after observing validation",
        ],
        "interpretation_guard": (
            "offline action sensitivity is not proof of closed-loop task completion"
        ),
    }


def _validate_m0(root: Path) -> dict[str, Any]:
    inventory = _verified_root(
        root,
        required={"dataset_manifest.json", "m0_authorization_report_v2.json"},
    )
    manifest_path = root / "dataset_manifest.json"
    manifest = _read_json(manifest_path)
    authorization = _read_json(root / "m0_authorization_report_v2.json")
    if (
        manifest.get("schema_version") != "sim_observable_cycle_export_v1"
        or manifest.get("stage") != "M0"
        or manifest.get("evidence_scope") != EVIDENCE_SCOPE
        or manifest.get("held_out_test_authorized") is not False
        or manifest.get("oracle_dependency") is not False
        or manifest.get("oracle_audit_referenced_by_main_artifacts") is not False
        or manifest.get("control_candidate") is not False
        or manifest.get("real_deployable") is not False
        or authorization.get("gate_preconditions_passed") is not True
        or authorization.get("m1_import_smoke_authorized_after_immutable_finalize")
        is not True
    ):
        raise ValueError("invalid M0 terminal evidence")
    manifest_sha256 = sha256_file(manifest_path)
    return {
        "root": root,
        "manifest_sha256": manifest_sha256,
        "identity": {
            **inventory,
            "manifest_sha256": manifest_sha256,
            "gate_preconditions_passed": True,
        },
    }


def _validate_m1(path: Path, *, m0_manifest_sha256: str) -> dict[str, Any]:
    report = _read_json(path)
    report_sha256 = sha256_file(path)
    if (
        report.get("schema") != "simverify_m1_import_smoke_v1"
        or report.get("stage") != "M1"
        or report.get("passed") is not True
        or report.get("m2_authorized") is not True
        or report.get("evidence_scope") != EVIDENCE_SCOPE
        or report.get("training_started") is not False
        or report.get("training_authorized") is not False
        or report.get("held_out_test_read") is not False
        or report.get("closed_loop_execution") is not False
        or report.get("package_dataset_manifest_sha256") != m0_manifest_sha256
    ):
        raise ValueError("invalid M1 import-smoke evidence")
    return {
        "report_sha256": report_sha256,
        "identity": {
            "path": str(path),
            "sha256": report_sha256,
            "size_bytes": int(path.stat().st_size),
            "passed": True,
        },
    }


def _validate_m2(
    root: Path,
    *,
    m0_manifest_sha256: str,
    m1_report_sha256: str,
) -> dict[str, Any]:
    inventory = _verified_root(
        root,
        required={
            "m2_manifest.json",
            "m2_authorization_report_v1.json",
            "expert_event_envelope_v1.json",
        },
    )
    manifest_path = root / "m2_manifest.json"
    manifest = _read_json(manifest_path)
    authorization = _read_json(root / "m2_authorization_report_v1.json")
    provenance = manifest.get("provenance", {})
    if (
        manifest.get("schema") != "simverify_m2_offline_eval_contract_v1"
        or manifest.get("stage") != "M2"
        or manifest.get("evidence_scope") != EVIDENCE_SCOPE
        or manifest.get("training_started") is not False
        or manifest.get("held_out_test_authorized") is not False
        or manifest.get("gate_thresholds_v1_status") != "not_generated"
        or manifest.get("source_splits") != ["train", "validation"]
        or provenance.get("held_out_episode_access_count") != 0
        or provenance.get("m0_dataset_manifest_sha256") != m0_manifest_sha256
        or provenance.get("m1_report_sha256") != m1_report_sha256
        or authorization.get("passed") is not True
        or authorization.get("m3_unconditioned_baseline_authorized") is not True
        or authorization.get("held_out_test_read") is not False
        or authorization.get("closed_loop_execution") is not False
    ):
        raise ValueError("invalid M2 evaluator evidence")
    manifest_sha256 = sha256_file(manifest_path)
    return {
        "root": root,
        "manifest_sha256": manifest_sha256,
        "identity": {
            **inventory,
            "manifest_sha256": manifest_sha256,
            "authorization_passed": True,
        },
    }


def _validate_g3(
    root: Path,
    *,
    m0_manifest_sha256: str,
    m2_manifest_sha256: str,
    m2_root: Path,
    replay_cache: dict[tuple[Path, str], dict[str, Any]],
) -> dict[str, Any]:
    inventory = _verified_root(
        root,
        required={"g3_manifest.json", "g3_calibration_v1.json"},
    )
    manifest_path = root / "g3_manifest.json"
    manifest = _read_json(manifest_path)
    calibration = _read_json(root / "g3_calibration_v1.json")
    if (
        manifest.get("schema") != "simverify_g3_manifest_v1"
        or manifest.get("decision") != "pass_recorded_observation_baseline"
        or set(manifest.get("authorizes", [])) != {"B1", "B2"}
        or manifest.get("evidence_scope") != EVIDENCE_SCOPE
        or manifest.get("held_out_test_read") is not False
        or manifest.get("closed_loop_execution") is not False
        or manifest.get("gate_thresholds_v1_generated") is not False
        or calibration.get("decision") != "pass_recorded_observation_baseline"
        or calibration.get("held_out_test_read") is not False
        or calibration.get("closed_loop_execution") is not False
        or manifest.get("expert_event_envelope_sha256")
        != sha256_file(m2_root / "expert_event_envelope_v1.json")
    ):
        raise ValueError("invalid G3 baseline evidence")
    replay_records = manifest.get("replay_packages", [])
    if len(replay_records) < 4:
        raise ValueError("G3 evidence requires train plus three validation replays")
    for record in replay_records:
        _validate_replay_reference(
            record,
            manifest_name="replay_manifest.json",
            m0_manifest_sha256=m0_manifest_sha256,
            m2_manifest_sha256=m2_manifest_sha256,
            replay_cache=replay_cache,
        )
    manifest_sha256 = sha256_file(manifest_path)
    return {
        "identity": {
            **inventory,
            "manifest_sha256": manifest_sha256,
            "decision": manifest["decision"],
        }
    }


def _validate_condition_gate(
    root: Path,
    *,
    m0_manifest_sha256: str,
    m2_manifest_sha256: str,
    replay_cache: dict[tuple[Path, str], dict[str, Any]],
) -> dict[str, Any]:
    inventory = _verified_root(
        root,
        required={
            "condition_causal_manifest.json",
            "condition_causal_gate_v2.json",
            "repeat_noise_v2.json",
            "semantic_permutations_v1.json",
            "source_episode_causal_metrics.jsonl",
        },
    )
    manifest_path = root / "condition_causal_manifest.json"
    gate_path = root / "condition_causal_gate_v2.json"
    manifest = _read_json(manifest_path)
    gate = _read_json(gate_path)
    replay_records = manifest.get("condition_replay_packages", [])
    candidate_replay_baselines = {
        str(record["baseline_id"])
        for record in replay_records
        if str(record["baseline_id"]) != "B2"
    }
    declared_baseline = manifest.get("candidate_baseline_id")
    if declared_baseline is None:
        if candidate_replay_baselines != {"B1"}:
            raise ValueError(
                "legacy condition Gate without candidate_baseline_id must "
                "resolve uniquely to B1"
            )
        baseline_id = "B1"
    else:
        baseline_id = str(declared_baseline)
    gate_baseline = gate.get("candidate_baseline_id")
    if (
        manifest.get("schema") != "simverify_condition_causal_manifest_v2"
        or baseline_id not in EXPECTED_CONDITION_BASELINES
        or candidate_replay_baselines != {baseline_id}
        or manifest.get("decision") != "condition_understanding_not_established"
        or manifest.get("evidence_scope") != EVIDENCE_SCOPE
        or manifest.get("held_out_test_read") is not False
        or manifest.get("closed_loop_execution") is not False
        or gate.get("schema") != "simverify_condition_causal_gate_v2"
        or (
            gate_baseline is None
            and declared_baseline is not None
        )
        or (
            gate_baseline is not None
            and str(gate_baseline) != baseline_id
        )
        or gate.get("decision") != "condition_understanding_not_established"
        or gate.get("recommended_terminal_status") != "revise_condition"
        or gate.get("condition_understanding_established") is not False
        or gate.get("held_out_test_read") is not False
        or gate.get("closed_loop_execution") is not False
    ):
        raise ValueError(f"invalid {baseline_id} condition Gate evidence")
    if len(replay_records) < 5:
        raise ValueError(f"{baseline_id} Gate requires candidate repeats, B2, and mask")
    for record in replay_records:
        _validate_replay_reference(
            record,
            manifest_name="condition_replay_manifest.json",
            m0_manifest_sha256=m0_manifest_sha256,
            m2_manifest_sha256=m2_manifest_sha256,
            replay_cache=replay_cache,
        )
    manifest_sha256 = sha256_file(manifest_path)
    return {
        "baseline_id": baseline_id,
        "decision": gate["decision"],
        "factor_pass": dict(gate["factor_pass"]),
        "source_episode_summary": _condition_summary(gate),
        "identity": {
            **inventory,
            "manifest_sha256": manifest_sha256,
            "gate_sha256": sha256_file(gate_path),
            "baseline_id": baseline_id,
            "decision": gate["decision"],
        },
    }


def _validate_replay_reference(
    record: Mapping[str, Any],
    *,
    manifest_name: str,
    m0_manifest_sha256: str,
    m2_manifest_sha256: str,
    replay_cache: dict[tuple[Path, str], dict[str, Any]],
) -> dict[str, Any]:
    root = Path(str(record["path"])).resolve(strict=True)
    key = (root, manifest_name)
    cached = replay_cache.get(key)
    if cached is None:
        verification = verify_checksums(root, root / "checksums.sha256")
        if not verification["ok"]:
            raise ValueError(f"replay checksum verification failed: {root}")
        manifest_path = root / manifest_name
        manifest = _read_json(manifest_path)
        provenance = manifest.get("provenance", {})
        episode_ids = {int(value) for value in manifest.get("episode_ids", [])}
        if (
            manifest.get("evidence_scope", provenance.get("evidence_scope"))
            != EVIDENCE_SCOPE
            or manifest.get("held_out_test_read") is not False
            or manifest.get("closed_loop_execution") is not False
            or episode_ids & HELD_OUT_EPISODES
            or manifest.get(
                "m0_dataset_manifest_sha256",
                provenance.get("m0_dataset_manifest_sha256"),
            )
            != m0_manifest_sha256
            or manifest.get(
                "m2_manifest_sha256",
                provenance.get("m2_manifest_sha256"),
            )
            != m2_manifest_sha256
        ):
            raise ValueError(f"replay violates M5 evidence boundary: {root}")
        cached = {
            "manifest_sha256": sha256_file(manifest_path),
            "checksums_sha256": sha256_file(root / "checksums.sha256"),
        }
        replay_cache[key] = cached
    if (
        record.get("manifest_sha256") != cached["manifest_sha256"]
        or record.get("checksums_sha256") != cached["checksums_sha256"]
    ):
        raise ValueError(f"replay identity mismatch: {root}")
    return cached


def _condition_summary(gate: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for factor in ("current_sector", "next_sector"):
        criteria = gate["criteria"][factor]
        permutations = criteria["semantic_identifiability"]["permutation_results"]
        result[factor] = {
            "action_effect_mean": _mean(
                criteria["action_sensitivity_vs_masked"]["b1_source_episode_values"]
            ),
            "signed_semantic_margin_mean": _mean(
                criteria["signed_semantic_margin_vs_b2"][
                    "b1_source_episode_values"
                ]
            ),
            "phase_specificity_mean": _mean(
                criteria["phase_specificity"]["positive_vs_masked"][
                    "b1_source_episode_values"
                ]
            ),
            "semantic_permutations_rejected": int(
                sum(bool(item["passed"]) for item in permutations.values())
            ),
            "semantic_permutation_count": int(len(permutations)),
            "factor_pass": bool(gate["factor_pass"][factor]),
        }
    return result


def _verified_root(root: Path, *, required: set[str]) -> dict[str, Any]:
    checksum_path = root / "checksums.sha256"
    verification = verify_checksums(root, checksum_path)
    if not verification["ok"]:
        raise ValueError(f"checksum verification failed: {root}")
    names = {
        line.split("  ", 1)[1]
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    missing = required - names
    if missing:
        raise ValueError(f"checksum inventory missing {sorted(missing)} in {root}")
    return {
        "path": str(root),
        "checksums_sha256": sha256_file(checksum_path),
        "verified_file_count": int(verification["verified_file_count"]),
    }


def _baseline_order(baseline_id: str) -> int:
    return {"B1": 0, "B1.1": 1, "B1.2": 2}[baseline_id]


def _mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("source-episode values must be finite and non-empty")
    return float(np.mean(array))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
