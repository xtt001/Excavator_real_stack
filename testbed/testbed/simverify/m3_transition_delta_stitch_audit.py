"""Reproducible stability audit of the G4-v3 completion-step Gate."""

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

EVIDENCE_SCOPE = "recorded-observation/offline development"
EXPECTED_V3_FAILURE = "validation_completion_steps"


def build_transition_delta_stitch_gate_audit(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    delta_stitch_root: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Audit v3 without mutating or replacing its immutable fail result."""

    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        not git.get("git_available")
        or git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
    ):
        raise ValueError("delta-stitch audit requires a clean SimVerify worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable delta-stitch audit exists: {destination}")
    source = Path(delta_stitch_root).resolve(strict=True)
    contract = Path(contract_path).resolve(strict=True)
    verification = verify_checksums(source, source / "checksums.sha256")
    if not verification["ok"]:
        raise ValueError("delta-stitch source checksum verification failed")

    gate = _read_json(source / "delta_stitch_gate_v1.json")
    if gate["decision"] != "offline_emulator_invalid_v3":
        raise ValueError("audit requires the immutable v3 fail decision")
    failed = {
        name
        for name, criterion in gate["delta_stitch"]["criteria"].items()
        if not bool(criterion["passed"])
    }
    if failed != {EXPECTED_V3_FAILURE}:
        raise ValueError(f"audit cannot waive non-step v3 failures: {sorted(failed)}")

    train = _read_jsonl(source / "train_source_episode_metrics.jsonl")
    validation = _read_jsonl(source / "validation_source_episode_metrics.jsonl")
    nested = nested_train_step_envelope_audit(train)
    criteria = development_prerequisite_criteria(
        train,
        validation,
        support_radius=float(gate["support_radius"]),
        maximum_steps=int(gate["maximum_steps"]),
    )
    passed = all(bool(row["passed"]) for row in criteria.values())
    decision = (
        "pass_expert_delta_stitch_development_prerequisite_v3_1"
        if passed
        else "offline_emulator_invalid_v3_1"
    )
    audit_gate = {
        "schema": "simverify_transition_delta_stitch_gate_audit_v1",
        "decision": decision,
        "authorizes_b1_4_policy_stitch_development": passed,
        "authorizes_independent_validation_claim": False,
        "v3_decision_preserved": gate["decision"],
        "v3_failed_criterion_preserved": EXPECTED_V3_FAILURE,
        "v3_failed_criterion_role": "diagnostic_not_hard_support_gate",
        "nested_train_step_envelope_audit": nested,
        "development_prerequisite": {
            "criteria": criteria,
            "passed": passed,
        },
        "evidence_scope": EVIDENCE_SCOPE,
        "validation_role": "development_reused_after_v3",
        "held_out_test_read": False,
        "closed_loop_execution": False,
    }

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    try:
        gate_identity = write_json(
            temporary / "delta_stitch_gate_audit_v1.json",
            audit_gate,
        )
        identities.append(gate_identity)
        manifest_identity = write_json(
            temporary / "delta_stitch_gate_audit_manifest.json",
            {
                "schema": "simverify_transition_delta_stitch_gate_audit_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "contract": {
                    "path": str(contract),
                    "sha256": sha256_file(contract),
                },
                "source_delta_stitch_package": {
                    "path": str(source),
                    "manifest_sha256": sha256_file(
                        source / "delta_stitch_manifest.json"
                    ),
                    "gate_sha256": sha256_file(source / "delta_stitch_gate_v1.json"),
                    "checksums_sha256": sha256_file(source / "checksums.sha256"),
                    "verified_file_count": verification["verified_file_count"],
                },
                "method_change": "gate_role_only; retrieval and rollout unchanged",
                "independent_validation": False,
                "validation_role": "development_reused_after_v3",
                "decision": decision,
                "evidence_scope": EVIDENCE_SCOPE,
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
        )
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
            "decision": decision,
            "authorizes_b1_4_policy_stitch_development": passed,
            "manifest_sha256": manifest_identity["sha256"],
            "gate_sha256": gate_identity["sha256"],
            "checksums_sha256": checksums_identity["sha256"],
            "held_out_test_read": False,
            "closed_loop_execution": False,
        }
    except BaseException as exc:
        if temporary.exists():
            write_json(
                temporary / "BUILD_FAILED.json",
                {
                    "schema": "simverify_transition_delta_stitch_audit_failure_v1",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "git": git,
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                },
            )
        raise


def nested_train_step_envelope_audit(
    train_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure false rejection of the old q97.5 envelope on train episodes."""

    failures = []
    for row in train_rows:
        others = [
            float(other["candidate_completed_steps_q97_5"])
            for other in train_rows
            if int(other["episode_id"]) != int(row["episode_id"])
        ]
        if not others:
            raise ValueError("nested audit requires multiple train episodes")
        threshold = float(np.quantile(others, 0.975))
        observed = float(row["candidate_completed_steps_q97_5"])
        failures.append(
            {
                "episode_id": int(row["episode_id"]),
                "observed": observed,
                "leave_one_episode_out_upper": threshold,
                "passed": observed <= threshold,
            }
        )
    failure_count = sum(not bool(row["passed"]) for row in failures)
    return {
        "schema": "simverify_nested_train_step_envelope_audit_v1",
        "rows": failures,
        "failure_count": failure_count,
        "source_episode_count": len(failures),
        "stable_as_zero_false_rejection_hard_gate": failure_count == 0,
    }


def development_prerequisite_criteria(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    support_radius: float,
    maximum_steps: int,
) -> dict[str, Any]:
    """Evaluate only capability and support criteria, not speed similarity."""

    combined = list(train_rows) + list(validation_rows)
    max_distance = max(
        float(row["candidate_max_retrieval_distance"]) for row in combined
    )
    max_observed_steps = max(
        float(row["candidate_completed_steps_q97_5"]) for row in combined
    )
    return {
        "candidate_completion_all_source_episodes": {
            "observed_min": min(
                float(row["candidate_completion_rate"]) for row in combined
            ),
            "required": 1.0,
            "passed": all(
                float(row["candidate_completion_rate"]) == 1.0 for row in combined
            ),
        },
        "median_action_null_incomplete_all_source_episodes": {
            "observed_max": max(
                float(row["median_action_null_completion_rate"]) for row in combined
            ),
            "required": 0.0,
            "passed": all(
                float(row["median_action_null_completion_rate"]) == 0.0
                for row in combined
            ),
        },
        "paired_action_dependence_all_source_episodes": {
            "observed_min": min(
                float(row["paired_completion_delta"]) for row in combined
            ),
            "required": 1.0,
            "passed": all(
                float(row["paired_completion_delta"]) == 1.0 for row in combined
            ),
        },
        "retrieval_inside_inherited_support": {
            "observed_max": max_distance,
            "maximum_allowed": support_radius,
            "passed": max_distance <= support_radius,
        },
        "completion_inside_frozen_rollout_budget": {
            "observed_max_episode_q97_5": max_observed_steps,
            "maximum_allowed": maximum_steps,
            "passed": max_observed_steps <= maximum_steps,
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
