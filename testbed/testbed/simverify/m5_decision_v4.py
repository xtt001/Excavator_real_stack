"""Terminal M5 decision for the B1.5/G4/G5.1/E04 experiment version."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from testbed.simverify.artifacts import write_checksums, write_json
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.m5_decision_v3 import _verified_package

EVIDENCE_SCOPE = "recorded-observation/offline"
HELD_OUT_EPISODES = {1, 13, 25, 33}
CANDIDATE_BASELINE_ID = "B1.5"
NULL_BASELINE_ID = "B2.5"
EXPERIMENT_VERSION = "B1.5_G4_G5.1_E04"


def build_m5_decision_v4(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    contract_path: str | Path,
    prior_m5_root: str | Path,
    g4_root: str | Path,
    g5_v1_root: str | Path,
    g5_1_root: str | Path,
    e04_root: str | Path,
) -> dict[str, Any]:
    """Validate the B1.5 evidence chain and write an immutable M5 package."""

    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        not git.get("git_available")
        or git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
    ):
        raise ValueError("M5 v4 requires a clean v2.0.0-simVerify worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable M5 v4 output exists: {destination}")
    contract = Path(contract_path).resolve(strict=True)

    prior_m5 = _verified_package(
        Path(prior_m5_root).resolve(strict=True),
        manifest_name="m5_manifest.json",
        gate_name="decision.json",
    )
    g4 = _verified_package(
        Path(g4_root).resolve(strict=True),
        manifest_name="next_condition_causal_manifest.json",
        gate_name="next_condition_causal_gate_v1.json",
    )
    g5_v1 = _verified_package(
        Path(g5_v1_root).resolve(strict=True),
        manifest_name="g5_two_cycle_manifest.json",
        gate_name="g5_core_gate_v1.json",
    )
    g5_1 = _verified_package(
        Path(g5_1_root).resolve(strict=True),
        manifest_name="g5_two_cycle_manifest.json",
        gate_name="g5_core_gate_v1.json",
    )
    e04 = _verified_package(
        Path(e04_root).resolve(strict=True),
        manifest_name="e04_manifest.json",
        gate_name="e04_gate_v1.json",
    )
    checkpoints = validate_chain_v4(
        prior_m5=prior_m5,
        g4=g4,
        g5_v1=g5_v1,
        g5_1=g5_1,
        e04=e04,
    )

    decision = decision_payload_v4(
        prior_m5=prior_m5,
        g4=g4,
        g5_v1=g5_v1,
        g5_1=g5_1,
        e04=e04,
        checkpoints=checkpoints,
    )
    manifest = {
        "schema": "simverify_m5_manifest_v4",
        "stage": "M5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "contract": {
            "path": str(contract),
            "sha256": sha256_file(contract),
        },
        "decision": "revise_condition",
        "terminal_for_experiment_version": True,
        "experiment_version": EXPERIMENT_VERSION,
        "candidate_baseline_id": CANDIDATE_BASELINE_ID,
        "null_baseline_id": NULL_BASELINE_ID,
        "candidate_checkpoint_sha256": checkpoints["candidate"],
        "null_checkpoint_sha256": checkpoints["null"],
        "evidence_scope": EVIDENCE_SCOPE,
        "closed_loop_execution": False,
        "held_out_test_read": False,
        "held_out_test_authorized": False,
        "control_candidate": False,
        "inputs": {
            "prior_m5_v3": prior_m5["identity"],
            "g4_b1_5_b2_5": g4["identity"],
            "g5_v1": g5_v1["identity"],
            "g5_1_b1_5_b2_5": g5_1["identity"],
            "e04_b1_5": e04["identity"],
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
                    "schema": "simverify_m5_build_failure_v4",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                    "git": git,
                },
            )
        raise


def decision_payload_v4(
    *,
    prior_m5: dict[str, Any],
    g4: dict[str, Any],
    g5_v1: dict[str, Any],
    g5_1: dict[str, Any],
    e04: dict[str, Any],
    checkpoints: Mapping[str, str],
) -> dict[str, Any]:
    """Materialize the only decision authorized by the verified Gate path."""

    camera_summary = _camera_summary(e04["gate"])
    return {
        "schema": "simverify_m5_decision_v4",
        "stage": "M5",
        "experiment_version": EXPERIMENT_VERSION,
        "decision": "revise_condition",
        "terminal_for_experiment_version": True,
        "candidate_baseline_id": CANDIDATE_BASELINE_ID,
        "null_baseline_id": NULL_BASELINE_ID,
        "candidate_checkpoint_sha256": checkpoints["candidate"],
        "null_checkpoint_sha256": checkpoints["null"],
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
                "gate": "G0_G1_G2_M1_M2_G3",
                "result": "pass_inherited_from_checksum_verified_M5_v3_chain",
                "evidence": prior_m5["identity"],
            },
            {
                "gate": "G4",
                "result": "next_condition_understanding_established_offline_development",
                "evidence": g4["identity"],
            },
            {
                "gate": "G5_v1",
                "result": "frozen_failed_router_lifecycle_predecessor",
                "evidence": g5_v1["identity"],
            },
            {
                "gate": "G5_1",
                "result": "core_two_cycle_continuity_established_development",
                "evidence": g5_1["identity"],
            },
            {
                "gate": "E04",
                "result": "camera_counterfactual_robustness_not_established",
                "evidence": e04["identity"],
            },
            {
                "gate": "E05",
                "result": "not_entered",
                "reason": "E04 did not authorize E05",
            },
            {
                "gate": "E06",
                "result": "not_entered",
                "reason": "E04 stop condition retained",
            },
            {
                "gate": "G6",
                "result": "not_entered",
                "reason": "complete camera robustness did not pass",
            },
        ],
        "camera_counterfactual_summary": camera_summary,
        "terminal_reason": (
            "B1.5 establishes supported next-condition semantics and full-camera "
            "two-cycle continuity, and removes the old negative video7-drop "
            "semantic direction, but the frozen targeted video7 retention "
            "requirement and the complete E04 camera robustness Gate still fail"
        ),
        "why_not_reject": (
            "M0/M1/M2 contracts, observable annotation, B0, B1.5 versus B2.5 "
            "condition causality, and full-camera G5.1 continuity are usable"
        ),
        "why_not_sim_observable_only": (
            "complete E04 failed and E05, E06, G6, and held-out test were not entered"
        ),
        "next_authorized_action": (
            "freeze a new experiment that separates condition-semantic "
            "causality, independently derived task-phase non-inferiority, and "
            "temporal-vision sensitivity diagnostics"
        ),
        "forbidden_next_actions": [
            "enter E05, E06, G6, or held-out test for this experiment version",
            "relax the completed B1.5 E04 thresholds after observing validation",
            "claim simulator or real closed-loop success",
            "promote B1.5 to real fine-tuning, shadow, control, or deployment",
            "change multiple primary training or runtime factors together",
        ],
        "interpretation_guard": (
            "recorded-observation action semantics do not prove environmental "
            "response or closed-loop excavation"
        ),
    }


def validate_chain_v4(
    *,
    prior_m5: dict[str, Any],
    g4: dict[str, Any],
    g5_v1: dict[str, Any],
    g5_1: dict[str, Any],
    e04: dict[str, Any],
) -> dict[str, str]:
    """Validate identities, decisions, checkpoint continuity, and safety locks."""

    prior = prior_m5["gate"]
    if (
        prior.get("schema") != "simverify_m5_decision_v3"
        or prior.get("experiment_version") != "B1.4_G5.1_E04"
        or prior.get("decision") != "revise_condition"
        or not prior.get("terminal_for_experiment_version")
        or prior.get("held_out_test_read")
        or prior.get("closed_loop_execution")
    ):
        raise ValueError("invalid prior M5 v3 evidence")

    if (
        g4["manifest"].get("candidate_baseline_id") != CANDIDATE_BASELINE_ID
        or g4["manifest"].get("null_baseline_id") != NULL_BASELINE_ID
        or g4["gate"].get("decision")
        != "next_condition_understanding_established_offline"
        or g4["manifest"].get("decision")
        != "next_condition_understanding_established_offline"
    ):
        raise ValueError("G4 B1.5/B2.5 next-condition Gate did not pass")

    if g5_v1["gate"].get(
        "decision"
    ) != "g5_core_two_cycle_condition_continuity_not_established" or g5_v1[
        "manifest"
    ].get("authorizes_remaining_g5_robustness"):
        raise ValueError("invalid G5 v1 failure evidence")

    if (
        g5_1["manifest"].get("candidate_baseline_id") != CANDIDATE_BASELINE_ID
        or g5_1["manifest"].get("null_baseline_id") != NULL_BASELINE_ID
        or g5_1["gate"].get("decision")
        != "g5_core_two_cycle_condition_continuity_established_development"
        or not g5_1["manifest"].get("authorizes_remaining_g5_robustness")
    ):
        raise ValueError("G5.1 B1.5/B2.5 core continuity did not pass")
    previous = g5_1["manifest"].get("previous_g5_core", {})
    if previous.get("manifest_sha256") != g5_v1["identity"]["manifest_sha256"]:
        raise ValueError("G5.1 does not bind the supplied G5 v1 artifact")

    if (
        e04["manifest"].get("candidate_baseline_id") != CANDIDATE_BASELINE_ID
        or e04["gate"].get("decision")
        != "e04_camera_counterfactual_robustness_not_established"
        or e04["manifest"].get("decision")
        != "e04_camera_counterfactual_robustness_not_established"
        or e04["manifest"].get("authorizes_e05")
    ):
        raise ValueError("E04 is not the B1.5 frozen stop decision")
    e04_previous = e04["manifest"].get("previous_g5", {})
    if e04_previous.get("manifest_sha256") != g5_1["identity"]["manifest_sha256"]:
        raise ValueError("E04 does not bind the supplied G5.1 artifact")

    g4_candidate = _g4_checkpoint(g4["manifest"], CANDIDATE_BASELINE_ID)
    g4_null = _g4_checkpoint(g4["manifest"], NULL_BASELINE_ID)
    g5_candidate = g5_1["manifest"]["bundles"][CANDIDATE_BASELINE_ID][
        "checkpoint_sha256"
    ]
    g5_null = g5_1["manifest"]["bundles"][NULL_BASELINE_ID]["checkpoint_sha256"]
    e04_candidate = e04["manifest"]["bundle"]["checkpoint_sha256"]
    if len({g4_candidate, g5_candidate, e04_candidate}) != 1:
        raise ValueError("B1.5 checkpoint differs across G4, G5.1, and E04")
    if g4_null != g5_null:
        raise ValueError("B2.5 checkpoint differs across G4 and G5.1")

    experiment = g5_1["manifest"]["bundles"][CANDIDATE_BASELINE_ID][
        "checkpoint_contract"
    ]["experiment_contract"]
    if (
        experiment.get("m5_v3_manifest_sha256")
        != prior_m5["identity"]["manifest_sha256"]
        or experiment.get("m5_v3_checksums_sha256")
        != prior_m5["identity"]["checksums_sha256"]
    ):
        raise ValueError("B1.5 checkpoint does not bind the supplied prior M5 v3")

    if set(map(int, e04["manifest"]["source_episode_ids"])) & HELD_OUT_EPISODES:
        raise ValueError("held-out episode entered E04")
    packages = (g4, g5_v1, g5_1, e04)
    if any(
        package["manifest"].get("held_out_test_read")
        or package["manifest"].get("closed_loop_execution")
        or package["gate"].get("held_out_test_read")
        or package["gate"].get("closed_loop_execution")
        for package in packages
    ):
        raise ValueError("offline-only or held-out boundary was violated")
    if any(
        bundle["checkpoint_contract"]["checkpoint_semantics"].get(
            "real_control_allowed"
        )
        for bundle in g5_1["manifest"]["bundles"].values()
    ):
        raise ValueError("M5 v4 input checkpoint permits real control")

    return {"candidate": g5_candidate, "null": g5_null}


def _g4_checkpoint(manifest: Mapping[str, Any], baseline_id: str) -> str:
    shas = {
        str(package["checkpoint_sha256"])
        for package in manifest["condition_replay_packages"]
        if package["baseline_id"] == baseline_id
    }
    if len(shas) != 1:
        raise ValueError(f"G4 does not have one checkpoint for {baseline_id}")
    return shas.pop()


def _camera_summary(gate: Mapping[str, Any]) -> dict[str, Any]:
    criteria = gate["criteria"]
    passing = sorted(
        variant for variant, result in criteria.items() if result["passed"]
    )
    failing = sorted(
        variant for variant, result in criteria.items() if not result["passed"]
    )
    drop_video7_sources = [
        {
            "episode_id": int(row["episode_id"]),
            "semantic_margin_mean": float(row["semantic_margin_mean"]),
            "condition_effect_mean": float(row["condition_effect_mean"]),
            "phase_coverage_mean": float(row["phase_coverage_mean"]),
            "failure_rate": float(row["failure_rate"]),
        }
        for row in gate["source_episode_metrics"]
        if row["camera_variant"] == "drop_video7"
    ]
    drop_video7_sources.sort(key=lambda row: row["episode_id"])
    return {
        "complete_e04_passed": False,
        "passing_variant_count": len(passing),
        "failing_variant_count": len(failing),
        "passing_variants": passing,
        "failing_variants": failing,
        "semantic_direction_positive_for_all_variants": all(
            result["semantic_margin_min_source_mean"] > 0.0
            for result in criteria.values()
        ),
        "drop_video7_frozen_criterion_passed": bool(
            criteria["drop_video7"]["passed"]
        ),
        "drop_video7_source_metrics": drop_video7_sources,
    }
