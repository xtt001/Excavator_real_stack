"""Terminal M5 decision for the B1.4/G5.1/E04 SimVerify experiment version."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from testbed.simverify.artifacts import (
    verify_checksums,
    write_checksums,
    write_json,
)
from testbed.simverify.contracts import git_provenance, sha256_file

EVIDENCE_SCOPE = "recorded-observation/offline"
HELD_OUT_EPISODES = {1, 13, 25, 33}


def build_m5_decision_v3(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    prior_m5_root: str | Path,
    g4_root: str | Path,
    g5_v1_root: str | Path,
    g5_1_root: str | Path,
    e04_root: str | Path,
) -> dict[str, Any]:
    """Validate the revised evidence chain and write an immutable M5 package."""

    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        not git.get("git_available")
        or git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
    ):
        raise ValueError("M5 v3 requires a clean v2.0.0-simVerify worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable M5 v3 output exists: {destination}")

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
    _validate_chain(
        prior_m5=prior_m5,
        g4=g4,
        g5_v1=g5_v1,
        g5_1=g5_1,
        e04=e04,
    )

    decision = decision_payload_v3(
        prior_m5=prior_m5,
        g4=g4,
        g5_v1=g5_v1,
        g5_1=g5_1,
        e04=e04,
    )
    manifest = {
        "schema": "simverify_m5_manifest_v3",
        "stage": "M5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "decision": "revise_condition",
        "terminal_for_experiment_version": True,
        "experiment_version": "B1.4_G5.1_E04",
        "evidence_scope": EVIDENCE_SCOPE,
        "closed_loop_execution": False,
        "held_out_test_read": False,
        "held_out_test_authorized": False,
        "control_candidate": False,
        "inputs": {
            "prior_m5": prior_m5["identity"],
            "g4_b1_4": g4["identity"],
            "g5_v1": g5_v1["identity"],
            "g5_1": g5_1["identity"],
            "e04": e04["identity"],
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
                    "schema": "simverify_m5_build_failure_v3",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                    "git": git,
                },
            )
        raise


def decision_payload_v3(
    *,
    prior_m5: dict[str, Any],
    g4: dict[str, Any],
    g5_v1: dict[str, Any],
    g5_1: dict[str, Any],
    e04: dict[str, Any],
) -> dict[str, Any]:
    """Materialize the only decision authorized by the verified Gate path."""

    return {
        "schema": "simverify_m5_decision_v3",
        "stage": "M5",
        "experiment_version": "B1.4_G5.1_E04",
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
                "gate": "G0_G1_G2_M1_M2_G3",
                "result": "pass_inherited_and_reverified_by_inputs",
                "evidence": prior_m5["identity"],
            },
            {
                "gate": "G4",
                "result": "next_condition_understanding_established_offline_development",
                "evidence": g4["identity"],
            },
            {
                "gate": "G5_v1",
                "result": "core_not_established_router_lifecycle_defect",
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
                "reason": "G5 camera robustness did not pass",
            },
        ],
        "terminal_reason": (
            "B1.4 understands the supported next condition on the complete "
            "four-camera recorded path and passes G5.1 core continuity, but "
            "E04 shows direction reversal or matched-response failure when "
            "stick views are removed, video7 is dropped, frames are fixed, "
            "or images are delayed; explicit camera-role sensitivity is also "
            "not established by role swaps"
        ),
        "why_not_reject": (
            "M0/M1/M2 contracts, observable annotation, B0, supported G4 "
            "condition response, and full-camera G5.1 continuity are usable"
        ),
        "why_not_sim_observable_only": (
            "the frozen G5 robustness family is incomplete and E04 failed"
        ),
        "next_authorized_action": (
            "freeze a new one-factor conditioned-visual robustness contract "
            "that addresses stick-up dependence and camera-role sensitivity"
        ),
        "forbidden_next_actions": [
            "enter E05, E06, G6, or held-out test for this experiment version",
            "relax E04 thresholds after observing validation",
            "claim simulator or real closed-loop success",
            "promote any sim checkpoint to real fine-tuning, shadow, or control",
            "change condition, camera augmentation, and runtime together",
        ],
        "interpretation_guard": (
            "recorded-observation action semantics do not prove environmental "
            "response or closed-loop excavation"
        ),
    }


def _validate_chain(
    *,
    prior_m5: dict[str, Any],
    g4: dict[str, Any],
    g5_v1: dict[str, Any],
    g5_1: dict[str, Any],
    e04: dict[str, Any],
) -> None:
    prior = prior_m5["gate"]
    if (
        prior.get("decision") != "revise_condition"
        or not prior.get("terminal_for_experiment_version")
        or prior.get("held_out_test_read")
        or prior.get("closed_loop_execution")
    ):
        raise ValueError("invalid prior M5 evidence")
    if (
        g4["manifest"].get("candidate_baseline_id") != "B1.4"
        or g4["gate"].get("decision")
        != "next_condition_understanding_established_offline"
        or g4["manifest"].get("decision")
        != "next_condition_understanding_established_offline"
    ):
        raise ValueError("G4 B1.4 next-condition Gate did not pass")
    if g5_v1["gate"].get(
        "decision"
    ) != "g5_core_two_cycle_condition_continuity_not_established" or g5_v1[
        "manifest"
    ].get("authorizes_remaining_g5_robustness"):
        raise ValueError("invalid G5 v1 failure evidence")
    if g5_1["gate"].get(
        "decision"
    ) != "g5_core_two_cycle_condition_continuity_established_development" or not g5_1[
        "manifest"
    ].get("authorizes_remaining_g5_robustness"):
        raise ValueError("G5.1 core continuity did not pass")
    previous = g5_1["manifest"].get("previous_g5_core", {})
    if previous.get("manifest_sha256") != g5_v1["identity"]["manifest_sha256"]:
        raise ValueError("G5.1 does not bind the supplied G5 v1 artifact")
    if e04["gate"].get(
        "decision"
    ) != "e04_camera_counterfactual_robustness_not_established" or e04["manifest"].get(
        "authorizes_e05"
    ):
        raise ValueError("E04 is not the frozen stop decision")
    e04_previous = e04["manifest"].get("previous_g5", {})
    if e04_previous.get("manifest_sha256") != g5_1["identity"]["manifest_sha256"]:
        raise ValueError("E04 does not bind the supplied G5.1 artifact")
    if (
        e04["manifest"]["bundle"]["checkpoint_sha256"]
        != g5_1["manifest"]["bundles"]["B1.4"]["checkpoint_sha256"]
    ):
        raise ValueError("E04 and G5.1 B1.4 checkpoints differ")
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


def _verified_package(
    root: Path,
    *,
    manifest_name: str,
    gate_name: str,
) -> dict[str, Any]:
    verification = verify_checksums(root, root / "checksums.sha256")
    if not verification["ok"]:
        raise ValueError(f"checksum verification failed: {root}")
    manifest_path = root / manifest_name
    gate_path = root / gate_name
    if not manifest_path.is_file() or not gate_path.is_file():
        raise ValueError(f"required M5 input file missing: {root}")
    manifest = _read_json(manifest_path)
    gate = _read_json(gate_path)
    identity = {
        "path": str(root),
        "manifest_path": manifest_name,
        "manifest_sha256": sha256_file(manifest_path),
        "gate_path": gate_name,
        "gate_sha256": sha256_file(gate_path),
        "checksums_sha256": sha256_file(root / "checksums.sha256"),
        "verified_file_count": verification["verified_file_count"],
    }
    return {
        "root": root,
        "manifest": manifest,
        "gate": gate,
        "identity": identity,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
