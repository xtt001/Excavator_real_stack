"""Immutable M5 decision for the expert-habit fixed-scenario experiment."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from testbed.simverify.agx_closed_loop_probe import validate_probe_bundle
from testbed.simverify.artifacts import (
    artifact_identity,
    verify_checksums,
    write_checksums,
    write_json,
)
from testbed.simverify.contracts import git_provenance

DECISION_SCHEMA = "simverify_expert_habit_m5_decision_v1"
MANIFEST_SCHEMA = "simverify_expert_habit_m5_manifest_v1"


def build_expert_habit_m5_decision(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    contract_path: str | Path,
    definition_root: str | Path,
    dataset_root: str | Path,
    b0_root: str | Path,
    b1_root: str | Path,
    b2_root: str | Path,
    validation_root: str | Path,
    offline_gate_root: str | Path,
    paired_branch_root: str | Path,
    repeat_same_root: str | Path,
    move_adjacent_root: str | Path,
) -> dict[str, Any]:
    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        not git.get("git_available")
        or git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
    ):
        raise ValueError("expert-habit M5 requires a clean SimVerify worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable M5 output exists: {destination}")

    packages = {
        "definition": _verified_package(
            definition_root,
            "definition_falsification_decision_v1.json",
        ),
        "dataset": _verified_package(dataset_root, "dataset_manifest.json"),
        "validation": _verified_package(
            validation_root,
            "validation_replay_manifest.json",
        ),
        "offline_gate": _verified_package(
            offline_gate_root,
            "gate_decision_v1.json",
        ),
        "paired_branch": _verified_package(
            paired_branch_root,
            "paired_branch_diagnostic.json",
        ),
        "repeat_same": _verified_package(repeat_same_root, "run_manifest.json"),
        "move_adjacent": _verified_package(
            move_adjacent_root,
            "run_manifest.json",
        ),
    }
    bundles = {
        baseline: validate_probe_bundle(root)
        for baseline, root in (
            ("B0", b0_root),
            ("B1", b1_root),
            ("B2", b2_root),
        )
    }
    _validate_evidence(packages=packages, bundles=bundles)

    contract = artifact_identity(Path(contract_path).resolve(strict=True))
    checkpoint_sha = bundles["B1"]["artifacts"]["policy_best.ckpt"]["sha256"]
    offline = packages["offline_gate"]["payload"]
    paired = packages["paired_branch"]["payload"]
    repeat = packages["repeat_same"]["payload"]
    adjacent = packages["move_adjacent"]["payload"]
    decision = {
        "schema": DECISION_SCHEMA,
        "terminal_decision": "sim_observable_only",
        "decision_scope": "technical_capability_classification_only",
        "status_flags": {
            "reject": False,
            "revise_annotation": False,
            "revise_condition": False,
            "sim_observable_only": True,
            "real_finetune_candidate": False,
            "control_candidate": False,
        },
        "offline_gate_preserved": {
            "decision": offline["decision"],
            "basic_capability_established_offline": offline[
                "basic_capability_established_offline"
            ],
            "condition_understanding_established_offline": offline[
                "condition_understanding_established_offline"
            ],
            "criteria_pass": offline["criteria_pass"],
            "rewritten_after_result": False,
        },
        "condition_closed_loop_evidence": {
            "checkpoint_sha256": checkpoint_sha,
            "shared_prefix_takeover_tick": paired["pairing_contract"][
                "takeover_tick"
            ],
            "condition_changes_rollout_above_repeat_variability": paired[
                "condition_response"
            ]["condition_changes_rollout_above_repeat_variability"],
            "paired_targets": {
                name: {
                    "scripted": row["scripted_target_sector"],
                    "realized": row["realized_target_sector"],
                    "completion_policy_tick": row["completion_policy_tick"],
                }
                for name, row in paired["observable_cycle_completion"][
                    "branches"
                ].items()
            },
        },
        "fixed_scenario_evidence": {
            "repeat_same": repeat["observable_cycle_contract"],
            "move_adjacent_then_stay": adjacent["observable_cycle_contract"],
        },
        "evidence_boundary": {
            "evidence_scope": "sim_closed_loop_diagnostic_non_promotable",
            "held_out_test_read": False,
            "physical_effect_validated": False,
            "real_machine_transfer_validated": False,
            "deployment_authorized": False,
            "real_control_candidate": False,
            "external_dirty_read_only_providers": True,
        },
        "plain_language_conclusion": (
            "B1 can execute the observable dig-dump-return cycle, use the "
            "committed condition to settle at left or center ready, and "
            "continue for a second fixed-scenario cycle. This does not prove "
            "soil-effect success, unseen-scene generalization, or real-machine "
            "readiness."
        ),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "stage": "M5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "contract": contract,
        "inputs": {
            key: {
                "root": value["root"],
                "primary": value["primary"],
                "checksums": value["checksums"],
                "verification": value["verification"],
            }
            for key, value in packages.items()
        },
        "training_bundles": bundles,
        "held_out_test_read": False,
        "physical_effect_validated": False,
        "real_control_candidate": False,
    }

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    try:
        decision_identity = write_json(temporary / "decision.json", decision)
        manifest_identity = write_json(temporary / "m5_manifest.json", manifest)
        checksums_identity = write_checksums(
            temporary,
            [decision_identity, manifest_identity],
            path=temporary / "checksums.sha256",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()
    verification = verify_checksums(destination, destination / "checksums.sha256")
    if not verification["ok"]:
        raise RuntimeError("written expert-habit M5 package failed verification")
    return {
        "terminal_decision": "sim_observable_only",
        "output_root": str(destination),
        "decision_sha256": decision_identity["sha256"],
        "manifest_sha256": manifest_identity["sha256"],
        "checksums_sha256": checksums_identity["sha256"],
        "verification": verification,
    }


def _verified_package(
    root: str | Path,
    primary_name: str,
) -> dict[str, Any]:
    package = Path(root).resolve(strict=True)
    checksums = package / "checksums.sha256"
    verification = verify_checksums(package, checksums)
    if not verification["ok"]:
        raise ValueError(f"checksum verification failed: {package}")
    primary = package / primary_name
    return {
        "root": str(package),
        "primary": artifact_identity(primary),
        "checksums": artifact_identity(checksums),
        "verification": verification,
        "payload": json.loads(primary.read_text(encoding="utf-8")),
    }


def _validate_evidence(
    *,
    packages: dict[str, dict[str, Any]],
    bundles: dict[str, dict[str, Any]],
) -> None:
    if set(bundles) != {"B0", "B1", "B2"} or any(
        bundles[key]["baseline_id"] != key for key in bundles
    ):
        raise ValueError("M5 requires matched B0/B1/B2 bundles")
    definition = packages["definition"]["payload"]
    dataset = packages["dataset"]["payload"]
    validation = packages["validation"]["payload"]
    offline = packages["offline_gate"]["payload"]
    paired = packages["paired_branch"]["payload"]
    repeat = packages["repeat_same"]["payload"]
    adjacent = packages["move_adjacent"]["payload"]
    if (
        definition.get("decision") != "accept"
        or definition["provenance"]["held_out_observation_read_count"] != 0
        or dataset["provenance"]["held_out_observation_read_count"] != 0
        or dataset["provenance"]["privilege_used"] is not False
        or validation.get("held_out_test_read") is not False
    ):
        raise ValueError("definition/dataset/validation evidence boundary failed")
    if (
        offline.get("basic_capability_established_offline") is not True
        or offline.get("condition_understanding_established_offline") is not False
        or offline.get("decision")
        != "condition_understanding_not_established_offline"
        or offline.get("held_out_test_read") is not False
    ):
        raise ValueError("frozen offline Gate was not preserved")
    branches = paired["observable_cycle_completion"]["branches"]
    if (
        paired.get("closed_loop_execution_after_shared_prefix") is not True
        or paired["condition_response"].get(
            "condition_changes_rollout_above_repeat_variability"
        )
        is not True
        or paired["observable_cycle_completion"].get("all_branches_completed")
        is not True
        or {(row["scripted_target_sector"], row["realized_target_sector"]) for row in branches.values()}
        != {("left", "left"), ("center", "center")}
        or paired.get("held_out_test_read") is not False
        or paired.get("physical_effect_validated") is not False
        or paired.get("real_control_candidate") is not False
    ):
        raise ValueError("paired closed-loop condition evidence failed")
    _validate_scenario(
        repeat,
        expected_targets=("left", "left"),
        checkpoint_sha=bundles["B1"]["artifacts"]["policy_best.ckpt"]["sha256"],
    )
    _validate_scenario(
        adjacent,
        expected_targets=("center", "center"),
        checkpoint_sha=bundles["B1"]["artifacts"]["policy_best.ckpt"]["sha256"],
    )


def _validate_scenario(
    manifest: dict[str, Any],
    *,
    expected_targets: tuple[str, str],
    checkpoint_sha: str,
) -> None:
    observable = manifest["observable_cycle_contract"]
    targets = tuple(
        row["realized_target_sector"] for row in observable["cycle_completions"]
    )
    if (
        manifest.get("closed_loop_execution") is not True
        or observable.get("observable_cycle_completed") is not True
        or observable.get("requested_cycle_count") != 2
        or observable.get("completed_cycle_count") != 2
        or targets != expected_targets
        or manifest.get("physical_effect_validated") is True
        or manifest.get("real_control_candidate") is not False
        or any(manifest["privilege_policy_input_scan"].values())
        or manifest["bundle_contract"]["artifacts"]["policy_best.ckpt"]["sha256"]
        != checkpoint_sha
    ):
        raise ValueError("continuous fixed-scenario evidence failed")
