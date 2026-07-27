"""Data-derived gate freeze for the expert-habit B0/B1/B2 experiment."""

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


def source_episode_bootstrap(
    values: Mapping[int, float],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a scalar by source episode, the frozen independence unit."""

    if repetitions < 10_000:
        raise ValueError("source-episode bootstrap requires at least 10000 draws")
    if len(values) < 2:
        raise ValueError("source-episode bootstrap requires at least two episodes")
    source_ids = sorted(map(int, values))
    vector = np.asarray([float(values[source_id]) for source_id in source_ids])
    if not np.isfinite(vector).all():
        raise ValueError("bootstrap values must be finite")
    generator = np.random.default_rng(seed)
    draws = generator.choice(
        vector,
        size=(repetitions, vector.size),
        replace=True,
    ).mean(axis=1)
    return {
        "unit": "source_episode",
        "source_episode_ids": source_ids,
        "source_episode_count": len(source_ids),
        "repetitions": repetitions,
        "seed": seed,
        "estimate": float(np.mean(vector)),
        "ci95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "source_values": {
            str(source_id): float(values[source_id]) for source_id in source_ids
        },
    }


def derive_habit_gate(
    cycle_rows: Sequence[Mapping[str, Any]],
    swap_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive thresholds and an offline-only decision from matched rows."""

    cycles = {
        (str(row["baseline_id"]), int(row["derived_episode_id"])): row
        for row in cycle_rows
    }
    episode_ids = sorted(
        {
            derived_id
            for baseline, derived_id in cycles
            if baseline == "B0"
        }
    )
    for derived_id in episode_ids:
        for baseline in ("B0", "B1", "B2"):
            if (baseline, derived_id) not in cycles:
                raise ValueError("cycle metrics are not matched across B0/B1/B2")
    source_by_derived = {
        derived_id: int(cycles[("B0", derived_id)]["source_episode_id"])
        for derived_id in episode_ids
    }

    basic_coverage = _source_means(
        {
            derived_id: float(
                cycles[("B1", derived_id)]["action_grammar"][
                    "required_event_coverage"
                ]
            )
            - float(
                cycles[("B1", derived_id)]["zero_action_grammar"][
                    "required_event_coverage"
                ]
            )
            for derived_id in episode_ids
        },
        source_by_derived,
    )
    basic_recall = _source_means(
        {
            derived_id: float(
                cycles[("B1", derived_id)]["action_grammar"][
                    "deadzone_effective_recall"
                ]
            )
            for derived_id in episode_ids
        },
        source_by_derived,
    )
    b1_vs_b0 = _source_means(
        {
            derived_id: float(
                cycles[("B0", derived_id)]["metrics"]["post_commit"]["overall"][
                    "mae"
                ]
            )
            - float(
                cycles[("B1", derived_id)]["metrics"]["post_commit"]["overall"][
                    "mae"
                ]
            )
            for derived_id in episode_ids
        },
        source_by_derived,
    )
    b1_vs_b2 = _source_means(
        {
            derived_id: float(
                cycles[("B2", derived_id)]["metrics"]["post_commit"]["overall"][
                    "mae"
                ]
            )
            - float(
                cycles[("B1", derived_id)]["metrics"]["post_commit"]["overall"][
                    "mae"
                ]
            )
            for derived_id in episode_ids
        },
        source_by_derived,
    )

    supported = [
        row
        for row in swap_rows
        if row["status"] == "supported_fixed_observation_intervention"
    ]
    swaps = {
        (str(row["baseline_id"]), int(row["derived_episode_id"])): row
        for row in supported
    }
    swap_ids = sorted(
        {
            derived_id
            for baseline, derived_id in swaps
            if baseline == "B1" and ("B2", derived_id) in swaps
        }
    )
    if not swap_ids:
        raise ValueError("gate has no matched supported condition interventions")
    swap_source = {
        derived_id: int(swaps[("B1", derived_id)]["source_episode_id"])
        for derived_id in swap_ids
    }
    direction_vs_chance = _source_means(
        {
            derived_id: float(
                bool(
                    swaps[("B1", derived_id)]["metrics"][
                        "semantic_direction_correct"
                    ]
                )
            )
            - 0.5
            for derived_id in swap_ids
        },
        swap_source,
    )
    direction_vs_b2 = _source_means(
        {
            derived_id: float(
                bool(
                    swaps[("B1", derived_id)]["metrics"][
                        "semantic_direction_correct"
                    ]
                )
            )
            - float(
                bool(
                    swaps[("B2", derived_id)]["metrics"][
                        "semantic_direction_correct"
                    ]
                )
            )
            for derived_id in swap_ids
        },
        swap_source,
    )
    pre_dump_effect_max = max(
        float(row["metrics"]["pre_dump_effect_l1"]) for row in supported
    )

    named_values = {
        "basic_event_coverage_vs_zero_action": basic_coverage,
        "basic_deadzone_effective_recall_vs_zero": basic_recall,
        "condition_post_commit_mae_advantage_vs_B0": b1_vs_b0,
        "condition_post_commit_mae_advantage_vs_B2": b1_vs_b2,
        "condition_semantic_direction_vs_chance": direction_vs_chance,
        "condition_semantic_direction_advantage_vs_B2": direction_vs_b2,
    }
    bootstrap = {
        name: source_episode_bootstrap(
            values,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed + index,
        )
        for index, (name, values) in enumerate(named_values.items())
    }
    criteria = {
        name: {
            "rule": "source_episode_bootstrap_ci95_lower_gt_zero",
            "threshold": 0.0,
            "observed": result,
            "passed": bool(result["ci95"][0] > 0.0),
        }
        for name, result in bootstrap.items()
    }
    criteria["condition_pre_dump_causal_localization"] = {
        "rule": "maximum_pre_dump_swap_effect_l1_le_numerical_tolerance",
        "threshold": 1e-7,
        "observed": pre_dump_effect_max,
        "passed": bool(pre_dump_effect_max <= 1e-7),
    }
    basic_names = (
        "basic_event_coverage_vs_zero_action",
        "basic_deadzone_effective_recall_vs_zero",
    )
    condition_names = (
        "condition_post_commit_mae_advantage_vs_B0",
        "condition_post_commit_mae_advantage_vs_B2",
        "condition_semantic_direction_vs_chance",
        "condition_semantic_direction_advantage_vs_B2",
        "condition_pre_dump_causal_localization",
    )
    basic_passed = all(bool(criteria[name]["passed"]) for name in basic_names)
    condition_passed = all(
        bool(criteria[name]["passed"]) for name in condition_names
    )
    decision_name = (
        "basic_capability_not_established_offline"
        if not basic_passed
        else "condition_understanding_not_established_offline"
        if not condition_passed
        else "offline_condition_evidence_accepted"
    )
    thresholds = {
        "schema": "simverify_habit_gate_thresholds_v1",
        "generation_contract": (
            "natural zero/chance nulls plus matched B0 and shuffled-target B2; "
            "all statistical comparisons bootstrap source episodes"
        ),
        "criteria": criteria,
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": bootstrap_seed,
        "held_out_test_read": False,
        "frozen_before_held_out_test": True,
    }
    decision = {
        "schema": "simverify_habit_gate_decision_v1",
        "decision": decision_name,
        "basic_capability_established_offline": basic_passed,
        "condition_understanding_established_offline": condition_passed,
        "basic_criteria": list(basic_names),
        "condition_criteria": list(condition_names),
        "criteria_pass": {
            name: bool(value["passed"]) for name, value in criteria.items()
        },
        "evidence_scope": "recorded-observation/offline",
        "teacher_forced_replay": True,
        "closed_loop_execution": False,
        "observable_cycle_completed_by_policy": False,
        "physical_effect_validated": False,
        "held_out_test_read": False,
        "interpretation_guard": (
            "passing establishes only recorded-observation action grammar and "
            "condition use; it does not establish target arrival or cycle completion"
        ),
    }
    return thresholds, decision


def build_habit_gate(
    *,
    repo_root: str | Path,
    replay_root: str | Path,
    output_root: str | Path,
    bootstrap_repetitions: int = 100_000,
    bootstrap_seed: int = 20_260_727,
) -> dict[str, Any]:
    """Freeze the gate and decision without opening any held-out observation."""

    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        not git.get("git_available")
        or git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
    ):
        raise ValueError("habit gate requires a clean SimVerify worktree")
    replay = Path(replay_root).resolve(strict=True)
    verification = verify_checksums(replay, replay / "checksums.sha256")
    if not verification["ok"]:
        raise ValueError("habit replay checksum verification failed")
    replay_manifest = _read_json(replay / "validation_replay_manifest.json")
    if replay_manifest.get("held_out_test_read") is not False:
        raise ValueError("habit replay does not prove held-out lock")
    cycle_rows = _read_jsonl(replay / "cycle_metrics.jsonl")
    swap_rows = _read_jsonl(replay / "condition_swap_metrics.jsonl")
    thresholds, decision = derive_habit_gate(
        cycle_rows,
        swap_rows,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    )

    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable habit gate exists: {destination}")
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    try:
        thresholds_identity = write_json(
            temporary / "gate_thresholds_v1.json",
            thresholds,
        )
        identities.append(thresholds_identity)
        identities.append(write_json(temporary / "gate_decision_v1.json", decision))
        manifest_identity = write_json(
            temporary / "gate_manifest.json",
            {
                "schema": "simverify_habit_gate_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "replay_manifest_sha256": sha256_file(
                    replay / "validation_replay_manifest.json"
                ),
                "replay_checksums_sha256": sha256_file(replay / "checksums.sha256"),
                "gate_thresholds_sha256": thresholds_identity["sha256"],
                "decision": decision["decision"],
                "evidence_scope": "recorded-observation/offline",
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
            "decision": decision["decision"],
            "basic_capability_established_offline": decision[
                "basic_capability_established_offline"
            ],
            "condition_understanding_established_offline": decision[
                "condition_understanding_established_offline"
            ],
            "gate_thresholds_sha256": thresholds_identity["sha256"],
            "manifest_sha256": manifest_identity["sha256"],
            "checksums_sha256": checksums_identity["sha256"],
            "held_out_test_read": False,
            "closed_loop_execution": False,
        }
    except BaseException as exc:
        if temporary.exists() and not (temporary / "BUILD_FAILED.json").exists():
            write_json(
                temporary / "BUILD_FAILED.json",
                {
                    "schema": "simverify_habit_gate_failure_v1",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "git": git,
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                },
            )
        raise


def _source_means(
    values: Mapping[int, float],
    source_by_derived: Mapping[int, int],
) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for derived_id, value in values.items():
        grouped.setdefault(int(source_by_derived[derived_id]), []).append(float(value))
    return {
        source_id: float(np.mean(source_values))
        for source_id, source_values in grouped.items()
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
