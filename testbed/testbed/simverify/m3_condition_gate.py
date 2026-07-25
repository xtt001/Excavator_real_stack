"""Source-episode G4 calibration for conditioned SimVerify ACT.

The implementation keeps the immutable M0 threshold contract as provenance,
but corrects two statistical defects before any held-out data are read:

* a paired B1-B2 contrast must not add the absolute B2 null a second time;
* rate uncertainty must be expressed as a rate, not as action magnitude.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from testbed.simverify.artifacts import (
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.m3_gate import bootstrap_episode_mean

FACTORS = ("current_sector", "next_sector")
HELD_OUT_EPISODES = {1, 13, 25, 33}


def build_g4_condition_calibration(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    b1_replay_roots: Sequence[str | Path],
    b2_replay_root: str | Path,
    g3_root: str | Path,
    m0_root: str | Path,
    m2_root: str | Path,
    bootstrap_repetitions: int = 100_000,
    bootstrap_seed: int = 20_260_725,
) -> dict[str, Any]:
    """Build an immutable, held-out-free G4 validation decision package."""

    if len(b1_replay_roots) < 3:
        raise ValueError("G4 calibration requires at least three B1 repeats")
    if bootstrap_repetitions < 10_000:
        raise ValueError("G4 calibration requires at least 10000 bootstrap draws")
    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
        or not git.get("git_available")
    ):
        raise ValueError("G4 calibration requires a clean SimVerify worktree")

    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable G4 output exists: {destination}")
    m0 = Path(m0_root).resolve(strict=True)
    m2 = Path(m2_root).resolve(strict=True)
    threshold_contract_path = m0 / "gate_thresholds_contract_v1.json"
    threshold_contract = _read_json(threshold_contract_path)
    if threshold_contract.get("held_out_test", {}).get("authorized") is not False:
        raise ValueError("M0 threshold contract does not lock held-out test")

    g3 = _validated_g3(Path(g3_root).resolve(strict=True))
    b1_packages = [
        _validated_condition_package(
            Path(root).resolve(strict=True),
            expected_baseline="B1",
        )
        for root in b1_replay_roots
    ]
    b2_package = _validated_condition_package(
        Path(b2_replay_root).resolve(strict=True),
        expected_baseline="B2",
    )
    _require_matched_packages(b1_packages, b2_package)
    repeat_ids = [int(package["manifest"]["repeat_id"]) for package in b1_packages]
    if len(set(repeat_ids)) != len(repeat_ids):
        raise ValueError("B1 repeat ids must be unique")
    reference_index = int(np.argmin(repeat_ids))
    reference = b1_packages[reference_index]
    other_repeats = [
        package for index, package in enumerate(b1_packages) if index != reference_index
    ]

    repeat_noise = _repeat_noise(reference, other_repeats)
    source_rows, factor_evidence = _source_episode_evidence(
        reference,
        other_repeats,
        b2_package,
        repeat_noise=repeat_noise,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    )
    formula_audit = _formula_audit()
    primary_criteria: dict[str, Any] = {}
    for factor in FACTORS:
        evidence = factor_evidence[factor]
        primary_criteria[f"{factor}_action_effect"] = evidence["action_effect"]
        primary_criteria[f"{factor}_direction"] = evidence["direction"]
        primary_criteria[f"{factor}_latency"] = evidence["latency"]
        primary_criteria[f"{factor}_condition_not_ignored"] = evidence[
            "condition_not_ignored"
        ]
        primary_criteria[f"{factor}_phase_preservation"] = evidence[
            "phase_preservation"
        ]
    validation_passed = all(
        bool(criterion["passed"]) for criterion in primary_criteria.values()
    )
    decision = (
        "pass_g4_validation_calibration" if validation_passed else "revise_condition"
    )
    calibration = {
        "schema": "simverify_g4_condition_calibration_v1",
        "gate": "G4",
        "decision": decision,
        "authorizes_g5": validation_passed,
        "recommended_terminal_status": None
        if validation_passed
        else "revise_condition",
        "evidence_scope": "recorded-observation/offline",
        "closed_loop_execution": False,
        "held_out_test_read": False,
        "gate_thresholds_v1_generated": False,
        "gate_thresholds_v1_reason": (
            "G4 validation did not pass; held-out and G5 remain locked"
            if not validation_passed
            else "G5 validation diagnostics are still required before the global "
            "threshold artifact can be frozen"
        ),
        "method_contract": {
            "name": "g4_source_episode_paired_contract_v2",
            "old_contract_preserved": True,
            "old_contract_sha256": sha256_file(threshold_contract_path),
            "correction_is_result_independent": True,
            "correction_summary": [
                "paired B1-B2 contrasts use repeat noise as the zero-margin; "
                "the B2 null is already present in the paired contrast",
                "condition-ignored rate uses repeat rate uncertainty rather "
                "than action-magnitude uncertainty",
            ],
        },
        "support": {factor: factor_evidence[factor]["support"] for factor in FACTORS},
        "criteria": primary_criteria,
        "same_token_repeat_consistency_threshold": repeat_noise[
            "same_token_repeat_consistency"
        ],
        "unsupported_anchors_are_failures": False,
        "unsupported_anchor_policy": (
            "retained in replay inventory and excluded from success denominator"
        ),
        "action_mae_is_gate": False,
    }

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    try:
        identities.append(
            write_jsonl(
                temporary / "source_episode_condition_metrics.jsonl",
                source_rows,
            )
        )
        identities.append(
            write_json(temporary / "b1_repeat_noise_v1.json", repeat_noise)
        )
        identities.append(
            write_json(temporary / "g4_formula_audit_v2.json", formula_audit)
        )
        identities.append(write_json(temporary / "g4_calibration_v1.json", calibration))
        manifest_identity = write_json(
            temporary / "g4_manifest.json",
            {
                "schema": "simverify_g4_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "decision": decision,
                "authorizes_g5": validation_passed,
                "evidence_scope": "recorded-observation/offline",
                "closed_loop_execution": False,
                "held_out_test_read": False,
                "gate_thresholds_v1_generated": False,
                "bootstrap": {
                    "unit": "source_episode",
                    "repetitions": bootstrap_repetitions,
                    "seed": bootstrap_seed,
                },
                "m0_dataset_manifest_sha256": sha256_file(m0 / "dataset_manifest.json"),
                "m2_manifest_sha256": sha256_file(m2 / "m2_manifest.json"),
                "m0_gate_thresholds_contract_v1_sha256": sha256_file(
                    threshold_contract_path
                ),
                "g3_manifest_sha256": sha256_file(g3["root"] / "g3_manifest.json"),
                "condition_replay_packages": [
                    _package_identity(package) for package in [*b1_packages, b2_package]
                ],
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
            "authorizes_g5": validation_passed,
            "manifest_sha256": manifest_identity["sha256"],
            "checksums_sha256": checksums_identity["sha256"],
            "held_out_test_read": False,
            "closed_loop_execution": False,
        }
    except BaseException as exc:
        failure = temporary / "BUILD_FAILED.json"
        if temporary.exists() and not failure.exists():
            write_json(
                failure,
                {
                    "schema": "simverify_g4_build_failure_v1",
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                    "git": git,
                },
            )
        raise


def paired_metric_result(
    b1_episode_values: np.ndarray,
    b2_episode_values: np.ndarray,
    *,
    repeat_noise: float,
    lower_is_better: bool,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate a same-unit paired contrast against repeat uncertainty."""

    b1 = np.asarray(b1_episode_values, dtype=np.float64)
    b2 = np.asarray(b2_episode_values, dtype=np.float64)
    if b1.shape != b2.shape or b1.ndim != 1 or b1.size < 2:
        raise ValueError("paired metric requires at least two matched episodes")
    if not np.isfinite(b1).all() or not np.isfinite(b2).all():
        raise ValueError("paired metric values must be finite")
    if not np.isfinite(repeat_noise) or repeat_noise < 0:
        raise ValueError("repeat noise must be finite and non-negative")
    delta = b1 - b2
    bootstrap = bootstrap_episode_mean(
        delta,
        repetitions=repetitions,
        seed=seed,
    )
    if lower_is_better:
        passed = bootstrap["p97_5"] < -repeat_noise
        comparison = "paired_bootstrap_p97_5 < -repeat_rate_noise_q97_5"
    else:
        passed = bootstrap["p02_5"] > repeat_noise
        comparison = "paired_bootstrap_p02_5 > repeat_metric_noise_q97_5"
    return {
        "b1_source_episode_values": b1.tolist(),
        "b2_source_episode_values": b2.tolist(),
        "paired_source_episode_deltas": delta.tolist(),
        "paired_bootstrap": bootstrap,
        "repeat_noise_margin": float(repeat_noise),
        "comparison": comparison,
        "passed": bool(passed),
    }


def first_effect_latency(values: Sequence[float], *, noise_floor: float) -> int:
    """Return the first tick above repeat noise, censoring after the window."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("latency values must be a finite vector")
    matches = np.flatnonzero(array > float(noise_floor))
    return int(matches[0]) if matches.size else int(array.size + 1)


def symmetric_trace_consistency(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    """One minus symmetric normalized L1 difference, bounded to [0, 1]."""

    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.shape != second.shape or first.size == 0:
        raise ValueError("consistency traces must have the same non-empty shape")
    numerator = float(np.sum(np.abs(first - second)))
    denominator = float(np.sum(np.abs(first)) + np.sum(np.abs(second)))
    if denominator == 0.0:
        return 1.0
    return float(np.clip(1.0 - numerator / denominator, 0.0, 1.0))


def _source_episode_evidence(
    reference: Mapping[str, Any],
    repeats: Sequence[Mapping[str, Any]],
    b2: Mapping[str, Any],
    *,
    repeat_noise: Mapping[str, Any],
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_rows = reference["supported_rows"]
    repeat_rows = [package["supported_rows"] for package in repeats]
    b2_rows = b2["supported_rows"]
    source_rows: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    for factor_index, factor in enumerate(FACTORS):
        anchor_ids = sorted(
            anchor_id
            for anchor_id, row in reference_rows.items()
            if row["changed_factor"] == factor
        )
        grouped: dict[int, list[int]] = defaultdict(list)
        for anchor_id in anchor_ids:
            grouped[int(reference_rows[anchor_id]["episode_id"])].append(anchor_id)
        if len(grouped) < 2:
            raise ValueError(f"{factor} lacks two supported source episodes")
        rows_by_episode: list[dict[str, Any]] = []
        for episode_id, episode_anchors in sorted(grouped.items()):
            b1_effect = _mean_metric(
                reference_rows, episode_anchors, "token_swap_action_effect"
            )
            b2_effect = _mean_metric(
                b2_rows, episode_anchors, "token_swap_action_effect"
            )
            b1_direction = _mean_metric(
                reference_rows, episode_anchors, "token_swap_direction_correct"
            )
            b2_direction = _mean_metric(
                b2_rows, episode_anchors, "token_swap_direction_correct"
            )
            effect_repeat_delta = float(
                np.mean(
                    [
                        max(
                            abs(
                                float(
                                    reference_rows[anchor_id]["metrics"][
                                        "token_swap_action_effect"
                                    ]
                                )
                                - float(
                                    rows[anchor_id]["metrics"][
                                        "token_swap_action_effect"
                                    ]
                                )
                            )
                            for rows in repeat_rows
                        )
                        for anchor_id in episode_anchors
                    ]
                )
            )
            direction_repeat_delta = max(
                abs(
                    b1_direction
                    - _mean_metric(
                        rows,
                        episode_anchors,
                        "token_swap_direction_correct",
                    )
                )
                for rows in repeat_rows
            )
            b1_latency = _episode_latency_q95(
                reference_rows,
                episode_anchors,
                noise_floor=float(repeat_noise["per_tick_effect_noise_floor"]),
            )
            repeat_latencies = [
                _episode_latency_q95(
                    rows,
                    episode_anchors,
                    noise_floor=float(repeat_noise["per_tick_effect_noise_floor"]),
                )
                for rows in repeat_rows
            ]
            latency_jitter = max(
                abs(b1_latency - candidate) for candidate in repeat_latencies
            )
            expert_latency = _episode_expert_latency_q95(
                reference,
                episode_anchors,
            )
            ignored_floor = float(repeat_noise["action_effect_noise_floor"])
            b1_ignored = _ignored_rate(
                reference_rows,
                episode_anchors,
                floor=ignored_floor,
            )
            b2_ignored = _ignored_rate(
                b2_rows,
                episode_anchors,
                floor=ignored_floor,
            )
            repeat_ignored_delta = max(
                abs(
                    b1_ignored
                    - _ignored_rate(rows, episode_anchors, floor=ignored_floor)
                )
                for rows in repeat_rows
            )
            phase_coverage_delta = _mean_metric(
                reference_rows,
                episode_anchors,
                "event_coverage_delta",
            )
            order_violation_rate = float(
                np.mean(
                    [
                        not bool(
                            reference_rows[anchor_id]["metrics"][
                                "target_event_order_valid"
                            ]
                        )
                        for anchor_id in episode_anchors
                    ]
                )
            )
            row = {
                "schema": "simverify_g4_source_episode_metrics_v1",
                "factor": factor,
                "episode_id": episode_id,
                "supported_anchor_count": len(episode_anchors),
                "b1_action_effect_mean": b1_effect,
                "b2_action_effect_mean": b2_effect,
                "b1_minus_b2_action_effect": b1_effect - b2_effect,
                "b1_direction_accuracy": b1_direction,
                "b2_direction_accuracy": b2_direction,
                "b1_minus_b2_direction_accuracy": b1_direction - b2_direction,
                "b1_repeat_action_effect_abs_delta": effect_repeat_delta,
                "b1_repeat_direction_rate_abs_delta": direction_repeat_delta,
                "expert_condition_relevant_action_onset_q95_ticks": expert_latency,
                "b1_response_latency_q95_ticks": b1_latency,
                "b1_repeat_latency_jitter_ticks": latency_jitter,
                "b1_condition_ignored_rate": b1_ignored,
                "b2_condition_ignored_rate": b2_ignored,
                "b1_repeat_condition_ignored_rate_abs_delta": (repeat_ignored_delta),
                "b1_event_coverage_delta_mean": phase_coverage_delta,
                "b1_target_event_order_violation_rate": order_violation_rate,
                "evidence_scope": "recorded-observation/offline",
            }
            rows_by_episode.append(row)
            source_rows.append(row)

        action_noise = _q(
            [row["b1_repeat_action_effect_abs_delta"] for row in rows_by_episode],
            0.975,
        )
        direction_noise = _q(
            [row["b1_repeat_direction_rate_abs_delta"] for row in rows_by_episode],
            0.975,
        )
        ignored_rate_noise = _q(
            [
                row["b1_repeat_condition_ignored_rate_abs_delta"]
                for row in rows_by_episode
            ],
            0.975,
        )
        seed = bootstrap_seed + factor_index * 100
        action_result = paired_metric_result(
            np.asarray([row["b1_action_effect_mean"] for row in rows_by_episode]),
            np.asarray([row["b2_action_effect_mean"] for row in rows_by_episode]),
            repeat_noise=action_noise,
            lower_is_better=False,
            repetitions=bootstrap_repetitions,
            seed=seed + 1,
        )
        direction_result = paired_metric_result(
            np.asarray([row["b1_direction_accuracy"] for row in rows_by_episode]),
            np.asarray([row["b2_direction_accuracy"] for row in rows_by_episode]),
            repeat_noise=direction_noise,
            lower_is_better=False,
            repetitions=bootstrap_repetitions,
            seed=seed + 2,
        )
        ignored_result = paired_metric_result(
            np.asarray([row["b1_condition_ignored_rate"] for row in rows_by_episode]),
            np.asarray([row["b2_condition_ignored_rate"] for row in rows_by_episode]),
            repeat_noise=ignored_rate_noise,
            lower_is_better=True,
            repetitions=bootstrap_repetitions,
            seed=seed + 3,
        )
        expert_onset_bound = _q(
            [
                row["expert_condition_relevant_action_onset_q95_ticks"]
                for row in rows_by_episode
            ],
            0.975,
        )
        latency_jitter_bound = _q(
            [row["b1_repeat_latency_jitter_ticks"] for row in rows_by_episode],
            0.975,
        )
        latency_bound = expert_onset_bound + latency_jitter_bound
        observed_latency = _q(
            [row["b1_response_latency_q95_ticks"] for row in rows_by_episode],
            0.975,
        )
        latency_result = {
            "expert_validation_condition_relevant_action_onset_ticks_q97_5": (
                expert_onset_bound
            ),
            "b1_repeat_latency_jitter_ticks_q97_5": latency_jitter_bound,
            "upper_bound_ticks": latency_bound,
            "b1_validation_response_latency_ticks_q97_5": observed_latency,
            "comparison": "observed_q97_5 <= expert_q97_5 + repeat_jitter_q97_5",
            "passed": bool(observed_latency <= latency_bound),
        }
        phase_result = {
            "event_coverage_delta_source_episode_values": [
                row["b1_event_coverage_delta_mean"] for row in rows_by_episode
            ],
            "target_event_order_violation_source_episode_values": [
                row["b1_target_event_order_violation_rate"] for row in rows_by_episode
            ],
            "comparison": "coverage_delta >= 0 and event-order violation == 0",
            "passed": bool(
                min(row["b1_event_coverage_delta_mean"] for row in rows_by_episode)
                >= 0.0
                and max(
                    row["b1_target_event_order_violation_rate"]
                    for row in rows_by_episode
                )
                == 0.0
            ),
        }
        evidence[factor] = {
            "support": {
                "supported_anchor_count": len(anchor_ids),
                "distinct_source_episode_count": len(grouped),
                "source_episode_ids": sorted(grouped),
                "minimum_distinct_source_episodes": 2,
                "passed": len(grouped) >= 2,
            },
            "action_effect": action_result,
            "direction": direction_result,
            "latency": latency_result,
            "condition_not_ignored": ignored_result,
            "phase_preservation": phase_result,
        }
    return source_rows, evidence


def _repeat_noise(
    reference: Mapping[str, Any],
    repeats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = reference["supported_rows"]
    repeat_rows = [package["supported_rows"] for package in repeats]
    per_anchor_effect_delta: dict[int, float] = {}
    per_anchor_tick_delta: dict[int, float] = {}
    per_anchor_consistency: dict[int, float] = {}
    trace_stage_max_abs_delta = {
        "raw_policy_chunk_normalized": 0.0,
        "raw_policy_chunk_direct": 0.0,
        "temporal_aggregation_action": 0.0,
        "future_runtime_safe_action": 0.0,
    }
    for anchor_id, row in rows.items():
        base_effect = float(row["metrics"]["token_swap_action_effect"])
        base_per_tick = np.asarray(
            row["metrics"]["per_tick_effect_l1"],
            dtype=np.float64,
        )
        per_anchor_effect_delta[anchor_id] = max(
            abs(
                base_effect
                - float(candidate[anchor_id]["metrics"]["token_swap_action_effect"])
            )
            for candidate in repeat_rows
        )
        per_anchor_tick_delta[anchor_id] = max(
            float(
                np.max(
                    np.abs(
                        base_per_tick
                        - np.asarray(
                            candidate[anchor_id]["metrics"]["per_tick_effect_l1"],
                            dtype=np.float64,
                        )
                    )
                )
            )
            for candidate in repeat_rows
        )
        reference_trace = _load_target_trace(reference, row)
        consistency_values = []
        for package, candidate in zip(repeats, repeat_rows, strict=True):
            candidate_trace = _load_target_trace(package, candidate[anchor_id])
            consistency_values.append(
                symmetric_trace_consistency(
                    reference_trace["future_runtime_safe_action"],
                    candidate_trace["future_runtime_safe_action"],
                )
            )
            for stage in trace_stage_max_abs_delta:
                trace_stage_max_abs_delta[stage] = max(
                    trace_stage_max_abs_delta[stage],
                    float(
                        np.max(
                            np.abs(
                                reference_trace[stage].astype(np.float64)
                                - candidate_trace[stage].astype(np.float64)
                            )
                        )
                    ),
                )
        per_anchor_consistency[anchor_id] = min(consistency_values)

    grouped: dict[int, list[int]] = defaultdict(list)
    for anchor_id, row in rows.items():
        grouped[int(row["episode_id"])].append(anchor_id)
    episode_effect_noise = [
        float(np.mean([per_anchor_effect_delta[index] for index in indices]))
        for _episode, indices in sorted(grouped.items())
    ]
    episode_tick_noise = [
        float(np.mean([per_anchor_tick_delta[index] for index in indices]))
        for _episode, indices in sorted(grouped.items())
    ]
    episode_consistency = [
        float(np.mean([per_anchor_consistency[index] for index in indices]))
        for _episode, indices in sorted(grouped.items())
    ]
    return {
        "schema": "simverify_b1_condition_repeat_noise_v1",
        "reference_repeat_id": reference["manifest"]["repeat_id"],
        "comparison_repeat_ids": [
            package["manifest"]["repeat_id"] for package in repeats
        ],
        "supported_anchor_count": len(rows),
        "distinct_source_episode_count": len(grouped),
        "action_effect_episode_noise_values": episode_effect_noise,
        "action_effect_noise_floor": _q(episode_effect_noise, 0.975),
        "per_tick_effect_episode_noise_values": episode_tick_noise,
        "per_tick_effect_noise_floor": _q(episode_tick_noise, 0.975),
        "trace_stage_max_abs_delta": trace_stage_max_abs_delta,
        "same_token_repeat_consistency": {
            "definition": (
                "one minus symmetric normalized L1 difference on the "
                "future-runtime-safe action; per-anchor minimum across repeats, "
                "then source-episode mean"
            ),
            "source_episode_values": episode_consistency,
            "validation_q02_5_lower_bound": _q(episode_consistency, 0.025),
            "held_out_comparison_pending": True,
        },
        "evidence_scope": "recorded-observation/offline",
        "closed_loop_execution": False,
    }


def _validated_condition_package(
    root: Path,
    *,
    expected_baseline: str,
) -> dict[str, Any]:
    verification = verify_checksums(root, root / "checksums.sha256")
    if not verification["ok"]:
        raise ValueError(f"condition replay checksum verification failed: {root}")
    manifest = _read_json(root / "condition_replay_manifest.json")
    if (
        manifest.get("baseline_id") != expected_baseline
        or manifest.get("split") != "validation"
        or manifest.get("held_out_test_read") is not False
        or manifest.get("closed_loop_execution") is not False
        or manifest.get("condition_input_used_by_policy") is not True
        or manifest.get("observation_history_changed") is not False
        or manifest.get("one_primary_factor_per_anchor") is not True
    ):
        raise ValueError(f"invalid {expected_baseline} condition replay: {root}")
    if set(map(int, manifest["episode_ids"])) & HELD_OUT_EPISODES:
        raise ValueError("held-out episode entered condition replay package")
    all_rows = _read_jsonl(root / "condition_swap_metrics.jsonl")
    supported_rows = {
        int(row["anchor_index"]): row for row in all_rows if row["supported"]
    }
    return {
        "root": root,
        "manifest": manifest,
        "all_rows": all_rows,
        "supported_rows": supported_rows,
        "verified_file_count": verification["verified_file_count"],
    }


def _validated_g3(root: Path) -> dict[str, Any]:
    verification = verify_checksums(root, root / "checksums.sha256")
    if not verification["ok"]:
        raise ValueError("G3 checksum verification failed")
    manifest = _read_json(root / "g3_manifest.json")
    if (
        manifest.get("decision") != "pass_recorded_observation_baseline"
        or manifest.get("held_out_test_read") is not False
        or manifest.get("closed_loop_execution") is not False
    ):
        raise ValueError("G4 requires the passing offline G3 baseline")
    return {"root": root, "manifest": manifest}


def _require_matched_packages(
    b1_packages: Sequence[Mapping[str, Any]],
    b2_package: Mapping[str, Any],
) -> None:
    packages = [*b1_packages, b2_package]
    binding_fields = (
        "m0_dataset_manifest_sha256",
        "m2_manifest_sha256",
        "anchor_registry_sha256",
    )
    for field in binding_fields:
        if len({package["manifest"][field] for package in packages}) != 1:
            raise ValueError(f"condition replays disagree on {field}")
    signatures = []
    for package in packages:
        signatures.append(
            [
                (
                    int(row["anchor_index"]),
                    int(row["episode_id"]),
                    int(row["cycle_id"]),
                    str(row["changed_factor"]),
                    bool(row["supported"]),
                )
                for row in package["all_rows"]
            ]
        )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("condition replay anchor inventories do not match")
    if not b1_packages[0]["supported_rows"]:
        raise ValueError("condition replay has no supported anchors")


def _package_identity(package: Mapping[str, Any]) -> dict[str, Any]:
    root = package["root"]
    manifest = package["manifest"]
    return {
        "path": str(root),
        "baseline_id": manifest["baseline_id"],
        "repeat_id": manifest["repeat_id"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "manifest_sha256": sha256_file(root / "condition_replay_manifest.json"),
        "checksums_sha256": sha256_file(root / "checksums.sha256"),
    }


def _mean_metric(
    rows: Mapping[int, Mapping[str, Any]],
    anchor_ids: Sequence[int],
    metric: str,
) -> float:
    return float(
        np.mean([float(rows[anchor_id]["metrics"][metric]) for anchor_id in anchor_ids])
    )


def _ignored_rate(
    rows: Mapping[int, Mapping[str, Any]],
    anchor_ids: Sequence[int],
    *,
    floor: float,
) -> float:
    return float(
        np.mean(
            [
                float(rows[anchor_id]["metrics"]["token_swap_action_effect"]) <= floor
                for anchor_id in anchor_ids
            ]
        )
    )


def _episode_latency_q95(
    rows: Mapping[int, Mapping[str, Any]],
    anchor_ids: Sequence[int],
    *,
    noise_floor: float,
) -> float:
    latencies = []
    for anchor_id in anchor_ids:
        metrics = rows[anchor_id]["metrics"]
        start, end = map(int, metrics["relevant_window_local"])
        values = metrics["per_tick_effect_l1"][start:end]
        latencies.append(first_effect_latency(values, noise_floor=noise_floor))
    return _q(latencies, 0.95)


def _episode_expert_latency_q95(
    package: Mapping[str, Any],
    anchor_ids: Sequence[int],
) -> float:
    latencies = []
    for anchor_id in anchor_ids:
        row = package["supported_rows"][anchor_id]
        start, end = map(int, row["metrics"]["relevant_window_local"])
        with np.load(package["root"] / row["base_trace_path"]) as trace:
            swing = np.abs(
                np.asarray(trace["expert_action"][start:end, 0], dtype=np.float64)
            )
        matches = np.flatnonzero(swing >= 0.05)
        latencies.append(int(matches[0]) if matches.size else int(swing.size + 1))
    return _q(latencies, 0.95)


def _load_target_trace(
    package: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    with np.load(package["root"] / row["target_trace_path"]) as trace:
        return {name: np.asarray(trace[name]).copy() for name in trace.files}


def _q(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("quantile values must be a finite non-empty vector")
    return float(np.quantile(array, quantile))


def _formula_audit() -> dict[str, Any]:
    return {
        "schema": "simverify_g4_formula_audit_v2",
        "audit_basis": "algebra_and_dimensional_consistency_before_held_out",
        "result_independent": True,
        "held_out_test_read": False,
        "old_contract_preserved": True,
        "findings": [
            {
                "metrics": [
                    "token_swap_action_effect",
                    "token_swap_direction_accuracy",
                    "current_sector_sensitivity",
                    "next_sector_sensitivity",
                ],
                "old_formula_issue": (
                    "paired CI(B1-B2) already subtracts B2, so comparing it to "
                    "B2 absolute q97.5 plus noise counts the null twice"
                ),
                "corrected_formula": (
                    "paired_bootstrap_CI95(B1-B2).lower > "
                    "B1_same_checkpoint_repeat_metric_noise_q97_5"
                ),
                "b2_null_still_used": True,
            },
            {
                "metrics": ["condition_ignored_rate"],
                "old_formula_issue": (
                    "the old formula subtracts action-magnitude repeat noise "
                    "from a dimensionless rate and also counts B2 twice"
                ),
                "corrected_formula": (
                    "paired_bootstrap_CI95(B1_rate-B2_rate).upper < "
                    "-B1_same_checkpoint_repeat_rate_noise_q97_5"
                ),
                "b2_null_still_used": True,
            },
            {
                "metrics": [
                    "token_response_latency_ticks",
                    "same_token_repeat_consistency",
                ],
                "old_formula_issue": None,
                "corrected_formula": "unchanged",
                "b2_null_still_used": False,
            },
        ],
        "scope": (
            "G4 validation calibration only; global gate_thresholds_v1.json "
            "remains ungenerated and held-out remains locked"
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
