"""Source-episode calibration for the SimVerify G3 B0 baseline."""

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

SEMANTIC_METRICS = (
    "required_event_coverage",
    "event_order_violation_rate",
    "missing_phase_rate",
    "deadzone_effective_recall",
    "opposite_direction_rate",
    "unexpected_effective_axis_rate",
)
ACTION_STAGES = (
    "raw_policy_chunk_normalized",
    "raw_policy_chunk_direct",
    "temporal_aggregation_action",
    "future_runtime_safe_action",
)


def build_g3_calibration(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    train_replay_root: str | Path,
    validation_replay_roots: Sequence[str | Path],
    m2_root: str | Path,
    bootstrap_repetitions: int = 100_000,
    bootstrap_seed: int = 20_260_725,
) -> dict[str, Any]:
    if len(validation_replay_roots) < 3:
        raise ValueError("G3 calibration requires at least three validation repeats")
    if bootstrap_repetitions < 10_000:
        raise ValueError("G3 calibration requires at least 10000 bootstrap draws")
    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
        or not git.get("git_available")
    ):
        raise ValueError("G3 calibration requires a clean SimVerify worktree")

    train_root = Path(train_replay_root).resolve(strict=True)
    validation_roots = [
        Path(path).resolve(strict=True) for path in validation_replay_roots
    ]
    m2 = Path(m2_root).resolve(strict=True)
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable G3 output exists: {destination}")

    replay_packages = [
        _validated_replay_package(train_root, expected_split="train"),
        *[
            _validated_replay_package(root, expected_split="validation")
            for root in validation_roots
        ],
    ]
    checkpoint_shas = {
        package["manifest"]["provenance"]["checkpoint"]["sha256"]
        for package in replay_packages
    }
    if len(checkpoint_shas) != 1:
        raise ValueError("G3 replay packages do not share one checkpoint")
    repeat_ids = [
        int(package["manifest"]["repeat_id"]) for package in replay_packages[1:]
    ]
    if len(set(repeat_ids)) != len(repeat_ids):
        raise ValueError("validation repeat ids must be unique")

    expert_envelope_path = m2 / "expert_event_envelope_v1.json"
    expert_envelope = _read_json(expert_envelope_path)
    expert_rows = {
        (int(row["episode_id"]), int(row["cycle_id"])): row
        for row in expert_envelope["validation_expert_distribution"]["cycles"]
    }
    validation_rows = replay_packages[1]["rows"]
    validation_by_key = {
        (int(row["episode_id"]), int(row["cycle_id"])): row for row in validation_rows
    }
    if set(validation_by_key) != set(expert_rows):
        raise ValueError("validation replay cycles do not match expert envelope")

    repeat_noise = _measure_repeat_noise(validation_roots)
    source_episode_rows = [
        *_source_episode_metrics(replay_packages[0]["rows"], split="train"),
        *_source_episode_metrics(validation_rows, split="validation"),
    ]
    coverage_deltas: dict[int, list[float]] = defaultdict(list)
    for key, policy_row in validation_by_key.items():
        coverage_deltas[key[0]].append(
            float(policy_row["metrics"]["required_event_coverage"])
            - float(expert_rows[key]["required_event_coverage"])
        )
    episode_coverage_deltas = np.asarray(
        [
            float(np.mean(values))
            for _episode_id, values in sorted(coverage_deltas.items())
        ],
        dtype=np.float64,
    )
    coverage_bootstrap = bootstrap_episode_mean(
        episode_coverage_deltas,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    validation_order_rate = float(
        np.mean(
            [row["metrics"]["event_order_violation_rate"] for row in validation_rows]
        )
    )
    expert_order_rate = float(
        expert_envelope["validation_expert_distribution"]["event_order_violation_rate"]
    )
    coverage_noise = float(
        repeat_noise["semantic_metric_max_abs_delta"]["required_event_coverage"]
    )
    order_noise = float(
        repeat_noise["semantic_metric_max_abs_delta"]["event_order_violation_rate"]
    )
    criteria = {
        "paired_coverage_source_episode_bootstrap_noninferior": {
            "observed_p02_5": coverage_bootstrap["p02_5"],
            "minimum_allowed": -coverage_noise,
            "passed": coverage_bootstrap["p02_5"] >= -coverage_noise,
            "reason": (
                "floor is zero minus measured semantic repeat noise; no "
                "subjective coverage percentage is inserted"
            ),
        },
        "event_order_not_worse_than_expert_plus_repeat_noise": {
            "observed": validation_order_rate,
            "maximum_allowed": expert_order_rate + order_noise,
            "passed": validation_order_rate <= expert_order_rate + order_noise,
        },
        "semantic_repeat_stability": {
            "semantic_metric_max_abs_delta": repeat_noise[
                "semantic_metric_max_abs_delta"
            ],
            "changed_missing_event_rows": repeat_noise["changed_missing_event_rows"],
            "changed_event_tick_rows": repeat_noise["changed_event_tick_rows"],
            "passed": (
                all(
                    value == 0.0
                    for value in repeat_noise["semantic_metric_max_abs_delta"].values()
                )
                and repeat_noise["changed_missing_event_rows"] == 0
                and repeat_noise["changed_event_tick_rows"] == 0
            ),
        },
    }
    passed = all(item["passed"] for item in criteria.values())
    calibration = {
        "schema": "simverify_g3_b0_calibration_v1",
        "gate": "G3",
        "baseline_id": "B0",
        "decision": (
            "pass_recorded_observation_baseline" if passed else "reject_or_revise_b0"
        ),
        "authorizes": ["B1", "B2"] if passed else [],
        "evidence_scope": "recorded-observation/offline",
        "closed_loop_execution": False,
        "criteria": criteria,
        "paired_validation_coverage_delta": {
            "source_episode_count": int(episode_coverage_deltas.size),
            "source_episode_means": episode_coverage_deltas.tolist(),
            "bootstrap": coverage_bootstrap,
        },
        "validation_cycle_policy_distribution": _cycle_metric_summary(validation_rows),
        "expert_validation_distribution": expert_envelope[
            "validation_expert_distribution"
        ],
        "direction_metrics_are_diagnostic_not_thresholded": True,
        "action_mae_is_auxiliary_not_gate": True,
        "gate_thresholds_v1_generated": False,
        "gate_thresholds_v1_reason": (
            "B2 shuffled-condition null and B1 paired effects do not yet exist"
        ),
        "held_out_test_read": False,
    }

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    try:
        identities.append(
            write_jsonl(
                temporary / "source_episode_metrics.jsonl",
                source_episode_rows,
            )
        )
        identities.append(write_json(temporary / "repeat_noise_v1.json", repeat_noise))
        identities.append(write_json(temporary / "g3_calibration_v1.json", calibration))
        manifest_identity = write_json(
            temporary / "g3_manifest.json",
            {
                "schema": "simverify_g3_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "evidence_scope": "recorded-observation/offline",
                "closed_loop_execution": False,
                "held_out_test_read": False,
                "gate_thresholds_v1_generated": False,
                "checkpoint_sha256": next(iter(checkpoint_shas)),
                "expert_event_envelope_sha256": sha256_file(expert_envelope_path),
                "bootstrap": {
                    "unit": "source_episode",
                    "repetitions": bootstrap_repetitions,
                    "seed": bootstrap_seed,
                },
                "replay_packages": [
                    {
                        "path": str(package["root"]),
                        "split": package["manifest"]["split"],
                        "repeat_id": package["manifest"]["repeat_id"],
                        "manifest_sha256": sha256_file(
                            package["root"] / "replay_manifest.json"
                        ),
                        "checksums_sha256": sha256_file(
                            package["root"] / "checksums.sha256"
                        ),
                    }
                    for package in replay_packages
                ],
                "decision": calibration["decision"],
                "authorizes": calibration["authorizes"],
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
            "decision": calibration["decision"],
            "authorizes": calibration["authorizes"],
            "g3_manifest_sha256": manifest_identity["sha256"],
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
                    "schema": "simverify_g3_build_failure_v1",
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                    "git": git,
                },
            )
        raise


def bootstrap_episode_mean(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        array.size,
        size=(repetitions, array.size),
    )
    draws = np.mean(array[indices], axis=1)
    return {
        "method": "source_episode_nonparametric_bootstrap_mean_v1",
        "repetitions": repetitions,
        "seed": seed,
        "observed_mean": float(np.mean(array)),
        "p02_5": float(np.quantile(draws, 0.025)),
        "p50": float(np.quantile(draws, 0.5)),
        "p97_5": float(np.quantile(draws, 0.975)),
    }


def _validated_replay_package(
    root: Path,
    *,
    expected_split: str,
) -> dict[str, Any]:
    verification = verify_checksums(root, root / "checksums.sha256")
    if not verification["ok"]:
        raise ValueError(f"replay checksum verification failed: {root}")
    manifest = _read_json(root / "replay_manifest.json")
    if (
        manifest.get("baseline_id") != "B0"
        or manifest.get("split") != expected_split
        or manifest.get("held_out_test_read") is not False
        or manifest.get("closed_loop_execution") is not False
        or manifest.get("condition_input_used_by_policy") is not False
    ):
        raise ValueError(f"invalid B0 replay manifest: {root}")
    return {
        "root": root,
        "manifest": manifest,
        "rows": _read_jsonl(root / "cycle_metrics.jsonl"),
        "verified_file_count": verification["verified_file_count"],
    }


def _source_episode_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_id"])].append(row["metrics"])
    result: list[dict[str, Any]] = []
    for episode_id, metrics in sorted(grouped.items()):
        result.append(
            {
                "schema": "simverify_g3_source_episode_metrics_v1",
                "split": split,
                "episode_id": episode_id,
                "cycle_count": len(metrics),
                "required_event_coverage_mean": float(
                    np.mean([row["required_event_coverage"] for row in metrics])
                ),
                "required_event_coverage_min": float(
                    min(row["required_event_coverage"] for row in metrics)
                ),
                "event_order_violation_rate": float(
                    np.mean([row["event_order_violation_rate"] for row in metrics])
                ),
                "deadzone_effective_recall": _weighted_rate(
                    metrics,
                    "same_direction_axis_ticks",
                    "expert_effective_axis_ticks",
                ),
                "opposite_direction_rate": _weighted_rate(
                    metrics,
                    "opposite_direction_axis_ticks",
                    "expert_effective_axis_ticks",
                ),
                "unexpected_effective_axis_rate": _weighted_rate(
                    metrics,
                    "unexpected_effective_axis_ticks",
                    "policy_effective_axis_ticks",
                ),
                "action_mae_auxiliary_mean": float(
                    np.mean([row["action_mae_auxiliary"] for row in metrics])
                ),
            }
        )
    return result


def _measure_repeat_noise(
    roots: Sequence[Path],
) -> dict[str, Any]:
    reference_rows = {
        (int(row["episode_id"]), int(row["cycle_id"])): row["metrics"]
        for row in _read_jsonl(roots[0] / "cycle_metrics.jsonl")
    }
    stage_delta = {stage: 0.0 for stage in ACTION_STAGES}
    semantic_delta = {metric: 0.0 for metric in SEMANTIC_METRICS}
    mae_delta = 0.0
    changed_missing = 0
    changed_ticks = 0
    compared_traces = 0
    trace_names = sorted(path.name for path in (roots[0] / "traces").glob("*.npz"))
    for root in roots[1:]:
        candidate_names = sorted(path.name for path in (root / "traces").glob("*.npz"))
        if candidate_names != trace_names:
            raise ValueError("validation repeats do not share trace inventory")
        candidate_rows = {
            (int(row["episode_id"]), int(row["cycle_id"])): row["metrics"]
            for row in _read_jsonl(root / "cycle_metrics.jsonl")
        }
        if set(candidate_rows) != set(reference_rows):
            raise ValueError("validation repeats do not share metric rows")
        for key, reference in reference_rows.items():
            candidate = candidate_rows[key]
            for metric in SEMANTIC_METRICS:
                semantic_delta[metric] = max(
                    semantic_delta[metric],
                    abs(float(reference[metric]) - float(candidate[metric])),
                )
            mae_delta = max(
                mae_delta,
                abs(
                    float(reference["action_mae_auxiliary"])
                    - float(candidate["action_mae_auxiliary"])
                ),
            )
            changed_missing += int(
                reference["missing_events"] != candidate["missing_events"]
            )
            changed_ticks += int(
                reference["event_ticks_local"] != candidate["event_ticks_local"]
            )
        for name in trace_names:
            with (
                np.load(roots[0] / "traces" / name) as reference,
                np.load(root / "traces" / name) as candidate,
            ):
                for stage in ACTION_STAGES:
                    stage_delta[stage] = max(
                        stage_delta[stage],
                        float(
                            np.max(
                                np.abs(
                                    reference[stage].astype(np.float64)
                                    - candidate[stage].astype(np.float64)
                                )
                            )
                        ),
                    )
            compared_traces += 1
    return {
        "schema": "simverify_b0_repeat_noise_v1",
        "reference_repeat_root": str(roots[0]),
        "comparison_repeat_roots": [str(root) for root in roots[1:]],
        "trace_comparison_count": compared_traces,
        "action_stage_max_abs_delta": stage_delta,
        "semantic_metric_max_abs_delta": semantic_delta,
        "action_mae_auxiliary_max_abs_delta": mae_delta,
        "changed_missing_event_rows": changed_missing,
        "changed_event_tick_rows": changed_ticks,
        "evidence_scope": "recorded-observation/offline",
        "closed_loop_execution": False,
    }


def _cycle_metric_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"cycle_count": len(rows)}
    for metric in (*SEMANTIC_METRICS, "action_mae_auxiliary"):
        values = np.asarray(
            [row["metrics"][metric] for row in rows],
            dtype=np.float64,
        )
        result[metric] = {
            "minimum": float(np.min(values)),
            "p02_5": float(np.quantile(values, 0.025)),
            "p50": float(np.quantile(values, 0.5)),
            "p97_5": float(np.quantile(values, 0.975)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
    return result


def _weighted_rate(
    rows: Sequence[Mapping[str, Any]],
    numerator: str,
    denominator: str,
) -> float | None:
    count = sum(int(row[numerator]) for row in rows)
    total = sum(int(row[denominator]) for row in rows)
    return float(count / total) if total else None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
