"""Causal qpos/qvel phase routing for the SimVerify B1.3 experiment."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.simverify.artifacts import (
    artifact_identity,
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance, sha256_file

ROUTE_NAMES = ("current", "neutral", "next")
ROUTE_CURRENT = 0
ROUTE_NEUTRAL = 1
ROUTE_NEXT = 2
HELD_OUT_EPISODES = {1, 13, 25, 33}
FEATURE_NAMES = (
    "qpos_swing",
    "qpos_boom",
    "qpos_stick",
    "qpos_bucket",
    "qvel_swing",
    "qvel_boom",
    "qvel_stick",
    "qvel_bucket",
)
DWELL_CANDIDATES = tuple(range(1, 11))
VARIANCE_FLOOR = 1.0e-6


def fit_diagonal_gaussian(
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Fit a deterministic three-class diagonal Gaussian classifier."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("phase-router features must have shape (N, 8)")
    if y.shape != (x.shape[0],):
        raise ValueError("phase-router labels must have shape (N,)")
    if not np.isfinite(x).all():
        raise ValueError("phase-router features must be finite")
    if set(np.unique(y).tolist()) != {0, 1, 2}:
        raise ValueError("phase-router fit requires all three route classes")

    center = np.median(x, axis=0)
    q25, q75 = np.quantile(x, (0.25, 0.75), axis=0)
    iqr = q75 - q25
    std = np.std(x, axis=0)
    scale = np.where(iqr > 0.0, iqr, np.where(std > 0.0, std, 1.0))
    z = (x - center) / scale
    means = np.stack([np.mean(z[y == route], axis=0) for route in range(3)])
    variances = np.stack([np.var(z[y == route], axis=0) for route in range(3)])
    variances = np.maximum(variances, VARIANCE_FLOOR)
    return {
        "schema": "simverify_observable_phase_router_classifier_v1",
        "feature_names": list(FEATURE_NAMES),
        "normalization_center": center.tolist(),
        "normalization_scale": scale.tolist(),
        "class_means": means.tolist(),
        "class_variances": variances.tolist(),
        "variance_floor": VARIANCE_FLOOR,
        "class_order": list(ROUTE_NAMES),
        "runtime_inputs": ["current_qpos", "current_qvel", "past_router_state"],
        "forbidden_runtime_inputs": [
            "condition",
            "action",
            "phase",
            "progress",
            "successor",
            "future_observation",
            "privilege",
        ],
    }


def predict_raw_routes(
    features: np.ndarray,
    classifier: Mapping[str, Any],
) -> np.ndarray:
    """Predict independent per-tick route classes."""

    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("phase-router features must have shape (N, 8)")
    if not np.isfinite(x).all():
        raise ValueError("phase-router features must be finite")
    center = np.asarray(classifier["normalization_center"], dtype=np.float64)
    scale = np.asarray(classifier["normalization_scale"], dtype=np.float64)
    means = np.asarray(classifier["class_means"], dtype=np.float64)
    variances = np.asarray(classifier["class_variances"], dtype=np.float64)
    if (
        center.shape != (8,)
        or scale.shape != (8,)
        or means.shape != (3, 8)
        or variances.shape != (3, 8)
    ):
        raise ValueError("phase-router classifier parameter shape mismatch")
    if np.any(scale <= 0.0) or np.any(variances <= 0.0):
        raise ValueError("phase-router classifier scales must be positive")
    z = (x - center) / scale
    nll = np.mean(
        ((z[:, None, :] - means[None, :, :]) ** 2) / variances[None, :, :]
        + np.log(variances[None, :, :]),
        axis=2,
    )
    return np.argmin(nll, axis=1).astype(np.int8)


def apply_monotonic_router(
    raw_routes: Sequence[int] | np.ndarray,
    *,
    dwell_steps: int,
) -> np.ndarray:
    """Apply the causal current -> neutral -> next state machine."""

    if int(dwell_steps) <= 0:
        raise ValueError("phase-router dwell_steps must be positive")
    raw = np.asarray(raw_routes, dtype=np.int64).reshape(-1)
    if raw.size and not np.isin(raw, (0, 1, 2)).all():
        raise ValueError("raw phase routes must be in {0,1,2}")
    routed = np.empty(raw.shape, dtype=np.int8)
    state = ROUTE_CURRENT
    consecutive = 0
    for index, predicted in enumerate(raw.tolist()):
        expected_next = state + 1 if state < ROUTE_NEXT else None
        if expected_next is not None:
            consecutive = consecutive + 1 if predicted == expected_next else 0
            if consecutive >= int(dwell_steps):
                state = expected_next
                consecutive = 0
        routed[index] = state
    return routed


class ObservablePhaseRouter:
    """Stateful runtime owner for the frozen causal router."""

    def __init__(self, classifier: Mapping[str, Any], *, dwell_steps: int):
        self.classifier = dict(classifier)
        self.dwell_steps = int(dwell_steps)
        if self.dwell_steps <= 0:
            raise ValueError("phase-router dwell_steps must be positive")
        self.reset()

    def reset(self) -> None:
        self.route = ROUTE_CURRENT
        self.consecutive = 0

    def step(self, qpos: np.ndarray, qvel: np.ndarray) -> int:
        qpos_arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        qvel_arr = np.asarray(qvel, dtype=np.float64).reshape(-1)
        if qpos_arr.shape != (4,) or qvel_arr.shape != (4,):
            raise ValueError("phase router requires qpos[4] and qvel[4]")
        raw = int(
            predict_raw_routes(
                np.concatenate((qpos_arr, qvel_arr), axis=0),
                self.classifier,
            )[0]
        )
        expected_next = self.route + 1 if self.route < ROUTE_NEXT else None
        if expected_next is not None:
            self.consecutive = (
                self.consecutive + 1 if raw == expected_next else 0
            )
            if self.consecutive >= self.dwell_steps:
                self.route = expected_next
                self.consecutive = 0
        return int(self.route)


def build_observable_phase_router(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    m0_root: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Build the immutable train-only router and validation prerequisite Gate."""

    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
        or not git.get("git_available")
    ):
        raise ValueError("phase-router build requires a clean SimVerify worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable phase-router artifact exists: {destination}")
    m0 = Path(m0_root).resolve(strict=True)
    contract = Path(contract_path).resolve(strict=True)
    checksum_verification = verify_checksums(m0, m0 / "checksums.sha256")
    if not checksum_verification["ok"]:
        raise ValueError("M0 checksum verification failed")

    split_manifest = _read_json(m0 / "split_groups.json")
    splits = {
        name: set(map(int, split_manifest["splits"][name]))
        for name in ("train", "validation", "held_out_test")
    }
    if splits["held_out_test"] != HELD_OUT_EPISODES:
        raise ValueError("held-out episode lock differs from frozen split")
    if (splits["train"] | splits["validation"]) & HELD_OUT_EPISODES:
        raise ValueError("held-out episode entered router development")

    annotations = [
        row
        for row in _read_jsonl(m0 / "cycle_annotations.jsonl")
        if row["quality"]["status"] == "accepted"
        and row["split"] in {"train", "validation"}
    ]
    if {int(row["episode_id"]) for row in annotations} & HELD_OUT_EPISODES:
        raise ValueError("held-out annotation entered router development")
    cycles = _load_cycle_rows(annotations, m0)
    train_cycles = [row for row in cycles if row["split"] == "train"]
    validation_cycles = [row for row in cycles if row["split"] == "validation"]
    if not train_cycles or not validation_cycles:
        raise ValueError("phase-router build requires train and validation cycles")

    dwell_audit: list[dict[str, Any]] = []
    loo_predictions: dict[tuple[int, int], np.ndarray] = {}
    raw_loo_predictions: dict[tuple[int, int], np.ndarray] = {}
    train_episode_ids = sorted({int(row["episode_id"]) for row in train_cycles})
    per_dwell_routes: dict[int, dict[tuple[int, int], np.ndarray]] = {
        dwell: {} for dwell in DWELL_CANDIDATES
    }
    for episode_id in train_episode_ids:
        fit_cycles = [
            row for row in train_cycles if int(row["episode_id"]) != episode_id
        ]
        query_cycles = [
            row for row in train_cycles if int(row["episode_id"]) == episode_id
        ]
        classifier = fit_diagonal_gaussian(
            np.concatenate([row["features"] for row in fit_cycles], axis=0),
            np.concatenate([row["true_route"] for row in fit_cycles], axis=0),
        )
        for row in query_cycles:
            key = (int(row["episode_id"]), int(row["cycle_id"]))
            raw = predict_raw_routes(row["features"], classifier)
            raw_loo_predictions[key] = raw
            for dwell in DWELL_CANDIDATES:
                per_dwell_routes[dwell][key] = apply_monotonic_router(
                    raw,
                    dwell_steps=dwell,
                )

    for dwell in DWELL_CANDIDATES:
        episode_metrics = _aggregate_by_episode(
            train_cycles,
            per_dwell_routes[dwell],
        )
        dwell_audit.append(
            {
                "dwell_steps": dwell,
                "selection_score_mean_source_episode_balanced_accuracy": float(
                    np.mean(
                        [row["balanced_accuracy"] for row in episode_metrics]
                    )
                ),
                "mean_source_episode_accuracy": float(
                    np.mean([row["accuracy"] for row in episode_metrics])
                ),
            }
        )
    selected = sorted(
        dwell_audit,
        key=lambda row: (
            -row["selection_score_mean_source_episode_balanced_accuracy"],
            row["dwell_steps"],
        ),
    )[0]
    selected_dwell = int(selected["dwell_steps"])
    loo_predictions = per_dwell_routes[selected_dwell]
    train_loo_episode_metrics = _aggregate_by_episode(
        train_cycles,
        loo_predictions,
    )

    final_classifier = fit_diagonal_gaussian(
        np.concatenate([row["features"] for row in train_cycles], axis=0),
        np.concatenate([row["true_route"] for row in train_cycles], axis=0),
    )
    final_predictions: dict[tuple[int, int], np.ndarray] = {}
    raw_final_predictions: dict[tuple[int, int], np.ndarray] = {}
    for row in [*train_cycles, *validation_cycles]:
        key = (int(row["episode_id"]), int(row["cycle_id"]))
        raw = predict_raw_routes(row["features"], final_classifier)
        raw_final_predictions[key] = raw
        final_predictions[key] = apply_monotonic_router(
            raw,
            dwell_steps=selected_dwell,
        )
    validation_episode_metrics = _aggregate_by_episode(
        validation_cycles,
        final_predictions,
    )
    runtime_parity = _verify_runtime_parity(
        validation_cycles,
        final_predictions,
        final_classifier,
        selected_dwell,
    )
    thresholds = _derive_thresholds(train_loo_episode_metrics)
    gate = _evaluate_gate(
        validation_cycles=validation_cycles,
        validation_episode_metrics=validation_episode_metrics,
        thresholds=thresholds,
        runtime_parity=runtime_parity,
    )

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    try:
        params_identity = write_json(
            temporary / "phase_router_params_v1.json",
            {
                **final_classifier,
                "dwell_steps": selected_dwell,
                "dwell_candidates": list(DWELL_CANDIDATES),
                "dwell_selection_rule": (
                    "maximize_train_source_episode_loo_mean_balanced_accuracy_"
                    "then_smallest_dwell"
                ),
                "route_transition_graph": ["current->neutral", "neutral->next"],
                "reset_route": "current",
                "fit_split": "train_only",
            },
        )
        identities.append(params_identity)
        assignments_identity = _write_assignments_npz(
            temporary / "phase_route_assignments_v1.npz",
            [*train_cycles, *validation_cycles],
            raw_final_predictions,
            final_predictions,
        )
        identities.append(assignments_identity)
        identities.append(
            write_jsonl(
                temporary / "train_loo_source_episode_metrics.jsonl",
                train_loo_episode_metrics,
            )
        )
        identities.append(
            write_jsonl(
                temporary / "validation_source_episode_metrics.jsonl",
                validation_episode_metrics,
            )
        )
        identities.append(
            write_json(
                temporary / "dwell_selection_v1.json",
                {
                    "schema": "simverify_phase_router_dwell_selection_v1",
                    "candidates": dwell_audit,
                    "selected_dwell_steps": selected_dwell,
                    "selection_split": "train_source_episode_leave_one_out",
                    "validation_used_for_selection": False,
                },
            )
        )
        identities.append(
            write_json(
                temporary / "phase_router_thresholds_v1.json",
                thresholds,
            )
        )
        identities.append(
            write_json(
                temporary / "phase_router_gate_v1.json",
                gate,
            )
        )
        manifest_identity = write_json(
            temporary / "phase_router_manifest.json",
            {
                "schema": "simverify_observable_phase_router_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "contract": {
                    "path": str(contract),
                    "sha256": sha256_file(contract),
                },
                "m0_root": str(m0),
                "m0_dataset_manifest_sha256": sha256_file(
                    m0 / "dataset_manifest.json"
                ),
                "m0_checksums_sha256": sha256_file(m0 / "checksums.sha256"),
                "split_manifest_sha256": sha256_file(m0 / "split_groups.json"),
                "annotation_sha256": sha256_file(m0 / "cycle_annotations.jsonl"),
                "params_sha256": params_identity["sha256"],
                "assignments_sha256": assignments_identity["sha256"],
                "selected_dwell_steps": selected_dwell,
                "train_episode_ids": train_episode_ids,
                "validation_episode_ids": sorted(
                    {int(row["episode_id"]) for row in validation_cycles}
                ),
                "held_out_episode_ids": "locked_unread",
                "train_cycle_count": len(train_cycles),
                "validation_cycle_count": len(validation_cycles),
                "decision": gate["decision"],
                "authorizes_b1_3_training": gate["authorizes_b1_3_training"],
                "evidence_scope": "recorded-observation/offline",
                "closed_loop_execution": False,
                "held_out_test_read": False,
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
            "decision": gate["decision"],
            "authorizes_b1_3_training": gate["authorizes_b1_3_training"],
            "selected_dwell_steps": selected_dwell,
            "manifest_sha256": manifest_identity["sha256"],
            "checksums_sha256": checksums_identity["sha256"],
            "held_out_test_read": False,
            "closed_loop_execution": False,
        }
    except BaseException as exc:
        if temporary.exists():
            write_json(
                temporary / "BUILD_FAILED.json",
                {
                    "schema": "simverify_observable_phase_router_failure_v1",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "git": git,
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                },
            )
        raise


def _load_cycle_rows(
    annotations: Sequence[Mapping[str, Any]],
    m0_root: Path,
) -> list[dict[str, Any]]:
    by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        by_episode[int(annotation["episode_id"])].append(annotation)
    rows: list[dict[str, Any]] = []
    for episode_id, episode_annotations in sorted(by_episode.items()):
        path = m0_root / f"episodes/episode_{episode_id}.hdf5"
        with h5py.File(path, "r") as handle:
            qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float64)
            qvel = np.asarray(handle["observations/qvel"][()], dtype=np.float64)
        if qpos.shape != qvel.shape or qpos.ndim != 2 or qpos.shape[1] != 4:
            raise ValueError(f"invalid qpos/qvel shape in {path}")
        for annotation in sorted(
            episode_annotations,
            key=lambda row: int(row["cycle_id"]),
        ):
            start, end = map(int, annotation["target_steps_20hz"])
            carry = int(
                annotation["observable_events"]["carry_transition_proxy"][
                    "representative_target_tick"
                ]
            )
            dump_end = int(
                annotation["observable_events"]["dump_end_proxy"][
                    "representative_target_tick"
                ]
            )
            if not 0 <= start <= carry < dump_end <= end < qpos.shape[0]:
                raise ValueError("observable phase boundaries are not ordered")
            ticks = np.arange(start, end + 1, dtype=np.int64)
            true_route = np.full(ticks.shape, ROUTE_NEUTRAL, dtype=np.int8)
            true_route[ticks <= carry] = ROUTE_CURRENT
            true_route[ticks >= dump_end] = ROUTE_NEXT
            rows.append(
                {
                    "episode_id": episode_id,
                    "cycle_id": int(annotation["cycle_id"]),
                    "split": str(annotation["split"]),
                    "ticks": ticks,
                    "features": np.concatenate(
                        (qpos[start : end + 1], qvel[start : end + 1]),
                        axis=1,
                    ),
                    "true_route": true_route,
                }
            )
    return rows


def _aggregate_by_episode(
    cycles: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[int, int], np.ndarray],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cycles:
        grouped[int(row["episode_id"])].append(row)
    output: list[dict[str, Any]] = []
    for episode_id, episode_cycles in sorted(grouped.items()):
        truth = np.concatenate(
            [np.asarray(row["true_route"]) for row in episode_cycles]
        )
        predicted = np.concatenate(
            [
                np.asarray(
                    predictions[(int(row["episode_id"]), int(row["cycle_id"]))]
                )
                for row in episode_cycles
            ]
        )
        confusion = np.bincount(
            truth.astype(np.int64) * 3 + predicted.astype(np.int64),
            minlength=9,
        ).reshape(3, 3)
        balanced = float(
            np.mean(
                [
                    np.mean(predicted[truth == route] == route)
                    for route in range(3)
                ]
            )
        )
        boundary_errors: dict[str, list[int]] = {"neutral": [], "next": []}
        all_routes_reached = True
        for row in episode_cycles:
            key = (int(row["episode_id"]), int(row["cycle_id"]))
            cycle_predicted = np.asarray(predictions[key])
            cycle_truth = np.asarray(row["true_route"])
            ticks = np.asarray(row["ticks"])
            for route, name in ((ROUTE_NEUTRAL, "neutral"), (ROUTE_NEXT, "next")):
                true_indices = np.flatnonzero(cycle_truth == route)
                predicted_indices = np.flatnonzero(cycle_predicted == route)
                if not true_indices.size or not predicted_indices.size:
                    all_routes_reached = False
                    continue
                boundary_errors[name].append(
                    int(
                        ticks[predicted_indices[0]]
                        - ticks[true_indices[0]]
                    )
                )
        abs_errors = [
            abs(value)
            for values in boundary_errors.values()
            for value in values
        ]
        output.append(
            {
                "schema": "simverify_phase_router_source_episode_metrics_v1",
                "episode_id": episode_id,
                "cycle_count": len(episode_cycles),
                "tick_count": int(truth.size),
                "accuracy": float(np.mean(truth == predicted)),
                "balanced_accuracy": balanced,
                "confusion_true_rows_predicted_columns": confusion.tolist(),
                "all_cycles_reached_neutral_and_next": all_routes_reached,
                "neutral_boundary_offset_ticks": boundary_errors["neutral"],
                "next_boundary_offset_ticks": boundary_errors["next"],
                "boundary_abs_offset_q97_5_ticks": (
                    float(np.quantile(abs_errors, 0.975))
                    if abs_errors
                    else None
                ),
            }
        )
    return output


def _derive_thresholds(
    train_episode_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    balanced = np.asarray(
        [row["balanced_accuracy"] for row in train_episode_metrics],
        dtype=np.float64,
    )
    boundary = [
        float(row["boundary_abs_offset_q97_5_ticks"])
        for row in train_episode_metrics
    ]
    return {
        "schema": "simverify_phase_router_thresholds_v1",
        "source": "train_source_episode_leave_one_out",
        "validation_source_episode_balanced_accuracy_lower": float(
            np.quantile(balanced, 0.025)
        ),
        "validation_source_episode_boundary_abs_offset_q97_5_upper_ticks": float(
            max(boundary)
        ),
        "require_all_cycles_reach_neutral_and_next": True,
        "require_runtime_assignment_exact_parity": True,
        "held_out_test_read": False,
    }


def _evaluate_gate(
    *,
    validation_cycles: Sequence[Mapping[str, Any]],
    validation_episode_metrics: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    runtime_parity: bool,
) -> dict[str, Any]:
    balanced_lower = float(
        thresholds["validation_source_episode_balanced_accuracy_lower"]
    )
    boundary_upper = float(
        thresholds[
            "validation_source_episode_boundary_abs_offset_q97_5_upper_ticks"
        ]
    )
    criteria = {
        "source_episode_balanced_accuracy": {
            "minimum_observed": float(
                min(row["balanced_accuracy"] for row in validation_episode_metrics)
            ),
            "minimum_allowed": balanced_lower,
        },
        "source_episode_boundary_abs_offset_q97_5_ticks": {
            "maximum_observed": float(
                max(
                    float(row["boundary_abs_offset_q97_5_ticks"])
                    for row in validation_episode_metrics
                )
            ),
            "maximum_allowed": boundary_upper,
        },
        "all_cycles_reach_neutral_and_next": {
            "observed": bool(
                all(
                    row["all_cycles_reached_neutral_and_next"]
                    for row in validation_episode_metrics
                )
            ),
            "cycle_count": len(validation_cycles),
        },
        "runtime_assignment_exact_parity": {
            "observed": bool(runtime_parity),
        },
        "forbidden_runtime_inputs_absent": {"observed": True},
    }
    passed = bool(
        criteria["source_episode_balanced_accuracy"]["minimum_observed"]
        >= balanced_lower
        and criteria["source_episode_boundary_abs_offset_q97_5_ticks"][
            "maximum_observed"
        ]
        <= boundary_upper
        and criteria["all_cycles_reach_neutral_and_next"]["observed"]
        and criteria["runtime_assignment_exact_parity"]["observed"]
        and criteria["forbidden_runtime_inputs_absent"]["observed"]
    )
    return {
        "schema": "simverify_observable_phase_router_gate_v1",
        "decision": (
            "pass_observable_phase_router_prerequisite"
            if passed
            else "revise_condition_router"
        ),
        "authorizes_b1_3_training": passed,
        "criteria": criteria,
        "evidence_scope": "recorded-observation/offline",
        "closed_loop_execution": False,
        "held_out_test_read": False,
    }


def _verify_runtime_parity(
    cycles: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[int, int], np.ndarray],
    classifier: Mapping[str, Any],
    dwell_steps: int,
) -> bool:
    for row in cycles:
        router = ObservablePhaseRouter(classifier, dwell_steps=dwell_steps)
        runtime = np.asarray(
            [
                router.step(feature[:4], feature[4:])
                for feature in np.asarray(row["features"])
            ],
            dtype=np.int8,
        )
        key = (int(row["episode_id"]), int(row["cycle_id"]))
        if not np.array_equal(runtime, np.asarray(predictions[key])):
            return False
    return True


def _write_assignments_npz(
    path: Path,
    cycles: Sequence[Mapping[str, Any]],
    raw_predictions: Mapping[tuple[int, int], np.ndarray],
    predictions: Mapping[tuple[int, int], np.ndarray],
) -> dict[str, Any]:
    episode_ids: list[np.ndarray] = []
    cycle_ids: list[np.ndarray] = []
    ticks: list[np.ndarray] = []
    splits: list[np.ndarray] = []
    raw_routes: list[np.ndarray] = []
    routes: list[np.ndarray] = []
    true_routes: list[np.ndarray] = []
    split_id = {"train": 0, "validation": 1}
    for row in cycles:
        key = (int(row["episode_id"]), int(row["cycle_id"]))
        count = int(np.asarray(row["ticks"]).size)
        episode_ids.append(np.full(count, key[0], dtype=np.int64))
        cycle_ids.append(np.full(count, key[1], dtype=np.int64))
        ticks.append(np.asarray(row["ticks"], dtype=np.int64))
        splits.append(
            np.full(count, split_id[str(row["split"])], dtype=np.int8)
        )
        raw_routes.append(np.asarray(raw_predictions[key], dtype=np.int8))
        routes.append(np.asarray(predictions[key], dtype=np.int8))
        true_routes.append(np.asarray(row["true_route"], dtype=np.int8))
    np.savez_compressed(
        path,
        episode_id=np.concatenate(episode_ids),
        cycle_id=np.concatenate(cycle_ids),
        tick=np.concatenate(ticks),
        split=np.concatenate(splits),
        raw_route=np.concatenate(raw_routes),
        route=np.concatenate(routes),
        true_route=np.concatenate(true_routes),
    )
    return artifact_identity(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

