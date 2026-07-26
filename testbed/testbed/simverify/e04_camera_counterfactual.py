"""E04 camera counterfactual replay for the SimVerify G5 robustness Gate."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from testbed.simverify.artifacts import (
    artifact_identity,
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.g5_two_cycle_replay import (
    CAMERA_VARIANTS,
    _load_policy,
    _read_json,
    _read_jsonl,
    _validate_bundle,
    build_two_cycle_condition_support,
    replay_two_cycle_arrays,
    two_cycle_trace_metrics,
)
from testbed.simverify.m3_transition_stitch import HELD_OUT_EPISODES, _q

EVIDENCE_SCOPE = "recorded-observation/offline teacher-forced development"
MODES = ("switched", "unchanged")


def build_e04_camera_counterfactual(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    m0_root: str | Path,
    m2_root: str | Path,
    b1_bundle_root: str | Path,
    previous_g5_root: str | Path,
    contract_path: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Build an immutable E04 development package."""

    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        not git.get("git_available")
        or git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
    ):
        raise ValueError("E04 requires a clean v2.0.0-simVerify worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable E04 package exists: {destination}")
    m0 = Path(m0_root).resolve(strict=True)
    m2 = Path(m2_root).resolve(strict=True)
    previous_g5 = Path(previous_g5_root).resolve(strict=True)
    contract = Path(contract_path).resolve(strict=True)
    verifications = {
        "m0": verify_checksums(m0, m0 / "checksums.sha256"),
        "m2": verify_checksums(m2, m2 / "checksums.sha256"),
        "previous_g5": verify_checksums(
            previous_g5,
            previous_g5 / "checksums.sha256",
        ),
    }
    if not all(row["ok"] for row in verifications.values()):
        raise ValueError("E04 input checksum verification failed")
    previous_gate = _read_json(previous_g5 / "g5_core_gate_v1.json")
    if (
        previous_gate["decision"]
        != "g5_core_two_cycle_condition_continuity_established_development"
        or not previous_gate["authorizes_remaining_g5_robustness"]
        or previous_gate["held_out_test_read"]
    ):
        raise ValueError("E04 requires the passing unread-test G5.1 package")

    split = _read_json(m0 / "split_groups.json")
    if set(map(int, split["splits"]["held_out_test"])) != HELD_OUT_EPISODES:
        raise ValueError("held-out split differs from frozen contract")
    anchors = _read_jsonl(m2 / "two_cycle_anchors_v1.jsonl")
    support = build_two_cycle_condition_support(
        anchors,
        _read_jsonl(m2 / "condition_counterfactual_anchors_v1.jsonl"),
    )
    supported_keys = {
        (row["split"], int(row["episode_id"]), int(row["first_cycle_id"]))
        for row in support["rows"]
        if row["next_target_changed"] and row["supported"]
    }
    validation = [
        row
        for row in anchors
        if (
            row["split"],
            int(row["episode_id"]),
            int(row["first_cycle_id"]),
        )
        in supported_keys
        and row["split"] == "validation"
    ]
    source_ids = sorted({int(row["episode_id"]) for row in validation})
    if len(source_ids) < 2:
        raise ValueError("E04 requires at least two validation source episodes")
    if any(episode_id in HELD_OUT_EPISODES for episode_id in source_ids):
        raise ValueError("held-out episode entered E04")

    envelope = _read_json(m2 / "expert_event_envelope_v1.json")
    templates = envelope["templates"]
    deadzone = list(map(float, envelope["effective_deadzone"]))
    direction = _read_json(previous_g5 / "g5_sector_action_direction_v1.json")
    previous_results = _read_jsonl(previous_g5 / "g5_two_cycle_results.jsonl")
    previous_by_key = {
        (
            int(row["episode_id"]),
            int(row["first_cycle_id"]),
            str(row["condition_mode"]),
        ): row
        for row in previous_results
        if row["baseline_id"] == "B1.4"
    }
    bundle = _validate_bundle(Path(b1_bundle_root).resolve(strict=True), "B1.4")
    max_steps = max(
        int(row["target_steps_20hz"][1]) - int(row["target_steps_20hz"][0]) + 1
        for row in validation
    )
    policy = _load_policy(bundle, max_steps=max_steps, device=device)

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    reproduction_max_abs_delta = 0.0
    try:
        grouped: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        for anchor_index, anchor in enumerate(validation):
            grouped[int(anchor["episode_id"])].append((anchor_index, anchor))
        for episode_id, episode_anchors in sorted(grouped.items()):
            with h5py.File(
                m0 / f"episodes/episode_{episode_id}.hdf5",
                "r",
            ) as episode:
                for anchor_index, anchor in episode_anchors:
                    for variant in CAMERA_VARIANTS:
                        repeats = (0, 1) if variant == "four_camera" else (0,)
                        for repeat in repeats:
                            mode_arrays: dict[str, dict[str, np.ndarray]] = {}
                            mode_rows: dict[str, dict[str, Any]] = {}
                            for mode in MODES:
                                arrays = replay_two_cycle_arrays(
                                    policy=policy,
                                    episode=episode,
                                    anchor=anchor,
                                    condition_mode=mode,
                                    reset_condition_cycle_at_boundary=True,
                                    camera_variant=variant,
                                )
                                metrics = two_cycle_trace_metrics(
                                    arrays,
                                    templates=templates,
                                    deadzone=deadzone,
                                )
                                relative = (
                                    Path("traces")
                                    / variant
                                    / (
                                        f"anchor_{anchor_index}_episode_{episode_id}_"
                                        f"cycles_{anchor['first_cycle_id']}_"
                                        f"{anchor['second_cycle_id']}_{mode}_"
                                        f"repeat{repeat}.npz"
                                    )
                                )
                                trace_path = temporary / relative
                                trace_path.parent.mkdir(parents=True, exist_ok=True)
                                np.savez_compressed(trace_path, **arrays)
                                identity = artifact_identity(trace_path)
                                identities.append(identity)
                                row = {
                                    "schema": "simverify_e04_camera_trace_result_v1",
                                    "anchor_index": anchor_index,
                                    "episode_id": episode_id,
                                    "first_cycle_id": int(anchor["first_cycle_id"]),
                                    "second_cycle_id": int(anchor["second_cycle_id"]),
                                    "camera_variant": variant,
                                    "repeat": repeat,
                                    "condition_mode": mode,
                                    "first_condition": anchor["first_condition"],
                                    "second_condition": anchor["second_condition"],
                                    "trace_path": str(relative),
                                    "trace_sha256": identity["sha256"],
                                    "evidence_scope": EVIDENCE_SCOPE,
                                    "closed_loop_execution": False,
                                    **metrics,
                                }
                                result_rows.append(row)
                                mode_rows[mode] = row
                                mode_arrays[mode] = arrays

                                if variant == "four_camera" and repeat == 0:
                                    prior = previous_by_key[
                                        (
                                            episode_id,
                                            int(anchor["first_cycle_id"]),
                                            mode,
                                        )
                                    ]
                                    with np.load(
                                        previous_g5 / prior["trace_path"],
                                        allow_pickle=False,
                                    ) as old:
                                        delta = float(
                                            np.max(
                                                np.abs(
                                                    arrays["future_runtime_safe_action"]
                                                    - np.asarray(
                                                        old[
                                                            "future_runtime_safe_action"
                                                        ],
                                                        dtype=np.float32,
                                                    )
                                                )
                                            )
                                        )
                                    reproduction_max_abs_delta = max(
                                        reproduction_max_abs_delta,
                                        delta,
                                    )
                            pair_rows.append(
                                camera_pair_metric(
                                    mode_rows["switched"],
                                    mode_rows["unchanged"],
                                    mode_arrays["switched"],
                                    mode_arrays["unchanged"],
                                    direction=direction,
                                )
                            )
        if reproduction_max_abs_delta != 0.0:
            raise ValueError("E04 four-camera replay does not exactly reproduce G5.1")

        thresholds = derive_e04_thresholds(pair_rows)
        source_rows = aggregate_e04_by_source(pair_rows, thresholds=thresholds)
        gate = evaluate_e04_gate(
            source_rows,
            thresholds=thresholds,
            ready_upper=float(
                previous_gate["criteria"]["b1_4_ready_boundary_discontinuity"][
                    "maximum_allowed"
                ]
            ),
        )
        identities.extend(
            [
                write_jsonl(temporary / "e04_trace_results_v1.jsonl", result_rows),
                write_jsonl(temporary / "e04_pair_metrics_v1.jsonl", pair_rows),
                write_jsonl(temporary / "e04_source_metrics_v1.jsonl", source_rows),
                write_json(
                    temporary / "e04_thresholds_v1.json",
                    thresholds,
                ),
            ]
        )
        gate_identity = write_json(temporary / "e04_gate_v1.json", gate)
        identities.append(gate_identity)
        manifest_identity = write_json(
            temporary / "e04_manifest.json",
            {
                "schema": "simverify_e04_camera_counterfactual_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "contract": {
                    "path": str(contract),
                    "sha256": sha256_file(contract),
                },
                "m0": _input_identity(
                    m0,
                    "dataset_manifest.json",
                    verifications["m0"],
                ),
                "m2": _input_identity(
                    m2,
                    "m2_manifest.json",
                    verifications["m2"],
                ),
                "previous_g5": _input_identity(
                    previous_g5,
                    "g5_two_cycle_manifest.json",
                    verifications["previous_g5"],
                ),
                "bundle": bundle["identity"],
                "camera_mapping_sha256": sha256_file(m0 / "camera_mapping.json"),
                "variants": list(CAMERA_VARIANTS),
                "supported_validation_pair_count": len(validation),
                "source_episode_ids": source_ids,
                "four_camera_reproduction_max_abs_delta": (reproduction_max_abs_delta),
                "decision": gate["decision"],
                "authorizes_e05": gate["authorizes_e05"],
                "validation_role": "development",
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
            "decision": gate["decision"],
            "authorizes_e05": gate["authorizes_e05"],
            "trace_count": len(result_rows),
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
                    "schema": "simverify_e04_build_failure_v1",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "git": git,
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                },
            )
        raise
    finally:
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def camera_pair_metric(
    switched_row: Mapping[str, Any],
    unchanged_row: Mapping[str, Any],
    switched_arrays: Mapping[str, np.ndarray],
    unchanged_arrays: Mapping[str, np.ndarray],
    *,
    direction: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure a condition switch on one camera intervention."""

    action_a = np.asarray(
        switched_arrays["future_runtime_safe_action"],
        dtype=np.float32,
    )
    action_b = np.asarray(
        unchanged_arrays["future_runtime_safe_action"],
        dtype=np.float32,
    )
    boundary = int(switched_arrays["shared_ready_boundary_local_index"])
    route_mask = (
        np.asarray(switched_arrays["condition_route_index"])[boundary:] == 2
    ) | (np.asarray(unchanged_arrays["condition_route_index"])[boundary:] == 2)
    delta = action_a[boundary:] - action_b[boundary:]
    effect = float(np.mean(np.abs(delta)))
    swing_delta = float(np.mean(delta[route_mask, 0])) if np.any(route_mask) else 0.0
    first = switched_row["first_condition"]["next_ready_sector"]
    second = switched_row["second_condition"]["next_ready_sector"]
    centers = direction["sector_swing_qpos_median"]
    expected_sign = int(
        np.sign(float(centers[second]) - float(centers[first]))
        * int(direction["action_to_qpos_direction_sign"])
    )
    return {
        "schema": "simverify_e04_camera_pair_metric_v1",
        "anchor_index": int(switched_row["anchor_index"]),
        "episode_id": int(switched_row["episode_id"]),
        "camera_variant": str(switched_row["camera_variant"]),
        "repeat": int(switched_row["repeat"]),
        "condition_switch_action_effect": effect,
        "route2_swing_delta_mean": swing_delta,
        "expected_swing_action_sign": expected_sign,
        "route2_semantic_margin": expected_sign * swing_delta,
        "two_cycle_phase_coverage": float(switched_row["two_cycle_phase_coverage"]),
        "event_order_valid": bool(switched_row["two_cycle_event_order_valid"]),
        "ready_boundary_discontinuity": float(
            switched_row["ready_boundary_discontinuity"]
        ),
        "second_cycle_route0_tick_count": int(
            switched_row["second_cycle_route0_tick_count"]
        ),
        "second_cycle_route2_tick_count": int(
            switched_row["second_cycle_route2_tick_count"]
        ),
        "shared_ready_boundary_route_index": int(
            switched_row["shared_ready_boundary_route_index"]
        ),
        "condition_cycle_router_reset_count": int(
            switched_row["condition_cycle_router_reset_count"]
        ),
        "switched_trace_sha256": switched_row["trace_sha256"],
        "unchanged_trace_sha256": unchanged_row["trace_sha256"],
        "closed_loop_execution": False,
    }


def derive_e04_thresholds(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive development thresholds from matched four-camera repeats."""

    full = [row for row in rows if row["camera_variant"] == "four_camera"]
    indexed = {
        (int(row["episode_id"]), int(row["anchor_index"]), int(row["repeat"])): row
        for row in full
    }
    episode_ids = sorted({int(row["episode_id"]) for row in full})
    source_baseline = []
    repeat_deltas = []
    for episode_id in episode_ids:
        anchor_ids = sorted(
            {
                int(row["anchor_index"])
                for row in full
                if int(row["episode_id"]) == episode_id
            }
        )
        zero = [indexed[(episode_id, anchor_id, 0)] for anchor_id in anchor_ids]
        one = [indexed[(episode_id, anchor_id, 1)] for anchor_id in anchor_ids]
        source_baseline.append(
            {
                "episode_id": episode_id,
                "condition_effect_mean": float(
                    np.mean([row["condition_switch_action_effect"] for row in zero])
                ),
                "phase_coverage_mean": float(
                    np.mean([row["two_cycle_phase_coverage"] for row in zero])
                ),
            }
        )
        repeat_deltas.append(
            {
                "episode_id": episode_id,
                "condition_effect_abs_delta": float(
                    np.mean(
                        [
                            abs(
                                left["condition_switch_action_effect"]
                                - right["condition_switch_action_effect"]
                            )
                            for left, right in zip(zero, one, strict=True)
                        ]
                    )
                ),
                "phase_coverage_abs_delta": float(
                    np.mean(
                        [
                            abs(
                                left["two_cycle_phase_coverage"]
                                - right["two_cycle_phase_coverage"]
                            )
                            for left, right in zip(zero, one, strict=True)
                        ]
                    )
                ),
                "failure_disagreement_rate": float(
                    np.mean(
                        [
                            (left["route2_semantic_margin"] > 0)
                            != (right["route2_semantic_margin"] > 0)
                            for left, right in zip(zero, one, strict=True)
                        ]
                    )
                ),
            }
        )
    effect_noise = _q(
        [row["condition_effect_abs_delta"] for row in repeat_deltas],
        0.975,
    )
    coverage_noise = _q(
        [row["phase_coverage_abs_delta"] for row in repeat_deltas],
        0.975,
    )
    condition_effect_lower = (
        _q([row["condition_effect_mean"] for row in source_baseline], 0.025)
        - effect_noise
    )
    phase_coverage_lower = (
        _q([row["phase_coverage_mean"] for row in source_baseline], 0.025)
        - coverage_noise
    )
    anchor_thresholds = {}
    for episode_id in episode_ids:
        anchor_ids = sorted(
            {
                int(row["anchor_index"])
                for row in full
                if int(row["episode_id"]) == episode_id
            }
        )
        for anchor_id in anchor_ids:
            zero = indexed[(episode_id, anchor_id, 0)]
            one = indexed[(episode_id, anchor_id, 1)]
            anchor_thresholds[f"{episode_id}:{anchor_id}"] = {
                "condition_effect_lower": (
                    float(zero["condition_switch_action_effect"])
                    - abs(
                        float(zero["condition_switch_action_effect"])
                        - float(one["condition_switch_action_effect"])
                    )
                ),
                "phase_coverage_lower": (
                    float(zero["two_cycle_phase_coverage"])
                    - abs(
                        float(zero["two_cycle_phase_coverage"])
                        - float(one["two_cycle_phase_coverage"])
                    )
                ),
            }
    partial_thresholds = {
        "condition_effect_lower": condition_effect_lower,
        "phase_coverage_lower": phase_coverage_lower,
        "anchor_thresholds": anchor_thresholds,
    }
    baseline_failure_rates = []
    repeat_disagreement_rates = []
    for episode_id in episode_ids:
        anchor_ids = sorted(
            {
                int(row["anchor_index"])
                for row in full
                if int(row["episode_id"]) == episode_id
            }
        )
        zero = [indexed[(episode_id, anchor_id, 0)] for anchor_id in anchor_ids]
        one = [indexed[(episode_id, anchor_id, 1)] for anchor_id in anchor_ids]
        zero_failed = [
            camera_pair_failed(row, thresholds=partial_thresholds) for row in zero
        ]
        one_failed = [
            camera_pair_failed(row, thresholds=partial_thresholds) for row in one
        ]
        baseline_failure_rates.append(
            {
                "episode_id": episode_id,
                "failure_rate": float(np.mean(zero_failed)),
            }
        )
        repeat_disagreement_rates.append(
            {
                "episode_id": episode_id,
                "failure_disagreement_rate": float(
                    np.mean(
                        [
                            left != right
                            for left, right in zip(
                                zero_failed,
                                one_failed,
                                strict=True,
                            )
                        ]
                    )
                ),
            }
        )
    failure_rate_upper = min(
        1.0,
        _q([row["failure_rate"] for row in baseline_failure_rates], 0.975)
        + _q(
            [row["failure_disagreement_rate"] for row in repeat_disagreement_rates],
            0.975,
        ),
    )
    return {
        "schema": "simverify_e04_thresholds_v1",
        "condition_effect_lower": condition_effect_lower,
        "phase_coverage_lower": phase_coverage_lower,
        "failure_rate_upper": failure_rate_upper,
        "same_checkpoint_repeat_condition_effect_delta_q97_5": effect_noise,
        "same_checkpoint_repeat_phase_coverage_delta_q97_5": coverage_noise,
        "source_baseline": source_baseline,
        "repeat_deltas": repeat_deltas,
        "anchor_thresholds": anchor_thresholds,
        "baseline_failure_rates": baseline_failure_rates,
        "repeat_failure_disagreement_rates": repeat_disagreement_rates,
        "formula": {
            "retention_lower": (
                "unperturbed_B1_validation_q02_5 - "
                "same_checkpoint_repeat_abs_delta_q97_5"
            ),
            "failure_upper": (
                "unperturbed_B1_failure_q97_5 + "
                "same_checkpoint_repeat_disagreement_q97_5"
            ),
        },
        "validation_role": "development",
        "held_out_test_read": False,
    }


def aggregate_e04_by_source(
    rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate matched camera pairs within source episode."""

    selected = [row for row in rows if int(row["repeat"]) == 0]
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(int(row["episode_id"]), str(row["camera_variant"]))].append(row)
    result = []
    for (episode_id, variant), values in sorted(grouped.items()):
        failures = [camera_pair_failed(row, thresholds=thresholds) for row in values]
        result.append(
            {
                "schema": "simverify_e04_source_camera_metric_v1",
                "episode_id": episode_id,
                "camera_variant": variant,
                "pair_count": len(values),
                "condition_effect_mean": float(
                    np.mean([row["condition_switch_action_effect"] for row in values])
                ),
                "phase_coverage_mean": float(
                    np.mean([row["two_cycle_phase_coverage"] for row in values])
                ),
                "semantic_margin_mean": float(
                    np.mean([row["route2_semantic_margin"] for row in values])
                ),
                "failure_rate": float(np.mean(failures)),
                "event_order_valid_rate": float(
                    np.mean([row["event_order_valid"] for row in values])
                ),
                "ready_boundary_discontinuity_q95": _q(
                    [row["ready_boundary_discontinuity"] for row in values],
                    0.95,
                ),
                "minimum_route0_ticks": min(
                    int(row["second_cycle_route0_tick_count"]) for row in values
                ),
                "minimum_route2_ticks": min(
                    int(row["second_cycle_route2_tick_count"]) for row in values
                ),
                "shared_ready_route0_rate": float(
                    np.mean(
                        [
                            int(row["shared_ready_boundary_route_index"]) == 0
                            for row in values
                        ]
                    )
                ),
                "router_reset_exactly_once_rate": float(
                    np.mean(
                        [
                            int(row["condition_cycle_router_reset_count"]) == 1
                            for row in values
                        ]
                    )
                ),
            }
        )
    return result


def camera_pair_failed(
    row: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
) -> bool:
    anchor_threshold = thresholds.get("anchor_thresholds", {}).get(
        f"{int(row['episode_id'])}:{int(row['anchor_index'])}",
        {},
    )
    effect_lower = anchor_threshold.get(
        "condition_effect_lower",
        thresholds["condition_effect_lower"],
    )
    coverage_lower = anchor_threshold.get(
        "phase_coverage_lower",
        thresholds["phase_coverage_lower"],
    )
    return bool(
        row["condition_switch_action_effect"] < effect_lower
        or row["two_cycle_phase_coverage"] < coverage_lower
        or row["route2_semantic_margin"] <= 0.0
        or not row["event_order_valid"]
        or int(row["second_cycle_route0_tick_count"]) < 1
        or int(row["second_cycle_route2_tick_count"]) < 1
        or int(row["shared_ready_boundary_route_index"]) != 0
        or int(row["condition_cycle_router_reset_count"]) != 1
    )


def evaluate_e04_gate(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any],
    ready_upper: float,
) -> dict[str, Any]:
    """Evaluate every frozen E04 variant without hiding source failures."""

    criteria: dict[str, Any] = {}
    for variant in CAMERA_VARIANTS:
        rows = [row for row in source_rows if row["camera_variant"] == variant]
        if not rows:
            raise ValueError(f"E04 source metrics missing {variant}")
        criteria[variant] = {
            "condition_effect_source_q02_5": _q(
                [row["condition_effect_mean"] for row in rows],
                0.025,
            ),
            "condition_effect_lower": thresholds["condition_effect_lower"],
            "phase_coverage_source_q02_5": _q(
                [row["phase_coverage_mean"] for row in rows],
                0.025,
            ),
            "phase_coverage_lower": thresholds["phase_coverage_lower"],
            "semantic_margin_min_source_mean": min(
                row["semantic_margin_mean"] for row in rows
            ),
            "failure_rate_source_q97_5": _q(
                [row["failure_rate"] for row in rows],
                0.975,
            ),
            "failure_rate_upper": thresholds["failure_rate_upper"],
            "ready_discontinuity_max_source_q95": max(
                row["ready_boundary_discontinuity_q95"] for row in rows
            ),
            "ready_discontinuity_upper": ready_upper,
            "passed": (
                _q(
                    [row["condition_effect_mean"] for row in rows],
                    0.025,
                )
                >= thresholds["condition_effect_lower"]
                and _q(
                    [row["phase_coverage_mean"] for row in rows],
                    0.025,
                )
                >= thresholds["phase_coverage_lower"]
                and min(row["semantic_margin_mean"] for row in rows) > 0.0
                and _q(
                    [row["failure_rate"] for row in rows],
                    0.975,
                )
                <= thresholds["failure_rate_upper"]
                and all(
                    row["event_order_valid_rate"] == 1.0
                    and row["ready_boundary_discontinuity_q95"] <= ready_upper
                    and row["minimum_route0_ticks"] >= 1
                    and row["minimum_route2_ticks"] >= 1
                    and row["shared_ready_route0_rate"] == 1.0
                    and row["router_reset_exactly_once_rate"] == 1.0
                    for row in rows
                )
            ),
        }
    passed = all(row["passed"] for row in criteria.values())
    return {
        "schema": "simverify_e04_camera_counterfactual_gate_v1",
        "decision": (
            "e04_camera_counterfactual_robustness_established_development"
            if passed
            else "e04_camera_counterfactual_robustness_not_established"
        ),
        "authorizes_e05": passed,
        "criteria": criteria,
        "thresholds": dict(thresholds),
        "source_episode_metrics": list(source_rows),
        "validation_role": "development",
        "evidence_scope": EVIDENCE_SCOPE,
        "held_out_test_read": False,
        "closed_loop_execution": False,
    }


def _input_identity(
    root: Path,
    manifest_name: str,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": str(root),
        "manifest_sha256": sha256_file(root / manifest_name),
        "checksums_sha256": sha256_file(root / "checksums.sha256"),
        "verified_file_count": verification["verified_file_count"],
    }
