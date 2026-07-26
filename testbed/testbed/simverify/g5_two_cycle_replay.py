"""Continuous two-cycle recorded-path replay for the SimVerify G5 core Gate."""

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
import torch

from testbed.simverify.artifacts import (
    artifact_identity,
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.m3_condition_delta_stitch import (
    _load_policy,
    _validate_bundle,
)
from testbed.simverify.m3_condition_replay import _fit_sector_action_direction
from testbed.simverify.m3_replay import (
    CAMERAS,
    _read_camera_image,
    cycle_action_metrics,
)
from testbed.simverify.m3_transition_stitch import HELD_OUT_EPISODES, _q

EVIDENCE_SCOPE = "recorded-observation/offline teacher-forced development"
MODES = ("switched", "unchanged")


def build_g5_two_cycle_replay(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    m0_root: str | Path,
    m2_root: str | Path,
    b1_bundle_root: str | Path,
    b2_bundle_root: str | Path,
    contract_path: str | Path,
    previous_g5_root: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Build the immutable B1.4/B2.4 G5.1 development artifact."""

    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        not git.get("git_available")
        or git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
    ):
        raise ValueError("G5 two-cycle replay requires a clean worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable G5 package exists: {destination}")
    m0 = Path(m0_root).resolve(strict=True)
    m2 = Path(m2_root).resolve(strict=True)
    contract = Path(contract_path).resolve(strict=True)
    previous_g5 = Path(previous_g5_root).resolve(strict=True)
    m0_verification = verify_checksums(m0, m0 / "checksums.sha256")
    m2_verification = verify_checksums(m2, m2 / "checksums.sha256")
    previous_g5_verification = verify_checksums(
        previous_g5,
        previous_g5 / "checksums.sha256",
    )
    if (
        not m0_verification["ok"]
        or not m2_verification["ok"]
        or not previous_g5_verification["ok"]
    ):
        raise ValueError("G5 input package checksum verification failed")
    previous_gate = _read_json(previous_g5 / "g5_core_gate_v1.json")
    if previous_gate[
        "decision"
    ] != "g5_core_two_cycle_condition_continuity_not_established" or bool(
        previous_gate["held_out_test_read"]
    ):
        raise ValueError("G5.1 requires the frozen failed G5 core development result")

    split_manifest = _read_json(m0 / "split_groups.json")
    if set(map(int, split_manifest["splits"]["held_out_test"])) != HELD_OUT_EPISODES:
        raise ValueError("held-out split differs from the frozen contract")
    anchors = _read_jsonl(m2 / "two_cycle_anchors_v1.jsonl")
    if any(int(row["episode_id"]) in HELD_OUT_EPISODES for row in anchors):
        raise ValueError("held-out episode entered two-cycle anchors")
    by_split = {
        split: [row for row in anchors if row["split"] == split]
        for split in ("train", "validation")
    }
    if not by_split["train"] or not by_split["validation"]:
        raise ValueError("G5 requires train and validation two-cycle anchors")
    for anchor in anchors:
        if (
            anchor["first_condition"]["next_ready_sector"]
            != anchor["second_condition"]["current_sector"]
        ):
            raise ValueError("two-cycle condition continuity mismatch")
    support = build_two_cycle_condition_support(
        anchors,
        _read_jsonl(m2 / "condition_counterfactual_anchors_v1.jsonl"),
    )
    support_by_key = {
        (
            row["split"],
            int(row["episode_id"]),
            int(row["first_cycle_id"]),
        ): row
        for row in support["rows"]
    }

    envelope = _read_json(m2 / "expert_event_envelope_v1.json")
    templates = envelope["templates"]
    deadzone = list(map(float, envelope["effective_deadzone"]))
    annotations = [
        row
        for row in _read_jsonl(m0 / "cycle_annotations.jsonl")
        if row["quality"]["status"] == "accepted" and row["split"] == "train"
    ]
    direction = _fit_sector_action_direction(
        annotations,
        m0,
        deadzone=deadzone,
    )
    expert_train = build_expert_two_cycle_metrics(
        by_split["train"],
        m0_root=m0,
        templates=templates,
        deadzone=deadzone,
    )
    expert_validation = build_expert_two_cycle_metrics(
        by_split["validation"],
        m0_root=m0,
        templates=templates,
        deadzone=deadzone,
    )
    expert_train_sources = aggregate_expert_by_episode(expert_train)
    expert_validation_sources = aggregate_expert_by_episode(expert_validation)
    thresholds = derive_expert_two_cycle_thresholds(expert_train_sources)
    expert_gate = evaluate_expert_two_cycle_gate(
        expert_validation_sources,
        thresholds,
    )

    max_steps = max(
        int(row["target_steps_20hz"][1]) - int(row["target_steps_20hz"][0]) + 1
        for row in by_split["validation"]
    )
    bundles = {
        "B1.4": _validate_bundle(Path(b1_bundle_root).resolve(strict=True), "B1.4"),
        "B2.4": _validate_bundle(Path(b2_bundle_root).resolve(strict=True), "B2.4"),
    }
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    try:
        for baseline_id, bundle in bundles.items():
            policy = _load_policy(bundle, max_steps=max_steps, device=device)
            grouped: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
            for anchor_index, anchor in enumerate(by_split["validation"]):
                grouped[int(anchor["episode_id"])].append((anchor_index, anchor))
            for episode_id, episode_anchors in sorted(grouped.items()):
                with h5py.File(
                    m0 / f"episodes/episode_{episode_id}.hdf5", "r"
                ) as episode:
                    for anchor_index, anchor in episode_anchors:
                        for mode in MODES:
                            arrays = replay_two_cycle_arrays(
                                policy=policy,
                                episode=episode,
                                anchor=anchor,
                                condition_mode=mode,
                                reset_condition_cycle_at_boundary=True,
                            )
                            metrics = two_cycle_trace_metrics(
                                arrays,
                                templates=templates,
                                deadzone=deadzone,
                            )
                            relative = (
                                Path("traces")
                                / baseline_id.lower().replace(".", "_")
                                / (
                                    f"anchor_{anchor_index}_episode_{episode_id}_"
                                    f"cycles_{anchor['first_cycle_id']}_"
                                    f"{anchor['second_cycle_id']}_{mode}.npz"
                                )
                            )
                            trace_path = temporary / relative
                            trace_path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(trace_path, **arrays)
                            identity = artifact_identity(trace_path)
                            identities.append(identity)
                            result_rows.append(
                                {
                                    "schema": "simverify_g5_two_cycle_result_v1",
                                    "anchor_index": anchor_index,
                                    "episode_id": episode_id,
                                    "first_cycle_id": int(anchor["first_cycle_id"]),
                                    "second_cycle_id": int(anchor["second_cycle_id"]),
                                    "baseline_id": baseline_id,
                                    "condition_mode": mode,
                                    "first_condition": anchor["first_condition"],
                                    "second_condition": anchor["second_condition"],
                                    "next_target_changed": (
                                        anchor["first_condition"]["next_ready_sector"]
                                        != anchor["second_condition"][
                                            "next_ready_sector"
                                        ]
                                    ),
                                    "condition_switch_counterfactual_supported": bool(
                                        support_by_key[
                                            (
                                                "validation",
                                                episode_id,
                                                int(anchor["first_cycle_id"]),
                                            )
                                        ]["supported"]
                                    ),
                                    "trace_path": str(relative),
                                    "trace_sha256": identity["sha256"],
                                    "evidence_scope": EVIDENCE_SCOPE,
                                    "closed_loop_execution": False,
                                    **metrics,
                                }
                            )
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        attach_switch_effects_to_results(result_rows, temporary)
        paired_rows = build_condition_switch_metrics(
            result_rows,
            direction=direction,
        )
        source_rows = aggregate_g5_by_episode(result_rows, paired_rows)
        gate = evaluate_g5_core_gate(
            expert_gate=expert_gate,
            thresholds=thresholds,
            source_rows=source_rows,
            support_threshold=int(support["train_source_episode_minimum"]),
        )
        identities.extend(
            [
                write_jsonl(
                    temporary / "expert_train_two_cycle_metrics.jsonl",
                    expert_train,
                ),
                write_jsonl(
                    temporary / "expert_validation_two_cycle_metrics.jsonl",
                    expert_validation,
                ),
                write_jsonl(
                    temporary / "g5_two_cycle_results.jsonl",
                    result_rows,
                ),
                write_jsonl(
                    temporary / "g5_condition_switch_metrics.jsonl",
                    paired_rows,
                ),
                write_jsonl(
                    temporary / "g5_source_episode_metrics.jsonl",
                    source_rows,
                ),
                write_json(
                    temporary / "g5_expert_thresholds_v1.json",
                    {
                        "schema": "simverify_g5_expert_threshold_package_v1",
                        "thresholds": thresholds,
                        "train_source_episode_metrics": expert_train_sources,
                        "validation_source_episode_metrics": (
                            expert_validation_sources
                        ),
                        "validation_gate": expert_gate,
                        "source": "train_adjacent_pair_then_source_episode",
                        "held_out_test_read": False,
                    },
                ),
                write_json(
                    temporary / "g5_sector_action_direction_v1.json",
                    direction,
                ),
                write_json(
                    temporary / "g5_condition_switch_support_v1.json",
                    support,
                ),
            ]
        )
        gate_identity = write_json(temporary / "g5_core_gate_v1.json", gate)
        identities.append(gate_identity)
        manifest_identity = write_json(
            temporary / "g5_two_cycle_manifest.json",
            {
                "schema": "simverify_g5_two_cycle_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "contract": {
                    "path": str(contract),
                    "sha256": sha256_file(contract),
                },
                "m0": {
                    "path": str(m0),
                    "dataset_manifest_sha256": sha256_file(
                        m0 / "dataset_manifest.json"
                    ),
                    "checksums_sha256": sha256_file(m0 / "checksums.sha256"),
                    "verified_file_count": m0_verification["verified_file_count"],
                },
                "m2": {
                    "path": str(m2),
                    "manifest_sha256": sha256_file(m2 / "m2_manifest.json"),
                    "checksums_sha256": sha256_file(m2 / "checksums.sha256"),
                    "verified_file_count": m2_verification["verified_file_count"],
                    "two_cycle_anchors_sha256": sha256_file(
                        m2 / "two_cycle_anchors_v1.jsonl"
                    ),
                },
                "previous_g5_core": {
                    "path": str(previous_g5),
                    "manifest_sha256": sha256_file(
                        previous_g5 / "g5_two_cycle_manifest.json"
                    ),
                    "gate_sha256": sha256_file(previous_g5 / "g5_core_gate_v1.json"),
                    "checksums_sha256": sha256_file(previous_g5 / "checksums.sha256"),
                    "verified_file_count": previous_g5_verification[
                        "verified_file_count"
                    ],
                    "decision": previous_gate["decision"],
                },
                "bundles": {
                    baseline_id: bundle["identity"]
                    for baseline_id, bundle in bundles.items()
                },
                "train_pair_count": len(by_split["train"]),
                "validation_pair_count": len(by_split["validation"]),
                "validation_changed_next_target_pair_count": sum(
                    row["first_condition"]["next_ready_sector"]
                    != row["second_condition"]["next_ready_sector"]
                    for row in by_split["validation"]
                ),
                "condition_update": "atomic_at_shared_ready_boundary",
                "condition_cycle_router_reset_at_boundary": True,
                "policy_reset_count_per_trace": 1,
                "temporal_aggregation_reset_at_boundary": False,
                "environment_response": "recorded_teacher_forced",
                "condition_switch_support": {
                    "train_supported_changed_pair_count": support[
                        "train_supported_changed_pair_count"
                    ],
                    "validation_supported_changed_pair_count": support[
                        "validation_supported_changed_pair_count"
                    ],
                    "train_source_episode_minimum": support[
                        "train_source_episode_minimum"
                    ],
                    "counterfactual_anchor_sha256": sha256_file(
                        m2 / "condition_counterfactual_anchors_v1.jsonl"
                    ),
                },
                "decision": gate["decision"],
                "authorizes_remaining_g5_robustness": gate[
                    "authorizes_remaining_g5_robustness"
                ],
                "independent_validation": False,
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
            "authorizes_remaining_g5_robustness": gate[
                "authorizes_remaining_g5_robustness"
            ],
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
                    "schema": "simverify_g5_two_cycle_failure_v1",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "git": git,
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                },
            )
        raise


def replay_two_cycle_arrays(
    *,
    policy: Any,
    episode: h5py.File,
    anchor: Mapping[str, Any],
    condition_mode: str,
    reset_condition_cycle_at_boundary: bool = False,
    camera_names: Sequence[str] = CAMERAS,
) -> dict[str, np.ndarray]:
    """Replay one adjacent pair with one reset and an atomic condition update."""

    if condition_mode not in MODES:
        raise ValueError(f"unknown two-cycle condition mode: {condition_mode}")
    start, end = map(int, anchor["target_steps_20hz"])
    boundary = int(anchor["shared_ready_boundary_tick"])
    total_steps = int(episode["action"].shape[0])
    if not 0 <= start < boundary < end < total_steps:
        raise ValueError("two-cycle ticks do not satisfy start < boundary < end")
    first = np.asarray(anchor["first_condition"]["vector"], dtype=np.float32)
    second = np.asarray(anchor["second_condition"]["vector"], dtype=np.float32)
    if first.shape != (6,) or second.shape != (6,):
        raise ValueError("two-cycle conditions must have shape (6,)")
    if hasattr(policy, "reset"):
        policy.reset()

    aggregated = []
    raw_normalized = []
    raw_direct = []
    delivered = []
    route_index = []
    route_pending = []
    condition_cycle_reset_count = 0
    ticks = np.arange(start, end + 1, dtype=np.int64)
    for tick in ticks.tolist():
        if tick == boundary and reset_condition_cycle_at_boundary:
            reset_condition_cycle = getattr(
                policy,
                "reset_condition_cycle",
                None,
            )
            if not callable(reset_condition_cycle):
                raise ValueError(
                    "condition-cycle boundary requires reset_condition_cycle"
                )
            reset_condition_cycle()
            condition_cycle_reset_count += 1
        condition = (
            first if condition_mode == "unchanged" or tick < boundary else second
        )
        observation: dict[str, Any] = {
            "qpos": np.asarray(episode["observations/qpos"][tick], dtype=np.float32),
            "qvel": np.asarray(episode["observations/qvel"][tick], dtype=np.float32),
            "cycle_condition_v1": condition.copy(),
        }
        for camera in camera_names:
            observation[f"image_{camera}"] = _read_camera_image(
                episode,
                camera,
                tick,
            )
        aggregated.append(
            np.asarray(policy.predict(observation), dtype=np.float32).reshape(4)
        )
        raw_normalized.append(
            np.asarray(policy.last_raw_action_chunk(), dtype=np.float32)
        )
        raw_direct.append(
            np.asarray(policy.last_raw_action_chunk_direct(), dtype=np.float32)
        )
        delivered.append(condition.copy())
        diagnostics = getattr(policy, "condition_route_diagnostics", None)
        route_index.append(
            -1 if diagnostics is None else int(diagnostics["route_index"])
        )
        route_pending.append(
            -1 if diagnostics is None else int(diagnostics["consecutive_pending"])
        )
    boundary_index = boundary - start
    return {
        "raw_policy_chunk_normalized": np.stack(raw_normalized).astype(np.float32),
        "raw_policy_chunk_direct": np.stack(raw_direct).astype(np.float32),
        "temporal_aggregation_action": np.stack(aggregated).astype(np.float32),
        "future_runtime_safe_action": np.stack(aggregated).astype(np.float32),
        "expert_action": np.asarray(
            episode["action"][start : end + 1], dtype=np.float32
        ),
        "condition": np.stack(delivered).astype(np.float32),
        "condition_route_index": np.asarray(route_index, dtype=np.int8),
        "condition_route_pending_count": np.asarray(route_pending, dtype=np.int16),
        "target_tick": ticks,
        "shared_ready_boundary_local_index": np.asarray(boundary_index, dtype=np.int64),
        "condition_cycle_router_reset_count": np.asarray(
            condition_cycle_reset_count, dtype=np.int8
        ),
    }


def two_cycle_trace_metrics(
    arrays: Mapping[str, np.ndarray],
    *,
    templates: Mapping[str, Mapping[str, Any]],
    deadzone: Sequence[float],
) -> dict[str, Any]:
    """Compute task continuity without counting the shared boundary twice."""

    boundary = int(arrays["shared_ready_boundary_local_index"])
    policy = np.asarray(arrays["temporal_aggregation_action"], dtype=np.float32)
    expert = np.asarray(arrays["expert_action"], dtype=np.float32)
    if not 0 < boundary < policy.shape[0] - 1 or expert.shape != policy.shape:
        raise ValueError("two-cycle trace shapes or boundary are invalid")
    first = cycle_action_metrics(
        {
            "temporal_aggregation_action": policy[: boundary + 1],
            "expert_action": expert[: boundary + 1],
        },
        templates=templates,
        deadzone=deadzone,
    )
    second = cycle_action_metrics(
        {
            "temporal_aggregation_action": policy[boundary:],
            "expert_action": expert[boundary:],
        },
        templates=templates,
        deadzone=deadzone,
    )
    discontinuity = float(
        np.sqrt(np.mean(np.square(policy[boundary] - policy[boundary - 1])))
    )
    return {
        "step_count": int(policy.shape[0]),
        "shared_ready_boundary_local_index": boundary,
        "first_cycle_required_event_coverage": float(first["required_event_coverage"]),
        "second_cycle_required_event_coverage": float(
            second["required_event_coverage"]
        ),
        "two_cycle_phase_coverage": float(
            0.5
            * (
                float(first["required_event_coverage"])
                + float(second["required_event_coverage"])
            )
        ),
        "first_cycle_event_order_valid": bool(first["event_order_valid"]),
        "second_cycle_event_order_valid": bool(second["event_order_valid"]),
        "two_cycle_event_order_valid": bool(
            first["event_order_valid"] and second["event_order_valid"]
        ),
        "ready_boundary_discontinuity": discontinuity,
        "second_cycle_route2_tick_count": int(
            np.sum(np.asarray(arrays["condition_route_index"])[boundary:] == 2)
        ),
        "second_cycle_route0_tick_count": int(
            np.sum(np.asarray(arrays["condition_route_index"])[boundary:] == 0)
        ),
        "shared_ready_boundary_route_index": int(
            np.asarray(arrays["condition_route_index"])[boundary]
        ),
        "condition_cycle_router_reset_count": int(
            arrays["condition_cycle_router_reset_count"]
        ),
    }


def build_expert_two_cycle_metrics(
    anchors: Sequence[Mapping[str, Any]],
    *,
    m0_root: Path,
    templates: Mapping[str, Mapping[str, Any]],
    deadzone: Sequence[float],
) -> list[dict[str, Any]]:
    """Build recorded expert continuity metrics with no policy inference."""

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        grouped[int(anchor["episode_id"])].append(anchor)
    result = []
    for episode_id, episode_anchors in sorted(grouped.items()):
        with h5py.File(m0_root / f"episodes/episode_{episode_id}.hdf5", "r") as episode:
            for anchor in episode_anchors:
                start, end = map(int, anchor["target_steps_20hz"])
                boundary = int(anchor["shared_ready_boundary_tick"]) - start
                expert = np.asarray(
                    episode["action"][start : end + 1], dtype=np.float32
                )
                first = cycle_action_metrics(
                    {
                        "temporal_aggregation_action": expert[: boundary + 1],
                        "expert_action": expert[: boundary + 1],
                    },
                    templates=templates,
                    deadzone=deadzone,
                )
                second = cycle_action_metrics(
                    {
                        "temporal_aggregation_action": expert[boundary:],
                        "expert_action": expert[boundary:],
                    },
                    templates=templates,
                    deadzone=deadzone,
                )
                result.append(
                    {
                        "schema": "simverify_g5_expert_two_cycle_metric_v1",
                        "episode_id": episode_id,
                        "first_cycle_id": int(anchor["first_cycle_id"]),
                        "second_cycle_id": int(anchor["second_cycle_id"]),
                        "two_cycle_phase_coverage": float(
                            0.5
                            * (
                                float(first["required_event_coverage"])
                                + float(second["required_event_coverage"])
                            )
                        ),
                        "two_cycle_event_order_valid": bool(
                            first["event_order_valid"] and second["event_order_valid"]
                        ),
                        "ready_boundary_discontinuity": float(
                            np.sqrt(
                                np.mean(
                                    np.square(expert[boundary] - expert[boundary - 1])
                                )
                            )
                        ),
                        "closed_loop_execution": False,
                    }
                )
    return result


def aggregate_expert_by_episode(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_id"])].append(row)
    return [
        {
            "episode_id": episode_id,
            "pair_count": len(episode_rows),
            "phase_coverage_mean": float(
                np.mean([row["two_cycle_phase_coverage"] for row in episode_rows])
            ),
            "event_order_valid_rate": float(
                np.mean([row["two_cycle_event_order_valid"] for row in episode_rows])
            ),
            "ready_boundary_discontinuity_q95": _q(
                [row["ready_boundary_discontinuity"] for row in episode_rows],
                0.95,
            ),
        }
        for episode_id, episode_rows in sorted(grouped.items())
    ]


def derive_expert_two_cycle_thresholds(
    train_source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "simverify_g5_expert_two_cycle_thresholds_v1",
        "two_cycle_phase_coverage_lower": _q(
            [row["phase_coverage_mean"] for row in train_source_rows],
            0.025,
        ),
        "two_cycle_event_order_valid_rate_lower": _q(
            [row["event_order_valid_rate"] for row in train_source_rows],
            0.025,
        ),
        "ready_boundary_discontinuity_q95_upper": _q(
            [row["ready_boundary_discontinuity_q95"] for row in train_source_rows],
            0.975,
        ),
        "source_episode_count": len(train_source_rows),
        "source": "train_adjacent_pair_then_source_episode",
    }


def evaluate_expert_two_cycle_gate(
    validation_source_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = {
        "phase_coverage": {
            "observed_min_source_episode_mean": min(
                row["phase_coverage_mean"] for row in validation_source_rows
            ),
            "minimum_allowed": thresholds["two_cycle_phase_coverage_lower"],
        },
        "event_order": {
            "observed_min_source_episode_rate": min(
                row["event_order_valid_rate"] for row in validation_source_rows
            ),
            "minimum_allowed": thresholds["two_cycle_event_order_valid_rate_lower"],
        },
        "ready_boundary_discontinuity": {
            "observed_max_source_episode_q95": max(
                row["ready_boundary_discontinuity_q95"]
                for row in validation_source_rows
            ),
            "maximum_allowed": thresholds["ready_boundary_discontinuity_q95_upper"],
        },
    }
    for criterion in criteria.values():
        observed_key = next(key for key in criterion if key.startswith("observed_"))
        if "minimum_allowed" in criterion:
            criterion["passed"] = bool(
                criterion[observed_key] >= criterion["minimum_allowed"]
            )
        else:
            criterion["passed"] = bool(
                criterion[observed_key] <= criterion["maximum_allowed"]
            )
    return {
        "schema": "simverify_g5_expert_validation_gate_v1",
        "criteria": criteria,
        "passed": all(bool(row["passed"]) for row in criteria.values()),
        "validation_source_episode_count": len(validation_source_rows),
    }


def build_two_cycle_condition_support(
    anchors: Sequence[Mapping[str, Any]],
    counterfactual_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join each second cycle to the frozen next-sector support decision."""

    counterfactual = {
        (
            str(row["split"]),
            int(row["episode_id"]),
            int(row["cycle_id"]),
            str(row["target_condition"]["next_sector"]),
        ): row
        for row in counterfactual_rows
        if row["changed_factors"] == ["next_sector"]
    }
    rows = []
    for anchor in anchors:
        base_target = str(anchor["first_condition"]["next_ready_sector"])
        switched_target = str(anchor["second_condition"]["next_ready_sector"])
        changed = base_target != switched_target
        support_row = counterfactual.get(
            (
                str(anchor["split"]),
                int(anchor["episode_id"]),
                int(anchor["second_cycle_id"]),
                base_target,
            )
        )
        if changed and support_row is None:
            raise ValueError("changed two-cycle pair lacks counterfactual registry row")
        rows.append(
            {
                "schema": "simverify_g5_condition_switch_support_row_v1",
                "split": str(anchor["split"]),
                "episode_id": int(anchor["episode_id"]),
                "first_cycle_id": int(anchor["first_cycle_id"]),
                "second_cycle_id": int(anchor["second_cycle_id"]),
                "unchanged_next_target": base_target,
                "switched_next_target": switched_target,
                "next_target_changed": changed,
                "supported": bool(
                    not changed or (support_row and support_row["supported"])
                ),
                "support_status": (
                    "not_applicable_same_target"
                    if not changed
                    else str(support_row["status"])
                ),
            }
        )
    counts: dict[str, dict[int, int]] = {
        "train": defaultdict(int),
        "validation": defaultdict(int),
    }
    for row in rows:
        if row["next_target_changed"] and row["supported"]:
            counts[row["split"]][int(row["episode_id"])] += 1
    train_values = list(counts["train"].values())
    if not train_values:
        raise ValueError("G5 train pairs have no supported condition switch")
    return {
        "schema": "simverify_g5_condition_switch_support_v1",
        "rows": rows,
        "train_supported_changed_pair_count": int(sum(counts["train"].values())),
        "validation_supported_changed_pair_count": int(
            sum(counts["validation"].values())
        ),
        "train_supported_source_episode_counts": {
            str(key): value for key, value in sorted(counts["train"].items())
        },
        "validation_supported_source_episode_counts": {
            str(key): value for key, value in sorted(counts["validation"].items())
        },
        "train_source_episode_minimum": int(min(train_values)),
        "held_out_test_read": False,
    }


def build_condition_switch_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    direction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    indexed = {
        (
            str(row["baseline_id"]),
            int(row["anchor_index"]),
            str(row["condition_mode"]),
        ): row
        for row in rows
    }
    centers = direction["sector_swing_qpos_median"]
    action_sign = int(direction["action_to_qpos_direction_sign"])
    result = []
    for baseline_id in ("B1.4", "B2.4"):
        anchor_ids = sorted(
            {
                int(row["anchor_index"])
                for row in rows
                if row["baseline_id"] == baseline_id
            }
        )
        for anchor_index in anchor_ids:
            switched = indexed[(baseline_id, anchor_index, "switched")]
            unchanged = indexed[(baseline_id, anchor_index, "unchanged")]
            result.append(
                _condition_switch_metric_from_rows(
                    switched,
                    unchanged,
                    centers=centers,
                    action_sign=action_sign,
                )
            )
    return result


def _condition_switch_metric_from_rows(
    switched: Mapping[str, Any],
    unchanged: Mapping[str, Any],
    *,
    centers: Mapping[str, float],
    action_sign: int,
) -> dict[str, Any]:
    """Placeholder populated by trace summaries attached before pairing."""

    action_effect = float(switched["switch_action_effect"])
    swing_delta = float(switched["switch_route2_swing_delta_mean"])
    base_sector = str(switched["first_condition"]["next_ready_sector"])
    target_sector = str(switched["second_condition"]["next_ready_sector"])
    qpos_direction = int(
        np.sign(float(centers[target_sector]) - float(centers[base_sector]))
    )
    expected_sign = qpos_direction * action_sign
    semantic_margin = expected_sign * swing_delta if expected_sign != 0 else 0.0
    return {
        "schema": "simverify_g5_condition_switch_metric_v1",
        "anchor_index": int(switched["anchor_index"]),
        "episode_id": int(switched["episode_id"]),
        "baseline_id": str(switched["baseline_id"]),
        "next_target_changed": bool(switched["next_target_changed"]),
        "counterfactual_supported": bool(
            switched["condition_switch_counterfactual_supported"]
        ),
        "base_next_sector": base_sector,
        "target_next_sector": target_sector,
        "switch_action_effect": action_effect,
        "route2_swing_delta_mean": swing_delta,
        "expected_swing_action_sign": expected_sign,
        "route2_semantic_margin": semantic_margin,
        "switched_trace_sha256": switched["trace_sha256"],
        "unchanged_trace_sha256": unchanged["trace_sha256"],
        "closed_loop_execution": False,
    }


def aggregate_g5_by_episode(
    result_rows: Sequence[Mapping[str, Any]],
    switch_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    switched = [row for row in result_rows if row["condition_mode"] == "switched"]
    grouped_results: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    grouped_switch: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in switched:
        grouped_results[(int(row["episode_id"]), str(row["baseline_id"]))].append(row)
    for row in switch_rows:
        if row["next_target_changed"] and row["counterfactual_supported"]:
            grouped_switch[(int(row["episode_id"]), str(row["baseline_id"]))].append(
                row
            )
    result = []
    for episode_id in sorted({key[0] for key in grouped_results}):
        b1 = grouped_results[(episode_id, "B1.4")]
        b1_switch = grouped_switch[(episode_id, "B1.4")]
        b2_switch = grouped_switch[(episode_id, "B2.4")]
        if not b1_switch or not b2_switch:
            raise ValueError("each validation source episode needs changed targets")
        b1_semantic = float(
            np.mean([row["route2_semantic_margin"] for row in b1_switch])
        )
        b2_semantic = float(
            np.mean([row["route2_semantic_margin"] for row in b2_switch])
        )
        result.append(
            {
                "schema": "simverify_g5_source_episode_metric_v1",
                "episode_id": episode_id,
                "pair_count": len(b1),
                "changed_target_pair_count": len(b1_switch),
                "b1_4_two_cycle_phase_coverage_min": min(
                    float(row["two_cycle_phase_coverage"]) for row in b1
                ),
                "b1_4_phase_coverage_mean": float(
                    np.mean([row["two_cycle_phase_coverage"] for row in b1])
                ),
                "b1_4_event_order_valid_rate": float(
                    np.mean([row["two_cycle_event_order_valid"] for row in b1])
                ),
                "b1_4_ready_boundary_discontinuity_q95": _q(
                    [row["ready_boundary_discontinuity"] for row in b1],
                    0.95,
                ),
                "b1_4_second_cycle_route2_min_ticks": min(
                    int(row["second_cycle_route2_tick_count"]) for row in b1
                ),
                "b1_4_second_cycle_route0_min_ticks": min(
                    int(row["second_cycle_route0_tick_count"]) for row in b1
                ),
                "b1_4_shared_ready_route0_rate": float(
                    np.mean(
                        [
                            int(row["shared_ready_boundary_route_index"]) == 0
                            for row in b1
                        ]
                    )
                ),
                "b1_4_condition_cycle_reset_count_min": min(
                    int(row["condition_cycle_router_reset_count"]) for row in b1
                ),
                "b1_4_condition_cycle_reset_count_max": max(
                    int(row["condition_cycle_router_reset_count"]) for row in b1
                ),
                "b1_4_switch_action_effect_mean": float(
                    np.mean([row["switch_action_effect"] for row in b1_switch])
                ),
                "b1_4_route2_semantic_margin_mean": b1_semantic,
                "b2_4_route2_semantic_margin_mean": b2_semantic,
                "b1_4_minus_b2_4_semantic_margin": (b1_semantic - b2_semantic),
            }
        )
    return result


def evaluate_g5_core_gate(
    *,
    expert_gate: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    support_threshold: int,
) -> dict[str, Any]:
    criteria = {
        "expert_validation_continuity": {
            "passed": bool(expert_gate["passed"]),
        },
        "b1_4_two_cycle_phase_coverage": {
            "observed_min_source_episode_mean": min(
                row["b1_4_phase_coverage_mean"] for row in source_rows
            ),
            "minimum_allowed": thresholds["two_cycle_phase_coverage_lower"],
        },
        "b1_4_two_cycle_event_order": {
            "observed_min_source_episode_rate": min(
                row["b1_4_event_order_valid_rate"] for row in source_rows
            ),
            "minimum_allowed": thresholds["two_cycle_event_order_valid_rate_lower"],
        },
        "b1_4_ready_boundary_discontinuity": {
            "observed_max_source_episode_q95": max(
                row["b1_4_ready_boundary_discontinuity_q95"] for row in source_rows
            ),
            "maximum_allowed": thresholds["ready_boundary_discontinuity_q95_upper"],
        },
        "b1_4_second_cycle_route_activation": {
            "observed_min_ticks": min(
                row["b1_4_second_cycle_route2_min_ticks"] for row in source_rows
            ),
            "minimum_allowed": 1,
        },
        "b1_4_second_cycle_route_restart": {
            "observed_min_route0_ticks": min(
                row["b1_4_second_cycle_route0_min_ticks"] for row in source_rows
            ),
            "minimum_allowed": 1,
        },
        "b1_4_shared_ready_route_is_current": {
            "observed_min_source_episode_rate": min(
                row["b1_4_shared_ready_route0_rate"] for row in source_rows
            ),
            "minimum_allowed": 1.0,
        },
        "condition_cycle_reset_exactly_once": {
            "observed_min_count": min(
                row["b1_4_condition_cycle_reset_count_min"] for row in source_rows
            ),
            "observed_max_count": max(
                row["b1_4_condition_cycle_reset_count_max"] for row in source_rows
            ),
            "required": 1,
            "passed": all(
                row["b1_4_condition_cycle_reset_count_min"] == 1
                and row["b1_4_condition_cycle_reset_count_max"] == 1
                for row in source_rows
            ),
        },
        "supported_condition_switch_coverage": {
            "observed_min_source_episode_count": min(
                row["changed_target_pair_count"] for row in source_rows
            ),
            "minimum_allowed": int(support_threshold),
        },
        "b1_4_condition_switch_effect": {
            "observed_min_source_episode_mean": min(
                row["b1_4_switch_action_effect_mean"] for row in source_rows
            ),
            "minimum_allowed": 0.0,
            "comparison": "strictly_greater",
            "passed": all(
                row["b1_4_switch_action_effect_mean"] > 0.0 for row in source_rows
            ),
        },
        "b1_4_route2_semantics": {
            "observed_min_source_episode_mean": min(
                row["b1_4_route2_semantic_margin_mean"] for row in source_rows
            ),
            "minimum_allowed": 0.0,
            "comparison": "strictly_greater",
            "passed": all(
                row["b1_4_route2_semantic_margin_mean"] > 0.0 for row in source_rows
            ),
        },
        "b1_4_exceeds_b2_4_semantics": {
            "observed_min_source_episode_delta": min(
                row["b1_4_minus_b2_4_semantic_margin"] for row in source_rows
            ),
            "minimum_allowed": 0.0,
            "comparison": "strictly_greater",
            "passed": all(
                row["b1_4_minus_b2_4_semantic_margin"] > 0.0 for row in source_rows
            ),
        },
    }
    for criterion in criteria.values():
        if "passed" in criterion:
            continue
        observed_key = next(key for key in criterion if key.startswith("observed_"))
        if "minimum_allowed" in criterion:
            criterion["passed"] = bool(
                criterion[observed_key] >= criterion["minimum_allowed"]
            )
        else:
            criterion["passed"] = bool(
                criterion[observed_key] <= criterion["maximum_allowed"]
            )
    passed = all(bool(row["passed"]) for row in criteria.values())
    return {
        "schema": "simverify_g5_core_gate_v1",
        "decision": (
            "g5_core_two_cycle_condition_continuity_established_development"
            if passed
            else "g5_core_two_cycle_condition_continuity_not_established"
        ),
        "authorizes_remaining_g5_robustness": passed,
        "criteria": criteria,
        "expert_validation_gate": expert_gate,
        "source_episode_metrics": list(source_rows),
        "condition_switch_support_minimum": int(support_threshold),
        "independent_validation": False,
        "validation_role": "development",
        "evidence_scope": EVIDENCE_SCOPE,
        "held_out_test_read": False,
        "closed_loop_execution": False,
    }


def attach_switch_effects_to_results(
    rows: list[dict[str, Any]],
    trace_root: Path,
) -> None:
    """Attach switched-vs-unchanged action effects before JSON serialization."""

    indexed = {
        (
            str(row["baseline_id"]),
            int(row["anchor_index"]),
            str(row["condition_mode"]),
        ): row
        for row in rows
    }
    for baseline_id in ("B1.4", "B2.4"):
        anchor_ids = sorted(
            {
                int(row["anchor_index"])
                for row in rows
                if row["baseline_id"] == baseline_id
            }
        )
        for anchor_id in anchor_ids:
            switched = indexed[(baseline_id, anchor_id, "switched")]
            unchanged = indexed[(baseline_id, anchor_id, "unchanged")]
            with (
                np.load(
                    trace_root / str(switched["trace_path"]), allow_pickle=False
                ) as first,
                np.load(
                    trace_root / str(unchanged["trace_path"]), allow_pickle=False
                ) as second,
            ):
                action_a = np.asarray(
                    first["future_runtime_safe_action"], dtype=np.float32
                )
                action_b = np.asarray(
                    second["future_runtime_safe_action"], dtype=np.float32
                )
                boundary = int(first["shared_ready_boundary_local_index"])
                if action_a.shape != action_b.shape:
                    raise ValueError("switched and unchanged trace shapes differ")
                route_mask = (
                    np.asarray(first["condition_route_index"])[boundary:] == 2
                ) | (np.asarray(second["condition_route_index"])[boundary:] == 2)
                delta = action_a[boundary:] - action_b[boundary:]
                effect = float(np.mean(np.abs(delta)))
                swing = (
                    float(np.mean(delta[route_mask, 0])) if np.any(route_mask) else 0.0
                )
            switched["switch_action_effect"] = effect
            switched["switch_route2_swing_delta_mean"] = swing
            unchanged["switch_action_effect"] = effect
            unchanged["switch_route2_swing_delta_mean"] = swing


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
