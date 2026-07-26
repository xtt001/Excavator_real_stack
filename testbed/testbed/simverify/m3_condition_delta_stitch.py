"""B1.4 versus B2.4 conditioned policy rollouts through G4-v3 stitching."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from testbed.policies.offline_eval import load_policy_for_episode
from testbed.simverify.artifacts import (
    artifact_identity,
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.m3_condition_replay import _validate_condition_checkpoint
from testbed.simverify.m3_replay import CAMERAS, _read_camera_image
from testbed.simverify.m3_transition_delta_stitch import (
    ACTION_DIM,
    TARGET_PROGRESS,
)
from testbed.simverify.m3_transition_stitch import (
    HELD_OUT_EPISODES,
    apply_standardization,
    compose_retrieval_vectors,
)

EVIDENCE_SCOPE = "recorded-observation/offline development"
SECTORS = ("left", "center", "right")


def build_condition_delta_stitch_experiment(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    m0_root: str | Path,
    m2_root: str | Path,
    transition_stitch_root: str | Path,
    delta_stitch_audit_root: str | Path,
    next_condition_support_root: str | Path,
    b1_bundle_root: str | Path,
    b2_bundle_root: str | Path,
    contract_path: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run immutable B1.4/B2.4 supported-path condition interventions."""

    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        not git.get("git_available")
        or git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
    ):
        raise ValueError("condition delta stitch requires a clean worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(
            f"immutable condition delta-stitch package exists: {destination}"
        )
    m0 = Path(m0_root).resolve(strict=True)
    m2 = Path(m2_root).resolve(strict=True)
    transition_root = Path(transition_stitch_root).resolve(strict=True)
    audit_root = Path(delta_stitch_audit_root).resolve(strict=True)
    support_root = Path(next_condition_support_root).resolve(strict=True)
    contract = Path(contract_path).resolve(strict=True)
    for root in (transition_root, audit_root, support_root):
        verification = verify_checksums(root, root / "checksums.sha256")
        if not verification["ok"]:
            raise ValueError(f"checksum verification failed: {root}")

    audit_gate = _read_json(audit_root / "delta_stitch_gate_audit_v1.json")
    if not bool(audit_gate["authorizes_b1_4_policy_stitch_development"]):
        raise ValueError("expert delta-stitch development prerequisite did not pass")
    support_gate = _read_json(support_root / "next_condition_support_gate_v1.json")
    if not bool(support_gate["authorizes_b1_4_training"]):
        raise ValueError("next-condition support prerequisite did not pass")
    eligible_episodes = set(
        map(int, support_gate["eligible_validation_source_episode_ids"])
    )
    if eligible_episodes & HELD_OUT_EPISODES:
        raise ValueError("held-out episode entered policy-stitch eligibility")

    source_gate = _read_json(transition_root / "emulator_gate_v1.json")
    source_thresholds = _read_json(transition_root / "expert_stitch_thresholds_v1.json")
    normalization = _read_json(transition_root / "retrieval_normalization_v1.json")
    support_radius = float(source_thresholds["one_step"]["retrieval_distance_upper"])
    if not bool(source_gate["one_step"]["passed"]):
        raise ValueError("inherited one-step transition Gate did not pass")
    train = _load_nodes(transition_root / "train_transition_bank_v1.npz")
    validation = _load_nodes(transition_root / "validation_transition_queries_v1.npz")
    max_steps = _maximum_cycle_transition_count(train)

    annotations = {
        (int(row["episode_id"]), int(row["cycle_id"])): row
        for row in _read_jsonl(m0 / "cycle_annotations.jsonl")
        if row["quality"]["status"] == "accepted"
        and row["split"] == "validation"
        and int(row["episode_id"]) in eligible_episodes
    }
    anchors = [
        row
        for row in _read_jsonl(m2 / "condition_counterfactual_anchors_v1.jsonl")
        if row["split"] == "validation"
        and row["changed_factors"] == ["next_sector"]
        and bool(row["supported"])
        and int(row["episode_id"]) in eligible_episodes
    ]
    if not anchors:
        raise ValueError("condition delta stitch selected no supported anchors")
    if len({(row["episode_id"], row["cycle_id"]) for row in anchors}) != len(anchors):
        raise ValueError("policy-stitch anchors must have unique cycle keys")
    if not {(int(row["episode_id"]), int(row["cycle_id"])) for row in anchors} <= set(
        annotations
    ):
        raise ValueError("supported policy-stitch anchor lacks annotation")

    initial_lookup = {
        (
            int(validation["episode_id"][index]),
            int(validation["cycle_id"][index]),
            int(validation["tick"][index]),
        ): index
        for index in range(validation["episode_id"].size)
    }
    bundles = {
        "B1.4": _validate_bundle(Path(b1_bundle_root).resolve(strict=True), "B1.4"),
        "B2.4": _validate_bundle(Path(b2_bundle_root).resolve(strict=True), "B2.4"),
    }
    retriever = CachedTransitionRetriever(
        train["retrieval"],
        train["episode_id"],
        device=device,
    )

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    try:
        with ExitStack() as stack:
            episodes = {
                episode_id: stack.enter_context(
                    h5py.File(m0 / f"episodes/episode_{episode_id}.hdf5", "r")
                )
                for episode_id in sorted(
                    set(map(int, train["episode_id"]))
                    | set(map(int, validation["episode_id"]))
                )
            }
            for baseline_id, bundle in bundles.items():
                policy = _load_policy(bundle, max_steps=max_steps, device=device)
                for anchor_index, anchor in enumerate(anchors):
                    key = (int(anchor["episode_id"]), int(anchor["cycle_id"]))
                    annotation = annotations[key]
                    start_tick = int(annotation["target_steps_20hz"][0])
                    initial_index = initial_lookup.get((key[0], key[1], start_tick))
                    if initial_index is None:
                        raise ValueError(f"missing validation start node: {key}")
                    for request_name in ("base", "target"):
                        condition = np.asarray(
                            anchor[f"{request_name}_condition"]["vector"],
                            dtype=np.float32,
                        )
                        arrays, summary = run_policy_delta_stitch_rollout(
                            policy=policy,
                            bank=train,
                            retriever=retriever,
                            episodes=episodes,
                            initial=validation,
                            initial_index=initial_index,
                            condition=condition,
                            support_radius=support_radius,
                            max_steps=max_steps,
                            action_normalization=normalization["action"],
                        )
                        relative = (
                            Path("traces")
                            / baseline_id.lower().replace(".", "_")
                            / (
                                f"anchor_{anchor_index}_episode_{key[0]}_"
                                f"cycle_{key[1]}_{request_name}.npz"
                            )
                        )
                        trace_path = temporary / relative
                        trace_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(trace_path, **arrays)
                        trace_identity = artifact_identity(trace_path)
                        identities.append(trace_identity)
                        result_rows.append(
                            {
                                "schema": (
                                    "simverify_condition_delta_stitch_result_v1"
                                ),
                                "anchor_index": anchor_index,
                                "episode_id": key[0],
                                "cycle_id": key[1],
                                "baseline_id": baseline_id,
                                "request_name": request_name,
                                "base_condition": anchor["base_condition"],
                                "target_condition": anchor["target_condition"],
                                "delivered_condition": anchor[
                                    f"{request_name}_condition"
                                ],
                                "trace_path": str(relative),
                                "trace_sha256": trace_identity["sha256"],
                                **summary,
                            }
                        )
                del policy
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        paired_rows = build_paired_condition_metrics(result_rows)
        source_rows = aggregate_condition_metrics_by_episode(paired_rows)
        gate = evaluate_condition_delta_stitch_gate(
            result_rows,
            source_rows,
        )
        decision = gate["decision"]
        identities.append(
            write_jsonl(
                temporary / "condition_delta_stitch_results.jsonl",
                result_rows,
            )
        )
        identities.append(
            write_jsonl(
                temporary / "condition_delta_stitch_paired_metrics.jsonl",
                paired_rows,
            )
        )
        identities.append(
            write_jsonl(
                temporary / "condition_delta_stitch_source_metrics.jsonl",
                source_rows,
            )
        )
        gate_identity = write_json(
            temporary / "condition_delta_stitch_gate_v1.json",
            gate,
        )
        identities.append(gate_identity)
        manifest_identity = write_json(
            temporary / "condition_delta_stitch_manifest.json",
            {
                "schema": "simverify_condition_delta_stitch_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "contract": {
                    "path": str(contract),
                    "sha256": sha256_file(contract),
                },
                "transition_package": _package_identity(
                    transition_root,
                    "transition_stitch_manifest.json",
                ),
                "delta_stitch_audit": _package_identity(
                    audit_root,
                    "delta_stitch_gate_audit_manifest.json",
                ),
                "next_condition_support": _package_identity(
                    support_root,
                    "next_condition_support_manifest.json",
                ),
                "bundles": {
                    baseline_id: bundle["identity"]
                    for baseline_id, bundle in bundles.items()
                },
                "m0_dataset_manifest_sha256": sha256_file(m0 / "dataset_manifest.json"),
                "m2_manifest_sha256": sha256_file(m2 / "m2_manifest.json"),
                "anchor_registry_sha256": sha256_file(
                    m2 / "condition_counterfactual_anchors_v1.jsonl"
                ),
                "eligible_validation_source_episode_ids": sorted(eligible_episodes),
                "anchor_count": len(anchors),
                "support_radius": support_radius,
                "maximum_steps": max_steps,
                "retrieval_input_fields": [
                    "observable_state",
                    "future_runtime_safe_policy_action",
                ],
                "retrieval_forbidden_fields": [
                    "condition",
                    "phase",
                    "progress",
                    "target_sector",
                    "successor_state",
                    "successor_identity",
                    "future_state",
                    "privilege",
                ],
                "decision": decision,
                "independent_validation": False,
                "validation_role": "development_reused_after_v3",
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
            "supported_path_effect_established": gate[
                "supported_path_effect_established"
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
                    "schema": "simverify_condition_delta_stitch_failure_v1",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "git": git,
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                },
            )
        raise


class CachedTransitionRetriever:
    """Exact GPU/CPU retriever with an immutable cached train bank."""

    def __init__(
        self,
        bank_vectors: np.ndarray,
        bank_episode_ids: np.ndarray,
        *,
        device: str,
    ) -> None:
        self.device = torch.device(device)
        self.bank = torch.from_numpy(np.asarray(bank_vectors, dtype=np.float32)).to(
            self.device
        )
        self.bank_norm = torch.sum(self.bank * self.bank, dim=1)
        self.episodes = torch.from_numpy(
            np.asarray(bank_episode_ids, dtype=np.int64)
        ).to(self.device)

    def nearest(
        self,
        query: np.ndarray,
        *,
        current_episode_id: int,
        seen_indices: set[int],
    ) -> tuple[float, int]:
        vector = torch.from_numpy(np.asarray(query, dtype=np.float32)).to(self.device)
        squared = (
            torch.sum(vector * vector) + self.bank_norm - 2.0 * (self.bank @ vector)
        )
        squared.clamp_(min=0.0)
        squared[self.episodes == int(current_episode_id)] = float("inf")
        if seen_indices:
            squared[
                torch.as_tensor(
                    sorted(seen_indices),
                    dtype=torch.int64,
                    device=self.device,
                )
            ] = float("inf")
        value, index = torch.min(squared, dim=0)
        return float(torch.sqrt(value).cpu()), int(index.cpu())


def run_policy_delta_stitch_rollout(
    *,
    policy: Any,
    bank: Mapping[str, np.ndarray],
    retriever: CachedTransitionRetriever,
    episodes: Mapping[int, h5py.File],
    initial: Mapping[str, np.ndarray],
    initial_index: int,
    condition: np.ndarray,
    support_radius: float,
    max_steps: int,
    action_normalization: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run one causal policy sequence over supported recorded successors."""

    if condition.shape != (6,):
        raise ValueError("policy-stitch condition must have shape (6,)")
    if hasattr(policy, "reset"):
        policy.reset()
    current_episode = int(initial["episode_id"][initial_index])
    current_tick = int(initial["tick"][initial_index])
    current_state = initial["state_standardized"][initial_index].copy()
    current_state_raw = initial["state"][initial_index].copy()
    seen: set[int] = set()
    donor_episodes: set[int] = set()
    accumulated = 0.0
    status = "max_train_duration_reached"
    raw_normalized = []
    raw_direct = []
    aggregated = []
    selected_indices = []
    retrieval_distances = []
    accumulated_rows = []
    observation_episode_ids = []
    observation_ticks = []
    route_indices = []
    route_pending = []
    for _step in range(max_steps):
        episode = episodes[current_episode]
        observation: dict[str, Any] = {
            "qpos": np.asarray(
                episode["observations/qpos"][current_tick],
                dtype=np.float32,
            ),
            "qvel": np.asarray(
                episode["observations/qvel"][current_tick],
                dtype=np.float32,
            ),
            "cycle_condition_v1": condition.copy(),
        }
        for camera in CAMERAS:
            observation[f"image_{camera}"] = _read_camera_image(
                episode,
                camera,
                current_tick,
            )
        action = np.asarray(policy.predict(observation), dtype=np.float32).reshape(4)
        chunk_normalized = np.asarray(
            policy.last_raw_action_chunk(),
            dtype=np.float32,
        )
        chunk_direct = np.asarray(
            policy.last_raw_action_chunk_direct(),
            dtype=np.float32,
        )
        action_standardized = apply_standardization(
            action.reshape(1, ACTION_DIM),
            action_normalization,
        )
        query = compose_retrieval_vectors(
            current_state.reshape(1, -1),
            action_standardized,
        )[0]
        distance, candidate = retriever.nearest(
            query,
            current_episode_id=current_episode,
            seen_indices=seen,
        )
        raw_normalized.append(chunk_normalized)
        raw_direct.append(chunk_direct)
        aggregated.append(action)
        observation_episode_ids.append(current_episode)
        observation_ticks.append(current_tick)
        route = getattr(policy, "condition_route_diagnostics", None)
        route_indices.append(-1 if route is None else int(route["route_index"]))
        route_pending.append(-1 if route is None else int(route["consecutive_pending"]))
        if not math.isfinite(distance) or distance > support_radius:
            status = "offline_support_exhausted"
            break
        if candidate in seen:
            raise AssertionError("policy stitch reused a transition")
        seen.add(candidate)
        selected_indices.append(candidate)
        retrieval_distances.append(distance)
        donor_episode = int(bank["episode_id"][candidate])
        donor_episodes.add(donor_episode)
        delta = float(bank["next_progress"][candidate] - bank["progress"][candidate])
        if not math.isfinite(delta) or delta <= 0.0:
            status = "invalid_local_progress_delta"
            break
        accumulated += delta
        accumulated_rows.append(accumulated)
        current_episode = donor_episode
        current_tick = int(bank["next_tick"][candidate])
        current_state = bank["next_state_standardized"][candidate].copy()
        current_state_raw = bank["next_state"][candidate].copy()
        if accumulated >= TARGET_PROGRESS - 1e-6:
            status = "completed_ready_to_ready_delta"
            break
    step_count = len(aggregated)
    if step_count == 0:
        raise AssertionError("policy stitch produced no action")
    arrays = {
        "raw_policy_chunk_normalized": np.stack(raw_normalized).astype(np.float32),
        "raw_policy_chunk_direct": np.stack(raw_direct).astype(np.float32),
        "temporal_aggregation_action": np.stack(aggregated).astype(np.float32),
        "future_runtime_safe_action": np.stack(aggregated).astype(np.float32),
        "selected_transition_index": np.asarray(selected_indices, dtype=np.int64),
        "retrieval_distance": np.asarray(retrieval_distances, dtype=np.float32),
        "accumulated_progress": np.asarray(accumulated_rows, dtype=np.float32),
        "observation_episode_id": np.asarray(observation_episode_ids, dtype=np.int64),
        "observation_tick": np.asarray(observation_ticks, dtype=np.int64),
        "condition": np.repeat(
            condition.reshape(1, 6),
            step_count,
            axis=0,
        ).astype(np.float32),
        "condition_route_index": np.asarray(route_indices, dtype=np.int8),
        "condition_route_pending_count": np.asarray(route_pending, dtype=np.int16),
        "endpoint_observable_state": current_state_raw.astype(np.float32),
    }
    return arrays, {
        "status": status,
        "completed": status == "completed_ready_to_ready_delta",
        "step_count": step_count,
        "selected_transition_count": len(selected_indices),
        "unique_selected_transition_count": len(seen),
        "unique_donor_episode_count": len(donor_episodes),
        "accumulated_progress": accumulated,
        "max_retrieval_distance": (
            max(retrieval_distances) if retrieval_distances else distance
        ),
        "endpoint_sector_similarity": {
            sector: float(current_state_raw[-3 + index])
            for index, sector in enumerate(SECTORS)
        },
        "selected_transition_index_sha256": hashlib.sha256(
            np.asarray(selected_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "condition_used_for_retrieval": False,
        "absolute_progress_used_for_rollout_state": False,
        "evidence_scope": EVIDENCE_SCOPE,
        "closed_loop_execution": False,
    }


def build_paired_condition_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Score base/target endpoint semantics for each baseline and anchor."""

    indexed = {
        (
            str(row["baseline_id"]),
            int(row["anchor_index"]),
            str(row["request_name"]),
        ): row
        for row in rows
    }
    result = []
    anchors = sorted({int(row["anchor_index"]) for row in rows})
    for baseline_id in ("B1.4", "B2.4"):
        for anchor_index in anchors:
            base = indexed[(baseline_id, anchor_index, "base")]
            target = indexed[(baseline_id, anchor_index, "target")]
            base_sector = str(base["base_condition"]["next_sector"])
            target_sector = str(base["target_condition"]["next_sector"])
            base_similarity = base["endpoint_sector_similarity"]
            target_similarity = target["endpoint_sector_similarity"]
            semantic_score = 0.5 * (
                float(base_similarity[base_sector])
                - float(base_similarity[target_sector])
                + float(target_similarity[target_sector])
                - float(target_similarity[base_sector])
            )
            result.append(
                {
                    "schema": "simverify_condition_delta_stitch_pair_metric_v1",
                    "anchor_index": anchor_index,
                    "episode_id": int(base["episode_id"]),
                    "cycle_id": int(base["cycle_id"]),
                    "baseline_id": baseline_id,
                    "base_next_sector": base_sector,
                    "target_next_sector": target_sector,
                    "both_completed": bool(base["completed"] and target["completed"]),
                    "endpoint_semantic_score": semantic_score,
                    "path_diverged": _path_diverged(base, target),
                    "closed_loop_execution": False,
                }
            )
    return result


def aggregate_condition_metrics_by_episode(
    paired_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        grouped[(int(row["episode_id"]), str(row["baseline_id"]))].append(row)
    result = []
    episode_ids = sorted({key[0] for key in grouped})
    for episode_id in episode_ids:
        b1 = grouped[(episode_id, "B1.4")]
        b2 = grouped[(episode_id, "B2.4")]
        if {int(row["anchor_index"]) for row in b1} != {
            int(row["anchor_index"]) for row in b2
        }:
            raise ValueError("B1.4 and B2.4 anchor support differs")
        b1_mean = float(np.mean([row["endpoint_semantic_score"] for row in b1]))
        b2_mean = float(np.mean([row["endpoint_semantic_score"] for row in b2]))
        result.append(
            {
                "schema": "simverify_condition_delta_stitch_source_metric_v1",
                "episode_id": episode_id,
                "anchor_count": len(b1),
                "b1_4_completion_rate": float(
                    np.mean([row["both_completed"] for row in b1])
                ),
                "b2_4_completion_rate": float(
                    np.mean([row["both_completed"] for row in b2])
                ),
                "b1_4_path_divergence_rate": float(
                    np.mean([row["path_diverged"] for row in b1])
                ),
                "b1_4_endpoint_semantic_score_mean": b1_mean,
                "b2_4_endpoint_semantic_score_mean": b2_mean,
                "b1_4_minus_b2_4_semantic_score": b1_mean - b2_mean,
            }
        )
    return result


def evaluate_condition_delta_stitch_gate(
    result_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    criteria = {
        "all_rollouts_complete_inside_support": {
            "observed_rate": float(
                np.mean([bool(row["completed"]) for row in result_rows])
            ),
            "required": 1.0,
            "passed": all(bool(row["completed"]) for row in result_rows),
        },
        "b1_4_path_diverges": {
            "observed_min_source_episode_rate": min(
                float(row["b1_4_path_divergence_rate"]) for row in source_rows
            ),
            "minimum_required": 0.0,
            "comparison": "strictly_greater",
            "passed": all(
                float(row["b1_4_path_divergence_rate"]) > 0.0 for row in source_rows
            ),
        },
        "b1_4_endpoint_semantics_positive": {
            "observed_min_source_episode_mean": min(
                float(row["b1_4_endpoint_semantic_score_mean"]) for row in source_rows
            ),
            "minimum_required": 0.0,
            "comparison": "strictly_greater",
            "passed": all(
                float(row["b1_4_endpoint_semantic_score_mean"]) > 0.0
                for row in source_rows
            ),
        },
        "b1_4_exceeds_b2_4": {
            "observed_min_source_episode_delta": min(
                float(row["b1_4_minus_b2_4_semantic_score"]) for row in source_rows
            ),
            "minimum_required": 0.0,
            "comparison": "strictly_greater",
            "passed": all(
                float(row["b1_4_minus_b2_4_semantic_score"]) > 0.0
                for row in source_rows
            ),
        },
        "transition_identity_not_reused": {
            "observed_violations": sum(
                int(row["selected_transition_count"])
                != int(row["unique_selected_transition_count"])
                for row in result_rows
            ),
            "maximum_allowed": 0,
            "passed": all(
                int(row["selected_transition_count"])
                == int(row["unique_selected_transition_count"])
                for row in result_rows
            ),
        },
    }
    passed = all(bool(row["passed"]) for row in criteria.values())
    decision = (
        "next_condition_supported_path_effect_established_development"
        if passed
        else "next_condition_supported_path_effect_not_established"
    )
    return {
        "schema": "simverify_condition_delta_stitch_gate_v1",
        "decision": decision,
        "supported_path_effect_established": passed,
        "criteria": criteria,
        "source_episode_metrics": list(source_rows),
        "independent_validation": False,
        "validation_role": "development_reused_after_v3",
        "evidence_scope": EVIDENCE_SCOPE,
        "held_out_test_read": False,
        "closed_loop_execution": False,
    }


def _path_diverged(
    base: Mapping[str, Any],
    target: Mapping[str, Any],
) -> bool:
    return bool(
        base["selected_transition_index_sha256"]
        != target["selected_transition_index_sha256"]
    )


def _load_policy(bundle: Mapping[str, Any], *, max_steps: int, device: str) -> Any:
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    return load_policy_for_episode(
        bundle_dir=bundle["root"],
        ckpt_path=bundle["checkpoint"],
        resolved_config_path=None,
        stats_path=None,
        max_episode_len=max_steps + 1,
        temporal_agg=True,
        device=device,
        inference_precision="fp32",
    )


def _validate_bundle(root: Path, baseline_id: str) -> dict[str, Any]:
    metadata = _read_json(root / "run_metadata.json")
    if metadata.get("status") != "completed":
        raise ValueError(f"{baseline_id} bundle is not completed")
    if metadata["experiment_contract"]["baseline_id"] != baseline_id:
        raise ValueError(f"expected {baseline_id} bundle")
    if (
        metadata["experiment_contract"]["condition_input"]
        != "cycle_condition_v1_next_sector_only"
    ):
        raise ValueError(f"{baseline_id} condition input changed")
    if metadata["checkpoint_semantics"]["real_control_allowed"] is not False:
        raise ValueError(f"{baseline_id} lacks real-control prohibition")
    checkpoint = root / "policy_best.ckpt"
    checkpoint_contract = _validate_condition_checkpoint(
        checkpoint,
        expected_baseline=baseline_id,
    )
    return {
        "root": root,
        "checkpoint": checkpoint,
        "identity": {
            "path": str(root),
            "baseline_id": baseline_id,
            "run_metadata_sha256": sha256_file(root / "run_metadata.json"),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_contract": checkpoint_contract,
        },
    }


def _package_identity(root: Path, manifest_name: str) -> dict[str, Any]:
    return {
        "path": str(root),
        "manifest_sha256": sha256_file(root / manifest_name),
        "checksums_sha256": sha256_file(root / "checksums.sha256"),
    }


def _maximum_cycle_transition_count(nodes: Mapping[str, np.ndarray]) -> int:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for episode_id, cycle_id in zip(
        nodes["episode_id"], nodes["cycle_id"], strict=True
    ):
        counts[(int(episode_id), int(cycle_id))] += 1
    return max(counts.values())


def _load_nodes(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as package:
        return {key: package[key] for key in package.files}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
