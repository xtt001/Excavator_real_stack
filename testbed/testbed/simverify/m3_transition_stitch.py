"""Observable-only non-parametric transition stitching for SimVerify G4-v2."""

from __future__ import annotations

import json
import math
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
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.features import FrozenResNet18FeatureExtractor

EVENT_NAMES = (
    "ready",
    "dig_entry_proxy",
    "carry_transition_proxy",
    "dump_start_proxy",
    "dump_end_proxy",
)
SECTORS = ("left", "center", "right")
POLICY_EYE_PAIR = ("video4", "video5")
POLICY_STICK_PAIR = ("video6", "video7")
HELD_OUT_EPISODES = {1, 13, 25, 33}
STATE_DIM = 4 + 4 + 5 + 5 + 3
ACTION_DIM = 4


def build_transition_stitch_calibration(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    m0_root: str | Path,
    m2_root: str | Path,
    contract_path: str | Path,
    feature_chunk_rows: int = 256,
    knn_query_batch_size: int = 256,
    device: str = "cuda",
) -> dict[str, Any]:
    """Build the train-only bank and expert self-replay prerequisite."""

    if feature_chunk_rows <= 0 or knn_query_batch_size <= 0:
        raise ValueError("feature and KNN batch sizes must be positive")
    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
        or not git.get("git_available")
    ):
        raise ValueError("transition stitching requires a clean SimVerify worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable transition package exists: {destination}")
    m0 = Path(m0_root).resolve(strict=True)
    m2 = Path(m2_root).resolve(strict=True)
    contract = Path(contract_path).resolve(strict=True)
    split_manifest = _read_json(m0 / "split_groups.json")
    splits = {
        name: list(map(int, split_manifest["splits"][name]))
        for name in ("train", "validation", "held_out_test")
    }
    if set(splits["held_out_test"]) != HELD_OUT_EPISODES:
        raise ValueError("held-out episode lock differs from frozen split")
    if (set(splits["train"]) | set(splits["validation"])) & HELD_OUT_EPISODES:
        raise ValueError("held-out episode entered train or validation")

    annotations = [
        row
        for row in _read_jsonl(m0 / "cycle_annotations.jsonl")
        if row["quality"]["status"] == "accepted"
        and row["split"] in {"train", "validation"}
    ]
    if {int(row["episode_id"]) for row in annotations} & HELD_OUT_EPISODES:
        raise ValueError("held-out annotation entered transition calibration")
    by_split = {
        split: [
            row
            for row in annotations
            if row["split"] == split and int(row["episode_id"]) in set(splits[split])
        ]
        for split in ("train", "validation")
    }
    if not by_split["train"] or not by_split["validation"]:
        raise ValueError("transition calibration requires train and validation cycles")

    feature_manifest = _read_json(m0 / "annotation_feature_input_manifest_v1.json")
    checkpoint_contract = feature_manifest["feature_extractor"]["checkpoint"]
    checkpoint = Path(checkpoint_contract["path"]).resolve(strict=True)
    extractor = FrozenResNet18FeatureExtractor(
        checkpoint,
        expected_checkpoint_sha256=checkpoint_contract["sha256"],
        device=device,
        batch_size=64,
    )
    prototypes = _load_prototypes(m0 / "annotation_feature_prototypes_v2.npz")

    node_sets: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "validation"):
        print(
            f"transition-stitch: extracting {split} observable features "
            f"for {len(by_split[split])} cycles",
            flush=True,
        )
        node_sets[split] = _build_nodes(
            annotations=by_split[split],
            m0_root=m0,
            extractor=extractor,
            prototypes=prototypes,
            chunk_rows=feature_chunk_rows,
        )
    train = node_sets["train"]
    validation = node_sets["validation"]
    normalization = fit_retrieval_normalization(
        train["state"],
        train["action"],
    )
    for nodes in node_sets.values():
        nodes["state_standardized"] = apply_standardization(
            nodes["state"],
            normalization["state"],
        )
        nodes["next_state_standardized"] = apply_standardization(
            nodes["next_state"],
            normalization["state"],
        )
        nodes["action_standardized"] = apply_standardization(
            nodes["action"],
            normalization["action"],
        )
        nodes["retrieval"] = compose_retrieval_vectors(
            nodes["state_standardized"],
            nodes["action_standardized"],
        )

    print(
        "transition-stitch: exact train leave-one-source-episode retrieval",
        flush=True,
    )
    train_neighbor_distance, train_neighbor_index = exact_nearest_indices(
        train["retrieval"],
        train["retrieval"],
        bank_episode_ids=train["episode_id"],
        query_episode_ids=train["episode_id"],
        exclude_same_episode=True,
        device=device,
        batch_size=knn_query_batch_size,
    )
    print(
        "transition-stitch: exact validation-to-train retrieval",
        flush=True,
    )
    validation_neighbor_distance, validation_neighbor_index = exact_nearest_indices(
        train["retrieval"],
        validation["retrieval"],
        bank_episode_ids=train["episode_id"],
        query_episode_ids=validation["episode_id"],
        exclude_same_episode=True,
        device=device,
        batch_size=knn_query_batch_size,
    )

    train_one_step = one_step_metrics(
        query=train,
        bank=train,
        neighbor_distance=train_neighbor_distance,
        neighbor_index=train_neighbor_index,
        split="train_leave_one_source_episode_out",
    )
    validation_one_step = one_step_metrics(
        query=validation,
        bank=train,
        neighbor_distance=validation_neighbor_distance,
        neighbor_index=validation_neighbor_index,
        split="validation",
    )
    train_one_step_episodes = aggregate_one_step_by_episode(train_one_step)
    validation_one_step_episodes = aggregate_one_step_by_episode(validation_one_step)
    one_step_thresholds = derive_one_step_thresholds(train_one_step_episodes)
    one_step_gate = evaluate_one_step_gate(
        validation_one_step_episodes,
        one_step_thresholds,
    )

    duration_max = max(
        int(row["target_steps_20hz"][1]) - int(row["target_steps_20hz"][0]) + 1
        for row in by_split["train"]
    )
    print(
        "transition-stitch: cumulative expert train leave-one-episode rollout",
        flush=True,
    )
    train_rollouts = run_expert_stitch_rollouts(
        bank=train,
        initial=train,
        initial_indices=_cycle_start_indices(train),
        support_radius=one_step_thresholds["retrieval_distance_upper"],
        max_steps=duration_max,
        device=device,
        batch_size=knn_query_batch_size,
    )
    print(
        "transition-stitch: cumulative expert validation rollout",
        flush=True,
    )
    validation_rollouts = run_expert_stitch_rollouts(
        bank=train,
        initial=validation,
        initial_indices=_cycle_start_indices(validation),
        support_radius=one_step_thresholds["retrieval_distance_upper"],
        max_steps=duration_max,
        device=device,
        batch_size=knn_query_batch_size,
    )
    train_rollout_episodes = aggregate_rollouts_by_episode(train_rollouts)
    validation_rollout_episodes = aggregate_rollouts_by_episode(validation_rollouts)
    cumulative_thresholds = derive_cumulative_thresholds(train_rollout_episodes)
    cumulative_gate = evaluate_cumulative_gate(
        train_rollout_episodes,
        validation_rollout_episodes,
        cumulative_thresholds,
    )
    emulator_passed = bool(one_step_gate["passed"] and cumulative_gate["passed"])
    gate = {
        "schema": "simverify_transition_stitch_emulator_gate_v1",
        "decision": (
            "pass_expert_transition_stitch_prerequisite"
            if emulator_passed
            else "offline_emulator_invalid"
        ),
        "authorizes_condition_rollout": emulator_passed,
        "one_step": one_step_gate,
        "cumulative": cumulative_gate,
        "retrieval_features_include_condition": False,
        "retrieval_features_include_phase_or_progress": False,
        "retrieval_features_include_successor": False,
        "evidence_scope": "recorded-observation/offline empirical rollout",
        "closed_loop_execution": False,
        "held_out_test_read": False,
    }

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    try:
        identities.append(
            _write_npz(
                temporary / "train_transition_bank_v1.npz",
                _serializable_node_arrays(train),
            )
        )
        identities.append(
            _write_npz(
                temporary / "validation_transition_queries_v1.npz",
                _serializable_node_arrays(validation),
            )
        )
        identities.append(
            write_json(
                temporary / "retrieval_normalization_v1.json",
                {
                    "schema": "simverify_transition_retrieval_normalization_v1",
                    "state_feature_names": _state_feature_names(),
                    "action_feature_names": [
                        "swing",
                        "boom",
                        "stick",
                        "bucket",
                    ],
                    "state": normalization["state"],
                    "action": normalization["action"],
                    "group_distance": (
                        "sqrt(mean(state_z_delta_squared)+mean(action_z_delta_squared))"
                    ),
                    "fit_split": "train_only",
                },
            )
        )
        identities.append(
            write_jsonl(
                temporary / "train_one_step_loo_metrics.jsonl",
                train_one_step,
            )
        )
        identities.append(
            write_jsonl(
                temporary / "validation_one_step_metrics.jsonl",
                validation_one_step,
            )
        )
        identities.append(
            write_jsonl(
                temporary / "train_expert_stitch_rollouts.jsonl",
                train_rollouts,
            )
        )
        identities.append(
            write_jsonl(
                temporary / "validation_expert_stitch_rollouts.jsonl",
                validation_rollouts,
            )
        )
        identities.append(
            write_json(
                temporary / "expert_stitch_thresholds_v1.json",
                {
                    "schema": "simverify_expert_stitch_thresholds_v1",
                    "one_step": one_step_thresholds,
                    "cumulative": cumulative_thresholds,
                    "source": "train_source_episode_leave_one_out",
                    "held_out_test_read": False,
                },
            )
        )
        identities.append(write_json(temporary / "emulator_gate_v1.json", gate))
        manifest_identity = write_json(
            temporary / "transition_stitch_manifest.json",
            {
                "schema": "simverify_transition_stitch_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "contract": {
                    "path": str(contract),
                    "sha256": sha256_file(contract),
                },
                "m0_dataset_manifest_sha256": sha256_file(m0 / "dataset_manifest.json"),
                "m2_manifest_sha256": sha256_file(m2 / "m2_manifest.json"),
                "split_manifest_sha256": sha256_file(m0 / "split_groups.json"),
                "visual_prototypes_sha256": sha256_file(
                    m0 / "annotation_feature_prototypes_v2.npz"
                ),
                "visual_extractor": extractor.provenance,
                "train_episode_ids": sorted(
                    set(map(int, train["episode_id"].tolist()))
                ),
                "validation_episode_ids": sorted(
                    set(map(int, validation["episode_id"].tolist()))
                ),
                "held_out_episode_ids": "locked_unread",
                "train_transition_count": int(train["episode_id"].size),
                "validation_transition_count": int(validation["episode_id"].size),
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "retrieval_input_fields": [
                    "qpos",
                    "qvel",
                    "eye_event_prototype_similarity",
                    "stick_event_prototype_similarity",
                    "eye_sector_prototype_similarity",
                    "executed_action",
                ],
                "retrieval_forbidden_fields": [
                    "condition",
                    "target_sector",
                    "phase",
                    "progress",
                    "successor_state",
                    "privilege",
                ],
                "expert_self_replay_decision": gate["decision"],
                "authorizes_condition_rollout": emulator_passed,
                "evidence_scope": ("recorded-observation/offline empirical rollout"),
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
            "authorizes_condition_rollout": emulator_passed,
            "train_transition_count": int(train["episode_id"].size),
            "validation_transition_count": int(validation["episode_id"].size),
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
                    "schema": "simverify_transition_stitch_failure_v1",
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "git": git,
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                },
            )
        raise


def fit_retrieval_normalization(
    state: np.ndarray,
    action: np.ndarray,
) -> dict[str, Any]:
    return {
        "state": _fit_robust_scale(state),
        "action": _fit_robust_scale(action),
    }


def apply_standardization(
    values: np.ndarray,
    contract: Mapping[str, Any],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    center = np.asarray(contract["center"], dtype=np.float32)
    scale = np.asarray(contract["scale"], dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != center.size:
        raise ValueError("standardization feature dimension mismatch")
    result = (array - center) / scale
    if not np.isfinite(result).all():
        raise ValueError("standardized values contain NaN or infinity")
    return result.astype(np.float32, copy=False)


def compose_retrieval_vectors(
    state_standardized: np.ndarray,
    action_standardized: np.ndarray,
) -> np.ndarray:
    state = np.asarray(state_standardized, dtype=np.float32)
    action = np.asarray(action_standardized, dtype=np.float32)
    if (
        state.ndim != 2
        or action.ndim != 2
        or state.shape[0] != action.shape[0]
        or state.shape[1] != STATE_DIM
        or action.shape[1] != ACTION_DIM
    ):
        raise ValueError("retrieval state/action shapes are invalid")
    return np.concatenate(
        (
            state / math.sqrt(float(STATE_DIM)),
            action / math.sqrt(float(ACTION_DIM)),
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def exact_nearest_indices(
    bank_vectors: np.ndarray,
    query_vectors: np.ndarray,
    *,
    bank_episode_ids: np.ndarray,
    query_episode_ids: np.ndarray,
    exclude_same_episode: bool,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact Euclidean nearest neighbors with optional episode masking."""

    bank = np.asarray(bank_vectors, dtype=np.float32)
    query = np.asarray(query_vectors, dtype=np.float32)
    bank_episodes = np.asarray(bank_episode_ids, dtype=np.int64)
    query_episodes = np.asarray(query_episode_ids, dtype=np.int64)
    if (
        bank.ndim != 2
        or query.ndim != 2
        or bank.shape[1] != query.shape[1]
        or bank.shape[0] != bank_episodes.size
        or query.shape[0] != query_episodes.size
    ):
        raise ValueError("nearest-neighbor array shapes are invalid")
    resolved = torch.device(device)
    bank_tensor = torch.from_numpy(bank).to(resolved)
    bank_norm = torch.sum(bank_tensor * bank_tensor, dim=1)
    bank_episode_tensor = torch.from_numpy(bank_episodes).to(resolved)
    distances: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for begin in range(0, query.shape[0], batch_size):
        end = min(query.shape[0], begin + batch_size)
        query_tensor = torch.from_numpy(query[begin:end]).to(resolved)
        squared = (
            torch.sum(query_tensor * query_tensor, dim=1, keepdim=True)
            + bank_norm.reshape(1, -1)
            - 2.0 * query_tensor @ bank_tensor.T
        )
        squared.clamp_(min=0.0)
        if exclude_same_episode:
            query_episode_tensor = torch.from_numpy(query_episodes[begin:end]).to(
                resolved
            )
            squared.masked_fill_(
                query_episode_tensor.reshape(-1, 1)
                == bank_episode_tensor.reshape(1, -1),
                float("inf"),
            )
        values, selected = torch.min(squared, dim=1)
        if not torch.isfinite(values).all():
            raise ValueError("no finite cross-episode transition candidate")
        distances.append(
            torch.sqrt(values).to(device="cpu", dtype=torch.float32).numpy()
        )
        indices.append(selected.to(device="cpu", dtype=torch.int64).numpy())
    return (
        np.concatenate(distances).astype(np.float32, copy=False),
        np.concatenate(indices).astype(np.int64, copy=False),
    )


def one_step_metrics(
    *,
    query: Mapping[str, np.ndarray],
    bank: Mapping[str, np.ndarray],
    neighbor_distance: np.ndarray,
    neighbor_index: np.ndarray,
    split: str,
) -> list[dict[str, Any]]:
    selected = np.asarray(neighbor_index, dtype=np.int64)
    successor_delta = (
        query["next_state_standardized"] - bank["next_state_standardized"][selected]
    )
    successor_error = np.sqrt(np.mean(successor_delta * successor_delta, axis=1))
    query_progress_delta = query["next_progress"] - query["progress"]
    bank_progress_delta = bank["next_progress"][selected] - bank["progress"][selected]
    progress_error = np.abs(query_progress_delta - bank_progress_delta)
    query_phase_delta = query["next_phase"] - query["phase"]
    bank_phase_delta = bank["next_phase"][selected] - bank["phase"][selected]
    result = []
    for index in range(selected.size):
        neighbor = int(selected[index])
        result.append(
            {
                "schema": "simverify_transition_one_step_metric_v1",
                "split": split,
                "episode_id": int(query["episode_id"][index]),
                "cycle_id": int(query["cycle_id"][index]),
                "tick": int(query["tick"][index]),
                "neighbor_episode_id": int(bank["episode_id"][neighbor]),
                "neighbor_cycle_id": int(bank["cycle_id"][neighbor]),
                "neighbor_tick": int(bank["tick"][neighbor]),
                "retrieval_distance": float(neighbor_distance[index]),
                "successor_state_rms_error": float(successor_error[index]),
                "progress_delta_abs_error": float(progress_error[index]),
                "phase_delta_agreement": bool(
                    query_phase_delta[index] == bank_phase_delta[index]
                ),
                "condition_used_for_retrieval": False,
                "phase_or_progress_used_for_retrieval": False,
                "successor_used_for_retrieval": False,
            }
        )
    return result


def aggregate_one_step_by_episode(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_id"])].append(row)
    result = []
    for episode_id, episode_rows in sorted(grouped.items()):
        result.append(
            {
                "episode_id": episode_id,
                "transition_count": len(episode_rows),
                "retrieval_distance_q97_5": _q(
                    [row["retrieval_distance"] for row in episode_rows],
                    0.975,
                ),
                "successor_state_rms_error_q97_5": _q(
                    [row["successor_state_rms_error"] for row in episode_rows],
                    0.975,
                ),
                "progress_delta_abs_error_q97_5": _q(
                    [row["progress_delta_abs_error"] for row in episode_rows],
                    0.975,
                ),
                "phase_delta_agreement_rate": float(
                    np.mean([row["phase_delta_agreement"] for row in episode_rows])
                ),
            }
        )
    return result


def derive_one_step_thresholds(
    train_episode_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "simverify_transition_one_step_thresholds_v1",
        "aggregation": "per_transition_then_source_episode",
        "retrieval_distance_upper": _q(
            [row["retrieval_distance_q97_5"] for row in train_episode_rows],
            0.975,
        ),
        "successor_state_rms_error_upper": _q(
            [row["successor_state_rms_error_q97_5"] for row in train_episode_rows],
            0.975,
        ),
        "progress_delta_abs_error_upper": _q(
            [row["progress_delta_abs_error_q97_5"] for row in train_episode_rows],
            0.975,
        ),
        "phase_delta_agreement_lower": _q(
            [row["phase_delta_agreement_rate"] for row in train_episode_rows],
            0.025,
        ),
        "source_episode_count": len(train_episode_rows),
        "source": "train_leave_one_source_episode_out",
    }


def evaluate_one_step_gate(
    validation_episode_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = {
        "retrieval_support": {
            "observed_max_episode_q97_5": max(
                row["retrieval_distance_q97_5"] for row in validation_episode_rows
            ),
            "maximum_allowed": thresholds["retrieval_distance_upper"],
        },
        "successor_state_error": {
            "observed_max_episode_q97_5": max(
                row["successor_state_rms_error_q97_5"]
                for row in validation_episode_rows
            ),
            "maximum_allowed": thresholds["successor_state_rms_error_upper"],
        },
        "progress_delta_error": {
            "observed_max_episode_q97_5": max(
                row["progress_delta_abs_error_q97_5"] for row in validation_episode_rows
            ),
            "maximum_allowed": thresholds["progress_delta_abs_error_upper"],
        },
        "phase_delta_agreement": {
            "observed_min_episode_rate": min(
                row["phase_delta_agreement_rate"] for row in validation_episode_rows
            ),
            "minimum_allowed": thresholds["phase_delta_agreement_lower"],
        },
    }
    for name, criterion in criteria.items():
        if name == "phase_delta_agreement":
            criterion["passed"] = bool(
                criterion["observed_min_episode_rate"] >= criterion["minimum_allowed"]
            )
        else:
            criterion["passed"] = bool(
                criterion["observed_max_episode_q97_5"] <= criterion["maximum_allowed"]
            )
    return {
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria.values()),
        "validation_source_episode_count": len(validation_episode_rows),
    }


def run_expert_stitch_rollouts(
    *,
    bank: Mapping[str, np.ndarray],
    initial: Mapping[str, np.ndarray],
    initial_indices: Sequence[int],
    support_radius: float,
    max_steps: int,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Run batched expert-action empirical rollouts without same-episode continuation."""

    starts = np.asarray(initial_indices, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("expert stitch rollout requires initial states")
    key_to_bank_index = {
        (int(episode), int(tick)): index
        for index, (episode, tick) in enumerate(
            zip(bank["episode_id"], bank["tick"], strict=True)
        )
    }
    active: list[dict[str, Any]] = []
    for start_index in starts.tolist():
        active.append(
            {
                "initial_index": start_index,
                "initial_episode_id": int(initial["episode_id"][start_index]),
                "initial_cycle_id": int(initial["cycle_id"][start_index]),
                "current_source": "initial",
                "current_index": start_index,
                "current_episode_id": int(initial["episode_id"][start_index]),
                "state": initial["state_standardized"][start_index].copy(),
                "action": initial["action_standardized"][start_index].copy(),
                "progress": float(initial["progress"][start_index]),
                "phase": int(initial["phase"][start_index]),
                "phases_seen": {int(initial["phase"][start_index])},
                "last_phase": int(initial["phase"][start_index]),
                "event_order_valid": True,
                "backward_progress_count": 0,
                "steps": 0,
                "max_distance": 0.0,
                "seen_nodes": set(),
                "status": "active",
            }
        )
    finished: list[dict[str, Any]] = []
    for _step in range(max_steps):
        running = [row for row in active if row["status"] == "active"]
        if not running:
            break
        states = np.stack([row["state"] for row in running])
        actions = np.stack([row["action"] for row in running])
        query_vectors = compose_retrieval_vectors(states, actions)
        query_episodes = np.asarray(
            [row["current_episode_id"] for row in running],
            dtype=np.int64,
        )
        distances, candidates = exact_nearest_indices(
            bank["retrieval"],
            query_vectors,
            bank_episode_ids=bank["episode_id"],
            query_episode_ids=query_episodes,
            exclude_same_episode=True,
            device=device,
            batch_size=batch_size,
        )
        for row, distance, candidate in zip(
            running,
            distances.tolist(),
            candidates.tolist(),
            strict=True,
        ):
            row["steps"] += 1
            row["max_distance"] = max(row["max_distance"], float(distance))
            if float(distance) > float(support_radius):
                row["status"] = "offline_support_exhausted"
                continue
            next_episode = int(bank["episode_id"][candidate])
            next_tick = int(bank["next_tick"][candidate])
            next_progress = float(bank["next_progress"][candidate])
            next_phase = int(bank["next_phase"][candidate])
            if next_progress + 1e-9 < row["progress"]:
                row["backward_progress_count"] += 1
            if next_phase < row["last_phase"] or next_phase > row["last_phase"] + 1:
                row["event_order_valid"] = False
            row["phases_seen"].add(next_phase)
            row["last_phase"] = next_phase
            row["progress"] = next_progress
            node_key = (next_episode, next_tick)
            if node_key in row["seen_nodes"]:
                row["status"] = "offline_deadlock"
                continue
            row["seen_nodes"].add(node_key)
            if next_progress >= 5.0 - 1e-6:
                row["status"] = "completed_ready_to_ready"
                continue
            next_index = key_to_bank_index.get(node_key)
            if next_index is None:
                row["status"] = "offline_support_exhausted"
                continue
            row["current_source"] = "bank"
            row["current_index"] = next_index
            row["current_episode_id"] = next_episode
            row["state"] = bank["state_standardized"][next_index].copy()
            row["action"] = bank["action_standardized"][next_index].copy()
    for row in active:
        if row["status"] == "active":
            row["status"] = "max_train_duration_reached"
        finished.append(
            {
                "schema": "simverify_expert_transition_stitch_rollout_v1",
                "episode_id": row["initial_episode_id"],
                "cycle_id": row["initial_cycle_id"],
                "status": row["status"],
                "steps": row["steps"],
                "final_progress": row["progress"],
                "phase_coverage": len(row["phases_seen"]) / 5.0,
                "event_order_valid": row["event_order_valid"],
                "backward_progress_count": row["backward_progress_count"],
                "max_retrieval_distance": row["max_distance"],
                "completed": row["status"] == "completed_ready_to_ready",
                "closed_loop_execution": False,
            }
        )
    return finished


def aggregate_rollouts_by_episode(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_id"])].append(row)
    result = []
    for episode_id, episode_rows in sorted(grouped.items()):
        result.append(
            {
                "episode_id": episode_id,
                "rollout_count": len(episode_rows),
                "completion_rate": float(
                    np.mean([row["completed"] for row in episode_rows])
                ),
                "event_order_valid_rate": float(
                    np.mean([row["event_order_valid"] for row in episode_rows])
                ),
                "phase_coverage_mean": float(
                    np.mean([row["phase_coverage"] for row in episode_rows])
                ),
                "support_exhaustion_rate": float(
                    np.mean(
                        [
                            row["status"] == "offline_support_exhausted"
                            for row in episode_rows
                        ]
                    )
                ),
                "deadlock_rate": float(
                    np.mean(
                        [row["status"] == "offline_deadlock" for row in episode_rows]
                    )
                ),
                "backward_progress_mean": float(
                    np.mean([row["backward_progress_count"] for row in episode_rows])
                ),
            }
        )
    return result


def derive_cumulative_thresholds(
    train_episode_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "simverify_transition_cumulative_thresholds_v1",
        "completion_rate_lower": _q(
            [row["completion_rate"] for row in train_episode_rows],
            0.025,
        ),
        "event_order_valid_rate_lower": _q(
            [row["event_order_valid_rate"] for row in train_episode_rows],
            0.025,
        ),
        "phase_coverage_mean_lower": _q(
            [row["phase_coverage_mean"] for row in train_episode_rows],
            0.025,
        ),
        "support_exhaustion_rate_upper": _q(
            [row["support_exhaustion_rate"] for row in train_episode_rows],
            0.975,
        ),
        "deadlock_rate_upper": _q(
            [row["deadlock_rate"] for row in train_episode_rows],
            0.975,
        ),
        "backward_progress_mean_upper": _q(
            [row["backward_progress_mean"] for row in train_episode_rows],
            0.975,
        ),
        "source_episode_count": len(train_episode_rows),
        "source": "train_leave_one_source_episode_out",
    }


def evaluate_cumulative_gate(
    train_episode_rows: Sequence[Mapping[str, Any]],
    validation_episode_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = {
        "train_identifies_nonzero_completion": {
            "observed": thresholds["completion_rate_lower"],
            "minimum_allowed": 0.0,
            "comparison": "strictly_greater",
            "passed": thresholds["completion_rate_lower"] > 0.0,
        },
        "validation_completion": {
            "observed_min_episode_rate": min(
                row["completion_rate"] for row in validation_episode_rows
            ),
            "minimum_allowed": thresholds["completion_rate_lower"],
        },
        "validation_event_order": {
            "observed_min_episode_rate": min(
                row["event_order_valid_rate"] for row in validation_episode_rows
            ),
            "minimum_allowed": thresholds["event_order_valid_rate_lower"],
        },
        "validation_phase_coverage": {
            "observed_min_episode_mean": min(
                row["phase_coverage_mean"] for row in validation_episode_rows
            ),
            "minimum_allowed": thresholds["phase_coverage_mean_lower"],
        },
        "validation_support_exhaustion": {
            "observed_max_episode_rate": max(
                row["support_exhaustion_rate"] for row in validation_episode_rows
            ),
            "maximum_allowed": thresholds["support_exhaustion_rate_upper"],
        },
        "validation_deadlock": {
            "observed_max_episode_rate": max(
                row["deadlock_rate"] for row in validation_episode_rows
            ),
            "maximum_allowed": thresholds["deadlock_rate_upper"],
        },
        "validation_backward_progress": {
            "observed_max_episode_mean": max(
                row["backward_progress_mean"] for row in validation_episode_rows
            ),
            "maximum_allowed": thresholds["backward_progress_mean_upper"],
        },
    }
    for name, criterion in criteria.items():
        if "passed" in criterion:
            continue
        if "minimum_allowed" in criterion:
            observed_key = next(key for key in criterion if key.startswith("observed_"))
            criterion["passed"] = bool(
                criterion[observed_key] >= criterion["minimum_allowed"]
            )
        else:
            observed_key = next(key for key in criterion if key.startswith("observed_"))
            criterion["passed"] = bool(
                criterion[observed_key] <= criterion["maximum_allowed"]
            )
    return {
        "criteria": criteria,
        "train_source_episode_count": len(train_episode_rows),
        "validation_source_episode_count": len(validation_episode_rows),
        "passed": all(item["passed"] for item in criteria.values()),
    }


def _build_nodes(
    *,
    annotations: Sequence[Mapping[str, Any]],
    m0_root: Path,
    extractor: FrozenResNet18FeatureExtractor,
    prototypes: Mapping[str, np.ndarray],
    chunk_rows: int,
) -> dict[str, np.ndarray]:
    by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        by_episode[int(annotation["episode_id"])].append(annotation)
    rows: dict[str, list[Any]] = defaultdict(list)
    for episode_id, episode_annotations in sorted(by_episode.items()):
        episode_path = m0_root / f"episodes/episode_{episode_id}.hdf5"
        node_metadata = []
        unique_ticks: set[int] = set()
        for annotation in sorted(
            episode_annotations,
            key=lambda item: int(item["cycle_id"]),
        ):
            start, end = map(int, annotation["target_steps_20hz"])
            progress = _progress_for_cycle(annotation)
            for tick in range(start, end):
                node_metadata.append(
                    (
                        annotation,
                        tick,
                        float(progress[tick - start]),
                        float(progress[tick - start + 1]),
                    )
                )
                unique_ticks.add(tick)
                unique_ticks.add(tick + 1)
        ticks = sorted(unique_ticks)
        feature_by_tick: dict[int, np.ndarray] = {}
        for begin in range(0, len(ticks), chunk_rows):
            chunk = ticks[begin : begin + chunk_rows]
            eye = extractor.extract_hdf5_camera_pair(
                episode_path,
                POLICY_EYE_PAIR,
                chunk,
            )
            stick = extractor.extract_hdf5_camera_pair(
                episode_path,
                POLICY_STICK_PAIR,
                chunk,
            )
            compact = compact_visual_features(
                eye,
                stick,
                prototypes=prototypes,
            )
            for index, tick in enumerate(chunk):
                feature_by_tick[tick] = compact[index]
        with h5py.File(episode_path, "r") as episode:
            qpos = episode["observations/qpos"]
            qvel = episode["observations/qvel"]
            action = episode["action"]
            for annotation, tick, progress, next_progress in node_metadata:
                state = np.concatenate(
                    (
                        np.asarray(qpos[tick], dtype=np.float32),
                        np.asarray(qvel[tick], dtype=np.float32),
                        feature_by_tick[tick],
                    )
                ).astype(np.float32)
                next_state = np.concatenate(
                    (
                        np.asarray(qpos[tick + 1], dtype=np.float32),
                        np.asarray(qvel[tick + 1], dtype=np.float32),
                        feature_by_tick[tick + 1],
                    )
                ).astype(np.float32)
                rows["state"].append(state)
                rows["next_state"].append(next_state)
                rows["action"].append(np.asarray(action[tick], dtype=np.float32))
                rows["episode_id"].append(episode_id)
                rows["cycle_id"].append(int(annotation["cycle_id"]))
                rows["tick"].append(tick)
                rows["next_tick"].append(tick + 1)
                rows["progress"].append(progress)
                rows["next_progress"].append(next_progress)
                rows["phase"].append(_phase(progress))
                rows["next_phase"].append(_phase(next_progress))
                rows["cycle_start"].append(
                    tick == int(annotation["target_steps_20hz"][0])
                )
                rows["condition"].append(
                    np.asarray(
                        annotation["policy_condition"]["vector"],
                        dtype=np.float32,
                    )
                )
                rows["current_sector"].append(
                    SECTORS.index(annotation["policy_condition"]["current_sector"])
                )
                rows["next_sector"].append(
                    SECTORS.index(annotation["policy_condition"]["next_ready_sector"])
                )
        print(
            f"transition-stitch: episode {episode_id} {len(node_metadata)} transitions",
            flush=True,
        )
    result = {
        "state": np.stack(rows["state"]).astype(np.float32),
        "next_state": np.stack(rows["next_state"]).astype(np.float32),
        "action": np.stack(rows["action"]).astype(np.float32),
        "episode_id": np.asarray(rows["episode_id"], dtype=np.int64),
        "cycle_id": np.asarray(rows["cycle_id"], dtype=np.int64),
        "tick": np.asarray(rows["tick"], dtype=np.int64),
        "next_tick": np.asarray(rows["next_tick"], dtype=np.int64),
        "progress": np.asarray(rows["progress"], dtype=np.float32),
        "next_progress": np.asarray(rows["next_progress"], dtype=np.float32),
        "phase": np.asarray(rows["phase"], dtype=np.int8),
        "next_phase": np.asarray(rows["next_phase"], dtype=np.int8),
        "cycle_start": np.asarray(rows["cycle_start"], dtype=np.uint8),
        "condition": np.stack(rows["condition"]).astype(np.float32),
        "current_sector": np.asarray(rows["current_sector"], dtype=np.int8),
        "next_sector": np.asarray(rows["next_sector"], dtype=np.int8),
    }
    if result["state"].shape[1] != STATE_DIM:
        raise AssertionError("transition state dimension changed")
    return result


def compact_visual_features(
    eye: np.ndarray,
    stick: np.ndarray,
    *,
    prototypes: Mapping[str, np.ndarray],
) -> np.ndarray:
    eye_array = np.asarray(eye, dtype=np.float32)
    stick_array = np.asarray(stick, dtype=np.float32)
    if (
        eye_array.ndim != 2
        or eye_array.shape != stick_array.shape
        or eye_array.shape[1] != 1024
    ):
        raise ValueError("eye/stick feature arrays must share shape (N,1024)")
    eye_events = np.stack(
        [prototypes[f"event_eye_{name}"] for name in EVENT_NAMES],
        axis=1,
    )
    stick_events = np.stack(
        [prototypes[f"event_stick_{name}"] for name in EVENT_NAMES],
        axis=1,
    )
    sectors = np.stack(
        [prototypes[f"sector_{sector}"] for sector in SECTORS],
        axis=1,
    )
    result = np.concatenate(
        (
            eye_array @ eye_events,
            stick_array @ stick_events,
            eye_array @ sectors,
        ),
        axis=1,
    )
    if result.shape[1] != 13 or not np.isfinite(result).all():
        raise ValueError("compact visual features are invalid")
    return result.astype(np.float32, copy=False)


def _progress_for_cycle(annotation: Mapping[str, Any]) -> np.ndarray:
    start, end = map(int, annotation["target_steps_20hz"])
    event_ticks = [
        start,
        int(
            annotation["observable_events"]["dig_entry_proxy"][
                "representative_target_tick"
            ]
        ),
        int(
            annotation["observable_events"]["carry_transition_proxy"][
                "representative_target_tick"
            ]
        ),
        int(
            annotation["observable_events"]["dump_start_proxy"][
                "representative_target_tick"
            ]
        ),
        int(
            annotation["observable_events"]["dump_end_proxy"][
                "representative_target_tick"
            ]
        ),
        end,
    ]
    if any(right < left for left, right in zip(event_ticks, event_ticks[1:])):
        raise ValueError("observable event ticks are not ordered")
    unique_ticks = []
    unique_progress = []
    for progress, tick in enumerate(event_ticks):
        if unique_ticks and tick == unique_ticks[-1]:
            unique_progress[-1] = float(progress)
        else:
            unique_ticks.append(tick)
            unique_progress.append(float(progress))
    return np.interp(
        np.arange(start, end + 1, dtype=np.float64),
        np.asarray(unique_ticks, dtype=np.float64),
        np.asarray(unique_progress, dtype=np.float64),
    ).astype(np.float32)


def _phase(progress: float) -> int:
    return int(np.clip(math.floor(float(progress) + 1e-6), 0, 4))


def _fit_robust_scale(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError("robust scale requires a finite two-dimensional array")
    center = np.median(array, axis=0)
    q25 = np.quantile(array, 0.25, axis=0)
    q75 = np.quantile(array, 0.75, axis=0)
    iqr = q75 - q25
    std = np.std(array, axis=0)
    scale = np.where(iqr > 0.0, iqr, np.where(std > 0.0, std, 1.0))
    return {
        "method": "train_median_iqr_std_fallback_v1",
        "center": center.tolist(),
        "scale": scale.tolist(),
        "zero_iqr_dimension_count": int(np.sum(iqr == 0.0)),
        "constant_dimension_count": int(np.sum(std == 0.0)),
    }


def _load_prototypes(path: Path) -> dict[str, np.ndarray]:
    required = {
        *(f"event_eye_{name}" for name in EVENT_NAMES),
        *(f"event_stick_{name}" for name in EVENT_NAMES),
        *(f"sector_{sector}" for sector in SECTORS),
    }
    with np.load(path) as archive:
        if not required <= set(archive.files):
            raise ValueError("visual prototype archive is incomplete")
        result = {
            name: np.asarray(archive[name], dtype=np.float32).copy()
            for name in required
        }
    for name, value in result.items():
        if value.shape != (1024,) or not np.isfinite(value).all():
            raise ValueError(f"invalid prototype {name}")
    return result


def _state_feature_names() -> list[str]:
    return [
        *(f"qpos_{axis}" for axis in ("swing", "boom", "stick", "bucket")),
        *(f"qvel_{axis}" for axis in ("swing", "boom", "stick", "bucket")),
        *(f"eye_event_{name}" for name in EVENT_NAMES),
        *(f"stick_event_{name}" for name in EVENT_NAMES),
        *(f"eye_sector_{sector}" for sector in SECTORS),
    ]


def _cycle_start_indices(nodes: Mapping[str, np.ndarray]) -> list[int]:
    return np.flatnonzero(nodes["cycle_start"] == 1).astype(int).tolist()


def _serializable_node_arrays(
    nodes: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {key: value for key, value in nodes.items() if isinstance(value, np.ndarray)}


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return artifact_identity(path)


def _q(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("quantile values must be a finite non-empty vector")
    return float(np.quantile(array, quantile))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
