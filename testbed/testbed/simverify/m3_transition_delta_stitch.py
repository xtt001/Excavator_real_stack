"""Novel-transition local-delta stitching for SimVerify G4-v3."""

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
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.m3_transition_stitch import (
    ACTION_DIM,
    HELD_OUT_EPISODES,
    _q,
    apply_standardization,
    compose_retrieval_vectors,
)

EVIDENCE_SCOPE = "recorded-observation/offline empirical rollout"
TARGET_PROGRESS = 5.0


def build_transition_delta_stitch_calibration(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    m0_root: str | Path,
    transition_stitch_root: str | Path,
    contract_path: str | Path,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Build the immutable expert and action-null G4-v3 prerequisite."""

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        not git.get("git_available")
        or git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
    ):
        raise ValueError("delta stitching requires a clean SimVerify worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable delta-stitch package exists: {destination}")
    m0 = Path(m0_root).resolve(strict=True)
    source = Path(transition_stitch_root).resolve(strict=True)
    contract = Path(contract_path).resolve(strict=True)
    checksum_verification = verify_checksums(source, source / "checksums.sha256")
    if not checksum_verification["ok"]:
        raise ValueError("source transition package checksum verification failed")

    source_manifest = _read_json(source / "transition_stitch_manifest.json")
    source_gate = _read_json(source / "emulator_gate_v1.json")
    source_thresholds = _read_json(source / "expert_stitch_thresholds_v1.json")
    normalization = _read_json(source / "retrieval_normalization_v1.json")
    if source_manifest.get("held_out_test_read") is not False:
        raise ValueError("source transition package held-out lock is invalid")
    if not bool(source_gate["one_step"]["passed"]):
        raise ValueError("delta stitching requires the passed one-step prerequisite")
    support_radius = float(source_thresholds["one_step"]["retrieval_distance_upper"])

    split_manifest = _read_json(m0 / "split_groups.json")
    if set(map(int, split_manifest["splits"]["held_out_test"])) != HELD_OUT_EPISODES:
        raise ValueError("held-out episode lock differs from frozen split")

    train = _load_nodes(source / "train_transition_bank_v1.npz")
    validation = _load_nodes(source / "validation_transition_queries_v1.npz")
    if set(map(int, train["episode_id"])) & HELD_OUT_EPISODES:
        raise ValueError("held-out episode entered the train bank")
    if set(map(int, validation["episode_id"])) & HELD_OUT_EPISODES:
        raise ValueError("held-out episode entered validation queries")
    _attach_successor_actions(train, m0, normalization["action"])
    _attach_successor_actions(validation, m0, normalization["action"])

    max_steps = maximum_cycle_transition_count(train)
    starts_train = np.flatnonzero(train["cycle_start"].astype(bool)).tolist()
    starts_validation = np.flatnonzero(validation["cycle_start"].astype(bool)).tolist()

    print("delta-stitch: train expert candidate", flush=True)
    train_candidate = run_delta_stitch_rollouts(
        bank=train,
        initial=train,
        initial_indices=starts_train,
        support_radius=support_radius,
        max_steps=max_steps,
        action_mode="recorded_expert",
        device=device,
        batch_size=batch_size,
    )
    print("delta-stitch: train median-action null", flush=True)
    train_null = run_delta_stitch_rollouts(
        bank=train,
        initial=train,
        initial_indices=starts_train,
        support_radius=support_radius,
        max_steps=max_steps,
        action_mode="median_action_null",
        device=device,
        batch_size=batch_size,
    )
    train_episode_rows = aggregate_paired_rollouts(train_candidate, train_null)
    thresholds = derive_delta_stitch_thresholds(train_episode_rows)

    print("delta-stitch: validation expert candidate", flush=True)
    validation_candidate = run_delta_stitch_rollouts(
        bank=train,
        initial=validation,
        initial_indices=starts_validation,
        support_radius=support_radius,
        max_steps=max_steps,
        action_mode="recorded_expert",
        device=device,
        batch_size=batch_size,
    )
    print("delta-stitch: validation median-action null", flush=True)
    validation_null = run_delta_stitch_rollouts(
        bank=train,
        initial=validation,
        initial_indices=starts_validation,
        support_radius=support_radius,
        max_steps=max_steps,
        action_mode="median_action_null",
        device=device,
        batch_size=batch_size,
    )
    validation_episode_rows = aggregate_paired_rollouts(
        validation_candidate,
        validation_null,
    )
    gate_evaluation = evaluate_delta_stitch_gate(
        validation_episode_rows,
        thresholds,
        support_radius=support_radius,
    )
    passed = bool(source_gate["one_step"]["passed"] and gate_evaluation["passed"])
    decision = (
        "pass_expert_delta_stitch_prerequisite"
        if passed
        else "offline_emulator_invalid_v3"
    )
    gate = {
        "schema": "simverify_transition_delta_stitch_gate_v1",
        "decision": decision,
        "authorizes_b1_4_policy_stitch": passed,
        "inherited_one_step_gate_passed": bool(source_gate["one_step"]["passed"]),
        "delta_stitch": gate_evaluation,
        "support_radius": support_radius,
        "maximum_steps": max_steps,
        "retrieval_features_include_condition": False,
        "retrieval_features_include_phase_or_progress": False,
        "retrieval_features_include_successor": False,
        "absolute_progress_used_for_rollout_state": False,
        "local_progress_delta_used_post_retrieval": True,
        "transition_reuse_allowed": False,
        "evidence_scope": EVIDENCE_SCOPE,
        "held_out_test_read": False,
        "closed_loop_execution": False,
    }

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    try:
        identities.extend(
            [
                write_jsonl(
                    temporary / "train_expert_delta_stitch_rollouts.jsonl",
                    train_candidate,
                ),
                write_jsonl(
                    temporary / "train_median_action_null_rollouts.jsonl",
                    train_null,
                ),
                write_jsonl(
                    temporary / "validation_expert_delta_stitch_rollouts.jsonl",
                    validation_candidate,
                ),
                write_jsonl(
                    temporary / "validation_median_action_null_rollouts.jsonl",
                    validation_null,
                ),
                write_jsonl(
                    temporary / "train_source_episode_metrics.jsonl",
                    train_episode_rows,
                ),
                write_jsonl(
                    temporary / "validation_source_episode_metrics.jsonl",
                    validation_episode_rows,
                ),
                write_json(
                    temporary / "expert_delta_stitch_thresholds_v1.json",
                    thresholds,
                ),
            ]
        )
        gate_identity = write_json(
            temporary / "delta_stitch_gate_v1.json",
            gate,
        )
        identities.append(gate_identity)
        manifest_identity = write_json(
            temporary / "delta_stitch_manifest.json",
            {
                "schema": "simverify_transition_delta_stitch_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "contract": {
                    "path": str(contract),
                    "sha256": sha256_file(contract),
                },
                "source_transition_package": {
                    "path": str(source),
                    "manifest_sha256": sha256_file(
                        source / "transition_stitch_manifest.json"
                    ),
                    "checksums_sha256": sha256_file(source / "checksums.sha256"),
                    "verified_file_count": checksum_verification["verified_file_count"],
                },
                "m0_dataset_manifest_sha256": sha256_file(m0 / "dataset_manifest.json"),
                "split_manifest_sha256": sha256_file(m0 / "split_groups.json"),
                "train_episode_ids": sorted(set(map(int, train["episode_id"]))),
                "validation_episode_ids": sorted(
                    set(map(int, validation["episode_id"]))
                ),
                "held_out_episode_ids": "locked_unread",
                "train_transition_count": int(train["episode_id"].size),
                "validation_transition_count": int(validation["episode_id"].size),
                "successor_action_source": (
                    "M0 episode HDF5 action[tick+1], source-domain actuator_speed_cmd"
                ),
                "successor_action_episode_sha256": _episode_sha_contract(
                    m0,
                    set(map(int, train["episode_id"]))
                    | set(map(int, validation["episode_id"])),
                ),
                "retrieval_input_fields": [
                    "observable_state",
                    "executed_action",
                ],
                "retrieval_forbidden_fields": [
                    "condition",
                    "phase",
                    "progress",
                    "successor_state",
                    "successor_identity",
                    "future_state",
                    "privilege",
                ],
                "selection_rule": (
                    "exact nearest cross-current-source-episode transition "
                    "excluding transitions already used by this rollout"
                ),
                "progress_rule": (
                    "sum selected local next_progress-progress after retrieval; "
                    "never adopt candidate absolute progress"
                ),
                "null": (
                    "standardized action replaced by train median zero at every step"
                ),
                "decision": decision,
                "authorizes_b1_4_policy_stitch": passed,
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
            "authorizes_b1_4_policy_stitch": passed,
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
                    "schema": "simverify_transition_delta_stitch_failure_v1",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "git": git,
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                },
            )
        raise


def exact_nearest_unseen_indices(
    bank_vectors: np.ndarray,
    query_vectors: np.ndarray,
    *,
    bank_episode_ids: np.ndarray,
    query_episode_ids: np.ndarray,
    seen_indices: Sequence[set[int]],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact cross-episode nearest transition not yet consumed."""

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
        or len(seen_indices) != query.shape[0]
    ):
        raise ValueError("unseen-neighbor array shapes are invalid")
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
        query_episode_tensor = torch.from_numpy(query_episodes[begin:end]).to(resolved)
        squared.masked_fill_(
            query_episode_tensor.reshape(-1, 1) == bank_episode_tensor.reshape(1, -1),
            float("inf"),
        )
        for local_index, consumed in enumerate(seen_indices[begin:end]):
            if consumed:
                selected = torch.as_tensor(
                    sorted(consumed),
                    dtype=torch.int64,
                    device=resolved,
                )
                squared[local_index, selected] = float("inf")
        values, selected = torch.min(squared, dim=1)
        distances.append(
            torch.sqrt(values).to(device="cpu", dtype=torch.float32).numpy()
        )
        indices.append(selected.to(device="cpu", dtype=torch.int64).numpy())
    return (
        np.concatenate(distances).astype(np.float32, copy=False),
        np.concatenate(indices).astype(np.int64, copy=False),
    )


def run_delta_stitch_rollouts(
    *,
    bank: Mapping[str, np.ndarray],
    initial: Mapping[str, np.ndarray],
    initial_indices: Sequence[int],
    support_radius: float,
    max_steps: int,
    action_mode: str,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Accumulate unique local transition deltas under expert or action null."""

    if action_mode not in {"recorded_expert", "median_action_null"}:
        raise ValueError(f"unsupported delta-stitch action mode: {action_mode}")
    active = []
    for initial_index in initial_indices:
        active.append(
            {
                "episode_id": int(initial["episode_id"][initial_index]),
                "cycle_id": int(initial["cycle_id"][initial_index]),
                "current_episode_id": int(initial["episode_id"][initial_index]),
                "state": initial["state_standardized"][initial_index].copy(),
                "action": initial["action_standardized"][initial_index].copy(),
                "accumulated_progress": 0.0,
                "steps": 0,
                "max_distance": 0.0,
                "seen": set(),
                "donor_episodes": set(),
                "status": "active",
            }
        )
    for _step in range(max_steps):
        running = [row for row in active if row["status"] == "active"]
        if not running:
            break
        states = np.stack([row["state"] for row in running])
        if action_mode == "recorded_expert":
            actions = np.stack([row["action"] for row in running])
        else:
            actions = np.zeros((len(running), ACTION_DIM), dtype=np.float32)
        queries = compose_retrieval_vectors(states, actions)
        distances, candidates = exact_nearest_unseen_indices(
            bank["retrieval"],
            queries,
            bank_episode_ids=bank["episode_id"],
            query_episode_ids=np.asarray(
                [row["current_episode_id"] for row in running],
                dtype=np.int64,
            ),
            seen_indices=[row["seen"] for row in running],
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
            if not math.isfinite(float(distance)):
                row["status"] = "offline_support_exhausted"
                continue
            row["max_distance"] = max(row["max_distance"], float(distance))
            if float(distance) > float(support_radius):
                row["status"] = "offline_support_exhausted"
                continue
            if candidate in row["seen"]:
                raise AssertionError("a transition contributed more than once")
            row["seen"].add(candidate)
            donor_episode = int(bank["episode_id"][candidate])
            row["donor_episodes"].add(donor_episode)
            delta = float(
                bank["next_progress"][candidate] - bank["progress"][candidate]
            )
            if not math.isfinite(delta) or delta <= 0.0:
                row["status"] = "invalid_local_progress_delta"
                continue
            row["accumulated_progress"] += delta
            row["state"] = bank["next_state_standardized"][candidate].copy()
            row["action"] = bank["next_action_standardized"][candidate].copy()
            row["current_episode_id"] = donor_episode
            if row["accumulated_progress"] >= TARGET_PROGRESS - 1e-6:
                row["status"] = "completed_ready_to_ready_delta"
    result = []
    for row in active:
        if row["status"] == "active":
            row["status"] = "max_train_duration_reached"
        result.append(
            {
                "schema": "simverify_expert_transition_delta_stitch_rollout_v1",
                "episode_id": row["episode_id"],
                "cycle_id": row["cycle_id"],
                "action_mode": action_mode,
                "status": row["status"],
                "steps": row["steps"],
                "accumulated_progress": row["accumulated_progress"],
                "unique_transition_count": len(row["seen"]),
                "unique_donor_episode_count": len(row["donor_episodes"]),
                "max_retrieval_distance": row["max_distance"],
                "completed": row["status"] == "completed_ready_to_ready_delta",
                "absolute_progress_used_for_rollout_state": False,
                "condition_used_for_retrieval": False,
                "closed_loop_execution": False,
            }
        )
    return result


def aggregate_paired_rollouts(
    candidate_rows: Sequence[Mapping[str, Any]],
    null_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate matched candidate/null cycles by source episode."""

    candidate = {
        (int(row["episode_id"]), int(row["cycle_id"])): row for row in candidate_rows
    }
    null = {(int(row["episode_id"]), int(row["cycle_id"])): row for row in null_rows}
    if set(candidate) != set(null):
        raise ValueError("candidate and null cycle keys differ")
    grouped: dict[int, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(
        list
    )
    for key in sorted(candidate):
        grouped[key[0]].append((candidate[key], null[key]))
    result = []
    for episode_id, pairs in sorted(grouped.items()):
        candidate_completion = float(
            np.mean([bool(pair[0]["completed"]) for pair in pairs])
        )
        null_completion = float(np.mean([bool(pair[1]["completed"]) for pair in pairs]))
        completed_steps = [
            int(pair[0]["steps"]) for pair in pairs if bool(pair[0]["completed"])
        ]
        result.append(
            {
                "schema": "simverify_delta_stitch_source_episode_metric_v1",
                "episode_id": episode_id,
                "cycle_count": len(pairs),
                "candidate_completion_rate": candidate_completion,
                "median_action_null_completion_rate": null_completion,
                "paired_completion_delta": candidate_completion - null_completion,
                "candidate_completed_steps_q97_5": (
                    _q(completed_steps, 0.975)
                    if completed_steps
                    else max(int(pair[0]["steps"]) for pair in pairs)
                ),
                "candidate_max_retrieval_distance": max(
                    float(pair[0]["max_retrieval_distance"]) for pair in pairs
                ),
            }
        )
    return result


def derive_delta_stitch_thresholds(
    train_episode_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze source-episode envelopes from train-only paired rollouts."""

    return {
        "schema": "simverify_expert_delta_stitch_thresholds_v1",
        "aggregation": "cycle_then_source_episode",
        "candidate_completion_rate_lower": _q(
            [row["candidate_completion_rate"] for row in train_episode_rows],
            0.025,
        ),
        "paired_completion_delta_lower": _q(
            [row["paired_completion_delta"] for row in train_episode_rows],
            0.025,
        ),
        "median_action_null_completion_rate_upper": _q(
            [row["median_action_null_completion_rate"] for row in train_episode_rows],
            0.975,
        ),
        "candidate_completed_steps_q97_5_upper": _q(
            [row["candidate_completed_steps_q97_5"] for row in train_episode_rows],
            0.975,
        ),
        "candidate_max_retrieval_distance_upper": _q(
            [row["candidate_max_retrieval_distance"] for row in train_episode_rows],
            0.975,
        ),
        "source_episode_count": len(train_episode_rows),
        "source": "train_leave_one_current_donor_episode_out",
    }


def evaluate_delta_stitch_gate(
    validation_episode_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    *,
    support_radius: float,
) -> dict[str, Any]:
    """Apply frozen train envelopes to validation source episodes."""

    criteria = {
        "train_candidate_identifies_completion": {
            "observed_train_lower": thresholds["candidate_completion_rate_lower"],
            "minimum_allowed": 0.0,
            "comparison": "strictly_greater",
            "passed": thresholds["candidate_completion_rate_lower"] > 0.0,
        },
        "train_action_dependence_is_nonzero": {
            "observed_train_lower": thresholds["paired_completion_delta_lower"],
            "minimum_allowed": 0.0,
            "comparison": "strictly_greater",
            "passed": thresholds["paired_completion_delta_lower"] > 0.0,
        },
        "validation_candidate_completion": {
            "observed_min_episode_rate": min(
                row["candidate_completion_rate"] for row in validation_episode_rows
            ),
            "minimum_allowed": thresholds["candidate_completion_rate_lower"],
        },
        "validation_action_dependence": {
            "observed_min_episode_delta": min(
                row["paired_completion_delta"] for row in validation_episode_rows
            ),
            "minimum_allowed": thresholds["paired_completion_delta_lower"],
        },
        "validation_median_action_null": {
            "observed_max_episode_rate": max(
                row["median_action_null_completion_rate"]
                for row in validation_episode_rows
            ),
            "maximum_allowed": thresholds["median_action_null_completion_rate_upper"],
        },
        "validation_completion_steps": {
            "observed_max_episode_q97_5": max(
                row["candidate_completed_steps_q97_5"]
                for row in validation_episode_rows
            ),
            "maximum_allowed": thresholds["candidate_completed_steps_q97_5_upper"],
        },
        "validation_retrieval_distance": {
            "observed_max_episode_distance": max(
                row["candidate_max_retrieval_distance"]
                for row in validation_episode_rows
            ),
            "maximum_allowed": min(
                support_radius,
                thresholds["candidate_max_retrieval_distance_upper"],
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
    return {
        "criteria": criteria,
        "passed": all(bool(row["passed"]) for row in criteria.values()),
        "validation_source_episode_count": len(validation_episode_rows),
    }


def maximum_cycle_transition_count(nodes: Mapping[str, np.ndarray]) -> int:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for episode_id, cycle_id in zip(
        nodes["episode_id"],
        nodes["cycle_id"],
        strict=True,
    ):
        counts[(int(episode_id), int(cycle_id))] += 1
    if not counts:
        raise ValueError("transition bank has no cycles")
    return max(counts.values())


def _attach_successor_actions(
    nodes: dict[str, np.ndarray],
    m0_root: Path,
    action_normalization: Mapping[str, Any],
) -> None:
    next_actions = np.empty((nodes["episode_id"].size, ACTION_DIM), dtype=np.float32)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, episode_id in enumerate(nodes["episode_id"].tolist()):
        grouped[int(episode_id)].append(index)
    for episode_id, indices in grouped.items():
        path = m0_root / f"episodes/episode_{episode_id}.hdf5"
        with h5py.File(path, "r") as episode:
            actions = episode["action"]
            for index in indices:
                tick = int(nodes["next_tick"][index])
                if tick >= actions.shape[0]:
                    raise ValueError(
                        f"episode {episode_id} lacks successor action at tick {tick}"
                    )
                next_actions[index] = np.asarray(actions[tick], dtype=np.float32)
    nodes["next_action"] = next_actions
    nodes["next_action_standardized"] = apply_standardization(
        next_actions,
        action_normalization,
    )


def _episode_sha_contract(
    m0_root: Path,
    episode_ids: set[int],
) -> dict[str, str]:
    return {
        str(episode_id): sha256_file(m0_root / f"episodes/episode_{episode_id}.hdf5")
        for episode_id in sorted(episode_ids)
    }


def _load_nodes(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as package:
        return {key: package[key] for key in package.files}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
