"""One-tick-history amendment for the SimVerify transition stitcher."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from testbed.simverify.artifacts import (
    artifact_identity,
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.m3_transition_stitch import (
    ACTION_DIM,
    STATE_DIM,
    _cycle_start_indices,
    _fit_robust_scale,
    aggregate_one_step_by_episode,
    aggregate_rollouts_by_episode,
    apply_standardization,
    derive_cumulative_thresholds,
    derive_one_step_thresholds,
    evaluate_cumulative_gate,
    evaluate_one_step_gate,
    exact_nearest_indices,
    one_step_metrics,
)


def build_history_stitch_calibration(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    base_stitch_root: str | Path,
    contract_path: str | Path,
    knn_query_batch_size: int = 256,
    device: str = "cuda",
) -> dict[str, Any]:
    if knn_query_batch_size <= 0:
        raise ValueError("KNN batch size must be positive")
    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
        or not git.get("git_available")
    ):
        raise ValueError("history stitching requires a clean SimVerify worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable history stitch exists: {destination}")
    base = Path(base_stitch_root).resolve(strict=True)
    contract = Path(contract_path).resolve(strict=True)
    verification = verify_checksums(base, base / "checksums.sha256")
    if not verification["ok"]:
        raise ValueError("base transition stitch checksum verification failed")
    base_manifest = _read_json(base / "transition_stitch_manifest.json")
    if (
        base_manifest.get("expert_self_replay_decision") != "offline_emulator_invalid"
        or base_manifest.get("held_out_test_read") is not False
        or base_manifest.get("closed_loop_execution") is not False
    ):
        raise ValueError("history amendment requires the failed offline v1 package")
    train = _load_nodes(base / "train_transition_bank_v1.npz")
    validation = _load_nodes(base / "validation_transition_queries_v1.npz")
    history_normalization = _fit_history_retrieval(train)
    for nodes in (train, validation):
        _attach_history_features(nodes, history_normalization)

    print("history-stitch: exact train leave-one-episode retrieval", flush=True)
    train_distance, train_neighbor = exact_nearest_indices(
        train["history_retrieval"],
        train["history_retrieval"],
        bank_episode_ids=train["episode_id"],
        query_episode_ids=train["episode_id"],
        exclude_same_episode=True,
        device=device,
        batch_size=knn_query_batch_size,
    )
    print("history-stitch: exact validation-to-train retrieval", flush=True)
    validation_distance, validation_neighbor = exact_nearest_indices(
        train["history_retrieval"],
        validation["history_retrieval"],
        bank_episode_ids=train["episode_id"],
        query_episode_ids=validation["episode_id"],
        exclude_same_episode=True,
        device=device,
        batch_size=knn_query_batch_size,
    )
    train_one_step = one_step_metrics(
        query=train,
        bank=train,
        neighbor_distance=train_distance,
        neighbor_index=train_neighbor,
        split="train_leave_one_source_episode_out_history_v1",
    )
    validation_one_step = one_step_metrics(
        query=validation,
        bank=train,
        neighbor_distance=validation_distance,
        neighbor_index=validation_neighbor,
        split="validation_history_v1",
    )
    train_one_step_episodes = aggregate_one_step_by_episode(train_one_step)
    validation_one_step_episodes = aggregate_one_step_by_episode(validation_one_step)
    one_step_thresholds = derive_one_step_thresholds(train_one_step_episodes)
    one_step_gate = evaluate_one_step_gate(
        validation_one_step_episodes,
        one_step_thresholds,
    )
    duration_max = _maximum_cycle_transition_count(train)
    print("history-stitch: cumulative expert train rollout", flush=True)
    train_rollouts = run_history_expert_rollouts(
        bank=train,
        initial=train,
        initial_indices=_cycle_start_indices(train),
        history_normalization=history_normalization,
        support_radius=one_step_thresholds["retrieval_distance_upper"],
        max_steps=duration_max,
        device=device,
        batch_size=knn_query_batch_size,
    )
    print("history-stitch: cumulative expert validation rollout", flush=True)
    validation_rollouts = run_history_expert_rollouts(
        bank=train,
        initial=validation,
        initial_indices=_cycle_start_indices(validation),
        history_normalization=history_normalization,
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
    passed = bool(one_step_gate["passed"] and cumulative_gate["passed"])
    gate = {
        "schema": "simverify_history_transition_stitch_emulator_gate_v1",
        "decision": (
            "pass_expert_history_stitch_prerequisite"
            if passed
            else "offline_emulator_invalid"
        ),
        "authorizes_condition_rollout": passed,
        "one_step": one_step_gate,
        "cumulative": cumulative_gate,
        "single_method_change": "add_one_tick_observable_state_action_delta",
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
                temporary / "train_history_transition_bank_v1.npz",
                train,
            )
        )
        identities.append(
            _write_npz(
                temporary / "validation_history_transition_queries_v1.npz",
                validation,
            )
        )
        identities.append(
            write_json(
                temporary / "history_retrieval_normalization_v1.json",
                {
                    "schema": ("simverify_history_retrieval_normalization_v1"),
                    "state_delta": history_normalization["state_delta"],
                    "action_delta": history_normalization["action_delta"],
                    "current_state_normalization": (
                        "inherited_from_base_transition_package"
                    ),
                    "current_action_normalization": (
                        "inherited_from_base_transition_package"
                    ),
                    "distance": (
                        "four_equal_rms_groups_current_state_state_delta_"
                        "current_action_action_delta"
                    ),
                    "fit_split": "train_only",
                },
            )
        )
        identities.append(
            write_jsonl(
                temporary / "train_one_step_history_metrics.jsonl",
                train_one_step,
            )
        )
        identities.append(
            write_jsonl(
                temporary / "validation_one_step_history_metrics.jsonl",
                validation_one_step,
            )
        )
        identities.append(
            write_jsonl(
                temporary / "train_expert_history_rollouts.jsonl",
                train_rollouts,
            )
        )
        identities.append(
            write_jsonl(
                temporary / "validation_expert_history_rollouts.jsonl",
                validation_rollouts,
            )
        )
        identities.append(
            write_json(
                temporary / "expert_history_thresholds_v1.json",
                {
                    "schema": "simverify_expert_history_thresholds_v1",
                    "one_step": one_step_thresholds,
                    "cumulative": cumulative_thresholds,
                    "source": "train_source_episode_leave_one_out",
                    "held_out_test_read": False,
                },
            )
        )
        identities.append(write_json(temporary / "history_emulator_gate_v1.json", gate))
        manifest_identity = write_json(
            temporary / "history_stitch_manifest.json",
            {
                "schema": "simverify_history_stitch_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "contract": {
                    "path": str(contract),
                    "sha256": sha256_file(contract),
                },
                "base_stitch_package": {
                    "path": str(base),
                    "manifest_sha256": sha256_file(
                        base / "transition_stitch_manifest.json"
                    ),
                    "checksums_sha256": sha256_file(base / "checksums.sha256"),
                    "decision": base_manifest["expert_self_replay_decision"],
                },
                "single_method_change": ("add_one_tick_observable_state_action_delta"),
                "train_transition_count": int(train["episode_id"].size),
                "validation_transition_count": int(validation["episode_id"].size),
                "held_out_episode_ids": "locked_unread",
                "expert_self_replay_decision": gate["decision"],
                "authorizes_condition_rollout": passed,
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
            "authorizes_condition_rollout": passed,
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
                    "schema": "simverify_history_stitch_failure_v1",
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "git": git,
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                },
            )
        raise


def compose_history_retrieval_vectors(
    current_state_standardized: np.ndarray,
    state_delta_standardized: np.ndarray,
    current_action_standardized: np.ndarray,
    action_delta_standardized: np.ndarray,
) -> np.ndarray:
    state = np.asarray(current_state_standardized, dtype=np.float32)
    state_delta = np.asarray(state_delta_standardized, dtype=np.float32)
    action = np.asarray(current_action_standardized, dtype=np.float32)
    action_delta = np.asarray(action_delta_standardized, dtype=np.float32)
    if (
        state.ndim != 2
        or state.shape != state_delta.shape
        or state.shape[1] != STATE_DIM
        or action.ndim != 2
        or action.shape != action_delta.shape
        or action.shape[1] != ACTION_DIM
        or state.shape[0] != action.shape[0]
    ):
        raise ValueError("history retrieval group shapes are invalid")
    return np.concatenate(
        (
            state / math.sqrt(float(STATE_DIM)),
            state_delta / math.sqrt(float(STATE_DIM)),
            action / math.sqrt(float(ACTION_DIM)),
            action_delta / math.sqrt(float(ACTION_DIM)),
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def derive_history_deltas(
    nodes: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    count = int(nodes["episode_id"].size)
    state_delta = np.zeros((count, STATE_DIM), dtype=np.float32)
    action_delta = np.zeros((count, ACTION_DIM), dtype=np.float32)
    previous: dict[tuple[int, int, int], int] = {}
    for index in range(count):
        key = (
            int(nodes["episode_id"][index]),
            int(nodes["cycle_id"][index]),
            int(nodes["tick"][index]),
        )
        previous_index = previous.get((key[0], key[1], key[2] - 1))
        if previous_index is not None:
            state_delta[index] = nodes["state"][index] - nodes["state"][previous_index]
            action_delta[index] = (
                nodes["action"][index] - nodes["action"][previous_index]
            )
        previous[key] = index
    return state_delta, action_delta


def run_history_expert_rollouts(
    *,
    bank: dict[str, np.ndarray],
    initial: dict[str, np.ndarray],
    initial_indices: list[int],
    history_normalization: dict[str, Any],
    support_radius: float,
    max_steps: int,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    key_to_bank_index = {
        (int(episode), int(tick)): index
        for index, (episode, tick) in enumerate(
            zip(bank["episode_id"], bank["tick"], strict=True)
        )
    }
    active: list[dict[str, Any]] = []
    for start_index in initial_indices:
        active.append(
            {
                "episode_id": int(initial["episode_id"][start_index]),
                "cycle_id": int(initial["cycle_id"][start_index]),
                "current_episode_id": int(initial["episode_id"][start_index]),
                "state": initial["state"][start_index].copy(),
                "action": initial["action"][start_index].copy(),
                "state_standardized": initial["state_standardized"][start_index].copy(),
                "action_standardized": initial["action_standardized"][
                    start_index
                ].copy(),
                "state_delta": np.zeros(STATE_DIM, dtype=np.float32),
                "action_delta": np.zeros(ACTION_DIM, dtype=np.float32),
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
    for _step in range(max_steps):
        running = [row for row in active if row["status"] == "active"]
        if not running:
            break
        state_delta = apply_standardization(
            np.stack([row["state_delta"] for row in running]),
            history_normalization["state_delta"],
        )
        action_delta = apply_standardization(
            np.stack([row["action_delta"] for row in running]),
            history_normalization["action_delta"],
        )
        query = compose_history_retrieval_vectors(
            np.stack([row["state_standardized"] for row in running]),
            state_delta,
            np.stack([row["action_standardized"] for row in running]),
            action_delta,
        )
        distances, candidates = exact_nearest_indices(
            bank["history_retrieval"],
            query,
            bank_episode_ids=bank["episode_id"],
            query_episode_ids=np.asarray(
                [row["current_episode_id"] for row in running],
                dtype=np.int64,
            ),
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
            next_state = bank["state"][next_index].copy()
            next_action = bank["action"][next_index].copy()
            row["state_delta"] = next_state - row["state"]
            row["action_delta"] = next_action - row["action"]
            row["state"] = next_state
            row["action"] = next_action
            row["state_standardized"] = bank["state_standardized"][next_index].copy()
            row["action_standardized"] = bank["action_standardized"][next_index].copy()
            row["current_episode_id"] = next_episode
    result = []
    for row in active:
        if row["status"] == "active":
            row["status"] = "max_train_duration_reached"
        result.append(
            {
                "schema": ("simverify_expert_history_transition_rollout_v1"),
                "episode_id": row["episode_id"],
                "cycle_id": row["cycle_id"],
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
    return result


def _fit_history_retrieval(
    train: dict[str, np.ndarray],
) -> dict[str, Any]:
    state_delta, action_delta = derive_history_deltas(train)
    return {
        "state_delta": _fit_robust_scale(state_delta),
        "action_delta": _fit_robust_scale(action_delta),
    }


def _attach_history_features(
    nodes: dict[str, np.ndarray],
    normalization: dict[str, Any],
) -> None:
    state_delta, action_delta = derive_history_deltas(nodes)
    nodes["state_delta"] = state_delta
    nodes["action_delta"] = action_delta
    nodes["state_delta_standardized"] = apply_standardization(
        state_delta,
        normalization["state_delta"],
    )
    nodes["action_delta_standardized"] = apply_standardization(
        action_delta,
        normalization["action_delta"],
    )
    nodes["history_retrieval"] = compose_history_retrieval_vectors(
        nodes["state_standardized"],
        nodes["state_delta_standardized"],
        nodes["action_standardized"],
        nodes["action_delta_standardized"],
    )


def _maximum_cycle_transition_count(nodes: dict[str, np.ndarray]) -> int:
    counts: dict[tuple[int, int], int] = {}
    for episode, cycle in zip(
        nodes["episode_id"],
        nodes["cycle_id"],
        strict=True,
    ):
        key = (int(episode), int(cycle))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) + 1


def _load_nodes(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _write_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return artifact_identity(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
