"""Recorded-observation B0 replay with immutable three-stage action traces."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from testbed.data.dataset import _read_camera_image
from testbed.policies.offline_eval import load_policy_for_episode
from testbed.simverify.artifacts import (
    artifact_identity,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.m2_eval import (
    effective_signature,
    extract_ordered_task_events,
    validate_replay_trace_arrays,
)

DEFAULT_M0_ROOT = Path(
    "/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3"
)
DEFAULT_M2_ROOT = Path(
    "/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3_m2_contract_v1"
)
DEFAULT_B0_BUNDLE = Path(
    "/data/pingfan/Excavator_real_stack_data/simverify_b0_unconditioned_v1_seed0"
)
CAMERAS = ("video4", "video5", "video6", "video7")


def replay_cycle_arrays(
    *,
    policy: Any,
    episode: h5py.File,
    annotation: Mapping[str, Any],
    camera_names: Sequence[str] = CAMERAS,
    condition_override: Sequence[float] | None = None,
    pass_condition_to_policy: bool = False,
) -> dict[str, np.ndarray]:
    """Replay one accepted cycle, including its shared ready-end boundary."""

    start, end = map(int, annotation["target_steps_20hz"])
    total_steps = int(episode["action"].shape[0])
    if not 0 <= start < end < total_steps:
        raise ValueError("cycle replay requires 0 <= start < end < episode length")
    condition = np.asarray(
        (
            annotation["policy_condition"]["vector"]
            if condition_override is None
            else condition_override
        ),
        dtype=np.float32,
    )
    if condition.shape != (6,):
        raise ValueError("accepted cycle condition must have shape (6,)")
    if hasattr(policy, "reset"):
        policy.reset()

    step_count = end - start + 1
    query_count: int | None = None
    raw_normalized_rows: list[np.ndarray] = []
    raw_direct_rows: list[np.ndarray] = []
    aggregated = np.zeros((step_count, 4), dtype=np.float32)
    qpos = episode["observations/qpos"]
    qvel = episode["observations/qvel"]
    for local_index, target_tick in enumerate(range(start, end + 1)):
        observation: dict[str, Any] = {
            "qpos": np.asarray(qpos[target_tick], dtype=np.float32),
            "qvel": np.asarray(qvel[target_tick], dtype=np.float32),
        }
        if pass_condition_to_policy:
            observation["cycle_condition_v1"] = condition.copy()
        for camera in camera_names:
            observation[f"image_{camera}"] = _read_camera_image(
                episode,
                camera,
                target_tick,
            )
        aggregated[local_index] = np.asarray(
            policy.predict(observation),
            dtype=np.float32,
        ).reshape(4)
        raw_normalized = np.asarray(
            policy.last_raw_action_chunk(),
            dtype=np.float32,
        )
        raw_direct = np.asarray(
            policy.last_raw_action_chunk_direct(),
            dtype=np.float32,
        )
        if (
            raw_normalized.ndim != 2
            or raw_normalized.shape[1] != 4
            or raw_direct.shape != raw_normalized.shape
        ):
            raise ValueError("policy raw chunks must share shape (Q, 4)")
        if query_count is None:
            query_count = int(raw_normalized.shape[0])
        if raw_normalized.shape[0] != query_count:
            raise ValueError("policy query count changed during cycle replay")
        raw_normalized_rows.append(raw_normalized.copy())
        raw_direct_rows.append(raw_direct.copy())

    source_index = np.asarray(
        episode["diagnostics/source_observation_index"][start : end + 1],
        dtype=np.int64,
    )
    target_tick = np.asarray(
        episode["diagnostics/target_tick"][start : end + 1],
        dtype=np.int64,
    )
    arrays = {
        "raw_policy_chunk_normalized": np.stack(raw_normalized_rows).astype(np.float32),
        "raw_policy_chunk_direct": np.stack(raw_direct_rows).astype(np.float32),
        "temporal_aggregation_action": aggregated.copy(),
        "future_runtime_safe_action": aggregated.copy(),
        "expert_action": np.asarray(
            episode["action"][start : end + 1],
            dtype=np.float32,
        ),
        "condition": np.repeat(
            condition.reshape(1, 6),
            step_count,
            axis=0,
        ).astype(np.float32),
        "target_tick": target_tick,
        "source_observation_index": source_index,
        "condition_cycle_id": np.full(
            step_count,
            int(annotation["cycle_id"]),
            dtype=np.int64,
        ),
        "condition_valid_mask": np.ones(step_count, dtype=np.uint8),
        "observation_age_ticks": np.zeros(step_count, dtype=np.int64),
        "action_age_ticks": np.zeros(step_count, dtype=np.int64),
    }
    validate_replay_trace_arrays(arrays, chunk_size=int(query_count or 0))
    return arrays


def cycle_action_metrics(
    arrays: Mapping[str, np.ndarray],
    *,
    templates: Mapping[str, Mapping[str, Any]],
    deadzone: Sequence[float],
) -> dict[str, Any]:
    expert = np.asarray(arrays["expert_action"], dtype=np.float32)
    policy = np.asarray(
        arrays["temporal_aggregation_action"],
        dtype=np.float32,
    )
    extracted = extract_ordered_task_events(
        policy,
        templates,
        deadzone=deadzone,
    )
    expert_signatures = [effective_signature(row, deadzone) for row in expert]
    policy_signatures = [effective_signature(row, deadzone) for row in policy]
    expert_effective = 0
    same_direction = 0
    opposite_direction = 0
    unexpected_effective = 0
    policy_effective = 0
    for expert_signature, policy_signature in zip(
        expert_signatures,
        policy_signatures,
    ):
        for expert_sign, policy_sign in zip(
            expert_signature,
            policy_signature,
        ):
            expert_effective += int(expert_sign != 0)
            policy_effective += int(policy_sign != 0)
            same_direction += int(expert_sign != 0 and policy_sign == expert_sign)
            opposite_direction += int(expert_sign != 0 and policy_sign == -expert_sign)
            unexpected_effective += int(expert_sign == 0 and policy_sign != 0)
    return {
        "schema": "simverify_b0_cycle_metrics_v1",
        "required_event_coverage": extracted["required_event_coverage"],
        "event_order_valid": extracted["event_order_valid"],
        "event_order_violation_rate": float(not extracted["event_order_valid"]),
        "missing_phase_rate": float(1.0 - extracted["required_event_coverage"]),
        "missing_events": extracted["missing_events"],
        "event_ticks_local": extracted["event_ticks"],
        "deadzone_effective_recall": _safe_rate(
            same_direction,
            expert_effective,
        ),
        "opposite_direction_rate": _safe_rate(
            opposite_direction,
            expert_effective,
        ),
        "unexpected_effective_axis_rate": _safe_rate(
            unexpected_effective,
            policy_effective,
        ),
        "expert_effective_axis_ticks": expert_effective,
        "policy_effective_axis_ticks": policy_effective,
        "same_direction_axis_ticks": same_direction,
        "opposite_direction_axis_ticks": opposite_direction,
        "unexpected_effective_axis_ticks": unexpected_effective,
        "action_mae_auxiliary": float(np.mean(np.abs(policy - expert))),
        "closed_loop_execution": False,
    }


def run_b0_replay(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    split_name: str,
    repeat_id: int,
    m0_root: str | Path = DEFAULT_M0_ROOT,
    m2_root: str | Path = DEFAULT_M2_ROOT,
    bundle_root: str | Path = DEFAULT_B0_BUNDLE,
    checkpoint_name: str = "policy_best.ckpt",
) -> dict[str, Any]:
    if split_name not in {"train", "validation"}:
        raise ValueError("B0 calibration replay split must be train or validation")
    if repeat_id < 0:
        raise ValueError("repeat_id must be non-negative")
    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
        or not git.get("git_available")
    ):
        raise ValueError("B0 replay requires a clean SimVerify worktree")

    m0 = Path(m0_root).resolve(strict=True)
    m2 = Path(m2_root).resolve(strict=True)
    bundle = Path(bundle_root).resolve(strict=True)
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable B0 replay output exists: {destination}")
    metadata = _read_json(bundle / "run_metadata.json")
    if metadata.get("status") != "completed":
        raise ValueError("B0 replay requires completed training metadata")
    if metadata["experiment_contract"]["condition_input"] != "absent":
        raise ValueError("B0 replay checkpoint must be unconditioned")
    if metadata["checkpoint_semantics"]["real_control_allowed"] is not False:
        raise ValueError("B0 replay checkpoint lacks real-control prohibition")
    split_key = "train_ids" if split_name == "train" else "val_ids"
    episode_ids = list(map(int, metadata["split"][split_key]))
    if set(episode_ids) & {1, 13, 25, 33}:
        raise ValueError("held-out episode entered B0 calibration replay")

    checkpoint = bundle / checkpoint_name
    checkpoint_contract = _validate_b0_checkpoint_contract(checkpoint)
    event_envelope = _read_json(m2 / "expert_event_envelope_v1.json")
    templates = event_envelope["templates"]
    deadzone = list(map(float, event_envelope["effective_deadzone"]))
    annotations = [
        row
        for row in _read_jsonl(m0 / "cycle_annotations.jsonl")
        if row["quality"]["status"] == "accepted"
        and int(row["episode_id"]) in set(episode_ids)
        and row["split"] == split_name
    ]
    if not annotations:
        raise ValueError("no accepted cycles selected for B0 replay")

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    policy = load_policy_for_episode(
        bundle_dir=bundle,
        ckpt_path=checkpoint,
        resolved_config_path=None,
        stats_path=None,
        max_episode_len=8000,
        temporal_agg=True,
        device="cuda",
        inference_precision="fp32",
    )

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    try:
        grouped: dict[int, list[Mapping[str, Any]]] = {}
        for annotation in annotations:
            grouped.setdefault(int(annotation["episode_id"]), []).append(annotation)
        for episode_id, episode_annotations in sorted(grouped.items()):
            episode_path = m0 / f"episodes/episode_{episode_id}.hdf5"
            with h5py.File(episode_path, "r") as episode:
                for annotation in sorted(
                    episode_annotations,
                    key=lambda row: int(row["cycle_id"]),
                ):
                    arrays = replay_cycle_arrays(
                        policy=policy,
                        episode=episode,
                        annotation=annotation,
                    )
                    metrics = cycle_action_metrics(
                        arrays,
                        templates=templates,
                        deadzone=deadzone,
                    )
                    cycle_id = int(annotation["cycle_id"])
                    relative = Path("traces") / (
                        f"episode_{episode_id}_cycle_{cycle_id}.npz"
                    )
                    trace_path = temporary / relative
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(trace_path, **arrays)
                    identity = artifact_identity(trace_path)
                    identities.append(identity)
                    cycle_rows.append(
                        {
                            "schema": "simverify_b0_cycle_replay_v1",
                            "episode_id": episode_id,
                            "cycle_id": cycle_id,
                            "split": split_name,
                            "target_steps_20hz": annotation["target_steps_20hz"],
                            "trace_path": str(relative),
                            "trace_sha256": identity["sha256"],
                            "trace_size_bytes": identity["size_bytes"],
                            "condition_recorded_for_eval_only": True,
                            "condition_input_used_by_policy": False,
                            "metrics": metrics,
                        }
                    )
        rows_identity = write_jsonl(
            temporary / "cycle_metrics.jsonl",
            cycle_rows,
        )
        identities.append(rows_identity)
        provenance = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git": git,
            "evidence_scope": "recorded-observation/offline",
            "closed_loop_execution": False,
            "held_out_test_read": False,
            "repeat_id": repeat_id,
            "inference_precision": "fp32",
            "temporal_aggregation": True,
            "future_runtime_safe_action_transform": (
                "identity_copy_offline_not_deployment"
            ),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "embedded_contract": checkpoint_contract,
            },
            "dataset_stats_sha256": sha256_file(bundle / "dataset_stats.pkl"),
            "resolved_config_sha256": sha256_file(bundle / "resolved_config.yaml"),
            "m0_dataset_manifest_sha256": sha256_file(m0 / "dataset_manifest.json"),
            "camera_mapping_sha256": sha256_file(m0 / "camera_mapping.json"),
            "condition_schema_sha256": sha256_file(
                m0 / "cycle_condition_v1.schema.json"
            ),
            "m2_manifest_sha256": sha256_file(m2 / "m2_manifest.json"),
            "test_intent_registry_sha256": sha256_file(
                m2 / "test_intent_registry_v1.json"
            ),
        }
        manifest_identity = write_json(
            temporary / "replay_manifest.json",
            {
                "schema": "simverify_b0_replay_manifest_v1",
                "baseline_id": "B0",
                "split": split_name,
                "repeat_id": repeat_id,
                "episode_ids": episode_ids,
                "cycle_count": len(cycle_rows),
                "cycle_id_range": [
                    min(int(row["cycle_id"]) for row in cycle_rows),
                    max(int(row["cycle_id"]) for row in cycle_rows),
                ],
                "target_tick_range": [
                    min(int(row["target_steps_20hz"][0]) for row in cycle_rows),
                    max(int(row["target_steps_20hz"][1]) for row in cycle_rows),
                ],
                "condition_input_used_by_policy": False,
                "condition_source": "hindsight_explicit_provenance",
                "command_source": "unknown_not_recorded",
                "checkpoint_training_status": "completed",
                "training_performed_by_replay": False,
                "gate_thresholds_v1_generated": False,
                "held_out_test_read": False,
                "closed_loop_execution": False,
                "provenance": provenance,
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
            "split": split_name,
            "repeat_id": repeat_id,
            "episode_count": len(episode_ids),
            "cycle_count": len(cycle_rows),
            "replay_manifest_sha256": manifest_identity["sha256"],
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
                    "schema": "simverify_b0_replay_failure_v1",
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "split": split_name,
                    "repeat_id": repeat_id,
                    "evidence_scope": "recorded-observation/offline",
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                    "git": git,
                },
            )
        raise


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_b0_checkpoint_contract(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no embedded config mapping")
    semantics = config.get("checkpoint_semantics")
    experiment = config.get("experiment_contract")
    if not isinstance(semantics, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError("checkpoint lacks embedded SimVerify contracts")
    if (
        semantics.get("domain") != "sim"
        or semantics.get("real_control_allowed") is not False
        or semantics.get("jetson_allowed") is not False
    ):
        raise ValueError("checkpoint embedded sim-domain prohibition is invalid")
    if (
        experiment.get("baseline_id") != "B0"
        or experiment.get("condition_input") != "absent"
    ):
        raise ValueError("checkpoint embedded experiment is not unconditioned B0")
    return {
        "checkpoint_semantics": dict(semantics),
        "experiment_contract": dict(experiment),
    }
