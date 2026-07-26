"""Condition-swap replay for the matched SimVerify B1/B2 checkpoints."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from testbed.policies.offline_eval import load_policy_for_episode
from testbed.simverify.artifacts import (
    artifact_identity,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.m3_replay import (
    CAMERAS,
    cycle_action_metrics,
    replay_cycle_arrays,
)

SECTORS = ("left", "center", "right")


def run_condition_swap_replay(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    bundle_root: str | Path,
    repeat_id: int,
    split_name: str = "validation",
    m0_root: str | Path = (
        "/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3"
    ),
    m2_root: str | Path = (
        "/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3_m2_contract_v1"
    ),
    condition_delivery_mode: str = "requested",
) -> dict[str, Any]:
    if split_name not in {"train", "validation"}:
        raise ValueError("condition replay split must be train or validation")
    if repeat_id < 0:
        raise ValueError("repeat_id must be non-negative")
    if condition_delivery_mode not in {"requested", "masked_canonical"}:
        raise ValueError("unknown condition delivery mode")
    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
        or not git.get("git_available")
    ):
        raise ValueError("condition replay requires a clean SimVerify worktree")

    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable condition replay exists: {destination}")
    bundle = Path(bundle_root).resolve(strict=True)
    m0 = Path(m0_root).resolve(strict=True)
    m2 = Path(m2_root).resolve(strict=True)
    metadata = _read_json(bundle / "run_metadata.json")
    if metadata.get("status") != "completed":
        raise ValueError("condition replay requires completed training")
    baseline_id = str(metadata["experiment_contract"]["baseline_id"])
    if baseline_id not in {"B1", "B1.1", "B2"}:
        raise ValueError("condition replay requires B1, B1.1, or B2")
    if metadata["experiment_contract"]["condition_input"] != (
        "cycle_condition_v1_low_dim"
    ):
        raise ValueError("checkpoint is not cycle_condition_v1 conditioned")
    if metadata["checkpoint_semantics"]["real_control_allowed"] is not False:
        raise ValueError("condition checkpoint lacks real-control prohibition")
    split_key = "train_ids" if split_name == "train" else "val_ids"
    episode_ids = list(map(int, metadata["split"][split_key]))
    if set(episode_ids) & {1, 13, 25, 33}:
        raise ValueError("held-out episode entered condition replay")

    checkpoint = bundle / "policy_best.ckpt"
    checkpoint_contract = _validate_condition_checkpoint(
        checkpoint,
        expected_baseline=baseline_id,
    )
    event_envelope = _read_json(m2 / "expert_event_envelope_v1.json")
    templates = event_envelope["templates"]
    deadzone = list(map(float, event_envelope["effective_deadzone"]))
    annotation_rows = _read_jsonl(m0 / "cycle_annotations.jsonl")
    canonical_condition = _most_frequent_train_condition(annotation_rows)
    annotations = {
        (int(row["episode_id"]), int(row["cycle_id"])): row
        for row in annotation_rows
        if row["quality"]["status"] == "accepted"
        and row["split"] == split_name
        and int(row["episode_id"]) in set(episode_ids)
    }
    anchors = [
        row
        for row in _read_jsonl(m2 / "condition_counterfactual_anchors_v1.jsonl")
        if row["split"] == split_name and int(row["episode_id"]) in set(episode_ids)
    ]
    if not anchors:
        raise ValueError("condition replay selected no anchors")
    supported_keys = {
        (int(row["episode_id"]), int(row["cycle_id"]))
        for row in anchors
        if row["supported"]
    }
    if not supported_keys <= set(annotations):
        raise ValueError("supported anchor has no accepted annotation")
    direction_contract = _fit_sector_action_direction(
        [
            row
            for row in annotation_rows
            if row["quality"]["status"] == "accepted"
            and row["split"] in {"train", "validation"}
        ],
        m0,
        deadzone=deadzone,
    )

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
    rows: list[dict[str, Any]] = []
    base_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    try:
        grouped_anchors: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
        for anchor_index, anchor in enumerate(anchors):
            grouped_anchors.setdefault(
                int(anchor["episode_id"]),
                [],
            ).append((anchor_index, anchor))
        for episode_id, episode_anchors in sorted(grouped_anchors.items()):
            episode_path = m0 / f"episodes/episode_{episode_id}.hdf5"
            with h5py.File(episode_path, "r") as episode:
                for anchor_index, anchor in episode_anchors:
                    key = (episode_id, int(anchor["cycle_id"]))
                    common = {
                        "schema": "simverify_condition_swap_result_v1",
                        "anchor_index": anchor_index,
                        "episode_id": episode_id,
                        "cycle_id": key[1],
                        "split": split_name,
                        "changed_factor": anchor["changed_factors"][0],
                        "base_condition": anchor["base_condition"],
                        "target_condition": anchor["target_condition"],
                        "condition_delivery_mode": condition_delivery_mode,
                        "delivered_base_condition": (
                            anchor["base_condition"]
                            if condition_delivery_mode == "requested"
                            else canonical_condition
                        ),
                        "delivered_target_condition": (
                            anchor["target_condition"]
                            if condition_delivery_mode == "requested"
                            else canonical_condition
                        ),
                        "support_evidence": anchor["support_evidence"],
                        "supported": bool(anchor["supported"]),
                        "included_in_success_denominator": bool(
                            anchor["included_in_success_denominator"]
                        ),
                    }
                    if not anchor["supported"]:
                        rows.append(
                            {
                                **common,
                                "status": "unsupported_counterfactual",
                                "base_trace_path": None,
                                "target_trace_path": None,
                                "metrics": None,
                            }
                        )
                        continue
                    annotation = annotations[key]
                    delivered_base = (
                        anchor["base_condition"]["vector"]
                        if condition_delivery_mode == "requested"
                        else canonical_condition["vector"]
                    )
                    delivered_target = (
                        anchor["target_condition"]["vector"]
                        if condition_delivery_mode == "requested"
                        else canonical_condition["vector"]
                    )
                    if key not in base_cache:
                        base_arrays = replay_cycle_arrays(
                            policy=policy,
                            episode=episode,
                            annotation=annotation,
                            camera_names=CAMERAS,
                            condition_override=delivered_base,
                            pass_condition_to_policy=True,
                        )
                        base_cache[key] = base_arrays
                        relative = Path("base_traces") / (
                            f"episode_{episode_id}_cycle_{key[1]}.npz"
                        )
                        path = temporary / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(path, **base_arrays)
                        identities.append(artifact_identity(path))
                    base_arrays = base_cache[key]
                    target_arrays = replay_cycle_arrays(
                        policy=policy,
                        episode=episode,
                        annotation=annotation,
                        camera_names=CAMERAS,
                        condition_override=delivered_target,
                        pass_condition_to_policy=True,
                    )
                    target_relative = Path("target_traces") / (
                        f"anchor_{anchor_index}_episode_{episode_id}_cycle_{key[1]}.npz"
                    )
                    target_path = temporary / target_relative
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target_path, **target_arrays)
                    identities.append(artifact_identity(target_path))
                    base_relative = Path("base_traces") / (
                        f"episode_{episode_id}_cycle_{key[1]}.npz"
                    )
                    rows.append(
                        {
                            **common,
                            "status": "supported_counterfactual",
                            "base_trace_path": str(base_relative),
                            "target_trace_path": str(target_relative),
                            "metrics": _condition_effect_metrics(
                                base_arrays,
                                target_arrays,
                                annotation=annotation,
                                changed_factor=anchor["changed_factors"][0],
                                base_sector=anchor["base_condition"][
                                    (
                                        "current_sector"
                                        if anchor["changed_factors"][0]
                                        == "current_sector"
                                        else "next_sector"
                                    )
                                ],
                                target_sector=anchor["support_evidence"][
                                    "target_sector"
                                ],
                                direction_contract=direction_contract,
                                deadzone=deadzone,
                                templates=templates,
                            ),
                        }
                    )
        identities.append(write_jsonl(temporary / "condition_swap_metrics.jsonl", rows))
        identities.append(
            write_json(
                temporary / "sector_action_direction_v1.json",
                direction_contract,
            )
        )
        manifest_identity = write_json(
            temporary / "condition_replay_manifest.json",
            {
                "schema": "simverify_condition_replay_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "baseline_id": baseline_id,
                "split": split_name,
                "repeat_id": repeat_id,
                "condition_delivery_mode": condition_delivery_mode,
                "masked_canonical_condition": (
                    canonical_condition
                    if condition_delivery_mode == "masked_canonical"
                    else None
                ),
                "requested_condition_diff_delivered_to_policy": (
                    condition_delivery_mode == "requested"
                ),
                "episode_ids": episode_ids,
                "anchor_count": len(rows),
                "supported_anchor_count": sum(bool(row["supported"]) for row in rows),
                "unsupported_anchor_count": sum(
                    not bool(row["supported"]) for row in rows
                ),
                "condition_input_used_by_policy": True,
                "observation_history_changed": False,
                "one_primary_factor_per_anchor": True,
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                    "embedded_contract": checkpoint_contract,
                },
                "dataset_stats_sha256": sha256_file(bundle / "dataset_stats.pkl"),
                "resolved_config_sha256": sha256_file(bundle / "resolved_config.yaml"),
                "m0_dataset_manifest_sha256": sha256_file(m0 / "dataset_manifest.json"),
                "m2_manifest_sha256": sha256_file(m2 / "m2_manifest.json"),
                "anchor_registry_sha256": sha256_file(
                    m2 / "condition_counterfactual_anchors_v1.jsonl"
                ),
                "evidence_scope": "recorded-observation/offline",
                "held_out_test_read": False,
                "gate_thresholds_v1_generated": False,
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
            "baseline_id": baseline_id,
            "split": split_name,
            "repeat_id": repeat_id,
            "condition_delivery_mode": condition_delivery_mode,
            "anchor_count": len(rows),
            "supported_anchor_count": sum(bool(row["supported"]) for row in rows),
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
                    "schema": "simverify_condition_replay_failure_v1",
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                    "git": git,
                },
            )
        raise


def _condition_effect_metrics(
    base: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    *,
    annotation: Mapping[str, Any],
    changed_factor: str,
    base_sector: str,
    target_sector: str,
    direction_contract: Mapping[str, Any],
    deadzone: Sequence[float],
    templates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    start, end = map(int, annotation["target_steps_20hz"])
    if changed_factor == "current_sector":
        window_start = 0
        window_end = (
            int(
                annotation["observable_events"]["carry_transition_proxy"][
                    "representative_target_tick"
                ]
            )
            - start
            + 1
        )
    elif changed_factor == "next_sector":
        window_start = (
            int(
                annotation["observable_events"]["dump_end_proxy"][
                    "representative_target_tick"
                ]
            )
            - start
        )
        window_end = end - start + 1
    else:
        raise ValueError("condition replay changes an unknown factor")
    base_action = _effective_action(
        base["temporal_aggregation_action"],
        deadzone,
    )
    target_action = _effective_action(
        target["temporal_aggregation_action"],
        deadzone,
    )
    delta = target_action - base_action
    relevant = delta[max(0, window_start) : min(delta.shape[0], window_end)]
    if relevant.shape[0] == 0:
        raise ValueError("condition response window is empty")
    swing_delta = float(np.mean(relevant[:, 0]))
    centers = direction_contract["sector_swing_qpos_median"]
    target_qpos_direction = int(
        np.sign(float(centers[target_sector]) - float(centers[base_sector]))
    )
    expected_action_sign = int(
        target_qpos_direction * int(direction_contract["action_to_qpos_direction_sign"])
    )
    observed_sign = int(np.sign(swing_delta))
    base_task = cycle_action_metrics(
        base,
        templates=templates,
        deadzone=deadzone,
    )
    target_task = cycle_action_metrics(
        target,
        templates=templates,
        deadzone=deadzone,
    )
    return {
        "schema": "simverify_condition_effect_metrics_v1",
        "relevant_window_local": [
            max(0, window_start),
            min(delta.shape[0], window_end),
        ],
        "token_swap_action_effect": float(np.mean(np.abs(relevant))),
        "swing_action_delta_mean": swing_delta,
        "non_target_axis_delta_mean_abs": float(np.mean(np.abs(relevant[:, 1:]))),
        "expected_swing_action_sign": expected_action_sign,
        "observed_swing_delta_sign": observed_sign,
        "token_swap_direction_correct": bool(
            expected_action_sign != 0 and observed_sign == expected_action_sign
        ),
        "per_tick_effect_l1": np.mean(np.abs(delta), axis=1).tolist(),
        "base_task_metrics": base_task,
        "target_task_metrics": target_task,
        "event_coverage_delta": float(
            target_task["required_event_coverage"]
            - base_task["required_event_coverage"]
        ),
        "target_event_order_valid": bool(target_task["event_order_valid"]),
        "closed_loop_execution": False,
    }


def _most_frequent_train_condition(
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    accepted = [
        row
        for row in annotations
        if row["split"] == "train" and row["quality"]["status"] == "accepted"
    ]
    if not accepted:
        raise ValueError("canonical condition fit has no accepted train cycles")
    keys = [
        (
            str(row["policy_condition"]["current_sector"]),
            str(row["policy_condition"]["next_ready_sector"]),
            tuple(map(float, row["policy_condition"]["vector"])),
        )
        for row in accepted
    ]
    counts = Counter(keys)
    selected, count = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]
    return {
        "schema": "simverify_masked_canonical_condition_v1",
        "current_sector": selected[0],
        "next_sector": selected[1],
        "vector": list(selected[2]),
        "train_accepted_cycle_count": len(accepted),
        "selected_count": int(count),
        "selection": ("most_frequent_accepted_train_condition_lexical_tie_break"),
    }


def _fit_sector_action_direction(
    annotations: Sequence[Mapping[str, Any]],
    m0_root: Path,
    *,
    deadzone: Sequence[float],
) -> dict[str, Any]:
    qpos_by_sector: dict[str, list[float]] = {sector: [] for sector in SECTORS}
    episode_ids: set[int] = set()
    for row in annotations:
        episode_ids.add(int(row["episode_id"]))
        current = row["numeric_sector_evidence"]["current_swing_qpos"]
        next_value = row["numeric_sector_evidence"]["next_swing_qpos"]
        if current is not None:
            qpos_by_sector[row["policy_condition"]["current_sector"]].append(
                float(current)
            )
        if next_value is not None:
            qpos_by_sector[row["policy_condition"]["next_ready_sector"]].append(
                float(next_value)
            )
    centers = {
        sector: float(np.median(values))
        for sector, values in qpos_by_sector.items()
        if values
    }
    if set(centers) != set(SECTORS):
        raise ValueError("sector direction fit lacks a sector center")
    products: list[float] = []
    for episode_id in sorted(episode_ids):
        with h5py.File(
            m0_root / f"episodes/episode_{episode_id}.hdf5",
            "r",
        ) as episode:
            action = np.asarray(episode["action"][:, 0], dtype=np.float64)
            qvel = np.asarray(
                episode["observations/qvel"][:, 0],
                dtype=np.float64,
            )
            mask = (np.abs(action) > float(deadzone[0])) & (np.abs(qvel) > 1e-6)
            products.extend((action[mask] * qvel[mask]).tolist())
    if not products:
        raise ValueError("cannot fit action-to-qpos direction")
    direction_sign = int(np.sign(np.median(products)))
    if direction_sign == 0:
        raise ValueError("action-to-qpos direction is unidentified")
    return {
        "schema": "simverify_sector_action_direction_v1",
        "fit_splits": ["train", "validation"],
        "sector_swing_qpos_median": centers,
        "sector_sample_count": {
            sector: len(values) for sector, values in qpos_by_sector.items()
        },
        "sector_order_by_qpos": sorted(centers, key=centers.get),
        "action_to_qpos_direction_sign": direction_sign,
        "action_qvel_product_count": len(products),
        "fit_uses_privilege": False,
    }


def _effective_action(
    action: np.ndarray,
    deadzone: Sequence[float],
) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    threshold = np.asarray(deadzone, dtype=np.float32).reshape(1, 4)
    return np.where(np.abs(values) > threshold, values, 0.0).astype(np.float32)


def _validate_condition_checkpoint(
    checkpoint: Path,
    *,
    expected_baseline: str,
) -> dict[str, Any]:
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("condition checkpoint lacks embedded config")
    experiment = config.get("experiment_contract")
    semantics = config.get("checkpoint_semantics")
    if not isinstance(experiment, Mapping) or not isinstance(
        semantics,
        Mapping,
    ):
        raise ValueError("condition checkpoint lacks embedded contracts")
    if (
        experiment.get("baseline_id") != expected_baseline
        or experiment.get("condition_input") != "cycle_condition_v1_low_dim"
    ):
        raise ValueError("condition checkpoint baseline/input mismatch")
    if (
        semantics.get("domain") != "sim"
        or semantics.get("real_control_allowed") is not False
        or semantics.get("jetson_allowed") is not False
    ):
        raise ValueError("condition checkpoint deployment prohibition invalid")
    return {
        "experiment_contract": dict(experiment),
        "checkpoint_semantics": dict(semantics),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
