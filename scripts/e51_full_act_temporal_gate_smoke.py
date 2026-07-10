#!/usr/bin/env python3
"""Run full ACT image inference through the causal temporal direction gate stack."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml

from scripts.cache_act_intent_probs import _image_tensor_for_policy
from scripts.e35_runtime_gate_smoke import (
    active_mask_from_gohome_gate,
    json_safe,
    load_gate_model,
    parse_gohome_gate_name,
    parse_phase_gate_name,
    predict_gate_probabilities,
)
from scripts.e36_build_policy_gate_package_manifest import verify_manifest
from scripts.e37_full_act_gate_smoke import (
    artifact_path_by_name,
    predict_action_and_intent_query0,
    read_train_ready_episode_ids_compatible,
    select_episode_ids,
)
from scripts.e41_intent_targeted_snap_probe import snap_actions_near_deadzone_with_intent
from scripts.e43_direction_gate_probe import FEATURE_NAMES, apply_direction_probability_gate
from testbed.data.dataset import _read_camera_image
from testbed.data.image_transforms import IMAGE_TRANSFORM_CHOICES, build_image_transform
from testbed.policies.deadzone_eval import AXIS_NAMES, load_deadzone_thresholds
from testbed.policies.gohome_eligibility import (
    aggregate_gohome_event_rows,
    gohome_event_metrics_from_active_mask,
)
from testbed.policies.offline_eval import (
    aggregate_episode_results,
    compute_action_metrics,
    episode_path,
    load_policy_for_episode,
    normalize_episode_id,
    write_collection_report,
)
from testbed.policies.phase_gate import apply_phase_gate_to_actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--temporal-direction-model", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--handoff-dataset-dir", type=Path, required=True)
    parser.add_argument("--train-ready-manifest", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--phase-gate-name", default=None)
    parser.add_argument("--gohome-gate-name", default=None)
    parser.add_argument("--direction-threshold", type=float, default=0.50)
    parser.add_argument("--direction-inactive-scale", type=float, default=0.75)
    parser.add_argument("--snap-margin", type=float, default=0.020)
    parser.add_argument("--snap-intent-threshold", type=float, default=0.70)
    parser.add_argument("--snap-epsilon", type=float, default=0.001)
    parser.add_argument("--image-transform", choices=IMAGE_TRANSFORM_CHOICES, default="none")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    package_manifest = _read_json(args.package_manifest)
    verify_report = verify_manifest(package_manifest)
    if not verify_report["ok"]:
        raise SystemExit(f"package manifest verification failed: {verify_report['errors']}")

    phase_gate_name = str(args.phase_gate_name or package_manifest["selected_gates"]["phase_gate"])
    gohome_gate_name = str(args.gohome_gate_name or package_manifest["selected_gates"]["gohome_gate"])
    phase_gate = parse_phase_gate_name(phase_gate_name)
    gohome_gate = parse_gohome_gate_name(gohome_gate_name)

    action_ckpt = Path(artifact_path_by_name(package_manifest, "action_policy_best"))
    stats_path = Path(artifact_path_by_name(package_manifest, "action_dataset_stats"))
    resolved_config = Path(artifact_path_by_name(package_manifest, "action_resolved_config"))
    phase_model = Path(artifact_path_by_name(package_manifest, "phase_gate_model"))
    tail_model = Path(artifact_path_by_name(package_manifest, "tail_candidate_model"))
    eligibility_model = Path(artifact_path_by_name(package_manifest, "gohome_eligibility_model"))

    resolved = _read_yaml(resolved_config)
    task_cfg = dict(resolved.get("task", {}) or {})
    camera_names = [str(cam) for cam in task_cfg.get("camera_names", ["fpv"])]
    available_ids = read_train_ready_episode_ids_compatible(args.train_ready_manifest)
    episode_ids = select_episode_ids(
        available=available_ids,
        requested=[normalize_episode_id(item) for item in args.episode_id],
        max_episodes=int(args.max_episodes),
    )
    episode_paths = [episode_path(args.dataset_dir, episode_id) for episode_id in episode_ids]
    missing = [path for path in episode_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    max_episode_len = max(_episode_len(path) for path in episode_paths)

    policy = load_policy_for_episode(
        bundle_dir=action_ckpt.parent,
        ckpt_path=action_ckpt,
        resolved_config_path=resolved_config,
        stats_path=stats_path,
        max_episode_len=max_episode_len,
        temporal_agg=True,
        device=args.device,
    )
    phase_bundle = load_gate_model(phase_model)
    tail_bundle = load_gate_model(tail_model)
    eligibility_bundle = load_gate_model(eligibility_model)
    temporal_direction_bundle = load_temporal_direction_model(args.temporal_direction_model)
    image_transform = build_image_transform(str(args.image_transform))
    thresholds = load_deadzone_thresholds(args.deadzone_json)
    pos_thresholds = np.asarray([thresholds[axis]["pos"] for axis in AXIS_NAMES], dtype=np.float32)
    neg_thresholds = np.asarray([thresholds[axis]["neg"] for axis in AXIS_NAMES], dtype=np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_groups: dict[str, list[dict[str, Any]]] = {
        "raw_action": [],
        "phase_gated_action": [],
        "snap_action": [],
        "temporal_direction_action": [],
    }
    gohome_rows = []
    act_latencies_ms: list[float] = []
    gate_latencies_ms: list[float] = []
    for index, episode_id in enumerate(episode_ids, start=1):
        print(f"[{index}/{len(episode_ids)}] E51 full ACT temporal gate smoke {episode_id}")
        result = evaluate_full_act_temporal_gate_episode(
            policy=policy,
            episode_id=episode_id,
            episode_file=episode_path(args.dataset_dir, episode_id),
            handoff_episode_file=episode_path(args.handoff_dataset_dir, episode_id),
            camera_names=camera_names,
            image_transform=image_transform,
            phase_bundle=phase_bundle,
            tail_bundle=tail_bundle,
            eligibility_bundle=eligibility_bundle,
            temporal_direction_bundle=temporal_direction_bundle,
            phase_gate=phase_gate,
            gohome_gate=gohome_gate,
            gohome_gate_name=gohome_gate_name,
            pos_thresholds=pos_thresholds,
            neg_thresholds=neg_thresholds,
            direction_threshold=float(args.direction_threshold),
            direction_inactive_scale=float(args.direction_inactive_scale),
            snap_margin=float(args.snap_margin),
            snap_intent_threshold=float(args.snap_intent_threshold),
            snap_epsilon=float(args.snap_epsilon),
            progress_every=max(0, int(args.progress_every)),
        )
        act_latencies_ms.extend(result["act_latencies_ms"])
        gate_latencies_ms.extend(result["gate_latencies_ms"])
        gohome_rows.append(result["gohome_row"])
        for key in result_groups:
            result_groups[key].append(_episode_result_for_aggregate(episode_id, result, key))
        _write_episode_outputs(output_dir, episode_id, result)

    aggregates: dict[str, dict[str, Any]] = {}
    replay_dirs = {
        "raw_action": "raw_action_replay",
        "phase_gated_action": "phase_gated_action_replay",
        "snap_action": "snap_action_replay",
        "temporal_direction_action": "temporal_direction_action_replay",
    }
    for key, rows in result_groups.items():
        aggregate = aggregate_episode_results(rows)
        aggregates[key] = aggregate
        write_collection_report(
            aggregate=aggregate,
            output_dir=output_dir / replay_dirs[key],
            metadata={
                "selection_mode": "e51_full_act_temporal_gate_smoke",
                "episode_ids": episode_ids,
                "action_key": key,
            },
        )

    gohome_summary = aggregate_gohome_event_rows(gohome_rows)
    _write_csv(output_dir / "full_act_gohome_events.csv", gohome_rows)
    latency_summary = {
        "steps": len(act_latencies_ms),
        "act_mean_ms": _mean(act_latencies_ms),
        "act_p50_ms": _percentile(act_latencies_ms, 50),
        "act_p95_ms": _percentile(act_latencies_ms, 95),
        "act_max_ms": max(act_latencies_ms) if act_latencies_ms else 0.0,
        "gate_mean_ms": _mean(gate_latencies_ms),
        "gate_p50_ms": _percentile(gate_latencies_ms, 50),
        "gate_p95_ms": _percentile(gate_latencies_ms, 95),
        "gate_max_ms": max(gate_latencies_ms) if gate_latencies_ms else 0.0,
    }
    compact = build_compact_summary(
        candidate_id="E51",
        episode_ids=episode_ids,
        aggregates=aggregates,
        gohome_summary=gohome_summary,
        latency_summary=latency_summary,
        artifact_manifest=str(args.package_manifest),
    )
    summary = {
        **compact,
        "package_verify": verify_report,
        "selected_gates": {
            "phase_gate": phase_gate_name,
            "gohome_gate": gohome_gate_name,
            "direction_threshold": float(args.direction_threshold),
            "direction_inactive_scale": float(args.direction_inactive_scale),
            "snap_margin": float(args.snap_margin),
            "snap_intent_threshold": float(args.snap_intent_threshold),
        },
        "camera_names": camera_names,
        "image_transform": str(args.image_transform),
        "dataset_dir": str(args.dataset_dir),
        "handoff_dataset_dir": str(args.handoff_dataset_dir),
        "train_ready_manifest": str(args.train_ready_manifest),
        "temporal_direction_model": str(args.temporal_direction_model),
        "temporal_context_offsets": temporal_direction_bundle.offsets,
        "latency_summary": latency_summary,
        "gohome_event_summary": gohome_summary,
    }
    _write_json(output_dir / "full_act_temporal_gate_smoke_summary.json", json_safe(summary))
    print(f"E51 full ACT temporal gate smoke summary: {output_dir / 'full_act_temporal_gate_smoke_summary.json'}")
    print(f"Gohome events: {output_dir / 'full_act_gohome_events.csv'}")


def build_causal_temporal_step_features(
    features: np.ndarray,
    feature_names: list[str],
    *,
    step: int,
    offsets: list[int],
) -> tuple[np.ndarray, list[str]]:
    base = np.asarray(features, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {base.shape}")
    if base.shape[1] != len(feature_names):
        raise ValueError(f"feature name count {len(feature_names)} does not match feature dim {base.shape[1]}")
    if not offsets:
        raise ValueError("offsets must not be empty")
    if any(int(offset) > 0 for offset in offsets):
        raise ValueError("future offsets are not allowed in causal temporal runtime features")
    if step < 0 or step >= base.shape[0]:
        raise IndexError(f"step {step} outside feature range 0..{base.shape[0] - 1}")
    chunks = []
    names = []
    for offset in offsets:
        source_index = max(0, int(step) + int(offset))
        chunks.append(base[source_index : source_index + 1])
        suffix = f"t{int(offset):+d}" if int(offset) else "t0"
        names.extend([f"{name}_{suffix}" for name in feature_names])
    return np.concatenate(chunks, axis=1).astype(np.float32), names


@dataclass(frozen=True)
class TemporalDirectionBundle:
    model: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]
    offsets: list[int]


def load_temporal_direction_model(path: Path) -> TemporalDirectionBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    feature_names, offsets = resolve_temporal_feature_names(path, payload)
    if any(offset > 0 for offset in offsets):
        raise ValueError(f"temporal direction model is non-causal; future offsets present: {offsets}")
    model = _DirectionGateMlp(input_dim=len(feature_names), hidden_dim=int(payload["hidden_dim"]), output_dim=8)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return TemporalDirectionBundle(
        model=model,
        mean=np.asarray(payload["feature_mean"], dtype=np.float32),
        std=np.asarray(payload["feature_std"], dtype=np.float32),
        feature_names=feature_names,
        offsets=offsets,
    )


def resolve_temporal_feature_names(model_path: Path, payload: dict[str, Any]) -> tuple[list[str], list[int]]:
    feature_names = list(payload.get("feature_names", []))
    feature_dim = int(np.asarray(payload["feature_mean"], dtype=np.float32).shape[0])
    if len(feature_names) == feature_dim and all("_t" in name for name in feature_names):
        offsets = _offsets_from_feature_names(feature_names)
        return feature_names, offsets

    metadata_path = Path(model_path).with_name("temporal_direction_gate_model_metadata.json")
    if not metadata_path.exists():
        raise ValueError(
            f"temporal feature_names are stale or missing and metadata file is absent: {metadata_path}"
        )
    metadata = _read_json(metadata_path)
    metadata_names = list(metadata.get("feature_names", []))
    if len(metadata_names) != feature_dim:
        raise ValueError(f"metadata feature dim {len(metadata_names)} does not match model feature dim {feature_dim}")
    offsets = [int(value) for value in metadata.get("context_offsets", _offsets_from_feature_names(metadata_names))]
    return metadata_names, offsets


def predict_temporal_direction_probabilities(bundle: TemporalDirectionBundle, features: np.ndarray) -> np.ndarray:
    base = np.asarray(features, dtype=np.float32)
    rows = []
    last_names: list[str] | None = None
    for step in range(base.shape[0]):
        row, names = build_causal_temporal_step_features(base, FEATURE_NAMES, step=step, offsets=bundle.offsets)
        rows.append(row)
        last_names = names
    if last_names != bundle.feature_names:
        raise ValueError("temporal feature names do not match model contract")
    x = np.concatenate(rows, axis=0)
    with torch.no_grad():
        logits = bundle.model(torch.as_tensor((x - bundle.mean) / bundle.std, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


def evaluate_full_act_temporal_gate_episode(
    *,
    policy: Any,
    episode_id: str,
    episode_file: Path,
    handoff_episode_file: Path,
    camera_names: list[str],
    image_transform: Any,
    phase_bundle: Any,
    tail_bundle: Any,
    eligibility_bundle: Any,
    temporal_direction_bundle: TemporalDirectionBundle,
    phase_gate: dict[str, Any],
    gohome_gate: dict[str, Any],
    gohome_gate_name: str,
    pos_thresholds: np.ndarray,
    neg_thresholds: np.ndarray,
    direction_threshold: float,
    direction_inactive_scale: float,
    snap_margin: float,
    snap_intent_threshold: float,
    snap_epsilon: float,
    progress_every: int,
) -> dict[str, Any]:
    if not handoff_episode_file.exists():
        raise FileNotFoundError(handoff_episode_file)
    with h5py.File(episode_file, "r") as f:
        qpos = np.asarray(f["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(f["observations/qvel"][()], dtype=np.float32)
        expert = np.asarray(f["action"][()], dtype=np.float32)
        n = int(min(qpos.shape[0], qvel.shape[0], expert.shape[0]))
        raw_action = np.zeros((n, len(AXIS_NAMES)), dtype=np.float32)
        intent_prob = np.zeros((n, len(AXIS_NAMES) * 2), dtype=np.float32)
        act_latencies_ms: list[float] = []
        if hasattr(policy, "reset"):
            policy.reset()
        for step in range(n):
            obs: dict[str, Any] = {"qpos": qpos[step], "qvel": qvel[step]}
            for camera_name in camera_names:
                image = _read_camera_image(f, camera_name, step)
                if image_transform is not None:
                    image = image_transform(image)
                obs[f"image_{camera_name}"] = image
            t0 = time.perf_counter()
            action, prob = predict_action_and_intent_query0(policy, obs)
            act_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            raw_action[step] = action
            intent_prob[step] = prob
            if progress_every > 0 and (step + 1) % progress_every == 0:
                print(f"  {episode_id}: full ACT replayed {step + 1}/{n} steps")
    features = np.concatenate([intent_prob, qpos[:n], qvel[:n]], axis=1).astype(np.float32)
    t_gate = time.perf_counter()
    phase_prob = predict_gate_probabilities(phase_bundle, features)
    candidate_prob = predict_gate_probabilities(tail_bundle, features)
    eligibility_prob = predict_gate_probabilities(eligibility_bundle, features)
    direction_prob = predict_temporal_direction_probabilities(temporal_direction_bundle, features)
    gate_ms_total = (time.perf_counter() - t_gate) * 1000.0
    gate_latencies_ms = [gate_ms_total / max(1, n)] * n
    phase_active = phase_prob >= float(phase_gate["threshold"])
    phase_gated_action = apply_phase_gate_to_actions(
        raw_action,
        phase_active,
        inactive_scale=float(phase_gate["inactive_scale"]),
    )
    snap_action = snap_actions_near_deadzone_with_intent(
        phase_gated_action,
        phase_active,
        intent_prob,
        pos_thresholds,
        neg_thresholds,
        margin=float(snap_margin),
        epsilon=float(snap_epsilon),
        intent_threshold=float(snap_intent_threshold),
    )
    temporal_direction_action = apply_direction_probability_gate(
        snap_action,
        direction_prob,
        threshold=float(direction_threshold),
        inactive_scale=float(direction_inactive_scale),
    )
    gohome_active = active_mask_from_gohome_gate(candidate_prob, eligibility_prob, gohome_gate)
    with h5py.File(handoff_episode_file, "r") as f:
        label = np.asarray(f["handoff/gohome_eligible_label"][()], dtype=bool)
        loss_mask = np.asarray(f["handoff/gohome_loss_mask"][()], dtype=bool)
        tail_idle = np.asarray(f["handoff/tail_idle_mask"][()], dtype=bool)
    m = int(min(n, label.shape[0], loss_mask.shape[0], tail_idle.shape[0]))
    gohome_row = gohome_event_metrics_from_active_mask(
        episode_id=episode_id,
        active_mask=gohome_active[:m],
        eligible_label=label[:m],
        loss_mask=loss_mask[:m],
        tail_idle_mask=tail_idle[:m],
        gate=gohome_gate_name,
    )
    return {
        "episode_id": episode_id,
        "expert_action": expert[:n],
        "raw_action": raw_action,
        "phase_gated_action": phase_gated_action,
        "snap_action": snap_action,
        "temporal_direction_action": temporal_direction_action,
        "intent_prob": intent_prob,
        "phase_prob": phase_prob,
        "tail_candidate_prob": candidate_prob,
        "gohome_eligibility_prob": eligibility_prob,
        "direction_prob": direction_prob,
        "gohome_active": gohome_active,
        "gohome_row": gohome_row,
        "act_latencies_ms": act_latencies_ms,
        "gate_latencies_ms": gate_latencies_ms,
    }


def build_compact_summary(
    *,
    candidate_id: str,
    episode_ids: list[str],
    aggregates: dict[str, dict[str, Any]],
    gohome_summary: dict[str, Any],
    latency_summary: dict[str, Any],
    artifact_manifest: str,
) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate_id),
        "artifact_manifest": str(artifact_manifest),
        "episode_ids": list(episode_ids),
        "episodes": len(episode_ids),
        "raw_action_mae": _metric(aggregates["raw_action"]["global_metrics"], "mae"),
        "phase_gated_action_mae": _metric(aggregates["phase_gated_action"]["global_metrics"], "mae"),
        "snap_action_mae": _metric(aggregates["snap_action"]["global_metrics"], "mae"),
        "temporal_direction_action_mae": _metric(
            aggregates["temporal_direction_action"]["global_metrics"],
            "mae",
        ),
        "temporal_direction_action_rmse": _metric(
            aggregates["temporal_direction_action"]["global_metrics"],
            "rmse",
        ),
        "gohome_event_recall": gohome_summary.get("event_recall", ""),
        "gohome_pre_tail_false_positive_episodes": gohome_summary.get(
            "pre_tail_false_positive_episodes",
            "",
        ),
        "gohome_pre_tail_active_frames": gohome_summary.get("pre_tail_active_frames", ""),
        "act_p95_ms": latency_summary.get("act_p95_ms", ""),
        "gate_p95_ms": latency_summary.get("gate_p95_ms", ""),
    }


def _episode_result_for_aggregate(episode_id: str, result: dict[str, Any], key: str) -> dict[str, Any]:
    expert = np.asarray(result["expert_action"], dtype=np.float32)
    policy = np.asarray(result[key], dtype=np.float32)
    return {
        "episode_id": episode_id,
        "episode_path": episode_id,
        "n_steps": int(expert.shape[0]),
        "dt": 0.05,
        "expert_action": expert,
        "policy_action": policy,
        "metrics": compute_action_metrics(expert, policy),
    }


def _write_episode_outputs(output_dir: Path, episode_id: str, result: dict[str, Any]) -> None:
    for replay_name, key in (
        ("raw_action_replay", "raw_action"),
        ("phase_gated_action_replay", "phase_gated_action"),
        ("snap_action_replay", "snap_action"),
        ("temporal_direction_action_replay", "temporal_direction_action"),
    ):
        episode_dir = output_dir / replay_name / "episodes" / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        steps = np.arange(np.asarray(result["expert_action"]).shape[0], dtype=np.float64)
        np.savez_compressed(
            episode_dir / "actions.npz",
            time_s=steps * 0.05,
            expert_action=np.asarray(result["expert_action"], dtype=np.float32),
            policy_action=np.asarray(result[key], dtype=np.float32),
        )
    prob_dir = output_dir / "episode_gate_probs"
    prob_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prob_dir / f"{episode_id}.npz",
        intent_prob=np.asarray(result["intent_prob"], dtype=np.float32),
        phase_prob=np.asarray(result["phase_prob"], dtype=np.float32),
        tail_candidate_prob=np.asarray(result["tail_candidate_prob"], dtype=np.float32),
        gohome_eligibility_prob=np.asarray(result["gohome_eligibility_prob"], dtype=np.float32),
        direction_prob=np.asarray(result["direction_prob"], dtype=np.float32),
        gohome_active=np.asarray(result["gohome_active"], dtype=bool),
    )


def _offsets_from_feature_names(feature_names: list[str]) -> list[int]:
    offsets: list[int] = []
    for name in feature_names:
        suffix = name.rsplit("_t", 1)[-1]
        offset = int(suffix)
        if offset not in offsets:
            offsets.append(offset)
    return offsets


def _metric(metrics: dict[str, Any], key: str) -> float | str:
    value = dict(metrics.get("overall", {})).get(key, "")
    return "" if value == "" else float(value)


def _episode_len(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(min(f["observations/qpos"].shape[0], f["observations/qvel"].shape[0], f["action"].shape[0]))


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class _DirectionGateMlp(torch.nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    main()
