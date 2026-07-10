#!/usr/bin/env python3
"""Run full ACT image inference through the packaged action and gate stack."""

from __future__ import annotations

import argparse
import json
import time
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
from testbed.data.dataset import _read_camera_image
from testbed.data.image_transforms import IMAGE_TRANSFORM_CHOICES, build_image_transform
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
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--handoff-dataset-dir", type=Path, required=True)
    parser.add_argument("--train-ready-manifest", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--phase-gate-name", default=None)
    parser.add_argument("--gohome-gate-name", default=None)
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
    image_transform = build_image_transform(str(args.image_transform))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_results = []
    gated_results = []
    gohome_rows = []
    act_latencies_ms: list[float] = []
    gate_latencies_ms: list[float] = []
    for index, episode_id in enumerate(episode_ids, start=1):
        print(f"[{index}/{len(episode_ids)}] full ACT gate smoke {episode_id}")
        result = evaluate_full_act_gate_episode(
            policy=policy,
            episode_id=episode_id,
            episode_file=episode_path(args.dataset_dir, episode_id),
            handoff_episode_file=episode_path(args.handoff_dataset_dir, episode_id),
            camera_names=camera_names,
            image_transform=image_transform,
            phase_bundle=phase_bundle,
            tail_bundle=tail_bundle,
            eligibility_bundle=eligibility_bundle,
            phase_gate=phase_gate,
            gohome_gate=gohome_gate,
            gohome_gate_name=gohome_gate_name,
            progress_every=max(0, int(args.progress_every)),
        )
        act_latencies_ms.extend(result["act_latencies_ms"])
        gate_latencies_ms.extend(result["gate_latencies_ms"])
        gohome_rows.append(result["gohome_row"])
        raw_results.append(_episode_result_for_aggregate(episode_id, result, "raw_policy_action"))
        gated_results.append(_episode_result_for_aggregate(episode_id, result, "phase_gated_action"))
        _write_episode_outputs(output_dir, episode_id, result)

    raw_aggregate = aggregate_episode_results(raw_results)
    gated_aggregate = aggregate_episode_results(gated_results)
    write_collection_report(
        aggregate=raw_aggregate,
        output_dir=output_dir / "raw_action_replay",
        metadata={"selection_mode": "e37_full_act_raw", "episode_ids": episode_ids},
    )
    write_collection_report(
        aggregate=gated_aggregate,
        output_dir=output_dir / "phase_gated_action_replay",
        metadata={
            "selection_mode": "e37_full_act_phase_gated",
            "episode_ids": episode_ids,
            "phase_gate_name": phase_gate_name,
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
        candidate_id="E37",
        episode_ids=episode_ids,
        raw_metrics=raw_aggregate["global_metrics"],
        gated_metrics=gated_aggregate["global_metrics"],
        gohome_summary=gohome_summary,
        latency_summary=latency_summary,
        artifact_manifest=str(args.package_manifest),
    )
    summary = {
        **compact,
        "package_verify": verify_report,
        "selected_gates": {"phase_gate": phase_gate_name, "gohome_gate": gohome_gate_name},
        "camera_names": camera_names,
        "image_transform": str(args.image_transform),
        "dataset_dir": str(args.dataset_dir),
        "handoff_dataset_dir": str(args.handoff_dataset_dir),
        "train_ready_manifest": str(args.train_ready_manifest),
        "latency_summary": latency_summary,
        "gohome_event_summary": gohome_summary,
    }
    _write_json(output_dir / "full_act_gate_smoke_summary.json", json_safe(summary))
    print(f"Full ACT gate smoke summary: {output_dir / 'full_act_gate_smoke_summary.json'}")
    print(f"Gohome events: {output_dir / 'full_act_gohome_events.csv'}")


def artifact_path_by_name(manifest: dict[str, Any], name: str) -> str:
    for artifact in list(manifest.get("artifacts", [])):
        if str(artifact.get("name", "")) == str(name):
            return str(artifact.get("path", ""))
    raise KeyError(f"missing artifact {name}")


def select_episode_ids(*, available: list[str], requested: list[str], max_episodes: int) -> list[str]:
    if requested:
        missing = [episode_id for episode_id in requested if episode_id not in set(available)]
        if missing:
            raise ValueError(f"requested episodes are not train-ready: {missing}")
        return list(requested)
    limit = max(1, int(max_episodes))
    return list(available[:limit])


def build_compact_summary(
    *,
    candidate_id: str,
    episode_ids: list[str],
    raw_metrics: dict[str, Any],
    gated_metrics: dict[str, Any],
    gohome_summary: dict[str, Any],
    latency_summary: dict[str, Any],
    artifact_manifest: str,
) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate_id),
        "artifact_manifest": str(artifact_manifest),
        "episode_ids": list(episode_ids),
        "episodes": len(episode_ids),
        "raw_action_mae": _metric(raw_metrics, "mae"),
        "raw_action_rmse": _metric(raw_metrics, "rmse"),
        "phase_gated_action_mae": _metric(gated_metrics, "mae"),
        "phase_gated_action_rmse": _metric(gated_metrics, "rmse"),
        "gohome_event_recall": gohome_summary.get("event_recall", ""),
        "gohome_pre_tail_false_positive_episodes": gohome_summary.get(
            "pre_tail_false_positive_episodes",
            "",
        ),
        "gohome_pre_tail_active_frames": gohome_summary.get("pre_tail_active_frames", ""),
        "act_p95_ms": latency_summary.get("act_p95_ms", ""),
        "gate_p95_ms": latency_summary.get("gate_p95_ms", ""),
    }


def read_train_ready_episode_ids_compatible(path: Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [normalize_episode_id(item) for item in payload]
    if "train_ready_episode_ids" in payload:
        return [normalize_episode_id(item) for item in payload["train_ready_episode_ids"]]
    strict = list(payload.get("strict_pass_episode_ids", []))
    warn = list(payload.get("warn_episode_ids", []))
    ids = strict + [item for item in warn if item not in set(strict)]
    if not ids:
        raise ValueError(f"manifest has no train-ready ids: {path}")
    return [normalize_episode_id(item) for item in ids]


def evaluate_full_act_gate_episode(
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
    phase_gate: dict[str, Any],
    gohome_gate: dict[str, Any],
    gohome_gate_name: str,
    progress_every: int,
) -> dict[str, Any]:
    if not handoff_episode_file.exists():
        raise FileNotFoundError(handoff_episode_file)
    with h5py.File(episode_file, "r") as f:
        qpos = np.asarray(f["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(f["observations/qvel"][()], dtype=np.float32)
        expert = np.asarray(f["action"][()], dtype=np.float32)
        n = int(min(qpos.shape[0], qvel.shape[0], expert.shape[0]))
        raw_action = np.zeros((n, 4), dtype=np.float32)
        intent_prob = np.zeros((n, 8), dtype=np.float32)
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
    gate_ms_total = (time.perf_counter() - t_gate) * 1000.0
    gate_latencies_ms = [gate_ms_total / max(1, n)] * n
    phase_active = phase_prob >= float(phase_gate["threshold"])
    phase_gated_action = apply_phase_gate_to_actions(
        raw_action,
        phase_active,
        inactive_scale=float(phase_gate["inactive_scale"]),
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
        "raw_policy_action": raw_action,
        "phase_gated_action": phase_gated_action,
        "intent_prob": intent_prob,
        "phase_prob": phase_prob,
        "tail_candidate_prob": candidate_prob,
        "gohome_eligibility_prob": eligibility_prob,
        "gohome_active": gohome_active,
        "gohome_row": gohome_row,
        "act_latencies_ms": act_latencies_ms,
        "gate_latencies_ms": gate_latencies_ms,
    }


def predict_action_and_intent_query0(policy: Any, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    proprio = policy._build_proprio(obs)
    proprio = (proprio - policy._proprio_mean) / policy._proprio_std
    image = _image_tensor_for_policy(policy, obs)
    if policy._model.training:
        policy._model.eval()
    with torch.inference_mode():
        a_hat, _, _, intent_logits = policy._unpack_model_output(policy._model(proprio, image, None))
    if intent_logits is None:
        raise ValueError("loaded ACT policy has no intent logits")
    if policy.temporal_agg:
        action = policy._aggregate(a_hat)
    else:
        if policy._t % policy._num_queries == 0:
            policy._cached_actions = a_hat.squeeze(0)
        step_in_chunk = policy._t % policy._num_queries
        action = policy._cached_actions[step_in_chunk].cpu().numpy()
    policy._t += 1
    action = action * policy.norm_stats["action_std"] + policy.norm_stats["action_mean"]
    prob = torch.sigmoid(intent_logits[0, 0]).detach().cpu().numpy()
    return action.astype(np.float32), np.asarray(prob, dtype=np.float32)


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
        ("raw_action_replay", "raw_policy_action"),
        ("phase_gated_action_replay", "phase_gated_action"),
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
        gohome_active=np.asarray(result["gohome_active"], dtype=bool),
    )


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


if __name__ == "__main__":
    main()
