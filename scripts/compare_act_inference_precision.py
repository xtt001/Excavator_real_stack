#!/usr/bin/env python3
"""Compare FP32 and FP16 ACT inference on identical recorded observations.

This is an offline teacher-forced replay and latency probe.  It verifies model
output precision and deadzone semantics; it does not prove closed-loop control
equivalence on the excavator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml

from testbed.data.dataset import _read_camera_image
from testbed.policies.deadzone_eval import AXIS_NAMES, load_deadzone_thresholds
from testbed.policies.offline_eval import load_policy_for_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--episode-file", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-action-abs-diff", type=float, default=None)
    parser.add_argument("--min-p95-speedup", type=float, default=1.0)
    parser.add_argument("--require-deadzone-equivalence", action="store_true")
    args = parser.parse_args()

    bundle = args.bundle_dir.resolve()
    episode_file = args.episode_file.resolve()
    resolved_path = bundle / "resolved_config.yaml"
    checkpoint = bundle / "policy_best.ckpt"
    stats_path = bundle / "dataset_stats.pkl"
    for path in (resolved_path, checkpoint, stats_path, episode_file, args.deadzone_json):
        if not path.exists():
            raise FileNotFoundError(path)

    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    camera_names = [
        str(name) for name in resolved.get("task", {}).get("camera_names", ["fpv"])
    ]
    observations = _load_observations(
        episode_file,
        camera_names=camera_names,
        max_steps=max(1, int(args.max_steps)),
        stride=max(1, int(args.stride)),
    )
    thresholds = load_deadzone_thresholds(args.deadzone_json)

    runs = {}
    for precision in ("fp32", "fp16"):
        runs[precision] = _run_precision(
            precision=precision,
            bundle=bundle,
            checkpoint=checkpoint,
            resolved_path=resolved_path,
            stats_path=stats_path,
            observations=observations,
            device=str(args.device),
            warmup_steps=max(0, int(args.warmup_steps)),
        )

    executed = _comparison(
        runs["fp32"]["actions"],
        runs["fp16"]["actions"],
        thresholds=thresholds,
    )
    raw_chunks = _comparison(
        runs["fp32"]["raw_chunks"],
        runs["fp16"]["raw_chunks"],
        thresholds=thresholds,
    )
    fp32_latency = runs["fp32"]["latency_ms"]
    fp16_latency = runs["fp16"]["latency_ms"]
    p50_speedup = _safe_ratio(
        np.percentile(fp32_latency, 50),
        np.percentile(fp16_latency, 50),
    )
    p95_speedup = _safe_ratio(
        np.percentile(fp32_latency, 95),
        np.percentile(fp16_latency, 95),
    )
    report = {
        "schema_version": 1,
        "evidence_scope": (
            "offline teacher-forced recorded-observation replay; not real closed-loop execution"
        ),
        "bundle_dir": str(bundle),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "episode_file": str(episode_file),
        "camera_names": camera_names,
        "steps": len(observations),
        "device": str(args.device),
        "temporal_aggregation": True,
        "latency_ms": {
            "fp32": _latency_summary(fp32_latency),
            "fp16": _latency_summary(fp16_latency),
            "p50_speedup": p50_speedup,
            "p95_speedup": p95_speedup,
        },
        "executed_action_comparison": executed,
        "raw_action_chunk_comparison": raw_chunks,
        "acceptance": _acceptance(
            executed=executed,
            raw_chunks=raw_chunks,
            max_action_abs_diff=args.max_action_abs_diff,
            p95_speedup=p95_speedup,
            min_p95_speedup=float(args.min_p95_speedup),
            require_deadzone_equivalence=bool(args.require_deadzone_equivalence),
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["acceptance"]["evaluated"] and not report["acceptance"]["passed"]:
        raise SystemExit(2)


def _load_observations(
    episode_file: Path,
    *,
    camera_names: list[str],
    max_steps: int,
    stride: int,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    with h5py.File(episode_file, "r") as h5:
        qpos = np.asarray(h5["observations/qpos"], dtype=np.float32)
        qvel = np.asarray(h5["observations/qvel"], dtype=np.float32)
        available_steps = min(len(qpos), len(qvel))
        for step in range(0, available_steps, stride):
            obs: dict[str, Any] = {"qpos": qpos[step], "qvel": qvel[step]}
            for camera_name in camera_names:
                obs[f"image_{camera_name}"] = _read_camera_image(
                    h5,
                    camera_name,
                    step,
                )
            observations.append(obs)
            if len(observations) >= max_steps:
                break
    if not observations:
        raise ValueError(f"episode contains no comparable observations: {episode_file}")
    return observations


def _run_precision(
    *,
    precision: str,
    bundle: Path,
    checkpoint: Path,
    resolved_path: Path,
    stats_path: Path,
    observations: list[dict[str, Any]],
    device: str,
    warmup_steps: int,
) -> dict[str, np.ndarray]:
    policy = load_policy_for_episode(
        bundle_dir=bundle,
        ckpt_path=checkpoint,
        resolved_config_path=resolved_path,
        stats_path=stats_path,
        max_episode_len=len(observations),
        temporal_agg=True,
        device=device,
        inference_precision=precision,
    )
    for index in range(warmup_steps):
        policy.predict(observations[index % len(observations)])
    policy.reset()

    actions = []
    raw_chunks = []
    latencies_ms = []
    for obs in observations:
        _synchronize(policy.device)
        start = time.perf_counter()
        actions.append(policy.predict(obs))
        _synchronize(policy.device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        normalized_chunk = policy.last_raw_action_chunk()
        raw_chunks.append(
            normalized_chunk * policy.norm_stats["action_std"]
            + policy.norm_stats["action_mean"]
        )
    return {
        "actions": np.asarray(actions, dtype=np.float32),
        "raw_chunks": np.asarray(raw_chunks, dtype=np.float32),
        "latency_ms": np.asarray(latencies_ms, dtype=np.float64),
    }


def _comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise ValueError(f"comparison shape mismatch: {reference.shape} vs {candidate.shape}")
    error = np.abs(candidate - reference)
    reference_class = _deadzone_class(reference, thresholds)
    candidate_class = _deadzone_class(candidate, thresholds)
    disagreements = reference_class != candidate_class
    return {
        "shape": list(reference.shape),
        "max_abs_diff": float(np.max(error)),
        "mean_abs_diff": float(np.mean(error)),
        "p95_abs_diff": float(np.percentile(error, 95)),
        "deadzone_class_disagreement_count": int(np.sum(disagreements)),
        "deadzone_class_total": int(disagreements.size),
        "deadzone_class_disagreement_rate": float(np.mean(disagreements)),
    }


def _deadzone_class(
    action: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> np.ndarray:
    result = np.zeros_like(action, dtype=np.int8)
    for axis_index, axis_name in enumerate(AXIS_NAMES):
        result[..., axis_index] = np.where(
            action[..., axis_index] >= float(thresholds[axis_name]["pos"]),
            1,
            np.where(
                action[..., axis_index] <= -float(thresholds[axis_name]["neg"]),
                -1,
                0,
            ),
        )
    return result


def _acceptance(
    *,
    executed: dict[str, Any],
    raw_chunks: dict[str, Any],
    max_action_abs_diff: float | None,
    p95_speedup: float | None,
    min_p95_speedup: float,
    require_deadzone_equivalence: bool,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "p95_latency_speedup": (
            p95_speedup is not None and p95_speedup >= min_p95_speedup
        )
    }
    if max_action_abs_diff is not None:
        checks["executed_action_max_abs_diff"] = (
            float(executed["max_abs_diff"]) <= float(max_action_abs_diff)
        )
    if require_deadzone_equivalence:
        checks["executed_action_deadzone_class"] = (
            int(executed["deadzone_class_disagreement_count"]) == 0
        )
        checks["raw_chunk_deadzone_class"] = (
            int(raw_chunks["deadzone_class_disagreement_count"]) == 0
        )
    return {
        "evaluated": bool(checks),
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "max_action_abs_diff_limit": max_action_abs_diff,
        "min_p95_speedup": min_p95_speedup,
        "require_deadzone_equivalence": require_deadzone_equivalence,
    }


def _latency_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0.0 else float(numerator / denominator)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
