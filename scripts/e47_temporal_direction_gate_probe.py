#!/usr/bin/env python3
"""Train a temporal-context axis-direction gate probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.e37_full_act_gate_smoke import read_train_ready_episode_ids_compatible
from scripts.e43_direction_gate_probe import (
    FEATURE_NAMES,
    _classification_metrics,
    _load_episode,
    _parse_float_list,
    _train_episode_heldout,
    _train_final_model,
    _write_csv,
    _write_direction_probs,
    _write_json,
    materialize_direction_gate,
)
from testbed.policies.deadzone_eval import load_deadzone_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-eval-dir", type=Path, required=True)
    parser.add_argument("--gate-prob-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-offsets", default="-10,-5,-2,-1,0,1,2,5,10")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--thresholds", default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--inactive-scales", default="0.25,0.50")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    thresholds = load_deadzone_thresholds(args.deadzone_json)
    offsets = [int(round(item)) for item in _parse_float_list(str(args.context_offsets))]
    episode_ids = read_train_ready_episode_ids_compatible(args.manifest)
    episodes = [
        _load_temporal_episode(
            episode_id=episode_id,
            source_eval_dir=args.source_eval_dir,
            gate_prob_dir=args.gate_prob_dir,
            dataset_dir=args.dataset_dir,
            thresholds=thresholds,
            offsets=offsets,
        )
        for episode_id in episode_ids
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows, direction_probs = _train_episode_heldout(
        episodes=episodes,
        folds=int(args.folds),
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    labels = np.concatenate([ep["label"] for ep in episodes], axis=0)
    probs = np.concatenate([direction_probs[str(ep["episode_id"])] for ep in episodes], axis=0)
    _write_json(
        args.output_dir / "fold_summary.json",
        {"folds": fold_rows, "overall_at_0.5": _classification_metrics(labels, probs, 0.5)},
    )
    _write_direction_probs(args.output_dir / "direction_probs", direction_probs)

    final_payload = _train_final_model(
        episodes=episodes,
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    final_payload["feature_names"] = episodes[0]["feature_names"] if episodes else []
    torch.save(final_payload, args.output_dir / "temporal_direction_gate_model.pt")
    _write_json(
        args.output_dir / "temporal_direction_gate_model_metadata.json",
        {
            "feature_names": episodes[0]["feature_names"] if episodes else [],
            "base_feature_names": FEATURE_NAMES,
            "context_offsets": offsets,
            "output_order": [
                "swing_pos",
                "swing_neg",
                "boom_pos",
                "boom_neg",
                "stick_pos",
                "stick_neg",
                "bucket_pos",
                "bucket_neg",
            ],
            "label": "expert_action_crosses_runtime_scaled_deadzone_per_axis_direction",
            "source_eval_dir": str(args.source_eval_dir),
            "gate_prob_dir": str(args.gate_prob_dir),
            "dataset_dir": str(args.dataset_dir),
            "deadzone_json": str(args.deadzone_json),
            "manifest": str(args.manifest),
        },
    )

    scan_rows = []
    for threshold in _parse_float_list(str(args.thresholds)):
        for inactive_scale in _parse_float_list(str(args.inactive_scales)):
            label = _label(threshold, inactive_scale)
            replay_dir = args.output_dir / label
            aggregate, scaled_frames = materialize_direction_gate(
                episodes=episodes,
                direction_probs=direction_probs,
                threshold=threshold,
                inactive_scale=inactive_scale,
                output_dir=replay_dir,
            )
            scan_row = {
                "label": label,
                "threshold": threshold,
                "inactive_scale": inactive_scale,
                "scaled_frames": scaled_frames,
                "mae": float(aggregate["global_metrics"]["overall"]["mae"]),
                "rmse": float(aggregate["global_metrics"]["overall"]["rmse"]),
                "replay_dir": str(replay_dir),
            }
            _write_json(replay_dir / "gate_summary.json", build_temporal_gate_summary(scan_row))
            scan_rows.append(scan_row)
    _write_csv(args.output_dir / "temporal_direction_gate_scan.csv", scan_rows)
    _write_json(
        args.output_dir / "temporal_direction_gate_manifest.json",
        {
            "source_eval_dir": str(args.source_eval_dir),
            "gate_prob_dir": str(args.gate_prob_dir),
            "dataset_dir": str(args.dataset_dir),
            "deadzone_json": str(args.deadzone_json),
            "manifest": str(args.manifest),
            "context_offsets": offsets,
            "thresholds": _parse_float_list(str(args.thresholds)),
            "inactive_scales": _parse_float_list(str(args.inactive_scales)),
            "scan_csv": str(args.output_dir / "temporal_direction_gate_scan.csv"),
        },
    )
    print(f"Temporal direction gate scan: {args.output_dir / 'temporal_direction_gate_scan.csv'}")


def build_temporal_context_features(
    features: np.ndarray,
    feature_names: list[str],
    *,
    offsets: list[int],
) -> tuple[np.ndarray, list[str]]:
    base = np.asarray(features, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {base.shape}")
    if base.shape[1] != len(feature_names):
        raise ValueError(f"feature name count {len(feature_names)} does not match feature dim {base.shape[1]}")
    if not offsets:
        raise ValueError("offsets must not be empty")
    chunks = []
    names = []
    max_index = base.shape[0] - 1
    for offset in offsets:
        indices = np.clip(np.arange(base.shape[0]) + int(offset), 0, max_index)
        chunks.append(base[indices])
        suffix = f"t{int(offset):+d}" if int(offset) else "t0"
        names.extend([f"{name}_{suffix}" for name in feature_names])
    return np.concatenate(chunks, axis=1).astype(np.float32), names


def build_temporal_gate_summary(scan_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_mode": "all_train_ready_oof_temporal_direction_gate",
        "gate_name": str(scan_row.get("label", "")),
        "scan_row": dict(scan_row),
    }


def _load_temporal_episode(
    *,
    episode_id: str,
    source_eval_dir: Path,
    gate_prob_dir: Path,
    dataset_dir: Path,
    thresholds: dict[str, dict[str, float]],
    offsets: list[int],
) -> dict[str, Any]:
    episode = _load_episode(
        episode_id=episode_id,
        source_eval_dir=source_eval_dir,
        gate_prob_dir=gate_prob_dir,
        dataset_dir=dataset_dir,
        thresholds=thresholds,
    )
    features, feature_names = build_temporal_context_features(
        episode["features"],
        FEATURE_NAMES,
        offsets=offsets,
    )
    episode["features"] = features
    episode["feature_names"] = feature_names
    return episode


def _label(threshold: float, inactive_scale: float) -> str:
    return f"tdir_t{int(round(float(threshold) * 100)):02d}_s{int(round(float(inactive_scale) * 100)):02d}"


if __name__ == "__main__":
    main()
