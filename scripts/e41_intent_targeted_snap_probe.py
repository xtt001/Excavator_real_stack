#!/usr/bin/env python3
"""Materialize intent-targeted deadzone snap calibration probes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.e37_full_act_gate_smoke import read_train_ready_episode_ids_compatible
from scripts.e40_deadzone_snap_probe import _load_actions, _load_gate_probs, _write_actions_npz
from testbed.policies.deadzone_eval import AXIS_NAMES, load_deadzone_thresholds
from testbed.policies.offline_eval import aggregate_episode_results, compute_action_metrics, write_collection_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-eval-dir", type=Path, required=True)
    parser.add_argument("--gate-prob-dir", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin", type=float, action="append", required=True)
    parser.add_argument("--intent-threshold", type=float, action="append", required=True)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--phase-threshold", type=float, default=0.15)
    args = parser.parse_args()

    thresholds = load_deadzone_thresholds(args.deadzone_json)
    pos = np.asarray([thresholds[axis]["pos"] for axis in AXIS_NAMES], dtype=np.float32)
    neg = np.asarray([thresholds[axis]["neg"] for axis in AXIS_NAMES], dtype=np.float32)
    episode_ids = read_train_ready_episode_ids_compatible(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scan_rows = []
    for margin in [float(value) for value in args.margin]:
        for intent_threshold in [float(value) for value in args.intent_threshold]:
            label = _label(margin, intent_threshold)
            replay_dir = args.output_dir / label
            results = []
            snapped_counts = []
            for episode_id in episode_ids:
                source = _load_actions(args.source_eval_dir, episode_id)
                probs = _load_gate_probs(args.gate_prob_dir, episode_id)
                n = min(
                    source["expert_action"].shape[0],
                    source["policy_action"].shape[0],
                    probs["phase_prob"].shape[0],
                    probs["intent_prob"].shape[0],
                )
                phase_active = np.asarray(probs["phase_prob"][:n], dtype=np.float32) >= float(args.phase_threshold)
                snapped = snap_actions_near_deadzone_with_intent(
                    source["policy_action"][:n],
                    phase_active,
                    probs["intent_prob"][:n],
                    pos,
                    neg,
                    margin=margin,
                    epsilon=float(args.epsilon),
                    intent_threshold=intent_threshold,
                )
                _write_actions_npz(
                    replay_dir / "episodes" / episode_id / "actions.npz",
                    expert=source["expert_action"][:n],
                    policy=snapped,
                )
                results.append(
                    {
                        "episode_id": episode_id,
                        "episode_path": episode_id,
                        "n_steps": int(n),
                        "dt": 0.05,
                        "expert_action": source["expert_action"][:n],
                        "policy_action": snapped,
                        "metrics": compute_action_metrics(source["expert_action"][:n], snapped),
                    }
                )
                snapped_counts.append(int(np.count_nonzero(np.any(snapped != source["policy_action"][:n], axis=1))))
            aggregate = aggregate_episode_results(results)
            write_collection_report(
                aggregate=aggregate,
                output_dir=replay_dir,
                metadata={
                    "selection_mode": "e41_intent_targeted_snap_probe",
                    "margin": margin,
                    "intent_threshold": intent_threshold,
                    "epsilon": float(args.epsilon),
                    "phase_threshold": float(args.phase_threshold),
                    "source_eval_dir": str(args.source_eval_dir),
                    "gate_prob_dir": str(args.gate_prob_dir),
                },
            )
            scan_rows.append(
                {
                    "label": label,
                    "margin": margin,
                    "intent_threshold": intent_threshold,
                    "epsilon": float(args.epsilon),
                    "phase_threshold": float(args.phase_threshold),
                    "snapped_frames": int(sum(snapped_counts)),
                    "mae": float(aggregate["global_metrics"]["overall"]["mae"]),
                    "rmse": float(aggregate["global_metrics"]["overall"]["rmse"]),
                    "replay_dir": str(replay_dir),
                }
            )
    _write_csv(args.output_dir / "intent_targeted_snap_scan.csv", scan_rows)
    _write_json(
        args.output_dir / "intent_targeted_snap_manifest.json",
        {
            "source_eval_dir": str(args.source_eval_dir),
            "gate_prob_dir": str(args.gate_prob_dir),
            "deadzone_json": str(args.deadzone_json),
            "manifest": str(args.manifest),
            "margins": [float(value) for value in args.margin],
            "intent_thresholds": [float(value) for value in args.intent_threshold],
            "epsilon": float(args.epsilon),
            "phase_threshold": float(args.phase_threshold),
            "scan_csv": str(args.output_dir / "intent_targeted_snap_scan.csv"),
        },
    )
    print(f"Intent-targeted snap probe scan: {args.output_dir / 'intent_targeted_snap_scan.csv'}")


def snap_actions_near_deadzone_with_intent(
    actions: np.ndarray,
    phase_active: np.ndarray,
    intent_prob: np.ndarray,
    pos_thresholds: np.ndarray,
    neg_thresholds: np.ndarray,
    *,
    margin: float,
    epsilon: float,
    intent_threshold: float,
) -> np.ndarray:
    source = np.asarray(actions, dtype=np.float32)
    active = np.asarray(phase_active, dtype=bool).reshape(-1)
    intent = np.asarray(intent_prob, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"actions must be rank-2, got {source.shape}")
    if source.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"actions must have {len(AXIS_NAMES)} axes, got {source.shape[1]}")
    if active.shape[0] != source.shape[0]:
        raise ValueError(f"phase_active length mismatch: {active.shape[0]} vs {source.shape[0]}")
    if intent.shape != (source.shape[0], len(AXIS_NAMES) * 2):
        raise ValueError(f"intent_prob must have shape {(source.shape[0], len(AXIS_NAMES) * 2)}, got {intent.shape}")
    pos = np.asarray(pos_thresholds, dtype=np.float32).reshape(1, -1)
    neg = np.asarray(neg_thresholds, dtype=np.float32).reshape(1, -1)
    if pos.shape[1] != source.shape[1] or neg.shape[1] != source.shape[1]:
        raise ValueError("threshold axis count must match action axis count")

    out = source.copy()
    active_mask = active.reshape(-1, 1)
    pos_intent = intent[:, 0::2] >= float(intent_threshold)
    neg_intent = intent[:, 1::2] >= float(intent_threshold)
    pos_near = active_mask & pos_intent & (source >= pos - float(margin)) & (source < pos)
    neg_near = active_mask & neg_intent & (source <= -neg + float(margin)) & (source > -neg)
    out[pos_near] = np.broadcast_to(pos + float(epsilon), source.shape)[pos_near]
    out[neg_near] = np.broadcast_to(-neg - float(epsilon), source.shape)[neg_near]
    return out


def _label(margin: float, intent_threshold: float) -> str:
    return f"snap_m{int(round(float(margin) * 10000)):04d}_i{int(round(float(intent_threshold) * 100)):02d}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
