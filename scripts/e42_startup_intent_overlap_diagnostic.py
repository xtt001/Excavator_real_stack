#!/usr/bin/env python3
"""Diagnose whether ACT intent probabilities separate startup same vs extra motion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.e37_full_act_gate_smoke import read_train_ready_episode_ids_compatible
from scripts.e40_deadzone_snap_probe import _load_actions, _load_gate_probs
from testbed.policies.deadzone_eval import effective_direction_mask, load_deadzone_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--gate-prob-dir", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-window-steps", type=int, default=40)
    parser.add_argument("--high-intent-threshold", type=float, action="append", default=[0.5, 0.7, 0.8, 0.9])
    args = parser.parse_args()

    thresholds = load_deadzone_thresholds(args.deadzone_json)
    rows = []
    for episode_id in read_train_ready_episode_ids_compatible(args.manifest):
        actions = _load_actions(args.eval_dir, episode_id)
        probs = _load_gate_probs(args.gate_prob_dir, episode_id)
        rows.append(
            diagnose_episode_intent_overlap(
                episode_id=episode_id,
                expert_action=actions["expert_action"],
                policy_action=actions["policy_action"],
                intent_prob=probs["intent_prob"],
                thresholds=thresholds,
                window_steps=int(args.startup_window_steps),
                high_intent_thresholds=tuple(float(v) for v in args.high_intent_threshold),
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "startup_intent_overlap_by_episode.csv", rows)
    summary = build_overlap_summary(rows)
    _write_json(args.output_dir / "startup_intent_overlap_summary.json", summary)
    print(f"Startup intent overlap: {args.output_dir / 'startup_intent_overlap_by_episode.csv'}")
    print(f"Startup intent overlap summary: {args.output_dir / 'startup_intent_overlap_summary.json'}")


def diagnose_episode_intent_overlap(
    *,
    episode_id: str,
    expert_action: np.ndarray,
    policy_action: np.ndarray,
    intent_prob: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    window_steps: int,
    high_intent_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    n = min(int(expert_action.shape[0]), int(policy_action.shape[0]), int(intent_prob.shape[0]))
    expert = np.asarray(expert_action[:n], dtype=np.float32)
    policy = np.asarray(policy_action[:n], dtype=np.float32)
    intent = np.asarray(intent_prob[:n], dtype=np.float32)
    expert_eff = effective_direction_mask(expert, thresholds)
    policy_eff = effective_direction_mask(policy, thresholds)
    expert_any = expert_eff.any(axis=(1, 2))
    start = int(np.flatnonzero(expert_any)[0]) if np.any(expert_any) else 0
    end = min(start + int(window_steps), n)
    summary = summarize_startup_intent_overlap(
        expert_eff[start:end],
        policy_eff[start:end],
        intent[start:end],
        high_intent_thresholds=high_intent_thresholds,
    )
    return {
        "episode_id": episode_id,
        "start_step": start,
        "end_step_exclusive": end,
        "steps": int(end - start),
        **summary,
    }


def summarize_startup_intent_overlap(
    expert_eff: np.ndarray,
    policy_eff: np.ndarray,
    intent_prob: np.ndarray,
    *,
    high_intent_thresholds: Iterable[float],
) -> dict[str, Any]:
    expert = np.asarray(expert_eff, dtype=bool)
    policy = np.asarray(policy_eff, dtype=bool)
    intent = np.asarray(intent_prob, dtype=np.float32)
    if expert.shape != policy.shape:
        raise ValueError(f"expert and policy masks must share shape, got {expert.shape} vs {policy.shape}")
    if expert.ndim != 3:
        raise ValueError(f"effectiveness masks must be rank-3, got {expert.shape}")
    expected_intent_shape = (expert.shape[0], expert.shape[1] * expert.shape[2])
    if intent.shape != expected_intent_shape:
        raise ValueError(f"intent_prob must have shape {expected_intent_shape}, got {intent.shape}")

    same_values = _intent_values(intent, expert & policy)
    extra_values = _intent_values(intent, policy & ~expert)
    missing_values = _intent_values(intent, expert & ~policy)
    row: dict[str, Any] = {
        "same_count": int(same_values.size),
        "extra_count": int(extra_values.size),
        "missing_count": int(missing_values.size),
        **_stats("same_intent", same_values),
        **_stats("extra_intent", extra_values),
        **_stats("missing_intent", missing_values),
    }
    for threshold in high_intent_thresholds:
        key = f"extra_intent_ge_{float(threshold):.2f}_pct"
        row[key] = _pct(int(np.count_nonzero(extra_values >= float(threshold))), int(extra_values.size))
    return row


def build_overlap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0}
    extra_count = int(sum(int(row["extra_count"]) for row in rows))
    missing_count = int(sum(int(row["missing_count"]) for row in rows))
    same_count = int(sum(int(row["same_count"]) for row in rows))
    high_keys = sorted(key for key in rows[0] if key.startswith("extra_intent_ge_"))
    return {
        "episodes": len(rows),
        "same_count": same_count,
        "extra_count": extra_count,
        "missing_count": missing_count,
        "mean_same_intent_mean": _mean(row["same_intent_mean"] for row in rows if row["same_count"]),
        "mean_extra_intent_mean": _mean(row["extra_intent_mean"] for row in rows if row["extra_count"]),
        "mean_missing_intent_mean": _mean(row["missing_intent_mean"] for row in rows if row["missing_count"]),
        "high_intent_extra_means": {key: _mean(row[key] for row in rows if row["extra_count"]) for key in high_keys},
        "worst_extra_count": [
            _compact(row)
            for row in sorted(rows, key=lambda item: int(item["extra_count"]), reverse=True)[:8]
            if int(row["extra_count"]) > 0
        ],
        "worst_missing_count": [
            _compact(row)
            for row in sorted(rows, key=lambda item: int(item["missing_count"]), reverse=True)[:8]
            if int(row["missing_count"]) > 0
        ],
    }


def _intent_values(intent: np.ndarray, mask: np.ndarray) -> np.ndarray:
    flat_mask = np.asarray(mask, dtype=bool).reshape(mask.shape[0], -1)
    return np.asarray(intent, dtype=np.float32)[flat_mask]


def _stats(prefix: str, values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            f"{prefix}_mean": "",
            f"{prefix}_p10": "",
            f"{prefix}_p50": "",
            f"{prefix}_p90": "",
            f"{prefix}_min": "",
            f"{prefix}_max": "",
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "episode_id",
        "same_count",
        "extra_count",
        "missing_count",
        "same_intent_mean",
        "extra_intent_mean",
        "missing_intent_mean",
        "extra_intent_ge_0.70_pct",
        "extra_intent_ge_0.80_pct",
    )
    return {key: row.get(key, "") for key in keys}


def _pct(num: int, denom: int) -> float:
    return 0.0 if denom <= 0 else float(num) * 100.0 / float(denom)


def _mean(values: Iterable[Any]) -> float:
    vals = [float(value) for value in values if value != ""]
    return float(np.mean(vals)) if vals else 0.0


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
