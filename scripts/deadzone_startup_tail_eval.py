#!/usr/bin/env python3
"""Compute startup and tail deadzone gates from offline policy eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.data.handoff_labels import compute_gohome_eligibility_labels
from testbed.policies.deadzone_eval import (
    effective_direction_mask,
    find_episode_action_files,
    load_deadzone_thresholds,
    parse_eval_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute first-effective startup and pre-gohome tail stability gates."
    )
    parser.add_argument("--eval", dest="eval_specs", action="append", required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="20Hz HDF5 dataset used by offline eval.")
    parser.add_argument("--raw-dataset-dir", type=Path, required=True, help="Raw source HDF5 dataset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-window-steps", type=int, default=40)
    parser.add_argument("--tail-idle-action-threshold", type=float, default=0.05)
    args = parser.parse_args()

    thresholds = load_deadzone_thresholds(args.deadzone_json)
    specs = [parse_eval_spec(value) for value in args.eval_specs]
    startup_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []

    for spec in specs:
        for episode_id, action_path in find_episode_action_files(spec.eval_dir, manifest=args.manifest):
            with np.load(action_path) as data:
                expert = np.asarray(data["expert_action"], dtype=np.float32)
                policy = np.asarray(data["policy_action"], dtype=np.float32)
            startup_rows.append(
                _startup_row(
                    model=spec.model,
                    episode_id=episode_id,
                    expert=expert,
                    policy=policy,
                    thresholds=thresholds,
                    window_steps=args.startup_window_steps,
                )
            )
            tail = _tail_row(
                model=spec.model,
                episode_id=episode_id,
                policy=policy,
                expert=expert,
                thresholds=thresholds,
                dataset_dir=args.dataset_dir,
                raw_dataset_dir=args.raw_dataset_dir,
                idle_action_threshold=args.tail_idle_action_threshold,
            )
            if tail is not None:
                tail_rows.append(tail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    startup_summary = args.output_dir / f"startup_first_expert_effective_{args.startup_window_steps}_summary.csv"
    startup_aggregate = args.output_dir / f"startup_first_expert_effective_{args.startup_window_steps}_aggregate.csv"
    tail_by_episode = args.output_dir / "tail_stability_by_episode.csv"
    tail_summary = args.output_dir / "tail_stability_summary.csv"
    write_csv(startup_summary, startup_rows)
    write_csv(startup_aggregate, _aggregate_startup(startup_rows))
    write_csv(tail_by_episode, tail_rows)
    tail_aggregate = _aggregate_tail(tail_rows)
    write_csv(tail_summary, tail_aggregate)
    (args.output_dir / "tail_stability_summary.json").write_text(
        json.dumps(
            {
                "deadzone_json": str(args.deadzone_json),
                "manifest": str(args.manifest),
                "dataset_dir": str(args.dataset_dir),
                "raw_dataset_dir": str(args.raw_dataset_dir),
                "startup_window_steps": args.startup_window_steps,
                "tail_idle_action_threshold": args.tail_idle_action_threshold,
                "startup_aggregate": _aggregate_startup(startup_rows),
                "tail_aggregate": tail_aggregate,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Startup summary: {startup_summary}")
    print(f"Startup aggregate: {startup_aggregate}")
    print(f"Tail by episode: {tail_by_episode}")
    print(f"Tail summary: {tail_summary}")


def _startup_row(
    *,
    model: str,
    episode_id: str,
    expert: np.ndarray,
    policy: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    window_steps: int,
) -> dict[str, Any]:
    expert_eff = effective_direction_mask(expert, thresholds)
    policy_eff = effective_direction_mask(policy, thresholds)
    expert_any = expert_eff.any(axis=(1, 2))
    if np.any(expert_any):
        start = int(np.flatnonzero(expert_any)[0])
    else:
        start = 0
    end = min(int(start + window_steps), int(expert.shape[0]))
    expert_slice = expert_eff[start:end]
    policy_slice = policy_eff[start:end]
    steps = int(max(0, end - start))
    expert_any_slice = expert_slice.any(axis=(1, 2)) if steps else np.zeros(0, dtype=bool)
    policy_any_slice = policy_slice.any(axis=(1, 2)) if steps else np.zeros(0, dtype=bool)
    same_axis_dir = (expert_slice & policy_slice).any(axis=(1, 2)) if steps else np.zeros(0, dtype=bool)
    extra_or_wrong = policy_any_slice & ~same_axis_dir if steps else np.zeros(0, dtype=bool)
    expert_effective_frames = int(expert_any_slice.sum())
    policy_effective_frames = int(policy_any_slice.sum())
    return {
        "model": model,
        "episode_id": episode_id,
        "start_step": start,
        "end_step_exclusive": end,
        "steps": steps,
        "expert_effective_frames": expert_effective_frames,
        "policy_effective_frames": policy_effective_frames,
        "policy_any_effective_pct": _pct(policy_effective_frames, steps),
        "same_axis_dir_pct_of_expert_effective": _pct(int(same_axis_dir.sum()), expert_effective_frames)
        if expert_effective_frames
        else "",
        "extra_or_wrong_pct_of_policy_effective": _pct(int(extra_or_wrong.sum()), policy_effective_frames)
        if policy_effective_frames
        else 0.0,
    }


def _tail_row(
    *,
    model: str,
    episode_id: str,
    policy: np.ndarray,
    expert: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    dataset_dir: Path,
    raw_dataset_dir: Path,
    idle_action_threshold: float,
) -> dict[str, Any] | None:
    raw_path = raw_dataset_dir / f"{episode_id}.hdf5"
    episode_path = dataset_dir / f"{episode_id}.hdf5"
    if not raw_path.exists() or not episode_path.exists():
        return None
    with h5py.File(raw_path, "r") as raw_f:
        labels = compute_gohome_eligibility_labels(
            actions=np.asarray(raw_f["action"][()], dtype=np.float32),
            go_home_requested=_dataset_or_none(raw_f, "diagnostics/go_home_requested"),
            go_home_start_accepted=_dataset_or_none(raw_f, "diagnostics/go_home_start_accepted"),
            go_home_running=_dataset_or_none(raw_f, "diagnostics/go_home_running"),
            idle_action_threshold=idle_action_threshold,
            dwell_min_steps=0,
        )
    if labels.t_stop is None or labels.t_go is None:
        return None
    with h5py.File(episode_path, "r") as f:
        source_idx = np.asarray(f["diagnostics/source_observation_index"][()], dtype=np.int64)
    n = min(policy.shape[0], expert.shape[0], source_idx.shape[0])
    policy = policy[:n]
    expert = expert[:n]
    source_idx = source_idx[:n]
    tail_mask = (source_idx >= int(labels.t_stop)) & (source_idx <= int(labels.t_go))
    policy_eff = effective_direction_mask(policy, thresholds)
    expert_eff = effective_direction_mask(expert, thresholds)
    policy_any = policy_eff.any(axis=(1, 2))
    expert_any = expert_eff.any(axis=(1, 2))
    max_abs = np.max(np.abs(policy), axis=1) if policy.size else np.zeros(0, dtype=np.float32)
    tail_steps = int(tail_mask.sum())
    policy_effective_frames = int(np.count_nonzero(policy_any & tail_mask))
    axis_rates: dict[str, float] = {}
    for axis_idx, axis in enumerate(("swing", "boom", "stick", "bucket")):
        axis_rates[f"policy_{axis}_effective_rate"] = _rate(
            int(np.count_nonzero(policy_eff[:, axis_idx, :].any(axis=1) & tail_mask)),
            tail_steps,
        )
    tail_max_abs = max_abs[tail_mask]
    return {
        "model": model,
        "episode_id": episode_id,
        "t_stop_raw": int(labels.t_stop),
        "t_go_raw": int(labels.t_go),
        "raw_stop_to_go_frames": int(labels.t_go - labels.t_stop),
        "tail_steps_20hz": tail_steps,
        "tail_first_source_idx": int(source_idx[tail_mask][0]) if tail_steps else "",
        "tail_last_source_idx": int(source_idx[tail_mask][-1]) if tail_steps else "",
        "expert_any_effective_rate": _rate(int(np.count_nonzero(expert_any & tail_mask)), tail_steps),
        "policy_any_effective_rate": _rate(policy_effective_frames, tail_steps),
        "policy_effective_frames": policy_effective_frames,
        "policy_mean_max_abs": float(np.mean(tail_max_abs)) if tail_max_abs.size else 0.0,
        "policy_p95_max_abs": float(np.percentile(tail_max_abs, 95)) if tail_max_abs.size else 0.0,
        "policy_max_abs": float(np.max(tail_max_abs)) if tail_max_abs.size else 0.0,
        **axis_rates,
    }


def _aggregate_startup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in sorted({str(row["model"]) for row in rows}):
        group = [row for row in rows if row["model"] == model]
        out.append(
            {
                "model": model,
                "episodes": len(group),
                "mean_policy_any_effective_pct": _mean(row["policy_any_effective_pct"] for row in group),
                "median_policy_any_effective_pct": _median(row["policy_any_effective_pct"] for row in group),
                "mean_same_axis_dir_pct_of_expert_effective": _mean(
                    _numeric_or_zero(row["same_axis_dir_pct_of_expert_effective"]) for row in group
                ),
                "mean_extra_or_wrong_pct_of_policy_effective": _mean(
                    row["extra_or_wrong_pct_of_policy_effective"] for row in group
                ),
                "episodes_policy_any_effective_ge50": sum(
                    float(row["policy_any_effective_pct"]) >= 50.0 for row in group
                ),
            }
        )
    return out


def _aggregate_tail(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in sorted({str(row["model"]) for row in rows}):
        group = [row for row in rows if row["model"] == model]
        total_tail_steps = sum(int(row["tail_steps_20hz"]) for row in group)
        total_policy_effective = sum(int(row["policy_effective_frames"]) for row in group)
        out.append(
            {
                "model": model,
                "episodes": len(group),
                "total_tail_steps_20hz": total_tail_steps,
                "total_policy_effective_frames": total_policy_effective,
                "tail_policy_any_effective_rate": _rate(total_policy_effective, total_tail_steps),
                "mean_episode_policy_any_effective_rate": _mean(row["policy_any_effective_rate"] for row in group),
                "median_episode_policy_any_effective_rate": _median(
                    row["policy_any_effective_rate"] for row in group
                ),
                "worst_episode_policy_any_effective_rate": max(
                    (float(row["policy_any_effective_rate"]) for row in group),
                    default=0.0,
                ),
                "mean_policy_p95_max_abs": _mean(row["policy_p95_max_abs"] for row in group),
                "max_policy_max_abs": max((float(row["policy_max_abs"]) for row in group), default=0.0),
                "mean_swing_effective_rate": _mean(row["policy_swing_effective_rate"] for row in group),
                "mean_boom_effective_rate": _mean(row["policy_boom_effective_rate"] for row in group),
                "mean_stick_effective_rate": _mean(row["policy_stick_effective_rate"] for row in group),
                "mean_bucket_effective_rate": _mean(row["policy_bucket_effective_rate"] for row in group),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_or_none(f: h5py.File, key: str) -> np.ndarray | None:
    return np.asarray(f[key][()]) if key in f else None


def _pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(count) / float(total)


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def _mean(values: Any) -> float:
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    return float(np.mean(array)) if array.size else 0.0


def _median(values: Any) -> float:
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    return float(np.median(array)) if array.size else 0.0


def _numeric_or_zero(value: Any) -> float:
    if value == "" or value is None:
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
