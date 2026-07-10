#!/usr/bin/env python3
"""Diagnose startup intent failures by comparing raw ACT and phase-gated actions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.e37_full_act_gate_smoke import read_train_ready_episode_ids_compatible
from testbed.policies.deadzone_eval import effective_direction_mask, load_deadzone_thresholds


AXIS_NAMES = ("swing", "boom", "stick", "bucket")
DIRECTION_NAMES = ("pos", "neg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-eval-dir", type=Path, required=True)
    parser.add_argument("--gated-eval-dir", type=Path, required=True)
    parser.add_argument("--gate-prob-dir", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--four-domain-csv", type=Path, default=None)
    parser.add_argument("--eye2-domain-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-window-steps", type=int, default=40)
    args = parser.parse_args()

    episode_ids = read_train_ready_episode_ids_compatible(args.manifest)
    thresholds = load_deadzone_thresholds(args.deadzone_json)
    four_domains = _read_domain_map(args.four_domain_csv)
    eye2_domains = _read_domain_map(args.eye2_domain_csv)

    rows = []
    for episode_id in episode_ids:
        raw = _load_actions(args.raw_eval_dir, episode_id)
        gated = _load_actions(args.gated_eval_dir, episode_id)
        gate_probs = _load_gate_probs(args.gate_prob_dir, episode_id)
        rows.append(
            diagnose_episode_startup(
                episode_id=episode_id,
                expert_action=raw["expert_action"],
                raw_policy_action=raw["policy_action"],
                gated_policy_action=gated["policy_action"],
                phase_prob=gate_probs["phase_prob"],
                thresholds=thresholds,
                window_steps=int(args.startup_window_steps),
                four_domain=four_domains.get(episode_id, ""),
                eye2_domain=eye2_domains.get(episode_id, ""),
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            float(row["gated_policy_any_effective_pct"]),
            -float(row["gated_extra_or_wrong_pct_of_policy_effective"]),
            -float(row["raw_to_gated_lost_policy_effective_frames"]),
            row["episode_id"],
        ),
    )
    _write_csv(args.output_dir / "startup_diagnostics.csv", rows_sorted)
    summary = build_diagnostic_summary(rows)
    _write_json(args.output_dir / "startup_diagnostic_summary.json", summary)
    print(f"Startup diagnostics: {args.output_dir / 'startup_diagnostics.csv'}")
    print(f"Startup diagnostic summary: {args.output_dir / 'startup_diagnostic_summary.json'}")


def summarize_window_effectiveness(expert_eff: np.ndarray, policy_eff: np.ndarray) -> dict[str, Any]:
    expert = np.asarray(expert_eff, dtype=bool)
    policy = np.asarray(policy_eff, dtype=bool)
    if expert.shape != policy.shape:
        raise ValueError(f"expert and policy masks must share shape, got {expert.shape} vs {policy.shape}")
    if expert.ndim != 3:
        raise ValueError(f"effectiveness masks must be rank-3, got {expert.shape}")
    steps = int(expert.shape[0])
    expert_any = expert.any(axis=(1, 2))
    policy_any = policy.any(axis=(1, 2))
    same_dir = (expert & policy).any(axis=(1, 2))
    extra_or_wrong = policy_any & ~same_dir
    expert_effective = int(expert_any.sum())
    policy_effective = int(policy_any.sum())
    same_frames = int(same_dir.sum())
    extra_frames = int(extra_or_wrong.sum())
    return {
        "steps": steps,
        "expert_effective_frames": expert_effective,
        "policy_effective_frames": policy_effective,
        "same_dir_frames": same_frames,
        "extra_or_wrong_frames": extra_frames,
        "policy_any_effective_pct": _pct(policy_effective, steps),
        "same_dir_pct_of_expert_effective": _pct(same_frames, expert_effective),
        "extra_or_wrong_pct_of_policy_effective": _pct(extra_frames, policy_effective),
    }


def diagnose_episode_startup(
    *,
    episode_id: str,
    expert_action: np.ndarray,
    raw_policy_action: np.ndarray,
    gated_policy_action: np.ndarray,
    phase_prob: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    window_steps: int,
    four_domain: str,
    eye2_domain: str,
) -> dict[str, Any]:
    n = min(
        int(expert_action.shape[0]),
        int(raw_policy_action.shape[0]),
        int(gated_policy_action.shape[0]),
        int(phase_prob.shape[0]),
    )
    expert = np.asarray(expert_action[:n], dtype=np.float32)
    raw = np.asarray(raw_policy_action[:n], dtype=np.float32)
    gated = np.asarray(gated_policy_action[:n], dtype=np.float32)
    phase = np.asarray(phase_prob[:n], dtype=np.float32).reshape(-1)
    expert_eff = effective_direction_mask(expert, thresholds)
    raw_eff = effective_direction_mask(raw, thresholds)
    gated_eff = effective_direction_mask(gated, thresholds)
    expert_any = expert_eff.any(axis=(1, 2))
    start = int(np.flatnonzero(expert_any)[0]) if np.any(expert_any) else 0
    end = min(start + int(window_steps), n)
    sl = slice(start, end)
    raw_summary = summarize_window_effectiveness(expert_eff[sl], raw_eff[sl])
    gated_summary = summarize_window_effectiveness(expert_eff[sl], gated_eff[sl])
    raw_any = raw_eff[sl].any(axis=(1, 2))
    gated_any = gated_eff[sl].any(axis=(1, 2))
    raw_same = (expert_eff[sl] & raw_eff[sl]).any(axis=(1, 2))
    gated_same = (expert_eff[sl] & gated_eff[sl]).any(axis=(1, 2))
    raw_extra = raw_any & ~raw_same
    gated_extra = gated_any & ~gated_same
    expert_any_window = expert_eff[sl].any(axis=(1, 2))
    phase_window = phase[sl]
    active = phase_window >= 0.15
    axis_rows = _axis_direction_counts(expert_eff[sl], raw_eff[sl], gated_eff[sl])
    row: dict[str, Any] = {
        "episode_id": episode_id,
        "four_domain": four_domain,
        "eye2_domain": eye2_domain,
        "start_step": start,
        "end_step_exclusive": end,
        "steps": int(end - start),
        "phase_active_pct": _pct(int(active.sum()), int(active.shape[0])),
        "phase_prob_mean": float(np.mean(phase_window)) if phase_window.size else 0.0,
        "phase_prob_p50": float(np.percentile(phase_window, 50)) if phase_window.size else 0.0,
        "phase_prob_p95": float(np.percentile(phase_window, 95)) if phase_window.size else 0.0,
        "expert_effective_frames": raw_summary["expert_effective_frames"],
        "raw_policy_effective_frames": raw_summary["policy_effective_frames"],
        "raw_policy_any_effective_pct": raw_summary["policy_any_effective_pct"],
        "raw_same_dir_pct_of_expert_effective": raw_summary["same_dir_pct_of_expert_effective"],
        "raw_extra_or_wrong_pct_of_policy_effective": raw_summary["extra_or_wrong_pct_of_policy_effective"],
        "gated_policy_effective_frames": gated_summary["policy_effective_frames"],
        "gated_policy_any_effective_pct": gated_summary["policy_any_effective_pct"],
        "gated_same_dir_pct_of_expert_effective": gated_summary["same_dir_pct_of_expert_effective"],
        "gated_extra_or_wrong_pct_of_policy_effective": gated_summary["extra_or_wrong_pct_of_policy_effective"],
        "raw_to_gated_lost_policy_effective_frames": int(np.count_nonzero(raw_any & ~gated_any)),
        "raw_to_gated_lost_same_dir_frames": int(np.count_nonzero(raw_same & ~gated_same)),
        "gated_new_extra_or_wrong_frames": int(np.count_nonzero(gated_extra & ~raw_extra)),
        "expert_effective_but_phase_inactive_frames": int(np.count_nonzero(expert_any_window & ~active)),
    }
    row.update(axis_rows)
    return row


def build_diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0}
    worst_effective = sorted(rows, key=lambda row: float(row["gated_policy_any_effective_pct"]))[:8]
    worst_extra = sorted(
        rows,
        key=lambda row: float(row["gated_extra_or_wrong_pct_of_policy_effective"]),
        reverse=True,
    )[:8]
    return {
        "episodes": len(rows),
        "mean_gated_policy_any_effective_pct": _mean(row["gated_policy_any_effective_pct"] for row in rows),
        "mean_gated_same_dir_pct": _mean(row["gated_same_dir_pct_of_expert_effective"] for row in rows),
        "mean_gated_extra_or_wrong_pct": _mean(row["gated_extra_or_wrong_pct_of_policy_effective"] for row in rows),
        "total_raw_to_gated_lost_same_dir_frames": int(
            sum(int(row["raw_to_gated_lost_same_dir_frames"]) for row in rows)
        ),
        "total_expert_effective_but_phase_inactive_frames": int(
            sum(int(row["expert_effective_but_phase_inactive_frames"]) for row in rows)
        ),
        "worst_startup_effective": [_compact_row(row) for row in worst_effective],
        "worst_extra_or_wrong": [_compact_row(row) for row in worst_extra],
    }


def _axis_direction_counts(
    expert_eff: np.ndarray,
    raw_eff: np.ndarray,
    gated_eff: np.ndarray,
) -> dict[str, int]:
    out: dict[str, int] = {}
    for axis_idx, axis in enumerate(AXIS_NAMES):
        for dir_idx, direction in enumerate(DIRECTION_NAMES):
            key = f"{axis}_{direction}"
            expert = expert_eff[:, axis_idx, dir_idx]
            raw = raw_eff[:, axis_idx, dir_idx]
            gated = gated_eff[:, axis_idx, dir_idx]
            out[f"expert_{key}_frames"] = int(np.count_nonzero(expert))
            out[f"raw_{key}_frames"] = int(np.count_nonzero(raw))
            out[f"gated_{key}_frames"] = int(np.count_nonzero(gated))
    return out


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "episode_id",
        "four_domain",
        "eye2_domain",
        "gated_policy_any_effective_pct",
        "gated_same_dir_pct_of_expert_effective",
        "gated_extra_or_wrong_pct_of_policy_effective",
        "raw_policy_any_effective_pct",
        "raw_same_dir_pct_of_expert_effective",
        "raw_extra_or_wrong_pct_of_policy_effective",
        "phase_active_pct",
        "expert_effective_but_phase_inactive_frames",
    )
    return {key: row.get(key, "") for key in keys}


def _load_actions(eval_dir: Path, episode_id: str) -> dict[str, np.ndarray]:
    path = Path(eval_dir) / "episodes" / episode_id / "actions.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {
            "expert_action": np.asarray(data["expert_action"], dtype=np.float32),
            "policy_action": np.asarray(data["policy_action"], dtype=np.float32),
        }


def _load_gate_probs(gate_prob_dir: Path, episode_id: str) -> dict[str, np.ndarray]:
    path = Path(gate_prob_dir) / f"{episode_id}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _read_domain_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        return {row["episode_id"]: row["dominant_domain"] for row in csv.DictReader(f)}


def _pct(num: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(num) / float(denom) * 100.0


def _mean(values: Any) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        return 0.0
    return float(np.mean(np.asarray(materialized, dtype=np.float64)))


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
