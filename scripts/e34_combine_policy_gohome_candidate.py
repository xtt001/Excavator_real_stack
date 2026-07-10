#!/usr/bin/env python3
"""Combine the current best action gate with a gohome request gate into one report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default="E34")
    parser.add_argument("--action-model", required=True)
    parser.add_argument("--action-gate-dir", type=Path, required=True)
    parser.add_argument("--action-deadzone-gate-dir", type=Path, required=True)
    parser.add_argument("--gohome-gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    action_gate_summary = _read_json(args.action_gate_dir / "gate_summary.json")
    startup_rows = _read_csv(args.action_deadzone_gate_dir / "startup_first_expert_effective_40_aggregate.csv")
    tail_rows = _read_csv(args.action_deadzone_gate_dir / "tail_stability_summary.csv")
    gohome_gate_summary = _read_json(args.gohome_gate_dir / "gate_summary.json")

    artifact_paths = {
        "action_gate_dir": str(args.action_gate_dir),
        "action_deadzone_gate_dir": str(args.action_deadzone_gate_dir),
        "gohome_gate_dir": str(args.gohome_gate_dir),
        "action_gate_summary": str(args.action_gate_dir / "gate_summary.json"),
        "startup_gate_summary": str(args.action_deadzone_gate_dir / "startup_first_expert_effective_40_aggregate.csv"),
        "tail_gate_summary": str(args.action_deadzone_gate_dir / "tail_stability_summary.csv"),
        "gohome_gate_summary": str(args.gohome_gate_dir / "gate_summary.json"),
    }
    summary = build_combined_summary(
        candidate_id=str(args.candidate_id),
        action_model=str(args.action_model),
        action_gate_summary=action_gate_summary,
        startup_row=select_model_row(startup_rows, str(args.action_model)),
        tail_row=select_model_row(tail_rows, str(args.action_model)),
        gohome_gate_summary=gohome_gate_summary,
        artifact_paths=artifact_paths,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "combined_candidate_summary.json", summary)
    _write_csv(args.output_dir / "combined_candidate_summary.csv", [summary])
    _write_json(args.output_dir / "candidate_artifacts.json", artifact_paths)
    print(f"Combined summary: {args.output_dir / 'combined_candidate_summary.json'}")
    print(f"Artifact manifest: {args.output_dir / 'candidate_artifacts.json'}")


def select_model_row(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    matches = [row for row in rows if str(row.get("model", "")) == str(model)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one row for model {model!r}, got {len(matches)}")
    return matches[0]


def build_combined_summary(
    *,
    candidate_id: str,
    action_model: str,
    action_gate_summary: dict[str, Any],
    startup_row: dict[str, Any],
    tail_row: dict[str, Any],
    gohome_gate_summary: dict[str, Any],
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    action_scan = dict(action_gate_summary.get("scan_row", {}))
    gohome_scan = dict(gohome_gate_summary.get("scan_row", {}))
    tail_effective_frames = int(_number(tail_row.get("total_policy_effective_frames", 0)))
    tail_effective_rate = _number(tail_row.get("tail_policy_any_effective_rate", 0.0))
    pre_tail_episodes = int(_number(gohome_scan.get("pre_tail_false_positive_episodes", 0)))
    pre_tail_frames = int(_number(gohome_scan.get("pre_tail_active_frames", 0)))
    event_recall = _number(gohome_scan.get("event_recall", 0.0))
    tail_stop_pass = tail_effective_frames == 0 and tail_effective_rate == 0.0
    gohome_pre_tail_pass = pre_tail_episodes == 0 and pre_tail_frames == 0
    gohome_recall_pass = event_recall >= 0.95

    return {
        "candidate_id": str(candidate_id),
        "action_model": str(action_model),
        "action_gate": str(action_gate_summary.get("gate_name", "")),
        "gohome_gate": str(gohome_gate_summary.get("gate", "")),
        "action_mae": _maybe_number(action_scan.get("mae")),
        "action_rmse": _maybe_number(action_scan.get("rmse")),
        "startup_policy_any_effective_pct": _first_number(
            startup_row.get("mean_policy_any_effective_pct"),
            action_scan.get("startup_policy_any_effective_pct"),
        ),
        "startup_same_dir_pct": _first_number(
            startup_row.get("mean_same_axis_dir_pct_of_expert_effective"),
            action_scan.get("startup_same_dir_pct"),
        ),
        "startup_extra_or_wrong_pct": _first_number(
            startup_row.get("mean_extra_or_wrong_pct_of_policy_effective"),
            action_scan.get("startup_extra_or_wrong_pct"),
        ),
        "main_policy_any_effective_pct": _maybe_number(action_scan.get("main_policy_any_effective_pct")),
        "main_same_dir_pct": _maybe_number(action_scan.get("main_same_dir_pct")),
        "main_extra_or_wrong_pct": _maybe_number(action_scan.get("main_extra_or_wrong_pct")),
        "tail_policy_effective_frames": tail_effective_frames,
        "tail_policy_any_effective_rate": tail_effective_rate,
        "tail_mean_policy_p95_max_abs": _maybe_number(tail_row.get("mean_policy_p95_max_abs")),
        "tail_max_policy_max_abs": _maybe_number(tail_row.get("max_policy_max_abs")),
        "gohome_event_recall": event_recall,
        "gohome_detected_episodes": int(_number(gohome_scan.get("detected_episodes", 0))),
        "gohome_episodes": int(_number(gohome_scan.get("episodes", 0))),
        "gohome_pre_tail_false_positive_episodes": pre_tail_episodes,
        "gohome_pre_tail_active_frames": pre_tail_frames,
        "gohome_early_false_positive_episodes": int(_number(gohome_scan.get("early_false_positive_episodes", 0))),
        "gohome_dwell_early_active_frames": int(_number(gohome_scan.get("dwell_early_active_frames", 0))),
        "gohome_mean_detection_delay_steps": _maybe_number(gohome_scan.get("mean_detection_delay_steps")),
        "gohome_mean_steps_before_t_go": _maybe_number(gohome_scan.get("mean_steps_before_t_go")),
        "tail_stop_pass": tail_stop_pass,
        "gohome_pre_tail_pass": gohome_pre_tail_pass,
        "gohome_recall_pass": gohome_recall_pass,
        "combined_offline_gate_pass": bool(tail_stop_pass and gohome_pre_tail_pass and gohome_recall_pass),
        "artifact_paths": dict(artifact_paths),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, dict):
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _first_number(*values: Any) -> float | str:
    for value in values:
        converted = _maybe_number(value)
        if converted != "":
            return converted
    return ""


def _maybe_number(value: Any) -> float | str:
    if value == "" or value is None:
        return ""
    return _number(value)


def _number(value: Any) -> float:
    return float(value)


if __name__ == "__main__":
    main()
