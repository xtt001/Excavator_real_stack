"""Run dense trajectory-wide deadzone liveness evaluation on replay actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from testbed.policies.trajectory_deadzone_liveness import (
    DEFAULT_HORIZONS,
    DEFAULT_PERSIST_STEPS,
    FORBIDDEN_HELDOUT,
    aggregate_trajectory_liveness,
    apply_mechanical_deadzone_assist,
    evaluate_trajectory_liveness,
    find_episode_action_files,
    load_deadzone_thresholds,
    parse_eval_spec,
    sha256_file,
    write_csv,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.trajectory_deadzone_liveness",
        description=(
            "Evaluate every expert-effective trajectory frame over multiple "
            "future deadzone-crossing horizons."
        ),
    )
    parser.add_argument(
        "--eval",
        dest="eval_specs",
        action="append",
        required=True,
        help="Replay directory in MODEL=DIR form; repeat for model comparisons.",
    )
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--horizons",
        default=",".join(str(value) for value in DEFAULT_HORIZONS),
        help="Comma-separated future horizons in control ticks.",
    )
    parser.add_argument("--persist-steps", type=int, default=DEFAULT_PERSIST_STEPS)
    parser.add_argument(
        "--include-mechanical-assist",
        action="store_true",
        help="Add a sequential mechanical-assist counterfactual for each raw model.",
    )
    parser.add_argument("--assist-trigger-fraction", type=float, default=0.5)
    parser.add_argument("--assist-min-consecutive-steps", type=int, default=2)
    parser.add_argument("--assist-margin", type=float, default=0.02)
    args = parser.parse_args()

    thresholds = load_deadzone_thresholds(args.deadzone_json)
    horizons = _parse_horizons(args.horizons)
    specs = [parse_eval_spec(value) for value in args.eval_specs]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_episode_reports: list[dict[str, Any]] = []
    all_opportunity_rows: list[dict[str, Any]] = []
    all_segment_rows: list[dict[str, Any]] = []
    all_axis_rows: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []

    for spec in specs:
        files = find_episode_action_files(spec.eval_dir, manifest=args.manifest)
        episode_ids = [episode_id for episode_id, _ in files]
        forbidden = sorted(set(episode_ids) & FORBIDDEN_HELDOUT)
        if forbidden:
            raise SystemExit(
                "held-out episode ids are forbidden for this evaluation: "
                + ", ".join(forbidden)
            )
        for episode_id, action_path in files:
            expert, policy = _load_actions(action_path)
            variants = [(str(spec.model), "raw", policy)]
            if args.include_mechanical_assist:
                assisted = apply_mechanical_deadzone_assist(
                    policy,
                    thresholds,
                    trigger_fraction=float(args.assist_trigger_fraction),
                    min_consecutive_steps=int(args.assist_min_consecutive_steps),
                    margin=float(args.assist_margin),
                )
                variants.append(
                    (f"{spec.model}+mechanical_assist", "mechanical_assist", assisted)
                )
            source_manifest.append(
                {
                    "model": str(spec.model),
                    "episode_id": episode_id,
                    "actions_path": str(action_path.resolve()),
                    "actions_sha256": sha256_file(action_path),
                    "steps": int(expert.shape[0]),
                    "variants": [variant for _, variant, _ in variants],
                }
            )
            for model, variant, action in variants:
                report = evaluate_trajectory_liveness(
                    episode_id=episode_id,
                    expert_action=expert,
                    policy_action=action,
                    thresholds=thresholds,
                    horizons=horizons,
                    persist_steps=int(args.persist_steps),
                    model=model,
                    variant=variant,
                )
                all_episode_reports.append(report)
                all_opportunity_rows.extend(report["opportunities"])
                all_segment_rows.extend(report["segments"])
                all_axis_rows.extend(report["axis_direction_summary"])

    aggregate_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for report in all_episode_reports:
        key = (
            str(report["episode_summary"]["model"]),
            str(report["episode_summary"]["variant"]),
        )
        aggregate_rows.setdefault(key, []).append(report)
    aggregates = []
    for key, reports in sorted(aggregate_rows.items()):
        aggregate = aggregate_trajectory_liveness(reports, horizons=horizons)
        aggregates.append(aggregate)

    write_csv(output_dir / "episode_summary.csv", [
        report["episode_summary"] for report in all_episode_reports
    ])
    write_csv(output_dir / "axis_direction_summary.csv", all_axis_rows)
    write_csv(output_dir / "segment_summary.csv", all_segment_rows)
    write_csv(output_dir / "opportunity_rows.csv", all_opportunity_rows)
    write_jsonl(output_dir / "opportunity_rows.jsonl", all_opportunity_rows)
    report_payload = {
        "schema_version": 1,
        "contract": "trajectory_deadzone_liveness_v1",
        "action_domain": "direct_policy_output",
        "policy_action_scale": [1.0, 1.0, 1.0, 1.0],
        "deadzone_json": str(args.deadzone_json.resolve()),
        "deadzone_json_sha256": sha256_file(args.deadzone_json),
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "manifest_sha256": sha256_file(args.manifest) if args.manifest else None,
        "horizons": list(horizons),
        "persist_steps": int(args.persist_steps),
        "mechanical_assist_counterfactual": {
            "enabled": bool(args.include_mechanical_assist),
            "trigger_fraction": float(args.assist_trigger_fraction),
            "min_consecutive_steps": int(args.assist_min_consecutive_steps),
            "margin": float(args.assist_margin),
        },
        "heldout_episode_ids": sorted(FORBIDDEN_HELDOUT),
        "heldout_evaluated": False,
        "source_manifest": source_manifest,
        "aggregates": aggregates,
        "paths": {
            "episode_summary": str((output_dir / "episode_summary.csv").resolve()),
            "axis_direction_summary": str((output_dir / "axis_direction_summary.csv").resolve()),
            "segment_summary": str((output_dir / "segment_summary.csv").resolve()),
            "opportunity_rows": str((output_dir / "opportunity_rows.csv").resolve()),
            "opportunity_rows_jsonl": str((output_dir / "opportunity_rows.jsonl").resolve()),
        },
    }
    report_path = output_dir / "trajectory_deadzone_liveness_report.json"
    report_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "aggregates": [item["aggregate"] for item in aggregates]}, ensure_ascii=False))


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(sorted({int(item.strip()) for item in str(value).split(",") if item.strip()}))
    except ValueError as exc:
        raise SystemExit(f"invalid --horizons: {value}") from exc
    if not horizons or any(item < 1 for item in horizons):
        raise SystemExit("--horizons must contain positive integers")
    return horizons


def _load_actions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "expert_action" not in data or "policy_action" not in data:
            raise SystemExit(f"{path} must contain expert_action and policy_action")
        expert = np.asarray(data["expert_action"], dtype=np.float32)
        policy = np.asarray(data["policy_action"], dtype=np.float32)
    if expert.shape != policy.shape:
        raise SystemExit(f"{path}: expert/policy shape mismatch {expert.shape} vs {policy.shape}")
    return expert, policy


if __name__ == "__main__":
    main()
