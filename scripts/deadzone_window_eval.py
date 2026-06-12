#!/usr/bin/env python3
"""Compute deadzone-window effectiveness metrics from offline policy eval outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.policies.deadzone_eval import (
    DEFAULT_WINDOWS,
    aggregate_window_rows,
    build_model_comparison_rows,
    load_deadzone_thresholds,
    load_rows_for_eval,
    parse_eval_spec,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read one or more tb-offline-policy-eval output directories and "
            "compute local action-effectiveness metrics using directional deadzones."
        )
    )
    parser.add_argument(
        "--eval",
        dest="eval_specs",
        action="append",
        required=True,
        help="Model replay directory in MODEL=DIR form. Repeat for comparisons.",
    )
    parser.add_argument(
        "--deadzone-json",
        type=Path,
        default=Path("artifacts/policy_effect_eval/deadzone_hdf5_20260612/deadzone_action_hdf5_estimate.json"),
        help="Directional deadzone JSON produced by calibration or HDF5 estimation.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest used to choose and order episode actions under each eval dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for deadzone_window_summary.csv, aggregate CSV and comparison CSV.",
    )
    parser.add_argument(
        "--windows",
        default=",".join(DEFAULT_WINDOWS),
        help="Comma-separated window names. Defaults to start40,end80,longest_expert_effective_segment_gap5,full_available.",
    )
    args = parser.parse_args()

    windows = [item.strip() for item in str(args.windows).split(",") if item.strip()]
    thresholds = load_deadzone_thresholds(args.deadzone_json)
    all_rows = []
    specs = [parse_eval_spec(value) for value in args.eval_specs]
    for spec in specs:
        print(f"Loading {spec.model}: {spec.eval_dir}")
        all_rows.extend(
            load_rows_for_eval(
                model=spec.model,
                eval_dir=spec.eval_dir,
                thresholds=thresholds,
                manifest=args.manifest,
                windows=windows,
            )
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "deadzone_window_summary.csv"
    aggregate_csv = output_dir / "deadzone_window_aggregate.csv"
    comparison_csv = output_dir / "deadzone_model_comparison.csv"
    summary_json = output_dir / "deadzone_window_summary.json"

    aggregate_rows = aggregate_window_rows(all_rows)
    comparison_rows = build_model_comparison_rows(all_rows)
    write_csv(summary_csv, all_rows)
    write_csv(aggregate_csv, aggregate_rows)
    write_csv(comparison_csv, comparison_rows)
    summary_json.write_text(
        json.dumps(
            {
                "deadzone_json": str(args.deadzone_json),
                "manifest": str(args.manifest) if args.manifest else None,
                "evals": [{"model": spec.model, "eval_dir": str(spec.eval_dir)} for spec in specs],
                "windows": windows,
                "rows": all_rows,
                "aggregate": aggregate_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Summary CSV: {summary_csv}")
    print(f"Aggregate CSV: {aggregate_csv}")
    print(f"Comparison CSV: {comparison_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
