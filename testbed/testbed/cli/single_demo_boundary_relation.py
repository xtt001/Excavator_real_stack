"""Build a single-demo command relation with observed-boundary context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from testbed.policies.deadzone_eval import load_deadzone_thresholds
from testbed.policies.single_demo_boundary_relation import (
    FORBIDDEN_HELDOUT,
    evaluate_single_demo_boundary_relation,
    fit_observed_boundary_proxy,
    write_single_demo_boundary_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.single_demo_boundary_relation",
        description=(
            "Report train-only observed-qpos-boundary context for commands "
            "that differ from one demo; no correctness gate is produced."
        ),
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="raw")
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    parser.add_argument("--progress-horizon", type=int, default=4)
    parser.add_argument("--boundary-margin-fraction", type=float, default=0.02)
    args = parser.parse_args()

    eval_dir = args.eval_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    split_path = args.split.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    episode_ids = sorted(
        int(path.parent.name.split("_", 1)[1])
        for path in (eval_dir / "episodes").glob("episode_*/actions.npz")
    )
    forbidden = sorted(set(episode_ids) & FORBIDDEN_HELDOUT)
    if forbidden:
        raise SystemExit(
            "held-out episode ids are forbidden: "
            + ", ".join(str(value) for value in forbidden)
        )
    split = yaml.safe_load(split_path.read_text(encoding="utf-8")) or {}
    train_ids = sorted(
        set(int(value) for value in split.get("train_ids", [])) & set(episode_ids)
    )
    if not train_ids:
        raise SystemExit("no train split episodes overlap the evaluation directory")
    thresholds = load_deadzone_thresholds(args.deadzone_json)
    boundary_proxy = fit_observed_boundary_proxy(
        dataset_dir=dataset_dir,
        train_episode_ids=train_ids,
        thresholds=thresholds,
        lower_quantile=float(args.lower_quantile),
        upper_quantile=float(args.upper_quantile),
        progress_horizon=int(args.progress_horizon),
        boundary_margin_fraction=float(args.boundary_margin_fraction),
    )
    report = evaluate_single_demo_boundary_relation(
        eval_episode_ids=episode_ids,
        dataset_dir=dataset_dir,
        eval_dir=eval_dir,
        thresholds=thresholds,
        boundary_proxy=boundary_proxy,
        model=str(args.model),
        variant=str(args.variant),
    )
    report_path = write_single_demo_boundary_report(
        output_dir=output_dir,
        report=report,
        boundary_proxy=boundary_proxy,
        source_paths={
            "deadzone_json": args.deadzone_json,
            "split": split_path,
        },
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "episodes": report["episodes"],
                "single_demo_direction_opportunities": report[
                    "single_demo_direction_opportunities"
                ],
                "demo_direction_boundary_exempt": report[
                    "demo_direction_boundary_exempt"
                ],
                "outside_single_demo_effective": report[
                    "outside_single_demo_effective"
                ],
                "outside_demo_boundary_exempt": report["outside_demo_boundary_exempt"],
                "outside_demo_nonexempt": report["outside_demo_nonexempt"],
                "physical_limit_ground_truth": report["physical_limit_ground_truth"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
