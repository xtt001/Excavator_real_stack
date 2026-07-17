"""Audit expert action-start distribution without touching source HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from testbed.policies.action_start_distribution import (
    FORBIDDEN_HELDOUT,
    analyze_action_start_distribution,
    write_action_start_distribution_report,
)
from testbed.policies.deadzone_eval import load_deadzone_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.audit_action_start_distribution",
        description=(
            "Audit idle-to-effective expert transitions, persistence, and "
            "qpos/qvel-conditioned action ambiguity."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", type=int, required=True)
    parser.add_argument("--persistence-horizon", type=int, default=4)
    parser.add_argument("--pre-window", type=int, default=5)
    parser.add_argument("--ambiguity-bins", type=int, default=5)
    args = parser.parse_args()

    episode_ids = sorted({int(value) for value in args.episode_id})
    forbidden = sorted(set(episode_ids) & FORBIDDEN_HELDOUT)
    if forbidden:
        raise SystemExit(
            "held-out episode ids are forbidden: "
            + ", ".join(str(value) for value in forbidden)
        )
    split_path = args.split.expanduser().resolve()
    split = yaml.safe_load(split_path.read_text(encoding="utf-8")) or {}
    train_ids = sorted(set(int(value) for value in split.get("train_ids", [])) & set(episode_ids))
    if not train_ids:
        raise SystemExit("train split has no overlap with requested episodes")
    thresholds = load_deadzone_thresholds(args.deadzone_json)
    report = analyze_action_start_distribution(
        dataset_dir=args.dataset_dir,
        episode_ids=episode_ids,
        train_episode_ids=train_ids,
        thresholds=thresholds,
        persistence_horizon=int(args.persistence_horizon),
        pre_window=int(args.pre_window),
        ambiguity_bins=int(args.ambiguity_bins),
    )
    report_path = write_action_start_distribution_report(
        output_dir=args.output_dir,
        report=report,
        source_paths={
            "deadzone_json": args.deadzone_json,
            "split": split_path,
        },
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "episodes": len(report["episode_ids"]),
                "transitions": {
                    key: value["transition_count"]
                    for key, value in report["transition_summary"].items()
                },
                "combo_counts": report["combo_counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
