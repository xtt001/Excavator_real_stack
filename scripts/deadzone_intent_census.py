#!/usr/bin/env python3
"""Census expert action deadzone intent over HDF5 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from testbed.policies.deadzone_eval import (
    aggregate_intent_census_rows,
    compute_intent_census_row,
    load_deadzone_thresholds,
    write_csv,
)
from testbed.policies.offline_eval import load_train_ready_episode_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count should-move vs should-stop expert frames using directional deadzones."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    thresholds = load_deadzone_thresholds(args.deadzone_json)
    rows = []
    for episode_id in load_train_ready_episode_ids(args.manifest):
        episode_path = args.dataset_dir / f"{episode_id}.hdf5"
        if not episode_path.exists():
            raise FileNotFoundError(f"missing train-ready episode: {episode_path}")
        with h5py.File(episode_path, "r") as f:
            actions = np.asarray(f["action"][()], dtype=np.float32)
        rows.append(
            compute_intent_census_row(
                episode_id=episode_id,
                action=actions,
                thresholds=thresholds,
            )
        )

    aggregate = aggregate_intent_census_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_episode_csv = args.output_dir / "deadzone_intent_census_by_episode.csv"
    summary_csv = args.output_dir / "deadzone_intent_census_summary.csv"
    summary_json = args.output_dir / "deadzone_intent_census_summary.json"
    write_csv(by_episode_csv, rows)
    write_csv(summary_csv, aggregate)
    summary_json.write_text(
        json.dumps(
            {
                "dataset_dir": str(args.dataset_dir),
                "manifest": str(args.manifest),
                "deadzone_json": str(args.deadzone_json),
                "by_episode": rows,
                "aggregate": aggregate,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"By episode: {by_episode_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
