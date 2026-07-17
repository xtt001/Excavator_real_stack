"""Build expert intent event sidecars for explicit train/validation episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.data.expert_intent_events import (
    MANIFEST_FILENAME,
    build_expert_intent_event_sidecar,
    load_episode_roles_from_split,
)
from testbed.policies.deadzone_eval import load_deadzone_thresholds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.audit_expert_intent_events",
        description=(
            "Derive immediate, near-future, and task-supported expert intent "
            "events without modifying source HDF5 files."
        ),
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--deadzone-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--support-horizon-ticks", required=True, type=int)
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--train-id", action="append", type=int)
    parser.add_argument("--val-id", action="append", type=int)
    args = parser.parse_args(argv)

    if args.split_path is not None:
        if args.train_id is not None or args.val_id is not None:
            parser.error("--split-path cannot be combined with --train-id/--val-id")
        train_ids, validation_ids = load_episode_roles_from_split(
            args.split_path,
            expected_dataset_dir=args.dataset_dir,
        )
    else:
        if not args.train_id or not args.val_id:
            parser.error(
                "provide --split-path or both --train-id and --val-id"
            )
        train_ids = args.train_id
        validation_ids = args.val_id

    manifest = build_expert_intent_event_sidecar(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        thresholds=load_deadzone_thresholds(args.deadzone_json),
        threshold_source_path=args.deadzone_json,
        train_episode_ids=train_ids,
        validation_episode_ids=validation_ids,
        support_horizon_ticks=args.support_horizon_ticks,
        split_path=args.split_path,
    )
    print(
        json.dumps(
            {
                "manifest": str(Path(args.output_dir).resolve() / MANIFEST_FILENAME),
                "episodes": len(manifest["episode_ids"]),
                "events": manifest["event_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
