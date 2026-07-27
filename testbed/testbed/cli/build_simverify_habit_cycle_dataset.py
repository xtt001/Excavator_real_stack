"""Build the immutable fixed-scenario ready-to-ready training dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.habit_cycle_dataset import (
    DEFAULT_DEFINITION_ROOT,
    DEFAULT_M0_ROOT,
    DEFAULT_OUTPUT_ROOT,
    build_habit_cycle_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-build-simverify-habit-cycle-dataset"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--definition-root", type=Path, default=DEFAULT_DEFINITION_ROOT)
    parser.add_argument("--m0-root", type=Path, default=DEFAULT_M0_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--action-chunk-size", type=int, default=20)
    parser.add_argument("--user-approved-freeze", action="store_true")
    args = parser.parse_args()
    result = build_habit_cycle_dataset(
        repo_root=args.repo_root,
        definition_root=args.definition_root,
        m0_root=args.m0_root,
        output_root=args.output_root,
        user_approved_freeze=args.user_approved_freeze,
        action_chunk_size=args.action_chunk_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
