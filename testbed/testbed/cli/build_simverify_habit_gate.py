"""Freeze the expert-habit offline gate and decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.habit_gate import build_habit_gate


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-habit-gate")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_727)
    args = parser.parse_args()
    result = build_habit_gate(
        repo_root=args.repo_root,
        replay_root=args.replay_root,
        output_root=args.output_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
