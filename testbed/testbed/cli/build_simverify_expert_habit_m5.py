"""Build the terminal expert-habit M5 evidence package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.habit_m5_decision import (
    build_expert_habit_m5_decision,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-expert-habit-m5")
    for name in (
        "repo-root",
        "output-root",
        "contract-path",
        "definition-root",
        "dataset-root",
        "b0-root",
        "b1-root",
        "b2-root",
        "validation-root",
        "offline-gate-root",
        "paired-branch-root",
        "repeat-same-root",
        "move-adjacent-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = build_expert_habit_m5_decision(
        repo_root=args.repo_root,
        output_root=args.output_root,
        contract_path=args.contract_path,
        definition_root=args.definition_root,
        dataset_root=args.dataset_root,
        b0_root=args.b0_root,
        b1_root=args.b1_root,
        b2_root=args.b2_root,
        validation_root=args.validation_root,
        offline_gate_root=args.offline_gate_root,
        paired_branch_root=args.paired_branch_root,
        repeat_same_root=args.repeat_same_root,
        move_adjacent_root=args.move_adjacent_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
