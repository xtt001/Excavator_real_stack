"""Run an immutable SimVerify B1/B2 condition-swap replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.m3_condition_replay import (
    run_condition_swap_replay,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-run-simverify-condition-replay")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
    )
    args = parser.parse_args()
    result = run_condition_swap_replay(
        repo_root=args.repo_root,
        output_root=args.output_root,
        bundle_root=args.bundle_root,
        repeat_id=args.repeat_id,
        split_name=args.split,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
