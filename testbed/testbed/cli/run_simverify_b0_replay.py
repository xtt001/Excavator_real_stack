"""Run one immutable B0 recorded-observation replay repetition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.m3_replay import run_b0_replay


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-run-simverify-b0-replay")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        required=True,
    )
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument("--bundle-root", type=Path, default=None)
    args = parser.parse_args()
    kwargs = {
        "repo_root": args.repo_root,
        "output_root": args.output_root,
        "split_name": args.split,
        "repeat_id": args.repeat_id,
    }
    if args.bundle_root is not None:
        kwargs["bundle_root"] = args.bundle_root
    print(json.dumps(run_b0_replay(**kwargs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
