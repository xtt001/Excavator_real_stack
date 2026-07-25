"""Build immutable source-episode calibration evidence for SimVerify G3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.m3_gate import build_g3_calibration


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-g3")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-replay-root", type=Path, required=True)
    parser.add_argument(
        "--validation-replay-root",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--m2-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_725)
    args = parser.parse_args()
    result = build_g3_calibration(
        repo_root=args.repo_root,
        output_root=args.output_root,
        train_replay_root=args.train_replay_root,
        validation_replay_roots=args.validation_replay_root,
        m2_root=args.m2_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
