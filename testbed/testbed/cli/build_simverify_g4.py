"""Build immutable source-episode calibration evidence for SimVerify G4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.m3_condition_gate import (
    build_g4_condition_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-g4")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--b1-replay-root",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--b2-replay-root", type=Path, required=True)
    parser.add_argument("--g3-root", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--m2-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_725)
    args = parser.parse_args()
    result = build_g4_condition_calibration(
        repo_root=args.repo_root,
        output_root=args.output_root,
        b1_replay_roots=args.b1_replay_root,
        b2_replay_root=args.b2_replay_root,
        g3_root=args.g3_root,
        m0_root=args.m0_root,
        m2_root=args.m2_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
