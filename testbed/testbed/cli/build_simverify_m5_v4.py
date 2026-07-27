"""Build the terminal B1.5/G4/G5.1/E04 SimVerify M5 v4 package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.m5_decision_v4 import build_m5_decision_v4


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-m5-v4")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract-path", type=Path, required=True)
    parser.add_argument("--prior-m5-root", type=Path, required=True)
    parser.add_argument("--g4-root", type=Path, required=True)
    parser.add_argument("--g5-v1-root", type=Path, required=True)
    parser.add_argument("--g5-1-root", type=Path, required=True)
    parser.add_argument("--e04-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_m5_decision_v4(
        repo_root=args.repo_root,
        output_root=args.output_root,
        contract_path=args.contract_path,
        prior_m5_root=args.prior_m5_root,
        g4_root=args.g4_root,
        g5_v1_root=args.g5_v1_root,
        g5_1_root=args.g5_1_root,
        e04_root=args.e04_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
