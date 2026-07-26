"""Build the immutable terminal SimVerify M5 decision package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.m5_decision import build_m5_decision


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-m5")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--m1-report", type=Path, required=True)
    parser.add_argument("--m2-root", type=Path, required=True)
    parser.add_argument("--g3-root", type=Path, required=True)
    parser.add_argument(
        "--condition-gate-root",
        type=Path,
        action="append",
        required=True,
    )
    args = parser.parse_args()
    result = build_m5_decision(
        repo_root=args.repo_root,
        output_root=args.output_root,
        m0_root=args.m0_root,
        m1_report_path=args.m1_report,
        m2_root=args.m2_root,
        g3_root=args.g3_root,
        condition_gate_roots=args.condition_gate_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
