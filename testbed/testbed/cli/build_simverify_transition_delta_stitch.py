"""Build the SimVerify G4-v3 expert delta-stitch prerequisite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.m3_transition_delta_stitch import (
    build_transition_delta_stitch_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-transition-delta-stitch")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--transition-stitch-root", type=Path, required=True)
    parser.add_argument("--contract-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    result = build_transition_delta_stitch_calibration(
        repo_root=args.repo_root,
        output_root=args.output_root,
        m0_root=args.m0_root,
        transition_stitch_root=args.transition_stitch_root,
        contract_path=args.contract_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
