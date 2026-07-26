"""Build the SimVerify G5 core continuous two-cycle replay artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.g5_two_cycle_replay import build_g5_two_cycle_replay


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-g5-two-cycle")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--m2-root", type=Path, required=True)
    parser.add_argument("--b1-bundle-root", type=Path, required=True)
    parser.add_argument("--b2-bundle-root", type=Path, required=True)
    parser.add_argument("--contract-path", type=Path, required=True)
    parser.add_argument("--previous-g5-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = build_g5_two_cycle_replay(
        repo_root=args.repo_root,
        output_root=args.output_root,
        m0_root=args.m0_root,
        m2_root=args.m2_root,
        b1_bundle_root=args.b1_bundle_root,
        b2_bundle_root=args.b2_bundle_root,
        contract_path=args.contract_path,
        previous_g5_root=args.previous_g5_root,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
