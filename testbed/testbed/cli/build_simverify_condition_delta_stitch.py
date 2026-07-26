"""Run the SimVerify B1.4/B2.4 condition delta-stitch experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.m3_condition_delta_stitch import (
    build_condition_delta_stitch_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-condition-delta-stitch")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--m2-root", type=Path, required=True)
    parser.add_argument("--transition-stitch-root", type=Path, required=True)
    parser.add_argument("--delta-stitch-audit-root", type=Path, required=True)
    parser.add_argument("--next-condition-support-root", type=Path, required=True)
    parser.add_argument("--b1-bundle-root", type=Path, required=True)
    parser.add_argument("--b2-bundle-root", type=Path, required=True)
    parser.add_argument("--contract-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = build_condition_delta_stitch_experiment(
        repo_root=args.repo_root,
        output_root=args.output_root,
        m0_root=args.m0_root,
        m2_root=args.m2_root,
        transition_stitch_root=args.transition_stitch_root,
        delta_stitch_audit_root=args.delta_stitch_audit_root,
        next_condition_support_root=args.next_condition_support_root,
        b1_bundle_root=args.b1_bundle_root,
        b2_bundle_root=args.b2_bundle_root,
        contract_path=args.contract_path,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
