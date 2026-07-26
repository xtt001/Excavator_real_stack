"""Build the SimVerify G4-v3.1 delta-stitch Gate audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.m3_transition_delta_stitch_audit import (
    build_transition_delta_stitch_gate_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-build-simverify-transition-delta-stitch-audit"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--delta-stitch-root", type=Path, required=True)
    parser.add_argument("--contract-path", type=Path, required=True)
    args = parser.parse_args()
    result = build_transition_delta_stitch_gate_audit(
        repo_root=args.repo_root,
        output_root=args.output_root,
        delta_stitch_root=args.delta_stitch_root,
        contract_path=args.contract_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
