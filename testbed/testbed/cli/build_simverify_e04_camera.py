"""Build the SimVerify E04 camera-counterfactual package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.e04_camera_counterfactual import (
    build_e04_camera_counterfactual,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-e04-camera")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--m2-root", type=Path, required=True)
    parser.add_argument("--b1-bundle-root", type=Path, required=True)
    parser.add_argument("--previous-g5-root", type=Path, required=True)
    parser.add_argument("--contract-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = build_e04_camera_counterfactual(
        repo_root=args.repo_root,
        output_root=args.output_root,
        m0_root=args.m0_root,
        m2_root=args.m2_root,
        b1_bundle_root=args.b1_bundle_root,
        previous_g5_root=args.previous_g5_root,
        contract_path=args.contract_path,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
