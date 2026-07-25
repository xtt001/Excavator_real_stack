"""Build the SimVerify G4-v2 transition-stitch expert prerequisite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.m3_transition_stitch import (
    build_transition_stitch_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-transition-stitch")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--m2-root", type=Path, required=True)
    parser.add_argument("--contract-path", type=Path, required=True)
    parser.add_argument("--feature-chunk-rows", type=int, default=256)
    parser.add_argument("--knn-query-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = build_transition_stitch_calibration(
        repo_root=args.repo_root,
        output_root=args.output_root,
        m0_root=args.m0_root,
        m2_root=args.m2_root,
        contract_path=args.contract_path,
        feature_chunk_rows=args.feature_chunk_rows,
        knn_query_batch_size=args.knn_query_batch_size,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
