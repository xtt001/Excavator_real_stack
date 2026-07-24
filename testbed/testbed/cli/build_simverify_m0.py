"""Build the immutable observable-only SimVerify M0 data package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.pipeline import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RESNET18_SHA256,
    DEFAULT_RESNET18_WEIGHTS,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_SPLIT_SEED,
    run_m0_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-build-simverify-m0",
        description=(
            "Build the immutable recorded-observation/offline SimVerify M0 "
            "package. This command does not train or invoke ACT."
        ),
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=DEFAULT_RESNET18_WEIGHTS,
    )
    parser.add_argument(
        "--expected-weights-sha256",
        default=DEFAULT_RESNET18_SHA256,
    )
    parser.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=256)
    parser.add_argument("--feature-device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    result = run_m0_pipeline(
        source_root=args.source_root,
        output_root=args.output_root,
        repo_root=args.repo_root,
        weights_path=args.weights_path,
        expected_weights_sha256=args.expected_weights_sha256,
        split_seed=args.split_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        feature_device=args.feature_device,
        feature_batch_size=args.feature_batch_size,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
