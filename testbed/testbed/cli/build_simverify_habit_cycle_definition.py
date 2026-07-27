"""Build the immutable expert-habit cycle definition audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.habit_cycle_audit import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_NULL_SAMPLES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RESNET18_SHA256,
    DEFAULT_RESNET18_WEIGHTS,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_SPLIT_SEED,
    run_habit_cycle_definition_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-build-simverify-habit-cycle-definition",
        description=(
            "Build a recorded-observation/offline falsification audit and "
            "candidate fixed scenarios. This command never trains a policy."
        ),
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_RESNET18_WEIGHTS)
    parser.add_argument(
        "--expected-weights-sha256",
        default=DEFAULT_RESNET18_SHA256,
    )
    parser.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument("--null-samples", type=int, default=DEFAULT_NULL_SAMPLES)
    parser.add_argument("--feature-device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    args = parser.parse_args()
    result = run_habit_cycle_definition_audit(
        source_root=args.source_root,
        output_root=args.output_root,
        repo_root=args.repo_root,
        weights_path=args.weights_path,
        expected_weights_sha256=args.expected_weights_sha256,
        split_seed=args.split_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        null_samples=args.null_samples,
        feature_device=args.feature_device,
        feature_batch_size=args.feature_batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
