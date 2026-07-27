"""Export the accepted v11 ready boundary for live AGX diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.habit_runtime_ready import (
    build_habit_runtime_ready_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-build-simverify-habit-runtime-ready"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--definition-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = build_habit_runtime_ready_calibration(
        repo_root=args.repo_root,
        definition_root=args.definition_root,
        source_root=args.source_root,
        output_root=args.output_root,
        weights_path=args.weights_path,
        expected_weights_sha256=args.expected_weights_sha256,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
