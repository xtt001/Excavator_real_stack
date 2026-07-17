"""Compare two state-hold liveness JSONL reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from testbed.policies.state_hold_comparison import compare_state_hold_files


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.offline_state_hold_compare",
        description=(
            "Strictly compare matched reference and candidate state-hold anchors."
        ),
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = compare_state_hold_files(
        reference_path=args.reference,
        candidate_path=args.candidate,
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
        output_dir=args.output_dir,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
