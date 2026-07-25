"""Build the immutable SimVerify M2 offline-evaluation contract package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.m2_eval import (
    DEFAULT_M0_ROOT,
    DEFAULT_M2_ROOT,
    run_m2_contract_builder,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-build-simverify-m2",
        description=(
            "Freeze observable-only M2 replay contracts and anchors. "
            "This command does not load a model or start training."
        ),
    )
    parser.add_argument("--m0-root", type=Path, default=DEFAULT_M0_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_M2_ROOT)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_m2_contract_builder(
        m0_root=args.m0_root,
        output_root=args.output_root,
        repo_root=args.repo_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
