"""Build the immutable B1.4 next-condition support prerequisite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.next_condition_support import (
    build_next_condition_support,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-build-simverify-next-condition-support"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--m0-root",
        type=Path,
        default=Path(
            "/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3"
        ),
    )
    parser.add_argument(
        "--m2-root",
        type=Path,
        default=Path(
            "/data/pingfan/Excavator_real_stack_data/"
            "sim_observable_cycle_v3_m2_contract_v1"
        ),
    )
    args = parser.parse_args()
    result = build_next_condition_support(
        repo_root=args.repo_root,
        output_root=args.output_root,
        m0_root=args.m0_root,
        m2_root=args.m2_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
