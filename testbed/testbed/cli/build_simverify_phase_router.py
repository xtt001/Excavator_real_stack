"""Build the causal observable SimVerify B1.3 phase router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.observable_phase_router import (
    build_observable_phase_router,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-phase-router")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--contract-path", type=Path, required=True)
    args = parser.parse_args()
    result = build_observable_phase_router(
        repo_root=args.repo_root,
        output_root=args.output_root,
        m0_root=args.m0_root,
        contract_path=args.contract_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

