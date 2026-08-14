"""CLI for v2.0.1 real-transition session preparation and package checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from testbed.tasks.home_side_contract import write_home_side_contract
from testbed.tasks.real_transition import (
    TransitionContractError,
    prepare_session_directory,
    verify_run_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-real-transition",
        description=(
            "Prepare immutable balanced multi-sequence recording plans and verify "
            "sealed real-transition run packages."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-session",
        help="Create deterministic sequence/split manifests without overwriting files.",
    )
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--session-id", required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument(
        "--created-at-utc",
        default=None,
        help="Optional fixed ISO timestamp for reproducible tests/audits.",
    )

    verify = subparsers.add_parser(
        "verify-run",
        help="Verify checksums, event order, and exact HDF5 step/time alignment.",
    )
    verify.add_argument("run_dir", type=Path)

    home = subparsers.add_parser(
        "build-home-contract",
        help="Resolve and freeze home/A/B support from accepted field windows.",
    )
    home.add_argument("--calibration", type=Path, required=True)
    home.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "prepare-session":
            result = prepare_session_directory(
                output_root=args.output_root,
                session_id=args.session_id,
                seed=args.seed,
                created_at_utc=args.created_at_utc,
            )
        elif args.command == "verify-run":
            result = verify_run_package(args.run_dir)
        elif args.command == "build-home-contract":
            result = write_home_side_contract(
                calibration_path=args.calibration,
                output_path=args.output,
            )
        else:  # pragma: no cover - argparse enforces the choices.
            parser.error(f"unsupported command {args.command!r}")
            return
    except TransitionContractError as exc:
        _print_json({"status": "FAIL", "error": str(exc)})
        raise SystemExit(2) from exc
    _print_json(result)


def _print_json(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
