"""CLI for v2.0.1 real-transition session preparation and package checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from testbed.tasks.home_side_calibration import (
    capture_home_calibration_window,
    initialise_home_calibration,
)
from testbed.tasks.home_side_contract import (
    write_home_side_contract,
    write_rule_ready_contract,
)
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
        help=(
            "Create deterministic sequence/split manifests and the fixed rule-based "
            "ready contract without overwriting files."
        ),
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

    ready = subparsers.add_parser(
        "build-ready-contract",
        help="Write the field-frozen v2 swing-only ready rule contract.",
    )
    ready.add_argument("--output", type=Path, required=True)

    home = subparsers.add_parser(
        "build-home-contract",
        help="Legacy: resolve home/A/B support from accepted calibration windows.",
    )
    home.add_argument("--calibration", type=Path, required=True)
    home.add_argument("--output", type=Path, required=True)

    init_home = subparsers.add_parser(
        "init-home-calibration",
        help="Create a portable field calibration input and home-config snapshot.",
    )
    init_home.add_argument("--output", type=Path, required=True)
    init_home.add_argument("--context-version", required=True)
    init_home.add_argument("--resolved-by", required=True)
    init_home.add_argument(
        "--physical-left-qpos-sign",
        type=int,
        choices=[-1, 1],
        required=True,
    )
    init_home.add_argument("--source-config", type=Path, required=True)
    init_home.add_argument("--source-value-path", required=True)
    init_home.add_argument(
        "--expected-cameras",
        default="video4,video5,video6,video7",
        help="Comma-separated camera names required in every accepted window.",
    )

    capture_home = subparsers.add_parser(
        "capture-home-window",
        help=(
            "Read one stable home/A/B window from the local slave gateway; "
            "never sends actions."
        ),
    )
    capture_home.add_argument("--calibration", type=Path, required=True)
    capture_home.add_argument("--side", choices=["home", "A", "B"], required=True)
    capture_home.add_argument("--reference-id", required=True)
    capture_home.add_argument("--host", default="127.0.0.1")
    capture_home.add_argument("--port", type=int, default=8765)
    capture_home.add_argument("--receiver-port", type=int, default=8770)
    capture_home.add_argument("--duration-s", type=float, default=0.5)
    capture_home.add_argument("--rate-hz", type=float, default=20.0)
    capture_home.add_argument("--timeout-s", type=float, default=2.0)
    capture_home.add_argument("--jpeg-quality", type=int, default=90)
    capture_home.add_argument(
        "--confirm-visual",
        action="store_true",
        required=True,
        help="Confirm that the operator visually checked this ready pose.",
    )
    capture_home.add_argument(
        "--confirm-no-software-action-source",
        action="store_true",
        required=True,
        help="Confirm that no sender/receiver/policy action source is active.",
    )

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
        elif args.command == "build-ready-contract":
            result = write_rule_ready_contract(args.output)
        elif args.command == "build-home-contract":
            result = write_home_side_contract(
                calibration_path=args.calibration,
                output_path=args.output,
            )
        elif args.command == "init-home-calibration":
            result = initialise_home_calibration(
                output_path=args.output,
                context_version=args.context_version,
                resolved_by=args.resolved_by,
                physical_left_qpos_sign=args.physical_left_qpos_sign,
                source_config=args.source_config,
                source_value_path=args.source_value_path,
                expected_cameras=args.expected_cameras,
            )
        elif args.command == "capture-home-window":
            result = capture_home_calibration_window(
                calibration_path=args.calibration,
                side=args.side,
                reference_id=args.reference_id,
                confirm_visual=args.confirm_visual,
                confirm_no_software_action_source=(
                    args.confirm_no_software_action_source
                ),
                host=args.host,
                port=args.port,
                receiver_port=args.receiver_port,
                duration_s=args.duration_s,
                rate_hz=args.rate_hz,
                timeout_s=args.timeout_s,
                jpeg_quality=args.jpeg_quality,
            )
        else:  # pragma: no cover - argparse enforces the choices.
            parser.error(f"unsupported command {args.command!r}")
            return
    except (OSError, TransitionContractError) as exc:
        _print_json({"status": "FAIL", "error": str(exc)})
        raise SystemExit(2) from exc
    _print_json(result)


def _print_json(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
