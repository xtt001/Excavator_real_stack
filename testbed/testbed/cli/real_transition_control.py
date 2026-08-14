"""Experimenter-side control client for real-transition task events."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from testbed.tasks.real_transition import TransitionContractError
from testbed.tasks.real_transition_runtime import (
    DEFAULT_TRANSITION_CONTROL_PORT,
    send_transition_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-real-transition-control",
        description="Select frozen runs and mark transition events; never sends actions.",
    )
    parser.add_argument("--host", required=True, help="Slave receiver host/IP.")
    parser.add_argument("--port", type=int, default=DEFAULT_TRANSITION_CONTROL_PORT)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status")
    start = subparsers.add_parser("start-run")
    start.add_argument("--run-id", default="")
    start.add_argument("--workface-reset-id", required=True)
    start.add_argument("--workface-action", required=True)
    start.add_argument("--soil-state", default="")
    start.add_argument("--lighting", default="")
    start.add_argument("--weather", default="")
    start.add_argument("--notes", default="")
    initial = subparsers.add_parser("initial-ready")
    initial.add_argument("--notes", default="")
    goal = subparsers.add_parser("goal")
    goal.add_argument(
        "--expected-return-swing-sign",
        type=int,
        choices=[-1, 1],
        default=None,
    )
    goal.add_argument("--notes", default="")
    dump = subparsers.add_parser("dump-end")
    dump.add_argument("--notes", default="")
    ready = subparsers.add_parser("target-ready")
    ready.add_argument("--notes", default="")
    intervention = subparsers.add_parser("intervention")
    intervention.add_argument("--reason", required=True)
    abort = subparsers.add_parser("abort")
    abort.add_argument("--reason", required=True)
    safety = subparsers.add_parser("safety-stop")
    safety.add_argument("--reason", required=True)
    args = parser.parse_args()

    try:
        if args.command == "status":
            result = _send(args, "status")
        elif args.command == "start-run":
            result = _send(
                args,
                "start-run",
                {
                    "run_id": args.run_id,
                    "field_context": {
                        "workface_reset_id": args.workface_reset_id,
                        "workface_action": args.workface_action,
                        "soil_state": args.soil_state,
                        "lighting": args.lighting,
                        "weather": args.weather,
                        "notes": args.notes,
                    },
                },
            )
            initial_side = result.get("initial_side")
            if initial_side in {"A", "B"}:
                sys.stdout.write(f"\n====== INITIAL READY SIDE {initial_side} ======\n")
                sys.stdout.flush()
        elif args.command == "goal":
            status = _send(args, "status")
            target = status.get("next_target_side")
            if target not in {"A", "B"}:
                raise TransitionContractError(
                    f"no goal can be committed while phase={status.get('phase')}"
                )
            sys.stdout.write(f"\n========== TARGET {target} ==========\n")
            sys.stdout.flush()
            result = _send(
                args,
                "commit-goal",
                {
                    "display_ack": True,
                    "expected_return_swing_sign": args.expected_return_swing_sign,
                    "notes": args.notes,
                },
            )
        elif args.command in {"initial-ready", "dump-end", "target-ready"}:
            result = _send(
                args,
                args.command,
                {"notes": args.notes},
            )
        elif args.command in {"intervention", "abort", "safety-stop"}:
            result = _send(args, args.command, {"reason": args.reason})
        else:  # pragma: no cover
            parser.error(f"unsupported command {args.command!r}")
            return
    except (OSError, TransitionContractError) as exc:
        _print_json({"status": "FAIL", "error": str(exc)})
        raise SystemExit(2) from exc
    _print_json({"status": "OK", **result})


def _send(
    args: argparse.Namespace,
    command: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return send_transition_command(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
        command=command,
        payload=payload,
    )


def _print_json(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
