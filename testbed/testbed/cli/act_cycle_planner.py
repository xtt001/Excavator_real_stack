"""Generate a goal-only A/B cycle schedule for a conditioned ACT policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from testbed.tasks.act_cycle_planner import (
    ABCyclePlanner,
    CyclePlannerError,
    ScriptCyclePlanner,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-act-cycle-planner",
        description=(
            "Generate an auditable A/B goal schedule. The planner emits only "
            "real_transition_condition_v1; ACT remains the action owner."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--pattern",
        default=None,
        help=(
            "Compact full side path including the initial ready side, e.g. ABBABABA. "
            "Separators such as A->B->B are accepted."
        ),
    )
    source.add_argument(
        "--script",
        type=Path,
        default=None,
        help=(
            "JSON/YAML variable-length script with initial_side and a steps "
            "list. Each step may be a side string or a target_side mapping."
        ),
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help=(
            "Number of goals to emit. Defaults to one pass (len(pattern)-1). "
            "Set this to repeat the path when --loop is enabled."
        ),
    )
    parser.add_argument(
        "--loop",
        dest="loop",
        action="store_true",
        default=None,
        help="Repeat a script after its last explicit step.",
    )
    parser.add_argument(
        "--no-loop",
        dest="loop",
        action="store_false",
        help="Stop after one pass of the pattern/script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output planner manifest; existing files are never overwritten.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing an existing planner manifest.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help=(
            "Optional ACT bundle to validate. Its resolved config must include "
            "real_transition_condition_v1 in policy.low_dim_keys."
        ),
    )
    args = parser.parse_args()

    try:
        if args.bundle_dir is not None:
            _validate_bundle(args.bundle_dir)
        if args.script is not None:
            planner = ScriptCyclePlanner.from_script(
                args.script,
                loop=args.loop,
                max_cycles=args.cycles,
            )
        else:
            planner = ABCyclePlanner(
                args.pattern or "ABBABABA",
                loop=True if args.loop is None else bool(args.loop),
                max_cycles=args.cycles,
            )
        planner.write_manifest(args.output, overwrite=bool(args.overwrite))
    except (CyclePlannerError, FileExistsError, OSError, ValueError) as exc:
        _print_json({"status": "FAIL", "error": str(exc)})
        raise SystemExit(2) from exc

    result = planner.manifest()
    result["status"] = "OK"
    result["output"] = str(args.output)
    _print_json(result)


def _validate_bundle(bundle_dir: Path) -> None:
    bundle = Path(bundle_dir)
    required = ("policy_best.ckpt", "dataset_stats.pkl", "resolved_config.yaml")
    missing = [name for name in required if not (bundle / name).is_file()]
    if missing:
        raise ValueError(
            f"ACT bundle is missing required file(s): {', '.join(missing)}"
        )
    resolved = yaml.safe_load(
        (bundle / "resolved_config.yaml").read_text(encoding="utf-8")
    ) or {}
    keys = list((resolved.get("policy", {}) or {}).get("low_dim_keys", ()))
    if "real_transition_condition_v1" not in keys:
        raise ValueError(
            "ACT bundle policy.low_dim_keys does not include "
            "real_transition_condition_v1"
        )


def _print_json(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
