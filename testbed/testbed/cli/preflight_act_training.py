"""Validate frozen ACT experiment configs without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.policies.act.training_preflight import preflight_act_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    reports = [
        preflight_act_training_config(path, verify_hashes=args.verify_hashes)
        for path in args.config
    ]
    payload = {"schema_version": 1, "reports": reports}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
