"""Create or verify provenance-rich symlink views for ACT training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.data.training_composite import (
    build_training_composite,
    validate_training_composite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--spec", type=Path, help="JSON/YAML composite source spec")
    mode.add_argument("--verify", type=Path, help="Existing composite directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()

    if args.spec is not None:
        result = build_training_composite(args.spec, output_dir=args.output_dir)
    else:
        result = validate_training_composite(
            args.verify, verify_hashes=args.verify_hashes
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
