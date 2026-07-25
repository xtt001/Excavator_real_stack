"""Run the bounded SimVerify M1 import smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.import_smoke import run_m1_import_smoke


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-run-simverify-m1",
        description=(
            "Validate one train and one validation SimVerify export. "
            "This command does not invoke ACT or start training."
        ),
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--train-episode-id", type=int)
    parser.add_argument("--validation-episode-id", type=int)
    args = parser.parse_args()
    result = run_m1_import_smoke(
        args.package_root,
        output_path=args.output,
        repo_root=args.repo_root,
        train_episode_id=args.train_episode_id,
        validation_episode_id=args.validation_episode_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
