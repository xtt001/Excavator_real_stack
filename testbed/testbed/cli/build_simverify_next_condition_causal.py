"""Build the support-aware B1.4 next-condition causal Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.next_condition_causal import (
    build_next_condition_causal_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-build-simverify-next-condition-causal"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--b1-replay-root",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--b2-replay-root", type=Path, required=True)
    parser.add_argument("--masked-b1-replay-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=100_000,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_726)
    args = parser.parse_args()
    result = build_next_condition_causal_gate(
        repo_root=args.repo_root,
        output_root=args.output_root,
        b1_replay_roots=args.b1_replay_root,
        b2_replay_root=args.b2_replay_root,
        masked_b1_replay_root=args.masked_b1_replay_root,
        support_root=args.support_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
