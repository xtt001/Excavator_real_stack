"""Run the matched B0/B1/B2 expert-habit validation replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.habit_cycle_eval import run_habit_validation_replay


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-run-simverify-habit-validation")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--b0-bundle", type=Path, required=True)
    parser.add_argument("--b1-bundle", type=Path, required=True)
    parser.add_argument("--b2-bundle", type=Path, required=True)
    parser.add_argument(
        "--event-envelope",
        type=Path,
        default=Path(
            "/data/pingfan/Excavator_real_stack_data/"
            "sim_observable_cycle_v3_m2_contract_v1/"
            "expert_event_envelope_v1.json"
        ),
    )
    args = parser.parse_args()
    result = run_habit_validation_replay(
        repo_root=args.repo_root,
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        event_envelope_path=args.event_envelope,
        bundle_roots={
            "B0": args.b0_bundle,
            "B1": args.b1_bundle,
            "B2": args.b2_bundle,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
