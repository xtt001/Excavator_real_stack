"""Compare saved model commands with a historical response envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from testbed.data.deadzone_intent_labels import AXIS_NAMES
from testbed.policies.response_support_eval import evaluate_policy_response_support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-jsonl", type=Path, required=True)
    parser.add_argument("--envelope-json", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="NAME=/path/to/open_loop_root",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = [
        json.loads(line)
        for line in args.events_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation_events = [event for event in events if event["split"] == "validation"]
    training_supported_directions = sorted(
        {
            direction
            for event in events
            if event["split"] == "train"
            for direction in event["single_demo_event_support_directions"]
        }
    )
    envelope = json.loads(args.envelope_json.read_text(encoding="utf-8"))
    deadzone_payload = json.loads(args.deadzone_json.read_text(encoding="utf-8"))
    deadzone = deadzone_payload["deadzone_action"]
    positive = [float(deadzone[axis]["pos"]) for axis in AXIS_NAMES]
    negative = [float(deadzone[axis]["neg"]) for axis in AXIS_NAMES]

    reports = []
    for spec in args.model:
        if "=" not in spec:
            raise ValueError("--model must use NAME=/path syntax")
        name, root_text = spec.split("=", 1)
        root = Path(root_text).expanduser().resolve()
        episode_ids = {int(event["episode_id"]) for event in validation_events}
        actions = {}
        for episode_id in episode_ids:
            path = root / "episodes" / f"episode_{episode_id}" / "actions.npz"
            with np.load(path) as payload:
                actions[episode_id] = np.asarray(
                    payload["policy_action"], dtype=np.float32
                )
        reports.append(
            evaluate_policy_response_support(
                model=name,
                events=validation_events,
                policy_actions=actions,
                positive_threshold=positive,
                negative_threshold=negative,
                envelope=envelope,
                training_supported_directions=training_supported_directions,
            )
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "policy_response_support_comparison_v3",
                "events_jsonl": str(args.events_jsonl.expanduser().resolve()),
                "envelope_json": str(args.envelope_json.expanduser().resolve()),
                "deadzone_json": str(args.deadzone_json.expanduser().resolve()),
                "sealed_test_read": False,
                "training_supported_directions": training_supported_directions,
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
