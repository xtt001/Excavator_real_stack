"""Offline counterfactual command-response feasibility report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from testbed.data.execution_feedback import load_split_episode_ids, sha256_file
from testbed.policies.counterfactual_response import (
    HELDOUT_EPISODES,
    aggregate_counterfactual_results,
    build_response_profile,
    simulate_state_hold_file,
)

SCHEMA_VERSION = 1


def run_counterfactual_response_simulation(
    *,
    response_dir: str | Path,
    split_path: str | Path,
    state_hold_specs: list[str],
    output_dir: str | Path,
    response_horizon: int = 1,
) -> dict[str, Any]:
    """Build a train-only response profile and replay explicit state-hold files."""

    response_root = Path(response_dir).expanduser().resolve()
    split = Path(split_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    manifest_path = response_root / "execution_response_manifest.json"
    events_path = response_root / "execution_response_events.jsonl"
    if not manifest_path.is_file() or not events_path.is_file():
        raise FileNotFoundError(
            f"response sidecar requires {manifest_path} and {events_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    train_ids, val_ids = load_split_episode_ids(split)
    if set(train_ids).intersection(HELDOUT_EPISODES) or set(val_ids).intersection(
        HELDOUT_EPISODES
    ):
        raise ValueError("formal split illegally contains held-out episode 105..109")

    train_id_set = set(train_ids)
    all_event_rows = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    train_event_rows = [
        row for row in all_event_rows if int(row["episode_id"]) in train_id_set
    ]
    profile = build_response_profile(
        train_event_rows,
        horizons=manifest["response_horizons"],
    )
    specs = [_parse_spec(spec) for spec in state_hold_specs]
    if not specs:
        raise ValueError("at least one --state-hold-spec label=path is required")

    runs: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for label, source in specs:
        state_hold_path = _resolve_state_hold_path(source)
        results = simulate_state_hold_file(
            state_hold_path,
            profiles=profile,
            positive_threshold=manifest["positive_threshold"],
            negative_threshold=manifest["negative_threshold"],
            response_horizon=response_horizon,
            supported_axes=manifest["supported_axes"],
        )
        if any(int(result.episode_id.split("_")[-1]) in HELDOUT_EPISODES for result in results):
            raise ValueError(f"state-hold spec {label!r} contains held-out episode")
        aggregate = aggregate_counterfactual_results(results)
        aggregate["source_path"] = str(state_hold_path)
        aggregate["source_sha256"] = sha256_file(state_hold_path)
        runs[label] = aggregate
        all_rows.extend(
            [{"pipeline": label, **result.as_dict()} for result in results]
        )

    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "counterfactual_anchor_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    report_path = output / "counterfactual_response_report.json"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": "counterfactual_command_response_feasibility_v1",
        "assumption": {
            "instant_same_direction_response": (
                "an already-effective target command is treated as receiving "
                "a same-direction response at the next tick"
            ),
            "empirical_response_profile": (
                "response probabilities and latency are descriptive train-fold "
                "teleoperation sidecar statistics only"
            ),
            "no_action_override": True,
            "no_source_hdf5_modified": True,
            "not_hydraulic_ground_truth": True,
        },
        "response_manifest_path": str(manifest_path),
        "response_manifest_sha256": sha256_file(manifest_path),
        "response_events_path": str(events_path),
        "response_events_sha256": sha256_file(events_path),
        "split_path": str(split),
        "split_sha256": sha256_file(split),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "heldout_episode_ids": sorted(HELDOUT_EPISODES),
        "heldout_evaluated": False,
        "response_horizon_for_empirical_flag": int(response_horizon),
        "train_event_rows": len(train_event_rows),
        "all_event_rows_in_sidecar": len(all_event_rows),
        "response_profile_train_only": {
            key: value.as_dict() for key, value in sorted(profile.items())
        },
        "runs": runs,
        "anchor_rows_path": str(rows_path),
        "report_path": str(report_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate whether existing policy state-hold traces are limited by "
            "command generation or by hypothetical plant response."
        )
    )
    parser.add_argument("--response-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument(
        "--state-hold-spec",
        action="append",
        required=True,
        help="repeat label=JSONL-or-directory; only explicit artifacts are read",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response-horizon", type=int, default=1)
    args = parser.parse_args(argv)
    report = run_counterfactual_response_simulation(
        response_dir=args.response_dir,
        split_path=args.split_path,
        state_hold_specs=args.state_hold_spec,
        output_dir=args.output_dir,
        response_horizon=args.response_horizon,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _parse_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"state-hold spec must be label=path, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise ValueError(f"state-hold spec must be label=path, got {spec!r}")
    return label.strip(), Path(raw_path).expanduser().resolve()


def _resolve_state_hold_path(source: Path) -> Path:
    if source.is_dir():
        source = source / "state_hold_anchors.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"state-hold anchors not found: {source}")
    return source


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("label_contract") != "direct_command_qvel_response_v1":
        raise ValueError("unexpected response sidecar label contract")
    if manifest.get("action_domain") != "direct_policy_output":
        raise ValueError("counterfactual simulator requires direct_policy_output")
    if [float(value) for value in manifest.get("policy_action_scale", [])] != [
        1.0,
        1.0,
        1.0,
        1.0,
    ]:
        raise ValueError("counterfactual simulator requires identity policy action scale")
    for key in (
        "positive_threshold",
        "negative_threshold",
        "supported_axes",
        "response_horizons",
        "qvel_noise_provenance",
    ):
        if key not in manifest:
            raise ValueError(f"response manifest is missing {key!r}")


if __name__ == "__main__":
    main()
