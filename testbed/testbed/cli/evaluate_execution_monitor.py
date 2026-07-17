"""Evaluate the causal execution monitor on an explicit train/val split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from testbed.data.execution_feedback import load_split_episode_ids, sha256_file
from testbed.policies.execution_monitor_eval import (
    aggregate_monitor_summaries,
    evaluate_response_sidecar,
)

MONITOR_EVAL_SCHEMA_VERSION = 1
HELDOUT_EPISODES = (105, 106, 107, 108, 109)
RESPONSE_WINDOW_TICKS = 20


def evaluate_execution_monitor_sidecars(
    *,
    response_dir: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Replay only the declared train/validation episodes.

    The response sidecar manifest may cover more episodes than the formal
    policy split.  Those extra records are reported as excluded and never
    enter aggregation.  No threshold or retry decision is fitted here.
    """

    response_root = Path(response_dir).expanduser().resolve()
    split = Path(split_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    manifest_path = response_root / "execution_response_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"response manifest not found: {manifest_path}")
    if not split.is_file():
        raise FileNotFoundError(f"split file not found: {split}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_response_manifest(manifest)
    train_ids, val_ids = load_split_episode_ids(split)
    selected_ids = _ordered_union(train_ids, val_ids)
    if set(selected_ids).intersection(HELDOUT_EPISODES):
        raise ValueError("formal split illegally contains held-out episode 105..109")
    manifest_ids = [int(value) for value in manifest.get("episode_ids", [])]
    missing = sorted(set(selected_ids).difference(manifest_ids))
    if missing:
        raise ValueError(f"response sidecar manifest is missing split IDs: {missing}")
    records = {
        int(record["episode_id"]): record
        for record in manifest.get("episodes", [])
        if isinstance(record, dict) and "episode_id" in record
    }
    missing_records = sorted(set(selected_ids).difference(records))
    if missing_records:
        raise ValueError(f"response sidecar records are missing IDs: {missing_records}")
    horizons = [int(value) for value in manifest["response_horizons"]]
    if RESPONSE_WINDOW_TICKS not in horizons:
        raise ValueError(
            f"response manifest does not contain locked {RESPONSE_WINDOW_TICKS}-tick horizon"
        )
    response_horizon_index = horizons.index(RESPONSE_WINDOW_TICKS)
    positive = [float(value) for value in manifest["positive_threshold"]]
    negative = [float(value) for value in manifest["negative_threshold"]]
    qvel_noise = [float(value) for value in manifest["qvel_noise"]]
    memberships = {episode_id: "train" for episode_id in train_ids}
    memberships.update({episode_id: "val" for episode_id in val_ids})

    summaries = []
    for episode_id in selected_ids:
        record = records[episode_id]
        sidecar_path = Path(record["sidecar_path"]).expanduser().resolve()
        summaries.append(
            evaluate_response_sidecar(
                sidecar_path=sidecar_path,
                episode_id=episode_id,
                split=memberships[episode_id],
                positive_threshold=positive,
                negative_threshold=negative,
                qvel_response_threshold=qvel_noise,
                response_window_ticks=RESPONSE_WINDOW_TICKS,
                response_horizon_index=response_horizon_index,
                supported_axes=manifest["supported_axes"],
            )
        )
    train_summaries = [summary for summary in summaries if summary.split == "train"]
    val_summaries = [summary for summary in summaries if summary.split == "val"]
    report: dict[str, Any] = {
        "schema_version": MONITOR_EVAL_SCHEMA_VERSION,
        "evaluation_contract": "causal_execution_monitor_response_replay_v1",
        "response_manifest_path": str(manifest_path),
        "response_manifest_sha256": sha256_file(manifest_path),
        "split_path": str(split),
        "split_sha256": sha256_file(split),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "selected_episode_ids": selected_ids,
        "excluded_response_manifest_episode_ids": sorted(
            set(manifest_ids).difference(selected_ids)
        ),
        "heldout_episode_ids": list(HELDOUT_EPISODES),
        "action_domain": manifest["action_domain"],
        "policy_action_scale": manifest["policy_action_scale"],
        "supported_axes": manifest["supported_axes"],
        "response_window_ticks": RESPONSE_WINDOW_TICKS,
        "response_horizons": horizons,
        "positive_threshold": positive,
        "negative_threshold": negative,
        "qvel_response_threshold": qvel_noise,
        "qvel_noise_provenance": manifest["qvel_noise_provenance"],
        "implementation_sha256": {
            "execution_monitor.py": sha256_file(
                Path(__file__).resolve().parents[1]
                / "policies"
                / "execution_monitor.py"
            ),
            "execution_monitor_eval.py": sha256_file(
                Path(__file__).resolve().parents[1]
                / "policies"
                / "execution_monitor_eval.py"
            ),
            "evaluate_execution_monitor.py": sha256_file(Path(__file__).resolve()),
        },
        "train": aggregate_monitor_summaries(train_summaries),
        "val": aggregate_monitor_summaries(val_summaries),
        "all_selected": aggregate_monitor_summaries(summaries),
        "episodes": [summary.as_dict() for summary in summaries],
        "source_sidecars_unchanged": True,
        "retry_policy_selected": False,
        "retry_policy_selection_reason": (
            "response sidecar has no policy-on intent/operator-correction labels; "
            "no retry threshold or candidate was fitted"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "execution_monitor_eval.json"
    report["report_path"] = str(report_path)
    _write_json_atomic(report_path, report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay post-command execution monitoring on existing response "
            "sidecars using an explicit train/validation split."
        )
    )
    parser.add_argument("--response-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_execution_monitor_sidecars(
        response_dir=args.response_dir,
        split_path=args.split_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _validate_response_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("label_contract") != "direct_command_qvel_response_v1":
        raise ValueError("unexpected response sidecar label contract")
    if manifest.get("action_domain") != "direct_policy_output":
        raise ValueError("monitor evaluation requires direct_policy_output domain")
    scale = [float(value) for value in manifest.get("policy_action_scale", [])]
    if scale != [1.0, 1.0, 1.0, 1.0]:
        raise ValueError("monitor evaluation requires identity policy action scale")
    for key in (
        "positive_threshold",
        "negative_threshold",
        "qvel_noise",
        "qvel_noise_provenance",
        "supported_axes",
        "response_horizons",
        "episodes",
    ):
        if key not in manifest:
            raise ValueError(f"response sidecar manifest is missing {key!r}")


def _ordered_union(first: list[int], second: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for episode_id in [*first, *second]:
        if episode_id not in seen:
            seen.add(episode_id)
            result.append(episode_id)
    return result


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
