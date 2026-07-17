"""Audit deadzone-aware idle/near/safe labels from existing HDF5 episodes.

The source HDF5 files are read-only.  This audit is deliberately separate from
the trainer so that the label census, split membership, threshold provenance,
and source hashes are reviewable before a checkpoint is trained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import yaml

from testbed.policies.act.action_state_effort import (
    AXIS_NAMES,
    compute_action_state_labels,
    resolve_action_state_effort_config,
    summarize_action_state_labels,
)

FORBIDDEN_HELDOUT = frozenset({105, 106, 107, 108, 109})


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.audit_action_state_effort",
        description=(
            "Build a read-only census of idle/near/safe action-state labels "
            "from existing direct-domain HDF5 actions."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--threshold-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", type=int, default=None)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--safe-margin", type=float, default=0.02)
    parser.add_argument("--persistence-steps", type=int, default=2)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    threshold_json = args.threshold_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_payload = json.loads(threshold_json.read_text(encoding="utf-8"))
    config = resolve_action_state_effort_config(
        {
            "enabled": True,
            "thresholds": threshold_payload.get(
                "deadzone_action", threshold_payload
            ),
            "safe_margin": float(args.safe_margin),
            "persistence_steps": int(args.persistence_steps),
        }
    )

    split_payload: dict[str, Any] | None = None
    if args.split is not None:
        split_path = args.split.expanduser().resolve()
        split_payload = yaml.safe_load(split_path.read_text(encoding="utf-8")) or {}
        if args.episode_id is None:
            episode_ids = [
                int(ep_id)
                for ep_id in split_payload.get("train_ids", [])
                + split_payload.get("val_ids", [])
            ]
        else:
            episode_ids = [int(ep_id) for ep_id in args.episode_id]
    elif args.episode_id is not None:
        episode_ids = [int(ep_id) for ep_id in args.episode_id]
    else:
        episode_ids = sorted(
            int(path.stem.split("_", 1)[1])
            for path in dataset_dir.glob("episode_*.hdf5")
        )
    episode_ids = sorted(dict.fromkeys(episode_ids))
    forbidden = sorted(set(episode_ids) & FORBIDDEN_HELDOUT)
    if forbidden:
        raise SystemExit(
            "held-out episode ids are forbidden for action-state label audit: "
            + ", ".join(str(ep_id) for ep_id in forbidden)
        )

    episodes: list[dict[str, Any]] = []
    total_counts = {
        axis: {state: 0 for state in ("idle", "pos_near", "pos_safe", "neg_near", "neg_safe")}
        for axis in AXIS_NAMES
    }
    total_steps = 0
    total_valid_axis_rows = 0
    total_persistent_events = 0
    for episode_id in episode_ids:
        path = dataset_dir / f"episode_{episode_id}.hdf5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            actions = handle["/action"][()]
        labels = compute_action_state_labels(
            actions=actions,
            thresholds=config["thresholds"],
            safe_margin=float(config["safe_margin"]),
            persistence_steps=int(config["persistence_steps"]),
        )
        summary = summarize_action_state_labels(labels)
        for axis in AXIS_NAMES:
            for state, count in summary["counts"][axis].items():
                total_counts[axis][state] += int(count)
        total_steps += int(summary["steps"])
        total_valid_axis_rows += int(summary["valid_axis_rows"])
        total_persistent_events += int(summary["persistent_effective_events"])
        episodes.append(
            {
                "episode_id": int(episode_id),
                "path": str(path),
                "sha256": _sha256(path),
                "summary": summary,
            }
        )

    split_hash = None
    if args.split is not None:
        split_hash = _sha256(args.split.expanduser().resolve())
    report = {
        "schema_version": 1,
        "contract": "action_state_effort_labels_v1",
        "action_domain": threshold_payload.get("metadata", {}).get(
            "action_domain", "direct_policy_output"
        ),
        "policy_action_scale": threshold_payload.get("metadata", {}).get(
            "policy_action_scale", [1.0, 1.0, 1.0, 1.0]
        ),
        "dataset_dir": str(dataset_dir),
        "threshold_json": str(threshold_json),
        "threshold_json_sha256": _sha256(threshold_json),
        "safe_margin": float(config["safe_margin"]),
        "persistence_steps": int(config["persistence_steps"]),
        "split": str(args.split.expanduser().resolve()) if args.split else None,
        "split_sha256": split_hash,
        "train_ids": list(split_payload.get("train_ids", []))
        if split_payload is not None
        else None,
        "val_ids": list(split_payload.get("val_ids", []))
        if split_payload is not None
        else None,
        "episode_ids": episode_ids,
        "heldout_episode_ids": sorted(FORBIDDEN_HELDOUT),
        "heldout_evaluated": False,
        "aggregate": {
            "steps": total_steps,
            "valid_axis_rows": total_valid_axis_rows,
            "persistent_effective_events": total_persistent_events,
            "counts": total_counts,
        },
        "episodes": episodes,
        "source_episodes_unchanged": True,
        "paths": {
            "report": str(output_dir / "action_state_effort_label_audit.json"),
            "episode_summary_csv": str(output_dir / "episode_summary.csv"),
            "axis_state_counts_csv": str(output_dir / "axis_state_counts.csv"),
        },
    }
    report_path = output_dir / "action_state_effort_label_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_episode_csv(output_dir / "episode_summary.csv", episodes)
    _write_axis_csv(output_dir / "axis_state_counts.csv", episodes)
    print(json.dumps({"report": str(report_path), "aggregate": report["aggregate"]}, ensure_ascii=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_episode_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    fields = (
        "episode_id",
        "steps",
        "valid_axis_rows",
        "persistent_effective_events",
        "source_sha256",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            summary = episode["summary"]
            writer.writerow(
                {
                    "episode_id": episode["episode_id"],
                    "steps": summary["steps"],
                    "valid_axis_rows": summary["valid_axis_rows"],
                    "persistent_effective_events": summary[
                        "persistent_effective_events"
                    ],
                    "source_sha256": episode["sha256"],
                }
            )


def _write_axis_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    fields = ("episode_id", "axis", "state", "count")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            for axis in AXIS_NAMES:
                for state, count in episode["summary"]["counts"][axis].items():
                    writer.writerow(
                        {
                            "episode_id": episode["episode_id"],
                            "axis": axis,
                            "state": state,
                            "count": count,
                        }
                    )


if __name__ == "__main__":
    main()
