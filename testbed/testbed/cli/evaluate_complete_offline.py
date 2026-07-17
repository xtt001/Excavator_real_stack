"""[Execution target G28/H1: Complete offline candidate audit]

Join independent offline diagnostics without turning one held-out demo into a
unique answer: open-loop windows, state-hold target reproduction, release/tail,
gohome label availability, and causal execution-monitor replay.  This command
never reads held-out episodes unless a caller explicitly supplies them; the
default report records 105..109 as forbidden and marks them unevaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from testbed.policies.deadzone_eval import (
    aggregate_window_rows,
    effective_direction_mask,
    load_deadzone_thresholds,
    load_rows_for_eval,
)

AXIS_NAMES = ("swing", "boom", "stick", "bucket")
FORBIDDEN_HELDOUT = (105, 106, 107, 108, 109)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_id_number(value: str) -> int:
    return int(str(value).split("_")[-1])


def _release_tail_metrics(
    *,
    open_loop_dir: Path,
    thresholds: dict[str, dict[str, float]],
    tail_steps: int = 80,
    release_window: int = 5,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((open_loop_dir / "episodes").glob("episode_*/actions.npz")):
        episode_id = path.parent.name
        with np.load(path) as data:
            expert = np.asarray(data["expert_action"], dtype=np.float32)
            policy = np.asarray(data["policy_action"], dtype=np.float32)
        expert_effective = effective_direction_mask(expert, thresholds)
        policy_effective = effective_direction_mask(policy, thresholds)
        expert_any = expert_effective.any(axis=(1, 2))
        policy_any = policy_effective.any(axis=(1, 2))
        tail_start = max(0, len(expert) - int(tail_steps))
        tail_outside_demo = int(
            np.count_nonzero(policy_any[tail_start:] & ~expert_any[tail_start:])
        )
        release_outside_demo = 0
        release_windows = 0
        for index in range(max(0, len(expert) - 1)):
            if not expert_any[index] or expert_any[index + 1]:
                continue
            end = min(len(expert), index + 1 + int(release_window))
            release_windows += 1
            release_outside_demo += int(
                np.count_nonzero(
                    policy_any[index + 1 : end] & ~expert_any[index + 1 : end]
                )
            )
        rows.append(
            {
                "episode_id": episode_id,
                "steps": int(len(expert)),
                "tail_start": int(tail_start),
                "tail_outside_single_demo_effective_frames": tail_outside_demo,
                "release_windows": release_windows,
                "release_outside_single_demo_effective_frames": (release_outside_demo),
                "policy_clip_violations": int(
                    np.count_nonzero(np.abs(policy) > 1.0 + 1.0e-6)
                ),
                "policy_nonfinite": int(np.count_nonzero(~np.isfinite(policy))),
                "policy_max_active_effective_axes": int(
                    policy_effective.any(axis=2).sum(axis=1).max(initial=0)
                ),
            }
        )
    return {
        "episode_count": len(rows),
        "rows": rows,
        "tail_outside_single_demo_effective_frames": int(
            sum(row["tail_outside_single_demo_effective_frames"] for row in rows)
        ),
        "release_windows": int(sum(row["release_windows"] for row in rows)),
        "release_outside_single_demo_effective_frames": int(
            sum(row["release_outside_single_demo_effective_frames"] for row in rows)
        ),
        "policy_clip_violations": int(
            sum(row["policy_clip_violations"] for row in rows)
        ),
        "policy_nonfinite": int(sum(row["policy_nonfinite"] for row in rows)),
        "max_active_effective_axes": int(
            max((row["policy_max_active_effective_axes"] for row in rows), default=0)
        ),
    }


def _gohome_estimability(dataset_dir: Path, episode_ids: list[str]) -> dict[str, Any]:
    eligible_paths = 0
    tail_paths = 0
    excluded_go_home = 0
    for episode_id in episode_ids:
        path = dataset_dir / f"{episode_id}.hdf5"
        with h5py.File(path, "r") as handle:
            if "handoff/gohome_eligible_label" in handle:
                eligible_paths += 1
            if "handoff/tail_idle_mask" in handle:
                tail_paths += 1
            metadata = dict(handle["metadata"].attrs) if "metadata" in handle else {}
            if bool(metadata.get("excluded_go_home", False)):
                excluded_go_home += 1
    estimable = eligible_paths == len(episode_ids) and tail_paths == len(episode_ids)
    return {
        "estimable": bool(estimable),
        "reason": (
            "all requested episodes contain handoff labels"
            if estimable
            else "formal train-ready HDF5 has no complete gohome/tail handoff labels; false positives are not estimable"
        ),
        "episodes": len(episode_ids),
        "episodes_with_gohome_eligible_label": eligible_paths,
        "episodes_with_tail_idle_mask": tail_paths,
        "episodes_excluded_go_home": excluded_go_home,
    }


def _state_hold_summary(run_summary_path: Path) -> dict[str, Any]:
    payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
    reports = {}
    for report in payload.get("reports", []):
        mode = str(report.get("mode", ""))
        summary_path = Path(str(report["paths"]["summary"]))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        overall = next(
            row for row in summary.get("aggregate", []) if row.get("group") == "overall"
        )
        reports[mode] = {
            "anchors": int(overall["anchors_total"]),
            "demo_target_reproduced": int(
                overall["state_hold_demo_target_reproduced_anchors"]
            ),
            "demo_target_not_reproduced": int(
                overall["state_hold_demo_target_not_reproduced_anchors"]
            ),
            "demo_target_reproduction_hidden": int(
                overall.get(
                    "demo_target_reproduction_hidden_by_teacher_forcing_anchors",
                    0,
                )
            ),
            "anchor_extra_effective_anchors": int(
                overall.get("state_hold_anchor_extra_effective_anchors", 0)
            ),
            "anchor_extra_effective_ticks": int(
                overall.get("state_hold_anchor_extra_effective_ticks", 0)
            ),
            "opposite_to_demo_target_ticks": int(
                overall.get("state_hold_opposite_to_demo_target_ticks", 0)
            ),
            "flips": int(overall.get("state_hold_direction_flips", 0)),
        }
    return {
        "candidate_id": payload.get("candidate_id"),
        "pipeline_mode": payload.get("pipeline_mode"),
        "episode_ids": payload.get("episode_ids", []),
        "reports": reports,
    }


def build_report(
    *,
    open_loop_dir: Path,
    state_hold_run_summary: Path,
    execution_monitor_json: Path,
    deadzone_json: Path,
    dataset_dir: Path,
    output_dir: Path,
    model_label: str,
    split_path: Path | None = None,
    fixed_qpos_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    thresholds = load_deadzone_thresholds(deadzone_json)
    rows = load_rows_for_eval(
        model=model_label,
        eval_dir=open_loop_dir,
        thresholds=thresholds,
    )
    aggregate = aggregate_window_rows(rows)
    episode_ids = sorted(
        (
            path.parent.name
            for path in (open_loop_dir / "episodes").glob("episode_*/actions.npz")
        ),
        key=_episode_id_number,
    )
    report: dict[str, Any] = {
        "schema_version": 2,
        "contract": "complete_offline_evidence_audit_v2",
        "model_label": model_label,
        "action_domain": "direct_policy_output",
        "policy_action_scale": [1.0, 1.0, 1.0, 1.0],
        "deadzone_json": str(deadzone_json),
        "deadzone_json_sha256": sha256_file(deadzone_json),
        "dataset_dir": str(dataset_dir),
        "episode_ids": episode_ids,
        "forbidden_heldout_episode_ids": list(FORBIDDEN_HELDOUT),
        "heldout_evaluated": False,
        "open_loop": {
            "source_dir": str(open_loop_dir),
            "window_rows": rows,
            "window_aggregate": aggregate,
        },
        "release_tail_single_demo_relation": _release_tail_metrics(
            open_loop_dir=open_loop_dir,
            thresholds=thresholds,
        ),
        "gohome": _gohome_estimability(dataset_dir, episode_ids),
        "state_hold": _state_hold_summary(state_hold_run_summary),
        "execution_monitor": json.loads(
            execution_monitor_json.read_text(encoding="utf-8")
        ),
        "fixed_qpos_multi_fpv": [
            {
                "output_dir": str(path),
                "summary": json.loads(
                    (path / "summary.json").read_text(encoding="utf-8")
                ),
            }
            for path in (fixed_qpos_dirs or [])
        ],
        "inputs": {
            "open_loop_collection_summary_sha256": sha256_file(
                open_loop_dir / "collection_summary.json"
            ),
            "state_hold_run_summary_sha256": sha256_file(state_hold_run_summary),
            "execution_monitor_json_sha256": sha256_file(execution_monitor_json),
        },
    }
    if split_path is not None:
        report["split_path"] = str(split_path)
        report["split_sha256"] = sha256_file(split_path)
        report["split"] = yaml.safe_load(split_path.read_text(encoding="utf-8")) or {}

    assist = report["state_hold"]["reports"].get("assist_enabled", {})
    report["single_demo_reproduction_diagnostic"] = {
        "promotion_gate": False,
        "correctness_estimable": False,
        "task_support_estimable": False,
        "assist_demo_target_reproduced": assist.get("demo_target_reproduced"),
        "assist_demo_target_not_reproduced": assist.get("demo_target_not_reproduced"),
        "assist_demo_target_reproduction_hidden": assist.get(
            "demo_target_reproduction_hidden"
        ),
        "heldout_status": (
            "heldout 105..109 remains unevaluated; this report makes no promotion "
            "decision from single-demo reproduction"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "window_rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "window_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "complete_offline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.evaluate_complete_offline"
    )
    parser.add_argument("--open-loop-dir", type=Path, required=True)
    parser.add_argument("--state-hold-run-summary", type=Path, required=True)
    parser.add_argument("--execution-monitor-json", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", default="candidate")
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument(
        "--fixed-qpos-dir",
        type=Path,
        action="append",
        default=[],
        help="Optional fixed-qpos/multi-FPV replay output directory; repeatable.",
    )
    args = parser.parse_args()
    report = build_report(
        open_loop_dir=args.open_loop_dir.resolve(),
        state_hold_run_summary=args.state_hold_run_summary.resolve(),
        execution_monitor_json=args.execution_monitor_json.resolve(),
        deadzone_json=args.deadzone_json.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        model_label=str(args.model_label),
        split_path=args.split_path.resolve() if args.split_path else None,
        fixed_qpos_dirs=[path.resolve() for path in args.fixed_qpos_dir],
    )
    print(json.dumps(report["single_demo_reproduction_diagnostic"], ensure_ascii=False))


if __name__ == "__main__":
    main()
