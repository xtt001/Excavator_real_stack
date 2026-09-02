"""Freeze a causal automatic task-progress contract from recorded cycles."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.policies.deadzone_eval import load_deadzone_thresholds
from testbed.tasks.real_transition import sha256_file, write_immutable_text

SCHEMA = "real_transition_task_state_v2_auto_progress_contract_v1"
AXES = ("swing", "boom", "stick", "bucket")
MIN_LIVENESS_DELTA_RAD = 0.05
MIN_BUCKET_EFFECTIVE_STEPS = 5
BUCKET_RELEASE_STEPS = 2
RETURN_IDLE_STEPS = 2


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-task-state-v2-auto-progress-audit")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task-state-manifest", type=Path, required=True)
    parser.add_argument("--work-context-manifest", type=Path, required=True)
    parser.add_argument("--deadzone-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_contract(
        dataset_root=args.dataset_root,
        task_state_manifest=args.task_state_manifest,
        work_context_manifest=args.work_context_manifest,
        deadzone_contract=args.deadzone_contract,
    )
    path = write_immutable_text(
        args.output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "output": str(path),
                "sha256": sha256_file(path),
                "episode_count": payload["population"]["episode_count"],
                "status": payload["status"],
            },
            ensure_ascii=False,
        )
    )


def build_contract(
    *,
    dataset_root: Path | str,
    task_state_manifest: Path | str,
    work_context_manifest: Path | str,
    deadzone_contract: Path | str,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    task_path = Path(task_state_manifest).resolve()
    work_path = Path(work_context_manifest).resolve()
    deadzone_path = Path(deadzone_contract).resolve()
    task = _json(task_path)
    work = _json(work_path)
    task_rows = {int(row["episode_id"]): row for row in task["episodes"]}
    work_rows = {int(row["episode_id"]): row for row in work["episodes"]}
    if set(task_rows) != set(work_rows):
        raise ValueError("task-state and work-context populations differ")
    if Path(str(task.get("dataset_root", ""))).resolve() != root:
        raise ValueError("task-state manifest dataset root mismatch")
    if Path(str(work.get("dataset_root", ""))).resolve() != root:
        raise ValueError("work-context manifest dataset root mismatch")

    thresholds = load_deadzone_thresholds(deadzone_path)
    positive = np.asarray([thresholds[name]["pos"] for name in AXES])
    negative = np.asarray([thresholds[name]["neg"] for name in AXES])
    rows: list[dict[str, Any]] = []
    for episode_id in sorted(task_rows):
        task_row = task_rows[episode_id]
        work_row = work_rows[episode_id]
        episode_path = root / str(task_row["episode_path"])
        if sha256_file(episode_path) != str(task_row["episode_sha256"]):
            raise ValueError(f"episode {episode_id} SHA-256 mismatch")
        with h5py.File(episode_path, "r") as handle:
            action = np.asarray(handle["action"][()], dtype=np.float64)
            qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float64)
        work_complete = int(task_row["work_complete_row"])
        return_start = int(task_row["return_effective_segment"][0])
        outbound_end = int(work_row["outbound_segment"][1])
        bucket_start, bucket_end = [int(value) for value in work_row["bucket_segment"]]
        if work_complete != bucket_end + 1:
            raise ValueError(
                f"episode {episode_id} work boundary is not bucket segment end + 1"
            )
        bucket_runs = _runs(
            action[outbound_end + 1 : return_start, 3] >= positive[3],
            offset=outbound_end + 1,
        )
        if not bucket_runs or bucket_runs[0] != (bucket_start, bucket_end):
            raise ValueError(
                f"episode {episode_id} first post-excursion bucket run changed"
            )
        bucket_release_decision = _first_consecutive(
            action[:, 3] < positive[3],
            start=work_complete,
            stop=return_start,
            count=BUCKET_RELEASE_STEPS,
        )
        mechanically_idle = np.all(
            (action < positive.reshape(1, -1)) & (action > -negative.reshape(1, -1)),
            axis=1,
        )
        return_idle_decision = _first_consecutive(
            mechanically_idle,
            start=work_complete,
            stop=return_start,
            count=RETURN_IDLE_STEPS,
        )
        qpos_span = np.ptp(qpos[:work_complete], axis=0)
        rows.append(
            {
                "episode_id": episode_id,
                "split": str(task_row["split"]),
                "transition_type": str(task_row["transition_type"]),
                "boom_qpos_span_rad": float(qpos_span[1]),
                "stick_qpos_span_rad": float(qpos_span[2]),
                "bucket_qpos_span_rad": float(qpos_span[3]),
                "boom_liveness_pass": bool(qpos_span[1] >= MIN_LIVENESS_DELTA_RAD),
                "bucket_liveness_pass": bool(qpos_span[3] >= MIN_LIVENESS_DELTA_RAD),
                "post_excursion_bucket_run_count": len(bucket_runs),
                "first_bucket_run_is_hindsight_work_segment": True,
                "bucket_effective_steps": bucket_end - bucket_start + 1,
                "bucket_release_decision_row": bucket_release_decision,
                "return_idle_decision_row": return_idle_decision,
                "return_effective_start_row": return_start,
                "bucket_release_before_return": bool(
                    bucket_release_decision is not None
                    and bucket_release_decision < return_start
                ),
                "return_idle_before_return": bool(
                    return_idle_decision is not None
                    and return_idle_decision < return_start
                ),
                "return_idle_margin_rows": (
                    None
                    if return_idle_decision is None
                    else return_start - return_idle_decision
                ),
            }
        )

    count = len(rows)
    required_checks = {
        "boom_liveness_coverage": sum(row["boom_liveness_pass"] for row in rows),
        "bucket_liveness_coverage": sum(row["bucket_liveness_pass"] for row in rows),
        "first_bucket_run_matches_work_segment": sum(
            row["first_bucket_run_is_hindsight_work_segment"] for row in rows
        ),
        "bucket_release_detected_before_return": sum(
            row["bucket_release_before_return"] for row in rows
        ),
        "return_idle_detected_before_return": sum(
            row["return_idle_before_return"] for row in rows
        ),
    }
    if not count or any(value != count for value in required_checks.values()):
        raise ValueError(
            f"automatic progress contract lacks full coverage: {required_checks}"
        )
    bucket_lengths = np.asarray(
        [row["bucket_effective_steps"] for row in rows], dtype=np.int64
    )
    idle_margins = np.asarray(
        [row["return_idle_margin_rows"] for row in rows], dtype=np.int64
    )
    if int(bucket_lengths.min()) < MIN_BUCKET_EFFECTIVE_STEPS:
        raise ValueError("configured bucket activity gate exceeds source support")
    return {
        "schema": SCHEMA,
        "status": "DATA_CONTRACT_PASS",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "dataset_root": str(root),
            "task_state_manifest": {
                "path": str(task_path),
                "sha256": sha256_file(task_path),
            },
            "work_context_manifest": {
                "path": str(work_path),
                "sha256": sha256_file(work_path),
            },
            "deadzone_contract": {
                "path": str(deadzone_path),
                "sha256": sha256_file(deadzone_path),
            },
        },
        "population": {
            "episode_count": count,
            "split_counts": {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "validation", "locked_test")
            },
            "required_checks": required_checks,
        },
        "runtime_config": {
            "advance_source": "automatic_policy_state",
            "required_liveness_axes": ["boom", "bucket"],
            "min_liveness_qpos_delta_rad": MIN_LIVENESS_DELTA_RAD,
            "require_positive_swing_excursion": True,
            "bucket_positive_action_threshold": float(positive[3]),
            "min_bucket_effective_steps": MIN_BUCKET_EFFECTIVE_STEPS,
            "bucket_release_steps": BUCKET_RELEASE_STEPS,
            "return_idle_steps": RETURN_IDLE_STEPS,
            "positive_action_thresholds": positive.tolist(),
            "negative_action_thresholds": negative.tolist(),
        },
        "observed_support": {
            "boom_qpos_span_rad": _summary([row["boom_qpos_span_rad"] for row in rows]),
            "stick_qpos_span_rad": _summary(
                [row["stick_qpos_span_rad"] for row in rows]
            ),
            "bucket_qpos_span_rad": _summary(
                [row["bucket_qpos_span_rad"] for row in rows]
            ),
            "bucket_effective_run_steps": _summary(bucket_lengths),
            "return_idle_margin_rows": _summary(idle_margins),
            "multi_bucket_run_episode_count": sum(
                row["post_excursion_bucket_run_count"] > 1 for row in rows
            ),
        },
        "episode_rows": rows,
        "semantics": {
            "work_complete": (
                "after physical boom/bucket liveness, positive swing excursion, "
                "a sustained mechanically effective positive bucket segment, and "
                "its causal release window"
            ),
            "return_commit": (
                "after work_complete and a causal all-axis action-idle window; "
                "no operator mark and no future observation are used"
            ),
            "failure_behavior": (
                "missing liveness, excursion, bucket activity, or idle evidence "
                "leaves the cycle uncommitted until the existing review/timeout gate"
            ),
        },
        "evidence_boundary": (
            "All thresholds and ordering checks are derived from recorded expert "
            "cycles. This is a causal runtime contract audit, not physical-policy "
            "closed-loop evidence."
        ),
    }


def _runs(mask: np.ndarray, *, offset: int) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [
        (int(offset + start), int(offset + end))
        for start, end in zip(starts, ends, strict=True)
    ]


def _first_consecutive(
    mask: np.ndarray, *, start: int, stop: int, count: int
) -> int | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    width = int(count)
    for end in range(int(start) + width - 1, int(stop)):
        if bool(np.all(values[end - width + 1 : end + 1])):
            return end
    return None


def _summary(values: Any) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(array)),
        "q10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
