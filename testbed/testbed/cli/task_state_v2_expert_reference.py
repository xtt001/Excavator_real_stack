"""Freeze raw-demonstration support for task-state-v2 acceptance metrics."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.data.action_primitive_islands import AXIS_NAMES
from testbed.policies.deadzone_eval import load_deadzone_thresholds
from testbed.tasks.real_transition import sha256_file, write_immutable_text

SCHEMA = "real_transition_task_state_v2_expert_reference_v1"
READY_SWING_QVEL_MAX = 0.015
DIAGNOSTIC_ALL_AXIS_QVEL_MAX = np.asarray(
    [0.015, 0.015, 0.020, 0.020], dtype=np.float32
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-task-state-v2-expert-reference")
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_reference(
        probe_manifest=args.probe_manifest, output_dir=args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_reference(
    *, probe_manifest: Path | str, output_dir: Path | str
) -> dict[str, Any]:
    probe_path = Path(probe_manifest).resolve()
    probe = _json(probe_path)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite expert reference: {output}")
    root = Path(str(probe["dataset_root"])).resolve()
    task_manifest_path = Path(str(probe["task_state_manifest"]["path"]))
    if sha256_file(task_manifest_path) != str(
        probe["task_state_manifest"]["sha256"]
    ):
        raise ValueError("task-state manifest changed after probe freeze")
    task_manifest = _json(task_manifest_path)
    deadzone_path = Path(str(probe["deadzone_thresholds"]["path"]))
    if sha256_file(deadzone_path) != str(probe["deadzone_thresholds"]["sha256"]):
        raise ValueError("deadzone thresholds changed after probe freeze")
    thresholds = load_deadzone_thresholds(deadzone_path)
    positive = np.asarray(
        [float(thresholds[axis]["pos"]) for axis in AXIS_NAMES], dtype=np.float32
    )
    negative = np.asarray(
        [float(thresholds[axis]["neg"]) for axis in AXIS_NAMES], dtype=np.float32
    )
    heldout_specs = [dict(row) for row in probe["population"]["cycles"]]
    heldout_rows = [
        _cycle_metrics(
            root=root,
            spec=spec,
            positive=positive,
            negative=negative,
        )
        for spec in heldout_specs
    ]
    task_rows = [dict(row) for row in task_manifest["episodes"]]
    commit_rows = [
        _commit_metrics(
            root=root,
            spec=spec,
            positive=positive,
            negative=negative,
        )
        for spec in task_rows
    ]
    heldout_ids = {int(row["episode_id"]) for row in heldout_specs}
    bta = [row for row in heldout_rows if row["transition_type"] == "B->A"]
    other = [row for row in heldout_rows if row["transition_type"] != "B->A"]
    normal = [row for row in commit_rows if row["work_complete_before_commit"]]
    heldout_normal = [row for row in normal if row["episode_id"] in heldout_ids]
    ready_bta = [row for row in commit_rows if row["episode_id"] in {r["episode_id"] for r in bta}]
    factual = {
        "heldout_all": _cycle_summary(heldout_rows),
        "heldout_b_to_a": _cycle_summary(bta),
        "heldout_other": _cycle_summary(other),
        "uncommitted_all_normal_order": _uncommitted_summary(normal),
        "uncommitted_heldout_normal_order": _uncommitted_summary(heldout_normal),
        "commit_all_cycles": _commit_summary(commit_rows),
        "b_to_a_ready_rows": _ready_summary(ready_bta),
    }
    payload = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "probe_manifest": {"path": str(probe_path), "sha256": sha256_file(probe_path)},
        "task_state_manifest": {
            "path": str(task_manifest_path),
            "sha256": sha256_file(task_manifest_path),
        },
        "deadzone_thresholds": {
            "path": str(deadzone_path),
            "sha256": sha256_file(deadzone_path),
            "positive": positive.tolist(),
            "negative_magnitude": negative.tolist(),
        },
        "qvel_interpretation": {
            "official_ready_gate": "only abs(swing_qvel) <= 0.015 rad/s",
            "official_non_swing_qvel": "record_only_unbounded",
            "diagnostic_all_axis_threshold": DIAGNOSTIC_ALL_AXIS_QVEL_MAX.tolist(),
            "near_exact_zero_threshold": 0.001,
        },
        "factual_reference": factual,
        "expert_aligned_gate_counts": {
            "b_to_a": {
                "population": 8,
                "direct_shortcut_max": 0,
                "correct_start_motion_min": 6,
                "tool_liveness_min": 6,
                "positive_excursion_min": 6,
                "bucket_liveness_min": 7,
                "return_negative_min": 7,
                "return_ready_crossing_min": 7,
                "ordered_proxy_min": 5,
            },
            "uncommitted_heldout": {
                "population": 29,
                "no_negative_swing_expert": 29,
                "no_negative_swing_min": 28,
                "full_stop_required": False,
            },
            "other_transitions": {
                "population": 22,
                "correct_start_motion_expert": 15,
                "correct_start_motion_min": 14,
                "one_cycle_tolerance": True,
            },
        },
        "diagnostic_only_metrics": [
            "all-axis full stop at return_commit",
            "all-axis qvel forced to exact zero",
            "minimum raw-chunk delta under token intervention",
            "minimum raw-chunk delta under qvel intervention",
            "counterfactual immediate return from a stopped ready observation",
        ],
        "interpretation": (
            "return_commit is a semantic permission boundary, not a full-stop "
            "label. Before commit, the factual invariant is absence of a "
            "mechanically effective negative swing; tool motion, sub-deadzone "
            "commands and residual qvel remain allowed."
        ),
        "candidate_results_used_to_set_thresholds": False,
        "physical_evidence": False,
    }
    output.mkdir(parents=True)
    result_path = write_immutable_text(
        output / "expert_reference.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(output / "heldout_expert_cycle_metrics.csv", heldout_rows)
    _write_csv(output / "commit_stop_metrics.csv", commit_rows)
    sums = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    write_immutable_text(output / "SHA256SUMS.txt", "\n".join(sums) + "\n")
    return {
        "expert_reference": str(result_path),
        "expert_reference_sha256": sha256_file(result_path),
    }


def _cycle_metrics(
    *, root: Path, spec: dict[str, Any], positive: np.ndarray, negative: np.ndarray
) -> dict[str, Any]:
    with h5py.File(root / str(spec["episode_path"]), "r") as handle:
        action = np.asarray(handle["action"][()], dtype=np.float32)
    direction = _direction(action, positive, negative)
    first_boundary = min(
        int(spec["work_complete_row"]), int(spec["return_commit_row"])
    )
    start = direction[: min(20, first_boundary)]
    outbound_start, outbound_end = (int(v) for v in spec["outbound_segment"])
    bucket_start, bucket_end = (int(v) for v in spec["bucket_segment"])
    return_start, return_end = (
        int(v) for v in spec["return_effective_segment"]
    )
    outbound_positive = bool(
        np.any(
            direction[
                outbound_start : min(outbound_end + 1, outbound_start + 20), 0
            ]
            == 1
        )
    )
    bucket_tool = bool(
        np.any(
            direction[
                bucket_start : min(bucket_end + 1, bucket_start + 20), 1:
            ]
            != 0
        )
    )
    return_negative = bool(
        np.any(
            direction[return_start : min(return_end + 1, return_start + 20), 0]
            == -1
        )
    )
    crossing_negative = None
    if spec.get("return_ready_crossing_row") is not None:
        crossing = int(spec["return_ready_crossing_row"])
        crossing_negative = bool(
            np.any(
                direction[crossing : min(return_end + 1, crossing + 20), 0]
                == -1
            )
        )
    no_negative = bool(not np.any(start[:, 0] == -1))
    any_effective = bool(np.any(start != 0))
    return {
        "episode_id": int(spec["episode_id"]),
        "split": str(spec["split"]),
        "transition_type": str(spec["transition_type"]),
        "start_no_negative_swing": no_negative,
        "start_any_axis_effective": any_effective,
        "start_non_swing_effective": bool(np.any(start[:, 1:] != 0)),
        "outbound_positive_swing": outbound_positive,
        "bucket_tool_liveness": bucket_tool,
        "bucket_positive": bool(
            np.any(
                direction[
                    bucket_start : min(bucket_end + 1, bucket_start + 20), 3
                ]
                == 1
            )
        ),
        "return_negative_swing": return_negative,
        "return_ready_crossing_negative_swing": crossing_negative,
        "ordered_action_proxy": bool(
            no_negative
            and any_effective
            and outbound_positive
            and bucket_tool
            and return_negative
        ),
    }


def _commit_metrics(
    *, root: Path, spec: dict[str, Any], positive: np.ndarray, negative: np.ndarray
) -> dict[str, Any]:
    work_complete = int(spec["work_complete_row"])
    commit = int(spec["return_commit_row"])
    return_start = int(spec["return_effective_segment"][0])
    with h5py.File(root / str(spec["episode_path"]), "r") as handle:
        action = np.asarray(handle["action"][()], dtype=np.float32)
        qvel = np.asarray(handle["observations/qvel"][()], dtype=np.float32)
    direction = _direction(action, positive, negative)
    action_idle = np.all(direction == 0, axis=1)
    all_qvel_stable = np.all(
        np.abs(qvel) <= DIAGNOSTIC_ALL_AXIS_QVEL_MAX.reshape(1, -1), axis=1
    )
    normal = work_complete < commit
    pre = slice(work_complete, commit)
    return {
        "episode_id": int(spec["episode_id"]),
        "split": str(spec["split"]),
        "transition_type": str(spec["transition_type"]),
        "work_complete_before_commit": normal,
        "work_complete_to_commit_rows": commit - work_complete,
        "commit_to_return_effective_rows": return_start - commit,
        "commit_action_mechanically_idle": bool(action_idle[commit]),
        "commit_swing_qvel_ready_stable": bool(
            abs(float(qvel[commit, 0])) <= READY_SWING_QVEL_MAX
        ),
        "commit_all_axis_qvel_diagnostic_stable": bool(all_qvel_stable[commit]),
        "commit_all_axis_qvel_le_0p001": bool(
            np.all(np.abs(qvel[commit]) <= 0.001)
        ),
        "commit_qvel_max_abs": float(np.max(np.abs(qvel[commit]))),
        "uncommitted_no_negative_swing": (
            None if not normal else bool(not np.any(direction[pre, 0] == -1))
        ),
        "uncommitted_all_rows_action_idle": (
            None if not normal else bool(np.all(action_idle[pre]))
        ),
        "uncommitted_action_idle_row_rate": (
            None if not normal else float(np.mean(action_idle[pre]))
        ),
        "uncommitted_any_tool_effective": (
            None if not normal else bool(np.any(direction[pre, 1:] != 0))
        ),
        "uncommitted_all_rows_all_axis_qvel_stable": (
            None if not normal else bool(np.all(all_qvel_stable[pre]))
        ),
        "uncommitted_has_any_all_axis_qvel_stable_row": (
            None if not normal else bool(np.any(all_qvel_stable[pre]))
        ),
        "ready_swing_qvel_stable": bool(
            abs(float(qvel[0, 0])) <= READY_SWING_QVEL_MAX
        ),
        "ready_all_axis_qvel_diagnostic_stable": bool(all_qvel_stable[0]),
        "ready_all_axis_qvel_le_0p001": bool(np.all(np.abs(qvel[0]) <= 0.001)),
    }


def _cycle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "start_no_negative_swing",
        "start_any_axis_effective",
        "start_non_swing_effective",
        "outbound_positive_swing",
        "bucket_tool_liveness",
        "bucket_positive",
        "return_negative_swing",
        "return_ready_crossing_negative_swing",
        "ordered_action_proxy",
    )
    return {"count": len(rows), **{f"{key}_rate": _optional_rate(rows, key) for key in keys}}


def _uncommitted_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "no_negative_swing_rate": _rate(rows, "uncommitted_no_negative_swing"),
        "all_rows_action_idle_rate": _rate(rows, "uncommitted_all_rows_action_idle"),
        "any_tool_effective_rate": _rate(rows, "uncommitted_any_tool_effective"),
        "all_rows_all_axis_qvel_stable_rate": _rate(
            rows, "uncommitted_all_rows_all_axis_qvel_stable"
        ),
        "has_any_all_axis_qvel_stable_row_rate": _rate(
            rows, "uncommitted_has_any_all_axis_qvel_stable_row"
        ),
        "duration_rows": _distribution(rows, "work_complete_to_commit_rows"),
    }


def _commit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "action_mechanically_idle_rate": _rate(
            rows, "commit_action_mechanically_idle"
        ),
        "swing_qvel_ready_stable_rate": _rate(
            rows, "commit_swing_qvel_ready_stable"
        ),
        "all_axis_qvel_diagnostic_stable_rate": _rate(
            rows, "commit_all_axis_qvel_diagnostic_stable"
        ),
        "all_axis_qvel_le_0p001_rate": _rate(
            rows, "commit_all_axis_qvel_le_0p001"
        ),
        "qvel_max_abs": _distribution(rows, "commit_qvel_max_abs"),
        "commit_to_return_effective_rows": _distribution(
            rows, "commit_to_return_effective_rows"
        ),
    }


def _ready_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "swing_qvel_ready_stable_rate": _rate(rows, "ready_swing_qvel_stable"),
        "all_axis_qvel_diagnostic_stable_rate": _rate(
            rows, "ready_all_axis_qvel_diagnostic_stable"
        ),
        "all_axis_qvel_le_0p001_rate": _rate(
            rows, "ready_all_axis_qvel_le_0p001"
        ),
    }


def _direction(
    action: np.ndarray, positive: np.ndarray, negative: np.ndarray
) -> np.ndarray:
    return np.where(
        action >= positive.reshape(1, -1),
        1,
        np.where(action <= -negative.reshape(1, -1), -1, 0),
    ).astype(np.int8)


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows]))


def _optional_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "min": float(values.min()),
        "q10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
