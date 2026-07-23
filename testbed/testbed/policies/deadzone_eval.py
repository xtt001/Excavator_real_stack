"""Deadzone-based single-demo relation metrics for offline policy replay."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from testbed.policies.offline_eval import AXIS_NAMES, load_train_ready_episode_ids

DEFAULT_WINDOWS = (
    "start40",
    "end80",
    "longest_single_demo_effective_segment_gap5",
    "full_available",
)


@dataclass(frozen=True)
class EvalSpec:
    model: str
    eval_dir: Path


def parse_eval_spec(value: str) -> EvalSpec:
    if "=" not in value:
        raise ValueError(f"--eval must be MODEL=DIR, got: {value}")
    model, path = value.split("=", 1)
    model = model.strip()
    if not model:
        raise ValueError(f"--eval model label is empty: {value}")
    return EvalSpec(model=model, eval_dir=Path(path).expanduser())


def load_deadzone_thresholds(path: str | Path) -> dict[str, dict[str, float]]:
    deadzone_path = Path(path)
    payload = json.loads(deadzone_path.read_text(encoding="utf-8"))
    raw = payload.get("deadzone_action", payload)
    thresholds: dict[str, dict[str, float]] = {}
    for axis in AXIS_NAMES:
        axis_raw = raw.get(axis)
        if not isinstance(axis_raw, dict):
            raise ValueError(
                f"deadzone table is missing axis {axis!r}: {deadzone_path}"
            )
        thresholds[axis] = {
            "pos": _threshold_value(axis_raw.get("pos"), axis=axis, direction="pos"),
            "neg": _threshold_value(axis_raw.get("neg"), axis=axis, direction="neg"),
        }
    return thresholds


def find_episode_action_files(
    eval_dir: str | Path,
    *,
    manifest: str | Path | None = None,
) -> list[tuple[str, Path]]:
    root = Path(eval_dir)
    if not root.exists():
        raise FileNotFoundError(f"offline eval directory does not exist: {root}")
    if manifest is not None:
        episode_ids = load_train_ready_episode_ids(manifest)
    else:
        episode_ids = sorted(
            path.parent.name
            for path in (root / "episodes").glob("episode_*/actions.npz")
        )
        if not episode_ids and (root / "actions.npz").exists():
            summary_path = root / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                episode_ids = [
                    str(
                        summary.get("selected_episode_id")
                        or Path(str(summary["episode_path"])).stem
                    )
                ]
            else:
                episode_ids = [root.name]

    files: list[tuple[str, Path]] = []
    for episode_id in episode_ids:
        action_path = root / "episodes" / episode_id / "actions.npz"
        if not action_path.exists() and len(episode_ids) == 1:
            action_path = root / "actions.npz"
        if not action_path.exists():
            raise FileNotFoundError(
                f"missing offline replay actions for {episode_id}: {action_path}"
            )
        files.append((episode_id, action_path))
    if not files:
        raise ValueError(f"no actions.npz files found under {root}")
    return files


def compute_deadzone_window_rows(
    *,
    model: str,
    episode_id: str,
    expert_action: np.ndarray,
    policy_action: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    windows: Iterable[str] = DEFAULT_WINDOWS,
    gap: int = 5,
) -> list[dict[str, Any]]:
    expert = _validate_action_array(expert_action, name="expert_action")
    policy = _validate_action_array(policy_action, name="policy_action")
    if expert.shape != policy.shape:
        raise ValueError(
            f"expert and policy actions must share shape, got {expert.shape} vs {policy.shape}"
        )

    expert_effective = effective_direction_mask(expert, thresholds)
    policy_effective = effective_direction_mask(policy, thresholds)
    window_ranges = build_window_ranges(
        expert_effective.any(axis=(1, 2)),
        total_steps=expert.shape[0],
        windows=windows,
        gap=gap,
    )

    return [
        compute_window_row(
            model=model,
            episode_id=episode_id,
            window=window,
            start=start,
            end=end,
            expert_effective=expert_effective,
            policy_effective=policy_effective,
        )
        for window, start, end in window_ranges
    ]


def effective_direction_mask(
    action: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> np.ndarray:
    actions = _validate_action_array(action, name="action")
    mask = np.zeros((actions.shape[0], len(AXIS_NAMES), 2), dtype=bool)
    for axis_idx, axis in enumerate(AXIS_NAMES):
        axis_thresholds = thresholds[axis]
        mask[:, axis_idx, 0] = actions[:, axis_idx] >= float(axis_thresholds["pos"])
        mask[:, axis_idx, 1] = actions[:, axis_idx] <= -float(axis_thresholds["neg"])
    return mask


def build_window_ranges(
    expert_any_effective: np.ndarray,
    *,
    total_steps: int,
    windows: Iterable[str] = DEFAULT_WINDOWS,
    gap: int = 5,
) -> list[tuple[str, int, int]]:
    mask = np.asarray(expert_any_effective, dtype=bool)
    if mask.shape != (total_steps,):
        raise ValueError(
            f"expert_any_effective must have shape ({total_steps},), got {mask.shape}"
        )
    ranges: list[tuple[str, int, int]] = []
    for window in windows:
        if window == "start40":
            start, end = 0, min(40, total_steps)
        elif window == "end80":
            start, end = max(0, total_steps - 80), total_steps
        elif window == "full_available":
            start, end = 0, total_steps
        elif window == "longest_single_demo_effective_segment_gap5":
            start, end = longest_true_segment_with_gap(mask, gap=gap)
        else:
            raise ValueError(f"unknown window: {window}")
        ranges.append((window, int(start), int(end)))
    return ranges


def longest_true_segment_with_gap(mask: np.ndarray, *, gap: int = 5) -> tuple[int, int]:
    true_indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if true_indices.size == 0:
        return 0, 0

    best_start = int(true_indices[0])
    best_end = int(true_indices[0]) + 1
    current_start = int(true_indices[0])
    previous = int(true_indices[0])
    for index in true_indices[1:]:
        index_int = int(index)
        if index_int - previous > gap + 1:
            current_end = previous + 1
            if current_end - current_start > best_end - best_start:
                best_start, best_end = current_start, current_end
            current_start = index_int
        previous = index_int

    current_end = previous + 1
    if current_end - current_start > best_end - best_start:
        best_start, best_end = current_start, current_end
    return best_start, best_end


def compute_window_row(
    *,
    model: str,
    episode_id: str,
    window: str,
    start: int,
    end: int,
    expert_effective: np.ndarray,
    policy_effective: np.ndarray,
) -> dict[str, Any]:
    expert_slice = expert_effective[start:end]
    policy_slice = policy_effective[start:end]
    steps = int(max(0, end - start))

    expert_any = expert_slice.any(axis=(1, 2)) if steps else np.zeros(0, dtype=bool)
    policy_any = policy_slice.any(axis=(1, 2)) if steps else np.zeros(0, dtype=bool)
    same_axis_dir = (
        (expert_slice & policy_slice).any(axis=(1, 2))
        if steps
        else np.zeros(0, dtype=bool)
    )
    outside_demo_frame = (
        (policy_any & ~same_axis_dir) if steps else np.zeros(0, dtype=bool)
    )
    expert_effective_frames = int(expert_any.sum())

    row: dict[str, Any] = {
        "model": model,
        "episode_id": episode_id,
        "window": window,
        "start_step": int(start),
        "end_step_exclusive": int(end),
        "steps": steps,
        "single_demo_any_effective_frames": expert_effective_frames,
        "single_demo_any_effective_pct": _pct(int(expert_any.sum()), steps),
        "policy_any_effective_frames": int(policy_any.sum()),
        "policy_any_effective_pct": _pct(int(policy_any.sum()), steps),
        "single_demo_same_axis_direction_effective_frames": int(same_axis_dir.sum()),
        "single_demo_same_axis_direction_effective_pct_of_demo_effective": (
            _pct(int(same_axis_dir.sum()), expert_effective_frames)
            if expert_effective_frames
            else ""
        ),
        "policy_outside_single_demo_frame_effective_frames": int(
            outside_demo_frame.sum()
        ),
        "policy_outside_single_demo_frame_effective_pct": _pct(
            int(outside_demo_frame.sum()), steps
        ),
    }

    for axis_idx, axis in enumerate(AXIS_NAMES):
        row[f"single_demo_{axis}_pos_eff_pct"] = _pct(
            int(expert_slice[:, axis_idx, 0].sum()), steps
        )
        row[f"single_demo_{axis}_neg_eff_pct"] = _pct(
            int(expert_slice[:, axis_idx, 1].sum()), steps
        )
        row[f"policy_{axis}_pos_eff_pct"] = _pct(
            int(policy_slice[:, axis_idx, 0].sum()), steps
        )
        row[f"policy_{axis}_neg_eff_pct"] = _pct(
            int(policy_slice[:, axis_idx, 1].sum()), steps
        )
    return row


def aggregate_window_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["model"]), str(row["window"])), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (model, window), group in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "model": model,
            "window": window,
            "episodes": len({str(row["episode_id"]) for row in group}),
            "mean_single_demo_any_effective_pct": _mean(
                row["single_demo_any_effective_pct"] for row in group
            ),
            "mean_policy_any_effective_pct": _mean(
                row["policy_any_effective_pct"] for row in group
            ),
            "median_policy_any_effective_pct": _median(
                row["policy_any_effective_pct"] for row in group
            ),
            "episodes_policy_any_effective_gt0": sum(
                float(row["policy_any_effective_pct"]) > 0.0 for row in group
            ),
            "episodes_policy_any_effective_ge50": sum(
                float(row["policy_any_effective_pct"]) >= 50.0 for row in group
            ),
            "mean_single_demo_same_axis_direction_effective_pct_of_demo_effective": _mean(
                _numeric_or_zero(
                    row[
                        "single_demo_same_axis_direction_effective_pct_of_demo_effective"
                    ]
                )
                for row in group
            ),
            "median_single_demo_same_axis_direction_effective_pct_of_demo_effective": _median(
                _numeric_or_zero(
                    row[
                        "single_demo_same_axis_direction_effective_pct_of_demo_effective"
                    ]
                )
                for row in group
            ),
            "mean_policy_outside_single_demo_frame_effective_pct": _mean(
                row["policy_outside_single_demo_frame_effective_pct"] for row in group
            ),
            "episodes_outside_single_demo_frame_ge20pct": sum(
                float(row["policy_outside_single_demo_frame_effective_pct"]) >= 20.0
                for row in group
            ),
        }
        for axis in AXIS_NAMES:
            aggregate[f"mean_policy_{axis}_pos_eff_pct"] = _mean(
                row[f"policy_{axis}_pos_eff_pct"] for row in group
            )
            aggregate[f"mean_policy_{axis}_neg_eff_pct"] = _mean(
                row[f"policy_{axis}_neg_eff_pct"] for row in group
            )
        aggregate_rows.append(aggregate)
    return aggregate_rows


def build_model_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["episode_id"]), str(row["window"])), {})[
            str(row["model"])
        ] = row

    comparison_rows: list[dict[str, Any]] = []
    for (episode_id, window), by_model in sorted(grouped.items()):
        output: dict[str, Any] = {"episode_id": episode_id, "window": window}
        for model in sorted(by_model):
            row = by_model[model]
            output[f"{model}_policy_any_effective_pct"] = row[
                "policy_any_effective_pct"
            ]
            output[f"{model}_single_demo_same_axis_direction_effective_pct"] = row[
                "single_demo_same_axis_direction_effective_pct_of_demo_effective"
            ]
            output[f"{model}_outside_single_demo_frame_effective_pct"] = row[
                "policy_outside_single_demo_frame_effective_pct"
            ]
        comparison_rows.append(output)
    return comparison_rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows_for_eval(
    *,
    model: str,
    eval_dir: str | Path,
    thresholds: dict[str, dict[str, float]],
    manifest: str | Path | None = None,
    windows: Iterable[str] = DEFAULT_WINDOWS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_id, action_path in find_episode_action_files(
        eval_dir, manifest=manifest
    ):
        with np.load(action_path) as data:
            expert = np.asarray(data["expert_action"], dtype=np.float32)
            policy = np.asarray(data["policy_action"], dtype=np.float32)
        rows.extend(
            compute_deadzone_window_rows(
                model=model,
                episode_id=episode_id,
                expert_action=expert,
                policy_action=policy,
                thresholds=thresholds,
                windows=windows,
            )
        )
    return rows


def compute_intent_census_row(
    *,
    episode_id: str,
    action: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    actions = _validate_action_array(action, name="action")
    effective = effective_direction_mask(actions, thresholds)
    per_step_events = effective.sum(axis=(1, 2))
    should_move = per_step_events > 0
    steps = int(actions.shape[0])
    should_move_frames = int(should_move.sum())
    row: dict[str, Any] = {
        "episode_id": episode_id,
        "steps": steps,
        "should_move_frames": should_move_frames,
        "should_move_pct": _pct(should_move_frames, steps),
        "should_stop_frames": int(steps - should_move_frames),
        "should_stop_pct": _pct(steps - should_move_frames, steps),
        "effective_axis_dir_events": int(effective.sum()),
        "multi_dir_move_frames": int((per_step_events > 1).sum()),
        "multi_dir_move_pct_of_move": _pct(
            int((per_step_events > 1).sum()), should_move_frames
        ),
        "mean_effective_dirs_per_move_frame": (
            float(per_step_events[should_move].mean()) if should_move_frames else 0.0
        ),
    }
    for axis_idx, axis in enumerate(AXIS_NAMES):
        row[f"{axis}_pos_frames"] = int(effective[:, axis_idx, 0].sum())
        row[f"{axis}_pos_pct"] = _pct(row[f"{axis}_pos_frames"], steps)
        row[f"{axis}_neg_frames"] = int(effective[:, axis_idx, 1].sum())
        row[f"{axis}_neg_pct"] = _pct(row[f"{axis}_neg_frames"], steps)
    return row


def aggregate_intent_census_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_steps = sum(int(row["steps"]) for row in rows)
    should_move_frames = sum(int(row["should_move_frames"]) for row in rows)
    multi_dir_move_frames = sum(int(row["multi_dir_move_frames"]) for row in rows)
    effective_events = sum(int(row["effective_axis_dir_events"]) for row in rows)
    aggregate: dict[str, Any] = {
        "episodes": len(rows),
        "total_steps": total_steps,
        "should_move_frames": should_move_frames,
        "should_move_pct": _pct(should_move_frames, total_steps),
        "should_stop_frames": total_steps - should_move_frames,
        "should_stop_pct": _pct(total_steps - should_move_frames, total_steps),
        "effective_axis_dir_events": effective_events,
        "multi_dir_move_frames": multi_dir_move_frames,
        "multi_dir_move_pct_of_move": _pct(multi_dir_move_frames, should_move_frames),
        "mean_effective_dirs_per_move_frame": (
            float(effective_events) / float(should_move_frames)
            if should_move_frames
            else 0.0
        ),
    }
    for axis in AXIS_NAMES:
        for direction in ("pos", "neg"):
            key = f"{axis}_{direction}_frames"
            value = sum(int(row[key]) for row in rows)
            aggregate[key] = value
            aggregate[f"{axis}_{direction}_pct"] = _pct(value, total_steps)
    return [aggregate]


def _threshold_value(value: Any, *, axis: str, direction: str) -> float:
    if isinstance(value, dict):
        value = value.get("threshold_action_abs")
    if value is None:
        raise ValueError(f"missing threshold for {axis}.{direction}")
    threshold = float(value)
    if threshold < 0.0:
        raise ValueError(
            f"threshold must be non-negative for {axis}.{direction}: {threshold}"
        )
    return threshold


def _validate_action_array(action: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(AXIS_NAMES):
        raise ValueError(
            f"{name} must have shape (T, {len(AXIS_NAMES)}), got {array.shape}"
        )
    return array


def _pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(count) / float(total)


def _mean(values: Iterable[Any]) -> float:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.mean(array))


def _median(values: Iterable[Any]) -> float:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.median(array))


def _numeric_or_zero(value: Any) -> float:
    if value == "" or value is None:
        return 0.0
    return float(value)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames
