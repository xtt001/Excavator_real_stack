"""Two-branch WORK/RETURN context derived from complete continuous cycles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from testbed.data.action_primitive_islands import AXIS_NAMES
from testbed.policies.deadzone_eval import load_deadzone_thresholds

WORK_CONTEXT_KEY = "real_transition_work_context_v1"
WORK_CONTEXT_SCHEMA = "real_transition_work_return_context_manifest_v1"
ROUTE_NAMES = ("work", "return")
TASK_HEAD_NAMES = ("work_A", "work_B", "return_A", "return_B")
SIDE_CODES = {"A": -1.0, "B": 1.0}


@dataclass(frozen=True)
class WorkReturnContext:
    boundary_row: int
    work_starts: np.ndarray
    return_starts: np.ndarray
    outbound_segment: tuple[int, int]
    bucket_segment: tuple[int, int]
    return_segment: tuple[int, int]


def resolve_work_return_context_config(raw: Any) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("work_return_context config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "condition_key": WORK_CONTEXT_KEY,
            "context_dim": 6,
            "route_names": ROUTE_NAMES,
            "action_window_steps": 1,
            "append_samples_per_episode": 0,
            "threshold_path": None,
            "positive_thresholds": np.zeros(4, dtype=np.float32),
            "negative_thresholds": np.zeros(4, dtype=np.float32),
            "manifest_path": None,
        }
    if cfg.get("condition_key") != WORK_CONTEXT_KEY:
        raise ValueError(
            f"work_return_context.condition_key must be {WORK_CONTEXT_KEY!r}"
        )
    threshold_raw = cfg.get("threshold_json")
    if threshold_raw is None or not str(threshold_raw).strip():
        raise ValueError("work_return_context.threshold_json is required")
    threshold_path = Path(str(threshold_raw)).resolve()
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"work_return_context threshold_json does not exist: {threshold_path}"
        )
    thresholds = load_deadzone_thresholds(threshold_path)
    positive = np.asarray(
        [float(thresholds[axis]["pos"]) for axis in AXIS_NAMES],
        dtype=np.float32,
    )
    negative = np.asarray(
        [float(thresholds[axis]["neg"]) for axis in AXIS_NAMES],
        dtype=np.float32,
    )
    manifest_raw = cfg.get("manifest_path")
    if manifest_raw is None or not str(manifest_raw).strip():
        raise ValueError("work_return_context.manifest_path is required")
    manifest_path = Path(str(manifest_raw)).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"work_return_context manifest does not exist: {manifest_path}"
        )
    append = _nonnegative_integer(
        cfg.get("append_samples_per_episode", 1),
        name="append_samples_per_episode",
    )
    if append != 1:
        raise ValueError(
            "work_return_context.append_samples_per_episode must be 1 so the "
            "base WORK and appended RETURN tiers are balanced"
        )
    return {
        "enabled": True,
        "condition_key": WORK_CONTEXT_KEY,
        "context_dim": 6,
        "route_names": ROUTE_NAMES,
        "action_window_steps": _positive_integer(
            cfg.get("action_window_steps", 20), name="action_window_steps"
        ),
        "append_samples_per_episode": append,
        "threshold_path": str(threshold_path),
        "positive_thresholds": positive,
        "negative_thresholds": negative,
        "manifest_path": str(manifest_path),
    }


def derive_work_return_context(
    actions: np.ndarray,
    *,
    positive_thresholds: Sequence[float],
    negative_thresholds: Sequence[float],
    action_window_steps: int,
    valid_starts: Sequence[int] | np.ndarray | None = None,
) -> WorkReturnContext:
    action = np.asarray(actions, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"actions must have shape (T, 4), got {action.shape}")
    if len(action) == 0 or not np.all(np.isfinite(action)):
        raise ValueError("actions must be non-empty and finite")
    positive = _thresholds(positive_thresholds, name="positive_thresholds")
    negative = _thresholds(negative_thresholds, name="negative_thresholds")
    window = _positive_integer(action_window_steps, name="action_window_steps")
    positive_effective = action >= positive.reshape(1, -1)
    negative_effective = action <= -negative.reshape(1, -1)
    outbound = _longest(_runs(positive_effective[:, 0]), name="positive swing")
    return_runs = [
        run for run in _runs(negative_effective[:, 0]) if run[0] > outbound[1]
    ]
    return_segment = _longest(return_runs, name="negative return swing")
    bucket_mask = np.zeros(len(action), dtype=bool)
    bucket_mask[outbound[1] + 1 : return_segment[0]] = positive_effective[
        outbound[1] + 1 : return_segment[0], 3
    ]
    bucket = _longest(_runs(bucket_mask), name="post-outbound positive bucket")
    boundary = int(bucket[1] + 1)
    if not 0 < boundary < return_segment[0] < len(action):
        raise ValueError(
            "work_complete boundary must precede the main negative return segment"
        )
    if valid_starts is None:
        allowed = np.arange(len(action), dtype=np.int64)
    else:
        allowed = np.unique(np.asarray(valid_starts, dtype=np.int64).reshape(-1))
        if np.any(allowed < 0) or np.any(allowed >= len(action)):
            raise ValueError("valid_starts must lie inside the episode")
    full = allowed[allowed + window <= len(action)]
    work = full[full + window <= boundary]
    returning = full[full >= boundary]
    if work.size == 0 or returning.size == 0:
        raise ValueError("WORK and RETURN must each contain a full action chunk")
    return WorkReturnContext(
        boundary_row=boundary,
        work_starts=work.astype(np.int64, copy=False),
        return_starts=returning.astype(np.int64, copy=False),
        outbound_segment=(int(outbound[0]), int(outbound[1])),
        bucket_segment=(int(bucket[0]), int(bucket[1])),
        return_segment=(int(return_segment[0]), int(return_segment[1])),
    )


def work_context_vector(
    *, current_anchor: str, dig_target: str, next_target: str, route: str
) -> np.ndarray:
    current = str(current_anchor)
    target = str(dig_target)
    destination = str(next_target)
    branch = str(route)
    if (
        current not in SIDE_CODES
        or target not in SIDE_CODES
        or destination not in SIDE_CODES
    ):
        raise ValueError("current_anchor, dig_target and next_target must be A or B")
    if branch not in ROUTE_NAMES:
        raise ValueError(f"route must be one of {ROUTE_NAMES}")
    if current != target:
        raise ValueError(
            "current_anchor must equal dig_target for the current data contract; "
            "independent combinations require a new supervised schema"
        )
    head = (
        f"work_{target}" if branch == "work" else f"return_{destination}"
    )
    one_hot = [1.0 if name == head else 0.0 for name in TASK_HEAD_NAMES]
    return np.asarray(
        [SIDE_CODES[current], SIDE_CODES[target], *one_hot],
        dtype=np.float32,
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.concatenate(([False], values, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _longest(runs: Sequence[tuple[int, int]], *, name: str) -> tuple[int, int]:
    if not runs:
        raise ValueError(f"missing mechanically effective {name} segment")
    return min(runs, key=lambda run: (-(run[1] - run[0] + 1), run[0]))


def _thresholds(value: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (4,) or not np.all(np.isfinite(array)) or not np.all(array > 0):
        raise ValueError(f"{name} must contain four finite positive values")
    return array


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"work_return_context.{name} must be boolean")


def _positive_integer(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0 or float(value) != float(parsed):
        raise ValueError(f"work_return_context.{name} must be a positive integer")
    return parsed


def _nonnegative_integer(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed < 0 or float(value) != float(parsed):
        raise ValueError(
            f"work_return_context.{name} must be a non-negative integer"
        )
    return parsed
