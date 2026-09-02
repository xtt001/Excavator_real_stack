"""High-confidence factual action islands for Real Transition ACT training.

The segmentation is intentionally non-exhaustive.  It identifies only action
chunks that remain inside a mechanically effective expert-action run.  Idle
gaps, short reversals, and boundary rows receive no primitive supervision.
The resulting primitive is an oracle command for offline development; this
module does not claim that it can be inferred online from observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from testbed.policies.deadzone_eval import load_deadzone_thresholds

ACTION_PRIMITIVE_KEY = "real_transition_action_primitive_v1"
ACTION_PRIMITIVE_SCHEMA = "real_transition_action_primitive_islands_v1"
PRIMITIVE_NAMES = (
    "tool_pre",
    "swing_out",
    "bucket_out",
    "swing_return",
)
AXIS_NAMES = ("swing", "boom", "stick", "bucket")


@dataclass(frozen=True)
class ActionPrimitiveIslands:
    """Derived segments and full-window candidate starts for one episode."""

    segments: dict[str, tuple[tuple[int, int], ...]]
    candidate_starts: dict[str, np.ndarray]
    evaluable: bool
    reasons: tuple[str, ...]


def resolve_action_primitive_config(raw: Any) -> dict[str, Any]:
    """Validate the oracle primitive sampling and input contract."""

    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("action_primitive_islands config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "condition_key": ACTION_PRIMITIVE_KEY,
            "primitive_names": PRIMITIVE_NAMES,
            "condition_dim": len(PRIMITIVE_NAMES),
            "action_window_steps": 1,
            "append_samples_per_episode": 0,
            "threshold_path": None,
            "positive_thresholds": np.zeros(len(AXIS_NAMES), dtype=np.float32),
            "negative_thresholds": np.zeros(len(AXIS_NAMES), dtype=np.float32),
            "manifest_path": None,
        }
    if cfg.get("condition_key") != ACTION_PRIMITIVE_KEY:
        raise ValueError(
            f"action_primitive_islands.condition_key must be {ACTION_PRIMITIVE_KEY!r}"
        )
    primitive_names = tuple(str(value) for value in cfg.get("primitive_names", ()))
    if primitive_names != PRIMITIVE_NAMES:
        raise ValueError(
            "action_primitive_islands.primitive_names must equal "
            f"{list(PRIMITIVE_NAMES)!r}"
        )
    threshold_raw = cfg.get("threshold_json")
    if threshold_raw is None or not str(threshold_raw).strip():
        raise ValueError("action_primitive_islands.threshold_json is required")
    threshold_path = Path(str(threshold_raw)).resolve()
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"action primitive threshold_json does not exist: {threshold_path}"
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
    if not np.all(np.isfinite(positive)) or not np.all(positive > 0.0):
        raise ValueError("action primitive positive thresholds must be finite and positive")
    if not np.all(np.isfinite(negative)) or not np.all(negative > 0.0):
        raise ValueError("action primitive negative thresholds must be finite and positive")
    append_samples = _nonnegative_integer(
        cfg.get("append_samples_per_episode", len(PRIMITIVE_NAMES) - 1),
        name="append_samples_per_episode",
    )
    if append_samples != len(PRIMITIVE_NAMES) - 1:
        raise ValueError(
            "action_primitive_islands.append_samples_per_episode must be 3 so "
            "the base tier plus appended tiers balance all four primitives"
        )
    manifest_raw = cfg.get("manifest_path")
    if manifest_raw is None or not str(manifest_raw).strip():
        raise ValueError("action_primitive_islands.manifest_path is required")
    manifest_path = Path(str(manifest_raw)).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"action primitive manifest does not exist: {manifest_path}"
        )
    return {
        "enabled": True,
        "condition_key": ACTION_PRIMITIVE_KEY,
        "primitive_names": primitive_names,
        "condition_dim": len(primitive_names),
        "action_window_steps": _positive_integer(
            cfg.get("action_window_steps", 20), name="action_window_steps"
        ),
        "append_samples_per_episode": append_samples,
        "threshold_path": str(threshold_path),
        "positive_thresholds": positive,
        "negative_thresholds": negative,
        "manifest_path": str(manifest_path),
    }


def derive_action_primitive_islands(
    actions: np.ndarray,
    *,
    positive_thresholds: Sequence[float],
    negative_thresholds: Sequence[float],
    action_window_steps: int,
    valid_starts: Sequence[int] | np.ndarray | None = None,
) -> ActionPrimitiveIslands:
    """Derive the four frozen factual primitive interiors for one episode."""

    action = np.asarray(actions, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != len(AXIS_NAMES):
        raise ValueError(
            f"actions must have shape (T, {len(AXIS_NAMES)}), got {action.shape}"
        )
    if len(action) == 0 or not np.all(np.isfinite(action)):
        raise ValueError("actions must be non-empty and finite")
    positive = _threshold_array(positive_thresholds, name="positive_thresholds")
    negative = _threshold_array(negative_thresholds, name="negative_thresholds")
    window = _positive_integer(action_window_steps, name="action_window_steps")
    if valid_starts is None:
        allowed = np.arange(len(action), dtype=np.int64)
    else:
        allowed = np.asarray(valid_starts, dtype=np.int64).reshape(-1)
        if np.any(allowed < 0) or np.any(allowed >= len(action)):
            raise ValueError("valid_starts must be inside the action sequence")
        allowed = np.unique(allowed)

    positive_effective = action >= positive.reshape(1, -1)
    negative_effective = action <= -negative.reshape(1, -1)
    swing_positive_runs = _true_runs(positive_effective[:, 0])
    reasons: list[str] = []
    outbound = _longest_run(swing_positive_runs)
    if outbound is None:
        reasons.append("missing_swing_out")
        return _empty_result(reasons)

    return_runs = [run for run in _true_runs(negative_effective[:, 0]) if run[0] > outbound[1]]
    swing_return = _longest_run(return_runs)
    if swing_return is None:
        reasons.append("missing_swing_return_after_swing_out")
        return _empty_result(reasons)

    bucket_search = np.zeros(len(action), dtype=bool)
    bucket_search[outbound[1] + 1 : swing_return[0]] = positive_effective[
        outbound[1] + 1 : swing_return[0], 3
    ]
    bucket_out = _longest_run(_true_runs(bucket_search))
    if bucket_out is None:
        reasons.append("missing_bucket_out_between_swing_runs")
        return _empty_result(reasons)

    any_swing_effective = positive_effective[:, 0] | negative_effective[:, 0]
    any_tool_effective = (
        positive_effective[:, 1:] | negative_effective[:, 1:]
    ).any(axis=1)
    tool_pre_mask = np.zeros(len(action), dtype=bool)
    tool_pre_mask[: outbound[0]] = (
        any_tool_effective[: outbound[0]] & ~any_swing_effective[: outbound[0]]
    )
    tool_pre_runs = _true_runs(tool_pre_mask)
    if not tool_pre_runs:
        reasons.append("missing_tool_pre_before_swing_out")
        return _empty_result(reasons)

    segment_map = {
        "tool_pre": tuple(tool_pre_runs),
        "swing_out": (outbound,),
        "bucket_out": (bucket_out,),
        "swing_return": (swing_return,),
    }
    candidates = {
        name: _candidate_starts_for_runs(
            runs,
            window=window,
            allowed_starts=allowed,
        )
        for name, runs in segment_map.items()
    }
    for name in PRIMITIVE_NAMES:
        if candidates[name].size == 0:
            reasons.append(f"missing_full_window_{name}")
    return ActionPrimitiveIslands(
        segments=segment_map,
        candidate_starts=candidates,
        evaluable=not reasons,
        reasons=tuple(reasons),
    )


def primitive_one_hot(name: str) -> np.ndarray:
    """Return the fixed-scale oracle command vector for one primitive."""

    text = str(name)
    if text not in PRIMITIVE_NAMES:
        raise ValueError(f"unknown action primitive {text!r}")
    value = np.zeros(len(PRIMITIVE_NAMES), dtype=np.float32)
    value[PRIMITIVE_NAMES.index(text)] = 1.0
    return value


def validate_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    expected_path: str | Path,
) -> None:
    """Fail closed when a training config points at a different contract."""

    if manifest.get("schema") != ACTION_PRIMITIVE_SCHEMA:
        raise ValueError(
            f"action primitive manifest schema must be {ACTION_PRIMITIVE_SCHEMA!r}"
        )
    if tuple(manifest.get("primitive_names", ())) != PRIMITIVE_NAMES:
        raise ValueError("action primitive manifest primitive_names changed")
    if Path(str(manifest.get("path", expected_path))).resolve() != Path(
        expected_path
    ).resolve():
        raise ValueError("action primitive manifest path identity changed")


def _empty_result(reasons: Sequence[str]) -> ActionPrimitiveIslands:
    return ActionPrimitiveIslands(
        segments={name: tuple() for name in PRIMITIVE_NAMES},
        candidate_starts={
            name: np.zeros(0, dtype=np.int64) for name in PRIMITIVE_NAMES
        },
        evaluable=False,
        reasons=tuple(str(value) for value in reasons),
    )


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.concatenate(([False], values, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _longest_run(runs: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    if not runs:
        return None
    return min(runs, key=lambda run: (-(int(run[1]) - int(run[0]) + 1), int(run[0])))


def _candidate_starts_for_runs(
    runs: Sequence[tuple[int, int]],
    *,
    window: int,
    allowed_starts: np.ndarray,
) -> np.ndarray:
    starts: list[np.ndarray] = []
    for first, last in runs:
        final_start = int(last) - int(window) + 1
        if final_start < int(first):
            continue
        starts.append(np.arange(int(first), final_start + 1, dtype=np.int64))
    if not starts:
        return np.zeros(0, dtype=np.int64)
    values = np.concatenate(starts)
    return values[np.isin(values, allowed_starts)].astype(np.int64, copy=False)


def _threshold_array(value: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (len(AXIS_NAMES),):
        raise ValueError(f"{name} must contain {len(AXIS_NAMES)} values")
    if not np.all(np.isfinite(array)) or not np.all(array > 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return array


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"action_primitive_islands.{name} must be boolean")


def _positive_integer(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0 or float(value) != float(parsed):
        raise ValueError(f"action_primitive_islands.{name} must be a positive integer")
    return parsed


def _nonnegative_integer(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed < 0 or float(value) != float(parsed):
        raise ValueError(
            f"action_primitive_islands.{name} must be a non-negative integer"
        )
    return parsed
