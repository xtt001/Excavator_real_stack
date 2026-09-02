"""Hindsight task-state contract for complete real-transition cycles.

The token keeps task semantics separate from measured robot state.  ``qpos``
and ``qvel`` remain the source of physical pose and motion.  This sidecar adds
only the recorded cycle context and two independently audited hindsight events.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TASK_STATE_V2_KEY = "real_transition_task_state_v2"
TASK_STATE_V2_SCHEMA = "real_transition_task_state_v2_manifest_v1"
TASK_STATE_V2_DIM = 5
TASK_STATE_V2_TIERS = (
    "work_start",
    "work_body",
    "boundary_state",
    "return_body",
)
SIDE_CODES = {"A": -1.0, "B": 1.0}


@dataclass(frozen=True)
class TaskStateV2Candidates:
    """Balanced factual sampling populations for one complete cycle."""

    work_start: np.ndarray
    work_body: np.ndarray
    boundary_state: np.ndarray
    return_body: np.ndarray

    def by_name(self) -> dict[str, np.ndarray]:
        return {
            "work_start": self.work_start,
            "work_body": self.work_body,
            "boundary_state": self.boundary_state,
            "return_body": self.return_body,
        }


def resolve_task_state_v2_config(raw: Any) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("task_state_v2 config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "condition_key": TASK_STATE_V2_KEY,
            "context_dim": TASK_STATE_V2_DIM,
            "tier_names": TASK_STATE_V2_TIERS,
            "action_window_steps": 1,
            "append_samples_per_episode": 0,
            "manifest_path": None,
        }
    if cfg.get("condition_key") != TASK_STATE_V2_KEY:
        raise ValueError(
            f"task_state_v2.condition_key must be {TASK_STATE_V2_KEY!r}"
        )
    manifest_raw = cfg.get("manifest_path")
    if manifest_raw is None or not str(manifest_raw).strip():
        raise ValueError("task_state_v2.manifest_path is required")
    manifest_path = Path(str(manifest_raw)).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"task_state_v2 manifest does not exist: {manifest_path}"
        )
    append = _nonnegative_integer(
        cfg.get("append_samples_per_episode", 3),
        name="append_samples_per_episode",
    )
    if append != len(TASK_STATE_V2_TIERS) - 1:
        raise ValueError(
            "task_state_v2.append_samples_per_episode must be 3 so the base "
            "sample plus three appended samples balance all four task states"
        )
    return {
        "enabled": True,
        "condition_key": TASK_STATE_V2_KEY,
        "context_dim": TASK_STATE_V2_DIM,
        "tier_names": TASK_STATE_V2_TIERS,
        "action_window_steps": _positive_integer(
            cfg.get("action_window_steps", 20), name="action_window_steps"
        ),
        "append_samples_per_episode": append,
        "manifest_path": str(manifest_path),
    }


def task_state_vector(
    *,
    current_side: str,
    dig_target: str,
    next_target: str,
    dig_complete: bool | int | float,
    return_commit: bool | int | float,
) -> np.ndarray:
    """Return ``[current, dig_target, complete, commit, gated_next]``.

    The next destination is intentionally hidden until ``return_commit``.  The
    two event bits are independent, so the audited ``commit-before-complete``
    overlap in episodes 63 and 70 remains representable instead of being
    silently rewritten.
    """

    current = str(current_side)
    target = str(dig_target)
    destination = str(next_target)
    if (
        current not in SIDE_CODES
        or target not in SIDE_CODES
        or destination not in SIDE_CODES
    ):
        raise ValueError("current_side, dig_target and next_target must be A or B")
    if current != target:
        raise ValueError(
            "current_side must equal dig_target for this recorded-data contract"
        )
    complete = _binary_float(dig_complete, name="dig_complete")
    committed = _binary_float(return_commit, name="return_commit")
    gated_next = SIDE_CODES[destination] if committed == 1.0 else 0.0
    return np.asarray(
        [SIDE_CODES[current], SIDE_CODES[target], complete, committed, gated_next],
        dtype=np.float32,
    )


def build_task_state_sequence(
    *,
    total_steps: int,
    current_side: str,
    dig_target: str,
    next_target: str,
    work_complete_row: int,
    return_commit_row: int,
) -> np.ndarray:
    """Materialise the complete per-row task-state sequence from boundaries."""

    length = _positive_integer(total_steps, name="total_steps")
    complete_row = _boundary_row(
        work_complete_row, total_steps=length, name="work_complete_row"
    )
    commit_row = _boundary_row(
        return_commit_row, total_steps=length, name="return_commit_row"
    )
    result = np.empty((length, TASK_STATE_V2_DIM), dtype=np.float32)
    for complete, committed in ((0, 0), (1, 0), (0, 1), (1, 1)):
        mask = (
            (np.arange(length) >= complete_row) == bool(complete)
        ) & ((np.arange(length) >= commit_row) == bool(committed))
        if np.any(mask):
            result[mask] = task_state_vector(
                current_side=current_side,
                dig_target=dig_target,
                next_target=next_target,
                dig_complete=complete,
                return_commit=committed,
            )
    return result


def task_state_candidate_starts(
    *,
    total_steps: int,
    work_complete_row: int,
    return_commit_row: int,
    action_window_steps: int,
    valid_starts: Sequence[int] | np.ndarray | None = None,
) -> TaskStateV2Candidates:
    """Build four factual populations without crossing a task-state change."""

    length = _positive_integer(total_steps, name="total_steps")
    window = _positive_integer(action_window_steps, name="action_window_steps")
    complete_row = _boundary_row(
        work_complete_row, total_steps=length, name="work_complete_row"
    )
    commit_row = _boundary_row(
        return_commit_row, total_steps=length, name="return_commit_row"
    )
    first_boundary = min(complete_row, commit_row)
    last_boundary = max(complete_row, commit_row)
    allowed = _valid_starts(valid_starts, total_steps=length)
    full = allowed[allowed + window <= length]

    work_start = np.asarray(
        [0] if 0 in full and window <= first_boundary else [], dtype=np.int64
    )
    work_body = full[
        (full > 0) & (full + window <= first_boundary)
    ].astype(np.int64, copy=False)
    boundary_state = np.asarray(
        [first_boundary] if first_boundary in allowed else [], dtype=np.int64
    )
    return_body = full[full >= last_boundary].astype(np.int64, copy=False)
    return TaskStateV2Candidates(
        work_start=work_start,
        work_body=work_body,
        boundary_state=boundary_state,
        return_body=return_body,
    )


def task_state_chunk_valid_mask(
    *,
    timestep: int,
    total_steps: int,
    action_chunk_size: int,
    work_complete_row: int,
    return_commit_row: int,
) -> np.ndarray:
    """Mask target actions at the next task-state transition or episode end."""

    length = _positive_integer(total_steps, name="total_steps")
    start = int(timestep)
    if start < 0 or start >= length:
        raise ValueError("timestep must lie inside the episode")
    width = _positive_integer(action_chunk_size, name="action_chunk_size")
    boundaries = sorted(
        {
            _boundary_row(
                work_complete_row,
                total_steps=length,
                name="work_complete_row",
            ),
            _boundary_row(
                return_commit_row,
                total_steps=length,
                name="return_commit_row",
            ),
        }
    )
    stop = min(
        [boundary for boundary in boundaries if boundary > start] + [length]
    )
    indices = start + np.arange(width, dtype=np.int64)
    return (indices < stop) & (indices < length)


def load_task_state_v2_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("task_state_v2 manifest must be a JSON object")
    if payload.get("schema") != TASK_STATE_V2_SCHEMA:
        raise ValueError("task_state_v2 manifest schema mismatch")
    if int(payload.get("task_state_dim", -1)) != TASK_STATE_V2_DIM:
        raise ValueError("task_state_v2 manifest dimension mismatch")
    if tuple(payload.get("tier_names", ())) != TASK_STATE_V2_TIERS:
        raise ValueError("task_state_v2 manifest tier names mismatch")
    return payload


def task_state_manifest_by_episode(
    payload: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    rows = payload.get("episodes", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("task_state_v2 manifest episodes must be a list")
    result: dict[int, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("task_state_v2 episode row must be a mapping")
        row = dict(raw)
        episode_id = int(row["episode_id"])
        if episode_id in result:
            raise ValueError(f"duplicate task_state_v2 episode {episode_id}")
        result[episode_id] = row
    return result


def _valid_starts(
    values: Sequence[int] | np.ndarray | None, *, total_steps: int
) -> np.ndarray:
    if values is None:
        return np.arange(total_steps, dtype=np.int64)
    allowed = np.unique(np.asarray(values, dtype=np.int64).reshape(-1))
    if np.any(allowed < 0) or np.any(allowed >= total_steps):
        raise ValueError("valid_starts must lie inside the episode")
    return allowed


def _boundary_row(value: Any, *, total_steps: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0 or parsed >= int(total_steps) or float(value) != float(parsed):
        raise ValueError(f"task_state_v2.{name} must lie strictly inside the episode")
    return parsed


def _binary_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    parsed = float(value)
    if parsed not in {0.0, 1.0}:
        raise ValueError(f"task_state_v2.{name} must be 0 or 1")
    return parsed


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"task_state_v2.{name} must be boolean")


def _positive_integer(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0 or float(value) != float(parsed):
        raise ValueError(f"task_state_v2.{name} must be a positive integer")
    return parsed


def _nonnegative_integer(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed < 0 or float(value) != float(parsed):
        raise ValueError(f"task_state_v2.{name} must be a non-negative integer")
    return parsed
