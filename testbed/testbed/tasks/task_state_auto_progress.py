"""Causal automatic owner for task-state-v2 progress events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

AUTO_PROGRESS_SCHEMA = "real_transition_task_state_v2_auto_progress_contract_v1"
AXES = ("swing", "boom", "stick", "bucket")
AXIS_INDEX = {name: index for index, name in enumerate(AXES)}


class TaskStateAutoProgressError(RuntimeError):
    """Raised when the frozen automatic progress contract is violated."""


class TaskStateAutoProgress:
    """Convert observed work progress into causal task-state transitions.

    The controller never predicts an action and never examines future state.
    It requires physical boom/bucket movement, the runtime's positive swing
    excursion latch, a sustained positive bucket command, the end of that
    command, and finally an all-axis mechanically idle action window.
    """

    def __init__(self, contract: Mapping[str, Any]) -> None:
        payload = dict(contract)
        if payload.get("schema") != AUTO_PROGRESS_SCHEMA:
            raise TaskStateAutoProgressError("automatic progress schema mismatch")
        if payload.get("status") != "DATA_CONTRACT_PASS":
            raise TaskStateAutoProgressError(
                "automatic progress source contract did not pass"
            )
        cfg = dict(payload.get("runtime_config", {}) or {})
        if cfg.get("advance_source") != "automatic_policy_state":
            raise TaskStateAutoProgressError(
                "automatic progress advance_source mismatch"
            )
        self.required_liveness_axes = tuple(
            str(value) for value in cfg.get("required_liveness_axes", ())
        )
        if not self.required_liveness_axes or any(
            axis not in AXIS_INDEX or axis == "swing"
            for axis in self.required_liveness_axes
        ):
            raise TaskStateAutoProgressError(
                "required_liveness_axes must contain non-swing axes"
            )
        self.min_liveness_qpos_delta_rad = _positive_float(
            cfg.get("min_liveness_qpos_delta_rad"),
            name="min_liveness_qpos_delta_rad",
        )
        self.require_positive_swing_excursion = _strict_bool(
            cfg.get("require_positive_swing_excursion"),
            name="require_positive_swing_excursion",
        )
        self.positive_action_thresholds = _positive_vector4(
            cfg.get("positive_action_thresholds"),
            name="positive_action_thresholds",
        )
        self.negative_action_thresholds = _positive_vector4(
            cfg.get("negative_action_thresholds"),
            name="negative_action_thresholds",
        )
        bucket_threshold = _positive_float(
            cfg.get("bucket_positive_action_threshold"),
            name="bucket_positive_action_threshold",
        )
        if not np.isclose(bucket_threshold, self.positive_action_thresholds[3]):
            raise TaskStateAutoProgressError(
                "bucket threshold disagrees with positive action table"
            )
        self.min_bucket_effective_steps = _positive_integer(
            cfg.get("min_bucket_effective_steps"),
            name="min_bucket_effective_steps",
        )
        self.bucket_release_steps = _positive_integer(
            cfg.get("bucket_release_steps"), name="bucket_release_steps"
        )
        self.return_idle_steps = _positive_integer(
            cfg.get("return_idle_steps"), name="return_idle_steps"
        )
        self.contract = payload
        self.reset()

    @classmethod
    def from_path(cls, path: Path | str) -> TaskStateAutoProgress:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskStateAutoProgressError(
                f"cannot read automatic progress contract {source}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise TaskStateAutoProgressError(
                "automatic progress contract must contain a JSON object"
            )
        return cls(payload)

    def reset(self) -> None:
        self._goal_qpos: np.ndarray | None = None
        self._max_qpos_delta = np.zeros(4, dtype=np.float64)
        self._bucket_effective_count = 0
        self._bucket_effective_observed = False
        self._bucket_release_count = 0
        self._return_idle_count = 0
        self._pending_event = ""
        self._last_event = ""

    def reset_goal(self, qpos: Any) -> None:
        values = _vector4(qpos, name="goal qpos")
        self.reset()
        self._goal_qpos = values

    def observe_qpos(self, qpos: Any) -> None:
        values = _vector4(qpos, name="observed qpos")
        if self._goal_qpos is None:
            return
        self._max_qpos_delta = np.maximum(
            self._max_qpos_delta, np.abs(values - self._goal_qpos)
        )

    def apply_pending(self, policy_source: Any) -> tuple[str, bool]:
        event = self._pending_event
        if not event:
            return "", False
        if event == "work_complete":
            changed = bool(policy_source.set_task_dig_complete(completed=True))
            self._last_event = "automatic_task_work_complete"
        elif event == "return_commit":
            changed = bool(policy_source.set_task_return_commit(committed=True))
            self._last_event = "automatic_task_return_commit"
        else:  # pragma: no cover - internal invariant guard.
            raise TaskStateAutoProgressError(f"unknown pending event {event!r}")
        self._pending_event = ""
        return event, changed

    def observe_policy_action(
        self,
        action: Any,
        *,
        excursion_observed: bool,
        task_dig_complete: bool,
        task_return_commit: bool,
    ) -> None:
        values = _vector4(action, name="policy action")
        if self._goal_qpos is None or self._pending_event or task_return_commit:
            return
        if not task_dig_complete:
            if not self.work_liveness_observed:
                self._reset_work_action_counts()
                return
            if self.require_positive_swing_excursion and not bool(excursion_observed):
                self._reset_work_action_counts()
                return
            bucket_effective = bool(values[3] >= self.positive_action_thresholds[3])
            if bucket_effective:
                self._bucket_effective_count += 1
                self._bucket_release_count = 0
                if self._bucket_effective_count >= self.min_bucket_effective_steps:
                    self._bucket_effective_observed = True
                return
            if not self._bucket_effective_observed:
                self._bucket_effective_count = 0
                return
            self._bucket_release_count += 1
            if self._bucket_release_count >= self.bucket_release_steps:
                self._pending_event = "work_complete"
                self._last_event = "automatic_work_complete_pending"
            return

        mechanically_idle = bool(
            np.all(values < self.positive_action_thresholds)
            and np.all(values > -self.negative_action_thresholds)
        )
        self._return_idle_count = (
            self._return_idle_count + 1 if mechanically_idle else 0
        )
        if self._return_idle_count >= self.return_idle_steps:
            self._pending_event = "return_commit"
            self._last_event = "automatic_return_commit_pending"

    @property
    def work_liveness_observed(self) -> bool:
        if self._goal_qpos is None:
            return False
        return all(
            self._max_qpos_delta[AXIS_INDEX[axis]] >= self.min_liveness_qpos_delta_rad
            for axis in self.required_liveness_axes
        )

    @property
    def pending_event(self) -> str:
        return str(self._pending_event)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "work_liveness_observed": bool(self.work_liveness_observed),
            "required_liveness_axes": list(self.required_liveness_axes),
            "min_liveness_qpos_delta_rad": float(self.min_liveness_qpos_delta_rad),
            "max_qpos_delta_rad": self._max_qpos_delta.astype(float).tolist(),
            "bucket_effective_count": int(self._bucket_effective_count),
            "bucket_effective_observed": bool(self._bucket_effective_observed),
            "bucket_release_count": int(self._bucket_release_count),
            "return_idle_count": int(self._return_idle_count),
            "pending_event": str(self._pending_event),
            "last_event": str(self._last_event),
        }

    def _reset_work_action_counts(self) -> None:
        self._bucket_effective_count = 0
        self._bucket_effective_observed = False
        self._bucket_release_count = 0


def resolve_auto_progress_contract_path(
    raw: Any, *, bundle_dir: Path | str | None
) -> Path:
    if raw is None or not str(raw).strip():
        raise TaskStateAutoProgressError(
            "task_state_v2.auto_progress_contract is required"
        )
    source = Path(str(raw))
    candidates = [source]
    if bundle_dir is not None:
        bundle = Path(bundle_dir)
        candidates.extend(
            (
                bundle / source,
                bundle / source.name,
                bundle / "contracts" / source.name,
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise TaskStateAutoProgressError(
        "automatic progress contract does not exist: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _vector4(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise TaskStateAutoProgressError(f"{name} must contain four finite values")
    return result


def _positive_vector4(value: Any, *, name: str) -> np.ndarray:
    result = _vector4(value, name=name)
    if np.any(result <= 0.0):
        raise TaskStateAutoProgressError(f"{name} must be positive")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise TaskStateAutoProgressError(f"{name} must be finite and positive")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    result = int(value)
    if result <= 0 or float(value) != float(result):
        raise TaskStateAutoProgressError(f"{name} must be a positive integer")
    return result


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TaskStateAutoProgressError(f"{name} must be boolean")
    return value
