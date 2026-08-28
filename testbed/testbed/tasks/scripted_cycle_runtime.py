"""Live lifecycle owner for a planner-conditioned scripted ACT sequence.

The neural policy owns continuous joint actions for one committed cycle.  This
module owns only the goal/ready lifecycle around it: verify the initial ready
side, commit the next scripted goal, require a real swing excursion, accept a
stable target-side ready window, and then commit the following goal.  It never
invents state from policy actions and never sends actuator commands itself.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from testbed.tasks.home_side_contract import (
    classify_ready_swing_qpos,
    validate_rule_ready_contract,
)


class ScriptedCycleRuntimeError(RuntimeError):
    """Raised when a live scripted-cycle contract cannot be satisfied."""


class ReadySideWindow:
    """Rolling, contract-backed A/B ready classifier for live qpos/qvel."""

    def __init__(
        self,
        contract: Mapping[str, Any],
        *,
        target_ranges: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        payload = dict(contract)
        validate_rule_ready_contract(payload)
        self.contract = payload
        self.target_ranges = {
            str(side): (float(bounds[0]), float(bounds[1]))
            for side, bounds in dict(target_ranges or {}).items()
        }
        self._samples: deque[tuple[int, np.ndarray, np.ndarray]] = deque()

    def reset(self) -> None:
        self._samples.clear()

    def update(self, *, timestamp_ns: int, qpos: Any, qvel: Any) -> dict[str, Any]:
        stamp = int(timestamp_ns)
        qpos_array = np.asarray(qpos, dtype=np.float64).reshape(-1)
        qvel_array = np.asarray(qvel, dtype=np.float64).reshape(-1)
        if stamp <= 0:
            raise ScriptedCycleRuntimeError(
                "ready observation timestamp must be positive"
            )
        if qpos_array.shape != (4,) or qvel_array.shape != (4,):
            raise ScriptedCycleRuntimeError(
                "ready observation qpos/qvel must both have shape (4,)"
            )
        if not np.isfinite(qpos_array).all() or not np.isfinite(qvel_array).all():
            raise ScriptedCycleRuntimeError(
                "ready observation qpos/qvel must be finite"
            )

        if self._samples and stamp <= self._samples[-1][0]:
            if stamp == self._samples[-1][0]:
                self._samples[-1] = (stamp, qpos_array.copy(), qvel_array.copy())
            else:
                self._samples.clear()
                self._samples.append((stamp, qpos_array.copy(), qvel_array.copy()))
        else:
            self._samples.append((stamp, qpos_array.copy(), qvel_array.copy()))

        stable_window_s = float(self.contract["swing_axis"]["stable_window_s"])
        keep_after_ns = stamp - int(max(1.0, 2.0 * stable_window_s) * 1e9)
        while self._samples and self._samples[0][0] < keep_after_ns:
            self._samples.popleft()
        return self.snapshot()

    @property
    def latest_swing_qpos(self) -> float | None:
        if not self._samples:
            return None
        return float(self._samples[-1][1][0])

    def snapshot(self) -> dict[str, Any]:
        swing = self.contract["swing_axis"]
        base = {
            "contract_schema": str(self.contract["schema"]),
            "window_required_s": float(swing["stable_window_s"]),
            "swing_qvel_limit_rad_s": float(swing["swing_qvel_abs_max_rad_s"]),
            "sample_count": 0,
            "window_duration_s": 0.0,
            "window_complete": False,
            "sample_gap_ok": False,
            "swing_stable": False,
            "clean_side_window": False,
            "target_support_window": False,
            "actual_side": "unknown",
            "blockers": [],
        }
        if not self._samples:
            return {**base, "blockers": ["no_ready_observations"]}

        samples = list(self._samples)
        latest_ns = int(samples[-1][0])
        window_ns = int(float(swing["stable_window_s"]) * 1e9)
        cutoff_ns = latest_ns - window_ns
        start_index = next(
            index for index, sample in enumerate(samples) if sample[0] >= cutoff_ns
        )
        if samples[start_index][0] > cutoff_ns and start_index > 0:
            start_index -= 1
        window = samples[start_index:]
        timestamps = np.asarray([sample[0] for sample in window], dtype=np.int64)
        qpos = np.stack([sample[1] for sample in window])
        qvel = np.stack([sample[2] for sample in window])

        duration_s = float((timestamps[-1] - timestamps[0]) * 1e-9)
        max_gap_s = (
            float(np.max(np.diff(timestamps)) * 1e-9) if len(timestamps) > 1 else None
        )
        classifications = [
            classify_ready_swing_qpos(self.contract, value) for value in qpos[:, 0]
        ]
        actual_side = str(classifications[-1])
        window_complete = duration_s + 1e-9 >= float(swing["stable_window_s"])
        sample_gap_ok = max_gap_s is not None and max_gap_s <= float(
            swing["max_sample_gap_s"]
        )
        swing_qvel_abs_max = float(np.max(np.abs(qvel[:, 0])))
        swing_stable = swing_qvel_abs_max <= float(swing["swing_qvel_abs_max_rad_s"])
        clean_side_window = actual_side in {"A", "B"} and all(
            value == actual_side for value in classifications
        )
        target_support_window = clean_side_window
        if clean_side_window and self.target_ranges:
            bounds = self.target_ranges.get(actual_side)
            if bounds is None:
                target_support_window = False
            else:
                low, high = bounds
                target_support_window = bool(
                    np.all((qpos[:, 0] >= low) & (qpos[:, 0] <= high))
                )
        blockers: list[str] = []
        if not window_complete:
            blockers.append("swing_window_too_short")
        if not sample_gap_ok:
            blockers.append("swing_window_sample_gap")
        if not swing_stable:
            blockers.append("swing_not_stable")
        if not clean_side_window:
            blockers.append(f"swing_side_{actual_side}")
        if clean_side_window and not target_support_window:
            blockers.append(f"swing_outside_{actual_side}_training_support")
        return {
            **base,
            "sample_count": int(len(window)),
            "window_start_ns": int(timestamps[0]),
            "window_end_ns": int(timestamps[-1]),
            "window_duration_s": duration_s,
            "window_complete": bool(window_complete),
            "max_sample_gap_s": max_gap_s,
            "sample_gap_ok": bool(sample_gap_ok),
            "swing_qpos_current_rad": float(qpos[-1, 0]),
            "swing_qpos_window_min_rad": float(np.min(qpos[:, 0])),
            "swing_qpos_window_max_rad": float(np.max(qpos[:, 0])),
            "swing_qvel_abs_max_rad_s": swing_qvel_abs_max,
            "swing_stable": bool(swing_stable),
            "clean_side_window": bool(clean_side_window),
            "target_support_window": bool(target_support_window),
            "actual_side": actual_side,
            "non_swing_qpos_current_rad": qpos[-1, 1:].tolist(),
            "non_swing_qvel_abs_max_rad_s": np.max(
                np.abs(qvel[:, 1:]), axis=0
            ).tolist(),
            "non_swing_axes_gate_ready": False,
            "blockers": blockers,
        }


class ScriptedCycleRuntime:
    """Advance a live ScriptCyclePlanner only on independently observed ready."""

    def __init__(
        self,
        *,
        policy_source: Any,
        ready_contract: Mapping[str, Any],
        target_ranges: Mapping[str, tuple[float, float]] | None = None,
        cycle_review_s: float = 45.0,
        cycle_stop_s: float = 60.0,
        run_stop_s: float = 240.0,
        stop_on_wrong_ready: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        planner = getattr(policy_source, "cycle_planner", None)
        if planner is None:
            raise ScriptedCycleRuntimeError(
                "scripted-cycle runtime requires teleop.policy.cycle_planner"
            )
        for method_name in (
            "commit_cycle_goal",
            "mark_cycle_target_ready",
            "cycle_planner_status",
        ):
            if not callable(getattr(policy_source, method_name, None)):
                raise ScriptedCycleRuntimeError(
                    f"policy source must implement {method_name}()"
                )
        self.policy_source = policy_source
        self.planner = planner
        self.ready = ReadySideWindow(
            ready_contract,
            target_ranges=target_ranges,
        )
        self.cycle_review_s = float(cycle_review_s)
        self.cycle_stop_s = float(cycle_stop_s)
        self.run_stop_s = float(run_stop_s)
        self.stop_on_wrong_ready = bool(stop_on_wrong_ready)
        self._clock = clock
        if not (0.0 < self.cycle_review_s < self.cycle_stop_s):
            raise ScriptedCycleRuntimeError(
                "scripted-cycle limits require 0 < cycle_review_s < cycle_stop_s"
            )
        if self.run_stop_s < self.cycle_stop_s:
            raise ScriptedCycleRuntimeError(
                "scripted-cycle run_stop_s must be at least cycle_stop_s"
            )
        self._reset_execution()

    @classmethod
    def from_config(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        policy_source: Any,
        bundle_dir: str | Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> ScriptedCycleRuntime | None:
        cfg = dict(raw or {})
        if not bool(cfg.get("enabled", False)):
            return None
        contract_path = _resolve_contract_path(
            cfg.get("ready_contract"), bundle_dir=bundle_dir
        )
        try:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScriptedCycleRuntimeError(
                f"cannot read scripted-cycle ready contract {contract_path}: {exc}"
            ) from exc
        target_ranges = _load_target_ranges(
            cfg.get("target_region_contract"), bundle_dir=bundle_dir
        )
        return cls(
            policy_source=policy_source,
            ready_contract=payload,
            target_ranges=target_ranges,
            cycle_review_s=float(cfg.get("cycle_review_s", 45.0)),
            cycle_stop_s=float(cfg.get("cycle_stop_s", 60.0)),
            run_stop_s=float(cfg.get("run_stop_s", 240.0)),
            stop_on_wrong_ready=bool(cfg.get("stop_on_wrong_ready", True)),
            clock=clock,
        )

    def reset(self) -> None:
        self.ready.reset()
        self._reset_execution()

    def prepare_new_run(self) -> None:
        """Clear execution state while retaining the latest manual ready window."""

        self._reset_execution()

    def observe(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        timestamp_ns = _observation_timestamp_ns(observation)
        status = self.ready.update(
            timestamp_ns=timestamp_ns,
            qpos=observation.get("qpos"),
            qvel=observation.get("qvel"),
        )
        self._last_ready = status
        return self.status()

    def activation_blocker(self) -> str:
        ready = self._last_ready
        blockers = list(ready.get("blockers", ()))
        if blockers:
            return "initial_ready:" + ",".join(str(value) for value in blockers)
        actual_side = str(ready.get("actual_side", "unknown"))
        initial_side = str(getattr(self.planner, "initial_side", ""))
        if actual_side != initial_side:
            return f"initial_side_mismatch:expected_{initial_side}:actual_{actual_side}"
        return ""

    def activate(self) -> dict[str, Any]:
        blocker = self.activation_blocker()
        if blocker:
            raise ScriptedCycleRuntimeError(blocker)
        self.prepare_new_run()
        now = float(self._clock())
        self._active = True
        self._run_started_s = now
        goal = self.policy_source.commit_cycle_goal()
        actual_side = str(self._last_ready["actual_side"])
        if str(goal.current_side) != actual_side:
            return self._fault(
                "planner_current_side_mismatch:"
                f"expected_{goal.current_side}:actual_{actual_side}"
            )
        self._commit_goal_state(goal=goal, now_s=now)
        self._last_event = "goal_committed"
        return self.status(goal_changed=True)

    def evaluate(self) -> dict[str, Any]:
        if not self._active:
            return self.status()
        now = float(self._clock())
        if (
            self._run_started_s is not None
            and now - self._run_started_s >= self.run_stop_s
        ):
            return self._fault("run_timeout")
        if self._cycle_started_s is not None:
            cycle_elapsed = now - self._cycle_started_s
            if cycle_elapsed >= self.cycle_stop_s:
                return self._fault("cycle_timeout")
            self._review_due = cycle_elapsed >= self.cycle_review_s

        goal = getattr(self.planner, "committed_goal", None)
        if goal is None:
            return self._fault("planner_goal_missing")
        current_swing = self.ready.latest_swing_qpos
        if current_swing is None:
            return self.status()

        if not self._excursion_observed:
            if self._goal_anchor_swing_qpos is None:
                return self._fault("goal_anchor_missing")
            swing_cfg = self.ready.contract["swing_axis"]
            delta = _shortest_angle(current_swing - self._goal_anchor_swing_qpos)
            threshold = float(swing_cfg["cycle_excursion_min_abs_delta_rad"])
            required = int(swing_cfg["cycle_excursion_min_consecutive_samples"])
            if abs(delta) < threshold:
                self._excursion_candidate_count = 0
                return self.status()
            self._excursion_candidate_count += 1
            if self._excursion_candidate_count < required:
                return self.status()
            self._excursion_observed = True
            self._last_event = "cycle_excursion_confirmed"
            return self.status()

        ready = self._last_ready
        if list(ready.get("blockers", ())):
            return self.status()
        actual_side = str(ready.get("actual_side", "unknown"))
        target_side = str(goal.target_side)
        if actual_side != target_side:
            if self.stop_on_wrong_ready:
                return self._fault(
                    f"stable_wrong_side:expected_{target_side}:actual_{actual_side}"
                )
            return self.status()

        self.policy_source.mark_cycle_target_ready(actual_side)
        if bool(getattr(self.planner, "done", False)):
            self._active = False
            self._completed = True
            self._last_event = "script_complete"
            return self.status(stop_policy=True, completed_now=True)

        next_goal = self.policy_source.commit_cycle_goal()
        self._commit_goal_state(goal=next_goal, now_s=now)
        self._last_event = "cycle_advanced"
        return self.status(goal_changed=True)

    def deactivate(self, reason: str) -> dict[str, Any]:
        if self._active:
            self._last_event = "deactivated"
        self._active = False
        self._stop_reason = str(reason)
        return self.status(stop_policy=True)

    def status(
        self,
        *,
        goal_changed: bool = False,
        stop_policy: bool = False,
        completed_now: bool = False,
    ) -> dict[str, Any]:
        planner_status = dict(self.policy_source.cycle_planner_status())
        ready = dict(self._last_ready)
        now = float(self._clock())
        return {
            "enabled": True,
            "active": bool(self._active),
            "completed": bool(self._completed),
            "completed_now": bool(completed_now),
            "fault": str(self._fault_reason),
            "stop_reason": str(self._stop_reason),
            "event": str(self._last_event),
            "goal_changed": bool(goal_changed),
            "stop_policy": bool(stop_policy),
            "review_due": bool(self._review_due),
            "excursion_observed": bool(self._excursion_observed),
            "excursion_candidate_count": int(self._excursion_candidate_count),
            "goal_anchor_swing_qpos_rad": self._goal_anchor_swing_qpos,
            "cycle_elapsed_s": (
                0.0
                if self._cycle_started_s is None
                else max(0.0, now - self._cycle_started_s)
            ),
            "run_elapsed_s": (
                0.0
                if self._run_started_s is None
                else max(0.0, now - self._run_started_s)
            ),
            "ready_actual_side": str(ready.get("actual_side", "unknown")),
            "ready_blockers": [str(value) for value in ready.get("blockers", ())],
            "ready_window_complete": bool(ready.get("window_complete", False)),
            "ready_swing_stable": bool(ready.get("swing_stable", False)),
            "ready_target_supported": bool(ready.get("target_support_window", False)),
            "ready_sample_gap_ok": bool(ready.get("sample_gap_ok", False)),
            "ready_swing_qpos_rad": ready.get("swing_qpos_current_rad"),
            "ready_swing_qvel_abs_max_rad_s": ready.get("swing_qvel_abs_max_rad_s"),
            "planner": planner_status,
        }

    def _fault(self, reason: str) -> dict[str, Any]:
        self._active = False
        self._fault_reason = str(reason)
        self._stop_reason = str(reason)
        self._last_event = "fault"
        return self.status(stop_policy=True)

    def _commit_goal_state(self, *, goal: Any, now_s: float) -> None:
        current_swing = self.ready.latest_swing_qpos
        if current_swing is None:
            raise ScriptedCycleRuntimeError(
                "goal commit requires a current swing observation"
            )
        self._goal_anchor_swing_qpos = float(current_swing)
        self._cycle_started_s = float(now_s)
        self._excursion_candidate_count = 0
        self._excursion_observed = False
        self._review_due = False
        self._stop_reason = ""
        self._fault_reason = ""
        if str(goal.target_side) not in {"A", "B"}:
            raise ScriptedCycleRuntimeError("planner goal target side must be A or B")

    def _reset_execution(self) -> None:
        self._active = False
        self._completed = False
        self._fault_reason = ""
        self._stop_reason = ""
        self._last_event = ""
        self._review_due = False
        self._run_started_s: float | None = None
        self._cycle_started_s: float | None = None
        self._goal_anchor_swing_qpos: float | None = None
        self._excursion_candidate_count = 0
        self._excursion_observed = False
        self._last_ready = self.ready.snapshot()


def _resolve_contract_path(raw: Any, *, bundle_dir: str | Path | None) -> Path:
    if raw is None or not str(raw).strip():
        raise ScriptedCycleRuntimeError(
            "scripted-cycle ready_contract path is required"
        )
    source = Path(str(raw))
    candidates = [source]
    if bundle_dir is not None:
        bundle = Path(bundle_dir)
        candidates.extend(
            (bundle / source, bundle / source.name, bundle / "contracts" / source.name)
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ScriptedCycleRuntimeError(
        "scripted-cycle ready contract does not exist: "
        + ", ".join(str(value) for value in candidates)
    )


def _load_target_ranges(
    raw: Any, *, bundle_dir: str | Path | None
) -> dict[str, tuple[float, float]]:
    path = _resolve_contract_path(raw, bundle_dir=bundle_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScriptedCycleRuntimeError(
            f"cannot read target-region contract {path}: {exc}"
        ) from exc
    if payload.get("schema") != "real_transition_target_release_contract_v1":
        raise ScriptedCycleRuntimeError("target-region contract schema is invalid")
    decision = dict(payload.get("decision_region", {}) or {})
    a_range = _finite_range(
        decision.get("train_A_endpoint_range_rad"), name="left target range"
    )
    b_range = _finite_range(
        decision.get("swing_qpos_range_rad"), name="right target range"
    )
    return {"A": a_range, "B": b_range}


def _finite_range(value: Any, *, name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ScriptedCycleRuntimeError(f"{name} must contain two finite values")
    low, high = float(array[0]), float(array[1])
    if low >= high:
        raise ScriptedCycleRuntimeError(f"{name} must be increasing")
    return low, high


def _observation_timestamp_ns(observation: Mapping[str, Any]) -> int:
    for key in ("timestamp_ns", "sync_timestamp_ns", "joint_timestamp_ns"):
        try:
            value = int(observation.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return time.time_ns()


def _shortest_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))
