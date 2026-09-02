"""Live lifecycle owner for a planner-conditioned scripted ACT sequence.

The neural policy owns continuous joint actions for one committed cycle.  This
module owns the goal/ready lifecycle around it and an optional measured-state
swing landing layer: verify the initial ready side, commit the next scripted
goal, require a real swing excursion, taper swing near the target, accept a
stable target-side ready window, and then commit the following goal.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from testbed.tasks.home_side_contract import (
    classify_ready_swing_qpos,
    validate_rule_ready_contract,
)
from testbed.tasks.task_state_auto_progress import (
    TaskStateAutoProgress,
    resolve_auto_progress_contract_path,
)


class ScriptedCycleRuntimeError(RuntimeError):
    """Raised when a live scripted-cycle contract cannot be satisfied."""


@dataclass(frozen=True)
class SwingLandingConfig:
    """Data-calibrated swing release and low-authority endpoint correction."""

    enabled: bool
    coast_stop_time_s: float
    edge_margin_rad: float
    p_gain: float
    d_gain: float
    return_confirm_drop_rad: float
    return_min_qvel_rad_s: float
    pd_blend_width_rad: float
    pd_blend_time_s: float
    policy_gain_time_s: float
    min_action_positive: float
    min_action_negative: float
    max_action_positive: float
    max_action_negative: float
    qvel_stable_rad_s: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> SwingLandingConfig:
        cfg = dict(raw or {})
        result = cls(
            enabled=bool(cfg.get("enabled", False)),
            coast_stop_time_s=float(cfg.get("coast_stop_time_s", 0.50)),
            edge_margin_rad=float(cfg.get("edge_margin_rad", 0.03)),
            p_gain=float(cfg.get("p_gain", 0.60)),
            d_gain=float(cfg.get("d_gain", 0.12)),
            return_confirm_drop_rad=float(cfg.get("return_confirm_drop_rad", 0.05)),
            return_min_qvel_rad_s=float(cfg.get("return_min_qvel_rad_s", 0.05)),
            pd_blend_width_rad=float(cfg.get("pd_blend_width_rad", 0.03)),
            pd_blend_time_s=float(cfg.get("pd_blend_time_s", 0.25)),
            policy_gain_time_s=float(cfg.get("policy_gain_time_s", 0.25)),
            min_action_positive=float(cfg.get("min_action_positive", 0.661)),
            min_action_negative=float(cfg.get("min_action_negative", 0.721)),
            max_action_positive=float(cfg.get("max_action_positive", 0.72)),
            max_action_negative=float(cfg.get("max_action_negative", 0.78)),
            qvel_stable_rad_s=float(cfg.get("qvel_stable_rad_s", 0.015)),
        )
        finite = (
            result.coast_stop_time_s,
            result.edge_margin_rad,
            result.p_gain,
            result.d_gain,
            result.return_confirm_drop_rad,
            result.return_min_qvel_rad_s,
            result.pd_blend_width_rad,
            result.pd_blend_time_s,
            result.policy_gain_time_s,
            result.min_action_positive,
            result.min_action_negative,
            result.max_action_positive,
            result.max_action_negative,
            result.qvel_stable_rad_s,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ScriptedCycleRuntimeError("swing_landing values must be finite")
        if result.coast_stop_time_s <= 0.0:
            raise ScriptedCycleRuntimeError(
                "swing_landing.coast_stop_time_s must be positive"
            )
        if result.edge_margin_rad < 0.0:
            raise ScriptedCycleRuntimeError(
                "swing_landing.edge_margin_rad must be non-negative"
            )
        if result.p_gain < 0.0 or result.d_gain < 0.0:
            raise ScriptedCycleRuntimeError(
                "swing_landing P/D gains must be non-negative"
            )
        if result.return_confirm_drop_rad <= 0.0:
            raise ScriptedCycleRuntimeError(
                "swing_landing.return_confirm_drop_rad must be positive"
            )
        if result.return_min_qvel_rad_s <= 0.0:
            raise ScriptedCycleRuntimeError(
                "swing_landing.return_min_qvel_rad_s must be positive"
            )
        if result.pd_blend_width_rad <= 0.0:
            raise ScriptedCycleRuntimeError(
                "swing_landing.pd_blend_width_rad must be positive"
            )
        if result.pd_blend_time_s <= 0.0:
            raise ScriptedCycleRuntimeError(
                "swing_landing.pd_blend_time_s must be positive"
            )
        if result.policy_gain_time_s <= 0.0:
            raise ScriptedCycleRuntimeError(
                "swing_landing.policy_gain_time_s must be positive"
            )
        if result.qvel_stable_rad_s <= 0.0:
            raise ScriptedCycleRuntimeError(
                "swing_landing.qvel_stable_rad_s must be positive"
            )
        if not 0.0 <= result.min_action_positive <= result.max_action_positive <= 1.0:
            raise ScriptedCycleRuntimeError(
                "swing_landing positive action limits are invalid"
            )
        if not 0.0 <= result.min_action_negative <= result.max_action_negative <= 1.0:
            raise ScriptedCycleRuntimeError(
                "swing_landing negative action limits are invalid"
            )
        return result


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

    @property
    def latest_swing_qvel(self) -> float | None:
        if not self._samples:
            return None
        return float(self._samples[-1][2][0])

    @property
    def latest_qpos(self) -> np.ndarray | None:
        if not self._samples:
            return None
        return self._samples[-1][1].copy()

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
        swing_landing: Mapping[str, Any] | SwingLandingConfig | None = None,
        cycle_review_s: float = 45.0,
        cycle_stop_s: float = 60.0,
        run_stop_s: float = 240.0,
        stop_on_wrong_ready: bool = True,
        task_state_v2: Mapping[str, Any] | None = None,
        task_state_auto_progress: TaskStateAutoProgress | None = None,
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
        self.swing_landing = (
            swing_landing
            if isinstance(swing_landing, SwingLandingConfig)
            else SwingLandingConfig.from_mapping(swing_landing)
        )
        self.cycle_review_s = float(cycle_review_s)
        self.cycle_stop_s = float(cycle_stop_s)
        self.run_stop_s = float(run_stop_s)
        self.stop_on_wrong_ready = bool(stop_on_wrong_ready)
        task_state_cfg = dict(task_state_v2 or {})
        self.task_state_v2_enabled = bool(task_state_cfg.get("enabled", False))
        self.task_state_advance_source = str(
            task_state_cfg.get("advance_source", "operator_mark")
        )
        self.require_excursion_before_work_complete = bool(
            task_state_cfg.get("require_excursion_before_work_complete", True)
        )
        self.task_state_auto_progress = task_state_auto_progress
        self._clock = clock
        policy_requires_task_state = bool(
            getattr(policy_source, "task_state_v2_enabled", False)
        )
        if policy_requires_task_state and not self.task_state_v2_enabled:
            raise ScriptedCycleRuntimeError(
                "task-state-v2 ACT requires scripted_cycle.task_state_v2.enabled"
            )
        if self.task_state_v2_enabled and not policy_requires_task_state:
            raise ScriptedCycleRuntimeError(
                "scripted_cycle.task_state_v2 is enabled but the policy does not "
                "declare real_transition_task_state_v2"
            )
        if self.task_state_v2_enabled:
            if self.task_state_advance_source not in {
                "operator_mark",
                "automatic_policy_state",
            }:
                raise ScriptedCycleRuntimeError(
                    "task_state_v2.advance_source must be operator_mark or "
                    "automatic_policy_state"
                )
            for method_name in (
                "set_task_dig_complete",
                "set_task_return_commit",
            ):
                if not callable(getattr(policy_source, method_name, None)):
                    raise ScriptedCycleRuntimeError(
                        f"task-state-v2 policy source must implement {method_name}()"
                    )
            if (
                self.task_state_advance_source == "automatic_policy_state"
                and self.task_state_auto_progress is None
            ):
                raise ScriptedCycleRuntimeError(
                    "automatic task-state-v2 requires its frozen progress contract"
                )
            if (
                self.task_state_advance_source == "operator_mark"
                and self.task_state_auto_progress is not None
            ):
                raise ScriptedCycleRuntimeError(
                    "operator-owned task-state-v2 cannot attach automatic progress"
                )
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
        task_state_cfg = dict(cfg.get("task_state_v2", {}) or {})
        auto_progress = None
        if task_state_cfg.get("advance_source") == "automatic_policy_state":
            auto_progress = TaskStateAutoProgress.from_path(
                resolve_auto_progress_contract_path(
                    task_state_cfg.get("auto_progress_contract"),
                    bundle_dir=bundle_dir,
                )
            )
        return cls(
            policy_source=policy_source,
            ready_contract=payload,
            target_ranges=target_ranges,
            swing_landing=cfg.get("swing_landing"),
            cycle_review_s=float(cfg.get("cycle_review_s", 45.0)),
            cycle_stop_s=float(cfg.get("cycle_stop_s", 60.0)),
            run_stop_s=float(cfg.get("run_stop_s", 240.0)),
            stop_on_wrong_ready=bool(cfg.get("stop_on_wrong_ready", True)),
            task_state_v2=task_state_cfg,
            task_state_auto_progress=auto_progress,
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
        observed_qpos = np.asarray(observation.get("qpos"), dtype=np.float64).reshape(
            -1
        )
        status = self.ready.update(
            timestamp_ns=timestamp_ns,
            qpos=observed_qpos,
            qvel=observation.get("qvel"),
        )
        self._last_observation_qpos = observed_qpos.copy()
        if self._active and self.task_state_auto_progress is not None:
            self.task_state_auto_progress.observe_qpos(observed_qpos)
        self._last_ready = status
        current_swing = self.ready.latest_swing_qpos
        if self._active and current_swing is not None:
            if self._goal_max_swing_qpos is None:
                self._goal_max_swing_qpos = float(current_swing)
            else:
                self._goal_max_swing_qpos = max(
                    self._goal_max_swing_qpos,
                    float(current_swing),
                )
        return self.status()

    def activation_blocker(self) -> str:
        ready = self._last_ready
        blockers = list(ready.get("blockers", ()))
        if blockers:
            return "initial_ready:" + ",".join(str(value) for value in blockers)
        actual_side = str(ready.get("actual_side", "unknown"))
        available_sides = tuple(
            getattr(self.planner, "available_initial_sides", ()) or ()
        )
        if available_sides:
            if actual_side not in available_sides:
                return f"initial_side_unavailable:actual_{actual_side}"
            return ""
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
        actual_side = str(self._last_ready["actual_side"])
        select_initial_side = getattr(self.planner, "select_initial_side", None)
        if callable(select_initial_side):
            select_initial_side(actual_side)
        goal = self.policy_source.commit_cycle_goal()
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
        self._task_state_changed_this_tick = False
        self._task_state_applied_event = ""
        if self.task_state_auto_progress is not None:
            applied_event, changed = self.task_state_auto_progress.apply_pending(
                self.policy_source
            )
            self._task_state_changed_this_tick = bool(changed)
            self._task_state_applied_event = str(applied_event)
            if changed:
                self._last_event = f"automatic_{applied_event}_applied"
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
            if delta < threshold:
                self._excursion_candidate_count = 0
                return self.status()
            self._excursion_candidate_count += 1
            if self._excursion_candidate_count < required:
                return self.status()
            self._excursion_observed = True
            excursion_setter = getattr(
                self.policy_source, "set_cycle_excursion_observed", None
            )
            excursion_changed = bool(
                callable(excursion_setter) and excursion_setter(observed=True)
            )
            self._last_event = "cycle_excursion_confirmed"
            return self.status(excursion_changed=excursion_changed)

        if not self._policy_return_phase_latched:
            peak = self._goal_max_swing_qpos
            swing_qvel = self.ready.latest_swing_qvel
            if peak is not None and swing_qvel is not None:
                return_drop = float(peak) - float(current_swing)
                if (
                    return_drop >= self.swing_landing.return_confirm_drop_rad
                    and swing_qvel <= -self.swing_landing.return_min_qvel_rad_s
                ):
                    self._policy_return_phase_latched = True
                    phase_setter = getattr(self.policy_source, "set_cycle_phase", None)
                    phase_changed = bool(
                        callable(phase_setter) and phase_setter(return_phase=True)
                    )
                    self._last_event = "cycle_return_phase_confirmed"
                    if phase_changed:
                        return self.status(phase_changed=True)

        if self.swing_landing.enabled and not self._return_phase_latched:
            # A/B endpoint ranges are crossed during outbound excavation as
            # well as during the final leftward return.  Never close a cycle
            # on the outbound crossing.
            return self.status()

        if self.task_state_v2_enabled:
            planner_status = dict(self.policy_source.cycle_planner_status())
            if not bool(planner_status.get("task_return_commit", False)):
                # A premature model return must never be accepted as a cycle
                # boundary before the explicit task permission is latched.
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

    def advance_task_state(self) -> dict[str, Any]:
        """Apply one explicit operator mark to the causal task-state token.

        The first accepted mark latches work complete.  The second latches
        return permission and exposes the already committed next target.  No
        transition is inferred from qpos, qvel, elapsed time, or policy output.
        """

        if not self.task_state_v2_enabled:
            raise ScriptedCycleRuntimeError(
                "task-state-v2 operator advance is disabled"
            )
        if self.task_state_advance_source != "operator_mark":
            raise ScriptedCycleRuntimeError(
                "task-state-v2 is owned by automatic_policy_state"
            )
        if not self._active:
            raise ScriptedCycleRuntimeError(
                "task-state-v2 operator advance requires an active scripted cycle"
            )
        planner_status = dict(self.policy_source.cycle_planner_status())
        if not bool(planner_status.get("committed", False)):
            raise ScriptedCycleRuntimeError(
                "task-state-v2 operator advance requires a committed goal"
            )
        work_complete = bool(planner_status.get("task_dig_complete", False))
        return_commit = bool(planner_status.get("task_return_commit", False))
        if not work_complete:
            if (
                self.require_excursion_before_work_complete
                and not self._excursion_observed
            ):
                raise ScriptedCycleRuntimeError(
                    "work_complete_requires_confirmed_positive_excursion"
                )
            changed = bool(self.policy_source.set_task_dig_complete(completed=True))
            self._last_event = "task_work_complete_committed"
            return self.status(task_state_changed=changed)
        if not return_commit:
            changed = bool(self.policy_source.set_task_return_commit(committed=True))
            self._last_event = "task_return_committed"
            return self.status(task_state_changed=changed)
        self._last_event = "task_state_advance_ignored_already_committed"
        return self.status(task_state_advance_ignored=True)

    def observe_policy_action(self, action: Any) -> dict[str, Any]:
        """Update automatic progress after one policy action without applying it."""

        if not self._active or self.task_state_auto_progress is None:
            return self.status()
        planner_status = dict(self.policy_source.cycle_planner_status())
        self.task_state_auto_progress.observe_policy_action(
            action,
            excursion_observed=bool(self._excursion_observed),
            task_dig_complete=bool(planner_status.get("task_dig_complete", False)),
            task_return_commit=bool(planner_status.get("task_return_commit", False)),
        )
        return self.status()

    def shape_policy_action(
        self,
        action: Any,
        observation: Mapping[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Blend ACT swing after measured target entry and bound PD correction."""

        shaped = np.asarray(action, dtype=np.float32).reshape(4).copy()
        diagnostics: dict[str, Any] = {
            "swing_landing_enabled": int(self.swing_landing.enabled),
            "swing_landing_active": 0,
            "swing_landing_mode": "disabled",
            "swing_landing_scale": 1.0,
            "swing_landing_policy_gain": 1.0,
            "swing_landing_policy_gain_desired": 1.0,
            "swing_landing_original_action": float(shaped[0]),
            "swing_landing_output_action": float(shaped[0]),
        }
        if not self.swing_landing.enabled or not self._active:
            return shaped, diagnostics
        goal = getattr(self.planner, "committed_goal", None)
        target_side = "" if goal is None else str(goal.target_side)
        if target_side not in {"A", "B"}:
            diagnostics["swing_landing_mode"] = "no_committed_target"
            return shaped, diagnostics
        bounds = self.ready.target_ranges.get(target_side)
        if bounds is None:
            diagnostics["swing_landing_mode"] = "target_range_missing"
            return shaped, diagnostics

        qpos = np.asarray(observation.get("qpos"), dtype=np.float64).reshape(-1)
        qvel = np.asarray(observation.get("qvel"), dtype=np.float64).reshape(-1)
        if qpos.shape != (4,) or qvel.shape != (4,):
            raise ScriptedCycleRuntimeError(
                "swing landing requires qpos/qvel shape (4,)"
            )
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ScriptedCycleRuntimeError("swing landing requires finite qpos/qvel")

        low, high = [float(value) for value in bounds]
        width = high - low
        if self.swing_landing.edge_margin_rad * 2.0 >= width:
            raise ScriptedCycleRuntimeError(
                "swing_landing.edge_margin_rad is too large for target range"
            )
        # Field cycles reach both A and B endpoints on the same return stroke:
        # swing qpos falls from the right-hand excavation excursion.  The
        # target side changes the endpoint range, not the landing direction.
        entry_edge = high
        far_edge = low
        center = 0.5 * (low + high)
        guard_edge = low
        swing_qpos = float(qpos[0])
        swing_qvel = float(qvel[0])
        timestamp_ns = _observation_timestamp_ns(observation)
        if self._landing_last_timestamp_ns is None:
            blend_dt_s = 0.0
        else:
            blend_dt_s = float(
                np.clip(
                    (timestamp_ns - self._landing_last_timestamp_ns) / 1e9,
                    0.0,
                    0.1,
                )
            )
        self._landing_last_timestamp_ns = timestamp_ns

        peak_qpos = (
            swing_qpos
            if self._goal_max_swing_qpos is None
            else float(self._goal_max_swing_qpos)
        )
        return_drop = max(0.0, peak_qpos - swing_qpos)
        if (
            not self._return_phase_latched
            and self._excursion_observed
            and return_drop >= self.swing_landing.return_confirm_drop_rad
            and swing_qvel <= -self.swing_landing.return_min_qvel_rad_s
            and float(shaped[0]) < 0.0
        ):
            self._return_phase_latched = True

        if not self._return_phase_latched:
            diagnostics.update(
                {
                    "swing_landing_mode": "waiting_return_phase",
                    "swing_landing_target_side": target_side,
                    "swing_landing_target_low_rad": low,
                    "swing_landing_target_high_rad": high,
                    "swing_landing_target_center_rad": center,
                    "swing_landing_guard_edge_rad": guard_edge,
                    "swing_landing_original_action": float(action[0]),
                    "swing_landing_output_action": float(shaped[0]),
                    "swing_landing_qpos_rad": swing_qpos,
                    "swing_landing_qvel_rad_s": swing_qvel,
                    "swing_landing_far_edge_rad": far_edge,
                    "swing_landing_return_phase": 0,
                    "swing_landing_peak_qpos_rad": peak_qpos,
                    "swing_landing_return_drop_rad": return_drop,
                    "swing_landing_pd_blend": 0.0,
                    "swing_landing_pd_blend_desired": 0.0,
                }
            )
            return shaped, diagnostics

        projected_qpos = swing_qpos + (
            self.swing_landing.coast_stop_time_s * swing_qvel
        )
        approach_span = entry_edge - center
        progress = (
            (entry_edge - swing_qpos) / approach_span if approach_span > 0.0 else 0.0
        )
        # Do not attenuate ACT before the measured joint has entered the
        # target's supported range.  A coast projection is useful telemetry,
        # but the hydraulic stopping distance is load-dependent; using it as
        # the gain boundary can leave the machine stalled outside the target.
        linear_scale = float(1.0 - np.clip(progress, 0.0, 1.0))
        mode = "model"
        desired_policy_gain = (
            linear_scale if float(shaped[0]) < 0.0 else self._landing_policy_gain
        )
        if desired_policy_gain < self._landing_policy_gain:
            self._landing_policy_gain = max(
                desired_policy_gain,
                self._landing_policy_gain
                - (blend_dt_s / self.swing_landing.policy_gain_time_s),
            )
        if float(shaped[0]) < 0.0 and self._landing_policy_gain < 1.0:
            shaped[0] = np.float32(float(shaped[0]) * self._landing_policy_gain)
            mode = "policy_gain"

        # PD is not permitted inside the supported endpoint range.  It starts
        # only after measured qpos has crossed the low/left boundary, then
        # blends in over time so it cannot replace ACT in one control tick.
        overshoot_depth = max(0.0, low - swing_qpos)
        desired_blend = float(
            np.clip(
                overshoot_depth / self.swing_landing.pd_blend_width_rad,
                0.0,
                1.0,
            )
        )
        max_blend_delta = blend_dt_s / self.swing_landing.pd_blend_time_s
        self._landing_pd_blend = float(
            self._landing_pd_blend
            + np.clip(
                desired_blend - self._landing_pd_blend,
                -max_blend_delta,
                max_blend_delta,
            )
        )
        pd_raw = (
            self.swing_landing.p_gain * (low - swing_qpos)
            - self.swing_landing.d_gain * swing_qvel
        )
        pd_action = 0.0
        if overshoot_depth > 0.0 and pd_raw > 0.0:
            pd_action = float(
                np.clip(
                    pd_raw,
                    self.swing_landing.min_action_positive,
                    self.swing_landing.max_action_positive,
                )
            )
        if self._landing_pd_blend > 0.0:
            policy_component = float(shaped[0])
            shaped[0] = np.float32(
                (1.0 - self._landing_pd_blend) * policy_component
                + self._landing_pd_blend * pd_action
            )
            mode = "pd_blend" if pd_action > 0.0 else "overshoot_gain_hold"

        diagnostics.update(
            {
                "swing_landing_active": int(mode != "model"),
                "swing_landing_mode": mode,
                "swing_landing_target_side": target_side,
                "swing_landing_target_low_rad": low,
                "swing_landing_target_high_rad": high,
                "swing_landing_target_center_rad": center,
                "swing_landing_guard_edge_rad": guard_edge,
                "swing_landing_projected_qpos_rad": projected_qpos,
                "swing_landing_scale": linear_scale,
                "swing_landing_policy_gain": self._landing_policy_gain,
                "swing_landing_policy_gain_desired": desired_policy_gain,
                "swing_landing_pd_raw": pd_raw,
                "swing_landing_pd_action": pd_action,
                "swing_landing_pd_blend": self._landing_pd_blend,
                "swing_landing_pd_blend_desired": desired_blend,
                "swing_landing_overshoot_depth_rad": overshoot_depth,
                "swing_landing_original_action": float(action[0]),
                "swing_landing_output_action": float(shaped[0]),
                "swing_landing_qpos_rad": swing_qpos,
                "swing_landing_qvel_rad_s": swing_qvel,
                "swing_landing_far_edge_rad": far_edge,
                "swing_landing_return_phase": 1,
                "swing_landing_peak_qpos_rad": peak_qpos,
                "swing_landing_return_drop_rad": return_drop,
            }
        )
        return shaped, diagnostics

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
        excursion_changed: bool = False,
        phase_changed: bool = False,
        task_state_changed: bool = False,
        task_state_advance_ignored: bool = False,
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
            "excursion_changed": bool(excursion_changed),
            "phase_changed": bool(phase_changed),
            "task_state_changed": bool(
                task_state_changed or self._task_state_changed_this_tick
            ),
            "task_state_advance_ignored": bool(task_state_advance_ignored),
            "task_state_v2_enabled": bool(self.task_state_v2_enabled),
            "task_state_advance_source": str(self.task_state_advance_source),
            "task_state_require_excursion_before_work_complete": bool(
                self.require_excursion_before_work_complete
            ),
            "task_state_stage": _task_state_stage(planner_status),
            "task_state_applied_event": str(self._task_state_applied_event),
            "task_state_auto_progress": (
                {"enabled": False}
                if self.task_state_auto_progress is None
                else self.task_state_auto_progress.status()
            ),
            "stop_policy": bool(stop_policy),
            "review_due": bool(self._review_due),
            "excursion_observed": bool(self._excursion_observed),
            "excursion_candidate_count": int(self._excursion_candidate_count),
            "goal_anchor_swing_qpos_rad": self._goal_anchor_swing_qpos,
            "goal_max_swing_qpos_rad": self._goal_max_swing_qpos,
            "return_phase_latched": bool(self._return_phase_latched),
            "policy_return_phase_latched": bool(self._policy_return_phase_latched),
            "landing_policy_gain": float(self._landing_policy_gain),
            "landing_pd_blend": float(self._landing_pd_blend),
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
        self._goal_max_swing_qpos = float(current_swing)
        self._return_phase_latched = False
        self._policy_return_phase_latched = False
        self._landing_policy_gain = 1.0
        self._landing_pd_blend = 0.0
        self._landing_last_timestamp_ns = None
        self._cycle_started_s = float(now_s)
        self._excursion_candidate_count = 0
        self._excursion_observed = False
        self._review_due = False
        self._stop_reason = ""
        self._fault_reason = ""
        if str(goal.target_side) not in {"A", "B"}:
            raise ScriptedCycleRuntimeError("planner goal target side must be A or B")
        if self.task_state_auto_progress is not None:
            if self._last_observation_qpos is None:
                raise ScriptedCycleRuntimeError(
                    "automatic task progress requires a current qpos observation"
                )
            self.task_state_auto_progress.reset_goal(self._last_observation_qpos)

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
        self._goal_max_swing_qpos: float | None = None
        self._return_phase_latched = False
        self._policy_return_phase_latched = False
        self._landing_policy_gain = 1.0
        self._landing_pd_blend = 0.0
        self._landing_last_timestamp_ns: int | None = None
        self._excursion_candidate_count = 0
        self._excursion_observed = False
        self._task_state_changed_this_tick = False
        self._task_state_applied_event = ""
        self._last_observation_qpos = self.ready.latest_qpos
        if self.task_state_auto_progress is not None:
            self.task_state_auto_progress.reset()
        self._last_ready = self.ready.snapshot()


def _task_state_stage(planner_status: Mapping[str, Any]) -> str:
    if not bool(planner_status.get("task_state_v2_enabled", False)):
        return "disabled"
    if not bool(planner_status.get("committed", False)):
        return "waiting_goal"
    if bool(planner_status.get("task_return_commit", False)):
        return "return_committed"
    if bool(planner_status.get("task_dig_complete", False)):
        return "work_complete"
    return "work"


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
