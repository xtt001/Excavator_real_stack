"""Policy-backed action source for real-machine shadow and guarded control."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from testbed.actions.base import ActionInfo, ActionSource
from testbed.backends.real.contracts import as_real_action
from testbed.data.action_primitive_islands import (
    ACTION_PRIMITIVE_KEY,
    PRIMITIVE_NAMES,
)
from testbed.data.task_state_v2 import TASK_STATE_V2_KEY, task_state_vector
from testbed.data.work_return_context import WORK_CONTEXT_KEY
from testbed.tasks.real_transition_excursion import EXCURSION_OBSERVED_KEY
from testbed.tasks.real_transition_phase import CYCLE_PHASE_KEY
from testbed.tasks.real_transition_return_commit import RETURN_COMMIT_KEY

POLICY_OUTPUT_MODES = ("control", "shadow_zero")
POLICY_QVEL_MODES = ("raw", "zero", "qpos_diff")
ACTION_AXIS_NAMES = ("swing", "boom", "stick", "bucket")


@dataclass(frozen=True)
class DeadzoneAssistConfig:
    enabled: bool
    axis_enabled: np.ndarray
    trigger_fraction: np.ndarray
    margin: np.ndarray
    deadzone_positive: np.ndarray
    deadzone_negative: np.ndarray
    min_consecutive_steps: int


@dataclass(frozen=True)
class PolicyActionSourceState:
    """Complete mutable action-pipeline state for deterministic branch replay."""

    step: int
    record_start_pending: bool
    last_qpos: np.ndarray | None
    last_obs_time_ns: int | None
    filtered_qvel: np.ndarray
    assist_last_sign: np.ndarray
    assist_consecutive_steps: np.ndarray
    policy_state: Any
    runtime_gate_state: Any | None
    planner_state: Any | None = None
    excursion_observed: float = 0.0
    excursion_observed_epoch: int = 0
    cycle_phase: float = 0.0
    cycle_phase_epoch: int = 0
    return_commit: float = 0.0
    return_commit_epoch: int = 0
    task_dig_complete: float = 0.0
    task_return_commit: float = 0.0
    task_state_epoch: int = 0


class PolicyActionSource(ActionSource):
    """
    Wrap a trained policy as an ActionSource.

    ``output_mode=control`` returns the scaled policy action. ``shadow_zero``
    records policy predictions in diagnostics but returns zero command, which is
    the safest first live test mode.
    """

    def __init__(
        self,
        *,
        policy: Any,
        source_id: str,
        camera_name: str = "fpv",
        camera_names: list[str] | tuple[str, ...] | None = None,
        action_scale: float | list[float] | tuple[float, ...] | np.ndarray = 1.0,
        clip: float = 1.0,
        output_mode: str = "shadow_zero",
        qvel_mode: str = "raw",
        qvel_diff_tau_s: float = 0.15,
        qvel_diff_clip_rad_s: float
        | list[float]
        | tuple[float, ...]
        | np.ndarray = 2.0,
        fail_safe_zero: bool = True,
        record_start_on_reset: bool = False,
        deadzone_assist: dict[str, Any] | None = None,
        runtime_gate_stack: Any | None = None,
        report_intent: bool = False,
        bundle_dir: str | Path | None = None,
        inference_warmup_steps: int = 0,
        frame_alignment_enabled: bool = False,
        cycle_planner: Any | None = None,
        reset_policy_on_goal: bool = True,
        reset_policy_on_phase_change: bool = True,
    ) -> None:
        if output_mode not in POLICY_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {POLICY_OUTPUT_MODES}, got {output_mode!r}"
            )
        if qvel_mode not in POLICY_QVEL_MODES:
            raise ValueError(
                f"qvel_mode must be one of {POLICY_QVEL_MODES}, got {qvel_mode!r}"
            )
        self._policy = policy
        self._source_id = str(source_id)
        self._camera_name = str(camera_name)
        self._camera_names = _camera_names_list(camera_names, default=self._camera_name)
        if not self._camera_names:
            raise ValueError("camera_names must not be empty")
        self._action_scale = _broadcast_action_scale(action_scale)
        self._clip = float(clip)
        self._output_mode = str(output_mode)
        self._qvel_mode = str(qvel_mode)
        self._qvel_diff_tau_s = max(0.0, float(qvel_diff_tau_s))
        self._qvel_diff_clip = _broadcast_qvel_clip(qvel_diff_clip_rad_s)
        self._fail_safe_zero = bool(fail_safe_zero)
        self._record_start_on_reset = bool(record_start_on_reset)
        self._deadzone_assist = _deadzone_assist_config(deadzone_assist)
        self._runtime_gate_stack = runtime_gate_stack
        self._report_intent = bool(report_intent)
        self._bundle_dir = None if bundle_dir is None else str(bundle_dir)
        self._inference_warmup_steps = int(inference_warmup_steps)
        self._frame_alignment_enabled = bool(frame_alignment_enabled)
        self._cycle_planner = cycle_planner
        self._reset_policy_on_goal = bool(reset_policy_on_goal)
        self._reset_policy_on_phase_change = bool(reset_policy_on_phase_change)
        policy_low_dim_keys = list(
            getattr(policy, "low_dim_keys", getattr(policy, "_low_dim_keys", ())) or ()
        )
        self._cycle_phase_enabled = CYCLE_PHASE_KEY in policy_low_dim_keys
        self._excursion_observed_enabled = EXCURSION_OBSERVED_KEY in policy_low_dim_keys
        self._return_commit_enabled = RETURN_COMMIT_KEY in policy_low_dim_keys
        self._task_state_v2_enabled = TASK_STATE_V2_KEY in policy_low_dim_keys
        self._excursion_observed = 0.0
        self._excursion_observed_epoch = 0
        self._cycle_phase = 0.0
        self._cycle_phase_epoch = 0
        self._return_commit = 0.0
        self._return_commit_epoch = 0
        self._task_dig_complete = 0.0
        self._task_return_commit = 0.0
        self._task_state_epoch = 0
        if self._cycle_planner is not None:
            for method_name in (
                "apply_condition",
                "commit_goal",
                "mark_target_ready",
                "reset",
            ):
                if not callable(getattr(self._cycle_planner, method_name, None)):
                    raise TypeError(f"cycle_planner must implement {method_name}()")
        if self._inference_warmup_steps < 0:
            raise ValueError("inference_warmup_steps must be non-negative")
        self._inference_prepared = False
        self._step = 0
        self._record_start_pending = self._record_start_on_reset
        self._last_qpos: np.ndarray | None = None
        self._last_obs_time_ns: int | None = None
        self._filtered_qvel = np.zeros(4, dtype=np.float32)
        self._assist_last_sign = np.zeros(4, dtype=np.int8)
        self._assist_consecutive_steps = np.zeros(4, dtype=np.int32)
        if self._cycle_planner is not None:
            self._cycle_planner.reset()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> PolicyActionSource:
        cfg = dict(config or {})
        bundle_dir = Path(cfg.get("bundle_dir", "policy_bundles/real_one_dig_v1"))
        cycle_planner = _build_cycle_planner(cfg.get("cycle_planner"))
        policy = load_act_policy_from_bundle(
            bundle_dir=bundle_dir,
            ckpt_path=cfg.get("ckpt_path"),
            resolved_config_path=cfg.get("resolved_config_path"),
            stats_path=cfg.get("stats_path"),
            device=cfg.get("device"),
            temporal_agg=bool(cfg.get("temporal_agg", True)),
            inference_precision=str(cfg.get("inference_precision", "fp32")),
            inference_compile=bool(cfg.get("inference_compile", False)),
            inference_compile_mode=str(
                cfg.get("inference_compile_mode", "reduce-overhead")
            ),
            inference_compile_dynamic=bool(cfg.get("inference_compile_dynamic", False)),
            device_uint8_preprocess=bool(cfg.get("device_uint8_preprocess", False)),
            temporal_aggregation_diagnostics=bool(
                cfg.get("temporal_aggregation_diagnostics", False)
            ),
        )
        camera_names = cfg.get("camera_names", cfg.get("cameras"))
        policy_camera_names = (
            list(getattr(policy, "camera_names"))
            if hasattr(policy, "camera_names")
            else []
        )
        if camera_names is None and policy_camera_names:
            camera_names = policy_camera_names
        parsed_camera_names = _camera_names_list(
            camera_names, default=str(cfg.get("camera", "fpv"))
        )
        if policy_camera_names and parsed_camera_names != policy_camera_names:
            raise ValueError(
                "teleop.policy.camera_names does not match the loaded policy bundle: "
                f"config={parsed_camera_names!r} bundle={policy_camera_names!r}"
            )
        camera_name = str(cfg.get("camera", parsed_camera_names[0]))
        runtime_gate_stack = None
        runtime_gates_cfg = cfg.get("runtime_gates")
        if runtime_gates_cfg is not None:
            if not isinstance(runtime_gates_cfg, dict):
                raise ValueError("teleop.policy.runtime_gates must be a mapping")
            if bool(runtime_gates_cfg.get("enabled", False)):
                from testbed.policies.runtime_gate_stack import RuntimeGateStack

                runtime_gate_stack = RuntimeGateStack.from_config(
                    runtime_gates_cfg,
                    default_bundle_dir=bundle_dir,
                )
        return cls(
            policy=policy,
            source_id=str(cfg.get("source_id", f"policy:act:{bundle_dir.name}")),
            camera_name=camera_name,
            camera_names=parsed_camera_names,
            action_scale=cfg.get("action_scale", 1.0),
            clip=float(cfg.get("clip", 1.0)),
            output_mode=str(cfg.get("output_mode", "shadow_zero")),
            qvel_mode=str(cfg.get("qvel_mode", "raw")),
            qvel_diff_tau_s=float(cfg.get("qvel_diff_tau_s", 0.15)),
            qvel_diff_clip_rad_s=cfg.get("qvel_diff_clip_rad_s", 2.0),
            fail_safe_zero=bool(cfg.get("fail_safe_zero", True)),
            record_start_on_reset=bool(cfg.get("record_start_on_reset", False)),
            deadzone_assist=cfg.get("deadzone_assist"),
            runtime_gate_stack=runtime_gate_stack,
            report_intent=bool(cfg.get("report_intent", False)),
            bundle_dir=bundle_dir,
            inference_warmup_steps=int(cfg.get("inference_warmup_steps", 0)),
            frame_alignment_enabled=bool(
                dict(cfg.get("frame_alignment", {}) or {}).get("enabled", False)
            ),
            cycle_planner=cycle_planner,
            reset_policy_on_goal=bool(cfg.get("reset_policy_on_goal", True)),
            reset_policy_on_phase_change=bool(
                cfg.get("reset_policy_on_phase_change", True)
            ),
        )

    def reset(self) -> None:
        self._step = 0
        self._record_start_pending = self._record_start_on_reset
        self._last_qpos = None
        self._last_obs_time_ns = None
        self._filtered_qvel.fill(0.0)
        self._assist_last_sign.fill(0)
        self._assist_consecutive_steps.fill(0)
        self._excursion_observed = 0.0
        self._excursion_observed_epoch = 0
        self._cycle_phase = 0.0
        self._cycle_phase_epoch = 0
        self._return_commit = 0.0
        self._return_commit_epoch = 0
        self._task_dig_complete = 0.0
        self._task_return_commit = 0.0
        self._task_state_epoch = 0
        if self._cycle_planner is not None:
            self._cycle_planner.reset()
        if self._runtime_gate_stack is not None:
            self._runtime_gate_stack.reset()
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def prepare(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Warm the inference runtime once, then restore clean episode state."""

        if self._inference_prepared or self._inference_warmup_steps == 0:
            return {
                "prepared": int(self._inference_prepared),
                "warmup_steps": 0,
                "elapsed_s": 0.0,
            }
        if self._cycle_planner is not None and (
            getattr(self._cycle_planner, "committed_goal", None) is None
        ):
            # Startup warm-up can happen before the first ready gate.  Do not
            # fabricate a goal merely to warm the network; the first real
            # inference will be enabled by an explicit commit_cycle_goal().
            return {
                "prepared": 0,
                "warmup_steps": 0,
                "elapsed_s": 0.0,
                "planner_waiting_for_commit": 1,
            }
        started = time.perf_counter()
        completed = 0
        try:
            policy_obs, _ = self._policy_obs(obs)
            if self._cycle_planner is not None:
                policy_obs = self._cycle_planner.apply_condition(policy_obs)
            if self._cycle_phase_enabled:
                policy_obs[CYCLE_PHASE_KEY] = np.asarray(
                    [self._cycle_phase], dtype=np.float32
                )
            if self._excursion_observed_enabled:
                policy_obs[EXCURSION_OBSERVED_KEY] = np.asarray(
                    [self._excursion_observed], dtype=np.float32
                )
            if self._return_commit_enabled:
                policy_obs[RETURN_COMMIT_KEY] = np.asarray(
                    [self._return_commit], dtype=np.float32
                )
            self._apply_task_state_v2(policy_obs)
            inference = (
                getattr(self._policy, "predict_action_and_intent", None)
                if self._runtime_gate_stack is not None or self._report_intent
                else getattr(self._policy, "predict", None)
            )
            if not callable(inference):
                raise TypeError(
                    "policy does not implement the configured inference API"
                )
            for _ in range(self._inference_warmup_steps):
                inference(policy_obs)
                completed += 1
        finally:
            self.reset()
        self._inference_prepared = True
        return {
            "prepared": 1,
            "warmup_steps": int(completed),
            "elapsed_s": float(time.perf_counter() - started),
        }

    def snapshot_state(self) -> PolicyActionSourceState:
        """Capture policy, gate, qvel, assist, and source sequence state."""

        policy_snapshot = getattr(self._policy, "snapshot_state", None)
        if not callable(policy_snapshot):
            raise TypeError("policy does not implement snapshot_state()")
        gate_state = None
        if self._runtime_gate_stack is not None:
            gate_snapshot = getattr(self._runtime_gate_stack, "snapshot_state", None)
            if not callable(gate_snapshot):
                raise TypeError(
                    "runtime gate stack does not implement snapshot_state()"
                )
            gate_state = gate_snapshot()
        planner_state = None
        if self._cycle_planner is not None:
            snapshot = getattr(self._cycle_planner, "snapshot_state", None)
            if not callable(snapshot):
                raise TypeError("cycle planner does not implement snapshot_state()")
            planner_state = snapshot()
        return PolicyActionSourceState(
            step=int(self._step),
            record_start_pending=bool(self._record_start_pending),
            last_qpos=(None if self._last_qpos is None else self._last_qpos.copy()),
            last_obs_time_ns=(
                None if self._last_obs_time_ns is None else int(self._last_obs_time_ns)
            ),
            filtered_qvel=self._filtered_qvel.copy(),
            assist_last_sign=self._assist_last_sign.copy(),
            assist_consecutive_steps=self._assist_consecutive_steps.copy(),
            policy_state=policy_snapshot(),
            runtime_gate_state=gate_state,
            planner_state=planner_state,
            excursion_observed=float(self._excursion_observed),
            excursion_observed_epoch=int(self._excursion_observed_epoch),
            cycle_phase=float(self._cycle_phase),
            cycle_phase_epoch=int(self._cycle_phase_epoch),
            return_commit=float(self._return_commit),
            return_commit_epoch=int(self._return_commit_epoch),
            task_dig_complete=float(self._task_dig_complete),
            task_return_commit=float(self._task_return_commit),
            task_state_epoch=int(self._task_state_epoch),
        )

    def restore_state(self, state: PolicyActionSourceState) -> None:
        """Restore a state produced by :meth:`snapshot_state` without aliasing."""

        if not isinstance(state, PolicyActionSourceState):
            raise TypeError("state must be PolicyActionSourceState")
        policy_restore = getattr(self._policy, "restore_state", None)
        if not callable(policy_restore):
            raise TypeError("policy does not implement restore_state()")
        if (self._runtime_gate_stack is None) != (state.runtime_gate_state is None):
            raise ValueError("runtime gate state/config mismatch")
        if (self._cycle_planner is None) != (state.planner_state is None):
            raise ValueError("cycle planner state/config mismatch")
        self._step = int(state.step)
        self._record_start_pending = bool(state.record_start_pending)
        self._last_qpos = None if state.last_qpos is None else state.last_qpos.copy()
        self._last_obs_time_ns = (
            None if state.last_obs_time_ns is None else int(state.last_obs_time_ns)
        )
        self._filtered_qvel = np.asarray(state.filtered_qvel, dtype=np.float32).copy()
        self._assist_last_sign = np.asarray(
            state.assist_last_sign, dtype=np.int8
        ).copy()
        self._assist_consecutive_steps = np.asarray(
            state.assist_consecutive_steps, dtype=np.int32
        ).copy()
        self._excursion_observed = float(state.excursion_observed)
        self._excursion_observed_epoch = int(state.excursion_observed_epoch)
        self._cycle_phase = float(state.cycle_phase)
        self._cycle_phase_epoch = int(state.cycle_phase_epoch)
        self._return_commit = float(state.return_commit)
        self._return_commit_epoch = int(state.return_commit_epoch)
        self._task_dig_complete = float(state.task_dig_complete)
        self._task_return_commit = float(state.task_return_commit)
        self._task_state_epoch = int(state.task_state_epoch)
        policy_restore(state.policy_state)
        if self._runtime_gate_stack is not None:
            gate_restore = getattr(self._runtime_gate_stack, "restore_state", None)
            if not callable(gate_restore):
                raise TypeError("runtime gate stack does not implement restore_state()")
            gate_restore(state.runtime_gate_state)
        if self._cycle_planner is not None:
            planner_restore = getattr(self._cycle_planner, "restore_state", None)
            if not callable(planner_restore):
                raise TypeError("cycle planner does not implement restore_state()")
            planner_restore(state.planner_state)

    @property
    def cycle_planner(self) -> Any | None:
        """Return the optional goal-only planner attached to this source."""

        return self._cycle_planner

    @property
    def task_state_v2_enabled(self) -> bool:
        """Whether the loaded ACT checkpoint requires the five-value task token."""

        return bool(self._task_state_v2_enabled)

    def commit_cycle_goal(self) -> Any:
        """Commit the next planner goal before requesting policy actions."""

        if self._cycle_planner is None:
            raise RuntimeError("no cycle planner is attached")
        goal = self._cycle_planner.commit_goal()
        self._excursion_observed = 0.0
        self._excursion_observed_epoch = int(getattr(goal, "goal_epoch", 0)) * 2
        self._cycle_phase = 0.0
        self._cycle_phase_epoch = int(getattr(goal, "goal_epoch", 0)) * 2
        self._return_commit = 0.0
        self._return_commit_epoch = int(getattr(goal, "goal_epoch", 0)) * 2
        self._task_dig_complete = 0.0
        self._task_return_commit = 0.0
        self._task_state_epoch = int(getattr(goal, "goal_epoch", 0)) * 3
        # A committed goal starts a new causal action sequence.  ACT's
        # temporal aggregation, cached chunk, visual history, and any
        # factorized aggregator must not carry proposals generated under the
        # previous goal into the first tick of this goal.  The planner still
        # owns only the condition; resetting the policy here is solely an
        # inference-cache lifecycle boundary.
        if self._reset_policy_on_goal:
            policy_reset = getattr(self._policy, "reset", None)
            if callable(policy_reset):
                policy_reset()
        return goal

    def set_cycle_excursion_observed(self, *, observed: bool) -> bool:
        """Latch causal positive excursion and reset conditioned ACT state."""

        if not self._excursion_observed_enabled:
            return False
        if (
            self._cycle_planner is not None
            and self._cycle_planner.committed_goal is None
        ):
            raise RuntimeError(
                "cycle excursion state requires a committed planner goal"
            )
        value = 1.0 if bool(observed) else 0.0
        if value == self._excursion_observed:
            return False
        if self._excursion_observed == 1.0 and value == 0.0:
            raise RuntimeError("cycle excursion state is monotonic within one goal")
        self._excursion_observed = value
        self._excursion_observed_epoch += 1
        if self._reset_policy_on_phase_change:
            policy_reset = getattr(self._policy, "reset", None)
            if callable(policy_reset):
                policy_reset()
        return True

    def set_cycle_phase(self, *, return_phase: bool) -> bool:
        """Latch the causal return phase and reset phase-conditioned ACT state."""

        if not self._cycle_phase_enabled:
            return False
        if (
            self._cycle_planner is not None
            and self._cycle_planner.committed_goal is None
        ):
            raise RuntimeError("cycle phase requires a committed planner goal")
        value = 1.0 if bool(return_phase) else 0.0
        if value == self._cycle_phase:
            return False
        if self._cycle_phase == 1.0 and value == 0.0:
            raise RuntimeError("cycle phase is monotonic within one committed goal")
        self._cycle_phase = value
        self._cycle_phase_epoch += 1
        if self._reset_policy_on_phase_change:
            policy_reset = getattr(self._policy, "reset", None)
            if callable(policy_reset):
                policy_reset()
        return True

    def set_return_commit(self, *, committed: bool) -> bool:
        """Latch planner-owned return intent without deriving it from observations."""

        if not self._return_commit_enabled:
            return False
        if (
            self._cycle_planner is not None
            and self._cycle_planner.committed_goal is None
        ):
            raise RuntimeError("return commit requires a committed planner goal")
        value = 1.0 if bool(committed) else 0.0
        if value == self._return_commit:
            return False
        if self._return_commit == 1.0 and value == 0.0:
            raise RuntimeError("return commit is monotonic within one committed goal")
        self._return_commit = value
        self._return_commit_epoch += 1
        if self._reset_policy_on_phase_change:
            policy_reset = getattr(self._policy, "reset", None)
            if callable(policy_reset):
                policy_reset()
        return True

    def set_task_dig_complete(self, *, completed: bool) -> bool:
        """Latch the operator/planner-owned work-complete event for task-state-v2."""

        if not self._task_state_v2_enabled:
            return False
        if (
            self._cycle_planner is not None
            and self._cycle_planner.committed_goal is None
        ):
            raise RuntimeError("task-state work complete requires a committed goal")
        value = 1.0 if bool(completed) else 0.0
        if value == self._task_dig_complete:
            return False
        if self._task_dig_complete == 1.0 and value == 0.0:
            raise RuntimeError("task-state work complete is monotonic within one goal")
        self._task_dig_complete = value
        self._task_state_epoch += 1
        self._reset_policy_for_task_state_change()
        return True

    def set_task_return_commit(self, *, committed: bool) -> bool:
        """Latch permission to expose the next target in task-state-v2."""

        if not self._task_state_v2_enabled:
            return False
        if (
            self._cycle_planner is not None
            and self._cycle_planner.committed_goal is None
        ):
            raise RuntimeError("task-state return commit requires a committed goal")
        value = 1.0 if bool(committed) else 0.0
        if value == self._task_return_commit:
            return False
        if self._task_return_commit == 1.0 and value == 0.0:
            raise RuntimeError("task-state return commit is monotonic within one goal")
        self._task_return_commit = value
        self._task_state_epoch += 1
        self._reset_policy_for_task_state_change()
        return True

    def mark_cycle_target_ready(self, realized_side: str) -> Any:
        """Advance the planner after an independently verified ready boundary."""

        if self._cycle_planner is None:
            raise RuntimeError("no cycle planner is attached")
        return self._cycle_planner.mark_target_ready(realized_side)

    def cycle_planner_status(self) -> dict[str, Any]:
        """Return compact planner status for a UI/logger."""

        planner = self._cycle_planner
        if planner is None:
            return {"enabled": False}
        goal = getattr(planner, "committed_goal", None)
        selected_planner = getattr(planner, "selected_planner", None)
        active_planner = selected_planner or planner
        planned_cycle_count = getattr(active_planner, "max_cycles", None)
        if planned_cycle_count is None:
            steps = getattr(active_planner, "steps", ()) or ()
            planned_cycle_count = len(steps) if steps else None
        return {
            "enabled": True,
            "cycle_index": int(getattr(planner, "cycle_index", -1)),
            "goal_epoch": int(getattr(planner, "goal_epoch", -1)),
            "done": bool(getattr(planner, "done", False)),
            "committed": goal is not None,
            "planner_type": (
                "side_matched_script"
                if hasattr(planner, "available_initial_sides")
                else "script"
            ),
            "selected_initial_side": str(
                getattr(planner, "selected_initial_side", "") or ""
            ),
            "available_initial_sides": list(
                getattr(planner, "available_initial_sides", ()) or ()
            ),
            "script_id": str(getattr(planner, "script_id", "") or ""),
            "script_path": str(getattr(planner, "source_path", "") or ""),
            "planned_cycle_count": (
                None if planned_cycle_count is None else int(planned_cycle_count)
            ),
            "target_side": None if goal is None else str(goal.target_side),
            "condition": (
                None if goal is None else [float(value) for value in goal.condition]
            ),
            "cycle_phase_enabled": bool(self._cycle_phase_enabled),
            "return_commit_enabled": bool(self._return_commit_enabled),
            "task_state_v2_enabled": bool(self._task_state_v2_enabled),
            "excursion_observed_enabled": bool(self._excursion_observed_enabled),
            "excursion_observed": float(self._excursion_observed),
            "excursion_observed_epoch": int(self._excursion_observed_epoch),
            "cycle_phase": float(self._cycle_phase),
            "cycle_phase_epoch": int(self._cycle_phase_epoch),
            "return_commit": float(self._return_commit),
            "return_commit_epoch": int(self._return_commit_epoch),
            "task_dig_complete": float(self._task_dig_complete),
            "task_return_commit": float(self._task_return_commit),
            "task_state_epoch": int(self._task_state_epoch),
            "task_state_v2": self._task_state_v2_for_goal(goal),
        }

    def next_action(self, obs: dict[str, Any]) -> tuple[np.ndarray, ActionInfo]:
        t0 = time.perf_counter()
        now_ns = time.time_ns()
        record_start_requested = self._consume_record_start_request()
        try:
            policy_obs, qvel_input = self._policy_obs(obs)
            planner_goal = None
            if self._cycle_planner is not None:
                policy_obs = self._cycle_planner.apply_condition(policy_obs)
                planner_goal = self._cycle_planner.committed_goal
            if self._cycle_phase_enabled:
                policy_obs[CYCLE_PHASE_KEY] = np.asarray(
                    [self._cycle_phase], dtype=np.float32
                )
            if self._excursion_observed_enabled:
                policy_obs[EXCURSION_OBSERVED_KEY] = np.asarray(
                    [self._excursion_observed], dtype=np.float32
                )
            if self._return_commit_enabled:
                policy_obs[RETURN_COMMIT_KEY] = np.asarray(
                    [self._return_commit], dtype=np.float32
                )
            self._apply_task_state_v2(policy_obs)
            gate_extras: dict[str, Any] = {}
            raw_gohome_requested = False
            intent_probabilities: np.ndarray | None = None
            if self._runtime_gate_stack is None and not self._report_intent:
                predicted_action = self._policy.predict(policy_obs)
            else:
                inference = getattr(self._policy, "predict_action_and_intent", None)
                if not callable(inference):
                    raise TypeError(
                        "runtime_gates/report_intent requires "
                        "policy.predict_action_and_intent(obs)"
                    )
                predicted_action, intent_probabilities = inference(policy_obs)
            policy_action = as_real_action(predicted_action, clip=False)
            policy_action = np.clip(policy_action, -self._clip, self._clip).astype(
                np.float32
            )
            gated_action = policy_action
            if self._runtime_gate_stack is not None:
                gate_result = self._runtime_gate_stack.step(
                    action=policy_action,
                    intent_probabilities=intent_probabilities,
                    qpos=policy_obs["qpos"],
                    qvel=qvel_input,
                )
                gated_action = as_real_action(gate_result.action, clip=False)
                gated_action = np.clip(gated_action, -self._clip, self._clip).astype(
                    np.float32
                )
                raw_gohome_requested = bool(gate_result.gohome_requested)
                gate_extras = dict(gate_result.diagnostics)
            scaled_action = np.clip(
                gated_action * self._action_scale,
                -self._clip,
                self._clip,
            ).astype(np.float32)
            assisted_action, assist_extras = self._apply_deadzone_assist(scaled_action)
            returned_action = (
                assisted_action
                if self._output_mode == "control"
                else np.zeros(4, dtype=np.float32)
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            extras = {
                "action_timestamp_ns": now_ns,
                "record_start_requested": record_start_requested,
                "policy_output_mode": self._output_mode,
                "policy_action": policy_action.copy(),
                "policy_scaled_action": scaled_action.copy(),
                "policy_assisted_action": assisted_action.copy(),
                "policy_returned_action": returned_action.copy(),
                "policy_action_scale": self._action_scale.copy(),
                "policy_qvel_mode": self._qvel_mode,
                "policy_qvel_input": qvel_input.copy(),
                "policy_previous_final_command_input": np.asarray(
                    policy_obs.get(
                        "previous_final_command",
                        np.zeros(len(ACTION_AXIS_NAMES), dtype=np.float32),
                    ),
                    dtype=np.float32,
                ).copy(),
                "policy_inference_latency_ms": latency_ms,
                "policy_frame_alignment_enabled": int(self._frame_alignment_enabled),
                "policy_frame_reused": 0,
                "policy_frame_reuse_count": 0,
                "policy_step": int(self._step),
                "policy_error": "",
                "policy_cycle_phase_enabled": int(self._cycle_phase_enabled),
                "policy_return_commit_enabled": int(self._return_commit_enabled),
                "policy_task_state_v2_enabled": int(self._task_state_v2_enabled),
                "policy_excursion_observed_enabled": int(
                    self._excursion_observed_enabled
                ),
                "policy_excursion_observed": float(self._excursion_observed),
                "policy_excursion_observed_epoch": int(self._excursion_observed_epoch),
                "policy_cycle_phase": float(self._cycle_phase),
                "policy_cycle_phase_epoch": int(self._cycle_phase_epoch),
                "policy_return_commit": float(self._return_commit),
                "policy_return_commit_epoch": int(self._return_commit_epoch),
                "policy_task_dig_complete": float(self._task_dig_complete),
                "policy_task_return_commit": float(self._task_return_commit),
                "policy_task_state_epoch": int(self._task_state_epoch),
                "policy_task_state_v2": np.asarray(
                    policy_obs.get(TASK_STATE_V2_KEY, np.zeros(5)),
                    dtype=np.float32,
                ).copy(),
                **assist_extras,
                **gate_extras,
            }
            if planner_goal is not None:
                extras.update(
                    {
                        "planner_cycle_index": int(planner_goal.cycle_index),
                        "planner_goal_epoch": int(planner_goal.goal_epoch),
                        "planner_current_side": str(planner_goal.current_side),
                        "planner_target_side": str(planner_goal.target_side),
                        "planner_target_side_code": int(planner_goal.target_side_code),
                        "planner_condition": np.asarray(
                            planner_goal.condition, dtype=np.float32
                        ),
                    }
                )
            factorized_diagnostics = getattr(
                self._policy, "factorized_diagnostics", None
            )
            if isinstance(factorized_diagnostics, dict):
                extras.update(factorized_diagnostics)
            temporal_aggregation_diagnostics = getattr(
                self._policy, "temporal_aggregation_diagnostics", None
            )
            if isinstance(temporal_aggregation_diagnostics, dict):
                extras.update(temporal_aggregation_diagnostics)
            if intent_probabilities is not None:
                extras["policy_intent_probabilities"] = np.asarray(
                    intent_probabilities, dtype=np.float32
                ).reshape(8)
            if self._runtime_gate_stack is not None:
                gohome_suppressed = (
                    raw_gohome_requested and self._output_mode == "shadow_zero"
                )
                extras["gohome_request_suppressed"] = int(gohome_suppressed)
                extras["gohome_request_suppression_reason"] = (
                    "policy_output_mode_shadow_zero" if gohome_suppressed else ""
                )
                extras["go_home_requested"] = bool(
                    raw_gohome_requested and self._output_mode == "control"
                )
            if self._bundle_dir is not None:
                extras["policy_bundle_dir"] = self._bundle_dir
            self._step += 1
            return (
                returned_action.astype(np.float32, copy=True),
                ActionInfo(
                    source_type="policy",
                    source_id=self._source_id,
                    latency_ms=latency_ms,
                    extras=extras,
                ),
            )
        except Exception as exc:
            if not self._fail_safe_zero:
                raise
            latency_ms = (time.perf_counter() - t0) * 1000.0
            zero = np.zeros(4, dtype=np.float32)
            extras = {
                "action_timestamp_ns": now_ns,
                "record_start_requested": record_start_requested,
                "policy_output_mode": self._output_mode,
                "policy_action": zero.copy(),
                "policy_scaled_action": zero.copy(),
                "policy_assisted_action": zero.copy(),
                "policy_returned_action": zero.copy(),
                "policy_action_scale": self._action_scale.copy(),
                "policy_qvel_mode": self._qvel_mode,
                "policy_qvel_input": zero.copy(),
                "policy_previous_final_command_input": zero.copy(),
                "policy_inference_latency_ms": latency_ms,
                "policy_frame_alignment_enabled": int(self._frame_alignment_enabled),
                "policy_frame_reused": 0,
                "policy_frame_reuse_count": 0,
                "policy_step": int(self._step),
                "policy_error": f"{type(exc).__name__}: {exc}",
                "policy_task_state_v2_enabled": int(self._task_state_v2_enabled),
                "policy_task_dig_complete": float(self._task_dig_complete),
                "policy_task_return_commit": float(self._task_return_commit),
                "policy_task_state_epoch": int(self._task_state_epoch),
                **_deadzone_assist_disabled_extras(self._deadzone_assist),
            }
            if self._report_intent:
                extras["policy_intent_probabilities"] = np.zeros(8, dtype=np.float32)
            if self._bundle_dir is not None:
                extras["policy_bundle_dir"] = self._bundle_dir
            self._step += 1
            return (
                zero,
                ActionInfo(
                    source_type="policy",
                    source_id=f"{self._source_id}:fail_safe_zero",
                    latency_ms=latency_ms,
                    extras=extras,
                ),
            )

    def _policy_obs(self, obs: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        qvel = self._policy_qvel(obs)
        policy_obs = _policy_obs_from_real_obs(
            obs,
            camera_name=self._camera_name,
            camera_names=self._camera_names,
            qvel_override=qvel,
        )
        return policy_obs, qvel

    def _apply_task_state_v2(self, policy_obs: dict[str, Any]) -> None:
        if not self._task_state_v2_enabled:
            return
        goal = (
            None
            if self._cycle_planner is None
            else getattr(self._cycle_planner, "committed_goal", None)
        )
        if goal is None:
            if TASK_STATE_V2_KEY not in policy_obs:
                raise RuntimeError(
                    "task-state-v2 ACT requires either a committed cycle goal or "
                    f"an explicit {TASK_STATE_V2_KEY} observation"
                )
            return
        policy_obs[TASK_STATE_V2_KEY] = self._task_state_v2_for_goal(goal)

    def _task_state_v2_for_goal(self, goal: Any | None) -> np.ndarray | None:
        if not self._task_state_v2_enabled or goal is None:
            return None
        return task_state_vector(
            current_side=str(goal.current_side),
            dig_target=str(goal.current_side),
            next_target=str(goal.target_side),
            dig_complete=self._task_dig_complete,
            return_commit=self._task_return_commit,
        )

    def _reset_policy_for_task_state_change(self) -> None:
        if not self._reset_policy_on_phase_change:
            return
        policy_reset = getattr(self._policy, "reset", None)
        if callable(policy_reset):
            policy_reset()

    def _policy_qvel(self, obs: dict[str, Any]) -> np.ndarray:
        raw_qvel = np.asarray(obs.get("qvel", np.zeros(4)), dtype=np.float32).reshape(
            -1
        )
        if raw_qvel.shape != (4,):
            raise ValueError(
                f"observation qvel must have shape (4,), got {raw_qvel.shape}"
            )
        if self._qvel_mode == "raw":
            return raw_qvel.astype(np.float32, copy=True)
        if self._qvel_mode == "zero":
            return np.zeros(4, dtype=np.float32)
        return self._qpos_diff_qvel(obs)

    def _qpos_diff_qvel(self, obs: dict[str, Any]) -> np.ndarray:
        qpos = np.asarray(obs.get("qpos", np.zeros(4)), dtype=np.float32).reshape(-1)
        if qpos.shape != (4,):
            raise ValueError(f"observation qpos must have shape (4,), got {qpos.shape}")
        obs_time_ns = _obs_time_ns(obs)
        if self._last_qpos is None or self._last_obs_time_ns is None:
            self._last_qpos = qpos.astype(np.float32, copy=True)
            self._last_obs_time_ns = obs_time_ns
            self._filtered_qvel.fill(0.0)
            return self._filtered_qvel.copy()
        dt = max(1e-3, float(obs_time_ns - self._last_obs_time_ns) * 1e-9)
        raw = (qpos - self._last_qpos) / dt
        raw = np.clip(raw, -self._qvel_diff_clip, self._qvel_diff_clip).astype(
            np.float32
        )
        if self._qvel_diff_tau_s <= 0.0:
            self._filtered_qvel = raw
        else:
            alpha = float(dt / (self._qvel_diff_tau_s + dt))
            self._filtered_qvel = (
                self._filtered_qvel + alpha * (raw - self._filtered_qvel)
            ).astype(np.float32)
        self._last_qpos = qpos.astype(np.float32, copy=True)
        self._last_obs_time_ns = obs_time_ns
        return self._filtered_qvel.copy()

    def _apply_deadzone_assist(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self._deadzone_assist
        if not cfg.enabled:
            return action.astype(
                np.float32, copy=True
            ), _deadzone_assist_disabled_extras(cfg)

        assisted = np.asarray(action, dtype=np.float32).copy()
        sign = np.sign(assisted).astype(np.int8)
        threshold = np.where(
            sign >= 0,
            cfg.deadzone_positive,
            cfg.deadzone_negative,
        ).astype(np.float32)
        magnitude = np.abs(assisted)
        intent = (
            cfg.axis_enabled
            & (sign != 0)
            & (magnitude >= cfg.trigger_fraction * threshold)
        )

        same_direction = intent & (sign == self._assist_last_sign)
        self._assist_consecutive_steps = np.where(
            same_direction,
            self._assist_consecutive_steps + 1,
            np.where(intent, 1, 0),
        ).astype(np.int32)
        self._assist_last_sign = np.where(intent, sign, 0).astype(np.int8)

        stable_intent = intent & (
            self._assist_consecutive_steps >= cfg.min_consecutive_steps
        )
        below_deadzone = magnitude < threshold
        assist_mask = stable_intent & below_deadzone
        target = np.minimum(threshold + cfg.margin, self._clip).astype(np.float32)
        assisted = np.where(assist_mask, sign.astype(np.float32) * target, assisted)
        assisted = np.clip(assisted, -self._clip, self._clip).astype(np.float32)
        axes = _assist_axes_text(assist_mask=assist_mask, sign=sign)
        extras = {
            "policy_deadzone_assist_enabled": 1,
            "policy_deadzone_assist_axis_enabled": cfg.axis_enabled.astype(np.int32),
            "policy_deadzone_assist_active": int(bool(np.any(assist_mask))),
            "policy_deadzone_assist_mask": assist_mask.astype(np.int32),
            "policy_deadzone_assist_axes": axes,
            "policy_deadzone_assist_trigger_fraction": cfg.trigger_fraction.copy(),
            "policy_deadzone_assist_min_consecutive_steps": int(
                cfg.min_consecutive_steps
            ),
            "policy_deadzone_assist_positive": cfg.deadzone_positive.copy(),
            "policy_deadzone_assist_negative": cfg.deadzone_negative.copy(),
        }
        return assisted, extras

    def close(self) -> None:
        close = getattr(self._policy, "close", None)
        if callable(close):
            close()

    def _consume_record_start_request(self) -> bool:
        if not self._record_start_pending:
            return False
        self._record_start_pending = False
        return True


def load_act_policy_from_bundle(
    *,
    bundle_dir: str | Path,
    ckpt_path: str | Path | None = None,
    resolved_config_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    device: str | None = None,
    temporal_agg: bool = True,
    inference_precision: str = "fp32",
    inference_compile: bool = False,
    inference_compile_mode: str = "reduce-overhead",
    inference_compile_dynamic: bool = False,
    device_uint8_preprocess: bool = False,
    temporal_aggregation_diagnostics: bool = False,
) -> Any:
    """Load the ACT policy bundle produced by excavator_testbed."""

    bundle = Path(bundle_dir)
    resolved_path = (
        Path(resolved_config_path)
        if resolved_config_path
        else bundle / "resolved_config.yaml"
    )
    if ckpt_path:
        checkpoint_source = Path(ckpt_path)
        ckpt = (
            checkpoint_source
            if checkpoint_source.is_absolute() or checkpoint_source.exists()
            else bundle / checkpoint_source
        )
    else:
        ckpt = bundle / "policy_best.ckpt"
    stats = Path(stats_path) if stats_path else bundle / "dataset_stats.pkl"
    missing = [path for path in (resolved_path, ckpt, stats) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing policy bundle file(s): " + ", ".join(str(path) for path in missing)
        )

    with resolved_path.open("r", encoding="utf-8") as f:
        resolved = yaml.safe_load(f) or {}
    policy_config = _act_policy_config_from_resolved(resolved)
    for section_name in (
        "intent_loss",
        "deadzone_loss",
        "window_deadzone_loss",
        "temporal_release_loss",
        "condition_adherence_loss",
        "task_state_v2_adherence_loss",
    ):
        section = policy_config.get(section_name)
        if isinstance(section, dict):
            _relocate_bundle_file(
                section,
                key="threshold_json",
                bundle=bundle,
            )

    from testbed.policies.act.adapter import ACTAdapter

    return ACTAdapter.from_checkpoint(
        ckpt_path=ckpt,
        policy_config=policy_config,
        norm_stats_path=stats,
        temporal_agg=bool(temporal_agg),
        inference_precision=inference_precision,
        inference_compile=bool(inference_compile),
        inference_compile_mode=str(inference_compile_mode),
        inference_compile_dynamic=bool(inference_compile_dynamic),
        device_uint8_preprocess=bool(device_uint8_preprocess),
        temporal_aggregation_diagnostics=bool(temporal_aggregation_diagnostics),
        device=str(device or resolved.get("policy", {}).get("device", "cuda")),
    )


def _relocate_bundle_file(config: dict[str, Any], *, key: str, bundle: Path) -> None:
    raw = config.get(key)
    if raw is None or not str(raw).strip():
        return
    source = Path(str(raw))
    try:
        if source.is_file():
            return
    except OSError:
        pass
    candidates = (bundle / source.name, bundle / "contracts" / source.name)
    for candidate in candidates:
        if candidate.is_file():
            config[key] = str(candidate)
            return


def _resolved_low_dim_state_dim(low_dim_keys: list[str]) -> int:
    dimensions = {
        "qpos": 4,
        "qvel": 4,
        "real_transition_condition_v1": 2,
        "real_transition_excursion_observed_v1": 1,
        "real_transition_cycle_phase_v1": 1,
        "real_transition_return_commit_v1": 1,
        "real_transition_action_primitive_v1": 4,
        "real_transition_work_context_v1": 6,
        TASK_STATE_V2_KEY: 5,
    }
    unknown = [key for key in low_dim_keys if key not in dimensions]
    if unknown:
        raise ValueError(f"unsupported ACT low_dim_keys: {unknown}")
    return int(sum(dimensions[key] for key in low_dim_keys))


def _act_policy_config_from_resolved(resolved: dict[str, Any]) -> dict[str, Any]:
    task_cfg = dict(resolved.get("task", {}) or {})
    policy_cfg = dict(resolved.get("policy", {}) or {})
    train_cfg = dict(resolved.get("train", {}) or {})
    act_params = dict(policy_cfg.get("act_params", {}) or {})
    camera_names = list(task_cfg.get("camera_names", ["fpv"]))
    low_dim_keys = list(policy_cfg.get("low_dim_keys", ["qpos", "qvel"]))
    state_dim = int(
        act_params.get("state_dim", _resolved_low_dim_state_dim(low_dim_keys))
    )
    episode_len = task_cfg.get("episode_len")
    max_episode_len = 400 if episode_len is None else int(episode_len)
    return {
        "lr": float(train_cfg.get("lr", 1e-5)),
        "num_queries": int(act_params.get("chunk_size", 100)),
        "kl_weight": float(act_params.get("kl_weight", 10)),
        "hidden_dim": int(act_params.get("hidden_dim", 512)),
        "dim_feedforward": int(act_params.get("dim_feedforward", 3200)),
        "vision_feature_scale": float(act_params.get("vision_feature_scale", 1.0)),
        "proprio_feature_scale": float(act_params.get("proprio_feature_scale", 1.0)),
        "camera_role_encoding": copy.deepcopy(
            act_params.get("camera_role_encoding", {}) or {}
        ),
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        # A full ACT checkpoint contains the ResNet parameters. Runtime bundle
        # loading must work on an offline field computer without consulting the
        # torchvision download cache.
        "backbone_pretrained": False,
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
        "equipment_model": task_cfg.get("equipment_model", "real_excavator"),
        "max_episode_len": max_episode_len,
        "low_dim_keys": low_dim_keys,
        "state_dim": state_dim,
        "train_with_zero_latent": bool(act_params.get("train_with_zero_latent", False)),
        "intent_loss": copy.deepcopy(
            train_cfg.get("intent_loss", policy_cfg.get("intent_loss", {})) or {}
        ),
        "factorized_intent_effort": copy.deepcopy(
            train_cfg.get(
                "factorized_intent_effort",
                policy_cfg.get("factorized_intent_effort", {}),
            )
            or {}
        ),
        "goal_effect": copy.deepcopy(
            train_cfg.get("goal_effect", policy_cfg.get("goal_effect", {})) or {}
        ),
        # These objectives are disabled during inference, but their resolved
        # architecture flags must be reconstructed so a conditioned ACT
        # checkpoint (including condition_action_head) can be loaded by the
        # offline evaluator and runtime adapter.
        "condition_adherence_loss": copy.deepcopy(
            train_cfg.get(
                "condition_adherence_loss",
                policy_cfg.get("condition_adherence_loss", {}),
            )
            or {}
        ),
        "condition_action_loss": copy.deepcopy(
            train_cfg.get(
                "condition_action_loss",
                policy_cfg.get("condition_action_loss", {}),
            )
            or {}
        ),
        "deadzone_loss": copy.deepcopy(
            train_cfg.get("deadzone_loss", policy_cfg.get("deadzone_loss", {})) or {}
        ),
        "window_deadzone_loss": copy.deepcopy(
            train_cfg.get(
                "window_deadzone_loss",
                policy_cfg.get("window_deadzone_loss", {}),
            )
            or {}
        ),
        "temporal_release_loss": copy.deepcopy(
            train_cfg.get(
                "temporal_release_loss",
                policy_cfg.get("temporal_release_loss", {}),
            )
            or {}
        ),
        # Action-state/effort is an auxiliary training head, but the model
        # architecture must still be reconstructed when loading its bundle so
        # the checkpoint's extra head is accepted.  It never changes the
        # runtime continuous action source.
        "action_state_effort": copy.deepcopy(
            train_cfg.get(
                "action_state_effort",
                policy_cfg.get("action_state_effort", {}),
            )
            or {}
        ),
        "effective_action": copy.deepcopy(
            train_cfg.get(
                "effective_action",
                policy_cfg.get("effective_action", {}),
            )
            or {}
        ),
        "temporal_input": copy.deepcopy(
            train_cfg.get(
                "temporal_input",
                policy_cfg.get("temporal_input", {}),
            )
            or {}
        ),
        "primitive_action_heads": copy.deepcopy(
            train_cfg.get(
                "primitive_action_heads",
                policy_cfg.get("primitive_action_heads", {}),
            )
            or {}
        ),
        "work_return_context": copy.deepcopy(
            train_cfg.get(
                "work_return_context",
                policy_cfg.get("work_return_context", {}),
            )
            or {}
        ),
        "state_visual_residual": copy.deepcopy(
            train_cfg.get(
                "state_visual_residual",
                policy_cfg.get("state_visual_residual", {}),
            )
            or {}
        ),
        "task_state_v2_adherence_loss": copy.deepcopy(
            train_cfg.get(
                "task_state_v2_adherence_loss",
                policy_cfg.get("task_state_v2_adherence_loss", {}),
            )
            or {}
        ),
    }


def _policy_obs_from_real_obs(
    obs: dict[str, Any],
    *,
    camera_name: str | None = None,
    camera_names: list[str] | tuple[str, ...] | None = None,
    qvel_override: np.ndarray | None = None,
) -> dict[str, Any]:
    if "qpos" not in obs:
        raise KeyError("observation missing qpos")
    if "qvel" not in obs:
        raise KeyError("observation missing qvel")
    names = _camera_names_list(camera_names, default=(camera_name or "fpv"))
    if not names:
        raise ValueError("camera_names must not be empty")
    policy_obs = {
        "qpos": np.asarray(obs["qpos"], dtype=np.float32),
        "qvel": (
            np.asarray(qvel_override, dtype=np.float32)
            if qvel_override is not None
            else np.asarray(obs["qvel"], dtype=np.float32)
        ),
    }
    if "real_transition_condition_v1" in obs:
        condition = np.asarray(
            obs["real_transition_condition_v1"], dtype=np.float32
        ).reshape(-1)
        if condition.shape != (2,):
            raise ValueError(
                "observation real_transition_condition_v1 must have shape (2), "
                f"got {condition.shape}"
            )
        if not np.isfinite(condition).all():
            raise ValueError("observation real_transition_condition_v1 must be finite")
        policy_obs["real_transition_condition_v1"] = condition.copy()
    if ACTION_PRIMITIVE_KEY in obs:
        primitive = np.asarray(obs[ACTION_PRIMITIVE_KEY], dtype=np.float32).reshape(-1)
        if (
            primitive.shape != (len(PRIMITIVE_NAMES),)
            or not np.isfinite(primitive).all()
            or not np.all(np.isin(primitive, [0.0, 1.0]))
            or not np.isclose(float(primitive.sum()), 1.0)
        ):
            raise ValueError(
                f"observation {ACTION_PRIMITIVE_KEY} must be a finite one-hot "
                f"vector of length {len(PRIMITIVE_NAMES)}"
            )
        policy_obs[ACTION_PRIMITIVE_KEY] = primitive.copy()
    if WORK_CONTEXT_KEY in obs:
        context = np.asarray(obs[WORK_CONTEXT_KEY], dtype=np.float32).reshape(-1)
        valid = (
            context.shape == (6,)
            and np.isfinite(context).all()
            and context[0] in {-1.0, 1.0}
            and context[1] in {-1.0, 1.0}
            and np.all(np.isin(context[2:], [0.0, 1.0]))
            and np.isclose(float(context[2:].sum()), 1.0)
        )
        if not valid:
            raise ValueError(
                f"observation {WORK_CONTEXT_KEY} must contain current A/B, "
                "dig-target A/B, and one WORK_A/WORK_B/RETURN_A/RETURN_B one-hot"
            )
        policy_obs[WORK_CONTEXT_KEY] = context.copy()
    if TASK_STATE_V2_KEY in obs:
        task_state = np.asarray(obs[TASK_STATE_V2_KEY], dtype=np.float32).reshape(-1)
        valid = (
            task_state.shape == (5,)
            and np.isfinite(task_state).all()
            and task_state[0] in {-1.0, 1.0}
            and task_state[1] == task_state[0]
            and task_state[2] in {0.0, 1.0}
            and task_state[3] in {0.0, 1.0}
            and (
                (task_state[3] == 0.0 and task_state[4] == 0.0)
                or (task_state[3] == 1.0 and task_state[4] in {-1.0, 1.0})
            )
        )
        if not valid:
            raise ValueError(
                f"observation {TASK_STATE_V2_KEY} must contain current side, "
                "matching dig target, independent complete/commit bits, and a "
                "next target gated by return_commit"
            )
        policy_obs[TASK_STATE_V2_KEY] = task_state.copy()
    if EXCURSION_OBSERVED_KEY in obs:
        excursion = np.asarray(obs[EXCURSION_OBSERVED_KEY], dtype=np.float32).reshape(
            -1
        )
        if (
            excursion.shape != (1,)
            or not np.isfinite(excursion).all()
            or excursion[0] not in {0.0, 1.0}
        ):
            raise ValueError(
                f"observation {EXCURSION_OBSERVED_KEY} must contain one finite "
                "0/1 value"
            )
        policy_obs[EXCURSION_OBSERVED_KEY] = excursion.copy()
    if CYCLE_PHASE_KEY in obs:
        phase = np.asarray(obs[CYCLE_PHASE_KEY], dtype=np.float32).reshape(-1)
        if (
            phase.shape != (1,)
            or not np.isfinite(phase).all()
            or phase[0]
            not in {
                0.0,
                1.0,
            }
        ):
            raise ValueError(
                f"observation {CYCLE_PHASE_KEY} must contain one finite 0/1 value"
            )
        policy_obs[CYCLE_PHASE_KEY] = phase.copy()
    if RETURN_COMMIT_KEY in obs:
        return_commit = np.asarray(obs[RETURN_COMMIT_KEY], dtype=np.float32).reshape(-1)
        if (
            return_commit.shape != (1,)
            or not np.isfinite(return_commit).all()
            or return_commit[0] not in {0.0, 1.0}
        ):
            raise ValueError(
                f"observation {RETURN_COMMIT_KEY} must contain one finite 0/1 value"
            )
        policy_obs[RETURN_COMMIT_KEY] = return_commit.copy()
    for name in names:
        policy_obs[f"image_{name}"] = _resolve_camera_image(obs, camera_name=name)
    # Keep causal image timing metadata for the opt-in temporal ACT adapter.
    # The legacy single-frame path simply ignores these extra keys.
    image_timestamps = obs.get("image_timestamp_ns")
    if isinstance(image_timestamps, dict):
        policy_obs["image_timestamp_ns"] = {
            str(name): int(value)
            for name, value in image_timestamps.items()
            if isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
        }
    elif isinstance(image_timestamps, (int, np.integer)) and not isinstance(
        image_timestamps, (bool, np.bool_)
    ):
        policy_obs["image_timestamp_ns"] = int(image_timestamps)
    for timestamp_key in ("sync_timestamp_ns", "timestamp_ns"):
        timestamp = obs.get(timestamp_key)
        if isinstance(timestamp, (int, np.integer)) and not isinstance(
            timestamp, (bool, np.bool_)
        ):
            policy_obs[timestamp_key] = int(timestamp)
    if "previous_final_command" in obs:
        previous_final_command = np.asarray(
            obs["previous_final_command"], dtype=np.float32
        ).reshape(-1)
        if previous_final_command.shape != (len(ACTION_AXIS_NAMES),):
            raise ValueError(
                "observation previous_final_command must have shape "
                f"({len(ACTION_AXIS_NAMES)},), got {previous_final_command.shape}"
            )
        if not np.isfinite(previous_final_command).all():
            raise ValueError("observation previous_final_command must be finite")
        policy_obs["previous_final_command"] = previous_final_command.copy()
    return policy_obs


def _camera_names_list(value: Any, *, default: str = "fpv") -> list[str]:
    if value is None:
        return [str(default)]
    if isinstance(value, str):
        return [value]
    return [str(name) for name in value]


def _resolve_camera_image(obs: dict[str, Any], *, camera_name: str) -> np.ndarray:
    images = obs.get("images") or {}
    if camera_name in images:
        return np.asarray(images[camera_name], dtype=np.uint8)
    direct_key = f"image_{camera_name}"
    if direct_key in obs:
        return np.asarray(obs[direct_key])
    encoded_images = obs.get("encoded_images") or {}
    if camera_name in encoded_images:
        return _decode_encoded_image(encoded_images[camera_name])
    raise KeyError(f"observation missing camera {camera_name!r}")


def _decode_encoded_image(payload: Any) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required to decode encoded FPV images"
        ) from exc
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("bytes", b""))
    if isinstance(payload, (bytes, bytearray, memoryview)):
        encoded = np.frombuffer(bytes(payload), dtype=np.uint8)
    else:
        encoded = np.asarray(payload, dtype=np.uint8).reshape(-1)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("failed to decode encoded camera image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _broadcast_action_scale(value: Any) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full(4, float(value), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"action_scale must be scalar or shape (4,), got {arr.shape}")
    return arr.astype(np.float32, copy=True)


def _broadcast_qvel_clip(value: Any) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full(4, max(0.0, float(value)), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(
            f"qvel_diff_clip_rad_s must be scalar or shape (4,), got {arr.shape}"
        )
    return np.maximum(arr, 0.0).astype(np.float32, copy=True)


def _deadzone_assist_config(config: dict[str, Any] | None) -> DeadzoneAssistConfig:
    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", False))
    axis_enabled = _broadcast_bool_vector4(
        cfg.get("axis_enabled", True),
        name="deadzone_assist.axis_enabled",
    )
    positive = _broadcast_nonnegative_vector4(
        cfg.get("deadzone_positive", cfg.get("deadzone", 0.0)),
        name="deadzone_assist.deadzone_positive",
    )
    negative = _broadcast_nonnegative_vector4(
        cfg.get("deadzone_negative", cfg.get("deadzone", positive)),
        name="deadzone_assist.deadzone_negative",
    )
    if enabled and (np.any(positive <= 0.0) or np.any(negative <= 0.0)):
        raise ValueError(
            "deadzone_assist requires positive deadzone_positive and "
            "deadzone_negative values when enabled"
        )
    trigger_fraction = _broadcast_fraction_vector4(
        cfg.get("trigger_fraction", 0.5),
        name="deadzone_assist.trigger_fraction",
    )
    margin = _broadcast_nonnegative_vector4(
        cfg.get("margin", 0.02),
        name="deadzone_assist.margin",
    )
    min_consecutive_steps = int(cfg.get("min_consecutive_steps", 2))
    if min_consecutive_steps < 1:
        raise ValueError("deadzone_assist.min_consecutive_steps must be >= 1")
    return DeadzoneAssistConfig(
        enabled=enabled,
        axis_enabled=axis_enabled,
        trigger_fraction=trigger_fraction,
        margin=margin,
        deadzone_positive=positive,
        deadzone_negative=negative,
        min_consecutive_steps=min_consecutive_steps,
    )


def _broadcast_nonnegative_vector4(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full(4, max(0.0, float(value)), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"{name} must be scalar or shape (4,), got {arr.shape}")
    return np.maximum(arr, 0.0).astype(np.float32, copy=True)


def _broadcast_fraction_vector4(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, (int, float)):
        arr = np.full(4, float(value), dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape != (4,):
            raise ValueError(f"{name} must be scalar or shape (4,), got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} must contain finite values in (0, 1]")
    return arr.astype(np.float32, copy=True)


def _broadcast_bool_vector4(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, (bool, int)):
        return np.full(4, bool(value), dtype=bool)
    arr = np.asarray(value).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"{name} must be scalar or shape (4,), got {arr.shape}")
    return arr.astype(bool, copy=True)


def _deadzone_assist_disabled_extras(
    cfg: DeadzoneAssistConfig,
) -> dict[str, Any]:
    return {
        "policy_deadzone_assist_enabled": int(bool(cfg.enabled)),
        "policy_deadzone_assist_axis_enabled": cfg.axis_enabled.astype(np.int32),
        "policy_deadzone_assist_active": 0,
        "policy_deadzone_assist_mask": np.zeros(4, dtype=np.int32),
        "policy_deadzone_assist_axes": "",
        "policy_deadzone_assist_trigger_fraction": cfg.trigger_fraction.copy(),
        "policy_deadzone_assist_min_consecutive_steps": int(cfg.min_consecutive_steps),
        "policy_deadzone_assist_positive": cfg.deadzone_positive.copy(),
        "policy_deadzone_assist_negative": cfg.deadzone_negative.copy(),
    }


def _assist_axes_text(*, assist_mask: np.ndarray, sign: np.ndarray) -> str:
    axes = []
    for idx, axis_name in enumerate(ACTION_AXIS_NAMES):
        if not bool(assist_mask[idx]):
            continue
        suffix = "+" if int(sign[idx]) >= 0 else "-"
        axes.append(f"{axis_name}{suffix}")
    return ",".join(axes)


def _obs_time_ns(obs: dict[str, Any]) -> int:
    for key in ("joint_timestamp_ns", "sync_timestamp_ns", "timestamp_ns"):
        value = obs.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return time.time_ns()


def policy_bundle_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    """Small helper for CLI/debug scripts to inspect a local bundle."""

    bundle = Path(bundle_dir)
    manifest = {"bundle_dir": str(bundle)}
    for name in (
        "policy_best.ckpt",
        "dataset_stats.pkl",
        "resolved_config.yaml",
        "run_metadata.json",
    ):
        path = bundle / name
        manifest[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
    resolved_path = bundle / "resolved_config.yaml"
    if resolved_path.exists():
        with resolved_path.open("r", encoding="utf-8") as f:
            resolved = yaml.safe_load(f) or {}
        manifest["resolved_task"] = resolved.get("task", {})
        manifest["resolved_policy"] = resolved.get("policy", {})
    metadata_path = bundle / "run_metadata.json"
    if metadata_path.exists():
        manifest["run_metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return manifest


def _build_cycle_planner(raw_config: Any) -> Any | None:
    """Build the opt-in goal-only planner used by a policy action source."""

    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("teleop.policy.cycle_planner must be a mapping")
    if not bool(raw_config.get("enabled", False)):
        return None
    from testbed.tasks.act_cycle_planner import (
        ABCyclePlanner,
        ScriptCyclePlanner,
        SideMatchedScriptCyclePlanner,
    )

    loop = None if "loop" not in raw_config else bool(raw_config["loop"])
    max_cycles = (
        None if raw_config.get("max_cycles") is None else int(raw_config["max_cycles"])
    )
    script_path = raw_config.get("script_path")
    script_paths_by_side = raw_config.get("script_paths_by_initial_side")
    if script_path and script_paths_by_side:
        raise ValueError(
            "cycle_planner cannot set both script_path and script_paths_by_initial_side"
        )
    if script_paths_by_side is not None:
        return SideMatchedScriptCyclePlanner.from_script_paths(
            script_paths_by_side,
            loop=loop,
            max_cycles=max_cycles,
        )
    if script_path:
        return ScriptCyclePlanner.from_script(
            str(script_path), loop=loop, max_cycles=max_cycles
        )
    script = raw_config.get("script")
    if script is not None:
        return ScriptCyclePlanner.from_mapping(script, loop=loop, max_cycles=max_cycles)
    return ABCyclePlanner(
        pattern=str(raw_config.get("pattern", "ABBABABA")),
        loop=True if loop is None else loop,
        max_cycles=max_cycles,
    )
