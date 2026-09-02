"""Remote-armed policy action source for live policy tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from testbed.actions.base import ActionInfo, ActionSource
from testbed.actions.policy import PolicyActionSource
from testbed.actions.remote import RemoteActionSource


class RemoteArmedPolicyActionSource(ActionSource):
    """
    Start with remote teleop, then switch to policy on policy-start.

    Continuous 4D actions come from the host remote before activation and from
    the policy after activation. Remote discrete events are still consumed after
    activation so status toggles, reset/discard/quit, and go-home remain usable.
    """

    def __init__(
        self,
        *,
        remote: ActionSource,
        policy: ActionSource,
        source_id: str = "policy_remote",
        start_in_policy: bool = False,
        infer_on_new_frame: bool = False,
        scripted_cycle_runtime: Any | None = None,
        scripted_cycle_auto_start_after_arm: bool = False,
    ) -> None:
        self._remote = remote
        self._policy = policy
        self._source_id = str(source_id)
        self._start_in_policy = bool(start_in_policy)
        self._infer_on_new_frame = bool(infer_on_new_frame)
        self._scripted_cycle_runtime = scripted_cycle_runtime
        self._scripted_cycle_auto_start_after_arm = bool(
            scripted_cycle_auto_start_after_arm
        )
        if self._scripted_cycle_auto_start_after_arm and scripted_cycle_runtime is None:
            raise ValueError(
                "scripted-cycle auto start requires scripted_cycle_runtime"
            )
        if self._scripted_cycle_runtime is not None and self._start_in_policy:
            raise ValueError("scripted-cycle policy_remote must start in manual mode")
        self._policy_active = bool(start_in_policy)
        self._step = 0
        self._activation_step: int | None = 0 if start_in_policy else None
        self._toggle_count = 0
        self._last_policy_frame_token: tuple[tuple[str, int], ...] | None = None
        self._cached_policy_action: np.ndarray | None = None
        self._cached_policy_info: ActionInfo | None = None
        self._policy_frame_reuse_count = 0
        self._script_stop_latched = False
        self._script_stop_reason = ""
        self._script_auto_armed = False
        self._script_auto_wait_reason = ""
        self._scripted_cycle_activation_rejected_reason = ""
        self._scripted_cycle_last_status = (
            _disabled_scripted_cycle_status()
            if self._scripted_cycle_runtime is None
            else self._scripted_cycle_runtime.status()
        )

    @classmethod
    def from_config(
        cls, teleop_cfg: Mapping[str, Any] | None
    ) -> RemoteArmedPolicyActionSource:
        cfg = dict(teleop_cfg or {})
        mode_cfg = dict(cfg.get("policy_remote", {}) or {})
        policy_cfg = dict(cfg.get("policy", {}) or {})
        frame_alignment_cfg = dict(policy_cfg.get("frame_alignment", {}) or {})
        policy = PolicyActionSource.from_config(policy_cfg)
        from testbed.tasks.scripted_cycle_runtime import ScriptedCycleRuntime

        scripted_cycle_runtime = ScriptedCycleRuntime.from_config(
            mode_cfg.get("scripted_cycle"),
            policy_source=policy,
            bundle_dir=policy_cfg.get("bundle_dir"),
        )
        return cls(
            remote=RemoteActionSource.from_config(cfg.get("remote", {})),
            policy=policy,
            source_id=str(mode_cfg.get("source_id", "policy_remote")),
            start_in_policy=bool(mode_cfg.get("start_in_policy", False)),
            infer_on_new_frame=bool(frame_alignment_cfg.get("enabled", False)),
            scripted_cycle_runtime=scripted_cycle_runtime,
            scripted_cycle_auto_start_after_arm=bool(
                dict(mode_cfg.get("scripted_cycle", {}) or {}).get(
                    "auto_start_after_arm", False
                )
            ),
        )

    def reset(self) -> None:
        self._remote.reset()
        self._policy.reset()
        self._policy_active = self._start_in_policy
        self._step = 0
        self._activation_step = 0 if self._start_in_policy else None
        self._toggle_count = 0
        self._script_stop_latched = False
        self._script_stop_reason = ""
        self._script_auto_armed = False
        self._script_auto_wait_reason = ""
        self._scripted_cycle_activation_rejected_reason = ""
        if self._scripted_cycle_runtime is not None:
            self._scripted_cycle_runtime.reset()
            self._scripted_cycle_last_status = self._scripted_cycle_runtime.status()
        self._clear_policy_frame_cache()

    def prepare(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Prepare the policy while leaving remote/manual state untouched."""

        prepare = getattr(self._policy, "prepare", None)
        if not callable(prepare):
            return {"prepared": 0, "warmup_steps": 0, "elapsed_s": 0.0}
        return dict(prepare(obs) or {})

    def next_action(self, obs: dict[str, Any]) -> tuple[np.ndarray, ActionInfo]:
        remote_action, remote_info = self._remote.next_action(obs)
        remote_extras = dict(getattr(remote_info, "extras", {}) or {})
        start_requested = bool(remote_extras.get("policy_start_requested", False))
        mark_requested = bool(remote_extras.get("mark_requested", False))
        record_start_requested = bool(
            remote_extras.get("record_start_requested", False)
        )
        activated_now = False
        deactivated_now = False
        activation_rejected_reason = ""
        if self._scripted_cycle_runtime is not None:
            try:
                self._scripted_cycle_last_status = self._scripted_cycle_runtime.observe(
                    obs
                )
            except Exception as exc:
                activation_rejected_reason = (
                    f"scripted_cycle_observation_error:{type(exc).__name__}:{exc}"
                )
                if self._policy_active:
                    self._scripted_cycle_last_status = (
                        self._scripted_cycle_runtime.deactivate(
                            activation_rejected_reason
                        )
                    )
                    self._latch_script_stop(activation_rejected_reason)

        if start_requested and self._script_stop_latched:
            self._script_stop_latched = False
            self._script_stop_reason = ""
            self._script_auto_armed = False
            self._script_auto_wait_reason = ""
            self._scripted_cycle_activation_rejected_reason = ""
            if self._scripted_cycle_runtime is not None:
                self._scripted_cycle_runtime.prepare_new_run()
                self._scripted_cycle_last_status = self._scripted_cycle_runtime.status()
            deactivated_now = True
        elif start_requested and not self._policy_active:
            if self._scripted_cycle_runtime is None:
                activated_now = self._set_policy_active_state(True)
            elif self._scripted_cycle_auto_start_after_arm:
                self._script_auto_armed = not self._script_auto_armed
                self._script_auto_wait_reason = (
                    "" if not self._script_auto_armed else "waiting_initial_ready"
                )
                self._scripted_cycle_activation_rejected_reason = ""
            else:
                activation_rejected_reason = (
                    activation_rejected_reason
                    or self._scripted_cycle_runtime.activation_blocker()
                )
                if not activation_rejected_reason:
                    activated_now = self._set_policy_active_state(True)
                    try:
                        self._scripted_cycle_last_status = (
                            self._scripted_cycle_runtime.activate()
                        )
                        self._scripted_cycle_activation_rejected_reason = ""
                        if bool(
                            self._scripted_cycle_last_status.get("stop_policy", False)
                        ):
                            self._latch_script_stop(
                                str(
                                    self._scripted_cycle_last_status.get("fault")
                                    or "scripted_cycle_activation_failed"
                                )
                            )
                    except Exception as exc:
                        activation_rejected_reason = f"scripted_cycle_activation_error:{type(exc).__name__}:{exc}"
                        self._scripted_cycle_last_status = (
                            self._scripted_cycle_runtime.deactivate(
                                activation_rejected_reason
                            )
                        )
                        self._set_policy_active_state(False)
        elif start_requested and self._policy_active:
            self._script_auto_armed = False
            self._script_auto_wait_reason = ""
            if self._scripted_cycle_runtime is not None:
                self._scripted_cycle_last_status = (
                    self._scripted_cycle_runtime.deactivate("operator_toggle")
                )
            deactivated_now = self._set_policy_active_state(False)
        if activation_rejected_reason:
            self._scripted_cycle_activation_rejected_reason = activation_rejected_reason

        if (
            self._script_auto_armed
            and not self._policy_active
            and not self._script_stop_latched
            and self._scripted_cycle_runtime is not None
        ):
            wait_reason = (
                activation_rejected_reason
                or self._scripted_cycle_runtime.activation_blocker()
            )
            self._script_auto_wait_reason = wait_reason
            if not wait_reason:
                activated_now = self._set_policy_active_state(True)
                try:
                    self._scripted_cycle_last_status = (
                        self._scripted_cycle_runtime.activate()
                    )
                    self._script_auto_armed = False
                    self._script_auto_wait_reason = ""
                    self._scripted_cycle_activation_rejected_reason = ""
                    if bool(self._scripted_cycle_last_status.get("stop_policy", False)):
                        self._latch_script_stop(
                            str(
                                self._scripted_cycle_last_status.get("fault")
                                or "scripted_cycle_activation_failed"
                            )
                        )
                except Exception as exc:
                    activation_rejected_reason = (
                        f"scripted_cycle_activation_error:{type(exc).__name__}:{exc}"
                    )
                    self._script_auto_armed = False
                    self._script_auto_wait_reason = ""
                    self._scripted_cycle_activation_rejected_reason = (
                        activation_rejected_reason
                    )
                    self._scripted_cycle_last_status = (
                        self._scripted_cycle_runtime.deactivate(
                            activation_rejected_reason
                        )
                    )
                    self._set_policy_active_state(False)

        task_state_changed = False
        task_state_advance_ignored = False
        task_state_rejected_reason = ""
        if mark_requested and self._scripted_cycle_runtime is not None:
            if not self._policy_active:
                task_state_rejected_reason = "task_state_advance_requires_active_policy"
            else:
                try:
                    self._scripted_cycle_last_status = (
                        self._scripted_cycle_runtime.advance_task_state()
                    )
                    task_state_changed = bool(
                        self._scripted_cycle_last_status.get(
                            "task_state_changed", False
                        )
                    )
                    task_state_advance_ignored = bool(
                        self._scripted_cycle_last_status.get(
                            "task_state_advance_ignored", False
                        )
                    )
                    if task_state_changed:
                        self._clear_policy_frame_cache()
                except Exception as exc:
                    task_state_rejected_reason = (
                        f"task_state_advance_error:{type(exc).__name__}:{exc}"
                    )
            if not self._policy_active:
                self._scripted_cycle_last_status.update(
                    {
                        "task_state_advance_requested": True,
                        "task_state_changed": False,
                        "task_state_advance_ignored": False,
                        "task_state_advance_rejected_reason": str(
                            task_state_rejected_reason
                        ),
                    }
                )

        if self._policy_active and self._scripted_cycle_runtime is not None:
            self._scripted_cycle_last_status = self._scripted_cycle_runtime.evaluate()
            self._scripted_cycle_last_status.update(
                {
                    "task_state_advance_requested": bool(mark_requested),
                    "task_state_changed": bool(
                        task_state_changed
                        or self._scripted_cycle_last_status.get(
                            "task_state_changed", False
                        )
                    ),
                    "task_state_advance_ignored": bool(task_state_advance_ignored),
                    "task_state_advance_rejected_reason": str(
                        task_state_rejected_reason
                    ),
                }
            )
            if bool(
                self._scripted_cycle_last_status.get("goal_changed", False)
                or self._scripted_cycle_last_status.get("phase_changed", False)
                or self._scripted_cycle_last_status.get("task_state_changed", False)
            ):
                self._clear_policy_frame_cache()
            if bool(self._scripted_cycle_last_status.get("stop_policy", False)):
                stop_reason = str(
                    self._scripted_cycle_last_status.get("fault")
                    or self._scripted_cycle_last_status.get("stop_reason")
                    or "script_complete"
                )
                self._latch_script_stop(stop_reason)

        if self._script_stop_latched:
            zero = np.zeros(4, dtype=np.float32)
            extras = dict(remote_extras)
            extras.update(
                {
                    "record_start_requested": record_start_requested,
                    "policy_start_requested": start_requested,
                    "policy_remote_mode": "script_stop",
                    "policy_remote_activated": 0,
                    "policy_remote_deactivated": int(deactivated_now),
                    "policy_remote_activation_step": -1,
                    "policy_remote_toggle_count": int(self._toggle_count),
                    "model_control": 0,
                    "scripted_cycle_stop_latched": 1,
                    "scripted_cycle_auto_start_after_arm": int(
                        self._scripted_cycle_auto_start_after_arm
                    ),
                    "scripted_cycle_auto_armed": int(self._script_auto_armed),
                    "scripted_cycle_auto_wait_reason": str(
                        self._script_auto_wait_reason
                    ),
                    "policy_action": zero.copy(),
                    "policy_scaled_action": zero.copy(),
                    "policy_assisted_action": zero.copy(),
                    "policy_returned_action": zero.copy(),
                    "policy_output_mode": "script_stop_zero",
                    "policy_error": "",
                    "scripted_cycle_activation_rejected_reason": (
                        activation_rejected_reason
                        or self._scripted_cycle_activation_rejected_reason
                    ),
                    **_scripted_cycle_extras(self._scripted_cycle_last_status),
                }
            )
            self._step += 1
            return zero, ActionInfo(
                source_type="policy",
                source_id=f"{self._source_id}:script_stop_zero",
                latency_ms=0.0,
                extras=extras,
            )

        if self._policy_active:
            frame_token = _observation_image_token(obs)
            reuse_policy_frame = bool(
                self._infer_on_new_frame
                and frame_token is not None
                and frame_token == self._last_policy_frame_token
                and self._cached_policy_action is not None
                and self._cached_policy_info is not None
            )
            if reuse_policy_frame:
                policy_action = self._cached_policy_action.copy()
                policy_info = self._cached_policy_info
                self._policy_frame_reuse_count += 1
            else:
                policy_action, policy_info = self._policy.next_action(obs)
                if self._infer_on_new_frame and frame_token is not None:
                    self._last_policy_frame_token = frame_token
                    self._cached_policy_action = np.asarray(
                        policy_action, dtype=np.float32
                    ).copy()
                    self._cached_policy_info = policy_info
                self._policy_frame_reuse_count = 0
            policy_extras = dict(getattr(policy_info, "extras", {}) or {})
            if self._scripted_cycle_runtime is not None:
                task_state_changed_before_action = bool(
                    self._scripted_cycle_last_status.get("task_state_changed", False)
                )
                raw_policy_action = np.asarray(
                    policy_extras.get("policy_action", policy_action),
                    dtype=np.float32,
                )
                self._scripted_cycle_last_status = (
                    self._scripted_cycle_runtime.observe_policy_action(
                        raw_policy_action
                    )
                )
                self._scripted_cycle_last_status["task_state_changed"] = bool(
                    task_state_changed_before_action
                    or self._scripted_cycle_last_status.get("task_state_changed", False)
                )
                policy_action, swing_landing_diagnostics = (
                    self._scripted_cycle_runtime.shape_policy_action(
                        policy_action,
                        obs,
                    )
                )
                policy_extras.update(swing_landing_diagnostics)
                policy_extras["policy_returned_action"] = np.asarray(
                    policy_action, dtype=np.float32
                ).copy()
            if reuse_policy_frame:
                # Requests are edge-triggered. Reusing the held action must not
                # replay a one-shot record/go-home event from the first pass.
                policy_extras["record_start_requested"] = False
                policy_extras["go_home_requested"] = False
            policy_extras["policy_frame_alignment_enabled"] = int(
                self._infer_on_new_frame
            )
            policy_extras["policy_frame_reused"] = int(reuse_policy_frame)
            policy_extras["policy_frame_reuse_count"] = int(
                self._policy_frame_reuse_count
            )
            extras = dict(remote_extras)
            extras.update(policy_extras)
            extras["record_start_requested"] = bool(
                record_start_requested
                or policy_extras.get("record_start_requested", False)
            )
            extras["policy_start_requested"] = start_requested
            extras["policy_remote_mode"] = "policy"
            extras["policy_remote_activated"] = int(activated_now)
            extras["policy_remote_deactivated"] = int(deactivated_now)
            extras["policy_remote_activation_step"] = int(
                -1 if self._activation_step is None else self._activation_step
            )
            extras["policy_remote_toggle_count"] = int(self._toggle_count)
            extras["model_control"] = 1
            extras["scripted_cycle_stop_latched"] = 0
            extras["scripted_cycle_auto_start_after_arm"] = int(
                self._scripted_cycle_auto_start_after_arm
            )
            extras["scripted_cycle_auto_armed"] = int(self._script_auto_armed)
            extras["scripted_cycle_auto_wait_reason"] = str(
                self._script_auto_wait_reason
            )
            extras["scripted_cycle_activation_rejected_reason"] = (
                activation_rejected_reason
                or self._scripted_cycle_activation_rejected_reason
            )
            extras.update(_scripted_cycle_extras(self._scripted_cycle_last_status))
            extras["policy_remote_remote_action"] = np.asarray(
                remote_action, dtype=np.float32
            ).copy()
            info = ActionInfo(
                source_type="policy",
                source_id=f"{getattr(policy_info, 'source_id', self._source_id)}:{self._source_id}",
                latency_ms=float(getattr(policy_info, "latency_ms", 0.0) or 0.0),
                extras=extras,
            )
            self._step += 1
            return np.asarray(policy_action, dtype=np.float32), info

        extras = dict(remote_extras)
        extras["policy_remote_mode"] = "manual"
        extras["policy_remote_activated"] = 0
        extras["policy_remote_deactivated"] = int(deactivated_now)
        extras["policy_remote_activation_step"] = -1
        extras["policy_remote_toggle_count"] = int(self._toggle_count)
        extras["model_control"] = 0
        extras["scripted_cycle_stop_latched"] = int(self._script_stop_latched)
        extras["scripted_cycle_auto_start_after_arm"] = int(
            self._scripted_cycle_auto_start_after_arm
        )
        extras["scripted_cycle_auto_armed"] = int(self._script_auto_armed)
        extras["scripted_cycle_auto_wait_reason"] = str(self._script_auto_wait_reason)
        extras["scripted_cycle_activation_rejected_reason"] = (
            activation_rejected_reason
            or self._scripted_cycle_activation_rejected_reason
        )
        extras.update(_scripted_cycle_extras(self._scripted_cycle_last_status))
        info = ActionInfo(
            source_type=getattr(remote_info, "source_type", "teleop"),
            source_id=f"{getattr(remote_info, 'source_id', 'remote')}:{self._source_id}",
            latency_ms=float(getattr(remote_info, "latency_ms", 0.0) or 0.0),
            extras=extras,
        )
        self._step += 1
        return np.asarray(remote_action, dtype=np.float32), info

    def close(self) -> None:
        try:
            self._remote.close()
        finally:
            self._policy.close()

    def publish_status(self, payload: Mapping[str, Any]) -> None:
        publish = getattr(self._remote, "publish_status", None)
        if callable(publish):
            publish(payload)

    def set_policy_active(self, active: bool) -> bool:
        if bool(active) and self._scripted_cycle_runtime is not None:
            raise RuntimeError(
                "scripted-cycle policy activation requires a live observation; "
                "use the configured remote policy-start event"
            )
        if not bool(active) and self._scripted_cycle_runtime is not None:
            self._scripted_cycle_last_status = self._scripted_cycle_runtime.deactivate(
                "external_disable"
            )
        return self._set_policy_active_state(active)

    def _set_policy_active_state(self, active: bool) -> bool:
        active = bool(active)
        if self._policy_active == active:
            return False
        self._policy_active = active
        self._toggle_count += 1
        if active:
            self._activation_step = self._step
            if hasattr(self._policy, "reset"):
                self._policy.reset()
            self._clear_policy_frame_cache()
        else:
            self._activation_step = None
            self._clear_policy_frame_cache()
        return True

    def policy_status(self) -> dict[str, Any]:
        mode = (
            "policy"
            if self._policy_active
            else ("script_stop" if self._script_stop_latched else "manual")
        )
        status = {
            "policy_remote_mode": mode,
            "model_control": int(self._policy_active),
            "policy_remote_activation_step": int(
                -1 if self._activation_step is None else self._activation_step
            ),
            "policy_remote_toggle_count": int(self._toggle_count),
            "scripted_cycle_stop_latched": int(self._script_stop_latched),
            "scripted_cycle_stop_reason": str(self._script_stop_reason),
            "scripted_cycle_auto_start_after_arm": int(
                self._scripted_cycle_auto_start_after_arm
            ),
            "scripted_cycle_auto_armed": int(self._script_auto_armed),
            "scripted_cycle_auto_wait_reason": str(self._script_auto_wait_reason),
            "scripted_cycle_activation_rejected_reason": str(
                self._scripted_cycle_activation_rejected_reason
            ),
        }
        status.update(_scripted_cycle_extras(self._scripted_cycle_last_status))
        return status

    def _latch_script_stop(self, reason: str) -> None:
        if self._policy_active:
            self._toggle_count += 1
        self._policy_active = False
        self._activation_step = None
        self._script_stop_latched = True
        self._script_stop_reason = str(reason)
        self._script_auto_armed = False
        self._script_auto_wait_reason = ""
        self._clear_policy_frame_cache()

    def _clear_policy_frame_cache(self) -> None:
        self._last_policy_frame_token = None
        self._cached_policy_action = None
        self._cached_policy_info = None
        self._policy_frame_reuse_count = 0


def _observation_image_token(
    obs: Mapping[str, Any],
) -> tuple[tuple[str, int], ...] | None:
    """Identify one immutable multi-camera observation without decoding images."""

    image_timestamps = obs.get("image_timestamp_ns")
    if not isinstance(image_timestamps, Mapping) or not image_timestamps:
        return None
    token: list[tuple[str, int]] = []
    for camera_name, timestamp_ns in sorted(image_timestamps.items()):
        try:
            value = int(timestamp_ns)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        token.append((str(camera_name), value))
    return tuple(token)


def _disabled_scripted_cycle_status() -> dict[str, Any]:
    return {
        "enabled": False,
        "active": False,
        "completed": False,
        "fault": "",
        "stop_reason": "",
        "event": "",
        "goal_changed": False,
        "task_state_advance_requested": False,
        "task_state_changed": False,
        "task_state_advance_ignored": False,
        "task_state_advance_rejected_reason": "",
        "task_state_v2_enabled": False,
        "task_state_stage": "disabled",
        "stop_policy": False,
        "ready_actual_side": "unknown",
        "ready_blockers": [],
        "planner": {"enabled": False},
    }


def _scripted_cycle_extras(status: Mapping[str, Any] | None) -> dict[str, Any]:
    value = dict(status or _disabled_scripted_cycle_status())
    planner = dict(value.get("planner", {}) or {})
    auto_progress = dict(value.get("task_state_auto_progress", {}) or {})
    blockers = [str(item) for item in value.get("ready_blockers", ())]
    planner_task_state = planner.get("task_state_v2")
    if planner_task_state is None:
        planner_task_state = [0.0] * 5
    return {
        "scripted_cycle_enabled": int(bool(value.get("enabled", False))),
        "scripted_cycle_active": int(bool(value.get("active", False))),
        "scripted_cycle_completed": int(bool(value.get("completed", False))),
        "scripted_cycle_fault": str(value.get("fault", "")),
        "scripted_cycle_stop_reason": str(value.get("stop_reason", "")),
        "scripted_cycle_event": str(value.get("event", "")),
        "scripted_cycle_goal_changed": int(bool(value.get("goal_changed", False))),
        "scripted_cycle_task_state_advance_requested": int(
            bool(value.get("task_state_advance_requested", False))
        ),
        "scripted_cycle_task_state_changed": int(
            bool(value.get("task_state_changed", False))
        ),
        "scripted_cycle_task_state_advance_ignored": int(
            bool(value.get("task_state_advance_ignored", False))
        ),
        "scripted_cycle_task_state_v2_enabled": int(
            bool(value.get("task_state_v2_enabled", False))
        ),
        "scripted_cycle_task_state_require_excursion": int(
            bool(value.get("task_state_require_excursion_before_work_complete", False))
        ),
        "scripted_cycle_task_state_stage": str(
            value.get("task_state_stage", "disabled")
        ),
        "scripted_cycle_task_state_advance_source": str(
            value.get("task_state_advance_source", "")
        ),
        "scripted_cycle_task_state_advance_rejected_reason": str(
            value.get("task_state_advance_rejected_reason", "")
        ),
        "scripted_cycle_task_auto_progress_enabled": int(
            bool(auto_progress.get("enabled", False))
        ),
        "scripted_cycle_task_auto_work_liveness": int(
            bool(auto_progress.get("work_liveness_observed", False))
        ),
        "scripted_cycle_task_auto_bucket_effective_observed": int(
            bool(auto_progress.get("bucket_effective_observed", False))
        ),
        "scripted_cycle_task_auto_bucket_effective_count": int(
            auto_progress.get("bucket_effective_count", 0) or 0
        ),
        "scripted_cycle_task_auto_bucket_release_count": int(
            auto_progress.get("bucket_release_count", 0) or 0
        ),
        "scripted_cycle_task_auto_return_idle_count": int(
            auto_progress.get("return_idle_count", 0) or 0
        ),
        "scripted_cycle_task_auto_pending_event": str(
            auto_progress.get("pending_event", "")
        ),
        "scripted_cycle_task_auto_last_event": str(auto_progress.get("last_event", "")),
        "scripted_cycle_task_auto_max_qpos_delta_rad": np.asarray(
            auto_progress.get("max_qpos_delta_rad", [0.0] * 4),
            dtype=np.float32,
        ),
        "scripted_cycle_task_state_applied_event": str(
            value.get("task_state_applied_event", "")
        ),
        "scripted_cycle_phase_changed": int(bool(value.get("phase_changed", False))),
        "scripted_cycle_excursion_changed": int(
            bool(value.get("excursion_changed", False))
        ),
        "scripted_cycle_excursion_observed": int(
            bool(value.get("excursion_observed", False))
        ),
        "scripted_cycle_return_phase_latched": int(
            bool(value.get("return_phase_latched", False))
        ),
        "scripted_cycle_policy_return_phase_latched": int(
            bool(value.get("policy_return_phase_latched", False))
        ),
        "scripted_cycle_landing_pd_blend": float(
            value.get("landing_pd_blend", 0.0) or 0.0
        ),
        "scripted_cycle_landing_policy_gain": float(
            value.get("landing_policy_gain", 1.0)
        ),
        "scripted_cycle_review_due": int(bool(value.get("review_due", False))),
        "scripted_cycle_cycle_elapsed_s": float(
            value.get("cycle_elapsed_s", 0.0) or 0.0
        ),
        "scripted_cycle_run_elapsed_s": float(value.get("run_elapsed_s", 0.0) or 0.0),
        "scripted_cycle_ready_side": str(value.get("ready_actual_side", "unknown")),
        "scripted_cycle_ready_blockers": ",".join(blockers),
        "scripted_cycle_ready_window_complete": int(
            bool(value.get("ready_window_complete", False))
        ),
        "scripted_cycle_ready_swing_stable": int(
            bool(value.get("ready_swing_stable", False))
        ),
        "scripted_cycle_ready_target_supported": int(
            bool(value.get("ready_target_supported", False))
        ),
        "scripted_cycle_ready_swing_qpos_rad": float(
            value.get("ready_swing_qpos_rad", 0.0) or 0.0
        ),
        "scripted_cycle_ready_swing_qvel_abs_max_rad_s": float(
            value.get("ready_swing_qvel_abs_max_rad_s", 0.0) or 0.0
        ),
        "planner_cycle_index": int(planner.get("cycle_index", -1)),
        "planner_goal_epoch": int(planner.get("goal_epoch", -1)),
        "planner_type": str(planner.get("planner_type", "") or ""),
        "planner_selected_initial_side": str(
            planner.get("selected_initial_side", "") or ""
        ),
        "planner_available_initial_sides": ",".join(
            str(side) for side in planner.get("available_initial_sides", ())
        ),
        "planner_script_id": str(planner.get("script_id", "") or ""),
        "planner_script_path": str(planner.get("script_path", "") or ""),
        "planner_planned_cycle_count": int(planner.get("planned_cycle_count", 0) or 0),
        "planner_target_side": str(planner.get("target_side") or ""),
        "planner_condition": np.asarray(
            planner.get("condition") or [0.0, 0.0], dtype=np.float32
        ),
        "planner_task_dig_complete": int(bool(planner.get("task_dig_complete", False))),
        "planner_task_return_commit": int(
            bool(planner.get("task_return_commit", False))
        ),
        "planner_task_state_epoch": int(planner.get("task_state_epoch", 0) or 0),
        "planner_task_state_v2": np.asarray(planner_task_state, dtype=np.float32),
    }
