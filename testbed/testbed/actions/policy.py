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

POLICY_OUTPUT_MODES = ("control", "shadow_zero")
POLICY_QVEL_MODES = ("raw", "zero", "qpos_diff")
ACTION_AXIS_NAMES = ("swing", "boom", "stick", "bucket")


@dataclass(frozen=True)
class DeadzoneAssistConfig:
    enabled: bool
    trigger_fraction: float
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
        bundle_dir: str | Path | None = None,
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
        self._bundle_dir = None if bundle_dir is None else str(bundle_dir)
        self._step = 0
        self._record_start_pending = self._record_start_on_reset
        self._last_qpos: np.ndarray | None = None
        self._last_obs_time_ns: int | None = None
        self._filtered_qvel = np.zeros(4, dtype=np.float32)
        self._assist_last_sign = np.zeros(4, dtype=np.int8)
        self._assist_consecutive_steps = np.zeros(4, dtype=np.int32)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> PolicyActionSource:
        cfg = dict(config or {})
        bundle_dir = Path(cfg.get("bundle_dir", "policy_bundles/real_one_dig_v1"))
        policy = load_act_policy_from_bundle(
            bundle_dir=bundle_dir,
            ckpt_path=cfg.get("ckpt_path"),
            resolved_config_path=cfg.get("resolved_config_path"),
            stats_path=cfg.get("stats_path"),
            device=cfg.get("device"),
            temporal_agg=bool(cfg.get("temporal_agg", True)),
            temporal_aggregation_diagnostics=bool(
                cfg.get("temporal_aggregation_diagnostics", False)
            ),
            inference_precision=str(cfg.get("inference_precision", "fp32")),
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
            bundle_dir=bundle_dir,
        )

    def reset(self) -> None:
        self._step = 0
        self._record_start_pending = self._record_start_on_reset
        self._last_qpos = None
        self._last_obs_time_ns = None
        self._filtered_qvel.fill(0.0)
        self._assist_last_sign.fill(0)
        self._assist_consecutive_steps.fill(0)
        if self._runtime_gate_stack is not None:
            self._runtime_gate_stack.reset()
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def snapshot_state(self) -> PolicyActionSourceState:
        """Capture policy, gate, qvel, assist, and source sequence state."""

        policy_snapshot = getattr(self._policy, "snapshot_state", None)
        if not callable(policy_snapshot):
            raise TypeError("policy does not implement snapshot_state()")
        gate_state = None
        if self._runtime_gate_stack is not None:
            gate_snapshot = getattr(self._runtime_gate_stack, "snapshot_state", None)
            if not callable(gate_snapshot):
                raise TypeError("runtime gate stack does not implement snapshot_state()")
            gate_state = gate_snapshot()
        return PolicyActionSourceState(
            step=int(self._step),
            record_start_pending=bool(self._record_start_pending),
            last_qpos=(
                None if self._last_qpos is None else self._last_qpos.copy()
            ),
            last_obs_time_ns=(
                None
                if self._last_obs_time_ns is None
                else int(self._last_obs_time_ns)
            ),
            filtered_qvel=self._filtered_qvel.copy(),
            assist_last_sign=self._assist_last_sign.copy(),
            assist_consecutive_steps=self._assist_consecutive_steps.copy(),
            policy_state=policy_snapshot(),
            runtime_gate_state=gate_state,
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
        self._step = int(state.step)
        self._record_start_pending = bool(state.record_start_pending)
        self._last_qpos = (
            None if state.last_qpos is None else state.last_qpos.copy()
        )
        self._last_obs_time_ns = (
            None
            if state.last_obs_time_ns is None
            else int(state.last_obs_time_ns)
        )
        self._filtered_qvel = np.asarray(
            state.filtered_qvel, dtype=np.float32
        ).copy()
        self._assist_last_sign = np.asarray(
            state.assist_last_sign, dtype=np.int8
        ).copy()
        self._assist_consecutive_steps = np.asarray(
            state.assist_consecutive_steps, dtype=np.int32
        ).copy()
        policy_restore(state.policy_state)
        if self._runtime_gate_stack is not None:
            gate_restore = getattr(self._runtime_gate_stack, "restore_state", None)
            if not callable(gate_restore):
                raise TypeError("runtime gate stack does not implement restore_state()")
            gate_restore(state.runtime_gate_state)

    def next_action(self, obs: dict[str, Any]) -> tuple[np.ndarray, ActionInfo]:
        t0 = time.perf_counter()
        now_ns = time.time_ns()
        record_start_requested = self._consume_record_start_request()
        try:
            policy_obs, qvel_input = self._policy_obs(obs)
            gate_extras: dict[str, Any] = {}
            raw_gohome_requested = False
            if self._runtime_gate_stack is None:
                predicted_action = self._policy.predict(policy_obs)
            else:
                inference = getattr(self._policy, "predict_action_and_intent", None)
                if not callable(inference):
                    raise TypeError(
                        "runtime_gates requires policy.predict_action_and_intent(obs)"
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
                "policy_inference_precision": str(
                    getattr(self._policy, "inference_precision", "fp32")
                ),
                "policy_step": int(self._step),
                "policy_error": "",
                **assist_extras,
                **gate_extras,
            }
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
                "policy_step": int(self._step),
                "policy_error": f"{type(exc).__name__}: {exc}",
                **_deadzone_assist_disabled_extras(self._deadzone_assist),
            }
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
        intent = (sign != 0) & (magnitude >= cfg.trigger_fraction * threshold)

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
            "policy_deadzone_assist_active": int(bool(np.any(assist_mask))),
            "policy_deadzone_assist_mask": assist_mask.astype(np.int32),
            "policy_deadzone_assist_axes": axes,
            "policy_deadzone_assist_trigger_fraction": float(cfg.trigger_fraction),
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
    temporal_aggregation_diagnostics: bool = False,
    inference_precision: str = "fp32",
) -> Any:
    """Load the ACT policy bundle produced by excavator_testbed."""

    bundle = Path(bundle_dir)
    resolved_path = (
        Path(resolved_config_path)
        if resolved_config_path
        else bundle / "resolved_config.yaml"
    )
    ckpt = Path(ckpt_path) if ckpt_path else bundle / "policy_best.ckpt"
    stats = Path(stats_path) if stats_path else bundle / "dataset_stats.pkl"
    missing = [path for path in (resolved_path, ckpt, stats) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing policy bundle file(s): " + ", ".join(str(path) for path in missing)
        )

    with resolved_path.open("r", encoding="utf-8") as f:
        resolved = yaml.safe_load(f) or {}
    policy_config = _act_policy_config_from_resolved(resolved)

    from testbed.policies.act.adapter import ACTAdapter

    return ACTAdapter.from_checkpoint(
        ckpt_path=ckpt,
        policy_config=policy_config,
        norm_stats_path=stats,
        temporal_agg=bool(temporal_agg),
        temporal_aggregation_diagnostics=bool(
            temporal_aggregation_diagnostics
        ),
        device=str(device or resolved.get("policy", {}).get("device", "cuda")),
        inference_precision=inference_precision,
    )


def _act_policy_config_from_resolved(resolved: dict[str, Any]) -> dict[str, Any]:
    task_cfg = dict(resolved.get("task", {}) or {})
    policy_cfg = dict(resolved.get("policy", {}) or {})
    train_cfg = dict(resolved.get("train", {}) or {})
    act_params = dict(policy_cfg.get("act_params", {}) or {})
    camera_names = list(task_cfg.get("camera_names", ["fpv"]))
    low_dim_keys = list(policy_cfg.get("low_dim_keys", ["qpos", "qvel"]))
    state_dim = int(act_params.get("state_dim", 4 * len(low_dim_keys)))
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
    trigger_fraction = float(cfg.get("trigger_fraction", 0.5))
    if not 0.0 < trigger_fraction <= 1.0:
        raise ValueError("deadzone_assist.trigger_fraction must be in (0, 1]")
    margin = _broadcast_nonnegative_vector4(
        cfg.get("margin", 0.02),
        name="deadzone_assist.margin",
    )
    min_consecutive_steps = int(cfg.get("min_consecutive_steps", 2))
    if min_consecutive_steps < 1:
        raise ValueError("deadzone_assist.min_consecutive_steps must be >= 1")
    return DeadzoneAssistConfig(
        enabled=enabled,
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


def _deadzone_assist_disabled_extras(
    cfg: DeadzoneAssistConfig,
) -> dict[str, Any]:
    return {
        "policy_deadzone_assist_enabled": int(bool(cfg.enabled)),
        "policy_deadzone_assist_active": 0,
        "policy_deadzone_assist_mask": np.zeros(4, dtype=np.int32),
        "policy_deadzone_assist_axes": "",
        "policy_deadzone_assist_trigger_fraction": float(cfg.trigger_fraction),
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
