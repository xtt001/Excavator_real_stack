"""Remote-armed policy action source for live policy tests."""

from __future__ import annotations

from typing import Any, Mapping

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
    ) -> None:
        self._remote = remote
        self._policy = policy
        self._source_id = str(source_id)
        self._start_in_policy = bool(start_in_policy)
        self._infer_on_new_frame = bool(infer_on_new_frame)
        self._policy_active = bool(start_in_policy)
        self._step = 0
        self._activation_step: int | None = 0 if start_in_policy else None
        self._toggle_count = 0
        self._last_policy_frame_token: tuple[tuple[str, int], ...] | None = None
        self._cached_policy_action: np.ndarray | None = None
        self._cached_policy_info: ActionInfo | None = None
        self._policy_frame_reuse_count = 0

    @classmethod
    def from_config(
        cls, teleop_cfg: Mapping[str, Any] | None
    ) -> "RemoteArmedPolicyActionSource":
        cfg = dict(teleop_cfg or {})
        mode_cfg = dict(cfg.get("policy_remote", {}) or {})
        policy_cfg = dict(cfg.get("policy", {}) or {})
        frame_alignment_cfg = dict(policy_cfg.get("frame_alignment", {}) or {})
        return cls(
            remote=RemoteActionSource.from_config(cfg.get("remote", {})),
            policy=PolicyActionSource.from_config(policy_cfg),
            source_id=str(mode_cfg.get("source_id", "policy_remote")),
            start_in_policy=bool(mode_cfg.get("start_in_policy", False)),
            infer_on_new_frame=bool(frame_alignment_cfg.get("enabled", False)),
        )

    def reset(self) -> None:
        self._remote.reset()
        self._policy.reset()
        self._policy_active = self._start_in_policy
        self._step = 0
        self._activation_step = 0 if self._start_in_policy else None
        self._toggle_count = 0
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
        record_start_requested = bool(
            remote_extras.get("record_start_requested", False)
        )
        activated_now = False
        deactivated_now = False
        if start_requested and not self._policy_active:
            activated_now = self.set_policy_active(True)
        elif start_requested and self._policy_active:
            deactivated_now = self.set_policy_active(False)

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
        mode = "policy" if self._policy_active else "manual"
        return {
            "policy_remote_mode": mode,
            "model_control": int(self._policy_active),
            "policy_remote_activation_step": int(
                -1 if self._activation_step is None else self._activation_step
            ),
            "policy_remote_toggle_count": int(self._toggle_count),
        }

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
