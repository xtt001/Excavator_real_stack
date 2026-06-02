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
    ) -> None:
        self._remote = remote
        self._policy = policy
        self._source_id = str(source_id)
        self._start_in_policy = bool(start_in_policy)
        self._policy_active = bool(start_in_policy)
        self._step = 0
        self._activation_step: int | None = 0 if start_in_policy else None

    @classmethod
    def from_config(
        cls, teleop_cfg: Mapping[str, Any] | None
    ) -> "RemoteArmedPolicyActionSource":
        cfg = dict(teleop_cfg or {})
        mode_cfg = dict(cfg.get("policy_remote", {}) or {})
        return cls(
            remote=RemoteActionSource.from_config(cfg.get("remote", {})),
            policy=PolicyActionSource.from_config(cfg.get("policy", {})),
            source_id=str(mode_cfg.get("source_id", "policy_remote")),
            start_in_policy=bool(mode_cfg.get("start_in_policy", False)),
        )

    def reset(self) -> None:
        self._remote.reset()
        self._policy.reset()
        self._policy_active = self._start_in_policy
        self._step = 0
        self._activation_step = 0 if self._start_in_policy else None

    def next_action(self, obs: dict[str, Any]) -> tuple[np.ndarray, ActionInfo]:
        remote_action, remote_info = self._remote.next_action(obs)
        remote_extras = dict(getattr(remote_info, "extras", {}) or {})
        start_requested = bool(remote_extras.get("policy_start_requested", False))
        record_start_requested = bool(
            remote_extras.get("record_start_requested", False)
        )
        activated_now = False
        if start_requested and not self._policy_active:
            self._policy_active = True
            self._activation_step = self._step
            activated_now = True

        if self._policy_active:
            policy_action, policy_info = self._policy.next_action(obs)
            policy_extras = dict(getattr(policy_info, "extras", {}) or {})
            extras = dict(remote_extras)
            extras.update(policy_extras)
            extras["record_start_requested"] = bool(
                record_start_requested
                or policy_extras.get("record_start_requested", False)
            )
            extras["policy_start_requested"] = start_requested
            extras["policy_remote_mode"] = "policy"
            extras["policy_remote_activated"] = int(activated_now)
            extras["policy_remote_activation_step"] = int(
                -1 if self._activation_step is None else self._activation_step
            )
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
        extras["policy_remote_activation_step"] = -1
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
