"""OEM remote-control action source boundary.

The first hardware-facing dataset should capture the command the operator meant
to send, not only the machine motion that happened later. This module provides
an import-safe adapter shape for reading the vendor remote stream when that
interface is available.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from testbed.actions.base import ActionInfo, ActionSource
from testbed.backends.real.contracts import as_real_action


class OemRemoteUnavailableError(RuntimeError):
    """Raised when OEM remote input is requested but no reader is configured."""


class OemRemoteActionSource(ActionSource):
    """Action source for a vendor remote, with an explicit zero-action stub."""

    def __init__(
        self,
        *,
        reader: Any | None = None,
        source_id: str = "oem_remote",
        allow_stub: bool = False,
    ) -> None:
        self._reader = reader
        self._source_id = str(source_id)
        self._allow_stub = bool(allow_stub)
        if self._reader is None and not self._allow_stub:
            raise OemRemoteUnavailableError(
                "OEM remote input was selected, but no remote reader is configured. "
                "Provide a reader in the hardware integration layer or set "
                "teleop.oem_remote.allow_stub=true for local interface checks."
            )

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "OemRemoteActionSource":
        cfg = dict(config or {})
        return cls(
            reader=cfg.get("reader"),
            source_id=str(cfg.get("source_id", "oem_remote")),
            allow_stub=bool(cfg.get("allow_stub", False)),
        )

    def reset(self) -> None:
        if hasattr(self._reader, "reset"):
            self._reader.reset()

    def next_action(self, obs: dict[str, Any]) -> tuple[np.ndarray, ActionInfo]:
        if self._reader is None:
            now_ns = time.time_ns()
            return (
                np.zeros(4, dtype=np.float32),
                ActionInfo(
                    source_type="teleop",
                    source_id=f"{self._source_id}_stub",
                    extras={
                        "action_timestamp_ns": now_ns,
                        "remote_available": False,
                        "learning_target": "operator_command",
                    },
                ),
            )

        read_started_ns = time.time_ns()
        result = self._read_remote(obs)
        action, timestamp_ns, extras = _parse_remote_result(result)
        now_ns = time.time_ns()
        extras = dict(extras)
        extras.setdefault("action_timestamp_ns", int(timestamp_ns or now_ns))
        extras.setdefault("remote_available", True)
        extras.setdefault("learning_target", "operator_command")
        latency_ms = max(0.0, (now_ns - int(extras["action_timestamp_ns"])) * 1e-6)
        extras.setdefault("read_duration_ms", max(0.0, (now_ns - read_started_ns) * 1e-6))
        return (
            as_real_action(action, clip=True),
            ActionInfo(
                source_type="teleop",
                source_id=self._source_id,
                latency_ms=latency_ms,
                extras=extras,
            ),
        )

    def close(self) -> None:
        if hasattr(self._reader, "close"):
            self._reader.close()

    def _read_remote(self, obs: dict[str, Any]) -> Any:
        if hasattr(self._reader, "read_action"):
            return self._reader.read_action(obs)
        if hasattr(self._reader, "read"):
            return self._reader.read()
        if callable(self._reader):
            return self._reader(obs)
        raise OemRemoteUnavailableError("configured OEM remote reader is not callable")


def _parse_remote_result(result: Any) -> tuple[np.ndarray, int | None, dict[str, Any]]:
    if isinstance(result, dict):
        if "action" in result:
            action = result["action"]
        elif "axes" in result:
            action = result["axes"]
        else:
            raise KeyError("OEM remote result dict must contain 'action' or 'axes'")
        timestamp_ns = result.get("timestamp_ns") or result.get("action_timestamp_ns")
        extras = dict(result.get("extras", {}))
        return as_real_action(action, clip=True), _optional_int(timestamp_ns), extras

    if isinstance(result, tuple):
        if len(result) == 2:
            action, second = result
            if isinstance(second, dict):
                timestamp_ns = second.get("timestamp_ns") or second.get("action_timestamp_ns")
                return as_real_action(action, clip=True), _optional_int(timestamp_ns), dict(second)
            return as_real_action(action, clip=True), _optional_int(second), {}
        if len(result) == 3:
            action, timestamp_ns, extras = result
            return as_real_action(action, clip=True), _optional_int(timestamp_ns), dict(extras or {})

    return as_real_action(result, clip=True), None, {}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
