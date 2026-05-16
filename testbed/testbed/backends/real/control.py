"""Low-level control interface for real excavator integrations.

The testbed owns normalized four-axis commands. Vendor/CAN/valve-specific
details must live behind LowLevelController implementations so data collection,
training, and policy code do not depend on a particular real machine protocol.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from testbed.backends.real.contracts import REAL_ACTION_DIM, as_real_action


ACTION_DIM = REAL_ACTION_DIM


@dataclass(frozen=True)
class ControlResult:
    """Result returned after sending one low-level control command."""

    ack: bool
    fault_code: str
    controller_timestamp_ns: int
    commanded_action: np.ndarray
    raw_low_level_command: np.ndarray | None = None


class LowLevelController(ABC):
    """Abstract adapter from normalized action to machine-specific control."""

    @abstractmethod
    def send(self, action: np.ndarray, state: dict[str, Any] | None = None) -> ControlResult:
        """Send one normalized four-axis action to the low-level controller."""

    def apply_status_toggle_mask(self, toggle_mask: int) -> bool:
        """Flip discrete machine status bits (excavator_api toggle_mask)."""

        return int(toggle_mask) == 0

    def close(self) -> None:
        """Release controller resources."""


def _as_action(action: np.ndarray) -> np.ndarray:
    return as_real_action(action, clip=False)


class MockLowLevelController(LowLevelController):
    """Controller used by mock backend and tests.

    It acknowledges every valid action and records the last command, while
    leaving the raw low-level command equal to the normalized command.
    """

    def __init__(self) -> None:
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.send_count = 0
        self.last_toggle_mask = 0
        self.status11: list[int] = [0] * 11

    def send(self, action: np.ndarray, state: dict[str, Any] | None = None) -> ControlResult:
        commanded = np.clip(_as_action(action), -1.0, 1.0).astype(np.float32)
        self.last_action = commanded.copy()
        self.send_count += 1
        return ControlResult(
            ack=True,
            fault_code="",
            controller_timestamp_ns=time.time_ns(),
            commanded_action=commanded,
            raw_low_level_command=commanded.copy(),
        )

    def apply_status_toggle_mask(self, toggle_mask: int) -> bool:
        from testbed.backends.real.contracts import apply_status_toggle_mask_to_status11

        mask = int(toggle_mask)
        self.last_toggle_mask = mask
        if mask == 0:
            return True
        apply_status_toggle_mask_to_status11(self.status11, mask)
        return True


class NoopLowLevelController(LowLevelController):
    """Controller that never drives hardware and always commands zeros."""

    def __init__(self) -> None:
        self.send_count = 0
        self.last_toggle_mask = 0

    def send(self, action: np.ndarray, state: dict[str, Any] | None = None) -> ControlResult:
        _as_action(action)
        self.send_count += 1
        zero = np.zeros(ACTION_DIM, dtype=np.float32)
        return ControlResult(
            ack=True,
            fault_code="noop",
            controller_timestamp_ns=time.time_ns(),
            commanded_action=zero,
            raw_low_level_command=zero.copy(),
        )

    def apply_status_toggle_mask(self, toggle_mask: int) -> bool:
        self.last_toggle_mask = int(toggle_mask)
        return True
