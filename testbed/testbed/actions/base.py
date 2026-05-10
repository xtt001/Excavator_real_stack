"""
ActionSource — abstract base class for all action generators.

Both teleop devices (joystick, keyboard) and policies (ACT, diffusion)
implement this interface so the runner loop never needs to know which one
is active.

Interface contract
──────────────────
  action_source.reset()
      Called once before each episode begins.

  action, info = action_source.next_action(obs)
      obs  — raw observation dict from the backend
      Returns (np.ndarray shape=(Na,), ActionInfo)

  action_source.close()
      Called when the session ends (release hardware, threads, etc.)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ActionInfo:
    """
    Per-step metadata about where the action came from.

    Stored in HDF5 /action_source.
    """
    source_type: str = "unknown"   # "teleop" | "policy" | "scripted"
    source_id:   str = "unknown"   # "joystick" | "keyboard" | "policy:act@ckpt"
    latency_ms:  float = 0.0       # optional, 0 = not measured
    extras:      dict[str, Any] = field(default_factory=dict)


class ActionSource(abc.ABC):
    """
    Abstract base for any action-generating entity.

    Subclasses
    ──────────
    JoystickActionSource  — reads a gamepad via pygame
    KeyboardActionSource  — reads keyboard via pygame
    PolicyActionSource    — wraps a Policy for offline inference utilities
    """

    @abc.abstractmethod
    def reset(self) -> None:
        """Called at the start of each episode."""

    @abc.abstractmethod
    def next_action(self, obs: dict) -> tuple[np.ndarray, ActionInfo]:
        """
        Produce the next action given the current observation.

        Returns
        -------
        action : np.ndarray  shape (Na,)
        info   : ActionInfo
        """

    def close(self) -> None:
        """Release any resources (hardware, threads). Default: no-op."""

    def __enter__(self) -> "ActionSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
