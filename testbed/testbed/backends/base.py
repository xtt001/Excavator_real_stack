"""Backend interface for hot-pluggable real excavator integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Backend(ABC):
    """
    Minimal backend interface consumed by record/train utilities.

    A backend exposes the same observation shape regardless of whether the
    state comes from a mock adapter, a TCP bridge, ROS topics, or direct CAN
    integration.
    """

    @abstractmethod
    def start_episode(self, seed: int | None = None) -> Any:
        """Mark an operator-controlled episode boundary and return a timestep."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> Any:
        """Apply one normalized 4-axis action and return the next timestep."""
        raise NotImplementedError

    @abstractmethod
    def render(self, camera_id: str, height: int = 480, width: int = 640) -> np.ndarray:
        """Return the latest image for a camera, resized if requested."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dt(self) -> float:
        """Control timestep in seconds."""
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources."""
        return None
