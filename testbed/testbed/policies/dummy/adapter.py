"""
DummyPolicy — proves hot-swap works without any model weights.

Useful for:
  • Smoke-testing policy hot-swap behavior quickly.
  • Verifying recorder and policy adapter plumbing before a real training run.
  • Baseline measurement of "random / zero" policy performance.

Modes
-----
  "zero"   Always returns a zero-filled action (default).
  "random" Returns uniform random actions in [-1, 1].
  "replay" Replays a fixed pre-loaded action sequence (provide sequence= arg).
"""

from __future__ import annotations

import numpy as np

from testbed.policies.base import Policy, register_policy


@register_policy("dummy")
class DummyPolicy(Policy):
    """
    A no-op policy that always outputs zeros (or random noise).

    Parameters
    ----------
    action_dim   Dimension of the action vector.
    mode         "zero" | "random" | "replay".
    sequence     (T, Na) array used when mode="replay".
    """

    def __init__(
        self,
        action_dim: int,
        mode: str = "zero",
        sequence: np.ndarray | None = None,
    ):
        if mode not in {"zero", "random", "replay"}:
            raise ValueError(f"Unknown DummyPolicy mode: {mode!r}")
        if mode == "replay" and sequence is None:
            raise ValueError("mode='replay' requires sequence= argument.")
        self.action_dim = action_dim
        self.mode       = mode
        self.sequence   = sequence
        self._t         = 0

    def reset(self) -> None:
        self._t = 0

    def predict(self, obs: dict) -> np.ndarray:
        if self.mode == "zero":
            return np.zeros(self.action_dim, dtype=np.float32)

        if self.mode == "random":
            return np.random.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

        # replay
        assert self.sequence is not None
        idx = min(self._t, len(self.sequence) - 1)
        action = self.sequence[idx].astype(np.float32)
        self._t += 1
        return action
