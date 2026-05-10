"""
Diffusion policy adapter — SKELETON (Step1+).

This file is intentionally left as a stub so that:
  1. The testbed package can be imported without any diffusion dependencies.
  2. CI and smoke tests confirm the Policy registry works.
  3. Filling in the implementation is a self-contained task.

To implement:
  • Install diffusion_policy or write a custom DDPM/DDIM network.
  • Populate __init__, predict, and register a DiffusionTrainer under
    testbed/policies/diffusion/trainer.py.
"""

from __future__ import annotations

import numpy as np

from testbed.policies.base import Policy, register_policy


@register_policy("diffusion")
class DiffusionAdapter(Policy):
    """
    Placeholder for a diffusion policy.

    Raises NotImplementedError on all calls — replace with a real
    implementation when needed.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DiffusionAdapter is not yet implemented. "
            "See testbed/policies/diffusion/adapter.py for instructions."
        )

    def predict(self, obs: dict) -> np.ndarray:
        raise NotImplementedError
