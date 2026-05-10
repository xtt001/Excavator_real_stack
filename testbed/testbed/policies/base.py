"""
Abstract base classes for policies and trainers.

The plugin contract:
  • Every deployable policy extends Policy and implements predict().
  • Every training loop extends Trainer and implements fit().
  • Register plugins with @register_policy("name") so the CLI can
    instantiate them from a config YAML without hardcoded if-else chains.

Example
-------
    @register_policy("my_policy")
    class MyPolicy(Policy):
        def predict(self, obs: dict) -> np.ndarray:
            ...
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import numpy as np


# ─── Policy ABC ───────────────────────────────────────────────────────────────

class Policy(abc.ABC):
    """
    Base class for all deployable policies.

    Subclasses must implement:
      predict(obs)  → action np.ndarray

    Optionally implement:
      reset()       → called at the start of each inference episode
      update(batch) → called during training (if the policy has an online
                      learning loop; most offline policies leave this as no-op)
    """

    @abc.abstractmethod
    def predict(self, obs: dict) -> np.ndarray:
        """
        Predict an action from an observation dict.

        Parameters
        ----------
        obs   Raw observation dict (or pre-converted dict from
              StructuredState.as_policy_input()).

        Returns
        -------
        action : np.ndarray  shape (Na,)
        """

    def reset(self) -> None:
        """Called at the beginning of every inference episode."""

    def update(self, batch: Any) -> dict[str, float]:
        """
        Optional online update step (no-op by default).
        Returns a dict of scalar metrics.
        """
        return {}


# ─── Trainer ABC ──────────────────────────────────────────────────────────────

class Trainer(abc.ABC):
    """
    Base class for offline policy trainers.

    Subclasses must implement:
      fit(train_loader, val_loader, config)
      save(ckpt_dir)
      load(ckpt_path) → Policy
    """

    @abc.abstractmethod
    def fit(
        self,
        train_loader,
        val_loader,
        config: dict,
    ) -> tuple[int, float, dict]:
        """
        Run the full training loop.

        Returns
        -------
        (best_epoch, min_val_loss, best_state_dict)
        """

    @abc.abstractmethod
    def save(self, ckpt_dir: Path | str, tag: str = "best") -> Path:
        """Save checkpoint to ckpt_dir / policy_{tag}.ckpt."""

    @abc.abstractmethod
    def load(self, ckpt_path: Path | str) -> Policy:
        """Load a checkpoint and return the runnable Policy."""


# ─── Registry ─────────────────────────────────────────────────────────────────

class PolicyRegistry:
    """
    Global registry mapping string names → Policy subclasses.

    Usage
    -----
    Policy class registration:
        @register_policy("act")
        class ACTAdapter(Policy): ...

    Instantiation from config:
        policy_cls = PolicyRegistry.get("act")
        policy = policy_cls(config)
    """

    _registry: dict[str, type[Policy]] = {}

    @classmethod
    def register(cls, name: str, policy_cls: type[Policy]) -> None:
        if name in cls._registry:
            raise ValueError(f"Policy '{name}' is already registered.")
        cls._registry[name] = policy_cls

    @classmethod
    def get(cls, name: str) -> type[Policy]:
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Policy '{name}' not found. Available: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())


def register_policy(name: str):
    """Decorator to register a Policy subclass by name."""
    def decorator(cls: type[Policy]) -> type[Policy]:
        PolicyRegistry.register(name, cls)
        return cls
    return decorator


# ─── Utility helpers shared across policies ───────────────────────────────────

def compute_dict_mean(epoch_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of loss dicts (one per batch) into one dict."""
    if not epoch_dicts:
        return {}
    result = {k: 0.0 for k in epoch_dicts[0]}
    for d in epoch_dicts:
        for k, v in d.items():
            result[k] += float(v)
    n = len(epoch_dicts)
    return {k: v / n for k, v in result.items()}


def detach_dict(d: dict) -> dict:
    return {k: v.detach() for k, v in d.items()}


def set_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
