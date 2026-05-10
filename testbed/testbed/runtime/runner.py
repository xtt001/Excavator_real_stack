"""
Runner helpers for YAML-driven training.

Recording is handled by the explicit ``tb-record-real`` CLI so hardware-facing
code stays behind the real backend/controller boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Runner:
    """Small orchestrator for config-driven policy training."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @classmethod
    def from_yaml(cls, *yaml_paths: str | Path) -> "Runner":
        """Load and merge one or more YAML configs."""

        merged: dict[str, Any] = {}
        for path in yaml_paths:
            with open(path) as f:
                merged.update(yaml.safe_load(f) or {})
        return cls(merged)

    def train(self) -> None:
        """Train the configured policy on recorded real-excavator episodes."""

        from testbed.runtime._train import train_policy

        train_policy(self.config)
