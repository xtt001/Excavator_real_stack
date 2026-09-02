"""Sampling-only balance for factual work and target semantics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "real_transition_simple_factual_semantic_sampling_manifest_v1"
TIER_NAMES = ("stable_work", "target_semantic")


def resolve_factual_semantic_sampling_config(raw: Any) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("factual_semantic_sampling config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "scope": "train_only",
            "append_samples_per_episode": 0,
            "manifest_path": None,
            "tier_names": TIER_NAMES,
        }
    scope = str(cfg.get("scope", "train_only"))
    if scope != "train_only":
        raise ValueError("factual_semantic_sampling.scope must be train_only")
    manifest_raw = cfg.get("manifest_path")
    if manifest_raw is None or not str(manifest_raw).strip():
        raise ValueError("factual_semantic_sampling.manifest_path is required")
    manifest_path = Path(str(manifest_raw)).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    append = _positive_integer(
        cfg.get("append_samples_per_episode", len(TIER_NAMES)),
        name="append_samples_per_episode",
    )
    if append != len(TIER_NAMES):
        raise ValueError(
            "factual_semantic_sampling.append_samples_per_episode must be 2"
        )
    return {
        "enabled": True,
        "scope": scope,
        "append_samples_per_episode": append,
        "manifest_path": str(manifest_path),
        "tier_names": TIER_NAMES,
    }


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"factual_semantic_sampling.{name} must be boolean")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0 or float(value) != float(parsed):
        raise ValueError(f"factual_semantic_sampling.{name} must be a positive integer")
    return parsed
