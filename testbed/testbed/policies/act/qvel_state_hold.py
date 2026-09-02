"""Direct qvel-zero state-hold supervision for ACT transition anchors."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_qvel_zero_state_hold_config(raw: Any) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("qvel_zero_state_hold_loss config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "weight": 0.0,
            "window_steps": 1,
            "margin": 0.0,
            "positive": np.zeros(4, dtype=np.float32),
            "negative": np.zeros(4, dtype=np.float32),
            "threshold_path": None,
        }
    threshold_raw = cfg.get("threshold_json")
    if threshold_raw is None or not str(threshold_raw).strip():
        raise ValueError("qvel_zero_state_hold_loss.threshold_json is required")
    threshold_path = Path(str(threshold_raw))
    if not threshold_path.is_file():
        raise FileNotFoundError(threshold_path)
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    deadzone = payload.get("deadzone_action")
    if not isinstance(deadzone, Mapping):
        raise ValueError("deadzone threshold JSON is missing deadzone_action")
    axes = ("swing", "boom", "stick", "bucket")
    positive = np.asarray([float(deadzone[axis]["pos"]) for axis in axes], np.float32)
    negative = np.asarray([float(deadzone[axis]["neg"]) for axis in axes], np.float32)
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("qvel-zero state-hold thresholds must be finite")
    if np.any(positive <= 0.0) or np.any(negative <= 0.0):
        raise ValueError("qvel-zero state-hold thresholds must be positive")
    return {
        "enabled": True,
        "weight": _nonnegative(cfg.get("weight", 1.0), name="weight"),
        "window_steps": _positive_integer(
            cfg.get("window_steps", 20), name="window_steps"
        ),
        "margin": _nonnegative(cfg.get("margin", 0.0), name="margin"),
        "positive": positive,
        "negative": negative,
        "threshold_path": str(threshold_path.resolve()),
    }


def qvel_zero_state_hold_loss_terms(
    *,
    policy_direct: torch.Tensor,
    transition_mask: torch.Tensor | None,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    zero = policy_direct.new_zeros(())
    result = {
        "qvel_zero_state_hold_pos_shortfall_loss": zero,
        "qvel_zero_state_hold_neg_shortfall_loss": zero,
        "qvel_zero_state_hold_unweighted_loss": zero,
        "qvel_zero_state_hold_loss": zero,
        "qvel_zero_state_hold_valid_count": zero,
        "qvel_zero_state_hold_direction_hit_rate": zero,
    }
    if not bool(config.get("enabled", False)):
        return result
    if policy_direct.ndim != 3 or policy_direct.shape[2] != 4:
        raise ValueError("qvel-zero state-hold policy chunk must have shape (B, Q, 4)")
    if transition_mask is None:
        raise ValueError("qvel-zero state-hold loss requires transition_mask")
    mask = transition_mask.to(device=policy_direct.device, dtype=torch.bool)
    if tuple(mask.shape) != (policy_direct.shape[0], 4, 2):
        raise ValueError("qvel-zero state-hold transition_mask must have shape (B,4,2)")
    window = min(int(config["window_steps"]), int(policy_direct.shape[1]))
    action = policy_direct[:, :window]
    positive = torch.as_tensor(
        np.asarray(config["positive"]), dtype=action.dtype, device=action.device
    ) + float(config["margin"])
    negative = torch.as_tensor(
        np.asarray(config["negative"]), dtype=action.dtype, device=action.device
    ) + float(config["margin"])
    max_action = action.max(dim=1).values
    min_action = action.min(dim=1).values
    pos_mask = mask[:, :, 0]
    neg_mask = mask[:, :, 1]
    pos_shortfall_values = torch.relu(positive.reshape(1, 4) - max_action)
    neg_shortfall_values = torch.relu(negative.reshape(1, 4) + min_action)
    pos_shortfall = _masked_mean(pos_shortfall_values, pos_mask)
    neg_shortfall = _masked_mean(neg_shortfall_values, neg_mask)
    unweighted = pos_shortfall + neg_shortfall
    hits = (max_action >= positive.reshape(1, 4)) & pos_mask
    hits |= (min_action <= -negative.reshape(1, 4)) & neg_mask
    valid_count = mask.sum().to(dtype=action.dtype)
    return {
        "qvel_zero_state_hold_pos_shortfall_loss": pos_shortfall,
        "qvel_zero_state_hold_neg_shortfall_loss": neg_shortfall,
        "qvel_zero_state_hold_unweighted_loss": unweighted,
        "qvel_zero_state_hold_loss": float(config["weight"]) * unweighted,
        "qvel_zero_state_hold_valid_count": valid_count,
        "qvel_zero_state_hold_direction_hit_rate": (
            action.new_zeros(())
            if not bool(mask.any())
            else hits.sum().to(dtype=action.dtype) / valid_count.clamp_min(1.0)
        ),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    return values.new_zeros(()) if selected.numel() == 0 else selected.mean()


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"qvel_zero_state_hold_loss.{name} must be a boolean")
    return bool(value)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"qvel_zero_state_hold_loss.{name} must be a positive integer")
    result = int(value)
    if result <= 0 or float(value) != float(result):
        raise ValueError(f"qvel_zero_state_hold_loss.{name} must be a positive integer")
    return result


def _nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"qvel_zero_state_hold_loss.{name} must be non-negative")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"qvel_zero_state_hold_loss.{name} must be non-negative")
    return result
