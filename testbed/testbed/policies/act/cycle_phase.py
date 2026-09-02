"""Direct action supervision for the causal Real Transition cycle phase."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from testbed.tasks.real_transition_phase import CYCLE_PHASE_KEY


def resolve_cycle_phase_loss_config(raw: Any) -> dict[str, Any]:
    """Validate the phase sampler and direct-action loss configuration."""

    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("cycle_phase_loss config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "scope": "train_only",
            "condition_key": CYCLE_PHASE_KEY,
            "weight": 0.0,
            "pre_positive_weight": 0.0,
            "return_preservation_weight": 0.0,
            "pre_return_guard_weight": 0.0,
            "contrast_weight": 0.0,
            "contrast_margin_scale": 1.0,
            "guard_margin": 0.0,
            "append_samples_per_episode": 0,
            "action_window_steps": 1,
            "axis_index": 0,
            "positive_deadzone": 0.0,
            "negative_deadzone": 0.0,
            "threshold_path": None,
        }
    if cfg.get("condition_key") != CYCLE_PHASE_KEY:
        raise ValueError(f"cycle_phase_loss.condition_key must be {CYCLE_PHASE_KEY!r}")
    scope = str(cfg.get("scope", "train_only"))
    if scope not in {"train_only", "train_and_validation"}:
        raise ValueError(
            "cycle_phase_loss.scope must be 'train_only' or 'train_and_validation'"
        )
    threshold_raw = cfg.get("threshold_json")
    if threshold_raw is None or not str(threshold_raw).strip():
        raise ValueError("cycle_phase_loss.threshold_json is required")
    threshold_path = Path(str(threshold_raw))
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"cycle_phase_loss threshold_json does not exist: {threshold_path}"
        )
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    deadzone = _mapping(payload.get("deadzone_action"), name="deadzone_action")
    swing = _mapping(deadzone.get("swing"), name="deadzone_action.swing")
    positive = _positive(swing.get("pos"), name="deadzone_action.swing.pos")
    negative = _positive(swing.get("neg"), name="deadzone_action.swing.neg")
    return {
        "enabled": True,
        "scope": scope,
        "condition_key": CYCLE_PHASE_KEY,
        "weight": _nonnegative(cfg.get("weight", 1.0), name="weight"),
        "pre_positive_weight": _nonnegative(
            cfg.get("pre_positive_weight", 1.0), name="pre_positive_weight"
        ),
        "return_preservation_weight": _nonnegative(
            cfg.get("return_preservation_weight", 1.0),
            name="return_preservation_weight",
        ),
        "pre_return_guard_weight": _nonnegative(
            cfg.get("pre_return_guard_weight", 1.0),
            name="pre_return_guard_weight",
        ),
        "contrast_weight": _nonnegative(
            cfg.get("contrast_weight", 1.0), name="contrast_weight"
        ),
        "contrast_margin_scale": _nonnegative(
            cfg.get("contrast_margin_scale", 1.0),
            name="contrast_margin_scale",
        ),
        "guard_margin": _nonnegative(
            cfg.get("guard_margin", 0.0), name="guard_margin"
        ),
        "append_samples_per_episode": _positive_integer(
            cfg.get("append_samples_per_episode", 1),
            name="append_samples_per_episode",
        ),
        "action_window_steps": _positive_integer(
            cfg.get("action_window_steps", 5), name="action_window_steps"
        ),
        "axis_index": 0,
        "positive_deadzone": positive,
        "negative_deadzone": negative,
        "threshold_path": str(threshold_path.resolve()),
    }


def cycle_phase_candidate_indices(
    *,
    actions: np.ndarray,
    phase: np.ndarray,
    valid_starts: np.ndarray,
    phase_valid_mask: np.ndarray | None,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Find executable positive pre-return and negative return anchors."""

    empty = {
        "pre_positive": np.zeros(0, dtype=np.int64),
        "return_negative": np.zeros(0, dtype=np.int64),
    }
    if not bool(config.get("enabled", False)):
        return empty
    action_array = np.asarray(actions, dtype=np.float32)
    phase_array = np.asarray(phase, dtype=np.float32).reshape(-1)
    if action_array.ndim != 2 or action_array.shape[0] != phase_array.size:
        raise ValueError("cycle phase actions/phase must have matching time length")
    if not np.all(np.isin(phase_array, [0.0, 1.0])):
        raise ValueError("cycle phase values must be 0 or 1")
    valid_phase = None
    if phase_valid_mask is not None:
        valid_phase = np.asarray(phase_valid_mask, dtype=bool)
        if valid_phase.ndim != 2 or valid_phase.shape[0] != len(phase_array):
            raise ValueError("cycle_phase_valid_mask must have shape (T, chunk_steps)")
    window = int(config["action_window_steps"])
    axis = int(config["axis_index"])
    positive = float(config["positive_deadzone"])
    negative = float(config["negative_deadzone"])
    pre: list[int] = []
    returned: list[int] = []
    for raw_start in np.asarray(valid_starts, dtype=np.int64).reshape(-1):
        start = int(raw_start)
        if start < 0 or start + window > len(phase_array):
            continue
        if valid_phase is not None and (
            valid_phase.shape[1] < window
            or not bool(valid_phase[start, :window].all())
        ):
            continue
        values = action_array[start : start + window, axis]
        if phase_array[start] == 0.0 and bool(np.all(values >= positive)):
            pre.append(start)
        elif phase_array[start] == 1.0 and bool(np.all(values <= -negative)):
            returned.append(start)
    return {
        "pre_positive": np.asarray(pre, dtype=np.int64),
        "return_negative": np.asarray(returned, dtype=np.int64),
    }


def cycle_phase_loss_terms(
    *,
    primary_direct: torch.Tensor,
    counterfactual_direct: torch.Tensor,
    return_primary: torch.Tensor | None,
    valid: torch.Tensor | None,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Penalise a return-like direct-to-target action while phase is pre-return."""

    zero = primary_direct.new_zeros(())
    result = {
        "cycle_phase_pre_positive_shortfall_loss": zero,
        "cycle_phase_return_preservation_loss": zero,
        "cycle_phase_pre_return_guard_loss": zero,
        "cycle_phase_contrast_loss": zero,
        "cycle_phase_unweighted_loss": zero,
        "cycle_phase_loss": zero,
        "cycle_phase_valid_count": zero,
        "cycle_phase_pre_positive_rate": zero,
        "cycle_phase_return_negative_rate": zero,
        "cycle_phase_pre_return_no_shortcut_rate": zero,
        "cycle_phase_return_pair_hit_rate": zero,
    }
    if not bool(config.get("enabled", False)):
        return result
    if tuple(primary_direct.shape) != tuple(counterfactual_direct.shape):
        raise ValueError("cycle phase primary/counterfactual chunks must match")
    if primary_direct.ndim != 3:
        raise ValueError("cycle phase action chunks must have shape (B, Q, A)")
    if return_primary is None or valid is None:
        raise ValueError("cycle phase loss requires return_primary and valid labels")
    valid_mask = valid.to(device=primary_direct.device, dtype=torch.bool).reshape(-1)
    return_mask_all = return_primary.to(
        device=primary_direct.device, dtype=torch.bool
    ).reshape(-1)
    if valid_mask.numel() != primary_direct.shape[0] or return_mask_all.numel() != primary_direct.shape[0]:
        raise ValueError("cycle phase labels must have one value per batch row")
    result["cycle_phase_valid_count"] = valid_mask.sum().to(
        dtype=primary_direct.dtype
    )
    if not bool(valid_mask.any()):
        return result

    axis = int(config["axis_index"])
    window = int(config["action_window_steps"])
    primary = primary_direct[valid_mask, :window, axis]
    counterfactual = counterfactual_direct[valid_mask, :window, axis]
    return_primary_mask = return_mask_all[valid_mask].reshape(-1, 1)
    pre_action = torch.where(return_primary_mask, counterfactual, primary)
    return_action = torch.where(return_primary_mask, primary, counterfactual)
    pre_rows = ~return_primary_mask.reshape(-1)
    return_rows = return_primary_mask.reshape(-1)
    positive = float(config["positive_deadzone"])
    negative = float(config["negative_deadzone"])
    guard_floor = -negative + float(config["guard_margin"])
    contrast_margin = negative * float(config["contrast_margin_scale"])

    pre_positive = _masked_mean(torch.relu(positive - pre_action), pre_rows)
    return_preservation = _masked_mean(
        torch.relu(negative + return_action), return_rows
    )
    pre_guard = _masked_mean(
        torch.relu(guard_floor - pre_action), return_rows
    )
    contrast = _masked_mean(
        torch.relu(contrast_margin - (pre_action - return_action)), return_rows
    )
    unweighted = (
        float(config["pre_positive_weight"]) * pre_positive
        + float(config["return_preservation_weight"]) * return_preservation
        + float(config["pre_return_guard_weight"]) * pre_guard
        + float(config["contrast_weight"]) * contrast
    )
    pre_positive_hit = pre_action >= positive
    return_negative_hit = return_action <= -negative
    pre_no_shortcut = pre_action > -negative
    return_pair_hit = return_negative_hit & pre_no_shortcut
    return {
        "cycle_phase_pre_positive_shortfall_loss": pre_positive,
        "cycle_phase_return_preservation_loss": return_preservation,
        "cycle_phase_pre_return_guard_loss": pre_guard,
        "cycle_phase_contrast_loss": contrast,
        "cycle_phase_unweighted_loss": unweighted,
        "cycle_phase_loss": float(config["weight"]) * unweighted,
        "cycle_phase_valid_count": result["cycle_phase_valid_count"],
        "cycle_phase_pre_positive_rate": _masked_mean(
            pre_positive_hit.to(primary.dtype), pre_rows
        ),
        "cycle_phase_return_negative_rate": _masked_mean(
            return_negative_hit.to(primary.dtype), return_rows
        ),
        "cycle_phase_pre_return_no_shortcut_rate": _masked_mean(
            pre_no_shortcut.to(primary.dtype), return_rows
        ),
        "cycle_phase_return_pair_hit_rate": _masked_mean(
            return_pair_hit.to(primary.dtype), return_rows
        ),
    }


def _masked_mean(values: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    selected = values[row_mask]
    return values.new_zeros(()) if selected.numel() == 0 else selected.mean()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"cycle phase {name} must be a mapping")
    return value


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"cycle_phase_loss.{name} must be a boolean")
    return bool(value)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"cycle_phase_loss.{name} must be a positive integer")
    result = int(value)
    if result <= 0 or float(value) != float(result):
        raise ValueError(f"cycle_phase_loss.{name} must be a positive integer")
    return result


def _nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"cycle_phase_loss.{name} must be non-negative")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"cycle_phase_loss.{name} must be non-negative")
    return result


def _positive(value: Any, *, name: str) -> float:
    result = _nonnegative(value, name=name)
    if result <= 0.0:
        raise ValueError(f"cycle_phase_loss.{name} must be positive")
    return result
