"""Direct action supervision for the causal positive-excursion latch."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from testbed.tasks.real_transition_excursion import EXCURSION_OBSERVED_KEY


def resolve_excursion_observed_loss_config(raw: Any) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("excursion_observed_loss config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "scope": "train_only",
            "condition_key": EXCURSION_OBSERVED_KEY,
            "weight": 0.0,
            "pre_positive_weight": 0.0,
            "post_negative_weight": 0.0,
            "pre_guard_weight": 0.0,
            "contrast_weight": 0.0,
            "contrast_margin_scale": 1.0,
            "guard_margin": 0.0,
            "append_samples_per_episode": 0,
            "action_window_steps": 1,
            "axis_index": 0,
            "positive_deadzone": 0.0,
            "negative_deadzone": 0.0,
            "qvel_stable_abs_max_rad_s": np.zeros(4, dtype=np.float32),
            "threshold_path": None,
        }
    if cfg.get("condition_key") != EXCURSION_OBSERVED_KEY:
        raise ValueError(
            "excursion_observed_loss.condition_key must be "
            f"{EXCURSION_OBSERVED_KEY!r}"
        )
    scope = str(cfg.get("scope", "train_only"))
    if scope not in {"train_only", "train_and_validation"}:
        raise ValueError(
            "excursion_observed_loss.scope must be train_only or "
            "train_and_validation"
        )
    threshold_raw = cfg.get("threshold_json")
    if threshold_raw is None or not str(threshold_raw).strip():
        raise ValueError("excursion_observed_loss.threshold_json is required")
    threshold_path = Path(str(threshold_raw))
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"excursion observed threshold_json does not exist: {threshold_path}"
        )
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    deadzone = _mapping(payload.get("deadzone_action"), name="deadzone_action")
    swing = _mapping(deadzone.get("swing"), name="deadzone_action.swing")
    qvel_limits = np.asarray(
        cfg.get("qvel_stable_abs_max_rad_s", [0.015, 0.015, 0.020, 0.020]),
        dtype=np.float32,
    )
    if qvel_limits.shape != (4,) or not np.isfinite(qvel_limits).all() or np.any(
        qvel_limits <= 0.0
    ):
        raise ValueError(
            "excursion_observed_loss.qvel_stable_abs_max_rad_s must contain "
            "four positive finite values"
        )
    return {
        "enabled": True,
        "scope": scope,
        "condition_key": EXCURSION_OBSERVED_KEY,
        "weight": _nonnegative(cfg.get("weight", 1.0), name="weight"),
        "pre_positive_weight": _nonnegative(
            cfg.get("pre_positive_weight", 1.0), name="pre_positive_weight"
        ),
        "post_negative_weight": _nonnegative(
            cfg.get("post_negative_weight", 1.0), name="post_negative_weight"
        ),
        "pre_guard_weight": _nonnegative(
            cfg.get("pre_guard_weight", 1.0), name="pre_guard_weight"
        ),
        "contrast_weight": _nonnegative(
            cfg.get("contrast_weight", 1.0), name="contrast_weight"
        ),
        "contrast_margin_scale": _nonnegative(
            cfg.get("contrast_margin_scale", 1.0), name="contrast_margin_scale"
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
        "positive_deadzone": _positive(
            swing.get("pos"), name="deadzone_action.swing.pos"
        ),
        "negative_deadzone": _positive(
            swing.get("neg"), name="deadzone_action.swing.neg"
        ),
        "qvel_stable_abs_max_rad_s": qvel_limits,
        "threshold_path": str(threshold_path.resolve()),
    }


def excursion_observed_candidate_indices(
    *,
    actions: np.ndarray,
    qvel: np.ndarray,
    excursion_observed: np.ndarray,
    return_phase: np.ndarray,
    valid_starts: np.ndarray,
    excursion_valid_mask: np.ndarray | None,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    empty = {
        "pre_positive": np.zeros(0, dtype=np.int64),
        "apex_negative": np.zeros(0, dtype=np.int64),
        "moving_negative": np.zeros(0, dtype=np.int64),
    }
    if not bool(config.get("enabled", False)):
        return empty
    action_array = np.asarray(actions, dtype=np.float32)
    qvel_array = np.asarray(qvel, dtype=np.float32)
    state = np.asarray(excursion_observed, dtype=np.float32).reshape(-1)
    phase = np.asarray(return_phase, dtype=np.float32).reshape(-1)
    if (
        action_array.ndim != 2
        or qvel_array.shape != action_array.shape
        or len(state) != len(action_array)
        or len(phase) != len(action_array)
    ):
        raise ValueError("excursion observed candidates require aligned action/qvel/state")
    if not np.all(np.isin(state, [0.0, 1.0])) or not np.all(
        np.isin(phase, [0.0, 1.0])
    ):
        raise ValueError("excursion observed and return phase must contain 0/1")
    valid_state = None
    if excursion_valid_mask is not None:
        valid_state = np.asarray(excursion_valid_mask, dtype=bool)
        if valid_state.ndim != 2 or valid_state.shape[0] != len(state):
            raise ValueError(
                "excursion_observed_valid_mask must have shape (T, chunk_steps)"
            )
    window = int(config["action_window_steps"])
    axis = int(config["axis_index"])
    positive = float(config["positive_deadzone"])
    negative = float(config["negative_deadzone"])
    qvel_limits = np.asarray(config["qvel_stable_abs_max_rad_s"], dtype=np.float32)
    pre: list[int] = []
    apex: list[int] = []
    moving: list[int] = []
    for raw_start in np.asarray(valid_starts, dtype=np.int64).reshape(-1):
        start = int(raw_start)
        if start < 0 or start + window > len(state):
            continue
        if valid_state is not None and (
            valid_state.shape[1] < window
            or not bool(valid_state[start, :window].all())
        ):
            continue
        values = action_array[start : start + window, axis]
        if state[start] == 0.0 and bool(np.all(values >= positive)):
            pre.append(start)
            continue
        if state[start] != 1.0 or not bool(np.all(values <= -negative)):
            continue
        if phase[start] == 0.0 and bool(
            np.all(np.abs(qvel_array[start]) <= qvel_limits)
        ):
            apex.append(start)
        elif phase[start] == 1.0:
            moving.append(start)
    return {
        "pre_positive": np.asarray(pre, dtype=np.int64),
        "apex_negative": np.asarray(apex, dtype=np.int64),
        "moving_negative": np.asarray(moving, dtype=np.int64),
    }


def excursion_observed_loss_terms(
    *,
    primary_direct: torch.Tensor,
    counterfactual_direct: torch.Tensor,
    post_excursion_primary: torch.Tensor | None,
    valid: torch.Tensor | None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    cfg = (
        dict(config)
        if isinstance(config, Mapping) and "positive_deadzone" in config
        else resolve_excursion_observed_loss_config(config)
    )
    zero = primary_direct.new_zeros(())
    result = {
        "excursion_pre_positive_shortfall_loss": zero,
        "excursion_post_negative_preservation_loss": zero,
        "excursion_pre_guard_loss": zero,
        "excursion_contrast_loss": zero,
        "excursion_unweighted_loss": zero,
        "excursion_observed_loss": zero,
        "excursion_observed_valid_count": zero,
        "excursion_pre_positive_rate": zero,
        "excursion_post_negative_rate": zero,
        "excursion_pre_no_shortcut_rate": zero,
        "excursion_pair_hit_rate": zero,
    }
    if not cfg["enabled"]:
        return result
    if tuple(primary_direct.shape) != tuple(counterfactual_direct.shape):
        raise ValueError("excursion observed primary/counterfactual chunks must match")
    if primary_direct.ndim != 3:
        raise ValueError("excursion observed chunks must have shape (B, Q, A)")
    if post_excursion_primary is None or valid is None:
        raise ValueError("excursion observed loss requires paired labels")
    valid_mask = valid.to(device=primary_direct.device, dtype=torch.bool).reshape(-1)
    post_all = post_excursion_primary.to(
        device=primary_direct.device, dtype=torch.bool
    ).reshape(-1)
    if valid_mask.numel() != primary_direct.shape[0] or post_all.numel() != primary_direct.shape[0]:
        raise ValueError("excursion observed labels must match batch size")
    result["excursion_observed_valid_count"] = valid_mask.sum().to(
        dtype=primary_direct.dtype
    )
    if not bool(valid_mask.any()):
        return result

    axis = int(cfg["axis_index"])
    window = int(cfg["action_window_steps"])
    primary = primary_direct[valid_mask, :window, axis]
    counterfactual = counterfactual_direct[valid_mask, :window, axis]
    post_mask_all = post_all[valid_mask].reshape(-1, 1)
    pre_action = torch.where(post_mask_all, counterfactual, primary)
    post_action = torch.where(post_mask_all, primary, counterfactual)
    pre_rows = ~post_mask_all.reshape(-1)
    post_rows = post_mask_all.reshape(-1)
    positive = float(cfg["positive_deadzone"])
    negative = float(cfg["negative_deadzone"])
    guard_floor = -negative + float(cfg["guard_margin"])
    contrast_margin = negative * float(cfg["contrast_margin_scale"])

    pre_positive = _masked_mean(torch.relu(positive - pre_action), pre_rows)
    post_negative = _masked_mean(torch.relu(negative + post_action), post_rows)
    pre_guard = _masked_mean(torch.relu(guard_floor - pre_action), post_rows)
    contrast = _masked_mean(
        torch.relu(contrast_margin - (pre_action - post_action)), post_rows
    )
    unweighted = (
        float(cfg["pre_positive_weight"]) * pre_positive
        + float(cfg["post_negative_weight"]) * post_negative
        + float(cfg["pre_guard_weight"]) * pre_guard
        + float(cfg["contrast_weight"]) * contrast
    )
    pre_hit = pre_action >= positive
    post_hit = post_action <= -negative
    pre_no_shortcut = pre_action > -negative
    pair_hit = post_hit & pre_no_shortcut
    return {
        "excursion_pre_positive_shortfall_loss": pre_positive,
        "excursion_post_negative_preservation_loss": post_negative,
        "excursion_pre_guard_loss": pre_guard,
        "excursion_contrast_loss": contrast,
        "excursion_unweighted_loss": unweighted,
        "excursion_observed_loss": float(cfg["weight"]) * unweighted,
        "excursion_observed_valid_count": result["excursion_observed_valid_count"],
        "excursion_pre_positive_rate": _masked_mean(
            pre_hit.to(primary.dtype), pre_rows
        ),
        "excursion_post_negative_rate": _masked_mean(
            post_hit.to(primary.dtype), post_rows
        ),
        "excursion_pre_no_shortcut_rate": _masked_mean(
            pre_no_shortcut.to(primary.dtype), post_rows
        ),
        "excursion_pair_hit_rate": _masked_mean(
            pair_hit.to(primary.dtype), post_rows
        ),
    }


def _masked_mean(values: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    selected = values[row_mask]
    return values.new_zeros(()) if selected.numel() == 0 else selected.mean()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"excursion observed {name} must be a mapping")
    return value


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"excursion_observed_loss.{name} must be a boolean")
    return bool(value)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"excursion_observed_loss.{name} must be a positive integer")
    result = int(value)
    if result <= 0 or float(value) != float(result):
        raise ValueError(f"excursion_observed_loss.{name} must be a positive integer")
    return result


def _positive(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"excursion observed {name} must be positive and finite")
    return result


def _nonnegative(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"excursion observed {name} must be non-negative and finite")
    return result
