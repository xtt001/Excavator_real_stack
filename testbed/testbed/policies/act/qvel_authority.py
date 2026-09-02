"""Direct-action authority for stable versus negative-moving swing qvel."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

AXES = ("swing", "boom", "stick", "bucket")


def resolve_qvel_authority_config(raw: Any) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("qvel_authority_loss config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "scope": "train_only",
            "weight": 0.0,
            "append_samples_per_episode": 0,
            "action_window_steps": 1,
            "stable_swing_qvel_abs_max": 0.015,
            "moving_swing_qvel_max": -0.05,
            "counterfactual_moving_swing_qvel": -0.266,
            "contrast_margin": 0.1,
            "stable_tool_weight": 0.0,
            "stable_guard_weight": 0.0,
            "moving_return_weight": 0.0,
            "contrast_weight": 0.0,
            "positive_deadzone_by_axis": np.ones(4, dtype=np.float32),
            "negative_deadzone_by_axis": np.ones(4, dtype=np.float32),
            "manifest_path": None,
            "threshold_path": None,
        }
    scope = str(cfg.get("scope", "train_only"))
    if scope not in {"train_only", "train_and_validation"}:
        raise ValueError(
            "qvel_authority_loss.scope must be train_only or train_and_validation"
        )
    threshold_raw = cfg.get("threshold_json")
    if threshold_raw is None or not str(threshold_raw).strip():
        raise ValueError("qvel_authority_loss.threshold_json is required")
    threshold_path = Path(str(threshold_raw)).resolve()
    if not threshold_path.is_file():
        raise FileNotFoundError(threshold_path)
    threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    deadzone = _mapping(threshold_payload.get("deadzone_action"), name="deadzone")
    positive = np.asarray(
        [
            _positive(
                _mapping(deadzone.get(axis), name=axis).get("pos"), name=f"{axis}.pos"
            )
            for axis in AXES
        ],
        dtype=np.float32,
    )
    negative = np.asarray(
        [
            _positive(
                _mapping(deadzone.get(axis), name=axis).get("neg"), name=f"{axis}.neg"
            )
            for axis in AXES
        ],
        dtype=np.float32,
    )
    manifest_raw = cfg.get("manifest_path")
    if manifest_raw is None or not str(manifest_raw).strip():
        raise ValueError("qvel_authority_loss.manifest_path is required")
    manifest_path = Path(str(manifest_raw)).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    append = _positive_integer(
        cfg.get("append_samples_per_episode", 2), name="append_samples_per_episode"
    )
    if append != 2:
        raise ValueError("qvel_authority_loss.append_samples_per_episode must be 2")
    moving = _negative(
        cfg.get("moving_swing_qvel_max", -0.05), name="moving_swing_qvel_max"
    )
    counterfactual = _negative(
        cfg.get("counterfactual_moving_swing_qvel", -0.266),
        name="counterfactual_moving_swing_qvel",
    )
    if counterfactual > moving:
        raise ValueError(
            "counterfactual moving qvel must be at least as negative as moving threshold"
        )
    return {
        "enabled": True,
        "scope": scope,
        "weight": _nonnegative(cfg.get("weight", 1.0), name="weight"),
        "append_samples_per_episode": append,
        "action_window_steps": _positive_integer(
            cfg.get("action_window_steps", 5), name="action_window_steps"
        ),
        "stable_swing_qvel_abs_max": _positive(
            cfg.get("stable_swing_qvel_abs_max", 0.015),
            name="stable_swing_qvel_abs_max",
        ),
        "moving_swing_qvel_max": moving,
        "counterfactual_moving_swing_qvel": counterfactual,
        "contrast_margin": _positive(
            cfg.get("contrast_margin", 0.1), name="contrast_margin"
        ),
        "stable_tool_weight": _nonnegative(
            cfg.get("stable_tool_weight", 1.0), name="stable_tool_weight"
        ),
        "stable_guard_weight": _nonnegative(
            cfg.get("stable_guard_weight", 1.0), name="stable_guard_weight"
        ),
        "moving_return_weight": _nonnegative(
            cfg.get("moving_return_weight", 1.0), name="moving_return_weight"
        ),
        "contrast_weight": _nonnegative(
            cfg.get("contrast_weight", 1.0), name="contrast_weight"
        ),
        "positive_deadzone_by_axis": positive,
        "negative_deadzone_by_axis": negative,
        "manifest_path": str(manifest_path),
        "threshold_path": str(threshold_path),
    }


def qvel_authority_candidate_starts(
    *,
    qvel: np.ndarray,
    valid_starts: np.ndarray,
    segments: Mapping[str, Sequence[Mapping[str, Any]]],
    chunk_steps: int,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not bool(config.get("enabled", False)):
        empty = np.zeros(0, dtype=np.int64)
        return {"stable_tool": empty, "moving_return": empty}
    velocity = np.asarray(qvel, dtype=np.float32)
    if velocity.ndim != 2 or velocity.shape[1] != 4:
        raise ValueError("qvel authority candidates require qvel shaped (T, 4)")
    allowed = set(int(value) for value in np.asarray(valid_starts).reshape(-1))
    window = int(chunk_steps)

    def starts_for(name: str) -> np.ndarray:
        values = []
        for segment in segments.get(name, ()):
            first = int(segment["start"])
            last = int(segment["end"]) - window + 1
            if last >= first:
                values.extend(range(first, last + 1))
        return np.asarray(
            [value for value in values if value in allowed], dtype=np.int64
        )

    stable = starts_for("tool_pre")
    stable = stable[
        np.abs(velocity[stable, 0]) <= float(config["stable_swing_qvel_abs_max"])
    ]
    moving = starts_for("swing_return")
    moving = moving[velocity[moving, 0] <= float(config["moving_swing_qvel_max"])]
    return {"stable_tool": stable, "moving_return": moving}


def stable_tool_direction_mask(
    action: np.ndarray, *, config: Mapping[str, Any]
) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.shape != (4,):
        raise ValueError("stable tool action must contain four axes")
    positive = np.asarray(config["positive_deadzone_by_axis"], dtype=np.float32)
    negative = np.asarray(config["negative_deadzone_by_axis"], dtype=np.float32)
    mask = np.stack((value >= positive, value <= -negative), axis=-1)
    mask[0] = False
    if not bool(mask[1:].any()):
        raise ValueError("stable tool anchor lacks a mechanically effective tool axis")
    return mask


def qvel_authority_loss_terms(
    *,
    primary_direct: torch.Tensor,
    counterfactual_direct: torch.Tensor,
    moving_primary: torch.Tensor | None,
    stable_tool_mask: torch.Tensor | None,
    valid: torch.Tensor | None,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    zero = primary_direct.new_zeros(())
    result = {
        "qvel_authority_stable_tool_loss": zero,
        "qvel_authority_stable_guard_loss": zero,
        "qvel_authority_moving_return_loss": zero,
        "qvel_authority_contrast_loss": zero,
        "qvel_authority_unweighted_loss": zero,
        "qvel_authority_loss": zero,
        "qvel_authority_valid_count": zero,
        "qvel_authority_stable_tool_rate": zero,
        "qvel_authority_stable_no_negative_rate": zero,
        "qvel_authority_moving_negative_rate": zero,
        "qvel_authority_pair_rate": zero,
    }
    if not bool(config.get("enabled", False)):
        return result
    if primary_direct.shape != counterfactual_direct.shape or primary_direct.ndim != 3:
        raise ValueError("qvel authority chunks must have matching shape (B, Q, A)")
    if moving_primary is None or stable_tool_mask is None or valid is None:
        raise ValueError("qvel authority loss requires labels and valid mask")
    valid_mask = valid.to(primary_direct.device, dtype=torch.bool).reshape(-1)
    moving_all = moving_primary.to(primary_direct.device, dtype=torch.bool).reshape(-1)
    tool_all = stable_tool_mask.to(primary_direct.device, dtype=torch.bool)
    if valid_mask.numel() != primary_direct.shape[0] or moving_all.numel() != len(
        valid_mask
    ):
        raise ValueError("qvel authority labels must match batch rows")
    if tool_all.shape != (primary_direct.shape[0], 4, 2):
        raise ValueError("stable tool mask must have shape (B, 4, 2)")
    result["qvel_authority_valid_count"] = valid_mask.sum().to(primary_direct.dtype)
    if not bool(valid_mask.any()):
        return result
    primary = primary_direct[valid_mask]
    counter = counterfactual_direct[valid_mask]
    moving = moving_all[valid_mask]
    tool_mask = tool_all[valid_mask]
    moving_selector = moving.reshape(-1, 1, 1)
    stable_action = torch.where(moving_selector, counter, primary)
    moving_action = torch.where(moving_selector, primary, counter)
    stable_rows = ~moving
    window = int(config["action_window_steps"])
    negative = torch.as_tensor(
        config["negative_deadzone_by_axis"],
        device=primary.device,
        dtype=primary.dtype,
    )
    positive = torch.as_tensor(
        config["positive_deadzone_by_axis"],
        device=primary.device,
        dtype=primary.dtype,
    )
    stable_swing = stable_action[:, :window, 0]
    moving_swing = moving_action[:, :window, 0]
    stable_guard = torch.relu(-negative[0] - stable_swing).mean()
    moving_return = torch.relu(negative[0] + moving_swing).mean()

    signed = torch.stack(
        (stable_action[:, :window], -stable_action[:, :window]), dim=-1
    )
    thresholds = torch.stack((positive, negative), dim=-1)
    margin = signed - thresholds.reshape(1, 1, 4, 2)
    masked_margin = torch.where(
        tool_mask[:, None], margin, torch.full_like(margin, -1e6)
    )
    best_tool_margin = masked_margin.amax(dim=(1, 2, 3))
    stable_tool = (
        zero
        if not bool(stable_rows.any())
        else torch.relu(-best_tool_margin[stable_rows]).mean()
    )
    contrast = torch.relu(
        float(config["contrast_margin"]) - (stable_swing - moving_swing)
    ).mean()
    unweighted = (
        float(config["stable_tool_weight"]) * stable_tool
        + float(config["stable_guard_weight"]) * stable_guard
        + float(config["moving_return_weight"]) * moving_return
        + float(config["contrast_weight"]) * contrast
    )
    stable_tool_hit = best_tool_margin >= 0.0
    stable_no_negative = (stable_swing > -negative[0]).all(dim=1)
    moving_negative = (moving_swing <= -negative[0]).any(dim=1)
    pair_hit = stable_no_negative & moving_negative
    return {
        "qvel_authority_stable_tool_loss": stable_tool,
        "qvel_authority_stable_guard_loss": stable_guard,
        "qvel_authority_moving_return_loss": moving_return,
        "qvel_authority_contrast_loss": contrast,
        "qvel_authority_unweighted_loss": unweighted,
        "qvel_authority_loss": float(config["weight"]) * unweighted,
        "qvel_authority_valid_count": result["qvel_authority_valid_count"],
        "qvel_authority_stable_tool_rate": _masked_rate(stable_tool_hit, stable_rows),
        "qvel_authority_stable_no_negative_rate": stable_no_negative.float().mean(),
        "qvel_authority_moving_negative_rate": moving_negative.float().mean(),
        "qvel_authority_pair_rate": pair_hit.float().mean(),
    }


def _masked_rate(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = value[mask]
    return (
        value.new_zeros((), dtype=torch.float32)
        if selected.numel() == 0
        else selected.float().mean()
    )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"qvel_authority_loss.{name} must be a mapping")
    return value


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"qvel_authority_loss.{name} must be boolean")
    return value


def _positive(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"qvel_authority_loss.{name} must be positive")
    return number


def _negative(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number >= 0.0:
        raise ValueError(f"qvel_authority_loss.{name} must be negative")
    return number


def _nonnegative(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"qvel_authority_loss.{name} must be non-negative")
    return number


def _positive_integer(value: Any, *, name: str) -> int:
    number = int(value)
    if number <= 0 or float(value) != float(number):
        raise ValueError(f"qvel_authority_loss.{name} must be a positive integer")
    return number
