"""Minimal direct-action guard for the uncommitted task-state-v2 interval."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from testbed.policies.deadzone_eval import load_deadzone_thresholds


def resolve_task_state_v2_adherence_config(raw: Any) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("task_state_v2_adherence_loss config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "weight": 0.0,
            "action_window_steps": 1,
            "negative_deadzone": 1.0,
            "guard_margin": 0.0,
            "reduction": "mean_all",
        }
    threshold_raw = cfg.get("threshold_json")
    if threshold_raw is None or not str(threshold_raw).strip():
        raise ValueError(
            "task_state_v2_adherence_loss.threshold_json is required"
        )
    threshold_path = Path(str(threshold_raw)).resolve()
    if not threshold_path.is_file():
        raise FileNotFoundError(
            "task_state_v2_adherence_loss threshold_json does not exist: "
            f"{threshold_path}"
        )
    thresholds = load_deadzone_thresholds(threshold_path)
    margin = _nonnegative_float(cfg.get("guard_margin", 0.0), name="guard_margin")
    negative = float(thresholds["swing"]["neg"])
    if margin >= negative:
        raise ValueError(
            "task_state_v2_adherence_loss.guard_margin must be below the "
            "negative swing deadzone"
        )
    reduction = str(cfg.get("reduction", "mean_all"))
    if reduction not in {"mean_all", "worst_query"}:
        raise ValueError(
            "task_state_v2_adherence_loss.reduction must be mean_all or "
            "worst_query"
        )
    return {
        "enabled": True,
        "weight": _positive_float(cfg.get("weight", 1.0), name="weight"),
        "action_window_steps": _positive_integer(
            cfg.get("action_window_steps", 20), name="action_window_steps"
        ),
        "negative_deadzone": negative,
        "guard_margin": margin,
        "reduction": reduction,
        "threshold_path": str(threshold_path),
    }


def task_state_v2_adherence_loss_terms(
    *,
    policy_direct: torch.Tensor,
    valid_mask: torch.Tensor,
    uncommitted: torch.Tensor | None,
    config: Mapping[str, Any] | None,
) -> dict[str, torch.Tensor]:
    cfg = (
        dict(config)
        if isinstance(config, Mapping) and "negative_deadzone" in config
        else resolve_task_state_v2_adherence_config(config)
    )
    zero = policy_direct.new_zeros(())
    result = {
        "task_state_v2_uncommitted_guard_loss": zero,
        "task_state_v2_adherence_loss": zero,
        "task_state_v2_uncommitted_valid_count": zero,
        "task_state_v2_uncommitted_no_negative_rate": zero,
    }
    if not bool(cfg["enabled"]):
        return result
    if policy_direct.ndim != 3 or policy_direct.shape[-1] < 1:
        raise ValueError("task_state_v2 policy action must have shape (B, Q, A)")
    if valid_mask.shape[:2] != policy_direct.shape[:2]:
        raise ValueError("task_state_v2 valid mask must align with policy chunks")
    if uncommitted is None:
        raise ValueError(
            "task_state_v2_adherence_loss requires uncommitted batch labels"
        )
    labels = uncommitted.to(
        device=policy_direct.device, dtype=torch.bool
    ).reshape(-1)
    if labels.numel() != policy_direct.shape[0]:
        raise ValueError("task_state_v2 labels must contain one value per row")
    window = min(int(cfg["action_window_steps"]), int(policy_direct.shape[1]))
    valid = valid_mask[:, :window]
    if valid.ndim == 3:
        valid = valid[..., 0]
    valid = valid.to(device=policy_direct.device, dtype=torch.bool)
    eligible = valid & labels.reshape(-1, 1)
    count = eligible.sum()
    result["task_state_v2_uncommitted_valid_count"] = count.to(
        dtype=policy_direct.dtype
    )
    if not bool(count):
        return result
    swing = policy_direct[:, :window, 0]
    floor = -float(cfg["negative_deadzone"]) + float(cfg["guard_margin"])
    violations = torch.relu(floor - swing)
    if cfg.get("reduction", "mean_all") == "worst_query":
        row_has_valid = eligible.any(dim=1)
        row_worst = torch.where(
            eligible,
            violations,
            torch.zeros_like(violations),
        ).amax(dim=1)
        guard = row_worst[row_has_valid].mean()
    else:
        guard = violations[eligible].mean()
    no_negative = (swing > -float(cfg["negative_deadzone"]))[eligible]
    result.update(
        {
            "task_state_v2_uncommitted_guard_loss": guard,
            "task_state_v2_adherence_loss": float(cfg["weight"]) * guard,
            "task_state_v2_uncommitted_no_negative_rate": no_negative.to(
                dtype=torch.float32
            ).mean(),
        }
    )
    return result


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"task_state_v2_adherence_loss.{name} must be boolean")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0 or float(value) != float(parsed):
        raise ValueError(
            f"task_state_v2_adherence_loss.{name} must be a positive integer"
        )
    return parsed


def _positive_float(value: Any, *, name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(
            f"task_state_v2_adherence_loss.{name} must be positive"
        )
    return parsed


def _nonnegative_float(value: Any, *, name: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise ValueError(
            f"task_state_v2_adherence_loss.{name} must be non-negative"
        )
    return parsed
