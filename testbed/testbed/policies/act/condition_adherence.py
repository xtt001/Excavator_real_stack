"""Reward-shaped counterfactual objective for Real Transition conditions.

The objective is deliberately action based.  A flipped goal must not retain
the recorded goal's terminal effective swing command, while the recorded goal
must keep an executable command and a deadzone-scale advantage.  This is an
offline differentiable surrogate, not a claim of online reinforcement learning
or physical outcome reward.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from testbed.data.deadzone_intent_labels import (
    AXIS_NAMES,
    compute_deadzone_intent_labels,
)

CONDITION_KEY = "real_transition_condition_v1"
ANCHOR_RULE = "terminal_effective_swing_transition"


def resolve_condition_action_loss_config(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    weight = float(cfg.get("weight", 1.0))
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("condition_action_loss.weight must be finite and non-negative")
    return {"enabled": enabled, "weight": weight if enabled else 0.0}


def resolve_condition_adherence_config(raw: Any) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("condition_adherence_loss config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "condition_key": CONDITION_KEY,
            "anchor_rule": ANCHOR_RULE,
            "directional_enabled": False,
            "weight": 0.0,
            "recorded_crossing_weight": 0.0,
            "counterfactual_violation_weight": 0.0,
            "contrast_weight": 0.0,
            "counterfactual_ceiling_fraction": 0.5,
            "advantage_margin_scale": 1.0,
            "thresholds": {},
            "pos": torch.zeros(len(AXIS_NAMES), dtype=torch.float32),
            "neg": torch.zeros(len(AXIS_NAMES), dtype=torch.float32),
        }

    if cfg.get("condition_key") != CONDITION_KEY:
        raise ValueError(f"condition_adherence_loss.condition_key must be {CONDITION_KEY!r}")
    if cfg.get("anchor_rule") != ANCHOR_RULE:
        raise ValueError(f"condition_adherence_loss.anchor_rule must be {ANCHOR_RULE!r}")
    scope = cfg.get("scope")
    if scope not in {"train_only", "train_and_validation"}:
        raise ValueError(
            "condition_adherence_loss.scope must be 'train_only' or "
            "'train_and_validation'"
        )
    action_label_scope = str(cfg.get("action_label_scope", "anchor_only"))
    if action_label_scope not in {"anchor_only", "all_active_steps"}:
        raise ValueError(
            "condition_adherence_loss.action_label_scope must be "
            "'anchor_only' or 'all_active_steps'"
        )
    thresholds = _resolve_thresholds(cfg)
    compute_deadzone_intent_labels(
        actions=np.empty((0, len(AXIS_NAMES)), dtype=np.float32),
        thresholds=thresholds,
    )
    values = {
        "directional_enabled": bool(cfg.get("directional_enabled", True)),
        "weight": _nonnegative(cfg.get("weight", 1.0), name="weight"),
        "recorded_crossing_weight": _nonnegative(
            cfg.get("recorded_crossing_weight", 1.0),
            name="recorded_crossing_weight",
        ),
        "counterfactual_violation_weight": _nonnegative(
            cfg.get("counterfactual_violation_weight", 1.0),
            name="counterfactual_violation_weight",
        ),
        "contrast_weight": _nonnegative(
            cfg.get("contrast_weight", 1.0), name="contrast_weight"
        ),
        "counterfactual_ceiling_fraction": _fraction(
            cfg.get("counterfactual_ceiling_fraction", 0.5),
            name="counterfactual_ceiling_fraction",
        ),
        "advantage_margin_scale": _nonnegative(
            cfg.get("advantage_margin_scale", 1.0),
            name="advantage_margin_scale",
        ),
    }
    return {
        "enabled": True,
        "condition_key": CONDITION_KEY,
        "anchor_rule": ANCHOR_RULE,
        "scope": scope,
        "action_label_scope": action_label_scope,
        "thresholds": thresholds,
        "pos": torch.as_tensor(
            [thresholds[axis]["pos"] for axis in AXIS_NAMES], dtype=torch.float32
        ),
        "neg": torch.as_tensor(
            [thresholds[axis]["neg"] for axis in AXIS_NAMES], dtype=torch.float32
        ),
        **values,
    }


def condition_adherence_loss_terms(
    *,
    recorded_direct: torch.Tensor,
    counterfactual_direct: torch.Tensor,
    anchor_mask: torch.Tensor | None,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    zero = recorded_direct.new_zeros(())
    result = {
        "condition_recorded_crossing_loss": zero,
        "condition_counterfactual_violation_loss": zero,
        "condition_contrast_loss": zero,
        "condition_adherence_loss": zero,
        "condition_adherence_anchor_count": zero,
    }
    if not bool(config.get("enabled", False)):
        return result
    if anchor_mask is None:
        raise ValueError("condition_adherence_loss requires condition_adherence_mask")
    if tuple(recorded_direct.shape) != tuple(counterfactual_direct.shape):
        raise ValueError("recorded and counterfactual action chunks must have equal shape")
    expected = (*recorded_direct.shape, 2)
    if tuple(anchor_mask.shape) != expected:
        raise ValueError(
            f"condition_adherence_mask must have shape {expected}, got {tuple(anchor_mask.shape)}"
        )

    mask = anchor_mask.to(device=recorded_direct.device, dtype=torch.bool)
    if bool((mask[..., 0] & mask[..., 1]).any()):
        raise ValueError("condition_adherence_mask cannot select both axis directions")
    count = mask.sum().to(dtype=recorded_direct.dtype)
    result["condition_adherence_anchor_count"] = count
    if not bool(mask.any()):
        return result

    pos = torch.as_tensor(
        config["pos"], dtype=recorded_direct.dtype, device=recorded_direct.device
    ).view(1, 1, -1)
    neg = torch.as_tensor(
        config["neg"], dtype=recorded_direct.dtype, device=recorded_direct.device
    ).view(1, 1, -1)
    positive = mask[..., 0]
    negative = mask[..., 1]
    threshold = torch.where(positive, pos, neg)
    signed_recorded = torch.where(positive, recorded_direct, -recorded_direct)
    signed_counterfactual = torch.where(
        positive, counterfactual_direct, -counterfactual_direct
    )
    selected = positive | negative
    denominator = count.clamp_min(1.0)

    crossing = (
        torch.relu(threshold - signed_recorded) * selected.to(recorded_direct.dtype)
    ).sum() / denominator
    ceiling = threshold * float(config["counterfactual_ceiling_fraction"])
    violation = (
        torch.relu(signed_counterfactual - ceiling)
        * selected.to(recorded_direct.dtype)
    ).sum() / denominator
    required_advantage = threshold * float(config["advantage_margin_scale"])
    contrast = (
        torch.relu(
            required_advantage - (signed_recorded - signed_counterfactual)
        )
        * selected.to(recorded_direct.dtype)
    ).sum() / denominator
    total = float(config["weight"]) * (
        float(config["recorded_crossing_weight"]) * crossing
        + float(config["counterfactual_violation_weight"]) * violation
        + float(config["contrast_weight"]) * contrast
    )
    return {
        "condition_recorded_crossing_loss": crossing,
        "condition_counterfactual_violation_loss": violation,
        "condition_contrast_loss": contrast,
        "condition_adherence_loss": total,
        "condition_adherence_anchor_count": count,
    }


def _resolve_thresholds(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    inline_present = cfg.get("thresholds") is not None
    path_raw = cfg.get("threshold_json")
    path_present = path_raw is not None and str(path_raw).strip() != ""
    if inline_present == path_present:
        raise ValueError(
            "condition_adherence_loss requires exactly one of thresholds or threshold_json"
        )
    if inline_present:
        payload: Any = cfg["thresholds"]
    else:
        path = Path(str(path_raw))
        if not path.is_file():
            raise FileNotFoundError(
                f"condition_adherence_loss threshold_json does not exist: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and "deadzone_action" in payload:
        payload = payload["deadzone_action"]
    if not isinstance(payload, Mapping):
        raise ValueError("condition_adherence_loss thresholds must be a mapping")
    return copy.deepcopy(dict(payload))


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"condition_adherence_loss.{name} must be a boolean")
    return bool(value)


def _nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"condition_adherence_loss.{name} must be non-negative")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"condition_adherence_loss.{name} must be non-negative")
    return result


def _fraction(value: Any, *, name: str) -> float:
    result = _nonnegative(value, name=name)
    if result > 1.0:
        raise ValueError(f"condition_adherence_loss.{name} must be in [0, 1]")
    return result
