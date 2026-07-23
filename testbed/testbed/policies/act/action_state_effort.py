"""Deadzone-aware action-state labels and training losses.

The continuous ACT action remains the only runtime command.  This module adds
an ordinal execution-state target per axis so that idle, barely-effective, and
safe-effort expert commands are not treated as one undifferentiated regression
quantity.  The state head is training/diagnostic supervision only; it never
projects or replaces the continuous action at inference.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

AXIS_NAMES = ("swing", "boom", "stick", "bucket")
STATE_NAMES = ("idle", "pos_near", "pos_safe", "neg_near", "neg_safe")
IDLE = 0
POS_NEAR = 1
POS_SAFE = 2
NEG_NEAR = 3
NEG_SAFE = 4
STATE_COUNT = len(STATE_NAMES)


@dataclass(frozen=True)
class ActionStateLabels:
    """Per-step labels derived from direct-domain expert actions."""

    state: np.ndarray
    valid: np.ndarray
    signed_margin: np.ndarray
    persistent_effective: np.ndarray


def resolve_action_state_effort_config(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve and validate the opt-in action-state/effort contract."""

    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "state_count": STATE_COUNT,
            "thresholds": {},
            "near_margin": 0.0,
            "safe_margin": 0.0,
            "required_margin": 0.0,
            "persistence_steps": 1,
            "current_steps": 0,
            "class_weights": np.ones(STATE_COUNT, dtype=np.float32),
            "classification_weight": 0.0,
            "margin_weight": 0.0,
            "idle_weight": 0.0,
            "wrong_weight": 0.0,
            "persistence_weight": 0.0,
        }
    thresholds = _resolve_thresholds(cfg)
    near_margin = _nonnegative(cfg.get("near_margin", 0.0), "near_margin")
    safe_margin = _nonnegative(cfg.get("safe_margin", 0.02), "safe_margin")
    if safe_margin < near_margin:
        raise ValueError("action_state_effort.safe_margin must be >= near_margin")
    required_margin = _nonnegative(
        cfg.get("required_margin", safe_margin), "required_margin"
    )
    persistence_steps = int(cfg.get("persistence_steps", 2))
    if persistence_steps < 1:
        raise ValueError("action_state_effort.persistence_steps must be >= 1")
    current_steps = int(cfg.get("current_steps", 0))
    if current_steps < 0:
        raise ValueError("action_state_effort.current_steps must be >= 0")
    class_weights = _vector(
        cfg.get("class_weights", [1.0, 8.0, 8.0, 8.0, 8.0]),
        STATE_COUNT,
        "class_weights",
    )
    if np.any(class_weights <= 0.0):
        raise ValueError("action_state_effort.class_weights must be positive")
    weights = {
        "classification_weight": _nonnegative(
            cfg.get("classification_weight", 0.5), "classification_weight"
        ),
        "margin_weight": _nonnegative(
            cfg.get("margin_weight", 1.0), "margin_weight"
        ),
        "idle_weight": _nonnegative(
            cfg.get("idle_weight", 0.1), "idle_weight"
        ),
        "wrong_weight": _nonnegative(
            cfg.get("wrong_weight", 0.1), "wrong_weight"
        ),
        "persistence_weight": _nonnegative(
            cfg.get("persistence_weight", 0.25), "persistence_weight"
        ),
    }
    return {
        "enabled": True,
        "state_count": STATE_COUNT,
        "thresholds": thresholds,
        "near_margin": near_margin,
        "safe_margin": safe_margin,
        "required_margin": required_margin,
        "persistence_steps": persistence_steps,
        # ``0`` preserves the original all-chunk auxiliary supervision.  A
        # positive value restricts the state/margin loss to the first N
        # decoder queries, which is the causal current-command contract.
        "current_steps": current_steps,
        "class_weights": class_weights,
        **weights,
    }


def compute_action_state_labels(
    *,
    actions: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
    safe_margin: float = 0.02,
    persistence_steps: int = 2,
    valid: np.ndarray | None = None,
) -> ActionStateLabels:
    """Label expert actions as idle/near/safe per axis.

    Values below the directional deadzone are not treated as a valid active
    behavior.  They remain ``idle`` in the expert label and are handled as a
    continuous under-deadzone failure by the training margin term.
    """

    action = _validate_actions(actions)
    pos, neg = _threshold_arrays(thresholds)
    safe = float(safe_margin)
    if not np.isfinite(safe) or safe < 0.0:
        raise ValueError("safe_margin must be finite and non-negative")
    steps = int(persistence_steps)
    if steps < 1:
        raise ValueError("persistence_steps must be >= 1")
    valid_steps = _valid_steps(valid, action.shape[0])

    positive = action >= pos.reshape(1, -1)
    negative = action <= -neg.reshape(1, -1)
    positive_safe = action >= (pos + safe).reshape(1, -1)
    negative_safe = action <= -(neg + safe).reshape(1, -1)

    state = np.full(action.shape, IDLE, dtype=np.int64)
    state[positive] = POS_NEAR
    state[positive_safe] = POS_SAFE
    state[negative] = NEG_NEAR
    state[negative_safe] = NEG_SAFE
    state[~valid_steps] = IDLE

    signed_margin = np.stack(
        [action - pos.reshape(1, -1), -action - neg.reshape(1, -1)],
        axis=-1,
    ).astype(np.float32)
    effective = np.stack([positive, negative], axis=-1) & valid_steps[:, None, None]
    persistent = np.zeros_like(effective, dtype=bool)
    if steps == 1:
        persistent[:] = effective
    else:
        for start in range(action.shape[0]):
            end = min(action.shape[0], start + steps)
            if end - start < steps:
                continue
            persistent[start] = np.all(effective[start:end], axis=0)

    return ActionStateLabels(
        state=state,
        valid=np.broadcast_to(valid_steps[:, None], state.shape).copy(),
        signed_margin=signed_margin,
        persistent_effective=persistent,
    )


def summarize_action_state_labels(labels: ActionStateLabels) -> dict[str, Any]:
    """Return a reviewable census without retaining source arrays."""

    state = np.asarray(labels.state, dtype=np.int64)
    valid = np.asarray(labels.valid, dtype=bool)
    if state.ndim != 2 or state.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"state must have shape (T, 4), got {state.shape}")
    counts = {}
    for axis_index, axis in enumerate(AXIS_NAMES):
        counts[axis] = {
            name: int(np.count_nonzero((state[:, axis_index] == index) & valid[:, axis_index]))
            for index, name in enumerate(STATE_NAMES)
        }
    return {
        "steps": int(state.shape[0]),
        "valid_axis_rows": int(np.count_nonzero(valid)),
        "axis_order": list(AXIS_NAMES),
        "state_order": list(STATE_NAMES),
        "counts": counts,
        "persistent_effective_events": int(
            np.count_nonzero(np.asarray(labels.persistent_effective, dtype=bool))
        ),
    }


def action_state_loss_terms(
    *,
    policy_direct: torch.Tensor,
    state_logits: torch.Tensor | None,
    state_labels: torch.Tensor | None,
    state_valid: torch.Tensor | None,
    persistent_effective: torch.Tensor | None,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Compute classification plus direct-domain effort boundary losses."""

    zero = policy_direct.new_zeros(())
    if not bool(config.get("enabled", False)):
        return _zero_terms(zero)
    if state_logits is None or state_labels is None or state_valid is None:
        raise ValueError(
            "action_state_effort requires state_logits, state_labels, and state_valid"
        )
    if policy_direct.ndim != 3 or policy_direct.shape[-1] != len(AXIS_NAMES):
        raise ValueError(f"policy_direct must have shape (B, C, 4), got {policy_direct.shape}")
    expected_logits = (*policy_direct.shape[:2], len(AXIS_NAMES) * STATE_COUNT)
    if tuple(state_logits.shape) != expected_logits:
        raise ValueError(
            f"state_logits must have shape {expected_logits}, got {tuple(state_logits.shape)}"
        )
    expected_labels = (*policy_direct.shape[:2], len(AXIS_NAMES))
    if tuple(state_labels.shape) != expected_labels or tuple(state_valid.shape) != expected_labels:
        raise ValueError("action state labels/valid mask shape does not match policy")

    logits = state_logits.reshape(*policy_direct.shape[:2], len(AXIS_NAMES), STATE_COUNT)
    labels = state_labels.to(device=policy_direct.device, dtype=torch.long)
    valid = state_valid.to(device=policy_direct.device, dtype=torch.bool)
    current_steps = int(config.get("current_steps", 0))
    if current_steps > 0:
        query_index = torch.arange(
            policy_direct.shape[1], device=policy_direct.device
        ).view(1, -1, 1)
        valid = valid & (query_index < current_steps)
    class_weights = torch.as_tensor(
        config["class_weights"], dtype=logits.dtype, device=logits.device
    )
    nll = F.cross_entropy(
        logits.reshape(-1, STATE_COUNT),
        labels.reshape(-1),
        weight=class_weights,
        reduction="none",
    ).reshape_as(labels)
    valid_float = valid.to(policy_direct.dtype)
    # ``cross_entropy(weight=...)`` normally divides by the sum of the
    # selected class weights.  Keep that normalization when applying our
    # padding mask; dividing only by the number of rows would silently scale
    # the auxiliary branch by the average class weight and let it dominate
    # the continuous imitation objective.
    selected_class_weight = class_weights[labels]
    class_nll = (nll * valid_float).sum() / (
        selected_class_weight * valid_float
    ).sum().clamp_min(1.0)

    pos, neg = _threshold_tensors(config, policy_direct)
    required_margin = float(config["required_margin"])
    positive_target = (labels == POS_NEAR) | (labels == POS_SAFE)
    negative_target = (labels == NEG_NEAR) | (labels == NEG_SAFE)
    positive_shortfall = torch.relu(pos + required_margin - policy_direct)
    negative_shortfall = torch.relu(policy_direct + neg + required_margin)
    active_mask = (positive_target | negative_target) & valid
    active_weights = torch.ones_like(policy_direct)
    if persistent_effective is not None:
        persistent_axis = persistent_effective.to(device=policy_direct.device, dtype=torch.bool).any(dim=-1)
        active_weights = active_weights + float(config["persistence_weight"]) * persistent_axis.to(policy_direct.dtype)
    margin_error = (
        positive_shortfall * positive_target.to(policy_direct.dtype)
        + negative_shortfall * negative_target.to(policy_direct.dtype)
    ) * active_weights * valid_float
    margin_loss = margin_error.sum() / active_mask.to(policy_direct.dtype).sum().clamp_min(1.0)

    policy_pos_effective = policy_direct >= pos
    policy_neg_effective = policy_direct <= -neg
    idle_mask = (labels == IDLE) & valid
    idle_error = (
        torch.relu(policy_direct - pos) * policy_pos_effective
        + torch.relu(-neg - policy_direct) * policy_neg_effective
    ) * idle_mask.to(policy_direct.dtype)
    idle_loss = idle_error.sum() / idle_mask.to(policy_direct.dtype).sum().clamp_min(1.0)

    wrong_positive = negative_target & policy_pos_effective & valid
    wrong_negative = positive_target & policy_neg_effective & valid
    wrong_error = (
        torch.relu(policy_direct - pos) * wrong_positive
        + torch.relu(-neg - policy_direct) * wrong_negative
    ).to(policy_direct.dtype)
    wrong_loss = wrong_error.sum() / (wrong_positive | wrong_negative).to(policy_direct.dtype).sum().clamp_min(1.0)

    classification_weight = float(config["classification_weight"])
    margin_weight = float(config["margin_weight"])
    idle_weight = float(config["idle_weight"])
    wrong_weight = float(config["wrong_weight"])
    total = (
        classification_weight * class_nll
        + margin_weight * margin_loss
        + idle_weight * idle_loss
        + wrong_weight * wrong_loss
    )
    return {
        "action_state_class_nll": class_nll,
        "action_state_margin_loss": margin_loss,
        "action_state_idle_loss": idle_loss,
        "action_state_wrong_loss": wrong_loss,
        "action_state_loss": total,
    }


def _zero_terms(zero: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "action_state_class_nll": zero,
        "action_state_margin_loss": zero,
        "action_state_idle_loss": zero,
        "action_state_wrong_loss": zero,
        "action_state_loss": zero,
    }


def _threshold_tensors(config: Mapping[str, Any], reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    thresholds = config["thresholds"]
    pos = torch.as_tensor(
        [float(thresholds[axis]["pos"]) for axis in AXIS_NAMES],
        dtype=reference.dtype,
        device=reference.device,
    )
    neg = torch.as_tensor(
        [float(thresholds[axis]["neg"]) for axis in AXIS_NAMES],
        dtype=reference.dtype,
        device=reference.device,
    )
    return pos.view(1, 1, -1), neg.view(1, 1, -1)


def _resolve_thresholds(raw: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    thresholds = raw.get("thresholds")
    if thresholds is None:
        path_raw = raw.get("threshold_json")
        if not path_raw:
            raise ValueError("action_state_effort requires thresholds or threshold_json")
        path = Path(str(path_raw))
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        thresholds = payload.get("deadzone_action", payload)
    if not isinstance(thresholds, Mapping):
        raise ValueError("action_state_effort thresholds must be a mapping")
    result: dict[str, dict[str, float]] = {}
    for axis in AXIS_NAMES:
        axis_raw = thresholds.get(axis)
        if not isinstance(axis_raw, Mapping):
            raise ValueError(f"action_state_effort thresholds missing axis {axis!r}")
        result[axis] = {
            "pos": _nonnegative(axis_raw.get("pos"), f"{axis}.pos"),
            "neg": _nonnegative(axis_raw.get("neg"), f"{axis}.neg"),
        }
    return result


def _threshold_arrays(thresholds: Mapping[str, Mapping[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([float(thresholds[axis]["pos"]) for axis in AXIS_NAMES], dtype=np.float32),
        np.asarray([float(thresholds[axis]["neg"]) for axis in AXIS_NAMES], dtype=np.float32),
    )


def _validate_actions(actions: np.ndarray) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"actions must have shape (T, 4), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("actions contain non-finite values")
    return array


def _valid_steps(valid: np.ndarray | None, steps: int) -> np.ndarray:
    if valid is None:
        return np.ones(steps, dtype=bool)
    result = np.asarray(valid, dtype=bool).reshape(-1)
    if result.size != steps:
        raise ValueError(f"valid mask must have length {steps}, got {result.size}")
    return result


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    if isinstance(value, (int, float)):
        result = np.full(size, float(value), dtype=np.float32)
    else:
        result = np.asarray(value, dtype=np.float32).reshape(-1)
        if result.size != size:
            raise ValueError(f"{name} must have {size} values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


__all__ = [
    "AXIS_NAMES",
    "ActionStateLabels",
    "IDLE",
    "NEG_NEAR",
    "NEG_SAFE",
    "POS_NEAR",
    "POS_SAFE",
    "STATE_COUNT",
    "STATE_NAMES",
    "action_state_loss_terms",
    "compute_action_state_labels",
    "resolve_action_state_effort_config",
    "summarize_action_state_labels",
]
