"""Deadzone-aware action targets for continuous ACT training.

The hydraulic deadzone is a semantic boundary, not a runtime safety gate.  A
recorded command inside the deadzone is therefore represented as ``neutral``
for the training target, while an active command keeps its signed magnitude.
The continuous ACT head remains the only command source at inference time;
the phase head in this module is auxiliary supervision only.
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
AXIS_COUNT = len(AXIS_NAMES)
NEUTRAL = 0
POSITIVE = 1
NEGATIVE = 2
PHASE_COUNT = 3


@dataclass(frozen=True)
class EffectiveActionLabels:
    """Direct-domain effective target and its per-axis semantic labels."""

    action: np.ndarray
    phase: np.ndarray
    valid: np.ndarray
    loss_weight: np.ndarray
    transition: np.ndarray
    persistent: np.ndarray


def resolve_effective_action_config(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve and validate the train-time effective-action contract."""

    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "thresholds": {},
            "neutral_weight": 1.0,
            "active_weight": 1.0,
            "persistent_weight": 1.0,
            "transition_weight": 1.0,
            "active_margin": 0.0,
            "transition_window_steps": 1,
            "persistence_steps": 1,
            "classification_weight": 0.0,
            "magnitude_weight": 0.0,
            "raw_continuity_weight": 0.0,
            "class_weights": np.ones(PHASE_COUNT, dtype=np.float32),
            "current_steps": 0,
        }

    thresholds = _resolve_thresholds(cfg)
    positive = {
        "neutral_weight": _positive(cfg.get("neutral_weight", 1.0), "neutral_weight"),
        "active_weight": _positive(cfg.get("active_weight", 1.5), "active_weight"),
        "persistent_weight": _positive(
            cfg.get("persistent_weight", 1.75), "persistent_weight"
        ),
        "transition_weight": _positive(
            cfg.get("transition_weight", 4.0), "transition_weight"
        ),
        "active_margin": _nonnegative(
            cfg.get("active_margin", 0.02), "active_margin"
        ),
    }
    transition_window_steps = int(cfg.get("transition_window_steps", 4))
    persistence_steps = int(cfg.get("persistence_steps", 4))
    current_steps = int(cfg.get("current_steps", 0))
    if transition_window_steps < 1:
        raise ValueError("effective_action.transition_window_steps must be >= 1")
    if persistence_steps < 1:
        raise ValueError("effective_action.persistence_steps must be >= 1")
    if current_steps < 0:
        raise ValueError("effective_action.current_steps must be >= 0")
    class_weights = _vector(
        cfg.get("class_weights", [1.0, 4.0, 4.0]),
        PHASE_COUNT,
        "class_weights",
    )
    if np.any(class_weights <= 0.0):
        raise ValueError("effective_action.class_weights must be positive")
    return {
        "enabled": True,
        "thresholds": thresholds,
        **positive,
        "transition_window_steps": transition_window_steps,
        "persistence_steps": persistence_steps,
        "classification_weight": _nonnegative(
            cfg.get("classification_weight", 0.15), "classification_weight"
        ),
        "magnitude_weight": _nonnegative(
            cfg.get("magnitude_weight", 0.10), "magnitude_weight"
        ),
        "raw_continuity_weight": _nonnegative(
            cfg.get("raw_continuity_weight", 0.25), "raw_continuity_weight"
        ),
        "class_weights": class_weights,
        "current_steps": current_steps,
    }


def compute_effective_action_labels(
    *,
    actions: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
    transition_window_steps: int = 4,
    persistence_steps: int = 4,
    valid: np.ndarray | None = None,
    neutral_weight: float = 1.0,
    active_weight: float = 1.5,
    persistent_weight: float = 1.75,
    transition_weight: float = 4.0,
    active_margin: float = 0.02,
) -> EffectiveActionLabels:
    """Map direct expert actions to neutral/positive/negative targets.

    The comparison is directional: ``pos`` and ``neg`` are kept separately,
    which is important because the excavator joystick has asymmetric deadzones.
    The transition and persistence masks are causal labels for training
    weighting; they do not modify the action sent at runtime.
    """

    action = _validate_actions(actions)
    pos, neg = _threshold_arrays(thresholds)
    valid_steps = _valid_steps(valid, action.shape[0])
    positive_mask = (action >= pos.reshape(1, -1)) & valid_steps[:, None]
    negative_mask = (action <= -neg.reshape(1, -1)) & valid_steps[:, None]

    phase = np.full(action.shape, NEUTRAL, dtype=np.int64)
    phase[positive_mask] = POSITIVE
    phase[negative_mask] = NEGATIVE
    # Keep active direction and relative effort, but give a command sitting
    # on the hydraulic boundary a small direct-domain margin. The mapping is
    # monotone and saturates at +/-1, so it cannot invent a direction or exceed
    # the joystick command range.
    positive_excess = np.clip(
        (action - pos.reshape(1, -1))
        / np.maximum(1.0 - pos.reshape(1, -1), 1.0e-6),
        0.0,
        1.0,
    )
    negative_excess = np.clip(
        (-action - neg.reshape(1, -1))
        / np.maximum(1.0 - neg.reshape(1, -1), 1.0e-6),
        0.0,
        1.0,
    )
    positive_target = pos.reshape(1, -1) + float(active_margin) + positive_excess * (
        1.0 - pos.reshape(1, -1) - float(active_margin)
    )
    negative_target = neg.reshape(1, -1) + float(active_margin) + negative_excess * (
        1.0 - neg.reshape(1, -1) - float(active_margin)
    )
    effective = np.where(
        positive_mask,
        np.minimum(positive_target, 1.0),
        np.where(negative_mask, -np.minimum(negative_target, 1.0), 0.0),
    ).astype(np.float32)
    effective[~valid_steps] = 0.0

    transition = np.zeros_like(phase, dtype=bool)
    previous = np.zeros_like(phase)
    if phase.shape[0] > 1:
        previous[1:] = phase[:-1]
    active = phase != NEUTRAL
    transition = active & (phase != previous)
    transition[~valid_steps] = False

    transition_window = np.zeros_like(transition, dtype=bool)
    for offset in range(max(1, int(transition_window_steps))):
        shifted = np.zeros_like(transition)
        if offset == 0:
            shifted = transition
        elif offset < transition.shape[0]:
            shifted[offset:] = transition[:-offset]
        # A window never carries a previous direction through a stop or a
        # sign change; only the current active direction receives the boost.
        transition_window |= shifted & active

    persistent = np.zeros_like(transition, dtype=bool)
    steps = max(1, int(persistence_steps))
    for start in range(action.shape[0]):
        end = start + steps
        if end > action.shape[0]:
            continue
        window = phase[start:end]
        persistent[start] = (
            (window[0] != NEUTRAL)
            & np.all(window == window[0:1], axis=0)
            & np.all(valid_steps[start:end, None], axis=0)
        )

    weights = np.full(action.shape, float(neutral_weight), dtype=np.float32)
    weights[active] = np.maximum(float(active_weight), float(persistent_weight))
    weights[persistent] = np.maximum(
        weights[persistent], float(persistent_weight)
    )
    weights[transition_window] = np.maximum(
        weights[transition_window], float(transition_weight)
    )
    weights[~valid_steps] = 0.0

    return EffectiveActionLabels(
        action=effective,
        phase=phase,
        valid=np.broadcast_to(valid_steps[:, None], phase.shape).copy(),
        loss_weight=weights,
        transition=transition,
        persistent=persistent,
    )


def effective_action_loss_terms(
    *,
    target_normalized: torch.Tensor,
    policy_normalized: torch.Tensor,
    phase_logits: torch.Tensor | None,
    phase_labels: torch.Tensor | None,
    phase_valid: torch.Tensor | None,
    valid_mask: torch.Tensor,
    loss_weight: torch.Tensor | None,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Return auxiliary phase/magnitude losses for effective targets."""

    zero = policy_normalized.new_zeros(())
    if not bool(config.get("enabled", False)):
        return {
            "effective_action_phase_nll": zero,
            "effective_action_active_l1": zero,
            "effective_action_loss": zero,
        }
    if phase_logits is None or phase_labels is None or phase_valid is None:
        raise ValueError(
            "effective_action requires phase logits, labels, and valid mask"
        )
    expected = (*policy_normalized.shape[:2], AXIS_COUNT * PHASE_COUNT)
    if tuple(phase_logits.shape) != expected:
        raise ValueError(
            f"effective_action phase logits must have shape {expected}, "
            f"got {tuple(phase_logits.shape)}"
        )
    labels = phase_labels.to(device=policy_normalized.device, dtype=torch.long)
    valid = phase_valid.to(device=policy_normalized.device, dtype=torch.bool)
    valid = valid & valid_mask.expand_as(valid)
    current_steps = int(config.get("current_steps", 0))
    if current_steps > 0:
        query_index = torch.arange(
            policy_normalized.shape[1], device=policy_normalized.device
        ).view(1, -1, 1)
        valid = valid & (query_index < current_steps)
    logits = phase_logits.reshape(*policy_normalized.shape[:2], AXIS_COUNT, PHASE_COUNT)
    class_weights = torch.as_tensor(
        config["class_weights"], dtype=logits.dtype, device=logits.device
    )
    nll = F.cross_entropy(
        logits.reshape(-1, PHASE_COUNT),
        labels.reshape(-1),
        weight=class_weights,
        reduction="none",
    ).reshape_as(labels)
    selected_weight = class_weights[labels]
    valid_float = valid.to(nll.dtype)
    phase_nll = (nll * valid_float).sum() / (
        selected_weight.mul(valid_float).sum().clamp_min(1.0)
    )

    mean = torch.as_tensor(action_mean, dtype=policy_normalized.dtype, device=policy_normalized.device)
    std = torch.as_tensor(action_std, dtype=policy_normalized.dtype, device=policy_normalized.device)
    target_direct = target_normalized * std + mean
    policy_direct = policy_normalized * std + mean
    active_mask = (labels != NEUTRAL) & valid
    magnitude_l1 = F.l1_loss(policy_direct, target_direct, reduction="none")
    if loss_weight is not None:
        weights = loss_weight.to(
            device=policy_normalized.device, dtype=policy_normalized.dtype
        )
        magnitude_l1 = magnitude_l1 * weights
    magnitude_l1 = (magnitude_l1 * active_mask.to(magnitude_l1.dtype)).sum() / (
        active_mask.to(magnitude_l1.dtype).sum().clamp_min(1.0)
    )
    return {
        "effective_action_phase_nll": phase_nll,
        "effective_action_active_l1": magnitude_l1,
        "effective_action_loss": (
            float(config["classification_weight"]) * phase_nll
            + float(config["magnitude_weight"]) * magnitude_l1
        ),
    }


def weighted_action_l1(
    *,
    expert: torch.Tensor,
    policy: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_weight: torch.Tensor | None,
    action_loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute weighted normalized L1 while preserving padding masks."""

    l1 = F.l1_loss(expert, policy, reduction="none")
    mask = valid_mask.to(dtype=torch.bool).expand_as(l1)
    if action_loss_mask is not None:
        mask &= action_loss_mask.to(device=expert.device, dtype=torch.bool).unsqueeze(-1)
    weights = torch.ones_like(l1)
    if loss_weight is not None:
        weights = loss_weight.to(device=expert.device, dtype=l1.dtype)
    weighted = l1 * weights * mask.to(l1.dtype)
    return weighted.sum() / (weights * mask.to(l1.dtype)).sum().clamp_min(1.0)


def summarize_effective_action_labels(labels: EffectiveActionLabels) -> dict[str, Any]:
    """Return compact counts for the saved run manifest."""

    phase = np.asarray(labels.phase, dtype=np.int64)
    valid = np.asarray(labels.valid, dtype=bool)
    return {
        "steps": int(phase.shape[0]),
        "valid_axis_rows": int(valid.sum()),
        "phase_counts": {
            "neutral": int(np.count_nonzero((phase == NEUTRAL) & valid)),
            "positive": int(np.count_nonzero((phase == POSITIVE) & valid)),
            "negative": int(np.count_nonzero((phase == NEGATIVE) & valid)),
        },
        "transition_axis_events": int(np.count_nonzero(labels.transition & valid)),
        "persistent_axis_events": int(np.count_nonzero(labels.persistent & valid)),
    }


def _resolve_thresholds(cfg: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    if "thresholds" in cfg:
        raw = cfg["thresholds"]
    else:
        path_raw = cfg.get("threshold_json")
        if not path_raw:
            raise ValueError(
                "effective_action.enabled requires threshold_json or thresholds"
            )
        path = Path(str(path_raw))
        if not path.exists():
            raise FileNotFoundError(f"effective_action threshold_json does not exist: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and "deadzone_action" in raw:
        raw = raw["deadzone_action"]
    if not isinstance(raw, Mapping):
        raise ValueError("effective_action thresholds must be a mapping")
    result: dict[str, dict[str, float]] = {}
    for axis in AXIS_NAMES:
        axis_raw = raw.get(axis)
        if not isinstance(axis_raw, Mapping):
            raise ValueError(f"effective_action thresholds missing axis {axis!r}")
        result[axis] = {
            "pos": _threshold_value(axis_raw.get("pos"), axis=axis, direction="pos"),
            "neg": _threshold_value(axis_raw.get("neg"), axis=axis, direction="neg"),
        }
    return result


def _threshold_arrays(
    thresholds: Mapping[str, Mapping[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    pos = []
    neg = []
    for axis in AXIS_NAMES:
        raw = thresholds.get(axis)
        if not isinstance(raw, Mapping):
            raise ValueError(f"thresholds missing axis {axis!r}")
        pos.append(_threshold_value(raw.get("pos"), axis=axis, direction="pos"))
        neg.append(_threshold_value(raw.get("neg"), axis=axis, direction="neg"))
    return np.asarray(pos, dtype=np.float32), np.asarray(neg, dtype=np.float32)


def _threshold_value(value: Any, *, axis: str, direction: str) -> float:
    if isinstance(value, Mapping):
        value = value.get("threshold_action_abs", value.get("value"))
    if value is None:
        raise ValueError(f"threshold for {axis}.{direction} is missing")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"threshold for {axis}.{direction} must be finite and >= 0")
    return result


def _validate_actions(actions: np.ndarray) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != AXIS_COUNT:
        raise ValueError(f"actions must have shape (T, {AXIS_COUNT}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("actions must contain finite values")
    return array


def _valid_steps(valid: np.ndarray | None, length: int) -> np.ndarray:
    if valid is None:
        return np.ones(length, dtype=bool)
    result = np.asarray(valid, dtype=bool).reshape(-1)
    if result.size != length:
        raise ValueError(f"valid mask must have length {length}, got {result.size}")
    return result


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"effective_action.{name} must be finite and > 0")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"effective_action.{name} must be finite and >= 0")
    return result


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    if isinstance(value, (int, float)):
        values = [float(value)] * length
    else:
        values = [float(item) for item in value]
    if len(values) != length or not np.isfinite(values).all():
        raise ValueError(f"effective_action.{name} must contain {length} finite values")
    return np.asarray(values, dtype=np.float32)
