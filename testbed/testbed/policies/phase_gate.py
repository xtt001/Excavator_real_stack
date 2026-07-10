"""Small offline/runtime helpers for policy phase gates."""

from __future__ import annotations

from typing import Any

import numpy as np

from testbed.policies.deadzone_eval import effective_direction_mask


def should_move_labels(
    expert_action: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> np.ndarray:
    """Return per-step labels for expert actions that cross any directional deadzone."""

    return effective_direction_mask(expert_action, thresholds).any(axis=(1, 2))


def direction_effective_labels(
    expert_action: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> np.ndarray:
    """Return per-step axis-direction labels in axis-major pos/neg order."""

    return effective_direction_mask(expert_action, thresholds).reshape(expert_action.shape[0], -1)


def build_hysteresis_mask(
    probability: np.ndarray,
    *,
    open_threshold: float,
    close_threshold: float,
    initial_active: bool = False,
) -> np.ndarray:
    """Convert per-step should-move probabilities into a stable active mask."""

    probs = np.asarray(probability, dtype=np.float32).reshape(-1)
    open_value = float(open_threshold)
    close_value = float(close_threshold)
    if not 0.0 <= close_value <= open_value <= 1.0:
        raise ValueError("phase gate thresholds must satisfy 0 <= close <= open <= 1.")

    active = bool(initial_active)
    mask = np.zeros(probs.shape[0], dtype=bool)
    for idx, prob in enumerate(probs):
        if active:
            active = bool(prob >= close_value)
        else:
            active = bool(prob >= open_value)
        mask[idx] = active
    return mask


def apply_phase_gate_to_actions(
    policy_action: np.ndarray,
    active_mask: np.ndarray,
    *,
    inactive_scale: float = 0.0,
) -> np.ndarray:
    """Scale policy actions on inactive phase-gate steps."""

    policy = np.asarray(policy_action, dtype=np.float32)
    if policy.ndim != 2 or policy.shape[1] != 4:
        raise ValueError(f"policy_action must have shape (T, 4), got {policy.shape}")
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    if active.shape != (policy.shape[0],):
        raise ValueError(f"active_mask must have shape ({policy.shape[0]},), got {active.shape}")
    scale = float(inactive_scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("inactive_scale must satisfy 0 <= inactive_scale <= 1.")
    gated = policy.copy()
    gated[~active] *= scale
    return gated


def apply_direction_gate_to_actions(
    policy_action: np.ndarray,
    active_direction_mask: np.ndarray,
    *,
    inactive_scale: float = 0.0,
) -> np.ndarray:
    """Scale policy action signs whose matching axis-direction gate is inactive."""

    policy = np.asarray(policy_action, dtype=np.float32)
    if policy.ndim != 2 or policy.shape[1] != 4:
        raise ValueError(f"policy_action must have shape (T, 4), got {policy.shape}")
    active = np.asarray(active_direction_mask, dtype=bool)
    if active.shape != (policy.shape[0], 8):
        raise ValueError(f"active_direction_mask must have shape ({policy.shape[0]}, 8), got {active.shape}")
    scale = float(inactive_scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("inactive_scale must satisfy 0 <= inactive_scale <= 1.")
    gated = policy.copy()
    pos_active = active[:, 0::2]
    neg_active = active[:, 1::2]
    gated[(policy > 0.0) & ~pos_active] *= scale
    gated[(policy < 0.0) & ~neg_active] *= scale
    return gated


def phase_gate_metadata(
    *,
    feature_names: list[str],
    open_threshold: float,
    close_threshold: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact metadata payload for saved phase-gate artifacts."""

    payload: dict[str, Any] = {
        "feature_names": list(feature_names),
        "open_threshold": float(open_threshold),
        "close_threshold": float(close_threshold),
    }
    if extra:
        payload.update(dict(extra))
    return payload
