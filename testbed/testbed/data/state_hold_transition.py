"""Transition-anchor sampling primitives for state-hold training.

This module owns only the NumPy-level sampling contract.  It deliberately
reuses the action-domain deadzone label owner so transition sampling cannot
drift from the intent labels used by training losses.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from testbed.data.deadzone_intent_labels import (
    AXIS_NAMES,
    compute_deadzone_intent_labels,
)


def resolve_state_hold_transition_config(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the canonical transition-sampling configuration.

    The feature is disabled by default so existing training configs retain
    their uniform sampling behavior.  Enabled configs must identify exactly
    one deadzone source: an inline ``thresholds`` mapping or ``threshold_json``.
    """

    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("state_hold_transition config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "thresholds": {},
            "probability": 0.0,
            "hold_horizon_steps": 1,
            "append_samples_per_episode": 0,
        }

    probability = _probability(cfg.get("probability", 0.0))
    hold_horizon_steps = _positive_integer(
        cfg.get("hold_horizon_steps", 1),
        name="hold_horizon_steps",
    )
    thresholds = _resolve_thresholds(cfg)
    append_samples = _nonnegative_integer(
        cfg.get("append_samples_per_episode", 0),
        name="append_samples_per_episode",
    )
    if append_samples not in {0, 1}:
        raise ValueError(
            "state_hold_transition.append_samples_per_episode must be 0 or 1"
        )
    # Reuse the intent-label owner as the schema validator.  An empty action
    # sequence validates every axis/direction without creating synthetic
    # transition evidence.
    compute_deadzone_intent_labels(
        actions=np.empty((0, len(AXIS_NAMES)), dtype=np.float32),
        thresholds=thresholds,
    )
    return {
        "enabled": True,
        "thresholds": thresholds,
        "probability": probability,
        "hold_horizon_steps": hold_horizon_steps,
        "append_samples_per_episode": append_samples,
    }


def compute_transition_direction_mask(
    *,
    actions: np.ndarray,
    thresholds: dict[str, dict[str, Any]],
    action_loss_mask: np.ndarray | None = None,
    tail_idle_mask: np.ndarray | None = None,
    owner_automation: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``(T, axis, direction)`` inactive-to-effective transitions."""

    labels = compute_deadzone_intent_labels(
        actions=actions,
        thresholds=thresholds,
        action_loss_mask=action_loss_mask,
        tail_idle_mask=tail_idle_mask,
        owner_automation=owner_automation,
    )
    effective = labels.move_mask
    previous = np.zeros_like(effective, dtype=bool)
    previous[1:] = effective[:-1]
    return (effective & ~previous).astype(bool)


def intersect_transition_starts(
    transition_mask: np.ndarray,
    valid_starts: Sequence[int] | np.ndarray,
    *,
    total_steps: int | None = None,
    hold_horizon_steps: int | None = None,
) -> np.ndarray:
    """Intersect transition anchors with valid starts and an optional horizon.

    When ``hold_horizon_steps`` is supplied, a start is retained only when the
    full state-hold window fits in the episode: ``start + horizon <= T``.
    """

    mask = _transition_mask(transition_mask)
    mask_steps = int(mask.shape[0])
    if total_steps is None:
        episode_steps = mask_steps
    else:
        episode_steps = _nonnegative_integer(total_steps, name="total_steps")
        if episode_steps != mask_steps:
            raise ValueError(
                "total_steps must match transition_mask length: "
                f"{episode_steps} != {mask_steps}"
            )

    starts = _start_indices(
        valid_starts,
        name="valid_starts",
        upper_bound=episode_steps,
    )
    selected = starts[mask.any(axis=(1, 2))[starts]]
    if hold_horizon_steps is not None:
        horizon = _positive_integer(
            hold_horizon_steps,
            name="hold_horizon_steps",
        )
        selected = selected[selected + horizon <= episode_steps]
    return selected.astype(np.int64, copy=False)


def sample_state_hold_start(
    *,
    valid_starts: Sequence[int] | np.ndarray,
    transition_starts: Sequence[int] | np.ndarray,
    probability: float,
    rng: Any | None = None,
) -> int:
    """Sample a transition start with ``probability``, else any valid start.

    ``rng`` may be a NumPy ``Generator``, ``RandomState``, or compatible test
    double.  Omitting it intentionally uses ``np.random`` so repository-wide
    NumPy seeding and monkeypatching retain their established behavior.
    """

    valid = _start_indices(valid_starts, name="valid_starts")
    if valid.size == 0:
        raise ValueError("valid_starts must not be empty")
    transitions = _start_indices(transition_starts, name="transition_starts")
    if transitions.size and not np.isin(transitions, valid).all():
        raise ValueError("transition_starts must be a subset of valid_starts")

    transition_probability = _probability(probability)
    random_source = np.random if rng is None else rng
    use_transition = False
    if transitions.size and transition_probability > 0.0:
        use_transition = transition_probability == 1.0 or (
            float(random_source.random()) < transition_probability
        )
    pool = transitions if use_transition else valid
    return int(random_source.choice(pool))


def anchor_transition_direction_mask(
    transition_mask: np.ndarray,
    start: int,
) -> np.ndarray:
    """Return the ``(axis, direction)`` transition mask at one sampled start."""

    mask = _transition_mask(transition_mask)
    index = _nonnegative_integer(start, name="start")
    if index >= mask.shape[0]:
        raise ValueError(
            f"start must be less than transition_mask length {mask.shape[0]}, got {index}"
        )
    return mask[index].copy()


def _resolve_thresholds(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    inline_present = cfg.get("thresholds") is not None
    path_raw = cfg.get("threshold_json")
    path_present = path_raw is not None and str(path_raw).strip() != ""
    if inline_present == path_present:
        raise ValueError(
            "state_hold_transition.enabled requires exactly one of thresholds "
            "or threshold_json"
        )

    if inline_present:
        payload: Any = cfg["thresholds"]
    else:
        path = Path(str(path_raw))
        if not path.is_file():
            raise FileNotFoundError(
                f"state_hold_transition threshold_json does not exist: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, Mapping) and "deadzone_action" in payload:
        payload = payload["deadzone_action"]
    if not isinstance(payload, Mapping):
        raise ValueError("state_hold_transition thresholds must be a mapping")
    return copy.deepcopy(dict(payload))


def _transition_mask(value: np.ndarray) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    expected_tail = (len(AXIS_NAMES), 2)
    if mask.ndim != 3 or mask.shape[1:] != expected_tail:
        raise ValueError(
            "transition_mask must have shape "
            f"(T, {expected_tail[0]}, {expected_tail[1]}), got {mask.shape}"
        )
    return mask


def _start_indices(
    value: Sequence[int] | np.ndarray,
    *,
    name: str,
    upper_bound: int | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {raw.shape}")
    if raw.size == 0:
        return np.zeros(0, dtype=np.int64)
    if raw.dtype.kind not in {"i", "u"} or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer indices")
    indices = raw.astype(np.int64, copy=False)
    if np.any(indices < 0):
        raise ValueError(f"{name} must contain nonnegative indices")
    if upper_bound is not None and np.any(indices >= int(upper_bound)):
        raise ValueError(f"{name} contains an index outside total_steps={upper_bound}")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} must not contain duplicate indices")
    return indices


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"state_hold_transition.{name} must be a boolean")
    return bool(value)


def _probability(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("state_hold_transition.probability must be in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "state_hold_transition.probability must be in [0, 1]"
        ) from exc
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("state_hold_transition.probability must be in [0, 1]")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result <= 0:
        raise ValueError(f"state_hold_transition.{name} must be positive")
    return result


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"state_hold_transition.{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"state_hold_transition.{name} must be nonnegative")
    return result
