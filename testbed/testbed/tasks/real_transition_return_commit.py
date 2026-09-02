"""Offline hindsight label for planner-owned Real Transition return intent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

RETURN_COMMIT_KEY = "real_transition_return_commit_v1"
RETURN_COMMIT_CONTRACT_SCHEMA = "real_transition_return_commit_contract_v1"
RETURN_COMMIT_ACTION_INTENT_THRESHOLD = 0.05


@dataclass(frozen=True)
class ReturnCommitDerivation:
    state: np.ndarray
    valid_mask: np.ndarray
    event_row: int | None
    evaluable: bool
    reason: str | None


def derive_return_commit(
    *,
    action: Any,
    excursion_observed: Any,
    return_phase: Any,
    chunk_steps: int,
    action_intent_threshold: float = RETURN_COMMIT_ACTION_INTENT_THRESHOLD,
) -> ReturnCommitDerivation:
    """Derive a monotonic hindsight label without claiming online observability."""

    actions = np.asarray(action, dtype=np.float64)
    excursion = np.asarray(excursion_observed, dtype=np.float64).reshape(-1)
    phase = np.asarray(return_phase, dtype=np.float64).reshape(-1)
    horizon = int(chunk_steps)
    threshold = float(action_intent_threshold)
    if actions.ndim != 2 or actions.shape[1] < 1:
        raise ValueError("return commit requires action shaped (T, A)")
    if len(actions) != len(excursion) or len(actions) != len(phase):
        raise ValueError("return commit action/state lengths must match")
    if horizon <= 0 or threshold <= 0.0:
        raise ValueError("return commit thresholds must be positive")
    if not np.isfinite(actions).all():
        raise ValueError("return commit actions must be finite")
    if not np.all(np.isin(excursion, [0.0, 1.0])):
        raise ValueError("excursion_observed must contain only 0/1")
    if not np.all(np.isin(phase, [0.0, 1.0])):
        raise ValueError("return_phase must contain only 0/1")

    event_row, reason = _find_event_row(
        swing_action=actions[:, 0],
        excursion=excursion,
        phase=phase,
        threshold=threshold,
    )
    state = np.zeros((len(actions), 1), dtype=np.float32)
    if event_row is None:
        return ReturnCommitDerivation(
            state=state,
            valid_mask=np.zeros((len(actions), horizon), dtype=np.uint8),
            event_row=None,
            evaluable=False,
            reason=reason,
        )
    state[event_row:, 0] = 1.0
    return ReturnCommitDerivation(
        state=state,
        valid_mask=return_commit_chunk_valid_mask(state, chunk_steps=horizon),
        event_row=event_row,
        evaluable=True,
        reason=None,
    )


def return_commit_chunk_valid_mask(state: Any, *, chunk_steps: int) -> np.ndarray:
    """Mask action queries that cross the hindsight DIG-to-RETURN boundary."""

    values = np.asarray(state, dtype=np.float32).reshape(-1)
    horizon = int(chunk_steps)
    if horizon <= 0:
        raise ValueError("return commit chunk_steps must be positive")
    if not np.all(np.isin(values, [0.0, 1.0])):
        raise ValueError("return commit state values must be 0 or 1")
    mask = np.zeros((len(values), horizon), dtype=np.uint8)
    for row in range(len(values)):
        width = min(horizon, len(values) - row)
        same = values[row : row + width] == values[row]
        if not bool(np.all(same)):
            width = int(np.flatnonzero(~same)[0])
        mask[row, :width] = 1
    return mask


def build_return_commit_contract(
    *, action_intent_threshold: float = RETURN_COMMIT_ACTION_INTENT_THRESHOLD
) -> dict[str, Any]:
    threshold = float(action_intent_threshold)
    if threshold <= 0.0:
        raise ValueError("return commit action_intent_threshold must be positive")
    return {
        "schema": RETURN_COMMIT_CONTRACT_SCHEMA,
        "condition_key": RETURN_COMMIT_KEY,
        "values": {"dig": 0, "return": 1},
        "derivation": {
            "causal_online": False,
            "hindsight_training_label": True,
            "source_fields": [
                "action[:,0]",
                "conditions/real_transition_excursion_observed_v1",
                "conditions/real_transition_cycle_phase_v1",
            ],
            "return_reference": "first cycle_phase=1 row",
            "negative_intent_action_at_or_below": -threshold,
            "event": (
                "start of the final contiguous negative-intent segment containing "
                "the first return row; if the return row is not negative-intent, "
                "use the final contiguous negative-intent segment ending before it"
            ),
            "required_state_at_event": {
                "excursion_observed": 1,
                "cycle_phase": 0,
            },
            "latch_monotonic": True,
        },
        "runtime": {
            "owner": "planner_or_explicit_task_command",
            "automatically_derived_from_observation": False,
            "goal_commit_resets_to": 0,
            "allowed_transition": "0_to_1_once_per_goal",
        },
        "chunk_contract": (
            "conditions/return_commit_valid_mask masks action queries that cross "
            "the hindsight DIG-to-RETURN boundary"
        ),
    }


def _find_event_row(
    *,
    swing_action: np.ndarray,
    excursion: np.ndarray,
    phase: np.ndarray,
    threshold: float,
) -> tuple[int | None, str | None]:
    excursion_rows = np.flatnonzero(excursion >= 0.5)
    if excursion_rows.size == 0:
        return None, "missing_excursion_observed"
    return_rows = np.flatnonzero(phase >= 0.5)
    if return_rows.size == 0:
        return None, "missing_return_phase"
    excursion_row = int(excursion_rows[0])
    return_row = int(return_rows[0])
    if return_row <= excursion_row:
        return None, "return_phase_not_after_excursion"
    negative = np.asarray(swing_action, dtype=np.float64) <= -float(threshold)
    cursor = return_row if bool(negative[return_row]) else return_row - 1
    while cursor >= excursion_row and not bool(negative[cursor]):
        cursor -= 1
    if cursor < excursion_row:
        return None, "missing_negative_intent_before_return"
    while cursor > excursion_row and bool(negative[cursor - 1]):
        cursor -= 1
    if float(excursion[cursor]) < 0.5:
        return None, "event_before_excursion"
    if float(phase[cursor]) >= 0.5:
        return None, "event_not_pre_return"
    return int(cursor), None
