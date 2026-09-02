"""Causal cycle-phase labels shared by materialization and runtime probes.

The phase is intentionally small and observable: ``0`` means that the measured
rightward excavation has not yet entered a confirmed leftward return, while
``1`` means that a positive excursion was observed and swing has subsequently
dropped from its running peak with negative velocity.  No future row, expert
action, or target-side hindsight is used to latch the phase.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CYCLE_PHASE_KEY = "real_transition_cycle_phase_v1"
CYCLE_PHASE_CONTRACT_SCHEMA = "real_transition_cycle_phase_contract_v1"
PRE_RETURN_PHASE = 0.0
RETURN_PHASE = 1.0
RETURN_CONFIRM_DROP_RAD = 0.05
RETURN_MIN_QVEL_RAD_S = 0.05


def derive_cycle_phase(
    *,
    qpos: Any,
    qvel: Any,
    excursion_min_delta_rad: float,
    excursion_min_consecutive_samples: int,
    return_confirm_drop_rad: float = RETURN_CONFIRM_DROP_RAD,
    return_min_qvel_rad_s: float = RETURN_MIN_QVEL_RAD_S,
) -> np.ndarray:
    """Derive a causal ``(T, 1)`` pre-return/return phase sequence."""

    qpos_array = np.asarray(qpos, dtype=np.float64)
    qvel_array = np.asarray(qvel, dtype=np.float64)
    if (
        qpos_array.ndim != 2
        or qpos_array.shape[1] < 1
        or qvel_array.shape != qpos_array.shape
    ):
        raise ValueError("cycle phase requires matching qpos/qvel arrays shaped (T, A)")
    if not np.isfinite(qpos_array).all() or not np.isfinite(qvel_array).all():
        raise ValueError("cycle phase qpos/qvel must be finite")
    if qpos_array.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)
    excursion_threshold = float(excursion_min_delta_rad)
    excursion_samples = int(excursion_min_consecutive_samples)
    return_drop = float(return_confirm_drop_rad)
    return_speed = float(return_min_qvel_rad_s)
    if excursion_threshold <= 0.0 or excursion_samples <= 0:
        raise ValueError("cycle phase excursion thresholds must be positive")
    if return_drop <= 0.0 or return_speed <= 0.0:
        raise ValueError("cycle phase return thresholds must be positive")

    swing = qpos_array[:, 0]
    anchor = float(swing[0])
    relative = _shortest_angle_array(swing - anchor)
    running_peak = float(relative[0])
    excursion_count = 0
    excursion_observed = False
    return_latched = False
    phase = np.zeros((len(relative), 1), dtype=np.float32)
    for index, (delta, swing_qvel) in enumerate(
        zip(relative, qvel_array[:, 0], strict=True)
    ):
        running_peak = max(running_peak, float(delta))
        if not excursion_observed:
            if float(delta) >= excursion_threshold:
                excursion_count += 1
            else:
                excursion_count = 0
            excursion_observed = excursion_count >= excursion_samples
        if (
            excursion_observed
            and not return_latched
            and running_peak - float(delta) >= return_drop
            and float(swing_qvel) <= -return_speed
        ):
            return_latched = True
        phase[index, 0] = RETURN_PHASE if return_latched else PRE_RETURN_PHASE
    return phase


def phase_chunk_valid_mask(phase: Any, *, chunk_steps: int) -> np.ndarray:
    """Return rows whose future chunk remains inside the current phase."""

    values = np.asarray(phase, dtype=np.float32).reshape(-1)
    horizon = int(chunk_steps)
    if horizon <= 0:
        raise ValueError("phase chunk_steps must be positive")
    if not np.all(np.isin(values, [PRE_RETURN_PHASE, RETURN_PHASE])):
        raise ValueError("cycle phase values must be 0 or 1")
    mask = np.zeros((len(values), horizon), dtype=np.uint8)
    for row in range(len(values)):
        width = min(horizon, len(values) - row)
        same = values[row : row + width] == values[row]
        if not bool(np.all(same)):
            width = int(np.flatnonzero(~same)[0])
        mask[row, :width] = 1
    return mask


def build_cycle_phase_contract(
    *,
    excursion_min_delta_rad: float,
    excursion_min_consecutive_samples: int,
) -> dict[str, Any]:
    """Build the traceable contract written beside a materialized dataset."""

    return {
        "schema": CYCLE_PHASE_CONTRACT_SCHEMA,
        "condition_key": CYCLE_PHASE_KEY,
        "values": {"pre_return": int(PRE_RETURN_PHASE), "return": int(RETURN_PHASE)},
        "derivation": {
            "causal": True,
            "source_fields": ["observations/qpos", "observations/qvel"],
            "swing_axis_index": 0,
            "excursion_direction": "positive",
            "excursion_min_delta_rad": float(excursion_min_delta_rad),
            "excursion_min_consecutive_samples": int(
                excursion_min_consecutive_samples
            ),
            "running_peak_drop_rad": RETURN_CONFIRM_DROP_RAD,
            "return_qvel_at_or_below_rad_s": -RETURN_MIN_QVEL_RAD_S,
            "latch_monotonic": True,
            "uses_action": False,
            "uses_future_rows": False,
        },
        "chunk_contract": (
            "conditions/cycle_phase_valid_mask masks action queries that cross "
            "the causal phase boundary"
        ),
    }


def _shortest_angle_array(values: Any) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi
