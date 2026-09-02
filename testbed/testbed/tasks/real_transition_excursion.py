"""Causal positive-excursion state for Real Transition cycles."""

from __future__ import annotations

from typing import Any

import numpy as np

EXCURSION_OBSERVED_KEY = "real_transition_excursion_observed_v1"
EXCURSION_CONTRACT_SCHEMA = "real_transition_excursion_observed_contract_v1"


def derive_excursion_observed(
    *,
    qpos: Any,
    minimum_delta_rad: float,
    minimum_consecutive_samples: int,
) -> np.ndarray:
    """Return a causal monotonic ``(T, 1)`` positive-excursion latch."""

    qpos_array = np.asarray(qpos, dtype=np.float64)
    if qpos_array.ndim != 2 or qpos_array.shape[1] < 1:
        raise ValueError("excursion state requires qpos shaped (T, A)")
    if not np.isfinite(qpos_array).all():
        raise ValueError("excursion state qpos must be finite")
    if qpos_array.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)
    threshold = float(minimum_delta_rad)
    required = int(minimum_consecutive_samples)
    if threshold <= 0.0 or required <= 0:
        raise ValueError("excursion state thresholds must be positive")

    anchor = float(qpos_array[0, 0])
    relative = _shortest_angle_array(qpos_array[:, 0] - anchor)
    result = np.zeros((len(relative), 1), dtype=np.float32)
    consecutive = 0
    latched = False
    for index, delta in enumerate(relative):
        if not latched:
            consecutive = consecutive + 1 if float(delta) >= threshold else 0
            latched = consecutive >= required
        result[index, 0] = 1.0 if latched else 0.0
    return result


def excursion_chunk_valid_mask(state: Any, *, chunk_steps: int) -> np.ndarray:
    """Return rows whose future action chunk stays in one excursion state."""

    values = np.asarray(state, dtype=np.float32).reshape(-1)
    horizon = int(chunk_steps)
    if horizon <= 0:
        raise ValueError("excursion chunk_steps must be positive")
    if not np.all(np.isin(values, [0.0, 1.0])):
        raise ValueError("excursion state values must be 0 or 1")
    mask = np.zeros((len(values), horizon), dtype=np.uint8)
    for row in range(len(values)):
        width = min(horizon, len(values) - row)
        same = values[row : row + width] == values[row]
        if not bool(np.all(same)):
            width = int(np.flatnonzero(~same)[0])
        mask[row, :width] = 1
    return mask


def build_excursion_contract(
    *,
    minimum_delta_rad: float,
    minimum_consecutive_samples: int,
) -> dict[str, Any]:
    return {
        "schema": EXCURSION_CONTRACT_SCHEMA,
        "condition_key": EXCURSION_OBSERVED_KEY,
        "values": {"pre_excursion": 0, "excursion_observed": 1},
        "derivation": {
            "causal": True,
            "source_fields": ["observations/qpos"],
            "swing_axis_index": 0,
            "anchor": "cycle_row_0_swing_qpos",
            "direction": "positive",
            "minimum_delta_rad": float(minimum_delta_rad),
            "minimum_consecutive_samples": int(minimum_consecutive_samples),
            "latch_monotonic": True,
            "uses_action": False,
            "uses_future_rows": False,
        },
        "chunk_contract": (
            "conditions/excursion_observed_valid_mask masks action queries that "
            "cross the causal positive-excursion boundary"
        ),
    }


def _shortest_angle_array(values: Any) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi
