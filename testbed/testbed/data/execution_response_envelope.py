"""Versioned all-axis command-to-motion response envelopes.

The historical ``execution_response`` sidecar remains the owner of causal
command alignment and its v1 artifact.  This module consumes that alignment and
adds a new, train-calibrated interpretation for the current four-axis dataset:

* command sign to measured-qvel sign is calibrated per axis from train only;
* stationary qvel noise is calibrated from train-only inactive dwell windows;
* effective command onsets are separated into from-rest, already-moving, and
  opposite-moving contexts;
* validation events are marked supported, weakly supported, or out of support
  by a train-only magnitude/qpos response envelope.

The resulting probabilities describe historical operator-command response.
They are not evidence that an unsent model command moved the machine.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from testbed.data.deadzone_intent_labels import AXIS_NAMES
from testbed.data.execution_response import ExecutionResponseEpisode

SCHEMA_VERSION = "all_axis_response_envelope_v1"
DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
MAGNITUDE_RATIO_EDGES = (1.0, 1.1, 1.25, 1.5)


@dataclass(frozen=True)
class ResponseSequence:
    """Causally aligned command and motion arrays for one dataset episode."""

    dataset_episode_id: int
    source_episode_id: int
    split: str
    command: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class ResponseCalibration:
    """Train-only physical sign and stationary-noise calibration."""

    qvel_direction_sign: tuple[int, int, int, int]
    qvel_noise: tuple[float, float, float, float]
    inactive_sample_count: int
    active_sample_count_by_axis: tuple[int, int, int, int]
    direction_agreement_by_axis: tuple[float, float, float, float]
    stationary_window_ticks: int
    response_offset_ticks: int
    source_split: str = "train_only"


def sequence_from_execution_response(
    result: ExecutionResponseEpisode,
    *,
    dataset_episode_id: int,
    split: str,
) -> ResponseSequence:
    """Adapt the existing causal-alignment result without changing v1."""

    valid = (
        result.alignment.valid_mask
        & ~result.alignment.reset_mask
        & np.isfinite(result.qpos).all(axis=1)
        & np.isfinite(result.qvel).all(axis=1)
        & np.isfinite(result.alignment.previous_final_command).all(axis=1)
    )
    return ResponseSequence(
        dataset_episode_id=int(dataset_episode_id),
        source_episode_id=int(result.episode_id),
        split=str(split),
        command=np.asarray(result.alignment.previous_final_command, dtype=np.float32),
        qpos=np.asarray(result.qpos, dtype=np.float32),
        qvel=np.asarray(result.qvel, dtype=np.float32),
        valid_mask=np.asarray(valid, dtype=bool),
    )


def calibrate_response_contract(
    sequences: Sequence[ResponseSequence],
    *,
    positive_threshold: Sequence[float],
    negative_threshold: Sequence[float],
    stationary_window_ticks: int = 9,
    response_offset_ticks: int = 3,
    noise_sigma_multiplier: float = 3.0,
    noise_floor: float = 0.006,
) -> ResponseCalibration:
    """Calibrate qvel sign and robust stationary noise from train only."""

    if not sequences:
        raise ValueError("at least one train sequence is required")
    if any(sequence.split != "train" for sequence in sequences):
        raise ValueError("response calibration may read train sequences only")
    window = int(stationary_window_ticks)
    offset = int(response_offset_ticks)
    if window < 3 or window % 2 == 0:
        raise ValueError("stationary_window_ticks must be odd and >= 3")
    if offset < 1:
        raise ValueError("response_offset_ticks must be positive")
    positive, negative = _thresholds(positive_threshold, negative_threshold)

    inactive_qvel: list[np.ndarray] = []
    signed_future: list[list[np.ndarray]] = [[] for _ in AXIS_NAMES]
    for sequence in sequences:
        _validate_sequence(sequence)
        effective, direction = _effective_direction(
            sequence.command, positive=positive, negative=negative
        )
        inactive = sequence.valid_mask & ~np.any(effective, axis=1)
        centered = _centered_all_true(inactive, window)
        if np.any(centered):
            inactive_qvel.append(sequence.qvel[centered])

        limit = len(sequence.command) - offset
        for axis_index in range(len(AXIS_NAMES)):
            mask = (
                effective[:limit, axis_index]
                & sequence.valid_mask[:limit]
                & sequence.valid_mask[offset:]
            )
            if not np.any(mask):
                continue
            command_direction = direction[:limit, axis_index][mask].astype(np.float32)
            future_qvel = sequence.qvel[offset:, axis_index][mask]
            signed_future[axis_index].append(command_direction * future_qvel)

    if not inactive_qvel:
        raise ValueError("no centered all-axis inactive dwell samples for calibration")
    stationary = np.concatenate(inactive_qvel, axis=0)
    qvel_noise: list[float] = []
    for axis_index in range(len(AXIS_NAMES)):
        values = stationary[:, axis_index]
        median = float(np.median(values))
        robust_sigma = 1.4826 * float(np.median(np.abs(values - median)))
        qvel_noise.append(
            max(float(noise_floor), float(noise_sigma_multiplier) * robust_sigma)
        )

    response_sign: list[int] = []
    sample_counts: list[int] = []
    agreements: list[float] = []
    for axis_index, parts in enumerate(signed_future):
        if not parts:
            raise ValueError(f"no active train samples for axis {AXIS_NAMES[axis_index]}")
        values = np.concatenate(parts)
        sample_counts.append(int(values.size))
        median = float(np.median(values))
        if median == 0.0:
            raise ValueError(
                f"cannot identify response direction for axis {AXIS_NAMES[axis_index]}"
            )
        sign = 1 if median > 0.0 else -1
        response_sign.append(sign)
        moving = np.abs(values) > qvel_noise[axis_index]
        agreements.append(
            float(np.mean(sign * values[moving] > 0.0)) if np.any(moving) else 0.0
        )

    return ResponseCalibration(
        qvel_direction_sign=tuple(response_sign),
        qvel_noise=tuple(qvel_noise),
        inactive_sample_count=int(stationary.shape[0]),
        active_sample_count_by_axis=tuple(sample_counts),
        direction_agreement_by_axis=tuple(agreements),
        stationary_window_ticks=window,
        response_offset_ticks=offset,
    )


def extract_response_events(
    sequences: Sequence[ResponseSequence],
    *,
    calibration: ResponseCalibration,
    positive_threshold: Sequence[float],
    negative_threshold: Sequence[float],
    response_horizons: Sequence[int] = DEFAULT_HORIZONS,
    minimum_pre_idle_ticks: int = 4,
    minimum_sustain_ticks: int = 4,
    baseline_ticks: int = 3,
) -> list[dict[str, Any]]:
    """Extract event-level historical response evidence."""

    positive, negative = _thresholds(positive_threshold, negative_threshold)
    horizons = _horizons(response_horizons)
    if minimum_pre_idle_ticks < 0 or minimum_sustain_ticks < 1 or baseline_ticks < 1:
        raise ValueError("invalid event dwell configuration")
    response_sign = np.asarray(calibration.qvel_direction_sign, dtype=np.int8)
    qvel_noise = np.asarray(calibration.qvel_noise, dtype=np.float32)
    rows: list[dict[str, Any]] = []

    for sequence in sequences:
        _validate_sequence(sequence)
        effective, direction = _effective_direction(
            sequence.command, positive=positive, negative=negative
        )
        n_steps = len(sequence.command)
        for timestep in range(n_steps):
            if not sequence.valid_mask[timestep]:
                continue
            for axis_index, axis in enumerate(AXIS_NAMES):
                if not effective[timestep, axis_index]:
                    continue
                sign = int(direction[timestep, axis_index])
                previous_same = (
                    timestep > 0
                    and sequence.valid_mask[timestep - 1]
                    and effective[timestep - 1, axis_index]
                    and int(direction[timestep - 1, axis_index]) == sign
                )
                if previous_same:
                    continue

                pre_idle = _preceding_idle_ticks(
                    effective[:, axis_index], sequence.valid_mask, timestep
                )
                sustain = _same_direction_run(
                    effective[:, axis_index],
                    direction[:, axis_index],
                    sequence.valid_mask,
                    timestep,
                    sign,
                )
                baseline_start = max(0, timestep - baseline_ticks + 1)
                baseline_valid = bool(
                    timestep - baseline_start + 1 == baseline_ticks
                    and np.all(sequence.valid_mask[baseline_start : timestep + 1])
                )
                projected_baseline = (
                    float(sign * response_sign[axis_index])
                    * sequence.qvel[baseline_start : timestep + 1, axis_index]
                )
                baseline = (
                    float(np.median(projected_baseline))
                    if baseline_valid and projected_baseline.size
                    else float("nan")
                )
                if not np.isfinite(baseline):
                    baseline_state = "unknown"
                elif baseline > qvel_noise[axis_index]:
                    baseline_state = "already_moving_same"
                elif baseline < -qvel_noise[axis_index]:
                    baseline_state = "already_moving_opposite"
                else:
                    baseline_state = "rest"

                threshold = positive[axis_index] if sign > 0 else negative[axis_index]
                command_value = float(sequence.command[timestep, axis_index])
                row: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_episode_id": int(sequence.dataset_episode_id),
                    "source_episode_id": int(sequence.source_episode_id),
                    "split": sequence.split,
                    "timestep": int(timestep),
                    "axis_index": axis_index,
                    "axis": axis,
                    "direction": "pos" if sign > 0 else "neg",
                    "command": command_value,
                    "command_magnitude_ratio": abs(command_value) / float(threshold),
                    "qpos": float(sequence.qpos[timestep, axis_index]),
                    "pre_idle_ticks": int(pre_idle),
                    "sustain_ticks": int(sustain),
                    "baseline_projected_qvel": (
                        baseline if np.isfinite(baseline) else None
                    ),
                    "baseline_state": baseline_state,
                    "eligible_from_rest": bool(
                        baseline_state == "rest"
                        and pre_idle >= minimum_pre_idle_ticks
                        and sustain >= minimum_sustain_ticks
                    ),
                    "direction_switch": bool(
                        timestep > 0
                        and sequence.valid_mask[timestep - 1]
                        and effective[timestep - 1, axis_index]
                        and int(direction[timestep - 1, axis_index]) != sign
                    ),
                }
                first_response: int | None = None
                for horizon in horizons:
                    end = timestep + horizon + 1
                    complete = end <= n_steps and bool(
                        np.all(sequence.valid_mask[timestep + 1 : end])
                    )
                    projected = (
                        float(sign * response_sign[axis_index])
                        * sequence.qvel[timestep + 1 : end, axis_index]
                    )
                    if complete and projected.size:
                        peak = float(np.max(projected))
                        opposite_peak = float(np.max(-projected))
                        observed = peak > float(qvel_noise[axis_index])
                        opposite = opposite_peak > float(qvel_noise[axis_index])
                        qpos_projected = (
                            float(sign * response_sign[axis_index])
                            * (
                                sequence.qpos[timestep + 1 : end, axis_index]
                                - sequence.qpos[timestep, axis_index]
                            )
                        )
                        qpos_peak = float(np.max(qpos_projected))
                        if observed and first_response is None:
                            first_response = horizon
                        row[f"response_{horizon}t"] = int(observed)
                        row[f"opposite_motion_{horizon}t"] = int(opposite)
                        row[f"qvel_peak_{horizon}t"] = peak
                        row[f"qpos_delta_peak_{horizon}t"] = qpos_peak
                    else:
                        row[f"response_{horizon}t"] = -1
                        row[f"opposite_motion_{horizon}t"] = -1
                        row[f"qvel_peak_{horizon}t"] = None
                        row[f"qpos_delta_peak_{horizon}t"] = None
                    row[f"horizon_complete_{horizon}t"] = complete
                row["first_response_horizon_ticks"] = first_response
                rows.append(row)
    return rows


def fit_response_envelope(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    response_horizons: Sequence[int] = DEFAULT_HORIZONS,
    minimum_supported_events: int = 10,
    minimum_weak_events: int = 3,
) -> dict[str, Any]:
    """Fit train-only response cells and qpos bins."""

    if not train_rows or any(row.get("split") != "train" for row in train_rows):
        raise ValueError("response envelope must be fit from non-empty train rows only")
    horizons = _horizons(response_horizons)
    qpos_edges: dict[str, list[float]] = {}
    for axis in AXIS_NAMES:
        values = np.asarray(
            [
                float(row["qpos"])
                for row in train_rows
                if row["axis"] == axis and bool(row["eligible_from_rest"])
            ],
            dtype=np.float64,
        )
        if values.size < minimum_weak_events:
            qpos_edges[axis] = []
        else:
            qpos_edges[axis] = [
                float(value) for value in np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
            ]

    grouped: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in train_rows:
        if not bool(row["eligible_from_rest"]):
            continue
        key = _cell_key(row, qpos_edges=qpos_edges)
        grouped[key].append(row)

    cells: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        axis, direction, magnitude_bin, qpos_bin = key
        count = len(rows)
        status = (
            "supported"
            if count >= minimum_supported_events
            else "weak_support"
            if count >= minimum_weak_events
            else "out_of_support"
        )
        cell: dict[str, Any] = {
            "axis": axis,
            "direction": direction,
            "magnitude_bin": magnitude_bin,
            "qpos_bin": qpos_bin,
            "event_count": count,
            "support_status": status,
        }
        for horizon in horizons:
            labels = [
                int(row[f"response_{horizon}t"])
                for row in rows
                if int(row[f"response_{horizon}t"]) >= 0
            ]
            cell[f"complete_count_{horizon}t"] = len(labels)
            cell[f"response_probability_{horizon}t"] = (
                float(np.mean(labels)) if labels else None
            )
        cells.append(cell)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_split": "train_only",
        "magnitude_ratio_edges": list(MAGNITUDE_RATIO_EDGES),
        "qpos_edges_by_axis": qpos_edges,
        "minimum_supported_events": int(minimum_supported_events),
        "minimum_weak_events": int(minimum_weak_events),
        "response_horizons": list(horizons),
        "eligible_train_event_count": int(
            sum(bool(row["eligible_from_rest"]) for row in train_rows)
        ),
        "cells": cells,
    }


def evaluate_response_envelope(
    rows: Sequence[Mapping[str, Any]],
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate historical validation events without inventing support."""

    qpos_edges = envelope["qpos_edges_by_axis"]
    cell_lookup = {
        (
            str(cell["axis"]),
            str(cell["direction"]),
            int(cell["magnitude_bin"]),
            int(cell["qpos_bin"]),
        ): cell
        for cell in envelope["cells"]
    }
    horizons = tuple(int(value) for value in envelope["response_horizons"])
    event_rows: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        key = _cell_key(source, qpos_edges=qpos_edges)
        cell = cell_lookup.get(key)
        status = "out_of_support" if cell is None else str(cell["support_status"])
        usable_cell = cell if status in {"supported", "weak_support"} else None
        row["envelope_support_status"] = status
        row["envelope_train_event_count"] = 0 if cell is None else int(cell["event_count"])
        for horizon in horizons:
            row[f"predicted_response_probability_{horizon}t"] = (
                None
                if usable_cell is None
                else usable_cell[f"response_probability_{horizon}t"]
            )
        event_rows.append(row)

    eligible = [row for row in event_rows if bool(row["eligible_from_rest"])]
    status_counts = Counter(row["envelope_support_status"] for row in eligible)
    brier: dict[str, float | None] = {}
    for horizon in horizons:
        pairs = [
            (
                float(row[f"predicted_response_probability_{horizon}t"]),
                int(row[f"response_{horizon}t"]),
            )
            for row in eligible
            if row[f"predicted_response_probability_{horizon}t"] is not None
            and int(row[f"response_{horizon}t"]) >= 0
        ]
        brier[str(horizon)] = (
            float(np.mean([(probability - label) ** 2 for probability, label in pairs]))
            if pairs
            else None
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "capability_boundaries": {
            "directly_measures": (
                "historical response of causally aligned operator commands in "
                "train-supported command/qpos cells"
            ),
            "does_not_measure": [
                "response to an unsent model command",
                "hydraulic pressure or terrain causality",
                "closed-loop task progress",
            ],
            "out_of_support_policy": "unknown, never converted to success or failure",
        },
        "event_count": len(event_rows),
        "eligible_from_rest_event_count": len(eligible),
        "support_status_counts": dict(sorted(status_counts.items())),
        "brier_score_by_horizon": brier,
        "events": event_rows,
    }


def query_response_envelope(
    *,
    axis: str,
    direction: str,
    command: float,
    qpos: float,
    positive_threshold: Sequence[float],
    negative_threshold: Sequence[float],
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    """Query historical response support for one unsent command proposal."""

    if axis not in AXIS_NAMES:
        raise ValueError(f"unknown axis: {axis}")
    if direction not in {"pos", "neg"}:
        raise ValueError(f"direction must be pos/neg, got {direction}")
    positive, negative = _thresholds(positive_threshold, negative_threshold)
    axis_index = AXIS_NAMES.index(axis)
    threshold = positive[axis_index] if direction == "pos" else negative[axis_index]
    ratio = abs(float(command)) / float(threshold)
    query = {
        "axis": axis,
        "direction": direction,
        "command_magnitude_ratio": ratio,
        "qpos": float(qpos),
    }
    key = _cell_key(query, qpos_edges=envelope["qpos_edges_by_axis"])
    cell = next(
        (
            candidate
            for candidate in envelope["cells"]
            if (
                str(candidate["axis"]),
                str(candidate["direction"]),
                int(candidate["magnitude_bin"]),
                int(candidate["qpos_bin"]),
            )
            == key
        ),
        None,
    )
    status = "out_of_support" if cell is None else str(cell["support_status"])
    usable = cell if status in {"supported", "weak_support"} else None
    return {
        "support_status": status,
        "train_event_count": 0 if cell is None else int(cell["event_count"]),
        "magnitude_ratio": ratio,
        "magnitude_bin": key[2],
        "qpos_bin": key[3],
        "predicted_response_probability_by_horizon": {
            str(horizon): (
                None
                if usable is None
                else usable[f"response_probability_{int(horizon)}t"]
            )
            for horizon in envelope["response_horizons"]
        },
        "claim_boundary": (
            "historical operator-command support only; this model command was not sent"
        ),
    }


def summarize_response_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    response_horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    """Return episode/axis-aware descriptive counts."""

    horizons = _horizons(response_horizons)
    groups: dict[str, Any] = {}
    for axis in AXIS_NAMES:
        for direction in ("pos", "neg"):
            selected = [
                row
                for row in rows
                if row["axis"] == axis and row["direction"] == direction
            ]
            eligible = [row for row in selected if bool(row["eligible_from_rest"])]
            group: dict[str, Any] = {
                "event_count": len(selected),
                "eligible_from_rest_event_count": len(eligible),
                "episode_count": len({int(row["dataset_episode_id"]) for row in selected}),
                "baseline_state_counts": dict(
                    sorted(Counter(str(row["baseline_state"]) for row in selected).items())
                ),
            }
            for horizon in horizons:
                labels = [
                    int(row[f"response_{horizon}t"])
                    for row in eligible
                    if int(row[f"response_{horizon}t"]) >= 0
                ]
                group[f"response_probability_{horizon}t"] = (
                    float(np.mean(labels)) if labels else None
                )
                group[f"complete_count_{horizon}t"] = len(labels)
            groups[f"{axis}:{direction}"] = group
    return {
        "schema_version": SCHEMA_VERSION,
        "event_count": len(rows),
        "episode_count": len({int(row["dataset_episode_id"]) for row in rows}),
        "eligible_from_rest_event_count": int(
            sum(bool(row["eligible_from_rest"]) for row in rows)
        ),
        "groups": groups,
    }


def calibration_to_dict(calibration: ResponseCalibration) -> dict[str, Any]:
    payload = asdict(calibration)
    payload["schema_version"] = SCHEMA_VERSION
    payload["axis_order"] = list(AXIS_NAMES)
    payload["qvel_direction_semantics"] = (
        "measured_qvel_sign = command_sign * qvel_direction_sign"
    )
    return payload


def _validate_sequence(sequence: ResponseSequence) -> None:
    n_steps = len(sequence.command)
    if sequence.command.shape != (n_steps, 4):
        raise ValueError("command must have shape (T, 4)")
    if sequence.qpos.shape != (n_steps, 4) or sequence.qvel.shape != (n_steps, 4):
        raise ValueError("qpos/qvel must have shape (T, 4)")
    if sequence.valid_mask.shape != (n_steps,):
        raise ValueError("valid_mask must have shape (T,)")


def _thresholds(
    positive_threshold: Sequence[float], negative_threshold: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(positive_threshold, dtype=np.float32)
    negative = np.asarray(negative_threshold, dtype=np.float32)
    if positive.shape != (4,) or negative.shape != (4,):
        raise ValueError("deadzone thresholds must have four entries")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("deadzone thresholds must be finite")
    if np.any(positive <= 0.0) or np.any(negative <= 0.0):
        raise ValueError("deadzone thresholds must be positive")
    return positive, negative


def _effective_direction(
    command: np.ndarray, *, positive: np.ndarray, negative: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positive_hit = command >= positive.reshape(1, 4)
    negative_hit = command <= -negative.reshape(1, 4)
    effective = positive_hit | negative_hit
    direction = np.zeros_like(command, dtype=np.int8)
    direction[positive_hit] = 1
    direction[negative_hit] = -1
    return effective, direction


def _centered_all_true(mask: np.ndarray, window: int) -> np.ndarray:
    counts = np.convolve(mask.astype(np.int32), np.ones(window, dtype=np.int32), mode="same")
    centered = counts == window
    radius = window // 2
    centered[:radius] = False
    centered[len(centered) - radius :] = False
    return centered


def _preceding_idle_ticks(
    effective: np.ndarray, valid: np.ndarray, timestep: int
) -> int:
    count = 0
    index = timestep - 1
    while index >= 0 and valid[index] and not effective[index]:
        count += 1
        index -= 1
    return count


def _same_direction_run(
    effective: np.ndarray,
    direction: np.ndarray,
    valid: np.ndarray,
    timestep: int,
    sign: int,
) -> int:
    count = 0
    for index in range(timestep, len(effective)):
        if not valid[index] or not effective[index] or int(direction[index]) != sign:
            break
        count += 1
    return count


def _horizons(values: Sequence[int]) -> tuple[int, ...]:
    horizons = tuple(sorted({int(value) for value in values}))
    if not horizons or horizons[0] < 1:
        raise ValueError("response horizons must be positive")
    return horizons


def _magnitude_bin(value: float) -> int:
    for index, upper in enumerate(MAGNITUDE_RATIO_EDGES[1:]):
        if value < upper:
            return index
    return len(MAGNITUDE_RATIO_EDGES) - 1


def _qpos_bin(value: float, edges: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), value, side="right"))


def _cell_key(
    row: Mapping[str, Any], *, qpos_edges: Mapping[str, Sequence[float]]
) -> tuple[str, str, int, int]:
    axis = str(row["axis"])
    return (
        axis,
        str(row["direction"]),
        _magnitude_bin(float(row["command_magnitude_ratio"])),
        _qpos_bin(float(row["qpos"]), qpos_edges[axis]),
    )
