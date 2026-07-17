"""Offline command-to-motion response labels from existing real episodes.

This owner deliberately does not modify the source HDF5 episodes.  It aligns
the latest causally prior final command, marks direct-domain deadzone crossings,
and emits conservative qvel-response labels for a bounded future horizon.  A
full-horizon no-response label is a review candidate, not a claim that the
hydraulics failed: transport acknowledgement is kept separate from motion.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.data.deadzone_intent_labels import AXIS_NAMES
from testbed.data.execution_feedback import (
    CausalCommandAlignment,
    align_causal_previous_commands,
    sha256_file,
)

EXECUTION_RESPONSE_SCHEMA_VERSION = 1
DEFAULT_RESPONSE_HORIZONS = (1, 2, 4, 8, 20)
DEFAULT_SUPPORTED_AXES = ("swing", "boom", "bucket")
LABEL_CONTRACT = "direct_command_qvel_response_v1"


@dataclass(frozen=True)
class ExecutionResponseEpisode:
    """Arrays and event rows for one resampled episode."""

    episode_id: int
    alignment: CausalCommandAlignment
    qpos: np.ndarray
    qvel: np.ndarray
    command_age_ns: np.ndarray
    effective_mask: np.ndarray
    direction: np.ndarray
    event_mask: np.ndarray
    response_mask: np.ndarray
    opposite_motion_mask: np.ndarray
    horizon_complete: np.ndarray
    qvel_peak: np.ndarray
    qvel_opposite_peak: np.ndarray
    qpos_delta_peak: np.ndarray
    event_rows: list[dict[str, Any]]
    resampled_path: Path
    raw_source_path: Path


def classify_effective_commands(
    commands: np.ndarray,
    *,
    positive_threshold: Sequence[float],
    negative_threshold: Sequence[float],
    supported_axes: Sequence[str] = DEFAULT_SUPPORTED_AXES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return direct-domain effective and signed direction masks.

    Unsupported axes are intentionally left false.  In the current task the
    stick axis is a structural zero axis, not a missing label.
    """

    command = np.asarray(commands, dtype=np.float32)
    if command.ndim != 2 or command.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"commands must have shape (T, 4), got {command.shape}")
    positive = _axis_vector(positive_threshold, name="positive_threshold")
    negative = _axis_vector(negative_threshold, name="negative_threshold")
    supported = _supported_axis_mask(supported_axes)
    effective = np.zeros_like(command, dtype=bool)
    direction = np.zeros_like(command, dtype=np.int8)
    positive_hit = command >= positive.reshape(1, -1)
    negative_hit = command <= -negative.reshape(1, -1)
    effective[:, supported] = (
        positive_hit[:, supported] | negative_hit[:, supported]
    )
    direction[positive_hit & effective] = 1
    direction[negative_hit & effective] = -1
    return effective, direction


def response_label(
    qvel_window: np.ndarray,
    *,
    direction: int,
    qvel_noise: float,
    complete: bool,
) -> tuple[int, float]:
    """Return ``-1`` unknown, ``0`` no response, or ``1`` response.

    The returned peak is signed in the requested command direction.  A
    truncated or invalid future window is unknown and cannot become a stalled
    training label.
    """

    values = np.asarray(qvel_window, dtype=np.float32).reshape(-1)
    if direction not in {-1, 1}:
        raise ValueError(f"direction must be +/-1, got {direction}")
    if not complete or values.size == 0 or not np.isfinite(values).all():
        return -1, float("nan")
    peak = float(np.max(float(direction) * values))
    return int(peak > float(qvel_noise)), peak


def build_execution_response_episode(
    *,
    resampled_path: str | Path,
    positive_threshold: Sequence[float],
    negative_threshold: Sequence[float],
    qvel_noise: Sequence[float],
    supported_axes: Sequence[str] = DEFAULT_SUPPORTED_AXES,
    response_horizons: Sequence[int] = DEFAULT_RESPONSE_HORIZONS,
) -> ExecutionResponseEpisode:
    """Build conservative response labels from one existing HDF5 episode."""

    resampled = Path(resampled_path).expanduser().resolve()
    if not resampled.is_file():
        raise FileNotFoundError(f"resampled episode not found: {resampled}")
    horizons = _horizons(response_horizons)
    positive = _axis_vector(positive_threshold, name="positive_threshold")
    negative = _axis_vector(negative_threshold, name="negative_threshold")
    noise = _axis_vector(qvel_noise, name="qvel_noise")

    with h5py.File(resampled, "r") as episode:
        raw_source = _resolve_raw_source(episode, resampled)
        observation_ts = _required_dataset(
            episode, "diagnostics/source_observation_timestamp_ns"
        )
        train_exclude = _required_dataset(
            episode, "diagnostics/train_exclude_mask"
        )
        source_gap = _required_dataset(
            episode, "diagnostics/source_time_gap_exceeds_threshold"
        )
        qpos = np.asarray(episode["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(episode["observations/qvel"][()], dtype=np.float32)
        episode_id = _episode_id(episode["metadata"].attrs.get("episode_id"))

    with h5py.File(raw_source, "r") as raw:
        raw_command = _required_dataset(raw, "diagnostics/commanded_action")
        raw_send_ts = _required_dataset(
            raw, "diagnostics/action_send_timestamp_ns"
        )

    if qpos.shape != qvel.shape or qvel.ndim != 2 or qvel.shape[1] != 4:
        raise ValueError(f"qpos/qvel must both have shape (T, 4), got {qpos.shape} and {qvel.shape}")
    alignment = align_causal_previous_commands(
        observation_timestamp_ns=observation_ts,
        raw_commanded_action=raw_command,
        raw_action_send_timestamp_ns=raw_send_ts,
        train_exclude_mask=train_exclude,
        source_time_gap_exceeds_threshold=source_gap,
    )
    if len(alignment.previous_final_command) != qvel.shape[0]:
        raise ValueError("causal command alignment length does not match qvel")

    effective, direction = classify_effective_commands(
        alignment.previous_final_command,
        positive_threshold=positive,
        negative_threshold=negative,
        supported_axes=supported_axes,
    )
    valid_frame = (
        alignment.valid_mask
        & ~alignment.reset_mask
        & np.isfinite(qpos).all(axis=1)
        & np.isfinite(qvel).all(axis=1)
    )
    n_steps = int(alignment.previous_final_command.shape[0])
    command_age_ns = np.full(n_steps, -1, dtype=np.int64)
    command_age_ns[alignment.valid_mask] = (
        alignment.observation_timestamp_ns[alignment.valid_mask]
        - alignment.command_send_timestamp_ns[alignment.valid_mask]
    )
    event_mask = np.zeros_like(effective, dtype=bool)
    response_mask = np.full(
        (len(horizons), *effective.shape), -1, dtype=np.int8
    )
    opposite_motion_mask = np.full(
        (len(horizons), *effective.shape), -1, dtype=np.int8
    )
    horizon_complete = np.zeros(
        (len(horizons), *effective.shape), dtype=bool
    )
    qvel_peak = np.full(
        (len(horizons), *effective.shape), np.nan, dtype=np.float32
    )
    qvel_opposite_peak = np.full(
        (len(horizons), *effective.shape), np.nan, dtype=np.float32
    )
    qpos_delta_peak = np.full(
        (len(horizons), *effective.shape), np.nan, dtype=np.float32
    )
    event_rows: list[dict[str, Any]] = []

    for timestep in range(n_steps):
        if not valid_frame[timestep]:
            continue
        for axis_index, axis_name in enumerate(AXIS_NAMES):
            if not effective[timestep, axis_index]:
                continue
            sign = int(direction[timestep, axis_index])
            previous_same = (
                timestep > 0
                and valid_frame[timestep - 1]
                and effective[timestep - 1, axis_index]
                and int(direction[timestep - 1, axis_index]) == sign
            )
            if previous_same:
                continue
            event_mask[timestep, axis_index] = True
            row: dict[str, Any] = {
                "episode_id": episode_id,
                "timestep": timestep,
                "axis_index": axis_index,
                "axis": axis_name,
                "direction": "pos" if sign > 0 else "neg",
                "command": float(alignment.previous_final_command[timestep, axis_index]),
                "command_age_ns": int(command_age_ns[timestep]),
                "direction_switch": bool(
                    timestep > 0
                    and valid_frame[timestep - 1]
                    and effective[timestep - 1, axis_index]
                    and int(direction[timestep - 1, axis_index]) != sign
                ),
            }
            for horizon_index, horizon in enumerate(horizons):
                end = timestep + horizon + 1
                complete = end <= n_steps and bool(
                    np.all(valid_frame[timestep + 1 : end])
                )
                horizon_complete[horizon_index, timestep, axis_index] = complete
                qvel_window = qvel[timestep + 1 : end, axis_index]
                label, peak = response_label(
                    qvel_window,
                    direction=sign,
                    qvel_noise=float(noise[axis_index]),
                    complete=complete,
                )
                response_mask[horizon_index, timestep, axis_index] = label
                qvel_peak[horizon_index, timestep, axis_index] = peak
                opposite_peak = float("nan")
                opposite_motion = -1
                if complete and qvel_window.size and np.isfinite(qvel_window).all():
                    opposite_peak = float(np.max(-float(sign) * qvel_window))
                    opposite_motion = int(opposite_peak > float(noise[axis_index]))
                opposite_motion_mask[
                    horizon_index, timestep, axis_index
                ] = opposite_motion
                qvel_opposite_peak[
                    horizon_index, timestep, axis_index
                ] = opposite_peak
                if complete:
                    qpos_window = qpos[timestep + 1 : end, axis_index]
                    signed_delta = float(sign) * (
                        qpos_window - qpos[timestep, axis_index]
                    )
                    qpos_delta_peak[horizon_index, timestep, axis_index] = float(
                        np.max(signed_delta)
                    )
                row[f"response_{horizon}t"] = label
                row[f"qvel_peak_{horizon}t"] = (
                    None if not np.isfinite(peak) else peak
                )
                row[f"opposite_motion_{horizon}t"] = opposite_motion
                row[f"qvel_opposite_peak_{horizon}t"] = (
                    None
                    if not np.isfinite(opposite_peak)
                    else opposite_peak
                )
                row[f"qpos_delta_peak_{horizon}t"] = (
                    None
                    if not np.isfinite(qpos_delta_peak[horizon_index, timestep, axis_index])
                    else float(qpos_delta_peak[horizon_index, timestep, axis_index])
                )
            event_rows.append(row)

    return ExecutionResponseEpisode(
        episode_id=episode_id,
        alignment=alignment,
        qpos=qpos,
        qvel=qvel,
        command_age_ns=command_age_ns,
        effective_mask=effective,
        direction=direction,
        event_mask=event_mask,
        response_mask=response_mask,
        opposite_motion_mask=opposite_motion_mask,
        horizon_complete=horizon_complete,
        qvel_peak=qvel_peak,
        qvel_opposite_peak=qvel_opposite_peak,
        qpos_delta_peak=qpos_delta_peak,
        event_rows=event_rows,
        resampled_path=resampled,
        raw_source_path=raw_source,
    )


def write_execution_response_episode(
    result: ExecutionResponseEpisode,
    *,
    output_dir: str | Path,
    positive_threshold: Sequence[float],
    negative_threshold: Sequence[float],
    qvel_noise: Sequence[float],
    supported_axes: Sequence[str],
    response_horizons: Sequence[int],
    qvel_noise_provenance: str,
) -> dict[str, Any]:
    """Write one immutable-source NPZ and return a manifest record."""

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    episode = result.episode_id
    npz_path = out / f"episode_{episode}.execution_response.npz"
    temporary = npz_path.with_name(f".{npz_path.name}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(
            file,
            schema_version=np.asarray(EXECUTION_RESPONSE_SCHEMA_VERSION, dtype=np.int64),
            label_contract=np.asarray(LABEL_CONTRACT),
            episode_id=np.asarray(str(episode)),
            previous_final_command=result.alignment.previous_final_command,
            command_send_timestamp_ns=result.alignment.command_send_timestamp_ns,
            observation_timestamp_ns=result.alignment.observation_timestamp_ns,
            command_age_ns=result.command_age_ns,
            valid_mask=result.alignment.valid_mask,
            reset_mask=result.alignment.reset_mask,
            train_exclude_mask=result.alignment.train_exclude_mask,
            source_time_gap_exceeds_threshold=result.alignment.source_time_gap_exceeds_threshold,
            qpos=result.qpos,
            qvel=result.qvel,
            effective_mask=result.effective_mask,
            direction=result.direction,
            event_mask=result.event_mask,
            response_mask=result.response_mask,
            opposite_motion_mask=result.opposite_motion_mask,
            horizon_complete=result.horizon_complete,
            qvel_peak=result.qvel_peak,
            qvel_opposite_peak=result.qvel_opposite_peak,
            qpos_delta_peak=result.qpos_delta_peak,
            resampled_path=np.asarray(str(result.resampled_path)),
            raw_source_path=np.asarray(str(result.raw_source_path)),
        )
    temporary.replace(npz_path)

    horizons = _horizons(response_horizons)
    event_counts = {
        axis: int(np.count_nonzero(result.event_mask[:, axis_index]))
        for axis_index, axis in enumerate(AXIS_NAMES)
    }
    response_counts: dict[str, dict[str, int]] = {}
    candidate_stalled: dict[str, int] = {}
    opposite_motion_counts: dict[str, dict[str, int]] = {}
    for horizon_index, horizon in enumerate(horizons):
        response_counts[str(horizon)] = {
            axis: int(
                np.count_nonzero(
                    result.response_mask[horizon_index, :, axis_index] == 1
                )
            )
            for axis_index, axis in enumerate(AXIS_NAMES)
        }
        candidate_stalled[str(horizon)] = int(
            np.count_nonzero(
                (result.response_mask[horizon_index] == 0)
                & (result.opposite_motion_mask[horizon_index] == 0)
            )
        )
        opposite_motion_counts[str(horizon)] = {
            axis: int(
                np.count_nonzero(
                    result.opposite_motion_mask[horizon_index, :, axis_index] == 1
                )
            )
            for axis_index, axis in enumerate(AXIS_NAMES)
        }
    record = {
        "episode_id": episode,
        "resampled_path": str(result.resampled_path),
        "resampled_sha256": sha256_file(result.resampled_path),
        "raw_source_path": str(result.raw_source_path),
        "raw_source_sha256": sha256_file(result.raw_source_path),
        "sidecar_path": str(npz_path),
        "sidecar_sha256": sha256_file(npz_path),
        "n_steps": int(result.alignment.previous_final_command.shape[0]),
        "valid_observation_count": int(np.count_nonzero(result.alignment.valid_mask)),
        "event_counts": event_counts,
        "response_counts_by_horizon": response_counts,
        "candidate_stalled_counts_by_horizon": candidate_stalled,
        "opposite_motion_counts_by_horizon": opposite_motion_counts,
        "unsupported_axes": [axis for axis in AXIS_NAMES if axis not in supported_axes],
        "label_contract": LABEL_CONTRACT,
        "response_horizons": list(horizons),
        "positive_threshold": [float(value) for value in positive_threshold],
        "negative_threshold": [float(value) for value in negative_threshold],
        "qvel_noise": [float(value) for value in qvel_noise],
        "qvel_noise_provenance": qvel_noise_provenance,
    }
    return record


def write_event_tables(
    rows: Sequence[Mapping[str, Any]],
    *,
    jsonl_path: str | Path,
    csv_path: str | Path,
) -> None:
    """Write review-friendly JSONL and CSV event tables."""

    jsonl = Path(jsonl_path)
    csv_file = Path(csv_path)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_response_latency(
    rows: Sequence[Mapping[str, Any]],
    *,
    response_horizons: Sequence[int],
) -> dict[str, Any]:
    """Summarize first same-direction response latency from event rows."""

    horizons = _horizons(response_horizons)
    grouped: dict[tuple[str, str], list[int]] = {}
    no_response = 0
    for row in rows:
        first: int | None = None
        for horizon in horizons:
            if int(row.get(f"response_{horizon}t", -1)) == 1:
                first = horizon
                break
        if first is None:
            no_response += 1
            continue
        key = (str(row["axis"]), str(row["direction"]))
        grouped.setdefault(key, []).append(first)
    groups: dict[str, Any] = {}
    for (axis, direction), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        counts = {
            str(horizon): int(np.count_nonzero(array == horizon))
            for horizon in horizons
            if np.count_nonzero(array == horizon)
        }
        groups[f"{axis}:{direction}"] = {
            "count": int(array.size),
            "first_response_tick_counts": counts,
            "median_ticks": float(np.percentile(array, 50)),
            "p95_ticks": float(np.percentile(array, 95)),
        }
    return {
        "response_horizons": list(horizons),
        "event_rows": len(rows),
        "same_direction_response_rows": int(len(rows) - no_response),
        "no_same_direction_response_rows": int(no_response),
        "groups": groups,
    }


def _resolve_raw_source(episode: h5py.File, resampled: Path) -> Path:
    raw = episode["metadata"].attrs.get("source_dataset_path")
    if isinstance(raw, bytes):
        raw = raw.decode()
    if not raw:
        raise ValueError(f"missing metadata.source_dataset_path: {resampled}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (resampled.parent / path).resolve()
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"raw source episode not found: {path}")
    return path


def _required_dataset(file: h5py.File, path: str) -> np.ndarray:
    if path not in file:
        raise ValueError(f"episode is missing required dataset {path!r}")
    return np.asarray(file[path][()])


def _axis_vector(value: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (len(AXIS_NAMES),) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain four finite values")
    return arr


def _supported_axis_mask(axes: Sequence[str]) -> np.ndarray:
    names = {str(axis) for axis in axes}
    unknown = names - set(AXIS_NAMES)
    if unknown:
        raise ValueError(f"unknown supported axes: {sorted(unknown)}")
    return np.asarray([axis in names for axis in AXIS_NAMES], dtype=bool)


def _horizons(value: Sequence[int]) -> tuple[int, ...]:
    horizons = tuple(sorted({int(item) for item in value}))
    if not horizons or horizons[0] <= 0:
        raise ValueError("response_horizons must contain positive integers")
    return horizons


def _episode_id(value: Any) -> int:
    text = str(value)
    if text.startswith("episode_"):
        text = text.split("_", 1)[1]
    try:
        episode = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid episode id {value!r}") from exc
    if episode < 0:
        raise ValueError("episode id must be nonnegative")
    return episode
