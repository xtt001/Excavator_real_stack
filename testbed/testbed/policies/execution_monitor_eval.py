"""Split-safe offline replay helpers for :mod:`execution_monitor`.

This module consumes immutable ``direct_command_qvel_response_v1`` NPZ
sidecars and replays the runtime monitor without synthesising actions or
feedback.  It is an evaluation/calibration audit only: the existing teleop
sidecars do not contain policy intent, operator correction, or confirmed
failed-actuation labels, so retry precision is reported as not estimable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from testbed.data.deadzone_intent_labels import AXIS_NAMES
from testbed.policies.execution_monitor import (
    ExecutionMonitor,
    ExecutionMonitorConfig,
    FeedbackSample,
    SentCommand,
)


@dataclass(frozen=True)
class MonitorEpisodeSummary:
    """One episode's response-replay counts and sidecar agreement."""

    episode_id: int
    split: str
    sidecar_path: str
    event_count: int
    responded_count: int
    stalled_count: int
    unknown_count: int
    sidecar_response_count: int
    sidecar_stalled_candidate_count: int
    response_mismatch_count: int
    incomplete_window_count: int
    by_axis: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": int(self.episode_id),
            "split": self.split,
            "sidecar_path": self.sidecar_path,
            "event_count": int(self.event_count),
            "responded_count": int(self.responded_count),
            "stalled_count": int(self.stalled_count),
            "unknown_count": int(self.unknown_count),
            "sidecar_response_count": int(self.sidecar_response_count),
            "sidecar_stalled_candidate_count": int(
                self.sidecar_stalled_candidate_count
            ),
            "response_mismatch_count": int(self.response_mismatch_count),
            "incomplete_window_count": int(self.incomplete_window_count),
            "by_axis": {
                axis: {key: int(value) for key, value in values.items()}
                for axis, values in self.by_axis.items()
            },
        }


def evaluate_response_sidecar(
    *,
    sidecar_path: str | Path,
    episode_id: int,
    split: str,
    positive_threshold: Sequence[float],
    negative_threshold: Sequence[float],
    qvel_response_threshold: Sequence[float],
    response_window_ticks: int,
    response_horizon_index: int,
    supported_axes: Sequence[str] = ("swing", "boom", "bucket"),
) -> MonitorEpisodeSummary:
    """Replay every effective onset in one response sidecar.

    The command send timestamp is taken from the sidecar's causally aligned
    ``command_send_timestamp_ns`` array.  A window with a reset, gap, invalid
    timestamp, or truncated horizon is counted as ``unknown`` rather than
    converted into a stalled label.
    """

    path = Path(sidecar_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"execution-response sidecar not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name])
            for name in (
                "qpos",
                "qvel",
                "previous_final_command",
                "command_send_timestamp_ns",
                "observation_timestamp_ns",
                "event_mask",
                "valid_mask",
                "reset_mask",
                "response_mask",
            )
            if name in data
        }
    required = {
        "qpos",
        "qvel",
        "previous_final_command",
        "command_send_timestamp_ns",
        "observation_timestamp_ns",
        "event_mask",
        "valid_mask",
        "reset_mask",
        "response_mask",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"sidecar is missing required arrays: {missing}")

    qpos = _state_matrix(arrays["qpos"], name="qpos")
    qvel = _state_matrix(arrays["qvel"], name="qvel")
    command = _state_matrix(
        arrays["previous_final_command"], name="previous_final_command"
    )
    n_steps = qvel.shape[0]
    if qpos.shape != qvel.shape or command.shape != qvel.shape:
        raise ValueError("qpos, qvel, and previous_final_command must have equal shape")
    send_ts = _vector(
        arrays["command_send_timestamp_ns"], name="command_send_timestamp_ns"
    )
    observation_ts = _vector(
        arrays["observation_timestamp_ns"], name="observation_timestamp_ns"
    )
    if send_ts.shape[0] != n_steps or observation_ts.shape[0] != n_steps:
        raise ValueError("timestamp lengths must match state arrays")
    event_mask = _mask(arrays["event_mask"], name="event_mask", shape=(n_steps, 4))
    valid_mask = _mask(
        arrays["valid_mask"], name="valid_mask", shape=(n_steps,)
    )
    reset_mask = _mask(
        arrays["reset_mask"], name="reset_mask", shape=(n_steps,)
    )
    response_mask = np.asarray(arrays["response_mask"], dtype=np.int8)
    if response_mask.ndim != 3 or response_mask.shape[1:] != (n_steps, 4):
        raise ValueError(
            "response_mask must have shape (H, T, 4), "
            f"got {response_mask.shape} for T={n_steps}"
        )
    horizon_index = int(response_horizon_index)
    if not 0 <= horizon_index < response_mask.shape[0]:
        raise ValueError(
            f"response_horizon_index {horizon_index} is out of range for "
            f"{response_mask.shape[0]} horizons"
        )

    config = ExecutionMonitorConfig(
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        qvel_response_threshold=qvel_response_threshold,
        response_window_ticks=int(response_window_ticks),
        # Retry eligibility is intentionally not evaluated from this teleop
        # sidecar, so no direction-confidence threshold is selected here.
        min_direction_confidence=0.0,
        supported_axes=supported_axes,
        min_response_ticks=1,
        max_retries_per_event=1,
    )
    by_axis = {
        axis: {
            "event_count": 0,
            "responded_count": 0,
            "stalled_count": 0,
            "unknown_count": 0,
            "sidecar_response_count": 0,
            "sidecar_stalled_candidate_count": 0,
            "response_mismatch_count": 0,
        }
        for axis in AXIS_NAMES
    }
    event_count = responded = stalled = unknown = 0
    sidecar_response = sidecar_stalled = mismatch = incomplete = 0
    for timestep, axis_index in zip(*np.where(event_mask)):
        timestep = int(timestep)
        axis_index = int(axis_index)
        axis_name = AXIS_NAMES[axis_index]
        counters = by_axis[axis_name]
        event_count += 1
        counters["event_count"] += 1
        complete = _complete_response_window(
            timestep=timestep,
            horizon=int(response_window_ticks),
            valid_mask=valid_mask,
            reset_mask=reset_mask,
            qpos=qpos,
            qvel=qvel,
            observation_ts=observation_ts,
            send_ts=send_ts,
        )
        label = int(response_mask[horizon_index, timestep, axis_index])
        if label == 1:
            sidecar_response += 1
            counters["sidecar_response_count"] += 1
        elif label == 0:
            sidecar_stalled += 1
            counters["sidecar_stalled_candidate_count"] += 1
        if not complete:
            status = "unknown"
            incomplete += 1
        else:
            monitor = ExecutionMonitor(config)
            monitor.on_command_sent(
                SentCommand(
                    command[timestep],
                    int(send_ts[timestep]),
                )
            )
            update = None
            for offset in range(1, int(response_window_ticks) + 1):
                update = monitor.observe_feedback(
                    FeedbackSample(
                        int(observation_ts[timestep + offset]),
                        qpos[timestep + offset],
                        qvel[timestep + offset],
                    )
                )
            if update is None:
                raise RuntimeError("monitor produced no update for a complete window")
            status = update.statuses[axis_index]
        if status == "responded":
            responded += 1
            counters["responded_count"] += 1
        elif status == "stalled":
            stalled += 1
            counters["stalled_count"] += 1
        elif status == "unknown":
            unknown += 1
            counters["unknown_count"] += 1
        else:
            raise RuntimeError(f"unexpected terminal monitor status: {status}")
        if complete and label in {0, 1}:
            expected = "responded" if label == 1 else "stalled"
            if status != expected:
                mismatch += 1
                counters["response_mismatch_count"] += 1

    return MonitorEpisodeSummary(
        episode_id=int(episode_id),
        split=str(split),
        sidecar_path=str(path),
        event_count=event_count,
        responded_count=responded,
        stalled_count=stalled,
        unknown_count=unknown,
        sidecar_response_count=sidecar_response,
        sidecar_stalled_candidate_count=sidecar_stalled,
        response_mismatch_count=mismatch,
        incomplete_window_count=incomplete,
        by_axis=by_axis,
    )


def aggregate_monitor_summaries(
    summaries: Sequence[MonitorEpisodeSummary],
) -> dict[str, Any]:
    """Aggregate summaries while keeping retry supervision explicitly absent."""

    result: dict[str, Any] = {
        "episode_count": len(summaries),
        "event_count": 0,
        "responded_count": 0,
        "stalled_count": 0,
        "unknown_count": 0,
        "sidecar_response_count": 0,
        "sidecar_stalled_candidate_count": 0,
        "response_mismatch_count": 0,
        "incomplete_window_count": 0,
        "by_axis": {
            axis: {
                "event_count": 0,
                "responded_count": 0,
                "stalled_count": 0,
                "unknown_count": 0,
                "sidecar_response_count": 0,
                "sidecar_stalled_candidate_count": 0,
                "response_mismatch_count": 0,
            }
            for axis in AXIS_NAMES
        },
        "retry_precision_estimable": False,
        "retry_precision_reason": (
            "existing sidecars are teleoperation response heuristics without "
            "policy intent, operator correction, or confirmed failed-actuation labels"
        ),
    }
    for summary in summaries:
        for key in (
            "event_count",
            "responded_count",
            "stalled_count",
            "unknown_count",
            "sidecar_response_count",
            "sidecar_stalled_candidate_count",
            "response_mismatch_count",
            "incomplete_window_count",
        ):
            result[key] += int(getattr(summary, key))
        for axis in AXIS_NAMES:
            for key, value in summary.by_axis[axis].items():
                result["by_axis"][axis][key] += int(value)
    return result


def _complete_response_window(
    *,
    timestep: int,
    horizon: int,
    valid_mask: np.ndarray,
    reset_mask: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    observation_ts: np.ndarray,
    send_ts: np.ndarray,
) -> bool:
    end = timestep + horizon + 1
    if timestep < 0 or end > qvel.shape[0]:
        return False
    if int(send_ts[timestep]) < 0:
        return False
    if not bool(valid_mask[timestep]):
        return False
    if not bool(np.all(valid_mask[timestep + 1 : end])):
        return False
    if bool(np.any(reset_mask[timestep + 1 : end])):
        return False
    if not int(send_ts[timestep]) < int(observation_ts[timestep + 1]):
        return False
    return bool(
        np.isfinite(qpos[timestep : end]).all()
        and np.isfinite(qvel[timestep : end]).all()
    )


def _state_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"{name} must have shape (T, 4), got {array.shape}")
    return array


def _vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _mask(value: np.ndarray, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array
