"""Dense trajectory-wide deadzone liveness metrics.

This module evaluates every expert-effective trajectory frame and each active
axis/direction over several future horizons.  It deliberately does not select
one startup anchor or classify a joint-action mode.  The expert effective set
at a frame may contain any number of axes, and every requested axis/direction
is scored independently.

The evaluator is command-level evidence.  It must be combined with the
recursive state-hold evaluator before making a live-control decision: an
open-loop future hit can still be hidden by teacher-forced state progression.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from testbed.policies.deadzone_eval import (
    AXIS_NAMES,
    effective_direction_mask,
    find_episode_action_files,
    load_deadzone_thresholds,
    parse_eval_spec,
)

DIRECTION_NAMES = ("pos", "neg")
DEFAULT_HORIZONS = (1, 4, 8, 20)
DEFAULT_PERSIST_STEPS = 2
FORBIDDEN_HELDOUT = frozenset({"episode_105", "episode_106", "episode_107", "episode_108", "episode_109"})


def apply_mechanical_deadzone_assist(
    actions: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
    *,
    trigger_fraction: float = 0.5,
    min_consecutive_steps: int = 2,
    margin: float | Sequence[float] = 0.02,
    clip: float = 1.0,
) -> np.ndarray:
    """Apply the existing sequential mechanical-assist contract offline.

    This is intentionally a counterfactual command transform for comparison;
    it does not change the policy checkpoint or pretend that the model emitted
    the assisted command.
    """

    source = _validate_action_array(actions, name="actions")
    if not 0.0 < float(trigger_fraction) <= 1.0:
        raise ValueError("trigger_fraction must be in (0, 1]")
    if int(min_consecutive_steps) < 1:
        raise ValueError("min_consecutive_steps must be >= 1")
    positive, negative = _threshold_arrays(thresholds)
    assist_margin = _vector4(margin, name="margin", nonnegative=True)
    result = source.copy()
    last_sign = np.zeros(4, dtype=np.int8)
    consecutive = np.zeros(4, dtype=np.int32)
    for step, action in enumerate(source):
        sign = np.sign(action).astype(np.int8)
        threshold = np.where(sign >= 0, positive, negative).astype(np.float32)
        intent = (sign != 0) & (np.abs(action) >= float(trigger_fraction) * threshold)
        same_direction = intent & (sign == last_sign)
        consecutive = np.where(
            same_direction,
            consecutive + 1,
            np.where(intent, 1, 0),
        ).astype(np.int32)
        last_sign = np.where(intent, sign, 0).astype(np.int8)
        assist_mask = intent & (consecutive >= int(min_consecutive_steps)) & (
            np.abs(action) < threshold
        )
        target = np.minimum(threshold + assist_margin, float(clip)).astype(np.float32)
        result[step] = np.clip(
            np.where(assist_mask, sign.astype(np.float32) * target, action),
            -float(clip),
            float(clip),
        )
    return result.astype(np.float32, copy=False)


def evaluate_trajectory_liveness(
    *,
    episode_id: str,
    expert_action: np.ndarray,
    policy_action: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    persist_steps: int = DEFAULT_PERSIST_STEPS,
    model: str = "model",
    variant: str = "raw",
) -> dict[str, Any]:
    """Evaluate every expert-effective frame without choosing an anchor.

    For each requested axis/direction at time ``t``, ``hit_hN`` means that the
    policy crosses that same directional deadzone at least once in
    ``[t, t + N)``.  ``persistent_hN`` additionally requires the effective
    command to stay effective for ``persist_steps`` consecutive ticks after the
    first crossing.  Margin metrics are signed relative to the directional
    deadzone, so positive values mean that the command is physically effective.
    """

    expert = _validate_action_array(expert_action, name="expert_action")
    policy = _validate_action_array(policy_action, name="policy_action")
    if expert.shape != policy.shape:
        raise ValueError(f"expert and policy shapes differ: {expert.shape} vs {policy.shape}")
    parsed_horizons = _normalize_horizons(horizons)
    if int(persist_steps) < 1:
        raise ValueError("persist_steps must be >= 1")

    expert_effective = effective_direction_mask(expert, dict(thresholds))
    policy_effective = effective_direction_mask(policy, dict(thresholds))
    signed_margin = _signed_margin(policy, thresholds)
    segment_lookup, segments = _build_direction_segments(
        episode_id=episode_id,
        model=model,
        variant=variant,
        expert_effective=expert_effective,
        policy_effective=policy_effective,
        signed_margin=signed_margin,
        horizons=parsed_horizons,
        persist_steps=int(persist_steps),
    )

    opportunity_rows: list[dict[str, Any]] = []
    for step in range(expert.shape[0]):
        targets = np.argwhere(expert_effective[step])
        if targets.size == 0:
            continue
        current_policy_dirs = {
            (int(axis_index), int(direction_index))
            for axis_index, direction_index in np.argwhere(policy_effective[step])
        }
        expert_target_dirs = {
            (int(axis_index), int(direction_index))
            for axis_index, direction_index in targets
        }
        wrong_extra_current = sorted(current_policy_dirs - expert_target_dirs)
        for axis_index, direction_index in targets:
            axis_index = int(axis_index)
            direction_index = int(direction_index)
            sign = 1.0 if direction_index == 0 else -1.0
            horizon_values: dict[str, Any] = {}
            for horizon in parsed_horizons:
                end = min(expert.shape[0], step + horizon)
                margins = signed_margin[step:end, axis_index, direction_index]
                effective = policy_effective[step:end, axis_index, direction_index]
                hit_indices = np.flatnonzero(effective)
                hit = bool(hit_indices.size)
                first_delay = int(hit_indices[0]) if hit else None
                persistent = bool(
                    hit
                    and _contains_true_run(effective[first_delay:], int(persist_steps))
                )
                horizon_values[f"hit_h{horizon}"] = hit
                horizon_values[f"persistent_h{horizon}"] = persistent
                horizon_values[f"delay_h{horizon}"] = first_delay
                horizon_values[f"max_margin_h{horizon}"] = (
                    float(np.max(margins)) if margins.size else float("nan")
                )
                horizon_values[f"same_sign_h{horizon}"] = bool(
                    np.any(sign * policy[step:end, axis_index] > 0.0)
                )
            max_horizon = parsed_horizons[-1]
            same_sign = bool(horizon_values[f"same_sign_h{max_horizon}"])
            current_effective = bool(policy_effective[step, axis_index, direction_index])
            current_same_sign = bool(sign * policy[step, axis_index] > 0.0)
            opportunity_rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "episode_id": str(episode_id),
                    "step": step,
                    "axis_index": axis_index,
                    "axis": AXIS_NAMES[axis_index],
                    "direction_index": direction_index,
                    "direction": DIRECTION_NAMES[direction_index],
                    "segment_id": segment_lookup.get((axis_index, direction_index, step)),
                    "threshold": float(
                        thresholds[AXIS_NAMES[axis_index]][DIRECTION_NAMES[direction_index]]
                    ),
                    "expert_action": float(expert[step, axis_index]),
                    "policy_action": float(policy[step, axis_index]),
                    "current_policy_effective": current_effective,
                    "current_same_sign": current_same_sign,
                    "current_signed_margin": float(
                        signed_margin[step, axis_index, direction_index]
                    ),
                    "same_sign_within_max_horizon": same_sign,
                    "underconfidence_within_max_horizon": bool(
                        (not bool(horizon_values[f"hit_h{max_horizon}"])) and same_sign
                    ),
                    "underconfidence_current": bool(
                        (not current_effective) and current_same_sign
                    ),
                    "wrong_extra_current_count": len(wrong_extra_current),
                    "wrong_extra_current_dirs": ",".join(
                        f"{AXIS_NAMES[a]}{DIRECTION_NAMES[d][0]}"
                        for a, d in wrong_extra_current
                    ),
                    **horizon_values,
                }
            )

    episode_summary = _episode_summary(
        model=model,
        variant=variant,
        episode_id=episode_id,
        expert_effective=expert_effective,
        policy_effective=policy_effective,
        opportunity_rows=opportunity_rows,
        horizons=parsed_horizons,
        segments=segments,
    )
    axis_direction_summary = _axis_direction_summary(
        model=model,
        variant=variant,
        episode_id=episode_id,
        opportunity_rows=opportunity_rows,
        expert_effective=expert_effective,
        policy_effective=policy_effective,
        signed_margin=signed_margin,
        horizons=parsed_horizons,
        segments=segments,
    )
    return {
        "episode_summary": episode_summary,
        "axis_direction_summary": axis_direction_summary,
        "segments": segments,
        "opportunities": opportunity_rows,
    }


def aggregate_trajectory_liveness(
    reports: Sequence[Mapping[str, Any]],
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    """Aggregate episode reports while preserving per-episode worst cases."""

    parsed_horizons = _normalize_horizons(horizons)
    episode_rows = [dict(report["episode_summary"]) for report in reports]
    axis_rows = [row for report in reports for row in report["axis_direction_summary"]]
    segment_rows = [row for report in reports for row in report["segments"]]
    if not episode_rows:
        return {
            "episodes": 0,
            "episode_rows": [],
            "axis_direction_rows": [],
            "segments": [],
            "aggregate": {},
        }

    aggregate: dict[str, Any] = {
        "model": episode_rows[0]["model"],
        "variant": episode_rows[0]["variant"],
        "episodes": len(episode_rows),
        "total_steps": int(sum(int(row["steps"]) for row in episode_rows)),
        "expert_effective_frames": int(
            sum(int(row["expert_effective_frames"]) for row in episode_rows)
        ),
        "opportunities": int(sum(int(row["opportunities"]) for row in episode_rows)),
        "segments": len(segment_rows),
        "segments_with_zero_hit_max_horizon": int(
            sum(
                not bool(row[f"any_hit_h{parsed_horizons[-1]}"])
                for row in segment_rows
            )
        ),
        "episodes_with_any_underconfidence": int(
            sum(bool(row["underconfidence_current"]) for row in episode_rows)
        ),
        "episodes_with_any_zero_liveness": int(
            sum(int(row[f"zero_liveness_h{parsed_horizons[-1]}"]) > 0 for row in episode_rows)
        ),
    }
    for horizon in parsed_horizons:
        aggregate[f"hit_{horizon}_rate"] = _rate(
            sum(int(row[f"hit_count_h{horizon}"]) for row in episode_rows),
            aggregate["opportunities"],
        )
        aggregate[f"persistent_{horizon}_rate"] = _rate(
            sum(int(row[f"persistent_count_h{horizon}"]) for row in episode_rows),
            aggregate["opportunities"],
        )
        aggregate[f"episodes_zero_liveness_{horizon}"] = int(
            sum(int(row[f"zero_liveness_h{horizon}"]) > 0 for row in episode_rows)
        )
        aggregate[f"segments_any_hit_{horizon}_rate"] = _rate(
            sum(bool(row[f"any_hit_h{horizon}"]) for row in segment_rows),
            len(segment_rows),
        )
    aggregate.update(
        {
            "current_same_direction_rate": _rate(
                sum(int(row["current_same_direction_count"]) for row in episode_rows),
                aggregate["opportunities"],
            ),
            "underconfidence_rate": _rate(
                sum(int(row["underconfidence_count"]) for row in episode_rows),
                aggregate["opportunities"],
            ),
            "same_sign_but_no_hit_rate": _rate(
                sum(int(row["same_sign_but_no_hit_count"]) for row in episode_rows),
                aggregate["opportunities"],
            ),
            "wrong_extra_active_frame_rate": _rate(
                sum(int(row["wrong_extra_active_frames"]) for row in episode_rows),
                sum(int(row["expert_effective_frames"]) for row in episode_rows),
            ),
            "policy_any_effective_frame_rate": _rate(
                sum(int(row["policy_any_effective_frames"]) for row in episode_rows),
                aggregate["total_steps"],
            ),
            "expert_any_effective_frame_rate": _rate(
                sum(int(row["expert_any_effective_frames"]) for row in episode_rows),
                aggregate["total_steps"],
            ),
        }
    )
    return {
        "episodes": len(episode_rows),
        "episode_rows": episode_rows,
        "axis_direction_rows": axis_rows,
        "segments": segment_rows,
        "aggregate": aggregate,
    }


def _episode_summary(
    *,
    model: str,
    variant: str,
    episode_id: str,
    expert_effective: np.ndarray,
    policy_effective: np.ndarray,
    opportunity_rows: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    opportunities = len(opportunity_rows)
    current_same = sum(bool(row["current_policy_effective"]) for row in opportunity_rows)
    underconfidence = sum(bool(row["underconfidence_current"]) for row in opportunity_rows)
    same_sign_no_hit = sum(
        bool(row["same_sign_within_max_horizon"])
        and not bool(row[f"hit_h{horizons[-1]}"])
        for row in opportunity_rows
    )
    wrong_extra_active_frames = int(
        np.count_nonzero(
            expert_effective.any(axis=(1, 2))
            & np.any(policy_effective & ~expert_effective, axis=(1, 2))
        )
    )
    row: dict[str, Any] = {
        "model": model,
        "variant": variant,
        "episode_id": str(episode_id),
        "steps": int(expert_effective.shape[0]),
        "expert_effective_frames": int(np.count_nonzero(expert_effective.any(axis=(1, 2)))),
        "expert_any_effective_frames": int(
            np.count_nonzero(expert_effective.any(axis=(1, 2)))
        ),
        "expert_effective_direction_events": int(np.count_nonzero(expert_effective)),
        "policy_any_effective_frames": int(np.count_nonzero(policy_effective.any(axis=(1, 2)))),
        "opportunities": opportunities,
        "current_same_direction_count": int(current_same),
        "underconfidence_count": int(underconfidence),
        "same_sign_but_no_hit_count": int(same_sign_no_hit),
        "wrong_extra_active_frames": wrong_extra_active_frames,
        "segments": len(segments),
        "underconfidence_current": bool(underconfidence),
        "underconfidence_within_max_horizon": bool(
            any(bool(item["underconfidence_within_max_horizon"]) for item in opportunity_rows)
        ),
    }
    for horizon in horizons:
        row[f"hit_count_h{horizon}"] = int(
            sum(bool(item[f"hit_h{horizon}"]) for item in opportunity_rows)
        )
        row[f"persistent_count_h{horizon}"] = int(
            sum(bool(item[f"persistent_h{horizon}"]) for item in opportunity_rows)
        )
        row[f"zero_liveness_h{horizon}"] = int(
            sum(not bool(item[f"hit_h{horizon}"]) for item in opportunity_rows)
        )
        row[f"any_hit_h{horizon}"] = bool(
            any(bool(item[f"hit_h{horizon}"]) for item in opportunity_rows)
        )
        row[f"hit_rate_h{horizon}"] = _rate(
            row[f"hit_count_h{horizon}"], opportunities
        )
        row[f"persistent_rate_h{horizon}"] = _rate(
            row[f"persistent_count_h{horizon}"], opportunities
        )
    row["underconfidence_rate"] = _rate(underconfidence, opportunities)
    row["current_same_direction_rate"] = _rate(current_same, opportunities)
    return row


def _axis_direction_summary(
    *,
    model: str,
    variant: str,
    episode_id: str,
    opportunity_rows: Sequence[Mapping[str, Any]],
    expert_effective: np.ndarray,
    policy_effective: np.ndarray,
    signed_margin: np.ndarray,
    horizons: Sequence[int],
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(AXIS_NAMES):
        for direction_index, direction in enumerate(DIRECTION_NAMES):
            group = [
                row
                for row in opportunity_rows
                if int(row["axis_index"]) == axis_index
                and int(row["direction_index"]) == direction_index
            ]
            segment_group = [
                row
                for row in segments
                if int(row["axis_index"]) == axis_index
                and int(row["direction_index"]) == direction_index
            ]
            row: dict[str, Any] = {
                "model": model,
                "variant": variant,
                "episode_id": str(episode_id),
                "axis_index": axis_index,
                "axis": axis,
                "direction_index": direction_index,
                "direction": direction,
                "expert_effective_frames": int(
                    np.count_nonzero(expert_effective[:, axis_index, direction_index])
                ),
                "opportunities": len(group),
                "segments": len(segment_group),
                "current_same_direction_count": int(
                    sum(bool(item["current_policy_effective"]) for item in group)
                ),
                "underconfidence_count": int(
                    sum(bool(item["underconfidence_current"]) for item in group)
                ),
                "margin_p10": _percentile(
                    [float(item[f"max_margin_h{horizons[-1]}"]) for item in group],
                    10.0,
                ),
                "margin_median": _percentile(
                    [float(item[f"max_margin_h{horizons[-1]}"]) for item in group],
                    50.0,
                ),
                "margin_min": _minimum(
                    [float(item[f"max_margin_h{horizons[-1]}"]) for item in group]
                ),
                "current_margin_p10": _percentile(
                    [float(item["current_signed_margin"]) for item in group],
                    10.0,
                ),
                "current_margin_median": _percentile(
                    [float(item["current_signed_margin"]) for item in group],
                    50.0,
                ),
                "current_margin_min": _minimum(
                    [float(item["current_signed_margin"]) for item in group]
                ),
            }
            for horizon in horizons:
                hits = int(sum(bool(item[f"hit_h{horizon}"]) for item in group))
                persistent = int(
                    sum(bool(item[f"persistent_h{horizon}"]) for item in group)
                )
                row[f"hit_count_h{horizon}"] = hits
                row[f"hit_rate_h{horizon}"] = _rate(hits, len(group))
                row[f"persistent_count_h{horizon}"] = persistent
                row[f"persistent_rate_h{horizon}"] = _rate(persistent, len(group))
                row[f"segments_any_hit_h{horizon}"] = int(
                    sum(bool(item[f"any_hit_h{horizon}"]) for item in segment_group)
                )
            row["current_same_direction_rate"] = _rate(
                row["current_same_direction_count"], len(group)
            )
            row["underconfidence_rate"] = _rate(
                row["underconfidence_count"], len(group)
            )
            rows.append(row)
    return rows


def _build_direction_segments(
    *,
    episode_id: str,
    model: str,
    variant: str,
    expert_effective: np.ndarray,
    policy_effective: np.ndarray,
    signed_margin: np.ndarray,
    horizons: Sequence[int],
    persist_steps: int,
) -> tuple[dict[tuple[int, int, int], str], list[dict[str, Any]]]:
    lookup: dict[tuple[int, int, int], str] = {}
    rows: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(AXIS_NAMES):
        for direction_index, direction in enumerate(DIRECTION_NAMES):
            active = expert_effective[:, axis_index, direction_index]
            start = 0
            while start < len(active):
                indices = np.flatnonzero(active[start:])
                if indices.size == 0:
                    break
                segment_start = start + int(indices[0])
                end_indices = np.flatnonzero(~active[segment_start:])
                segment_end = (
                    segment_start + int(end_indices[0])
                    if end_indices.size
                    else len(active)
                )
                segment_id = (
                    f"{episode_id}:{axis}:{direction}:{segment_start}:{segment_end}"
                )
                for step in range(segment_start, segment_end):
                    lookup[(axis_index, direction_index, step)] = segment_id
                opportunity_mask = active[segment_start:segment_end]
                policy_segment = policy_effective[segment_start:segment_end, axis_index, direction_index]
                row: dict[str, Any] = {
                    "model": model,
                    "variant": variant,
                    "episode_id": str(episode_id),
                    "segment_id": segment_id,
                    "axis_index": axis_index,
                    "axis": axis,
                    "direction_index": direction_index,
                    "direction": direction,
                    "start_step": segment_start,
                    "end_step_exclusive": segment_end,
                    "expert_effective_frames": int(np.count_nonzero(opportunity_mask)),
                    "policy_current_effective_frames": int(np.count_nonzero(policy_segment)),
                    "policy_current_effective_rate": _rate(
                        int(np.count_nonzero(policy_segment)),
                        int(np.count_nonzero(opportunity_mask)),
                    ),
                    "max_consecutive_current_miss": _max_false_run(policy_segment),
                }
                for horizon in horizons:
                    segment_opportunities = []
                    for step in range(segment_start, segment_end):
                        end = min(len(active), step + horizon)
                        segment_opportunities.append(
                            bool(
                                np.any(
                                    policy_effective[
                                        step:end, axis_index, direction_index
                                    ]
                                )
                            )
                        )
                    row[f"hit_count_h{horizon}"] = int(sum(segment_opportunities))
                    row[f"any_hit_h{horizon}"] = bool(any(segment_opportunities))
                    row[f"hit_rate_h{horizon}"] = _rate(
                        row[f"hit_count_h{horizon}"], row["expert_effective_frames"]
                    )
                    persistent_count = 0
                    for step in range(segment_start, segment_end):
                        end = min(len(active), step + horizon)
                        effective = policy_effective[
                            step:end, axis_index, direction_index
                        ]
                        first = np.flatnonzero(effective)
                        if first.size and _contains_true_run(
                            effective[int(first[0]) :], persist_steps
                        ):
                            persistent_count += 1
                    row[f"persistent_count_h{horizon}"] = persistent_count
                    row[f"persistent_rate_h{horizon}"] = _rate(
                        persistent_count, row["expert_effective_frames"]
                    )
                rows.append(row)
                start = segment_end
    return lookup, rows


def _signed_margin(
    action: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    values = _validate_action_array(action, name="action")
    positive, negative = _threshold_arrays(thresholds)
    result = np.empty((values.shape[0], len(AXIS_NAMES), 2), dtype=np.float32)
    result[:, :, 0] = values - positive.reshape(1, -1)
    result[:, :, 1] = -values - negative.reshape(1, -1)
    return result


def _threshold_arrays(
    thresholds: Mapping[str, Mapping[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(
        [float(thresholds[axis]["pos"]) for axis in AXIS_NAMES], dtype=np.float32
    )
    negative = np.asarray(
        [float(thresholds[axis]["neg"]) for axis in AXIS_NAMES], dtype=np.float32
    )
    if np.any(~np.isfinite(positive)) or np.any(~np.isfinite(negative)):
        raise ValueError("deadzone thresholds must be finite")
    if np.any(positive < 0.0) or np.any(negative < 0.0):
        raise ValueError("deadzone thresholds must be non-negative")
    return positive, negative


def _normalize_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in horizons}))
    if not values or any(value < 1 for value in values):
        raise ValueError("horizons must contain positive integers")
    return values


def _validate_action_array(action: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"{name} must have shape (N, {len(AXIS_NAMES)}), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def _vector4(value: float | Sequence[float], *, name: str, nonnegative: bool) -> np.ndarray:
    if isinstance(value, (int, float)):
        values = np.full(4, float(value), dtype=np.float32)
    else:
        values = np.asarray(value, dtype=np.float32).reshape(-1)
        if values.shape != (4,):
            raise ValueError(f"{name} must be scalar or shape (4,), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if nonnegative and np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return values.astype(np.float32, copy=False)


def _contains_true_run(values: np.ndarray, run_length: int) -> bool:
    if int(run_length) <= 1:
        return bool(np.any(values))
    count = 0
    for value in np.asarray(values, dtype=bool):
        count = count + 1 if value else 0
        if count >= int(run_length):
            return True
    return False


def _max_false_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        current = 0 if value else current + 1
        longest = max(longest, current)
    return int(longest)


def _rate(numerator: int, denominator: int) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.percentile(finite, percentile)) if finite.size else None


def _minimum(values: Sequence[float]) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.min(finite)) if finite.size else None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False))
            stream.write("\n")


def _load_actions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "expert_action" not in data or "policy_action" not in data:
            raise ValueError(f"{path} must contain expert_action and policy_action")
        expert = np.asarray(data["expert_action"], dtype=np.float32)
        policy = np.asarray(data["policy_action"], dtype=np.float32)
    return expert, policy


def _build_direction_segment_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row["variant"])), []).append(row)
    return grouped


__all__ = [
    "DEFAULT_HORIZONS",
    "DEFAULT_PERSIST_STEPS",
    "FORBIDDEN_HELDOUT",
    "aggregate_trajectory_liveness",
    "apply_mechanical_deadzone_assist",
    "evaluate_trajectory_liveness",
    "find_episode_action_files",
    "load_deadzone_thresholds",
    "parse_eval_spec",
    "sha256_file",
    "write_csv",
    "write_jsonl",
]
