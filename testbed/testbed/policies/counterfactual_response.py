"""Counterfactual command-response feasibility checks.

This module is intentionally offline-only.  It does not claim that a policy
action was sent to the machine.  Instead, it answers a narrower question:
given a recorded policy trace, would one demonstrated target remain
non-reproduced after assuming that an already-effective command receives an
immediate, same-direction response?

The important separation is between two failures:

* ``command_limited``: the target axis/direction never crossed its direct
  mechanical deadzone, so no plant-response assumption can make that target
  move;
* ``response_limited``: an effective target command was present, but the
  state-hold evaluator still did not reproduce the demo target.  This is
  the only class for which a response model can plausibly help.

The empirical response profile is fitted from train-fold teleoperation
sidecar events only. Validation traces are never used to fit it, and held-out
episode IDs are rejected. These classes do not determine task correctness.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from testbed.data.deadzone_intent_labels import AXIS_NAMES

HELDOUT_EPISODES = frozenset({105, 106, 107, 108, 109})
DEFAULT_RESPONSE_HORIZONS = (1, 2, 4, 8, 20)
DEFAULT_SUPPORTED_AXES = ("swing", "boom", "bucket")


@dataclass(frozen=True)
class ResponseGroupProfile:
    """Train-fold response statistics for one axis and direction."""

    axis: str
    direction: str
    event_count: int
    response_counts: Mapping[int, int]
    valid_counts: Mapping[int, int]
    first_response_tick_counts: Mapping[int, int]

    @property
    def median_first_response_ticks(self) -> float | None:
        values: list[int] = []
        for tick, count in self.first_response_tick_counts.items():
            values.extend([int(tick)] * int(count))
        if not values:
            return None
        return float(np.median(np.asarray(values, dtype=np.float64)))

    @property
    def p_response_at_one_tick(self) -> float | None:
        return self.response_probability(1)

    def response_probability(self, horizon: int) -> float | None:
        valid = int(self.valid_counts.get(int(horizon), 0))
        if valid <= 0:
            return None
        return float(self.response_counts.get(int(horizon), 0)) / float(valid)

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "direction": self.direction,
            "event_count": self.event_count,
            "response_counts": {
                str(key): int(value) for key, value in self.response_counts.items()
            },
            "valid_counts": {
                str(key): int(value) for key, value in self.valid_counts.items()
            },
            "first_response_tick_counts": {
                str(key): int(value)
                for key, value in self.first_response_tick_counts.items()
            },
            "response_probability_1t": self.p_response_at_one_tick,
            "median_first_response_ticks": self.median_first_response_ticks,
        }


@dataclass(frozen=True)
class CounterfactualAnchorResult:
    """A causal attribution result for one state-hold anchor."""

    episode_id: str
    anchor_step: int
    anchor_group: str
    target_axis: str
    target_direction: str
    observed_state_hold_status: str
    observed_demo_target_not_reproduced: bool
    demo_target_reproduction_hidden_by_teacher_forcing: bool
    demo_target_effective_ticks: int
    opposite_to_demo_target_ticks: int
    demo_target_first_effective_tick: int | None
    demo_target_response_probability_1t: float | None
    demo_target_median_response_latency_ticks: float | None
    optimistic_instant_response: bool
    empirical_response_at_horizon: bool | None
    command_limited: bool
    response_limited: bool
    oracle_demo_target_injection_reproduces: bool
    non_target_effective_ticks: int
    effective_axis_direction_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["effective_axis_direction_counts"] = dict(
            self.effective_axis_direction_counts
        )
        return payload


def build_response_profile(
    event_rows: Iterable[Mapping[str, Any]],
    *,
    horizons: Sequence[int] = DEFAULT_RESPONSE_HORIZONS,
) -> dict[str, ResponseGroupProfile]:
    """Fit response rates and latency histograms from train-fold events."""

    resolved_horizons = tuple(int(value) for value in horizons)
    if not resolved_horizons or any(value <= 0 for value in resolved_horizons):
        raise ValueError("horizons must contain positive integers")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        axis = str(row.get("axis", ""))
        direction = str(row.get("direction", ""))
        if axis not in AXIS_NAMES or direction not in {"pos", "neg"}:
            raise ValueError(f"invalid event axis/direction: {axis!r}/{direction!r}")
        grouped[f"{axis}:{direction}"].append(row)

    profiles: dict[str, ResponseGroupProfile] = {}
    for key, rows in grouped.items():
        axis, direction = key.split(":", 1)
        response_counts: Counter[int] = Counter()
        valid_counts: Counter[int] = Counter()
        first_response_ticks: Counter[int] = Counter()
        for row in rows:
            first_response: int | None = None
            for horizon in resolved_horizons:
                value = _optional_int(row.get(f"response_{horizon}t"))
                if value not in {0, 1}:
                    continue
                valid_counts[horizon] += 1
                response_counts[horizon] += value
                if value == 1 and first_response is None:
                    first_response = horizon
            if first_response is not None:
                first_response_ticks[first_response] += 1
        profiles[key] = ResponseGroupProfile(
            axis=axis,
            direction=direction,
            event_count=len(rows),
            response_counts=dict(response_counts),
            valid_counts=dict(valid_counts),
            first_response_tick_counts=dict(first_response_ticks),
        )
    return profiles


def classify_trace_effects(
    action_trace: Sequence[Sequence[float]],
    *,
    positive_threshold: Sequence[float],
    negative_threshold: Sequence[float],
    supported_axes: Sequence[str] = DEFAULT_SUPPORTED_AXES,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify direct-domain effective actions in a policy trace."""

    actions = np.asarray(action_trace, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"action_trace must have shape (T, 4), got {actions.shape}")
    positive = _axis_vector(positive_threshold, "positive_threshold")
    negative = _axis_vector(negative_threshold, "negative_threshold")
    supported = np.asarray(
        [axis in set(supported_axes) for axis in AXIS_NAMES], dtype=bool
    )
    effective = np.zeros_like(actions, dtype=bool)
    direction = np.zeros_like(actions, dtype=np.int8)
    effective[:, supported] = (actions[:, supported] >= positive[supported]) | (
        actions[:, supported] <= -negative[supported]
    )
    positive_hit = actions >= positive.reshape(1, -1)
    negative_hit = actions <= -negative.reshape(1, -1)
    direction[positive_hit & effective] = 1
    direction[negative_hit & effective] = -1
    return effective, direction


def simulate_anchor(
    row: Mapping[str, Any],
    *,
    profiles: Mapping[str, ResponseGroupProfile],
    positive_threshold: Sequence[float],
    negative_threshold: Sequence[float],
    response_horizon: int = 1,
    supported_axes: Sequence[str] = DEFAULT_SUPPORTED_AXES,
) -> CounterfactualAnchorResult:
    """Run the optimistic/empirical response counterfactual for one anchor."""

    action_trace = row.get("state_hold_action_trace")
    if not isinstance(action_trace, list):
        raise ValueError("state-hold row has no state_hold_action_trace list")
    effective, direction = classify_trace_effects(
        action_trace,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        supported_axes=supported_axes,
    )
    target_axis = str(row["axis"])
    target_direction = str(row["direction"])
    target_index = AXIS_NAMES.index(target_axis)
    target_sign = 1 if target_direction == "pos" else -1
    target_hits = effective[:, target_index] & (
        direction[:, target_index] == target_sign
    )
    target_opposite = effective[:, target_index] & (
        direction[:, target_index] == -target_sign
    )
    target_effective_ticks = int(np.count_nonzero(target_hits))
    target_opposite_ticks = int(np.count_nonzero(target_opposite))
    first_target = int(np.flatnonzero(target_hits)[0]) if np.any(target_hits) else None
    group = profiles.get(f"{target_axis}:{target_direction}")
    response_probability = group.p_response_at_one_tick if group else None
    median_latency = group.median_first_response_ticks if group else None
    empirical_probability = (
        group.response_probability(response_horizon) if group else None
    )
    empirical_recovery = (
        None
        if target_effective_ticks == 0 or empirical_probability is None
        else bool(empirical_probability > 0.0)
    )
    observed_not_reproduced = bool(
        row.get("state_hold_demo_target_not_reproduced", False)
    )
    optimistic_reproduction = target_effective_ticks > 0
    command_limited = observed_not_reproduced and target_effective_ticks == 0
    response_limited = observed_not_reproduced and target_effective_ticks > 0
    effective_direction_counts: Counter[str] = Counter()
    for axis_index, axis_name in enumerate(AXIS_NAMES):
        for sign, label in ((1, "pos"), (-1, "neg")):
            count = int(
                np.count_nonzero(
                    effective[:, axis_index] & (direction[:, axis_index] == sign)
                )
            )
            if count:
                effective_direction_counts[f"{axis_name}:{label}"] = count
    non_target = int(np.count_nonzero(effective)) - int(target_effective_ticks)
    return CounterfactualAnchorResult(
        episode_id=str(row["episode_id"]),
        anchor_step=int(row["anchor_step"]),
        anchor_group=str(row.get("anchor_group", "")),
        target_axis=target_axis,
        target_direction=target_direction,
        observed_state_hold_status=str(row.get("state_hold_status", "")),
        observed_demo_target_not_reproduced=observed_not_reproduced,
        demo_target_reproduction_hidden_by_teacher_forcing=bool(
            row.get("demo_target_reproduction_hidden_by_teacher_forcing", False)
        ),
        demo_target_effective_ticks=target_effective_ticks,
        opposite_to_demo_target_ticks=target_opposite_ticks,
        demo_target_first_effective_tick=first_target,
        demo_target_response_probability_1t=response_probability,
        demo_target_median_response_latency_ticks=median_latency,
        optimistic_instant_response=optimistic_reproduction,
        empirical_response_at_horizon=empirical_recovery,
        command_limited=command_limited,
        response_limited=response_limited,
        oracle_demo_target_injection_reproduces=True,
        non_target_effective_ticks=non_target,
        effective_axis_direction_counts=dict(effective_direction_counts),
    )


def simulate_state_hold_file(
    path: str | Path,
    *,
    profiles: Mapping[str, ResponseGroupProfile],
    positive_threshold: Sequence[float],
    negative_threshold: Sequence[float],
    response_horizon: int = 1,
    supported_axes: Sequence[str] = DEFAULT_SUPPORTED_AXES,
) -> list[CounterfactualAnchorResult]:
    """Simulate all JSONL anchors in one state-hold artifact."""

    source = Path(path).expanduser().resolve()
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    return [
        simulate_anchor(
            row,
            profiles=profiles,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            response_horizon=response_horizon,
            supported_axes=supported_axes,
        )
        for row in rows
    ]


def aggregate_counterfactual_results(
    results: Sequence[CounterfactualAnchorResult],
) -> dict[str, Any]:
    """Summarize command-vs-response attribution and optimistic upper bounds."""

    not_reproduced = [
        result for result in results if result.observed_demo_target_not_reproduced
    ]
    return {
        "anchors_total": len(results),
        "observed_demo_target_reproduced": sum(
            not result.observed_demo_target_not_reproduced for result in results
        ),
        "observed_demo_target_not_reproduced": len(not_reproduced),
        "observed_demo_target_reproduction_hidden": sum(
            result.demo_target_reproduction_hidden_by_teacher_forcing
            for result in results
        ),
        "optimistic_instant_demo_target_reproduced": sum(
            result.optimistic_instant_response for result in results
        ),
        "optimistic_instant_demo_target_not_reproduced": sum(
            not result.optimistic_instant_response for result in results
        ),
        "optimistic_demo_target_reproduction_gain": sum(
            result.observed_demo_target_not_reproduced
            and result.optimistic_instant_response
            for result in results
        ),
        "command_limited_demo_target_nonreproductions": sum(
            result.command_limited for result in results
        ),
        "response_limited_demo_target_nonreproductions": sum(
            result.response_limited for result in results
        ),
        "oracle_demo_target_injection_reproduced": len(results),
        "demo_target_nonreproduction_rows": [
            result.as_dict() for result in not_reproduced
        ],
        "demo_target_effective_tick_histogram": _histogram(
            result.demo_target_effective_ticks for result in results
        ),
        "demo_target_response_probability_1t": _mean_optional(
            result.demo_target_response_probability_1t for result in results
        ),
        "correctness_estimable": False,
        "task_support_estimable": False,
    }


def _axis_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(AXIS_NAMES),) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must have finite shape (4), got {vector.shape}")
    if np.any(vector <= 0):
        raise ValueError(f"{name} must be positive")
    return vector


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer in {-1, 0, 1} else None


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def _mean_optional(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.mean(np.asarray(finite, dtype=np.float64)))
