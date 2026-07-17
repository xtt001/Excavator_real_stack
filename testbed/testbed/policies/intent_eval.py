"""Single-demo similarity scoring for saved teacher-forced open-loop actions.

This module thresholds an already-saved continuous policy output and compares
the resulting executable direction set with ``ExpertIntentEvent`` labels.  It
does not expose latent model intent and does not evaluate closed-loop recovery
or physical machine response.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from testbed.data.expert_intent_events import SCHEMA_VERSION as EVENT_SCHEMA_VERSION
from testbed.policies.deadzone_eval import AXIS_NAMES, effective_direction_mask

SCHEMA_VERSION = "single_demo_open_loop_similarity_v4"
INFERENCE_SOURCE = "teacher_forced_open_loop_continuous_output"
DIRECTION_LABELS = tuple(f"{axis}{sign}" for axis in AXIS_NAMES for sign in ("+", "-"))
WINDOW_SPECS = (
    ("anchor_current", "anchor_intent", 0, 0),
    ("immediate_0_1", "immediate_intent_0_1", 0, 1),
    ("near_2_5", "near_intent_2_5", 2, 5),
    ("near_6_10", "near_intent_6_10", 6, 10),
)


def evaluate_open_loop_intent(
    *,
    model: str,
    events: Sequence[Mapping[str, Any]],
    policy_actions: Mapping[int, np.ndarray],
    thresholds: Mapping[str, Mapping[str, float]],
    sampling_hz: float = 20.0,
) -> dict[str, Any]:
    """Score one model without treating event windows as frame samples."""

    label = str(model).strip()
    if not label:
        raise ValueError("model label must not be empty")
    hz = float(sampling_hz)
    if not np.isfinite(hz) or hz <= 0.0:
        raise ValueError("sampling_hz must be finite and positive")
    expected_ids = {int(event["episode_id"]) for event in events}
    actual_ids = {int(episode_id) for episode_id in policy_actions}
    if actual_ids != expected_ids:
        raise ValueError(
            "policy episode IDs do not exactly match event episode IDs: "
            f"expected={sorted(expected_ids)}, actual={sorted(actual_ids)}"
        )

    rows: list[dict[str, Any]] = []
    masks: dict[int, np.ndarray] = {}
    for episode_id, actions in policy_actions.items():
        array = _validate_action_array(actions, name=f"policy_actions[{episode_id}]")
        masks[int(episode_id)] = effective_direction_mask(array, dict(thresholds))

    for event in events:
        _validate_event(event)
        episode_id = int(event["episode_id"])
        effective = masks[episode_id]
        _validate_event_bounds(event, total_steps=int(effective.shape[0]))
        onset = int(event["onset_step"])
        anchor = _label_set(event["anchor_intent"], name="anchor_intent")
        supported = _label_set(
            event["single_demo_event_support_directions"],
            name="single_demo_event_support_directions",
        )
        onset_delays = _direction_onset_delays(event, supported=supported)
        if not anchor <= supported:
            raise ValueError(
                f"{event['event_id']} anchor intent is outside single-demo event support"
            )

        for window, required_field, start_offset, stop_offset in WINDOW_SPECS:
            required = _label_set(event[required_field], name=required_field)
            if not required <= supported:
                raise ValueError(
                    f"{event['event_id']} {required_field} is outside single-demo event support"
                )
            start = onset + start_offset
            stop_exclusive = min(onset + stop_offset + 1, effective.shape[0])
            if start >= effective.shape[0]:
                predicted: set[str] = set()
                observed_ticks = 0
            else:
                predicted = _labels_from_mask(
                    np.any(effective[start:stop_exclusive], axis=0)
                )
                observed_ticks = stop_exclusive - start
            matched = predicted & required
            outside_demo_window = predicted - required
            later_in_demo = outside_demo_window & supported
            demo_onset_later = {
                direction
                for direction in later_in_demo
                if onset_delays[direction] > stop_offset
            }
            outside_demo_event_support = predicted - supported
            opposite = predicted & _opposites(anchor)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "inference_source": INFERENCE_SOURCE,
                    "model": label,
                    "event_id": str(event["event_id"]),
                    "episode_id": episode_id,
                    "event_index": int(event["event_index"]),
                    "onset_step": onset,
                    "window": window,
                    "window_start_offset": start_offset,
                    "window_stop_offset_inclusive": stop_offset,
                    "window_observed_ticks": observed_ticks,
                    "window_complete": observed_ticks == stop_offset - start_offset + 1,
                    "demonstrated_directions": sorted(required, key=_direction_order),
                    "predicted_directions": sorted(predicted, key=_direction_order),
                    "matched_demonstrated_directions": sorted(
                        matched, key=_direction_order
                    ),
                    "outside_demonstrated_window_directions": sorted(
                        outside_demo_window, key=_direction_order
                    ),
                    "single_demo_later_supported_directions": sorted(
                        later_in_demo, key=_direction_order
                    ),
                    "single_demo_direction_onset_later_directions": sorted(
                        demo_onset_later, key=_direction_order
                    ),
                    "outside_single_demo_event_support_directions": sorted(
                        outside_demo_event_support, key=_direction_order
                    ),
                    "opposite_to_single_demo_anchor_directions": sorted(
                        opposite, key=_direction_order
                    ),
                    "demonstrated_count": len(required),
                    "matched_demonstrated_count": len(matched),
                    "single_demo_direction_recall": (
                        len(matched) / len(required) if required else None
                    ),
                    "single_demo_exact_set": predicted == required,
                    "has_outside_demonstrated_window_direction": bool(
                        outside_demo_window
                    ),
                    "has_single_demo_later_supported_direction": bool(later_in_demo),
                    "has_single_demo_direction_onset_later": bool(demo_onset_later),
                    "has_outside_single_demo_event_support_direction": bool(
                        outside_demo_event_support
                    ),
                    "has_opposite_to_single_demo_anchor_direction": bool(opposite),
                    "single_demo_event_support_directions": sorted(
                        supported, key=_direction_order
                    ),
                    "cluster_unit": "episode",
                    "event_rows_non_independent_within_episode": True,
                }
            )

    aggregates: dict[str, Any] = {}
    for scope, selected_events in (
        (
            "first_event",
            {
                str(event["event_id"])
                for event in events
                if int(event["event_index"]) == 0
            },
        ),
        ("all_events", {str(event["event_id"]) for event in events}),
    ):
        aggregates[scope] = {}
        for window, *_ in WINDOW_SPECS:
            selected_rows = [
                row
                for row in rows
                if row["window"] == window and row["event_id"] in selected_events
            ]
            aggregates[scope][window] = aggregate_event_rows(selected_rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "inference_source": INFERENCE_SOURCE,
        "capability_boundaries": {
            "directly_measures": (
                "deadzone-thresholded direction sets in saved teacher-forced "
                "open-loop continuous outputs"
            ),
            "does_not_measure": [
                "latent model intent",
                "closed-loop or state-hold recovery",
                "command realization after projection or safety layers",
                "physical machine response",
                "task success",
                "task-wide behavioral support or correctness",
            ],
            "correctness_estimable": False,
            "task_support_estimable": False,
            "physical_validity_estimable": False,
            "statistical_unit": "event with episode-clustered macro reporting",
            "overlap_warning": (
                "event rows may overlap in episode time and are non-independent; "
                "no frame-micro aggregate is reported"
            ),
            "startup_readiness_proxy": {
                "recording_wait_semantics": (
                    "the pre-expert-onset recording interval may be observation or "
                    "recording preparation and is not idle ground truth"
                ),
                "candidate_rule": (
                    "only the first deadzone-effective policy output from step 0 is "
                    "used; all later teacher-forced outputs are ignored"
                ),
                "post_single_demo_rule": (
                    "a first effective output after the demo onset is labelled "
                    "post_single_demo_trajectory_not_initial_readiness and excluded from "
                    "pre-or-at-onset single-demo-similarity summaries"
                ),
                "no_required_startup_axis": True,
                "single_demo_similarity_only": (
                    "anchor overlap, exact anchor, local support, and opposite "
                    "metrics are descriptive similarity to this expert recording only"
                ),
                "gate_policy": (
                    "startup single-demo-similarity metrics are not promotion or safety gates"
                ),
                "claim_boundary": (
                    "this is a stronger startup-readiness proxy, but no command was "
                    "sent and it cannot prove safety, machine response, or task success"
                ),
            },
            "metric_semantics": {
                "demonstrated": "direction is present in this demo event window",
                "single_demo_later_supported": (
                    "predicted later in this same demo event but absent from the "
                    "current scoring window"
                ),
                "single_demo_direction_onset_later": (
                    "same-demo-supported outside-window prediction whose first "
                    "demonstrated onset is strictly later than the scoring window; "
                    "this is single-demo timing disagreement and "
                    "does not imply unsafe or premature execution"
                ),
                "outside_single_demo_event_support": (
                    "predicted outside this recording's event support horizon; "
                    "task-wide support and correctness remain unknown"
                ),
                "opposite_to_single_demo_anchor": (
                    "predicted opposite to this demo's direction at event onset; "
                    "for later windows this is descriptive and not automatically unsafe"
                ),
            },
        },
        "model": label,
        "sampling_hz": hz,
        "event_count": len(events),
        "episode_count": len(expected_ids),
        "rows": rows,
        "aggregates": aggregates,
        "startup_readiness": _startup_readiness(
            events=events,
            policy_effective=masks,
            sampling_hz=hz,
        ),
    }


def _startup_readiness(
    *,
    events: Sequence[Mapping[str, Any]],
    policy_effective: Mapping[int, np.ndarray],
    sampling_hz: float,
) -> dict[str, Any]:
    first_events: dict[int, Mapping[str, Any]] = {}
    for event in events:
        if int(event["event_index"]) != 0:
            continue
        episode_id = int(event["episode_id"])
        if episode_id in first_events:
            raise ValueError(f"episode_{episode_id} has duplicate first events")
        first_events[episode_id] = event
    expected_ids = set(policy_effective)
    if set(first_events) != expected_ids:
        raise ValueError(
            "first ExpertIntentEvent episode IDs do not exactly match policy episodes"
        )

    episode_rows: list[dict[str, Any]] = []
    for episode_id in sorted(expected_ids):
        event = first_events[episode_id]
        effective = policy_effective[episode_id]
        effective_steps = np.flatnonzero(np.any(effective, axis=(1, 2)))
        expert_onset = int(event["onset_step"])
        anchor = _label_set(event["anchor_intent"], name="anchor_intent")
        supported = _label_set(
            event["single_demo_event_support_directions"],
            name="single_demo_event_support_directions",
        )
        onset_delays = _direction_onset_delays(event, supported=supported)
        if effective_steps.size == 0:
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "event_id": str(event["event_id"]),
                    "status": "none",
                    "first_effective_step": None,
                    "single_demo_first_onset_step": expert_onset,
                    "event_task_support_horizon_requested_ticks": int(
                        event["support_horizon_requested_ticks"]
                    ),
                    "relative_to_single_demo_onset_ticks": None,
                    "relative_to_single_demo_onset_seconds": None,
                    "first_direction_set": [],
                    "included_in_pre_or_at_demo_similarity_summary": False,
                    "first_effective_before_or_at_single_demo_onset": None,
                    "intersects_single_demo_anchor": None,
                    "single_demo_exact_anchor": None,
                    "within_single_demo_local_support": None,
                    "outside_single_demo_local_support_directions": [],
                    "has_outside_single_demo_local_support_direction": None,
                    "opposite_to_single_demo_anchor_directions": [],
                    "has_opposite_to_single_demo_anchor_direction": None,
                    "single_demo_direction_onset_later_directions": [],
                }
            )
            continue

        first_step = int(effective_steps[0])
        predicted = _labels_from_mask(effective[first_step])
        before_or_at = first_step <= expert_onset
        unsupported = predicted - supported
        opposite = predicted & _opposites(anchor)
        onset_early = {
            direction
            for direction in predicted & supported
            if first_step < expert_onset + onset_delays[direction]
        }
        relative_ticks = first_step - expert_onset
        episode_rows.append(
            {
                "episode_id": episode_id,
                "event_id": str(event["event_id"]),
                "status": (
                    "before_or_at_single_demo_onset"
                    if before_or_at
                    else "post_single_demo_trajectory_not_initial_readiness"
                ),
                "first_effective_step": first_step,
                "single_demo_first_onset_step": expert_onset,
                "event_task_support_horizon_requested_ticks": int(
                    event["support_horizon_requested_ticks"]
                ),
                "relative_to_single_demo_onset_ticks": relative_ticks,
                "relative_to_single_demo_onset_seconds": relative_ticks / sampling_hz,
                "first_direction_set": sorted(predicted, key=_direction_order),
                "included_in_pre_or_at_demo_similarity_summary": before_or_at,
                "first_effective_before_or_at_single_demo_onset": before_or_at,
                "intersects_single_demo_anchor": bool(predicted & anchor),
                "single_demo_exact_anchor": predicted == anchor,
                "within_single_demo_local_support": predicted <= supported,
                "outside_single_demo_local_support_directions": sorted(
                    unsupported, key=_direction_order
                ),
                "has_outside_single_demo_local_support_direction": bool(unsupported),
                "opposite_to_single_demo_anchor_directions": sorted(
                    opposite, key=_direction_order
                ),
                "has_opposite_to_single_demo_anchor_direction": bool(opposite),
                "single_demo_direction_onset_later_directions": sorted(
                    onset_early, key=_direction_order
                ),
            }
        )

    episode_count = len(episode_rows)
    effective_rows = [row for row in episode_rows if row["status"] != "none"]
    pre_or_at_rows = [
        row
        for row in episode_rows
        if row["included_in_pre_or_at_demo_similarity_summary"]
    ]
    post_rows = [
        row
        for row in episode_rows
        if row["status"] == "post_single_demo_trajectory_not_initial_readiness"
    ]
    none_count = episode_count - len(effective_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "sampling_hz": sampling_hz,
        "sampling_contract": (
            f"saved open-loop actions are interpreted at {sampling_hz:g} Hz"
        ),
        "episode_count": episode_count,
        "first_effective_count": len(effective_rows),
        "none_count": none_count,
        "none_rate_of_episodes": none_count / episode_count,
        "first_effective_before_or_at_single_demo_onset_count": len(pre_or_at_rows),
        "first_effective_before_or_at_single_demo_onset_rate_of_episodes": (
            len(pre_or_at_rows) / episode_count
        ),
        "post_single_demo_trajectory_not_initial_readiness_count": len(post_rows),
        "post_single_demo_trajectory_not_initial_readiness_rate_of_episodes": (
            len(post_rows) / episode_count
        ),
        "first_candidate_single_demo_similarity": _startup_single_demo_similarity(
            pre_or_at_rows
        ),
        "timing_distribution": _startup_timing_distribution(
            effective_rows, sampling_hz=sampling_hz
        ),
        "episode_rows": episode_rows,
    }


def _startup_single_demo_similarity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    count = len(rows)
    fields = (
        ("anchor_overlap", "intersects_single_demo_anchor"),
        ("exact_anchor", "single_demo_exact_anchor"),
        ("within_local_support", "within_single_demo_local_support"),
        (
            "outside_local_support",
            "has_outside_single_demo_local_support_direction",
        ),
        (
            "opposite_to_anchor",
            "has_opposite_to_single_demo_anchor_direction",
        ),
    )
    result: dict[str, Any] = {"pre_or_at_single_demo_onset_candidate_count": count}
    for output_name, row_field in fields:
        matches = sum(bool(row[row_field]) for row in rows)
        result[f"{output_name}_count"] = matches
        result[f"{output_name}_rate"] = matches / count if count else None
    return result


def _startup_timing_distribution(
    rows: Sequence[Mapping[str, Any]], *, sampling_hz: float
) -> dict[str, Any]:
    ticks = [int(row["relative_to_single_demo_onset_ticks"]) for row in rows]
    if not ticks:
        return {"count": 0, "ticks": None, "seconds": None}
    values = np.asarray(ticks, dtype=np.float64)
    percentiles = {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }
    return {
        "count": len(ticks),
        "ticks": percentiles,
        "seconds": {key: value / sampling_hz for key, value in percentiles.items()},
    }


def aggregate_event_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one event scope/window with an episode macro."""

    if not rows:
        return {
            "event_count": 0,
            "episode_count": 0,
            "single_demo_direction_recall": None,
            "event_mean_single_demo_direction_recall": None,
            "single_demo_exact_set_rate": None,
            "outside_demonstrated_window_rate": None,
            "single_demo_later_supported_rate": None,
            "single_demo_direction_onset_later_rate": None,
            "outside_single_demo_event_support_rate": None,
            "opposite_to_single_demo_anchor_rate": None,
            "incomplete_window_rate": None,
            "episode_macro": {},
            "axis_direction_breakdown": {},
        }
    demonstrated = sum(int(row["demonstrated_count"]) for row in rows)
    matched = sum(int(row["matched_demonstrated_count"]) for row in rows)
    per_event_recall = [
        float(row["single_demo_direction_recall"])
        for row in rows
        if row["single_demo_direction_recall"] is not None
    ]
    result = {
        "event_count": len(rows),
        "episode_count": len({int(row["episode_id"]) for row in rows}),
        "demonstrated_direction_count": demonstrated,
        "matched_demonstrated_direction_count": matched,
        "single_demo_direction_recall": (
            matched / demonstrated if demonstrated else None
        ),
        "event_mean_single_demo_direction_recall": _mean_or_none(per_event_recall),
        "single_demo_exact_set_rate": _binary_rate(rows, "single_demo_exact_set"),
        "outside_demonstrated_window_rate": _binary_rate(
            rows, "has_outside_demonstrated_window_direction"
        ),
        "single_demo_later_supported_rate": _binary_rate(
            rows, "has_single_demo_later_supported_direction"
        ),
        "single_demo_direction_onset_later_rate": _binary_rate(
            rows, "has_single_demo_direction_onset_later"
        ),
        "outside_single_demo_event_support_rate": _binary_rate(
            rows, "has_outside_single_demo_event_support_direction"
        ),
        "opposite_to_single_demo_anchor_rate": _binary_rate(
            rows, "has_opposite_to_single_demo_anchor_direction"
        ),
        "incomplete_window_rate": sum(not bool(row["window_complete"]) for row in rows)
        / len(rows),
        "episode_macro": _episode_macro(rows),
        "axis_direction_breakdown": _axis_direction_breakdown(rows),
        "clustered_non_independent": True,
    }
    return result


def _episode_macro(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_id"])].append(row)
    episode_rows = []
    for episode_id, selected in sorted(grouped.items()):
        demonstrated = sum(int(row["demonstrated_count"]) for row in selected)
        matched = sum(int(row["matched_demonstrated_count"]) for row in selected)
        episode_rows.append(
            {
                "episode_id": episode_id,
                "event_count": len(selected),
                "single_demo_direction_recall": (
                    matched / demonstrated if demonstrated else None
                ),
                "single_demo_exact_set_rate": _binary_rate(
                    selected, "single_demo_exact_set"
                ),
                "outside_demonstrated_window_rate": _binary_rate(
                    selected, "has_outside_demonstrated_window_direction"
                ),
                "single_demo_later_supported_rate": _binary_rate(
                    selected, "has_single_demo_later_supported_direction"
                ),
                "single_demo_direction_onset_later_rate": _binary_rate(
                    selected, "has_single_demo_direction_onset_later"
                ),
                "outside_single_demo_event_support_rate": _binary_rate(
                    selected, "has_outside_single_demo_event_support_direction"
                ),
                "opposite_to_single_demo_anchor_rate": _binary_rate(
                    selected, "has_opposite_to_single_demo_anchor_direction"
                ),
            }
        )
    metric_names = (
        "single_demo_direction_recall",
        "single_demo_exact_set_rate",
        "outside_demonstrated_window_rate",
        "single_demo_later_supported_rate",
        "single_demo_direction_onset_later_rate",
        "outside_single_demo_event_support_rate",
        "opposite_to_single_demo_anchor_rate",
    )
    return {
        "episode_count": len(episode_rows),
        **{
            name: _mean_or_none(
                [float(row[name]) for row in episode_rows if row[name] is not None]
            )
            for name in metric_names
        },
        "episode_rows": episode_rows,
    }


def _axis_direction_breakdown(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int | float | None]]:
    result: dict[str, dict[str, int | float | None]] = {}
    for direction in DIRECTION_LABELS:
        demonstrated_events = sum(
            direction in row["demonstrated_directions"] for row in rows
        )
        matched_events = sum(
            direction in row["matched_demonstrated_directions"] for row in rows
        )
        result[direction] = {
            "demonstrated_events": demonstrated_events,
            "matched_demo_events": matched_events,
            "unmatched_demo_events": demonstrated_events - matched_events,
            "single_demo_recall": (
                matched_events / demonstrated_events if demonstrated_events else None
            ),
            "outside_demo_window_events": sum(
                direction in row["outside_demonstrated_window_directions"]
                for row in rows
            ),
            "single_demo_later_supported_events": sum(
                direction in row["single_demo_later_supported_directions"]
                for row in rows
            ),
            "single_demo_direction_onset_later_events": sum(
                direction in row["single_demo_direction_onset_later_directions"]
                for row in rows
            ),
            "outside_single_demo_event_support_events": sum(
                direction in row["outside_single_demo_event_support_directions"]
                for row in rows
            ),
            "opposite_to_single_demo_anchor_events": sum(
                direction in row["opposite_to_single_demo_anchor_directions"]
                for row in rows
            ),
        }
    return result


def _validate_event(event: Mapping[str, Any]) -> None:
    if event.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise ValueError(
            f"event {event.get('event_id')} has unsupported schema_version: "
            f"{event.get('schema_version')}"
        )
    for field in (
        "anchor_intent",
        "immediate_intent_0_1",
        "near_intent_2_5",
        "near_intent_6_10",
        "single_demo_event_support_directions",
    ):
        _label_set(event.get(field), name=field)


def _validate_event_bounds(event: Mapping[str, Any], *, total_steps: int) -> None:
    onset = int(event["onset_step"])
    end = int(event["support_end_step_exclusive"])
    requested = int(event["support_horizon_requested_ticks"])
    observed = int(event["support_horizon_observed_ticks"])
    if not 0 <= onset < total_steps:
        raise ValueError(f"{event['event_id']} onset_step is out of bounds")
    expected_end = min(total_steps, onset + requested)
    if end != expected_end or observed != end - onset:
        raise ValueError(f"{event['event_id']} support/window bounds are inconsistent")


def _direction_onset_delays(
    event: Mapping[str, Any], *, supported: set[str]
) -> dict[str, int]:
    details = event.get("direction_details")
    if isinstance(details, (str, bytes)) or not isinstance(details, Sequence):
        raise ValueError(f"{event['event_id']} direction_details must be a list")
    result: dict[str, int] = {}
    for detail in details:
        if not isinstance(detail, Mapping):
            raise ValueError(f"{event['event_id']} direction detail must be a mapping")
        direction = str(detail.get("direction", ""))
        if direction not in DIRECTION_LABELS or direction not in supported:
            raise ValueError(
                f"{event['event_id']} direction detail is outside single-demo support: "
                f"{direction!r}"
            )
        if direction in result:
            raise ValueError(
                f"{event['event_id']} has duplicate direction detail: {direction}"
            )
        delay = detail.get("onset_delay_ticks")
        if isinstance(delay, bool) or not isinstance(delay, Integral):
            raise ValueError(
                f"{event['event_id']} has unusable onset delay for {direction}"
            )
        result[direction] = int(delay)
    if set(result) != supported:
        missing = sorted(supported - set(result), key=_direction_order)
        raise ValueError(
            f"{event['event_id']} direction_details do not cover single-demo support: "
            f"{missing}"
        )
    return result


def _validate_action_array(action: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"{name} must have shape (T, {len(AXIS_NAMES)})")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _label_set(values: Any, *, name: str) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a direction-label list")
    labels = [str(value) for value in values]
    if len(labels) != len(set(labels)) or not set(labels) <= set(DIRECTION_LABELS):
        raise ValueError(f"{name} contains invalid or duplicate directions: {labels}")
    return set(labels)


def _labels_from_mask(mask: np.ndarray) -> set[str]:
    labels: set[str] = set()
    for axis_index, axis in enumerate(AXIS_NAMES):
        if bool(mask[axis_index, 0]):
            labels.add(f"{axis}+")
        if bool(mask[axis_index, 1]):
            labels.add(f"{axis}-")
    return labels


def _opposites(labels: set[str]) -> set[str]:
    return {f"{label[:-1]}{'-' if label[-1] == '+' else '+'}" for label in labels}


def _direction_order(label: str) -> int:
    return DIRECTION_LABELS.index(label)


def _binary_rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


__all__ = [
    "DIRECTION_LABELS",
    "INFERENCE_SOURCE",
    "SCHEMA_VERSION",
    "WINDOW_SPECS",
    "aggregate_event_rows",
    "evaluate_open_loop_intent",
]
