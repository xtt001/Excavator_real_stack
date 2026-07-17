"""Natural startup activation under an explicitly armed frozen observation.

This diagnostic advances policy state through an observe-only recording prefix,
then repeatedly presents the final demo-ineffective observation.  Any axis may
start.  Single-demo comparisons are descriptive only and never alter liveness.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np

from testbed.data.expert_intent_events import (
    SCHEMA_VERSION as EVENT_SCHEMA_VERSION,
)
from testbed.policies.deadzone_eval import AXIS_NAMES, effective_direction_mask
from testbed.policies.state_hold_demo_target import StatefulStepSource, StepOutput

SCHEMA_VERSION = "startup_activation_v2"
CURVE_TICKS = (1, 3, 5, 10, 20)


def evaluate_startup_activation(
    *,
    episode_id: int | str,
    first_event: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Mapping[str, float]],
    step_source: StatefulStepSource,
    hold_horizon_steps: int,
    sampling_hz: float,
    instrumentation: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate first natural effective action after observe-only warmup and arm."""

    _validate_first_event(first_event, episode_id=episode_id)
    horizon = int(hold_horizon_steps)
    if horizon <= 0:
        raise ValueError("hold_horizon_steps must be positive")
    hz = float(sampling_hz)
    if not np.isfinite(hz) or hz <= 0.0:
        raise ValueError("sampling_hz must be finite and positive")
    onset = int(first_event["onset_step"])
    arm_step = max(0, onset - 1)
    if len(observations) <= arm_step:
        raise ValueError(
            f"observations do not cover arm_step {arm_step}: length={len(observations)}"
        )

    step_source.reset()
    warmup_effective_ticks: list[int] = []
    for step in range(arm_step):
        observation = _zero_previous_command(observations[step])
        action = _step_action(step_source, observation)
        _increment(instrumentation, "source_step_calls")
        if _effective_labels(action, thresholds):
            warmup_effective_ticks.append(step)

    frozen = _frozen_arm_observation(observations[arm_step])
    arm_qpos = _finite_axis_vector(frozen.get("qpos"), name="arm qpos").copy()
    first_action: np.ndarray | None = None
    first_directions: set[str] = set()
    delay: int | None = None
    for tick in range(horizon):
        action = _step_action(step_source, copy.deepcopy(frozen))
        _increment(instrumentation, "source_step_calls")
        directions = _effective_labels(action, thresholds)
        if directions:
            first_action = action
            first_directions = directions
            delay = tick
            break

    anchor = _direction_set(first_event.get("anchor_intent"), name="anchor_intent")
    support = _direction_set(
        first_event.get("single_demo_event_support_directions"),
        name="single_demo_event_support_directions",
    )
    if not anchor <= support:
        raise ValueError("first-event anchor intent is outside single-demo support")
    outside_support = first_directions - support
    opposite = first_directions & _opposite_directions(anchor)
    liveness = delay is not None
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": int(episode_id),
        "event_id": str(first_event["event_id"]),
        "single_demo_first_onset_step": onset,
        "arm_step": arm_step,
        "arm_reference_semantics": "last_single_demo_ineffective_frame_reference",
        "arm_reference_is_ground_truth_go_signal": False,
        "warmup_ticks": arm_step,
        "warmup_observe_only": True,
        "warmup_commands_suppressed": True,
        "warmup_effective_output_ticks": warmup_effective_ticks,
        "warmup_any_effective_output": bool(warmup_effective_ticks),
        "warmup_effective_outputs_ignored_for_liveness": True,
        "frozen_observation_step": arm_step,
        "frozen_observation_repeated": True,
        "frozen_qpos": [float(value) for value in arm_qpos],
        "frozen_qvel_zero": True,
        "frozen_previous_final_command_zero_when_present": True,
        "hold_horizon_steps": horizon,
        "sampling_hz": hz,
        "status": "effective_action" if liveness else "horizon_no_effective_action",
        "natural_liveness": liveness,
        "activation_delay_ticks": delay,
        "activation_delay_seconds": delay / hz if delay is not None else None,
        "ticks_evaluated_after_arm": delay + 1 if delay is not None else horizon,
        "first_action_vector": (
            [float(value) for value in first_action]
            if first_action is not None
            else None
        ),
        "first_direction_set": _ordered_directions(first_directions),
        "startup_axis_requirement": "none",
        "single_demo_similarity_only": True,
        "promotion_gate": False,
        "safety_gate": False,
        "single_demo_anchor_directions": _ordered_directions(anchor),
        "single_demo_local_support_directions": _ordered_directions(support),
        "single_demo_similarity": {
            "exact_anchor": liveness and first_directions == anchor,
            "overlap_anchor": bool(first_directions & anchor),
            "wholly_within_local_support": liveness and first_directions <= support,
            "outside_local_support_directions": _ordered_directions(outside_support),
            "opposite_to_anchor_directions": _ordered_directions(opposite),
        },
    }
    for ticks in CURVE_TICKS:
        row[f"within_{ticks}_ticks"] = delay is not None and delay < ticks
    return row


def aggregate_startup_activation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate episode-level liveness without turning expert match into a gate."""

    if not rows:
        raise ValueError("startup activation rows must not be empty")
    episode_ids = [int(row["episode_id"]) for row in rows]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("startup activation rows contain duplicate episode IDs")
    total = len(rows)
    live_rows = [row for row in rows if bool(row["natural_liveness"])]
    delays = [int(row["activation_delay_ticks"]) for row in live_rows]
    direction_counts = Counter(
        ",".join(str(value) for value in row["first_direction_set"])
        for row in live_rows
    )
    similarity_fields = (
        "exact_anchor",
        "overlap_anchor",
        "wholly_within_local_support",
    )
    result: dict[str, Any] = {
        "episode_count": total,
        "natural_liveness_count": len(live_rows),
        "natural_liveness_rate": len(live_rows) / total,
        "horizon_no_effective_action_count": total - len(live_rows),
        "activation_delay_ticks": {
            "count": len(delays),
            "min": min(delays) if delays else None,
            "median": float(np.median(delays)) if delays else None,
            "mean": float(np.mean(delays)) if delays else None,
            "max": max(delays) if delays else None,
        },
        "first_direction_set_counts": dict(sorted(direction_counts.items())),
        "warmup_any_effective_output_count": sum(
            bool(row["warmup_any_effective_output"]) for row in rows
        ),
        "startup_axis_requirement": "none",
        "single_demo_similarity_only": True,
        "promotion_gate": False,
        "safety_gate": False,
        "single_demo_similarity_live_denominator": len(live_rows),
    }
    for ticks in CURVE_TICKS:
        count = sum(bool(row[f"within_{ticks}_ticks"]) for row in rows)
        result[f"within_{ticks}_ticks"] = {
            "count": count,
            "rate": count / total,
            "semantics": f"activation_delay_ticks < {ticks}",
        }
    for field in similarity_fields:
        count = sum(bool(row["single_demo_similarity"][field]) for row in live_rows)
        result[f"single_demo_{field}"] = {
            "count": count,
            "rate_among_live": count / len(live_rows) if live_rows else None,
        }
    result["outside_single_demo_local_support_count"] = sum(
        bool(row["single_demo_similarity"]["outside_local_support_directions"])
        for row in live_rows
    )
    result["opposite_to_single_demo_anchor_count"] = sum(
        bool(row["single_demo_similarity"]["opposite_to_anchor_directions"])
        for row in live_rows
    )
    return result


def capability_boundaries() -> dict[str, Any]:
    return {
        "directly_measures": (
            "whether local raw policy inference emits any deadzone-effective action "
            "after observe-only warmup and explicit arm under one frozen observation"
        ),
        "startup_axis_requirement": "none",
        "single_demo_similarity_only": True,
        "promotion_gate": False,
        "safety_gate": False,
        "correctness_estimable": False,
        "task_support_estimable": False,
        "physical_validity_estimable": False,
        "does_not_measure": [
            "physical machine response",
            "safety",
            "task success",
            "a unique correct startup axis",
            "task-wide behavioral support",
            "terrain generalization",
            "closed-loop rollout",
        ],
        "arm_reference_warning": (
            "last_single_demo_ineffective_frame_reference is a reproducible evaluation "
            "reference, not a ground-truth go signal"
        ),
        "command_claim": "no command is sent by this offline diagnostic",
    }


def _validate_first_event(event: Mapping[str, Any], *, episode_id: int | str) -> None:
    if str(event.get("schema_version")) != EVENT_SCHEMA_VERSION:
        raise ValueError("startup activation requires a single-demo intent v2 event")
    if int(event.get("event_index", -1)) != 0:
        raise ValueError("startup activation requires event_index == 0")
    if str(event.get("split")) != "validation":
        raise ValueError("startup activation requires a validation event")
    if int(event.get("episode_id", -1)) != int(episode_id):
        raise ValueError("first event episode_id does not match requested episode")
    if int(event.get("onset_step", -1)) < 0:
        raise ValueError("first event onset_step must be nonnegative")


def _step_action(
    step_source: StatefulStepSource,
    observation: Mapping[str, Any],
) -> np.ndarray:
    output = step_source.step(observation)
    action = output.action if isinstance(output, StepOutput) else output
    return _finite_axis_vector(action, name="policy action")


def _finite_axis_vector(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (len(AXIS_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite ({len(AXIS_NAMES)},) vector")
    return result


def _zero_previous_command(observation: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(observation))
    if "previous_final_command" in result:
        command = _finite_axis_vector(
            result["previous_final_command"], name="previous_final_command"
        )
        result["previous_final_command"] = np.zeros_like(command)
    return result


def _frozen_arm_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    result = _zero_previous_command(observation)
    qvel = _finite_axis_vector(result.get("qvel"), name="arm qvel")
    result["qvel"] = np.zeros_like(qvel)
    _finite_axis_vector(result.get("qpos"), name="arm qpos")
    return result


def _effective_labels(
    action: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
) -> set[str]:
    mask = effective_direction_mask(action.reshape(1, -1), dict(thresholds))[0]
    labels: set[str] = set()
    for index, axis in enumerate(AXIS_NAMES):
        if bool(mask[index, 0]):
            labels.add(f"{axis}+")
        if bool(mask[index, 1]):
            labels.add(f"{axis}-")
    return labels


def _direction_set(value: Any, *, name: str) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a direction sequence")
    result = {str(item) for item in value}
    valid = {f"{axis}{sign}" for axis in AXIS_NAMES for sign in ("+", "-")}
    invalid = sorted(result - valid)
    if invalid:
        raise ValueError(f"{name} contains invalid direction(s): {invalid}")
    return result


def _opposite_directions(directions: set[str]) -> set[str]:
    return {
        f"{direction[:-1]}{'-' if direction.endswith('+') else '+'}"
        for direction in directions
    }


def _ordered_directions(directions: set[str]) -> list[str]:
    order = {
        f"{axis}{sign}": index
        for index, (axis, sign) in enumerate(
            axis_sign for axis in AXIS_NAMES for axis_sign in ((axis, "+"), (axis, "-"))
        )
    }
    return sorted(directions, key=order.__getitem__)


def _increment(instrumentation: MutableMapping[str, Any] | None, key: str) -> None:
    if instrumentation is not None:
        instrumentation[key] = int(instrumentation.get(key, 0)) + 1
