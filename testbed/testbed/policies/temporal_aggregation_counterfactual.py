"""Rescore saved ACT aggregation traces without changing held observations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from testbed.policies.action_start_distribution import FORBIDDEN_HELDOUT
from testbed.policies.deadzone_eval import AXIS_NAMES, effective_direction_mask
from testbed.policies.state_hold_demo_relation import (
    evaluate_state_hold_trace_demo_relation,
)

MODES = ("legacy", "newest", "recency")
_DIRECTIONS = ("pos", "neg")
_ACTION_KEYS = {
    "legacy": "policy_temporal_aggregation_legacy_action",
    "newest": "policy_temporal_aggregation_newest_action",
    "recency": "policy_temporal_aggregation_recency_action",
}
_DIAGNOSTIC_KEYS = (
    "policy_temporal_aggregation_action_domain",
    "policy_temporal_aggregation_exponential_k",
    "policy_action_scale",
    "policy_deadzone_assist_enabled",
    "policy_error",
    "policy_temporal_aggregation_query_step",
    "policy_temporal_aggregation_source_steps",
    "policy_temporal_aggregation_query_offsets",
    "policy_temporal_aggregation_population",
    *_ACTION_KEYS.values(),
)


def evaluate_temporal_aggregation_counterfactual(
    *,
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Return per-anchor and aggregate command counterfactual metrics."""

    if not rows_by_model:
        raise ValueError("rows_by_model must not be empty")
    normalized_thresholds = {
        axis: {
            direction: float(thresholds[axis][direction]) for direction in _DIRECTIONS
        }
        for axis in AXIS_NAMES
    }
    per_anchor: list[dict[str, Any]] = []
    anchor_ids_by_model: dict[str, set[str]] = {}
    model_labels: list[str] = []
    for raw_model, source_rows in rows_by_model.items():
        model = str(raw_model).strip()
        if not model or model in anchor_ids_by_model:
            raise ValueError(f"invalid or duplicate model label: {raw_model!r}")
        if not source_rows:
            raise ValueError(f"model {model!r} has no anchor rows")
        model_labels.append(model)
        model_rows = [
            _evaluate_anchor(
                model=model,
                source_row=row,
                thresholds=normalized_thresholds,
            )
            for row in source_rows
        ]
        anchor_ids = {str(row["anchor_id"]) for row in model_rows}
        if len(anchor_ids) != len(model_rows):
            raise ValueError(f"model {model!r} contains duplicate anchor ids")
        anchor_ids_by_model[model] = anchor_ids
        per_anchor.extend(model_rows)

    reference_model = model_labels[0]
    reference_anchor_ids = anchor_ids_by_model[reference_model]
    for model in model_labels[1:]:
        if anchor_ids_by_model[model] != reference_anchor_ids:
            raise ValueError(
                f"model {model!r} anchor ids differ from {reference_model!r}"
            )

    per_anchor.sort(key=_anchor_sort_key)
    return {
        "schema_version": 1,
        "contract": "temporal_aggregation_command_counterfactual_v1",
        "action_domain": "direct_policy_output",
        "selected_action_mode": "legacy",
        "counterfactual_modes": list(MODES),
        "model_labels": model_labels,
        "anchor_ids": sorted(reference_anchor_ids, key=_anchor_id_sort_key),
        "anchors_per_model": len(reference_anchor_ids),
        "forbidden_heldout_episode_ids": sorted(FORBIDDEN_HELDOUT),
        "heldout_evaluated": False,
        "mechanical_assist": {
            "estimated": False,
            "reason": (
                "stored traces are assist-disabled and do not prove equivalent "
                "assist priming for alternative commands"
            ),
        },
        "limitations": {
            "observation_control": (
                "newest and recency actions did not control observations; all "
                "modes are rescored on the same held-observation branch"
            ),
            "scope": (
                "this is an exact command counterfactual for the stored held "
                "branch, not evidence of live closed-loop behavior"
            ),
            "demo_target_semantics": (
                "target reproduction and anchor-extra actions are relations to one "
                "demo anchor, not task correctness or invalid-action labels"
            ),
            "correctness_estimable": False,
            "task_support_estimable": False,
            "physical_validity_estimable": False,
        },
        "per_anchor": per_anchor,
        "aggregate": {
            model: _aggregate_model(
                [row for row in per_anchor if row["model"] == model]
            )
            for model in model_labels
        },
    }


def _evaluate_anchor(
    *,
    model: str,
    source_row: Mapping[str, Any],
    thresholds: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    required = (
        "episode_id",
        "anchor_step",
        "anchor_group",
        "axis_index",
        "axis",
        "direction",
        "deadzone_threshold",
        "expert_action_vector",
        "hold_horizon_steps",
        "trace_full_horizon_after_reproduction",
        "state_hold_qvel_zero",
        "teacher_forced_status",
        "state_hold_status",
        "state_hold_demo_target_reproduction_delay_ticks",
        "state_hold_ticks_evaluated",
        "state_hold_full_horizon_complete",
        "state_hold_action_trace",
        "state_hold_diagnostics_trace",
        "state_hold_temporal_aggregation_decomposition_complete",
        "state_hold_temporal_aggregation_decomposition_ticks_recorded",
    )
    missing = [key for key in required if key not in source_row]
    if missing:
        raise ValueError("anchor row is missing field(s): " + ", ".join(missing))

    episode_id = str(source_row["episode_id"])
    if _episode_number(episode_id) in FORBIDDEN_HELDOUT:
        raise ValueError(f"held-out episode is forbidden: {episode_id}")
    anchor_step = int(source_row["anchor_step"])
    axis_index = int(source_row["axis_index"])
    if not 0 <= axis_index < len(AXIS_NAMES):
        raise ValueError(f"axis_index is out of range: {axis_index}")
    axis = str(source_row["axis"])
    if axis != AXIS_NAMES[axis_index]:
        raise ValueError(f"axis/axis_index mismatch: {axis}/{axis_index}")
    direction = str(source_row["direction"])
    if direction not in _DIRECTIONS:
        raise ValueError(f"invalid anchor direction: {direction!r}")
    expected_threshold = float(thresholds[axis][direction])
    if not np.isclose(
        float(source_row["deadzone_threshold"]),
        expected_threshold,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("anchor deadzone threshold does not match supplied table")

    horizon = int(source_row["hold_horizon_steps"])
    if horizon <= 0:
        raise ValueError("hold_horizon_steps must be positive")
    if not bool(source_row["trace_full_horizon_after_reproduction"]):
        raise ValueError("counterfactual requires full-horizon reproduction traces")
    if not bool(source_row["state_hold_qvel_zero"]):
        raise ValueError("counterfactual requires state-hold qvel zero")
    if not bool(source_row["state_hold_full_horizon_complete"]):
        raise ValueError("state-hold trace is incomplete")
    if int(source_row["state_hold_ticks_evaluated"]) != horizon:
        raise ValueError("state-hold tick count does not match hold horizon")
    if not bool(source_row["state_hold_temporal_aggregation_decomposition_complete"]):
        raise ValueError("temporal aggregation decomposition is incomplete")
    if (
        int(source_row["state_hold_temporal_aggregation_decomposition_ticks_recorded"])
        != horizon
    ):
        raise ValueError("decomposition tick count does not match hold horizon")

    selected_trace = _validate_action_trace(
        source_row["state_hold_action_trace"],
        expected_steps=horizon,
        name="state_hold_action_trace",
    )
    diagnostics_trace = source_row["state_hold_diagnostics_trace"]
    if not isinstance(diagnostics_trace, Sequence) or isinstance(
        diagnostics_trace, (str, bytes)
    ):
        raise ValueError("state_hold_diagnostics_trace must be a sequence")
    if len(diagnostics_trace) != horizon:
        raise ValueError("state_hold_diagnostics_trace is incomplete")

    mode_actions = {
        mode: np.zeros((horizon, len(AXIS_NAMES)), dtype=np.float32) for mode in MODES
    }
    query_steps: list[int] = []
    source_steps_by_tick: list[list[int]] = []
    for delay, raw_diagnostics in enumerate(diagnostics_trace):
        if not isinstance(raw_diagnostics, Mapping):
            raise ValueError(f"diagnostics tick {delay} must be a mapping")
        diagnostics = raw_diagnostics
        missing_diagnostics = [
            key for key in _DIAGNOSTIC_KEYS if key not in diagnostics
        ]
        if missing_diagnostics:
            raise ValueError(
                f"diagnostics tick {delay} is missing field(s): "
                + ", ".join(missing_diagnostics)
            )
        if diagnostics.get("policy_temporal_aggregation_action_domain") != (
            "direct_policy_output"
        ):
            raise ValueError("aggregation action domain must be direct_policy_output")
        if not np.isclose(
            float(diagnostics.get("policy_temporal_aggregation_exponential_k")),
            0.01,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("aggregation exponential k must be 0.01")
        if np.asarray(
            diagnostics.get("policy_action_scale"), dtype=np.float32
        ).tolist() != [1.0, 1.0, 1.0, 1.0]:
            raise ValueError("policy action scale must be identity")
        if int(diagnostics.get("policy_deadzone_assist_enabled", -1)) != 0:
            raise ValueError("counterfactual requires assist-disabled traces")
        if str(diagnostics.get("policy_error", "")):
            raise ValueError("source policy trace contains an inference error")

        query_step = int(diagnostics["policy_temporal_aggregation_query_step"])
        if query_step != anchor_step + delay:
            raise ValueError("aggregation query steps are not anchor-causal")
        source_steps = [
            int(value)
            for value in diagnostics["policy_temporal_aggregation_source_steps"]
        ]
        if (
            not source_steps
            or source_steps != sorted(set(source_steps))
            or any(value < 0 or value > query_step for value in source_steps)
        ):
            raise ValueError("aggregation trace contains noncausal source steps")
        expected_offsets = [query_step - value for value in source_steps]
        query_offsets = [
            int(value)
            for value in diagnostics["policy_temporal_aggregation_query_offsets"]
        ]
        if query_offsets != expected_offsets:
            raise ValueError("aggregation query offsets do not match source steps")
        if int(diagnostics["policy_temporal_aggregation_population"]) != len(
            source_steps
        ):
            raise ValueError("aggregation population does not match source steps")
        query_steps.append(query_step)
        source_steps_by_tick.append(source_steps)
        for mode, action_key in _ACTION_KEYS.items():
            mode_actions[mode][delay] = _validate_action(
                diagnostics[action_key], name=f"{mode} action at delay {delay}"
            )

    if not np.array_equal(mode_actions["legacy"], selected_trace):
        raise ValueError("legacy diagnostic action differs from selected action trace")

    expert_action = _validate_action(
        source_row["expert_action_vector"], name="expert_action_vector"
    )
    teacher_status = str(source_row["teacher_forced_status"])
    mode_results = {
        mode: _evaluate_mode(
            action_trace=actions,
            expert_action=expert_action,
            thresholds=thresholds,
            target_axis_index=axis_index,
            target_direction=direction,
            teacher_status=teacher_status,
        )
        for mode, actions in mode_actions.items()
    }
    _validate_stored_legacy(source_row=source_row, result=mode_results["legacy"])

    anchor_id = f"{episode_id}:{anchor_step}:{axis}{'+' if direction == 'pos' else '-'}"
    return {
        "model": model,
        "anchor_id": anchor_id,
        "episode_id": episode_id,
        "anchor_step": anchor_step,
        "anchor_group": str(source_row["anchor_group"]),
        "axis_index": axis_index,
        "axis": axis,
        "direction": direction,
        "deadzone_threshold": expected_threshold,
        "expert_action_vector": expert_action.astype(float).tolist(),
        "stored_teacher_forced_status": teacher_status,
        "trace_provenance": {
            "hold_horizon_steps": horizon,
            "diagnostic_ticks": len(diagnostics_trace),
            "query_steps": query_steps,
            "source_steps_by_tick": source_steps_by_tick,
            "action_domain": "direct_policy_output",
            "exponential_k": 0.01,
            "selected_action_mode": "legacy",
            "alternatives_controlled_observations": False,
        },
        "modes": mode_results,
        "comparisons": {
            f"{mode}_vs_legacy": _compare_with_legacy(
                legacy=mode_results["legacy"], alternative=mode_results[mode]
            )
            for mode in ("newest", "recency")
        },
    }


def _evaluate_mode(
    *,
    action_trace: np.ndarray,
    expert_action: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
    target_axis_index: int,
    target_direction: str,
    teacher_status: str,
) -> dict[str, Any]:
    effective = effective_direction_mask(action_trace, dict(thresholds))
    target_direction_index = 0 if target_direction == "pos" else 1
    reproduction_ticks = np.flatnonzero(
        effective[:, target_axis_index, target_direction_index]
    )
    reproduced = bool(reproduction_ticks.size)
    delay = int(reproduction_ticks[0]) if reproduced else None
    demo_relation = evaluate_state_hold_trace_demo_relation(
        expert_action=expert_action,
        action_trace=action_trace,
        thresholds=thresholds,
        target_axis_index=target_axis_index,
        target_direction=target_direction,
    )
    direction_counts, axis_counts = _anchor_extra_breakdown(
        expert_action=expert_action,
        action_trace=action_trace,
        thresholds=thresholds,
    )
    return {
        "status": (
            "demo_target_reproduced" if reproduced else "demo_target_not_reproduced"
        ),
        "demo_target_reproduced": reproduced,
        "demo_target_not_reproduced": not reproduced,
        "first_demo_target_reproduction_delay_ticks": delay,
        "demo_target_reproduction_hidden_by_stored_teacher_forcing": bool(
            teacher_status == "demo_target_reproduced" and not reproduced
        ),
        "anchor_extra_effective": bool(
            demo_relation["state_hold_anchor_extra_effective"]
        ),
        "anchor_extra_effective_tick_count": int(
            demo_relation["state_hold_anchor_extra_effective_tick_count"]
        ),
        "anchor_extra_effective_direction_activation_count": int(
            demo_relation["state_hold_anchor_extra_effective_direction_count"]
        ),
        "anchor_extra_effective_tick_indices": list(
            demo_relation["state_hold_anchor_extra_effective_tick_indices"]
        ),
        "anchor_extra_effective_directions": list(
            demo_relation["state_hold_anchor_extra_effective_directions"]
        ),
        "anchor_extra_effective_direction_tick_counts": direction_counts,
        "anchor_extra_effective_axis_tick_counts": axis_counts,
        "opposite_to_demo_target_tick_count": int(
            demo_relation["state_hold_opposite_to_demo_target_tick_count"]
        ),
        "demo_target_effective_tick_count": int(
            demo_relation["state_hold_demo_target_effective_tick_count"]
        ),
        "direction_flip_count": int(demo_relation["state_hold_direction_flip_count"]),
        "max_effective_axes": int(demo_relation["state_hold_max_effective_axes"]),
    }


def _anchor_extra_breakdown(
    *,
    expert_action: np.ndarray,
    action_trace: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, int], dict[str, int]]:
    expert_effective = effective_direction_mask(
        expert_action.reshape(1, -1), dict(thresholds)
    )[0]
    trace_effective = effective_direction_mask(action_trace, dict(thresholds))
    anchor_extra = trace_effective & ~expert_effective[None, :, :]
    direction_counts = {
        f"{axis}{'+' if direction == 'pos' else '-'}": int(
            np.count_nonzero(anchor_extra[:, axis_index, direction_index])
        )
        for axis_index, axis in enumerate(AXIS_NAMES)
        for direction_index, direction in enumerate(_DIRECTIONS)
    }
    axis_counts = {
        axis: int(np.count_nonzero(anchor_extra[:, axis_index, :].any(axis=1)))
        for axis_index, axis in enumerate(AXIS_NAMES)
    }
    return direction_counts, axis_counts


def _validate_stored_legacy(
    *, source_row: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    expected_delay = source_row["state_hold_demo_target_reproduction_delay_ticks"]
    if expected_delay is not None:
        expected_delay = int(expected_delay)
    checks = {
        "state_hold_status": result["status"],
        "state_hold_demo_target_not_reproduced": result["demo_target_not_reproduced"],
        "state_hold_demo_target_reproduction_delay_ticks": result[
            "first_demo_target_reproduction_delay_ticks"
        ],
        "demo_target_reproduction_hidden_by_teacher_forcing": result[
            "demo_target_reproduction_hidden_by_stored_teacher_forcing"
        ],
        "state_hold_anchor_extra_effective": result["anchor_extra_effective"],
        "state_hold_anchor_extra_effective_tick_count": result[
            "anchor_extra_effective_tick_count"
        ],
        "state_hold_anchor_extra_effective_direction_count": result[
            "anchor_extra_effective_direction_activation_count"
        ],
        "state_hold_opposite_to_demo_target_tick_count": result[
            "opposite_to_demo_target_tick_count"
        ],
        "state_hold_direction_flip_count": result["direction_flip_count"],
    }
    normalized_source = dict(source_row)
    normalized_source["state_hold_demo_target_reproduction_delay_ticks"] = (
        expected_delay
    )
    for key, expected in checks.items():
        if key not in normalized_source or normalized_source[key] != expected:
            raise ValueError(f"stored legacy metric mismatch for {key}")


def _compare_with_legacy(
    *, legacy: Mapping[str, Any], alternative: Mapping[str, Any]
) -> dict[str, Any]:
    legacy_reproduced = bool(legacy["demo_target_reproduced"])
    alternative_reproduced = bool(alternative["demo_target_reproduced"])
    both_reproduced = legacy_reproduced and alternative_reproduced
    delta = None
    if both_reproduced:
        delta = int(alternative["first_demo_target_reproduction_delay_ticks"]) - int(
            legacy["first_demo_target_reproduction_delay_ticks"]
        )
    return {
        "legacy_nonreproduction_changed_to_reproduction": bool(
            not legacy_reproduced and alternative_reproduced
        ),
        "legacy_reproduction_changed_to_nonreproduction": bool(
            legacy_reproduced and not alternative_reproduced
        ),
        "both_reproduced_demo_target": both_reproduced,
        "alternative_faster": bool(delta is not None and delta < 0),
        "alternative_slower": bool(delta is not None and delta > 0),
        "same_reproduction_delay": bool(delta == 0) if delta is not None else False,
        "reproduction_delay_delta_ticks": delta,
    }


def _aggregate_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "anchors": len(rows),
        "modes": {
            mode: _aggregate_mode([row["modes"][mode] for row in rows])
            for mode in MODES
        },
        "comparisons": {
            f"{mode}_vs_legacy": _aggregate_comparison(
                rows=rows, comparison_key=f"{mode}_vs_legacy"
            )
            for mode in ("newest", "recency")
        },
    }


def _aggregate_mode(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    delays = [
        int(result["first_demo_target_reproduction_delay_ticks"])
        for result in results
        if result["first_demo_target_reproduction_delay_ticks"] is not None
    ]
    direction_counts: Counter[str] = Counter()
    axis_counts: Counter[str] = Counter()
    for result in results:
        direction_counts.update(result["anchor_extra_effective_direction_tick_counts"])
        axis_counts.update(result["anchor_extra_effective_axis_tick_counts"])
    return {
        "demo_target_reproduced_anchors": sum(
            bool(result["demo_target_reproduced"]) for result in results
        ),
        "demo_target_not_reproduced_anchors": sum(
            bool(result["demo_target_not_reproduced"]) for result in results
        ),
        "demo_target_reproduction_hidden_by_stored_teacher_forcing_anchors": sum(
            bool(result["demo_target_reproduction_hidden_by_stored_teacher_forcing"])
            for result in results
        ),
        "first_demo_target_reproduction_delay_ticks": _distribution(delays),
        "anchor_extra_effective_anchors": sum(
            bool(result["anchor_extra_effective"]) for result in results
        ),
        "anchor_extra_effective_ticks": sum(
            int(result["anchor_extra_effective_tick_count"]) for result in results
        ),
        "anchor_extra_effective_direction_activations": sum(
            int(result["anchor_extra_effective_direction_activation_count"])
            for result in results
        ),
        "anchor_extra_effective_direction_tick_counts": dict(direction_counts),
        "anchor_extra_effective_axis_tick_counts": dict(axis_counts),
        "opposite_to_demo_target_ticks": sum(
            int(result["opposite_to_demo_target_tick_count"]) for result in results
        ),
        "direction_flips": sum(
            int(result["direction_flip_count"]) for result in results
        ),
        "max_effective_axes": max(
            (int(result["max_effective_axes"]) for result in results), default=0
        ),
    }


def _aggregate_comparison(
    *, rows: list[dict[str, Any]], comparison_key: str
) -> dict[str, Any]:
    comparisons = [row["comparisons"][comparison_key] for row in rows]

    def anchor_ids(key: str) -> list[str]:
        return [
            str(row["anchor_id"])
            for row in rows
            if bool(row["comparisons"][comparison_key][key])
        ]

    deltas = [
        int(comparison["reproduction_delay_delta_ticks"])
        for comparison in comparisons
        if comparison["reproduction_delay_delta_ticks"] is not None
    ]
    return {
        "legacy_nonreproduction_changed_to_reproduction_anchors": len(
            anchor_ids("legacy_nonreproduction_changed_to_reproduction")
        ),
        "legacy_nonreproduction_changed_to_reproduction_anchor_ids": anchor_ids(
            "legacy_nonreproduction_changed_to_reproduction"
        ),
        "legacy_reproduction_changed_to_nonreproduction_anchors": len(
            anchor_ids("legacy_reproduction_changed_to_nonreproduction")
        ),
        "legacy_reproduction_changed_to_nonreproduction_anchor_ids": anchor_ids(
            "legacy_reproduction_changed_to_nonreproduction"
        ),
        "both_reproduced_demo_target_anchors": sum(
            bool(comparison["both_reproduced_demo_target"])
            for comparison in comparisons
        ),
        "alternative_faster_anchors": len(anchor_ids("alternative_faster")),
        "alternative_faster_anchor_ids": anchor_ids("alternative_faster"),
        "alternative_slower_anchors": len(anchor_ids("alternative_slower")),
        "alternative_slower_anchor_ids": anchor_ids("alternative_slower"),
        "same_reproduction_delay_anchors": len(anchor_ids("same_reproduction_delay")),
        "same_reproduction_delay_anchor_ids": anchor_ids("same_reproduction_delay"),
        "reproduction_delay_delta_ticks": _distribution(deltas),
    }


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.percentile(array, 50)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _validate_action_trace(value: Any, *, expected_steps: int, name: str) -> np.ndarray:
    trace = np.asarray(value, dtype=np.float32)
    if trace.shape != (expected_steps, len(AXIS_NAMES)):
        raise ValueError(
            f"{name} must have shape ({expected_steps}, {len(AXIS_NAMES)})"
        )
    if not np.isfinite(trace).all():
        raise ValueError(f"{name} must be finite")
    return trace


def _validate_action(value: Any, *, name: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32).reshape(-1)
    if action.shape != (len(AXIS_NAMES),) or not np.isfinite(action).all():
        raise ValueError(f"{name} must be finite shape ({len(AXIS_NAMES)},)")
    return action


def _episode_number(value: str) -> int:
    try:
        return int(str(value).split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"invalid episode id: {value!r}") from exc


def _anchor_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["model"]),
        _episode_number(str(row["episode_id"])),
        int(row["anchor_step"]),
        int(row["axis_index"]),
        str(row["direction"]),
    )


def _anchor_id_sort_key(anchor_id: str) -> tuple[Any, ...]:
    episode_id, raw_step, raw_direction = anchor_id.split(":", 2)
    return (_episode_number(episode_id), int(raw_step), raw_direction)


__all__ = ["MODES", "evaluate_temporal_aggregation_counterfactual"]
