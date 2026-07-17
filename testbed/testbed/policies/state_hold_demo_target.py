"""Counterfactual reproduction of one demo target under held observation.

The normal offline replay advances through recorded observations regardless of
whether the predicted action could have moved the machine.  This module adds a
falsification check: at every expert inactive-to-effective transition, rebuild
the policy history and repeatedly present the same observation with zero qvel.
Only an action crossing the calibrated deadzone on the same axis and direction
counts as reproduction of that demo target.

Neither reproduction nor non-reproduction determines task correctness, safety,
or generic liveness; the held-out trajectory supplies one sampled target only.
"""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from testbed.policies.deadzone_eval import effective_direction_mask
from testbed.policies.offline_eval import AXIS_NAMES
from testbed.policies.state_hold_demo_relation import (
    evaluate_state_hold_trace_demo_relation,
)


@dataclass(frozen=True)
class ShouldMoveAnchor:
    """One expert transition from ineffective to effective motion intent."""

    step: int
    axis_index: int
    axis: str
    direction: str
    group: str
    threshold: float
    expert_action: float


@dataclass(frozen=True)
class StepOutput:
    """Action plus optional stage diagnostics emitted by a step source."""

    action: np.ndarray
    diagnostics: Mapping[str, Any] | None = None


class StatefulStepSource(Protocol):
    """Small injectable interface implemented by raw or full runtime sources."""

    def reset(self) -> None: ...

    def step(self, observation: Mapping[str, Any]) -> StepOutput | np.ndarray: ...

    def snapshot_state(self) -> Any: ...

    def restore_state(self, state: Any) -> None: ...


class PredictPolicyStepSource:
    """Adapt an object exposing ``reset``/``predict`` to ``StatefulStepSource``."""

    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def step(self, observation: Mapping[str, Any]) -> StepOutput:
        return StepOutput(action=np.asarray(self._policy.predict(dict(observation))))

    def snapshot_state(self) -> Any:
        snapshot = getattr(self._policy, "snapshot_state", None)
        if not callable(snapshot):
            raise TypeError("policy does not implement snapshot_state()")
        return snapshot()

    def restore_state(self, state: Any) -> None:
        restore = getattr(self._policy, "restore_state", None)
        if not callable(restore):
            raise TypeError("policy does not implement restore_state()")
        restore(state)


class RuntimeActionStepSource:
    """Adapt ``PolicyActionSource.next_action`` and retain its stage diagnostics."""

    def __init__(self, source: Any) -> None:
        self._source = source

    def reset(self) -> None:
        self._source.reset()

    def step(self, observation: Mapping[str, Any]) -> StepOutput:
        action, info = self._source.next_action(dict(observation))
        diagnostics = getattr(info, "extras", None)
        return StepOutput(action=np.asarray(action), diagnostics=diagnostics)

    def snapshot_state(self) -> Any:
        return self._source.snapshot_state()

    def restore_state(self, state: Any) -> None:
        self._source.restore_state(state)

    def close(self) -> None:
        close = getattr(self._source, "close", None)
        if callable(close):
            close()


def extract_should_move_anchors(
    expert_action: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> list[ShouldMoveAnchor]:
    """Return every per-axis/direction ineffective-to-effective transition."""

    expert = _validate_actions(expert_action, name="expert_action")
    effective = effective_direction_mask(expert, thresholds)
    first_move_steps = np.flatnonzero(effective.any(axis=(1, 2)))
    startup_step = int(first_move_steps[0]) if first_move_steps.size else None
    anchors: list[ShouldMoveAnchor] = []
    for step in range(expert.shape[0]):
        for axis_index, axis in enumerate(AXIS_NAMES):
            for direction_index, direction in enumerate(("pos", "neg")):
                if not effective[step, axis_index, direction_index]:
                    continue
                if step > 0 and effective[step - 1, axis_index, direction_index]:
                    continue
                anchors.append(
                    ShouldMoveAnchor(
                        step=step,
                        axis_index=axis_index,
                        axis=axis,
                        direction=direction,
                        group="startup" if step == startup_step else "mid_cycle",
                        threshold=float(thresholds[axis][direction]),
                        expert_action=float(expert[step, axis_index]),
                    )
                )
    return anchors


def evaluate_state_hold_demo_target(
    *,
    episode_id: str,
    observations: Sequence[Mapping[str, Any]],
    expert_action: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    step_source: StatefulStepSource,
    hold_horizon_steps: int,
    trace_full_horizon_after_reproduction: bool = False,
    instrumentation: MutableMapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compare teacher-forced and held-observation demo-target reproduction.

    Each run starts from ``reset`` and warms the stateful source with recorded
    observations strictly before the anchor.  The state-hold branch then repeats
    the anchor image/qpos while replacing qvel with zeros.  By default it stops
    at first target reproduction; the opt-in full-horizon trace keeps later
    anchor-relative differences observable without judging them invalid.
    """

    expert = _validate_actions(expert_action, name="expert_action")
    if len(observations) < expert.shape[0]:
        raise ValueError(
            "observations must cover every expert action: "
            f"{len(observations)} < {expert.shape[0]}"
        )
    if int(hold_horizon_steps) <= 0:
        raise ValueError("hold_horizon_steps must be positive")
    _validate_observation_structure(observations, expert.shape[0])

    anchors = extract_should_move_anchors(expert, thresholds)
    shared_results = _evaluate_shared_teacher_prefix(
        observations=observations,
        source=step_source,
        anchors=anchors,
        horizon_steps=int(hold_horizon_steps),
        trace_full_horizon_after_reproduction=bool(
            trace_full_horizon_after_reproduction
        ),
        instrumentation=instrumentation,
    )
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        if shared_results is None:
            teacher = _run_from_anchor(
                observations=observations,
                source=step_source,
                anchor=anchor,
                horizon_steps=int(hold_horizon_steps),
                state_hold=False,
                trace_full_horizon_after_reproduction=bool(
                    trace_full_horizon_after_reproduction
                ),
                instrumentation=instrumentation,
            )
            held = _run_from_anchor(
                observations=observations,
                source=step_source,
                anchor=anchor,
                horizon_steps=int(hold_horizon_steps),
                state_hold=True,
                trace_full_horizon_after_reproduction=bool(
                    trace_full_horizon_after_reproduction
                ),
                instrumentation=instrumentation,
            )
        else:
            teacher, held = shared_results[anchor]
        demo_relation = evaluate_state_hold_trace_demo_relation(
            expert_action=expert[anchor.step],
            action_trace=held["action_trace"],
            thresholds=thresholds,
            target_axis_index=anchor.axis_index,
            target_direction=anchor.direction,
        )
        temporal_aggregation_evidence = _state_hold_temporal_aggregation_evidence(
            diagnostics_trace=held["diagnostics_trace"],
            anchor=anchor,
        )
        rows.append(
            {
                "episode_id": str(episode_id),
                "anchor_step": anchor.step,
                "anchor_group": anchor.group,
                "axis_index": anchor.axis_index,
                "axis": anchor.axis,
                "direction": anchor.direction,
                "deadzone_threshold": anchor.threshold,
                "expert_action": anchor.expert_action,
                "expert_action_vector": [float(value) for value in expert[anchor.step]],
                "hold_horizon_steps": int(hold_horizon_steps),
                "trace_full_horizon_after_reproduction": bool(
                    trace_full_horizon_after_reproduction
                ),
                "state_hold_qvel_zero": True,
                "execution_feedback_recursive": _observation_has_key(
                    observations,
                    anchor.step,
                    "previous_final_command",
                ),
                "teacher_forced_status": teacher["status"],
                "teacher_forced_demo_target_reproduction_delay_ticks": teacher[
                    "demo_target_reproduction_delay_ticks"
                ],
                "teacher_forced_ticks_evaluated": teacher["ticks_evaluated"],
                "teacher_forced_trace_termination": teacher["trace_termination"],
                "teacher_forced_full_horizon_complete": teacher[
                    "full_horizon_complete"
                ],
                "teacher_forced_action_trace": teacher["action_trace"],
                "teacher_forced_diagnostics_trace": teacher["diagnostics_trace"],
                "state_hold_status": held["status"],
                "state_hold_demo_target_not_reproduced": (
                    held["status"] == "demo_target_not_reproduced"
                ),
                "state_hold_demo_target_reproduction_delay_ticks": held[
                    "demo_target_reproduction_delay_ticks"
                ],
                "state_hold_ticks_evaluated": held["ticks_evaluated"],
                "state_hold_trace_termination": held["trace_termination"],
                "state_hold_full_horizon_complete": held["full_horizon_complete"],
                "state_hold_action_trace": held["action_trace"],
                "state_hold_diagnostics_trace": held["diagnostics_trace"],
                "demo_target_reproduction_hidden_by_teacher_forcing": (
                    teacher["status"] == "demo_target_reproduced"
                    and held["status"] == "demo_target_not_reproduced"
                ),
                **temporal_aggregation_evidence,
                **demo_relation,
            }
        )
    return rows


def aggregate_state_hold_demo_target_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize demo-target reproduction without inferring correctness."""

    summaries: list[dict[str, Any]] = []
    groups = (
        ("overall", list(rows)),
        ("startup", [row for row in rows if row["anchor_group"] == "startup"]),
        ("mid_cycle", [row for row in rows if row["anchor_group"] == "mid_cycle"]),
    )
    for group_name, group in groups:
        reproduced_delays = sorted(
            int(row["state_hold_demo_target_reproduction_delay_ticks"])
            for row in group
            if row["state_hold_demo_target_reproduction_delay_ticks"] is not None
        )
        total = len(group)
        not_reproduced = sum(
            bool(row["state_hold_demo_target_not_reproduced"]) for row in group
        )
        summaries.append(
            {
                "group": group_name,
                "anchors_total": total,
                "state_hold_demo_target_reproduced_anchors": len(reproduced_delays),
                "state_hold_demo_target_not_reproduced_anchors": not_reproduced,
                "state_hold_demo_target_nonreproduction_rate": (
                    float(not_reproduced) / float(total)
                )
                if total
                else 0.0,
                "teacher_forced_demo_target_reproduced_anchors": sum(
                    row["teacher_forced_status"] == "demo_target_reproduced"
                    for row in group
                ),
                "demo_target_reproduction_hidden_by_teacher_forcing_anchors": sum(
                    bool(row["demo_target_reproduction_hidden_by_teacher_forcing"])
                    for row in group
                ),
                "state_hold_anchor_extra_effective_anchors": sum(
                    bool(row.get("state_hold_anchor_extra_effective", False))
                    for row in group
                ),
                "state_hold_anchor_extra_effective_ticks": sum(
                    int(row.get("state_hold_anchor_extra_effective_tick_count", 0))
                    for row in group
                ),
                "state_hold_anchor_extra_effective_directions": sum(
                    int(
                        row.get(
                            "state_hold_anchor_extra_effective_direction_count",
                            0,
                        )
                    )
                    for row in group
                ),
                "state_hold_opposite_to_demo_target_ticks": sum(
                    int(row.get("state_hold_opposite_to_demo_target_tick_count", 0))
                    for row in group
                ),
                "state_hold_direction_flips": sum(
                    int(row.get("state_hold_direction_flip_count", 0)) for row in group
                ),
                "state_hold_max_effective_axes": max(
                    (int(row.get("state_hold_max_effective_axes", 0)) for row in group),
                    default=0,
                ),
                "temporal_aggregation_decomposition_complete_anchors": sum(
                    bool(
                        row.get(
                            "state_hold_temporal_aggregation_decomposition_complete",
                            False,
                        )
                    )
                    for row in group
                ),
                "newest_crosses_legacy_misses_anchors": sum(
                    int(
                        row.get(
                            "state_hold_newest_crosses_legacy_misses_tick_count",
                            0,
                        )
                    )
                    > 0
                    for row in group
                ),
                "newest_crosses_legacy_misses_ticks": sum(
                    int(
                        row.get(
                            "state_hold_newest_crosses_legacy_misses_tick_count",
                            0,
                        )
                    )
                    for row in group
                ),
                "recency_crosses_legacy_misses_anchors": sum(
                    int(
                        row.get(
                            "state_hold_recency_crosses_legacy_misses_tick_count",
                            0,
                        )
                    )
                    > 0
                    for row in group
                ),
                "recency_crosses_legacy_misses_ticks": sum(
                    int(
                        row.get(
                            "state_hold_recency_crosses_legacy_misses_tick_count",
                            0,
                        )
                    )
                    for row in group
                ),
                "state_hold_demo_target_reproduction_delay_ticks": reproduced_delays,
                "state_hold_demo_target_reproduction_delay_mean_ticks": _mean_or_none(
                    reproduced_delays
                ),
                "state_hold_demo_target_reproduction_delay_median_ticks": (
                    _percentile_or_none(reproduced_delays, 50)
                ),
                "state_hold_demo_target_reproduction_delay_p95_ticks": (
                    _percentile_or_none(reproduced_delays, 95)
                ),
                "state_hold_demo_target_reproduction_delay_max_ticks": (
                    max(reproduced_delays) if reproduced_delays else None
                ),
            }
        )
    return summaries


def write_state_hold_demo_target_report(
    *,
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write per-anchor JSONL/CSV and aggregate JSON artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe_rows = [_json_safe(row) for row in rows]
    aggregates = _json_safe(aggregate_state_hold_demo_target_rows(rows))
    rows_jsonl = output / "state_hold_anchors.jsonl"
    rows_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in safe_rows),
        encoding="utf-8",
    )
    rows_csv = output / "state_hold_anchors.csv"
    fieldnames = list(dict.fromkeys(key for row in safe_rows for key in row))
    with rows_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in safe_rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
    summary = output / "state_hold_summary.json"
    summary.write_text(
        json.dumps(
            {
                "diagnostic": "single_demo_target_state_hold_counterfactual",
                "capability_boundaries": {
                    "directly_measures": (
                        "whether the held-observation policy reproduces one demonstrated "
                        "axis-direction target within the configured horizon"
                    ),
                    "correctness_estimable": False,
                    "task_support_estimable": False,
                    "physical_validity_estimable": False,
                    "nonreproduction_is_generic_deadlock": False,
                    "anchor_extra_is_invalid_action": False,
                    "limitations": (
                        "Reproduction does not prove field success; non-reproduction "
                        "does not prove generic deadlock; anchor-extra actions are "
                        "differences from one demo, not correctness or safety failures."
                    ),
                },
                "metadata": _json_safe(dict(metadata or {})),
                "aggregate": aggregates,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"rows_jsonl": rows_jsonl, "rows_csv": rows_csv, "summary": summary}


def _validate_observation_structure(
    observations: Sequence[Mapping[str, Any]],
    required_steps: int,
) -> None:
    validator = getattr(observations, "validate_state_hold_structure", None)
    if callable(validator):
        validator(required_steps=int(required_steps))
        return
    for index in range(int(required_steps)):
        if "qvel" not in observations[index]:
            raise ValueError(f"observation {index} is missing qvel")


def _observation_has_key(
    observations: Sequence[Mapping[str, Any]],
    index: int,
    key: str,
) -> bool:
    checker = getattr(observations, "has_observation_key", None)
    if callable(checker):
        return bool(checker(key))
    return key in observations[index]


def _evaluate_shared_teacher_prefix(
    *,
    observations: Sequence[Mapping[str, Any]],
    source: StatefulStepSource,
    anchors: Sequence[ShouldMoveAnchor],
    horizon_steps: int,
    trace_full_horizon_after_reproduction: bool,
    instrumentation: MutableMapping[str, Any] | None,
) -> dict[ShouldMoveAnchor, tuple[dict[str, Any], dict[str, Any]]] | None:
    """Run one teacher prefix and restore its pre-anchor states for held branches."""

    if not anchors:
        _counter_set(instrumentation, "evaluation_path", "no_anchors")
        return {}
    snapshot = getattr(source, "snapshot_state", None)
    restore = getattr(source, "restore_state", None)
    if not callable(snapshot) or not callable(restore):
        _counter_set(instrumentation, "evaluation_path", "legacy_replay")
        return None

    source.reset()
    try:
        initial_state = snapshot()
        restore(initial_state)
    except (AttributeError, NotImplementedError, TypeError):
        source.reset()
        _counter_set(instrumentation, "evaluation_path", "legacy_replay")
        return None

    _counter_set(instrumentation, "evaluation_path", "shared_teacher_prefix")
    anchors_by_step: dict[int, list[ShouldMoveAnchor]] = {}
    for anchor in anchors:
        anchors_by_step.setdefault(anchor.step, []).append(anchor)
    last_teacher_step = max(
        min(len(observations), anchor.step + int(horizon_steps)) for anchor in anchors
    )
    held_results: dict[ShouldMoveAnchor, dict[str, Any]] = {}
    teacher_outputs: list[StepOutput] = []
    for step in range(last_teacher_step):
        step_anchors = anchors_by_step.get(step, [])
        if step_anchors:
            branch_state = snapshot()
            _counter_add(instrumentation, "snapshots_captured")
            for anchor in step_anchors:
                restore(branch_state)
                _counter_add(instrumentation, "snapshots_restored")
                held_results[anchor] = _run_horizon(
                    observations=observations,
                    source=source,
                    anchor=anchor,
                    horizon_steps=horizon_steps,
                    state_hold=True,
                    trace_full_horizon_after_reproduction=(
                        trace_full_horizon_after_reproduction
                    ),
                    cached_teacher_outputs=None,
                    instrumentation=instrumentation,
                )
            restore(branch_state)
            _counter_add(instrumentation, "snapshots_restored")
        output = _step_source(
            source,
            _copy_observation(observations[step], zero_qvel=False),
            instrumentation=instrumentation,
            counter="shared_teacher_steps",
        )
        teacher_outputs.append(_clone_step_output(output))

    results: dict[ShouldMoveAnchor, tuple[dict[str, Any], dict[str, Any]]] = {}
    for anchor in anchors:
        teacher = _run_horizon(
            observations=observations,
            source=source,
            anchor=anchor,
            horizon_steps=horizon_steps,
            state_hold=False,
            trace_full_horizon_after_reproduction=(
                trace_full_horizon_after_reproduction
            ),
            cached_teacher_outputs=teacher_outputs,
            instrumentation=instrumentation,
        )
        results[anchor] = (teacher, held_results[anchor])
    return results


def _run_from_anchor(
    *,
    observations: Sequence[Mapping[str, Any]],
    source: StatefulStepSource,
    anchor: ShouldMoveAnchor,
    horizon_steps: int,
    state_hold: bool,
    trace_full_horizon_after_reproduction: bool,
    instrumentation: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    source.reset()
    for warmup_step in range(anchor.step):
        _step_source(
            source,
            _copy_observation(observations[warmup_step], zero_qvel=False),
            instrumentation=instrumentation,
            counter="legacy_warmup_steps",
        )
    return _run_horizon(
        observations=observations,
        source=source,
        anchor=anchor,
        horizon_steps=horizon_steps,
        state_hold=state_hold,
        trace_full_horizon_after_reproduction=trace_full_horizon_after_reproduction,
        cached_teacher_outputs=None,
        instrumentation=instrumentation,
    )


def _run_horizon(
    *,
    observations: Sequence[Mapping[str, Any]],
    source: StatefulStepSource,
    anchor: ShouldMoveAnchor,
    horizon_steps: int,
    state_hold: bool,
    trace_full_horizon_after_reproduction: bool,
    cached_teacher_outputs: Sequence[StepOutput] | None,
    instrumentation: MutableMapping[str, Any] | None,
) -> dict[str, Any]:

    action_trace: list[list[float]] = []
    diagnostics_trace: list[dict[str, Any] | None] = []
    reproduction_delay: int | None = None
    held_observation = (
        _copy_observation(observations[anchor.step], zero_qvel=True)
        if state_hold
        else None
    )
    for delay in range(horizon_steps):
        observation_step = anchor.step if state_hold else anchor.step + delay
        if observation_step >= len(observations):
            if reproduction_delay is not None:
                return {
                    "status": "demo_target_reproduced",
                    "demo_target_reproduction_delay_ticks": reproduction_delay,
                    "ticks_evaluated": len(action_trace),
                    "action_trace": action_trace,
                    "diagnostics_trace": diagnostics_trace,
                    "trace_termination": (
                        "recorded_data_exhausted_after_demo_target_reproduction"
                    ),
                    "full_horizon_complete": False,
                }
            return {
                "status": "recorded_data_exhausted",
                "demo_target_reproduction_delay_ticks": None,
                "ticks_evaluated": len(action_trace),
                "action_trace": action_trace,
                "diagnostics_trace": diagnostics_trace,
                "trace_termination": "recorded_data_exhausted",
                "full_horizon_complete": False,
            }
        if cached_teacher_outputs is not None:
            if observation_step >= len(cached_teacher_outputs):
                raise RuntimeError(
                    "shared teacher trace does not cover required observation step"
                )
            output = _clone_step_output(cached_teacher_outputs[observation_step])
        else:
            observation = (
                _copy_observation(held_observation, zero_qvel=False)
                if held_observation is not None
                else _copy_observation(observations[observation_step], zero_qvel=False)
            )
            output = _step_source(
                source,
                observation,
                instrumentation=instrumentation,
                counter=("held_branch_steps" if state_hold else "legacy_branch_steps"),
            )
        action = _validate_action(output.action)
        action_trace.append([float(value) for value in action])
        diagnostics_trace.append(_json_safe(output.diagnostics))
        if (
            held_observation is not None
            and "previous_final_command" in held_observation
        ):
            held_observation["previous_final_command"] = action.astype(
                np.float32, copy=True
            )
        if _demo_target_is_effective(action, anchor):
            if reproduction_delay is None:
                reproduction_delay = delay
            if not trace_full_horizon_after_reproduction:
                return {
                    "status": "demo_target_reproduced",
                    "demo_target_reproduction_delay_ticks": reproduction_delay,
                    "ticks_evaluated": len(action_trace),
                    "action_trace": action_trace,
                    "diagnostics_trace": diagnostics_trace,
                    "trace_termination": "demo_target_reproduced",
                    "full_horizon_complete": len(action_trace) == horizon_steps,
                }
    if reproduction_delay is not None:
        return {
            "status": "demo_target_reproduced",
            "demo_target_reproduction_delay_ticks": reproduction_delay,
            "ticks_evaluated": len(action_trace),
            "action_trace": action_trace,
            "diagnostics_trace": diagnostics_trace,
            "trace_termination": "horizon_complete_after_demo_target_reproduction",
            "full_horizon_complete": True,
        }
    return {
        "status": "demo_target_not_reproduced",
        "demo_target_reproduction_delay_ticks": None,
        "ticks_evaluated": len(action_trace),
        "action_trace": action_trace,
        "diagnostics_trace": diagnostics_trace,
        "trace_termination": "demo_target_not_reproduced",
        "full_horizon_complete": True,
    }


def _step_source(
    source: StatefulStepSource,
    observation: Mapping[str, Any],
    *,
    instrumentation: MutableMapping[str, Any] | None,
    counter: str,
) -> StepOutput:
    output = _coerce_step_output(source.step(observation))
    _counter_add(instrumentation, "source_step_calls")
    _counter_add(instrumentation, counter)
    return output


def _clone_step_output(output: StepOutput) -> StepOutput:
    return StepOutput(
        action=np.asarray(output.action, dtype=np.float32).copy(),
        diagnostics=copy.deepcopy(output.diagnostics),
    )


def _counter_add(
    instrumentation: MutableMapping[str, Any] | None,
    key: str,
    value: int = 1,
) -> None:
    if instrumentation is not None:
        instrumentation[key] = int(instrumentation.get(key, 0)) + int(value)


def _counter_set(
    instrumentation: MutableMapping[str, Any] | None,
    key: str,
    value: Any,
) -> None:
    if instrumentation is not None:
        instrumentation[key] = value


def _copy_observation(
    observation: Mapping[str, Any],
    *,
    zero_qvel: bool,
) -> dict[str, Any]:
    copied = {key: copy.deepcopy(value) for key, value in observation.items()}
    if zero_qvel:
        copied["qvel"] = np.zeros_like(np.asarray(copied["qvel"], dtype=np.float32))
    return copied


def _coerce_step_output(value: StepOutput | np.ndarray) -> StepOutput:
    if isinstance(value, StepOutput):
        return value
    return StepOutput(action=np.asarray(value), diagnostics=None)


def _demo_target_is_effective(action: np.ndarray, anchor: ShouldMoveAnchor) -> bool:
    value = float(action[anchor.axis_index])
    if anchor.direction == "pos":
        return value >= anchor.threshold
    return value <= -anchor.threshold


def _state_hold_temporal_aggregation_evidence(
    *,
    diagnostics_trace: Sequence[Mapping[str, Any] | None],
    anchor: ShouldMoveAnchor,
) -> dict[str, Any]:
    """Extract target-deadzone evidence from opt-in ACT aggregation traces."""

    action_keys = {
        "legacy": "policy_temporal_aggregation_legacy_action",
        "newest": "policy_temporal_aggregation_newest_action",
        "recency": "policy_temporal_aggregation_recency_action",
    }
    required_keys = (
        "policy_temporal_aggregation_action_domain",
        "policy_temporal_aggregation_query_step",
        "policy_temporal_aggregation_source_steps",
        *action_keys.values(),
    )
    trace: list[dict[str, Any]] = []
    newest_miss_ticks: list[int] = []
    recency_miss_ticks: list[int] = []
    for held_delay, diagnostics in enumerate(diagnostics_trace):
        if not isinstance(diagnostics, Mapping):
            continue
        if not any(key in diagnostics for key in action_keys.values()):
            continue
        missing = [key for key in required_keys if key not in diagnostics]
        if missing:
            raise ValueError(
                "temporal aggregation diagnostics are missing field(s): "
                + ", ".join(missing)
            )
        if diagnostics["policy_temporal_aggregation_action_domain"] != (
            "direct_policy_output"
        ):
            raise ValueError(
                "temporal aggregation diagnostics must use direct_policy_output"
            )
        query_step = int(diagnostics["policy_temporal_aggregation_query_step"])
        source_steps = [
            int(value)
            for value in diagnostics["policy_temporal_aggregation_source_steps"]
        ]
        if source_steps != sorted(source_steps) or any(
            value > query_step for value in source_steps
        ):
            raise ValueError(
                "temporal aggregation diagnostics contain non-causal source steps"
            )
        actions = {
            name: _validate_action(np.asarray(diagnostics[key], dtype=np.float32))
            for name, key in action_keys.items()
        }
        effective = {
            name: _demo_target_is_effective(action, anchor)
            for name, action in actions.items()
        }
        newest_miss = bool(effective["newest"] and not effective["legacy"])
        recency_miss = bool(effective["recency"] and not effective["legacy"])
        if newest_miss:
            newest_miss_ticks.append(held_delay)
        if recency_miss:
            recency_miss_ticks.append(held_delay)
        trace.append(
            {
                "held_delay": held_delay,
                "query_step": query_step,
                "source_steps": source_steps,
                "legacy_target_value": float(actions["legacy"][anchor.axis_index]),
                "newest_target_value": float(actions["newest"][anchor.axis_index]),
                "recency_target_value": float(actions["recency"][anchor.axis_index]),
                "legacy_crosses_target_deadzone": bool(effective["legacy"]),
                "newest_crosses_target_deadzone": bool(effective["newest"]),
                "recency_crosses_target_deadzone": bool(effective["recency"]),
                "newest_crosses_legacy_misses": newest_miss,
                "recency_crosses_legacy_misses": recency_miss,
            }
        )
    return {
        "state_hold_temporal_aggregation_decomposition_available": bool(trace),
        "state_hold_temporal_aggregation_decomposition_complete": bool(trace)
        and len(trace) == len(diagnostics_trace),
        "state_hold_temporal_aggregation_decomposition_ticks_recorded": len(trace),
        "state_hold_temporal_aggregation_decomposition_trace": trace,
        "state_hold_newest_crosses_legacy_misses_ticks": newest_miss_ticks,
        "state_hold_newest_crosses_legacy_misses_tick_count": len(newest_miss_ticks),
        "state_hold_recency_crosses_legacy_misses_ticks": recency_miss_ticks,
        "state_hold_recency_crosses_legacy_misses_tick_count": len(recency_miss_ticks),
    }


def _validate_actions(action: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(AXIS_NAMES):
        raise ValueError(
            f"{name} must have shape (T, {len(AXIS_NAMES)}), got {array.shape}"
        )
    return array


def _validate_action(action: np.ndarray) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32).reshape(-1)
    if array.shape != (len(AXIS_NAMES),):
        raise ValueError(
            f"step action must have shape ({len(AXIS_NAMES)},), got {array.shape}"
        )
    return array


def _mean_or_none(values: list[int]) -> float | None:
    return float(np.mean(values)) if values else None


def _percentile_or_none(values: list[int], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
