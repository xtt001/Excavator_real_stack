"""Contract and evidence audit for bounded short-horizon policy rollouts.

The owner is deliberately passive: it validates a recorded trace and never
sends, clips, retries, or rewrites a command.  A state counts as self-generated
only when its observation explicitly names the preceding acknowledged command
that produced it in the declared rollout world.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from testbed.policies.deadzone_eval import AXIS_NAMES
from testbed.policies.task_sequence_compatibility import activation_motif

SCHEMA_VERSION = "short_horizon_rollout_trace_v1"
MAX_HORIZON_SECONDS = 2.0

StateOrigin = Literal[
    "teacher_forced",
    "state_hold",
    "learned_dynamics",
    "hybrid_lowdim",
    "simulator",
    "live_policy_on",
]
ControlAuthority = Literal["observe_only", "bounded_control"]

STATE_ORIGINS = (
    "teacher_forced",
    "state_hold",
    "learned_dynamics",
    "hybrid_lowdim",
    "simulator",
    "live_policy_on",
)
CONTROL_AUTHORITIES = ("observe_only", "bounded_control")
TERMINATION_REASONS = (
    "horizon_complete",
    "operator_abort",
    "safety_abort",
    "controller_fault",
    "timing_gap",
    "missing_observation",
    "policy_error",
    "gohome_requested",
)


@dataclass(frozen=True)
class ShortHorizonRolloutContract:
    """Immutable rollout authority, provenance, and safety boundary."""

    test_id: str
    state_origin: StateOrigin
    control_authority: ControlAuthority
    policy_id: str
    checkpoint_sha256: str
    resolved_config_sha256: str
    sampling_hz: float
    horizon_ticks: int
    camera_names: tuple[str, ...]
    max_observation_gap_ms: float
    max_camera_age_ms: float
    deadzone_positive: tuple[float, ...]
    deadzone_negative: tuple[float, ...]
    command_abs_limit: tuple[float, ...] | None = None
    command_delta_limit: tuple[float, ...] | None = None
    qvel_abort_limit: tuple[float, ...] | None = None
    qpos_lower_limit: tuple[float, ...] | None = None
    qpos_upper_limit: tuple[float, ...] | None = None
    allowed_direction_mask: tuple[tuple[bool, bool], ...] | None = None
    require_deadman: bool = True
    require_controller_ack: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ShortHorizonRolloutContract:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"contract schema_version must be {SCHEMA_VERSION!r}")
        state_origin = str(payload.get("state_origin", ""))
        if state_origin not in STATE_ORIGINS:
            raise ValueError(f"unsupported state_origin: {state_origin!r}")
        authority = str(payload.get("control_authority", ""))
        if authority not in CONTROL_AUTHORITIES:
            raise ValueError(f"unsupported control_authority: {authority!r}")
        if state_origin == "live_policy_on" and authority != "bounded_control":
            raise ValueError("live_policy_on requires bounded_control authority")
        if (
            state_origin in {"teacher_forced", "state_hold"}
            and authority != "observe_only"
        ):
            raise ValueError(f"{state_origin} requires observe_only authority")

        sampling_hz = _positive(payload.get("sampling_hz"), "sampling_hz")
        horizon_ticks = _positive_int(payload.get("horizon_ticks"), "horizon_ticks")
        duration_seconds = horizon_ticks / sampling_hz
        if duration_seconds > MAX_HORIZON_SECONDS + 1e-12:
            raise ValueError(
                f"rollout duration {duration_seconds:.6f}s exceeds "
                f"{MAX_HORIZON_SECONDS:.1f}s"
            )
        cameras = _unique_nonempty_strings(payload.get("camera_names"), "camera_names")
        positive_deadzone = _positive_axis_vector(
            payload.get("deadzone_positive"), "deadzone_positive"
        )
        negative_deadzone = _positive_axis_vector(
            payload.get("deadzone_negative"), "deadzone_negative"
        )

        bounded_fields: dict[str, tuple[float, ...] | None] = {}
        for name in (
            "command_abs_limit",
            "command_delta_limit",
            "qvel_abort_limit",
        ):
            raw = payload.get(name)
            bounded_fields[name] = (
                _positive_axis_vector(raw, name) if raw is not None else None
            )
        for name in ("qpos_lower_limit", "qpos_upper_limit"):
            raw = payload.get(name)
            bounded_fields[name] = (
                _finite_axis_vector(raw, name) if raw is not None else None
            )
        direction_mask = _direction_mask(payload.get("allowed_direction_mask"))

        if authority == "bounded_control":
            missing = [
                name
                for name, value in (
                    *bounded_fields.items(),
                    ("allowed_direction_mask", direction_mask),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "bounded_control contract is missing safety field(s): "
                    + ", ".join(missing)
                )
            lower = np.asarray(bounded_fields["qpos_lower_limit"])
            upper = np.asarray(bounded_fields["qpos_upper_limit"])
            if np.any(lower >= upper):
                raise ValueError("qpos_lower_limit must be below qpos_upper_limit")

        return cls(
            test_id=_nonempty(payload.get("test_id"), "test_id"),
            state_origin=state_origin,  # type: ignore[arg-type]
            control_authority=authority,  # type: ignore[arg-type]
            policy_id=_nonempty(payload.get("policy_id"), "policy_id"),
            checkpoint_sha256=_sha256(
                payload.get("checkpoint_sha256"), "checkpoint_sha256"
            ),
            resolved_config_sha256=_sha256(
                payload.get("resolved_config_sha256"), "resolved_config_sha256"
            ),
            sampling_hz=sampling_hz,
            horizon_ticks=horizon_ticks,
            camera_names=cameras,
            max_observation_gap_ms=_positive(
                payload.get("max_observation_gap_ms"), "max_observation_gap_ms"
            ),
            max_camera_age_ms=_nonnegative(
                payload.get("max_camera_age_ms"), "max_camera_age_ms"
            ),
            deadzone_positive=positive_deadzone,
            deadzone_negative=negative_deadzone,
            command_abs_limit=bounded_fields["command_abs_limit"],
            command_delta_limit=bounded_fields["command_delta_limit"],
            qvel_abort_limit=bounded_fields["qvel_abort_limit"],
            qpos_lower_limit=bounded_fields["qpos_lower_limit"],
            qpos_upper_limit=bounded_fields["qpos_upper_limit"],
            allowed_direction_mask=direction_mask,
            require_deadman=_strict_bool(
                payload.get("require_deadman", True), "require_deadman"
            ),
            require_controller_ack=_strict_bool(
                payload.get("require_controller_ack", True),
                "require_controller_ack",
            ),
        )

    def thresholds(self) -> dict[str, dict[str, float]]:
        return {
            axis: {
                "pos": float(self.deadzone_positive[index]),
                "neg": float(self.deadzone_negative[index]),
            }
            for index, axis in enumerate(AXIS_NAMES)
        }


@dataclass(frozen=True)
class ShortHorizonRolloutStep:
    """One observation, policy decision, and resulting controller send record."""

    tick: int
    observation_timestamp_ns: int
    camera_timestamps_ns: dict[str, int]
    camera_frame_ids: dict[str, str]
    observation_origin: str
    generated_by_command_id: int | None
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    policy_action: tuple[float, ...]
    policy_returned_action: tuple[float, ...]
    safe_action: tuple[float, ...]
    commanded_action: tuple[float, ...]
    command_sent: bool
    command_id: int | None
    send_timestamp_ns: int | None
    controller_ack: bool
    deadman_pressed: bool
    estop_active: bool
    manual_override_active: bool
    sensor_stale: bool
    safety_reasons: tuple[str, ...]
    policy_error: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ShortHorizonRolloutStep:
        command_sent = _strict_bool(payload.get("command_sent"), "command_sent")
        command_id = _optional_nonnegative_int(payload.get("command_id"), "command_id")
        send_timestamp = _optional_nonnegative_int(
            payload.get("send_timestamp_ns"), "send_timestamp_ns"
        )
        if command_sent and (command_id is None or send_timestamp is None):
            raise ValueError("sent command requires command_id and send_timestamp_ns")
        if not command_sent and (command_id is not None or send_timestamp is not None):
            raise ValueError(
                "unsent command must not have command_id/send_timestamp_ns"
            )
        camera_timestamps = _timestamp_mapping(
            payload.get("camera_timestamps_ns"), "camera_timestamps_ns"
        )
        camera_ids = _string_mapping(
            payload.get("camera_frame_ids"), "camera_frame_ids"
        )
        return cls(
            tick=_nonnegative_int(payload.get("tick"), "tick"),
            observation_timestamp_ns=_nonnegative_int(
                payload.get("observation_timestamp_ns"),
                "observation_timestamp_ns",
            ),
            camera_timestamps_ns=camera_timestamps,
            camera_frame_ids=camera_ids,
            observation_origin=_nonempty(
                payload.get("observation_origin"), "observation_origin"
            ),
            generated_by_command_id=_optional_nonnegative_int(
                payload.get("generated_by_command_id"),
                "generated_by_command_id",
            ),
            qpos=_finite_axis_vector(payload.get("qpos"), "qpos"),
            qvel=_finite_axis_vector(payload.get("qvel"), "qvel"),
            policy_action=_finite_axis_vector(
                payload.get("policy_action"), "policy_action"
            ),
            policy_returned_action=_finite_axis_vector(
                payload.get("policy_returned_action"), "policy_returned_action"
            ),
            safe_action=_finite_axis_vector(payload.get("safe_action"), "safe_action"),
            commanded_action=_finite_axis_vector(
                payload.get("commanded_action"), "commanded_action"
            ),
            command_sent=command_sent,
            command_id=command_id,
            send_timestamp_ns=send_timestamp,
            controller_ack=_strict_bool(
                payload.get("controller_ack", False), "controller_ack"
            ),
            deadman_pressed=_strict_bool(
                payload.get("deadman_pressed", False), "deadman_pressed"
            ),
            estop_active=_strict_bool(
                payload.get("estop_active", False), "estop_active"
            ),
            manual_override_active=_strict_bool(
                payload.get("manual_override_active", False),
                "manual_override_active",
            ),
            sensor_stale=_strict_bool(
                payload.get("sensor_stale", False), "sensor_stale"
            ),
            safety_reasons=_unique_nonempty_strings(
                payload.get("safety_reasons", []),
                "safety_reasons",
                allow_empty=True,
            ),
            policy_error=str(payload.get("policy_error", "")),
        )


def evaluate_short_horizon_rollout(
    *,
    contract: ShortHorizonRolloutContract,
    steps: Sequence[ShortHorizonRolloutStep],
    termination_reason: str,
) -> dict[str, Any]:
    """Audit one immutable trace without synthesizing missing feedback."""

    if not isinstance(contract, ShortHorizonRolloutContract):
        raise TypeError("contract must be ShortHorizonRolloutContract")
    if termination_reason not in TERMINATION_REASONS:
        raise ValueError(f"unsupported termination_reason: {termination_reason!r}")
    trace = tuple(steps)
    if not trace:
        raise ValueError("steps must not be empty")
    if any(not isinstance(step, ShortHorizonRolloutStep) for step in trace):
        raise TypeError("all steps must be ShortHorizonRolloutStep")

    integrity_errors: list[str] = []
    contract_breaches: list[str] = []
    abort_triggers: list[dict[str, Any]] = []
    causal_transition_count = 0
    nonzero_causal_transition_count = 0
    command_ids: set[int] = set()
    expected_cameras = set(contract.camera_names)
    max_gap_ns = int(round(contract.max_observation_gap_ms * 1_000_000.0))
    max_camera_age_ns = int(round(contract.max_camera_age_ms * 1_000_000.0))

    previous: ShortHorizonRolloutStep | None = None
    previous_command = np.zeros(len(AXIS_NAMES), dtype=np.float32)
    for index, step in enumerate(trace):
        if step.tick != index:
            integrity_errors.append(
                f"tick_{step.tick}: expected contiguous tick {index}"
            )
        if step.observation_origin != contract.state_origin:
            integrity_errors.append(
                f"tick_{index}: observation_origin differs from contract"
            )
        if set(step.camera_timestamps_ns) != expected_cameras:
            integrity_errors.append(f"tick_{index}: camera timestamp keys differ")
        if set(step.camera_frame_ids) != expected_cameras:
            integrity_errors.append(f"tick_{index}: camera frame-id keys differ")
        for camera in contract.camera_names:
            if camera not in step.camera_timestamps_ns:
                continue
            timestamp = step.camera_timestamps_ns[camera]
            age = step.observation_timestamp_ns - timestamp
            if age < 0 or age > max_camera_age_ns:
                integrity_errors.append(
                    f"tick_{index}: {camera} frame age {age}ns outside contract"
                )
            if previous is not None and camera in previous.camera_timestamps_ns:
                if (
                    contract.state_origin != "state_hold"
                    and timestamp <= previous.camera_timestamps_ns[camera]
                ):
                    integrity_errors.append(
                        f"tick_{index}: {camera} timestamp is not increasing"
                    )

        if previous is None:
            if step.generated_by_command_id is not None:
                integrity_errors.append(
                    "tick_0: initial observation has a parent command"
                )
        else:
            gap = step.observation_timestamp_ns - previous.observation_timestamp_ns
            if gap <= 0 or gap > max_gap_ns:
                integrity_errors.append(
                    f"tick_{index}: observation gap {gap}ns outside contract"
                )
            linked = bool(
                previous.command_sent
                and previous.controller_ack
                and previous.command_id is not None
                and step.generated_by_command_id == previous.command_id
                and previous.send_timestamp_ns is not None
                and step.observation_timestamp_ns > previous.send_timestamp_ns
            )
            if linked:
                causal_transition_count += 1
                if np.any(np.asarray(previous.commanded_action) != 0.0):
                    nonzero_causal_transition_count += 1
            if (
                contract.state_origin
                in {
                    "learned_dynamics",
                    "hybrid_lowdim",
                    "simulator",
                    "live_policy_on",
                }
                and not linked
            ):
                integrity_errors.append(
                    f"tick_{index}: self-generated observation lacks causal command link"
                )
            if (
                contract.state_origin in {"teacher_forced", "state_hold"}
                and step.generated_by_command_id is not None
            ):
                integrity_errors.append(
                    f"tick_{index}: noncausal observation must not claim a parent command"
                )

        commanded = np.asarray(step.commanded_action, dtype=np.float32)
        if contract.control_authority == "observe_only" and np.any(commanded != 0.0):
            contract_breaches.append(
                f"tick_{index}: observe_only trace contains nonzero command"
            )
        if step.command_sent:
            assert step.command_id is not None and step.send_timestamp_ns is not None
            if step.command_id in command_ids:
                integrity_errors.append(f"tick_{index}: duplicate command_id")
            command_ids.add(step.command_id)
            if step.send_timestamp_ns < step.observation_timestamp_ns:
                integrity_errors.append(
                    f"tick_{index}: command sent before its source observation"
                )
            if contract.require_controller_ack and not step.controller_ack:
                abort_triggers.append(
                    {"tick": index, "reason": "controller_not_acknowledged"}
                )
        elif step.controller_ack:
            integrity_errors.append(
                f"tick_{index}: unsent command cannot be acknowledged"
            )

        if contract.control_authority == "bounded_control":
            _audit_bounded_command(
                contract=contract,
                step=step,
                index=index,
                previous_command=previous_command,
                breaches=contract_breaches,
                abort_triggers=abort_triggers,
            )
        previous_command = commanded
        previous = step

    if len(trace) > contract.horizon_ticks:
        contract_breaches.append("trace exceeds horizon_ticks")
    if (
        termination_reason == "horizon_complete"
        and len(trace) != contract.horizon_ticks
    ):
        integrity_errors.append("horizon_complete requires exactly horizon_ticks steps")
    if (
        termination_reason != "horizon_complete"
        and len(trace) >= contract.horizon_ticks
    ):
        integrity_errors.append(
            "abort termination must occur before horizon completion"
        )
    if abort_triggers:
        first_abort_tick = int(abort_triggers[0]["tick"])
        if termination_reason not in {
            "operator_abort",
            "safety_abort",
            "controller_fault",
            "timing_gap",
            "missing_observation",
            "policy_error",
        }:
            contract_breaches.append(
                "abort trigger did not produce an abort termination"
            )
        if first_abort_tick != len(trace) - 1:
            contract_breaches.append("trace continued after the first abort trigger")

    policy_actions = np.asarray(
        [step.policy_action for step in trace], dtype=np.float32
    )
    commanded_actions = np.asarray(
        [step.commanded_action for step in trace], dtype=np.float32
    )
    qpos = np.asarray([step.qpos for step in trace], dtype=np.float32)
    qvel = np.asarray([step.qvel for step in trace], dtype=np.float32)
    eligible_transitions = max(0, len(trace) - 1)
    causal_fraction = (
        float(causal_transition_count / eligible_transitions)
        if eligible_transitions
        else 0.0
    )
    trace_integrity_valid = not integrity_errors
    contract_compliant = trace_integrity_valid and not contract_breaches
    evidence_level = _evidence_level(
        contract=contract,
        trace_integrity_valid=trace_integrity_valid,
        causal_fraction=causal_fraction,
        nonzero_causal_transition_count=nonzero_causal_transition_count,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "test_id": contract.test_id,
        "trace_integrity_valid": trace_integrity_valid,
        "contract_compliant": contract_compliant,
        "integrity_errors": integrity_errors,
        "contract_breaches": contract_breaches,
        "abort_triggers": abort_triggers,
        "termination_reason": termination_reason,
        "step_count": len(trace),
        "duration_seconds_from_timestamps": float(
            (trace[-1].observation_timestamp_ns - trace[0].observation_timestamp_ns)
            * 1e-9
        ),
        "causal_state_progression": {
            "eligible_transition_count": eligible_transitions,
            "causally_linked_transition_count": causal_transition_count,
            "causal_link_fraction": causal_fraction,
            "nonzero_command_linked_transition_count": nonzero_causal_transition_count,
            "self_generated_state_evidence": evidence_level,
        },
        "action_chain": {
            "policy_nonzero_tick_count": int(
                np.sum(np.any(policy_actions != 0.0, axis=1))
            ),
            "commanded_nonzero_tick_count": int(
                np.sum(np.any(commanded_actions != 0.0, axis=1))
            ),
            "command_sent_count": int(sum(step.command_sent for step in trace)),
            "controller_ack_count": int(sum(step.controller_ack for step in trace)),
            "policy_activation_motif": [
                list(token)
                for token in activation_motif(
                    policy_actions, thresholds=contract.thresholds()
                )
            ],
            "commanded_activation_motif": [
                list(token)
                for token in activation_motif(
                    commanded_actions, thresholds=contract.thresholds()
                )
            ],
        },
        "observed_state_change": {
            "qpos_delta": (qpos[-1] - qpos[0]).astype(float).tolist(),
            "max_abs_qvel": np.max(np.abs(qvel), axis=0).astype(float).tolist(),
            "claim_boundary": (
                "descriptive only; task progress requires an independent phase or "
                "goal label"
            ),
        },
        "capability_boundaries": {
            "directly_measures": (
                "trace integrity, action-chain realization, and causal linkage "
                "between a sent command and the next observation in the declared world"
            ),
            "task_progress_estimable": False,
            "task_success_estimable": False,
            "safety_proven": False,
            "terrain_generalization_estimable": False,
            "physical_response_estimable": bool(
                evidence_level == "direct_physical_short_horizon"
            ),
            "no_promotion_from_noncausal_world": contract.state_origin
            in {"teacher_forced", "state_hold"},
        },
    }


def _audit_bounded_command(
    *,
    contract: ShortHorizonRolloutContract,
    step: ShortHorizonRolloutStep,
    index: int,
    previous_command: np.ndarray,
    breaches: list[str],
    abort_triggers: list[dict[str, Any]],
) -> None:
    command = np.asarray(step.commanded_action, dtype=np.float32)
    assert contract.command_abs_limit is not None
    assert contract.command_delta_limit is not None
    assert contract.qvel_abort_limit is not None
    assert contract.qpos_lower_limit is not None
    assert contract.qpos_upper_limit is not None
    assert contract.allowed_direction_mask is not None
    if np.any(np.abs(command) > np.asarray(contract.command_abs_limit) + 1e-7):
        breaches.append(f"tick_{index}: command_abs_limit exceeded")
    if np.any(
        np.abs(command - previous_command)
        > np.asarray(contract.command_delta_limit) + 1e-7
    ):
        breaches.append(f"tick_{index}: command_delta_limit exceeded")
    for axis_index, axis in enumerate(AXIS_NAMES):
        if (
            command[axis_index] > 0.0
            and not contract.allowed_direction_mask[axis_index][0]
        ):
            breaches.append(f"tick_{index}: {axis}+ is outside allowed directions")
        if (
            command[axis_index] < 0.0
            and not contract.allowed_direction_mask[axis_index][1]
        ):
            breaches.append(f"tick_{index}: {axis}- is outside allowed directions")
    unsafe_input = (
        (contract.require_deadman and not step.deadman_pressed)
        or step.estop_active
        or step.manual_override_active
        or step.sensor_stale
        or bool(step.policy_error)
    )
    if unsafe_input:
        abort_triggers.append({"tick": index, "reason": "runtime_safety_boundary"})
        if np.any(command != 0.0):
            breaches.append(f"tick_{index}: nonzero command at runtime safety boundary")
    qpos = np.asarray(step.qpos)
    qvel = np.asarray(step.qvel)
    if np.any(qpos < np.asarray(contract.qpos_lower_limit)) or np.any(
        qpos > np.asarray(contract.qpos_upper_limit)
    ):
        abort_triggers.append({"tick": index, "reason": "qpos_abort_limit"})
    if np.any(np.abs(qvel) > np.asarray(contract.qvel_abort_limit)):
        abort_triggers.append({"tick": index, "reason": "qvel_abort_limit"})


def _evidence_level(
    *,
    contract: ShortHorizonRolloutContract,
    trace_integrity_valid: bool,
    causal_fraction: float,
    nonzero_causal_transition_count: int,
) -> str:
    if not trace_integrity_valid:
        return "not_estimable_invalid_trace"
    if contract.state_origin in {"teacher_forced", "state_hold"}:
        return "none_noncausal_observation_source"
    if causal_fraction < 1.0:
        return "incomplete_causal_trace"
    if nonzero_causal_transition_count == 0:
        return "zero_command_state_stream_only"
    if contract.state_origin == "live_policy_on":
        return "direct_physical_short_horizon"
    if contract.state_origin == "simulator":
        return "direct_in_declared_simulator"
    return "synthetic_state_progression_proxy"


def _finite_axis_vector(value: Any, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (len(AXIS_NAMES),) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape ({len(AXIS_NAMES)},)")
    return tuple(float(item) for item in array)


def _positive_axis_vector(value: Any, name: str) -> tuple[float, ...]:
    result = _finite_axis_vector(value, name)
    if any(item <= 0.0 for item in result):
        raise ValueError(f"{name} must be positive")
    return result


def _direction_mask(value: Any) -> tuple[tuple[bool, bool], ...] | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape != (len(AXIS_NAMES), 2):
        raise ValueError("allowed_direction_mask must have shape (4, 2)")
    if not np.all(np.isin(array, [False, True, 0, 1])):
        raise ValueError("allowed_direction_mask must contain booleans")
    return tuple(tuple(bool(item) for item in row) for row in array)  # type: ignore[return-value]


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (int, np.integer)):
        result = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        result = int(value)
    else:
        raise ValueError(f"{name} must be a nonnegative integer")
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    return None if value is None else _nonnegative_int(value, name)


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _nonempty(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _sha256(value: Any, name: str) -> str:
    result = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _unique_nonempty_strings(
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(_nonempty(item, name) for item in value)
    if not result and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _timestamp_mapping(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {
        _nonempty(key, name): _nonnegative_int(item, name)
        for key, item in value.items()
    }


def _string_mapping(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {_nonempty(key, name): _nonempty(item, name) for key, item in value.items()}


__all__ = [
    "CONTROL_AUTHORITIES",
    "MAX_HORIZON_SECONDS",
    "SCHEMA_VERSION",
    "STATE_ORIGINS",
    "TERMINATION_REASONS",
    "ShortHorizonRolloutContract",
    "ShortHorizonRolloutStep",
    "evaluate_short_horizon_rollout",
]
