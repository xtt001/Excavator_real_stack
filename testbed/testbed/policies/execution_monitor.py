"""Causal post-command execution monitoring for a continuous policy.

The monitor is deliberately downstream of the action source. It observes the
final command that was actually sent and subsequent feedback; it does not
scale, gate, zero, or otherwise rewrite the baseline policy action. A retry is
an explicit opt-in decision for a candidate supplied by the caller and is only
issued after a completed response window reports ``stalled``.

This module owns runtime state and the input/output contract. The offline
``testbed.data.execution_response`` owner remains responsible for constructing
response labels from recorded episodes. Neither owner treats opposite qvel as
ground-truth wrong-direction motion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from testbed.data.deadzone_intent_labels import AXIS_NAMES

MONITOR_SCHEMA_VERSION = 1
_SUPPORTED_DEFAULT = ("swing", "boom", "bucket")

AxisStatus = Literal["inactive", "pending", "responded", "stalled", "unknown"]


@dataclass(frozen=True)
class ExecutionMonitorConfig:
    """Immutable monitor thresholds and bounded retry policy.

    All action thresholds are in the direct command domain. In particular,
    this config has no ``action_scale``: model output scaling belongs to the
    joystick/action-source boundary and must not compress a policy proposal.
    """

    positive_threshold: Sequence[float]
    negative_threshold: Sequence[float]
    qvel_response_threshold: Sequence[float]
    response_window_ticks: int
    min_direction_confidence: float
    supported_axes: Sequence[str] = _SUPPORTED_DEFAULT
    min_response_ticks: int = 1
    max_retries_per_event: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "positive_threshold",
            _axis_vector(self.positive_threshold, name="positive_threshold"),
        )
        object.__setattr__(
            self,
            "negative_threshold",
            _axis_vector(self.negative_threshold, name="negative_threshold"),
        )
        object.__setattr__(
            self,
            "qvel_response_threshold",
            _axis_vector(
                self.qvel_response_threshold,
                name="qvel_response_threshold",
            ),
        )
        supported = tuple(str(axis) for axis in self.supported_axes)
        if len(set(supported)) != len(supported):
            raise ValueError("supported_axes must not contain duplicates")
        unknown = sorted(set(supported).difference(AXIS_NAMES))
        if unknown:
            raise ValueError(f"supported_axes contains unknown axes: {unknown}")
        object.__setattr__(self, "supported_axes", supported)

        window = int(self.response_window_ticks)
        if window <= 0:
            raise ValueError("response_window_ticks must be positive")
        object.__setattr__(self, "response_window_ticks", window)

        min_ticks = int(self.min_response_ticks)
        if not 1 <= min_ticks <= window:
            raise ValueError(
                "min_response_ticks must satisfy 1 <= min_response_ticks "
                "<= response_window_ticks"
            )
        object.__setattr__(self, "min_response_ticks", min_ticks)

        confidence = float(self.min_direction_confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("min_direction_confidence must be in [0, 1]")
        object.__setattr__(self, "min_direction_confidence", confidence)

        retries = int(self.max_retries_per_event)
        if retries < 0:
            raise ValueError("max_retries_per_event must be non-negative")
        object.__setattr__(self, "max_retries_per_event", retries)


@dataclass(frozen=True)
class SentCommand:
    """The command that crossed the controller boundary.

    ``command`` must be the final commanded action after mechanical assist and
    independent safety processing. A policy proposal is intentionally not a
    required field: the monitor cannot infer whether an action was actually
    sent from a model prediction alone.
    """

    command: Sequence[float]
    send_timestamp_ns: int
    controller_ack: bool = True
    safety_blocked: bool = False
    retry_token: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", _action_vector(self.command, name="command"))
        timestamp = int(self.send_timestamp_ns)
        if timestamp < 0:
            raise ValueError("send_timestamp_ns must be non-negative")
        object.__setattr__(self, "send_timestamp_ns", timestamp)
        object.__setattr__(self, "controller_ack", bool(self.controller_ack))
        object.__setattr__(self, "safety_blocked", bool(self.safety_blocked))
        if self.retry_token is not None:
            token = int(self.retry_token)
            if token < 0:
                raise ValueError("retry_token must be non-negative")
            object.__setattr__(self, "retry_token", token)


@dataclass(frozen=True)
class FeedbackSample:
    """One observation used after a sent command.

    ``reset``/``gap``/``safety_active`` are hard causal boundaries. They
    produce ``unknown`` rather than a stalled label, because no retry is safe
    without knowing which command the feedback belongs to.
    """

    observation_timestamp_ns: int
    qpos: Sequence[float]
    qvel: Sequence[float]
    reset: bool = False
    gap: bool = False
    safety_active: bool = False

    def __post_init__(self) -> None:
        timestamp = int(self.observation_timestamp_ns)
        if timestamp < 0:
            raise ValueError("observation_timestamp_ns must be non-negative")
        object.__setattr__(self, "observation_timestamp_ns", timestamp)
        object.__setattr__(self, "qpos", _action_vector(self.qpos, name="qpos"))
        object.__setattr__(self, "qvel", _action_vector(self.qvel, name="qvel"))
        object.__setattr__(self, "reset", bool(self.reset))
        object.__setattr__(self, "gap", bool(self.gap))
        object.__setattr__(self, "safety_active", bool(self.safety_active))


@dataclass(frozen=True)
class ExecutionMonitorUpdate:
    """Snapshot of the monitor state after registration or feedback."""

    event_id: int | None
    statuses: tuple[AxisStatus, ...]
    effective_mask: np.ndarray
    direction: np.ndarray
    age_ticks: int
    same_direction_peak: np.ndarray
    opposite_direction_peak: np.ndarray
    retry_eligible_mask: np.ndarray
    retry_count: int
    terminal: bool
    reason: str


@dataclass(frozen=True)
class RetryDecision:
    """A permission to send an already-proposed same-direction candidate."""

    allowed: bool
    action: np.ndarray | None
    eligible_mask: np.ndarray
    event_id: int | None
    retry_token: int | None
    reason: str


@dataclass
class _PendingEvent:
    event_id: int
    command: np.ndarray
    effective_mask: np.ndarray
    direction: np.ndarray
    sent_timestamp_ns: int
    retry_count: int
    statuses: list[AxisStatus]
    age_ticks: int
    same_direction_peak: np.ndarray
    opposite_direction_peak: np.ndarray
    consecutive_response_ticks: np.ndarray


@dataclass(frozen=True)
class _IssuedRetry:
    event_id: int
    action: np.ndarray
    token: int


class ExecutionMonitor:
    """Stateful causal response monitor with an abstaining retry contract."""

    def __init__(self, config: ExecutionMonitorConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Drop pending command history at an explicit episode boundary."""

        self._next_event_id = 0
        self._active: _PendingEvent | None = None
        self._last_sent_timestamp_ns = -1
        self._last_observation_timestamp_ns = -1
        self._issued_retry: _IssuedRetry | None = None
        self._last_update: ExecutionMonitorUpdate | None = None

    @property
    def last_update(self) -> ExecutionMonitorUpdate | None:
        """Return the latest immutable snapshot for diagnostics."""

        return self._last_update

    def on_command_sent(self, command: SentCommand) -> ExecutionMonitorUpdate:
        """Register the final sent command and return the current event state.

        Repeated commands with the same effective sign continue one response
        event. A release, a new axis, or a direction switch closes unresolved
        axes as ``unknown`` and starts a new event. The monitor never rewrites
        the command returned by this method.
        """

        if not isinstance(command, SentCommand):
            raise TypeError("command must be a SentCommand")
        if command.send_timestamp_ns <= self._last_sent_timestamp_ns:
            raise ValueError(
                "send_timestamp_ns must be strictly increasing across commands"
            )
        self._last_sent_timestamp_ns = command.send_timestamp_ns
        effective, direction = self._classify(command.command)

        if command.retry_token is not None:
            return self._register_retry(command, effective, direction)

        if self._active is not None:
            same_shape = np.array_equal(
                effective, self._active.effective_mask
            ) and np.array_equal(direction, self._active.direction)
            if same_shape:
                if not command.controller_ack or command.safety_blocked:
                    self._mark_pending_unknown("transport_or_safety_blocked")
                return self._snapshot("same_direction_command")
            self._mark_pending_unknown("command_replaced_before_response")
            self._active = None
            self._issued_retry = None

        if not np.any(effective):
            update = self._inactive_snapshot(reason="no_effective_command")
            self._last_update = update
            return update

        event = self._new_event(
            command=command.command,
            effective=effective,
            direction=direction,
            sent_timestamp_ns=command.send_timestamp_ns,
            retry_count=0,
        )
        self._active = event
        if not command.controller_ack or command.safety_blocked:
            self._mark_pending_unknown("transport_or_safety_blocked")
        return self._snapshot("command_sent")

    def observe_feedback(
        self, feedback: FeedbackSample
    ) -> ExecutionMonitorUpdate | None:
        """Consume one causally ordered feedback sample.

        Same-direction qvel evidence resolves an axis as ``responded`` after
        ``min_response_ticks`` consecutive samples. Only a complete, finite,
        acked, safety-free window with no same-direction evidence resolves as
        ``stalled``. Every other interruption is ``unknown``.
        """

        if not isinstance(feedback, FeedbackSample):
            raise TypeError("feedback must be a FeedbackSample")
        if feedback.observation_timestamp_ns <= self._last_observation_timestamp_ns:
            raise ValueError(
                "observation_timestamp_ns must be strictly increasing across feedback"
            )
        self._last_observation_timestamp_ns = feedback.observation_timestamp_ns
        if self._active is None:
            return None

        if feedback.reset or feedback.gap or feedback.safety_active:
            self._mark_pending_unknown("reset_gap_or_safety_boundary")
            return self._snapshot("reset_gap_or_safety_boundary")

        if feedback.observation_timestamp_ns <= self._active.sent_timestamp_ns:
            return self._snapshot("awaiting_causal_observation")

        finite_qpos = bool(np.isfinite(feedback.qpos).all())
        finite_qvel = bool(np.isfinite(feedback.qvel).all())
        if not finite_qpos or not finite_qvel:
            self._mark_pending_unknown("nonfinite_feedback")
            return self._snapshot("nonfinite_feedback")

        event = self._active
        event.age_ticks += 1
        for axis_index, status in enumerate(event.statuses):
            if status != "pending":
                continue
            signed_qvel = float(event.direction[axis_index]) * float(
                feedback.qvel[axis_index]
            )
            opposite_qvel = -signed_qvel
            event.same_direction_peak[axis_index] = max(
                float(event.same_direction_peak[axis_index]), signed_qvel
            )
            event.opposite_direction_peak[axis_index] = max(
                float(event.opposite_direction_peak[axis_index]), opposite_qvel
            )
            if signed_qvel > float(self.config.qvel_response_threshold[axis_index]):
                event.consecutive_response_ticks[axis_index] += 1
            else:
                event.consecutive_response_ticks[axis_index] = 0
            if (
                event.consecutive_response_ticks[axis_index]
                >= self.config.min_response_ticks
            ):
                event.statuses[axis_index] = "responded"

        if event.age_ticks >= self.config.response_window_ticks:
            for axis_index, status in enumerate(event.statuses):
                if status == "pending":
                    event.statuses[axis_index] = "stalled"
            reason = "response_window_complete"
        else:
            reason = "response_window_in_progress"
        return self._snapshot(reason)

    def request_retry(
        self,
        candidate_action: Sequence[float],
        direction_confidence: Sequence[float],
    ) -> RetryDecision:
        """Validate, but do not send, a bounded same-direction retry.

        The candidate must be the caller's already-assisted action. Every
        effective candidate axis must be one that this event marked stalled,
        and every effective sign must match the original sent command.
        Sub-deadzone residual values are retained because they cannot create
        a new direct-domain command after assist has already run. A caller must
        attach the returned ``retry_token`` to the actual :class:`SentCommand`;
        a proposal that is never sent cannot change monitor state.
        """

        candidate = _action_vector(candidate_action, name="candidate_action")
        confidence = _axis_vector(
            direction_confidence,
            name="direction_confidence",
            nonnegative=True,
        )
        if np.any(candidate < -1.0) or np.any(candidate > 1.0):
            return self._deny_retry("candidate_action_out_of_range")
        if np.any(confidence > 1.0):
            return self._deny_retry("direction_confidence_out_of_range")
        event = self._active
        if event is None:
            return self._deny_retry("no_active_event")
        if self._issued_retry is not None:
            return self._deny_retry("retry_already_issued")
        if event.retry_count >= self.config.max_retries_per_event:
            return self._deny_retry("retry_limit_reached")

        candidate_effective, candidate_direction = self._classify(candidate)
        eligible = np.asarray(
            [
                status == "stalled"
                and bool(confidence[index] >= self.config.min_direction_confidence)
                for index, status in enumerate(event.statuses)
            ],
            dtype=bool,
        )
        if np.any(
            candidate_effective
            & (~event.effective_mask | (candidate_direction != event.direction))
        ):
            return self._deny_retry("candidate_introduces_new_or_opposite_direction")
        if not np.any(candidate_effective):
            return self._deny_retry("candidate_has_no_effective_axis")
        if np.any(candidate_effective & ~eligible):
            return self._deny_retry("candidate_axis_not_high_confidence_stalled")

        token = int(self._next_event_id + 1_000_000)
        self._issued_retry = _IssuedRetry(
            event_id=event.event_id,
            action=candidate.copy(),
            token=token,
        )
        return RetryDecision(
            allowed=True,
            action=candidate.copy(),
            eligible_mask=eligible.copy(),
            event_id=event.event_id,
            retry_token=token,
            reason="stalled_high_confidence_same_direction",
        )

    def _register_retry(
        self,
        command: SentCommand,
        effective: np.ndarray,
        direction: np.ndarray,
    ) -> ExecutionMonitorUpdate:
        issued = self._issued_retry
        event = self._active
        if issued is None or event is None:
            raise ValueError("retry command has no matching issued retry token")
        if issued.token != command.retry_token:
            raise ValueError("retry_token does not match the issued retry")
        if not np.array_equal(command.command, issued.action):
            raise ValueError("sent retry command differs from the approved candidate")
        if not np.any(effective):
            raise ValueError("sent retry command has no effective axis")
        if np.any(
            effective & (~event.effective_mask | (direction != event.direction))
        ):
            raise ValueError("sent retry command changes the approved direction set")
        eligible = np.asarray(
            [status == "stalled" for status in event.statuses], dtype=bool
        )
        if np.any(effective & ~eligible):
            raise ValueError("sent retry command includes a non-stalled axis")
        retry_count = event.retry_count + 1
        self._issued_retry = None
        self._active = self._new_event(
            command=command.command,
            effective=effective,
            direction=direction,
            sent_timestamp_ns=command.send_timestamp_ns,
            retry_count=retry_count,
        )
        if not command.controller_ack or command.safety_blocked:
            self._mark_pending_unknown("retry_transport_or_safety_blocked")
        return self._snapshot("retry_command_sent")

    def _new_event(
        self,
        *,
        command: np.ndarray,
        effective: np.ndarray,
        direction: np.ndarray,
        sent_timestamp_ns: int,
        retry_count: int,
    ) -> _PendingEvent:
        event = _PendingEvent(
            event_id=self._next_event_id,
            command=command.copy(),
            effective_mask=effective.copy(),
            direction=direction.copy(),
            sent_timestamp_ns=int(sent_timestamp_ns),
            retry_count=int(retry_count),
            statuses=["pending" if bool(value) else "inactive" for value in effective],
            age_ticks=0,
            same_direction_peak=np.full(4, -np.inf, dtype=np.float32),
            opposite_direction_peak=np.full(4, -np.inf, dtype=np.float32),
            consecutive_response_ticks=np.zeros(4, dtype=np.int32),
        )
        self._next_event_id += 1
        return event

    def _classify(self, action: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        values = _action_vector(action, name="action")
        positive = values >= np.asarray(self.config.positive_threshold)
        negative = values <= -np.asarray(self.config.negative_threshold)
        supported = np.asarray(
            [axis in self.config.supported_axes for axis in AXIS_NAMES], dtype=bool
        )
        effective = (positive | negative) & supported
        direction = np.where(positive, 1, np.where(negative, -1, 0)).astype(np.int8)
        direction[~effective] = 0
        return effective.astype(bool), direction

    def _mark_pending_unknown(self, reason: str) -> None:
        if self._active is None:
            return
        self._active.statuses = [
            "unknown" if status == "pending" else status
            for status in self._active.statuses
        ]
        self._last_update = self._snapshot(reason)

    def _snapshot(self, reason: str) -> ExecutionMonitorUpdate:
        event = self._active
        if event is None:
            return self._inactive_snapshot(reason=reason)
        statuses = tuple(event.statuses)
        retry_eligible = np.asarray(
            [status == "stalled" for status in statuses], dtype=bool
        )
        update = ExecutionMonitorUpdate(
            event_id=event.event_id,
            statuses=statuses,
            effective_mask=event.effective_mask.copy(),
            direction=event.direction.copy(),
            age_ticks=int(event.age_ticks),
            same_direction_peak=_finite_peak(event.same_direction_peak),
            opposite_direction_peak=_finite_peak(event.opposite_direction_peak),
            retry_eligible_mask=retry_eligible,
            retry_count=int(event.retry_count),
            terminal=all(
                status in {"inactive", "responded", "stalled", "unknown"}
                for status in statuses
            ),
            reason=str(reason),
        )
        self._last_update = update
        return update

    def _inactive_snapshot(self, *, reason: str) -> ExecutionMonitorUpdate:
        return ExecutionMonitorUpdate(
            event_id=None,
            statuses=("inactive",) * len(AXIS_NAMES),
            effective_mask=np.zeros(4, dtype=bool),
            direction=np.zeros(4, dtype=np.int8),
            age_ticks=0,
            same_direction_peak=np.full(4, np.nan, dtype=np.float32),
            opposite_direction_peak=np.full(4, np.nan, dtype=np.float32),
            retry_eligible_mask=np.zeros(4, dtype=bool),
            retry_count=0,
            terminal=True,
            reason=str(reason),
        )

    def _deny_retry(self, reason: str) -> RetryDecision:
        return RetryDecision(
            allowed=False,
            action=None,
            eligible_mask=np.zeros(4, dtype=bool),
            event_id=None if self._active is None else self._active.event_id,
            retry_token=None,
            reason=str(reason),
        )


def replay_execution_monitor_trace(
    monitor: ExecutionMonitor,
    inputs: Sequence[SentCommand | FeedbackSample],
) -> tuple[ExecutionMonitorUpdate, ...]:
    """Replay an interleaved sent-command/feedback trace for offline eval.

    This is intentionally a thin evaluation interface: it does not synthesize
    feedback or train a retry policy. A sidecar adapter may construct the
    interleaved trace from causally aligned timestamps, while an on-policy
    collector can pass the same objects from its execution logger.
    """

    if not isinstance(monitor, ExecutionMonitor):
        raise TypeError("monitor must be an ExecutionMonitor")
    updates: list[ExecutionMonitorUpdate] = []
    for item in inputs:
        if isinstance(item, SentCommand):
            updates.append(monitor.on_command_sent(item))
        elif isinstance(item, FeedbackSample):
            update = monitor.observe_feedback(item)
            if update is not None:
                updates.append(update)
        else:
            raise TypeError(
                "trace inputs must contain only SentCommand or FeedbackSample"
            )
    return tuple(updates)


def _axis_vector(
    value: Sequence[float],
    *,
    name: str,
    nonnegative: bool = True,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (len(AXIS_NAMES),):
        raise ValueError(f"{name} must have shape (4,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must contain non-negative values")
    return array.copy()


def _action_vector(value: Sequence[float], *, name: str) -> np.ndarray:
    return _axis_vector(value, name=name, nonnegative=False)


def _finite_peak(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    result[~np.isfinite(result)] = np.nan
    return result
