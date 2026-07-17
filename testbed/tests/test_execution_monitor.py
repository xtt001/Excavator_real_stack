from __future__ import annotations

import numpy as np
import pytest

from testbed.policies.execution_monitor import (
    ExecutionMonitor,
    ExecutionMonitorConfig,
    FeedbackSample,
    SentCommand,
    replay_execution_monitor_trace,
)

POSITIVE = [0.661, 0.259, 0.5, 0.408]
NEGATIVE = [0.721, 0.357, 0.5, 0.508]


def _monitor(*, window: int = 3) -> ExecutionMonitor:
    return ExecutionMonitor(
        ExecutionMonitorConfig(
            positive_threshold=POSITIVE,
            negative_threshold=NEGATIVE,
            qvel_response_threshold=[0.1, 0.1, 0.1, 0.1],
            response_window_ticks=window,
            min_direction_confidence=0.8,
        )
    )


def _feedback(timestamp: int, qvel: list[float], **kwargs: object) -> FeedbackSample:
    return FeedbackSample(
        observation_timestamp_ns=timestamp,
        qpos=[0.0, 0.0, 0.0, 0.0],
        qvel=qvel,
        **kwargs,
    )


def test_stick_is_structural_zero_and_not_a_monitor_event() -> None:
    monitor = _monitor()

    update = monitor.on_command_sent(SentCommand([0.0, 0.0, 0.9, 0.0], 10))

    assert update.event_id is None
    assert update.statuses == ("inactive",) * 4


def test_response_is_causal_and_preserves_opposite_motion_as_diagnostic() -> None:
    monitor = _monitor(window=3)
    sent = SentCommand([0.0, 0.3, 0.0, 0.0], 100)

    start = monitor.on_command_sent(sent)
    assert start.statuses[1] == "pending"
    waiting = monitor.observe_feedback(_feedback(100, [0.0, 0.2, 0.0, 0.0]))
    assert waiting is not None
    assert waiting.statuses[1] == "pending"
    assert waiting.age_ticks == 0

    result = monitor.observe_feedback(_feedback(101, [0.0, -0.2, 0.0, 0.0]))

    assert result is not None
    assert result.statuses[1] == "pending"
    np.testing.assert_allclose(result.same_direction_peak[1], -0.2)
    np.testing.assert_allclose(result.opposite_direction_peak[1], 0.2)
    assert not bool(result.retry_eligible_mask[1])


def test_complete_finite_window_marks_stalled_and_allows_only_high_confidence_retry() -> None:
    monitor = _monitor(window=3)
    monitor.on_command_sent(SentCommand([0.0, 0.3, 0.0, 0.0], 100))

    for timestamp in (101, 102, 103):
        result = monitor.observe_feedback(_feedback(timestamp, [0.0, 0.0, 0.0, 0.0]))
    assert result is not None
    assert result.statuses[1] == "stalled"
    assert result.retry_eligible_mask[1]

    denied = monitor.request_retry(
        [0.0, 0.35, 0.0, 0.0], [0.0, 0.79, 0.0, 0.0]
    )
    assert not denied.allowed
    assert denied.reason == "candidate_axis_not_high_confidence_stalled"

    retry = monitor.request_retry(
        [0.0, 0.35, 0.0, 0.0], [0.0, 0.95, 0.0, 0.0]
    )
    assert retry.allowed
    assert retry.retry_token is not None
    np.testing.assert_allclose(retry.action, [0.0, 0.35, 0.0, 0.0])

    retry_start = monitor.on_command_sent(
        SentCommand(
            [0.0, 0.35, 0.0, 0.0],
            104,
            retry_token=retry.retry_token,
        )
    )
    assert retry_start.retry_count == 1
    assert retry_start.statuses[1] == "pending"

    retry_result = monitor.observe_feedback(_feedback(105, [0.0, 0.2, 0.0, 0.0]))
    assert retry_result is not None
    assert retry_result.statuses[1] == "responded"


def test_retry_abstains_on_new_axis_opposite_sign_or_non_stalled_axis() -> None:
    monitor = _monitor(window=1)
    monitor.on_command_sent(SentCommand([0.0, 0.3, 0.0, 0.0], 1))
    stalled = monitor.observe_feedback(_feedback(2, [0.0, 0.0, 0.0, 0.0]))
    assert stalled is not None and stalled.statuses[1] == "stalled"

    for candidate, reason in (
        ([0.0, -0.4, 0.0, 0.0], "candidate_introduces_new_or_opposite_direction"),
        ([0.0, 0.35, 0.0, 0.45], "candidate_introduces_new_or_opposite_direction"),
        ([0.0, 0.0, 0.0, 0.0], "candidate_has_no_effective_axis"),
    ):
        decision = monitor.request_retry(candidate, [0.0, 0.99, 0.0, 0.99])
        assert not decision.allowed
        assert decision.reason == reason


def test_retry_ignores_sub_deadzone_noise_when_final_command_is_already_assisted() -> None:
    monitor = _monitor(window=1)
    monitor.on_command_sent(SentCommand([0.0, 0.3, 0.0, 0.0], 1))
    stalled = monitor.observe_feedback(_feedback(2, [0.0, 0.0, 0.0, 0.0]))
    assert stalled is not None and stalled.statuses[1] == "stalled"

    decision = monitor.request_retry(
        [0.03, 0.35, -0.02, 0.01], [0.0, 0.99, 0.0, 0.0]
    )

    assert decision.allowed


def test_reset_gap_safety_and_transport_failures_are_unknown_not_stalled() -> None:
    monitor = _monitor(window=2)
    monitor.on_command_sent(SentCommand([0.0, 0.3, 0.0, 0.0], 10))

    result = monitor.observe_feedback(
        _feedback(11, [0.0, 0.0, 0.0, 0.0], gap=True)
    )
    assert result is not None
    assert result.statuses[1] == "unknown"
    assert not monitor.request_retry([0.0, 0.35, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]).allowed

    monitor.reset()
    result = monitor.on_command_sent(
        SentCommand([0.0, 0.3, 0.0, 0.0], 20, controller_ack=False)
    )
    assert result.statuses[1] == "unknown"


def test_release_or_direction_switch_cannot_become_a_false_stall() -> None:
    monitor = _monitor(window=3)
    monitor.on_command_sent(SentCommand([0.0, 0.3, 0.0, 0.0], 1))
    inactive = monitor.on_command_sent(SentCommand([0.0, 0.0, 0.0, 0.0], 2))

    assert inactive.event_id is None
    assert monitor.last_update is not None
    assert monitor.last_update.statuses[1] == "inactive"


def test_timestamps_must_be_strictly_monotonic() -> None:
    monitor = _monitor()
    monitor.on_command_sent(SentCommand([0.0, 0.3, 0.0, 0.0], 10))

    with pytest.raises(ValueError, match="strictly increasing"):
        monitor.on_command_sent(SentCommand([0.0, 0.3, 0.0, 0.0], 10))
    with pytest.raises(ValueError, match="strictly increasing"):
        monitor.observe_feedback(_feedback(11, [0.0, 0.0, 0.0, 0.0]))
        monitor.observe_feedback(_feedback(11, [0.0, 0.0, 0.0, 0.0]))


def test_interleaved_trace_replay_is_a_thin_offline_evaluation_interface() -> None:
    monitor = _monitor(window=2)
    trace = [
        SentCommand([0.0, 0.3, 0.0, 0.0], 10),
        _feedback(10, [0.0, 0.0, 0.0, 0.0]),
        _feedback(11, [0.0, 0.0, 0.0, 0.0]),
        _feedback(12, [0.0, 0.0, 0.0, 0.0]),
    ]

    updates = replay_execution_monitor_trace(monitor, trace)

    assert updates[0].reason == "command_sent"
    assert updates[-1].statuses[1] == "stalled"
