from __future__ import annotations

import numpy as np
import pytest
import torch

from testbed.policies.act.adapter import ACTAdapter
from testbed.policies.act.phase_routed_condition import RoutedConditionProjection
from testbed.simverify.observable_phase_router import (
    ObservablePhaseRouter,
    apply_monotonic_router,
    fit_diagonal_gaussian,
    predict_raw_routes,
)


def _training_rows() -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(
        [[-2.0] * 8, [-1.8] * 8, [-2.2] * 8],
        dtype=np.float64,
    )
    neutral = np.asarray(
        [[0.0] * 8, [0.2] * 8, [-0.2] * 8],
        dtype=np.float64,
    )
    next_rows = np.asarray(
        [[2.0] * 8, [1.8] * 8, [2.2] * 8],
        dtype=np.float64,
    )
    return (
        np.concatenate((current, neutral, next_rows), axis=0),
        np.repeat(np.arange(3), 3),
    )


def test_diagonal_classifier_identifies_three_routes() -> None:
    features, labels = _training_rows()
    classifier = fit_diagonal_gaussian(features, labels)
    predicted = predict_raw_routes(features, classifier)
    np.testing.assert_array_equal(predicted, labels)


def test_monotonic_router_requires_next_state_and_dwell() -> None:
    raw = np.asarray([0, 2, 1, 1, 0, 2, 2, 1, 0], dtype=np.int8)
    routed = apply_monotonic_router(raw, dwell_steps=2)
    np.testing.assert_array_equal(
        routed,
        np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int8),
    )


def test_runtime_router_reset_and_parity() -> None:
    features, labels = _training_rows()
    classifier = fit_diagonal_gaussian(features, labels)
    sequence = np.concatenate(
        (
            np.full((2, 8), -2.0),
            np.full((2, 8), 0.0),
            np.full((2, 8), 2.0),
        ),
        axis=0,
    )
    expected = apply_monotonic_router(
        predict_raw_routes(sequence, classifier),
        dwell_steps=2,
    )
    router = ObservablePhaseRouter(classifier, dwell_steps=2)
    observed = np.asarray(
        [router.step(row[:4], row[4:]) for row in sequence],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(observed, expected)
    router.reset()
    assert router.route == 0


def test_condition_cycle_reset_preserves_temporal_adapter_state() -> None:
    class _Router:
        route = 2
        consecutive = 7

        def reset(self) -> None:
            self.route = 0
            self.consecutive = 0

    adapter = object.__new__(ACTAdapter)
    adapter._condition_phase_router = _Router()
    adapter._last_condition_route_diagnostics = {"route_index": 2}
    adapter._t = 37
    adapter._all_time_actions = torch.ones((2, 3, 4))
    adapter._cached_actions = torch.ones((5, 4))
    all_time_actions = adapter._all_time_actions
    cached_actions = adapter._cached_actions

    adapter.reset_condition_cycle()

    assert adapter._condition_phase_router.route == 0
    assert adapter._condition_phase_router.consecutive == 0
    assert adapter._last_condition_route_diagnostics is None
    assert adapter._t == 37
    assert adapter._all_time_actions is all_time_actions
    assert adapter._cached_actions is cached_actions


def test_router_rejects_non_finite_observation() -> None:
    features, labels = _training_rows()
    classifier = fit_diagonal_gaussian(features, labels)
    router = ObservablePhaseRouter(classifier, dwell_steps=1)
    with pytest.raises(ValueError, match="finite"):
        router.step(np.asarray([np.nan, 0.0, 0.0, 0.0]), np.zeros(4))


def test_routed_projection_separates_factors_and_neutral_is_exactly_invariant() -> None:
    projection = RoutedConditionProjection(hidden_dim=3)
    with torch.no_grad():
        projection.state.weight.zero_()
        projection.state.bias.zero_()
        projection.current.weight.copy_(torch.eye(3))
        projection.current.bias.zero_()
        projection.next.weight.copy_(2.0 * torch.eye(3))
        projection.next.bias.zero_()
    proprio = torch.zeros((4, 14), dtype=torch.float32)
    proprio[0, 8] = 1.0
    proprio[0, 12] = 1.0
    proprio[1, 9] = 1.0
    proprio[1, 13] = 1.0
    proprio[2] = proprio[0]
    proprio[3] = proprio[1]

    current = projection(proprio[:2], torch.tensor([0, 0]))
    neutral = projection(proprio[2:], torch.tensor([1, 1]))
    next_route = projection(proprio[:2], torch.tensor([2, 2]))

    torch.testing.assert_close(
        current,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    torch.testing.assert_close(neutral, torch.zeros_like(neutral))
    torch.testing.assert_close(
        next_route,
        torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
    )


def test_routed_projection_fails_closed_without_valid_route() -> None:
    projection = RoutedConditionProjection(hidden_dim=4)
    with pytest.raises(ValueError, match="routes"):
        projection(torch.zeros((1, 14)), torch.tensor([3]))


def test_next_only_projection_never_reads_current_sector() -> None:
    projection = RoutedConditionProjection(
        hidden_dim=3,
        factor_mode="next_only",
    )
    assert projection.current is None
    with torch.no_grad():
        projection.state.weight.zero_()
        projection.state.bias.zero_()
        projection.next.weight.copy_(torch.eye(3))
        projection.next.bias.zero_()
    left_current = torch.zeros((3, 14), dtype=torch.float32)
    right_current = left_current.clone()
    left_current[:, 8] = 1.0
    right_current[:, 10] = 1.0
    left_current[:, 12] = 1.0
    right_current[:, 12] = 1.0
    routes = torch.tensor([0, 1, 2])

    left_output = projection(left_current, routes)
    right_output = projection(right_current, routes)

    torch.testing.assert_close(left_output, right_output)
    torch.testing.assert_close(left_output[:2], torch.zeros((2, 3)))
    torch.testing.assert_close(
        left_output[2:],
        torch.tensor([[0.0, 1.0, 0.0]]),
    )
