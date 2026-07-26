from __future__ import annotations

import numpy as np
import pytest
import torch

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
