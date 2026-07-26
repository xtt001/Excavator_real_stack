from __future__ import annotations

import numpy as np
import pytest

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

