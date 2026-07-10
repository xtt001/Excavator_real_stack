from __future__ import annotations

import numpy as np

from testbed.policies.trajectory_transition_eval import (
    build_transition_samples,
    fit_feature_support_model,
    fit_linear_transition_model,
)


def _thresholds(value: float = 0.2) -> dict[str, dict[str, float]]:
    return {
        axis: {"pos": value, "neg": value}
        for axis in ("swing", "boom", "stick", "bucket")
    }


def test_build_transition_samples_applies_axis_sign_contracts() -> None:
    action = np.zeros((6, 4), dtype=np.float64)
    action[:, 0] = 0.6
    action[:, 1] = 0.6
    qvel = np.zeros((6, 4), dtype=np.float64)
    qvel[:, 0] = 0.5
    qvel[:, 1] = 0.5
    qpos = np.zeros((6, 4), dtype=np.float64)
    qpos[:, 0] = np.arange(6) * 0.05
    qpos[:, 1] = -np.arange(6) * 0.05

    samples = build_transition_samples(
        qpos=qpos,
        qvel=qvel,
        action=action,
        thresholds=_thresholds(),
        dt=0.05,
        horizon_steps=2,
        stride=2,
        qvel_to_qpos_sign=np.array([1.0, -1.0, 1.0, 1.0]),
        action_to_qpos_sign=np.array([1.0, -1.0, 1.0, 1.0]),
    )

    assert samples.start_steps.tolist() == [0, 2]
    np.testing.assert_allclose(samples.target_qpos_delta[:, :2], [[0.1, -0.1], [0.1, -0.1]])
    np.testing.assert_allclose(samples.initial_qvel_displacement[:, :2], [[0.05, -0.05], [0.05, -0.05]])
    np.testing.assert_allclose(samples.action_impulse[:, 0], 0.05)
    np.testing.assert_allclose(samples.action_impulse[:, 1], -0.05)


def test_build_transition_samples_wraps_swing_delta() -> None:
    qpos = np.zeros((4, 4), dtype=np.float64)
    qpos[:, 0] = [3.10, 3.13, -3.12, -3.10]

    samples = build_transition_samples(
        qpos=qpos,
        qvel=np.zeros_like(qpos),
        action=np.zeros_like(qpos),
        thresholds=_thresholds(),
        dt=0.05,
        horizon_steps=1,
        stride=1,
        qvel_to_qpos_sign=np.ones(4),
        action_to_qpos_sign=np.ones(4),
    )

    assert np.all(np.abs(samples.target_qpos_delta[:, 0]) < 0.05)


def test_linear_transition_and_support_models_are_training_only() -> None:
    qvel = np.linspace(-1.0, 1.0, 21)
    action = np.linspace(0.5, -0.5, 21)
    target = 0.1 + 2.0 * qvel - 3.0 * action
    model = fit_linear_transition_model(qvel, action, target)

    np.testing.assert_allclose(model.predict(qvel, action), target, atol=1.0e-12)
    support = fit_feature_support_model(np.column_stack([qvel, action]), quantile=0.9)
    train_coverage = np.mean(
        support.distances(np.column_stack([qvel, action])) <= support.distance_threshold
    )
    outside_distance = support.distances(np.array([[10.0, 10.0]]))[0]

    assert train_coverage >= 0.85
    assert outside_distance > support.distance_threshold
