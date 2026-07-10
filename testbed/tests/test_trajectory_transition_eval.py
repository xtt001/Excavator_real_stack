from __future__ import annotations

import numpy as np

from testbed.policies.trajectory_transition_eval import build_transition_samples


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
