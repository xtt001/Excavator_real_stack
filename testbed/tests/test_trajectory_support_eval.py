from __future__ import annotations

import numpy as np
import pytest

from testbed.policies.trajectory_support_eval import (
    compute_intent_horizon_rows,
    cumulative_intent,
    effective_action_channels,
    impulse_metrics,
)


def _thresholds(value: float = 0.2) -> dict[str, dict[str, float]]:
    return {
        axis: {"pos": value, "neg": value}
        for axis in ("swing", "boom", "stick", "bucket")
    }


def test_effective_action_channels_remove_directional_deadzone() -> None:
    actions = np.array(
        [
            [0.2, 0.6, -0.6, 0.0],
            [-0.2, -1.0, 1.0, 0.1],
        ],
        dtype=np.float32,
    )

    channels = effective_action_channels(actions, _thresholds())

    assert channels.shape == (2, 4, 2)
    np.testing.assert_allclose(channels[0, 0], [0.0, 0.0])
    np.testing.assert_allclose(channels[0, 1], [0.5, 0.0])
    np.testing.assert_allclose(channels[0, 2], [0.0, 0.5])
    np.testing.assert_allclose(channels[1, 1], [0.0, 1.0])
    np.testing.assert_allclose(channels[1, 2], [1.0, 0.0])


def test_cumulative_intent_has_zero_origin_and_uses_dt() -> None:
    channels = np.zeros((3, 4, 2), dtype=np.float32)
    channels[:, 1, 0] = [0.25, 0.5, 1.0]

    cumulative = cumulative_intent(channels, dt=0.05)

    assert cumulative.shape == (4, 4, 2)
    np.testing.assert_allclose(cumulative[0], 0.0)
    np.testing.assert_allclose(cumulative[:, 1, 0], [0.0, 0.0125, 0.0375, 0.0875])


def test_impulse_metrics_do_not_hide_opposite_direction_cancellation() -> None:
    expert = np.zeros((4, 2), dtype=np.float64)
    policy = np.zeros((4, 2), dtype=np.float64)
    expert[0, 0] = 1.0
    policy[0] = [1.0, 1.0]

    metrics = impulse_metrics(expert, policy)

    assert metrics["net_axis_l1_error"] == pytest.approx(1.0)
    assert metrics["channel_l1_error"] == pytest.approx(1.0)
    assert metrics["policy_cancellation_ratio"] == pytest.approx(1.0)
    assert metrics["expert_cancellation_ratio"] == pytest.approx(0.0)


def test_impulse_metrics_mark_zero_expert_denominators_invalid() -> None:
    expert = np.zeros((4, 2), dtype=np.float64)
    policy = np.zeros((4, 2), dtype=np.float64)
    policy[2, 0] = 0.5

    metrics = impulse_metrics(expert, policy)

    assert metrics["magnitude_ratio"] is None
    assert metrics["direction_cosine"] is None
    assert metrics["missing_expert_impulse"] == 0.0
    assert metrics["extra_policy_impulse"] == pytest.approx(0.5)


def test_horizon_rows_align_windows_and_preserve_partial_range() -> None:
    expert = np.zeros((6, 4), dtype=np.float32)
    policy = np.zeros((6, 4), dtype=np.float32)
    expert[1:5, 0] = 1.0
    policy[2:5, 0] = 1.0

    rows = compute_intent_horizon_rows(
        expert,
        policy,
        _thresholds(),
        dt=0.05,
        horizons=(2, 4, 8),
        start=1,
        end=6,
        stride=2,
    )

    assert [(row["horizon_steps"], row["start_step"], row["end_step_exclusive"]) for row in rows] == [
        (2, 1, 3),
        (2, 3, 5),
        (4, 1, 5),
    ]
    assert rows[0]["missing_expert_impulse"] > 0.0
    assert rows[1]["missing_expert_impulse"] == pytest.approx(0.0)
    assert all("cumulative_path_mean_channel_l1" in row for row in rows)


@pytest.mark.parametrize("dt", [0.0, -0.05])
def test_cumulative_intent_rejects_non_positive_dt(dt: float) -> None:
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        cumulative_intent(np.zeros((2, 4, 2), dtype=np.float32), dt=dt)
