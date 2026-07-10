import numpy as np

from testbed.policies.phase_gate import (
    apply_direction_gate_to_actions,
    apply_phase_gate_to_actions,
    build_hysteresis_mask,
    direction_effective_labels,
    should_move_labels,
)


def test_hysteresis_mask_opens_high_and_holds_until_close() -> None:
    probs = np.asarray([0.10, 0.35, 0.25, 0.18, 0.21, 0.10, 0.40], dtype=np.float32)

    mask = build_hysteresis_mask(probs, open_threshold=0.30, close_threshold=0.20)

    np.testing.assert_array_equal(mask, [False, True, True, False, False, False, True])


def test_apply_phase_gate_zeroes_inactive_policy_steps() -> None:
    policy = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=np.float32,
    )
    active = np.asarray([False, True], dtype=bool)

    gated = apply_phase_gate_to_actions(policy, active)

    np.testing.assert_allclose(gated[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(gated[1], policy[1])
    assert gated.dtype == np.float32


def test_apply_phase_gate_can_attenuate_inactive_policy_steps() -> None:
    policy = np.asarray(
        [
            [0.4, -0.2, 0.1, -0.8],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=np.float32,
    )
    active = np.asarray([False, True], dtype=bool)

    gated = apply_phase_gate_to_actions(policy, active, inactive_scale=0.25)

    np.testing.assert_allclose(gated[0], [0.1, -0.05, 0.025, -0.2])
    np.testing.assert_allclose(gated[1], policy[1])


def test_should_move_labels_follow_any_directional_deadzone_crossing() -> None:
    thresholds = {
        "swing": {"pos": 0.5, "neg": 0.5},
        "boom": {"pos": 0.4, "neg": 0.4},
        "stick": {"pos": 0.3, "neg": 0.3},
        "bucket": {"pos": 0.2, "neg": 0.2},
    }
    expert = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.51, 0.0, 0.0, 0.0],
            [0.0, -0.41, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.19],
        ],
        dtype=np.float32,
    )

    labels = should_move_labels(expert, thresholds)

    np.testing.assert_array_equal(labels, [False, True, True, False])


def test_direction_effective_labels_keep_axis_direction_shape() -> None:
    thresholds = {
        "swing": {"pos": 0.5, "neg": 0.5},
        "boom": {"pos": 0.4, "neg": 0.4},
        "stick": {"pos": 0.3, "neg": 0.3},
        "bucket": {"pos": 0.2, "neg": 0.2},
    }
    expert = np.asarray(
        [
            [0.51, -0.41, 0.0, 0.21],
            [-0.52, 0.0, 0.31, -0.22],
        ],
        dtype=np.float32,
    )

    labels = direction_effective_labels(expert, thresholds)

    assert labels.shape == (2, 8)
    np.testing.assert_array_equal(
        labels,
        [
            [True, False, False, True, False, False, True, False],
            [False, True, False, False, True, False, False, True],
        ],
    )


def test_apply_direction_gate_to_actions_scales_only_inactive_directions() -> None:
    policy = np.asarray([[0.8, -0.6, 0.4, -0.2]], dtype=np.float32)
    active = np.asarray(
        [[True, False, False, True, False, True, True, False]],
        dtype=bool,
    )

    gated = apply_direction_gate_to_actions(policy, active, inactive_scale=0.25)

    np.testing.assert_allclose(gated, [[0.8, -0.6, 0.1, -0.05]], rtol=1e-6)
