import numpy as np

from scripts.e40_deadzone_snap_probe import snap_actions_near_deadzone


def test_snap_actions_near_deadzone_pushes_active_positive_margin() -> None:
    actions = np.asarray([[0.49, 0.0], [0.50, 0.0]], dtype=np.float32)
    active = np.asarray([True, True])
    pos = np.asarray([0.5, 1.0], dtype=np.float32)
    neg = np.asarray([0.5, 1.0], dtype=np.float32)

    snapped = snap_actions_near_deadzone(actions, active, pos, neg, margin=0.02, epsilon=0.001)

    np.testing.assert_allclose(snapped[:, 0], [0.501, 0.50], rtol=1e-6)


def test_snap_actions_near_deadzone_respects_inactive_phase() -> None:
    actions = np.asarray([[0.49], [-0.49]], dtype=np.float32)
    active = np.asarray([False, False])
    pos = np.asarray([0.5], dtype=np.float32)
    neg = np.asarray([0.5], dtype=np.float32)

    snapped = snap_actions_near_deadzone(actions, active, pos, neg, margin=0.02, epsilon=0.001)

    np.testing.assert_allclose(snapped, actions)


def test_snap_actions_near_deadzone_pushes_active_negative_margin() -> None:
    actions = np.asarray([[-0.49], [-0.50]], dtype=np.float32)
    active = np.asarray([True, True])
    pos = np.asarray([0.5], dtype=np.float32)
    neg = np.asarray([0.5], dtype=np.float32)

    snapped = snap_actions_near_deadzone(actions, active, pos, neg, margin=0.02, epsilon=0.001)

    np.testing.assert_allclose(snapped[:, 0], [-0.501, -0.50], rtol=1e-6)
