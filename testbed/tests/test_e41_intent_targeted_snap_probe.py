import numpy as np

from scripts.e41_intent_targeted_snap_probe import snap_actions_near_deadzone_with_intent


def test_snap_actions_near_deadzone_with_intent_requires_matching_direction() -> None:
    actions = np.asarray([[0.49, -0.49, 0.0, 0.0]], dtype=np.float32)
    active = np.asarray([True])
    intent = np.asarray([[0.9, 0.1, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    pos = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    neg = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    snapped = snap_actions_near_deadzone_with_intent(
        actions,
        active,
        intent,
        pos,
        neg,
        margin=0.02,
        epsilon=0.001,
        intent_threshold=0.8,
    )

    np.testing.assert_allclose(snapped[0, :2], [0.501, -0.501], rtol=1e-6)
    np.testing.assert_allclose(snapped[0, 2:], [0.0, 0.0], rtol=1e-6)


def test_snap_actions_near_deadzone_with_intent_rejects_low_intent() -> None:
    actions = np.asarray([[0.49, 0.0, 0.0, 0.0]], dtype=np.float32)
    active = np.asarray([True])
    intent = np.asarray([[0.79, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    pos = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    neg = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    snapped = snap_actions_near_deadzone_with_intent(
        actions,
        active,
        intent,
        pos,
        neg,
        margin=0.02,
        epsilon=0.001,
        intent_threshold=0.8,
    )

    np.testing.assert_allclose(snapped, actions)


def test_snap_actions_near_deadzone_with_intent_respects_inactive_phase() -> None:
    actions = np.asarray([[0.49, 0.0, 0.0, 0.0]], dtype=np.float32)
    active = np.asarray([False])
    intent = np.asarray([[0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    pos = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    neg = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    snapped = snap_actions_near_deadzone_with_intent(
        actions,
        active,
        intent,
        pos,
        neg,
        margin=0.02,
        epsilon=0.001,
        intent_threshold=0.8,
    )

    np.testing.assert_allclose(snapped, actions)
