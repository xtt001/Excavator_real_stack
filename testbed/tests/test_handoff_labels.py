from __future__ import annotations

import numpy as np

from testbed.data.handoff_labels import compute_gohome_eligibility_labels


def test_gohome_eligibility_starts_after_human_idle_dwell_and_ignores_after_go() -> None:
    actions = np.zeros((12, 4), dtype=np.float32)
    actions[4, 1] = 0.08
    requested = np.zeros(12, dtype=np.uint8)
    accepted = np.zeros(12, dtype=np.uint8)
    running = np.zeros(12, dtype=np.uint8)
    requested[10] = 1
    accepted[9] = 1
    running[11] = 1

    labels = compute_gohome_eligibility_labels(
        actions=actions,
        go_home_requested=requested,
        go_home_start_accepted=accepted,
        go_home_running=running,
        idle_action_threshold=0.05,
        dwell_min_steps=2,
    )

    assert labels.t_go == 9
    assert labels.t_stop == 5
    assert labels.eligible_start == 7
    np.testing.assert_array_equal(
        labels.gohome_eligible_label.astype(np.uint8),
        np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        labels.gohome_loss_mask.astype(np.uint8),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        labels.tail_idle_mask.astype(np.uint8),
        np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        labels.action_loss_mask.astype(np.uint8),
        np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        labels.owner_automation.astype(np.uint8),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.uint8),
    )


def test_gohome_eligibility_can_have_no_positive_when_dwell_exceeds_human_idle() -> None:
    actions = np.zeros((8, 4), dtype=np.float32)
    actions[5, 0] = -0.07
    requested = np.zeros(8, dtype=np.uint8)
    requested[7] = 1

    labels = compute_gohome_eligibility_labels(
        actions=actions,
        go_home_requested=requested,
        go_home_start_accepted=None,
        go_home_running=None,
        idle_action_threshold=0.05,
        dwell_min_steps=3,
    )

    assert labels.t_go == 7
    assert labels.t_stop == 6
    assert labels.eligible_start == 9
    assert not np.any(labels.gohome_eligible_label)
    assert np.all(labels.gohome_loss_mask[:8])
    assert not np.any(labels.action_loss_mask[6:])


def test_gohome_eligibility_without_gohome_marker_disables_handoff_supervision() -> None:
    actions = np.zeros((6, 4), dtype=np.float32)
    labels = compute_gohome_eligibility_labels(
        actions=actions,
        go_home_requested=None,
        go_home_start_accepted=None,
        go_home_running=None,
        idle_action_threshold=0.05,
        dwell_min_steps=2,
    )

    assert labels.t_go is None
    assert labels.t_stop is None
    assert labels.eligible_start is None
    assert not np.any(labels.gohome_eligible_label)
    assert not np.any(labels.gohome_loss_mask)
    assert not np.any(labels.tail_idle_mask)
    assert np.all(labels.action_loss_mask)
    assert not np.any(labels.owner_automation)
