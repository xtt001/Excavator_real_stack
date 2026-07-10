from __future__ import annotations

import numpy as np
import pytest

from testbed.data.deadzone_intent_labels import (
    compute_deadzone_intent_labels,
    masked_action_stats,
)


def _thresholds() -> dict[str, dict[str, float]]:
    return {
        "swing": {"pos": 0.5, "neg": 0.5},
        "boom": {"pos": 0.4, "neg": 0.4},
        "stick": {"pos": 0.3, "neg": 0.3},
        "bucket": {"pos": 0.2, "neg": 0.2},
    }


def test_deadzone_intent_labels_mark_move_stop_and_wrong_directions() -> None:
    actions = np.asarray(
        [
            [0.60, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, -0.50, 0.00, 0.00],
        ],
        dtype=np.float32,
    )

    labels = compute_deadzone_intent_labels(actions=actions, thresholds=_thresholds())

    assert labels.move_mask.shape == (3, 4, 2)
    assert labels.stop_mask.shape == (3,)
    assert labels.wrong_mask.shape == (3, 4, 2)
    assert labels.action_loss_mask.tolist() == [True, True, True]

    assert labels.move_mask[0, 0, 0]
    assert not labels.move_mask[0, 0, 1]
    assert labels.move_mask[2, 1, 1]
    assert labels.stop_mask.tolist() == [False, True, False]

    assert not labels.wrong_mask[0, 0, 0]
    assert labels.wrong_mask[0, 0, 1]
    assert labels.wrong_mask[0, 1, 0]
    assert labels.wrong_mask[1].all()


def test_deadzone_intent_labels_force_tail_and_automation_to_stop_intent() -> None:
    actions = np.asarray(
        [
            [0.60, 0.00, 0.00, 0.00],
            [0.60, 0.00, 0.00, 0.00],
            [0.00, -0.50, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    action_loss_mask = np.asarray([True, False, False], dtype=bool)
    tail_idle_mask = np.asarray([False, True, False], dtype=bool)
    owner_automation = np.asarray([False, False, True], dtype=bool)

    labels = compute_deadzone_intent_labels(
        actions=actions,
        thresholds=_thresholds(),
        action_loss_mask=action_loss_mask,
        tail_idle_mask=tail_idle_mask,
        owner_automation=owner_automation,
    )

    assert labels.action_loss_mask.tolist() == [True, False, False]
    assert labels.stop_mask.tolist() == [False, True, True]
    assert labels.move_mask[0, 0, 0]
    assert not labels.move_mask[1].any()
    assert not labels.move_mask[2].any()
    assert labels.wrong_mask[1].all()
    assert labels.wrong_mask[2].all()


def test_masked_action_stats_ignore_automation_tail_actions() -> None:
    actions = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [-9.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mask = np.asarray([True, True, False], dtype=bool)

    mean, std = masked_action_stats(actions, mask)

    assert mean.shape == (4,)
    assert std.shape == (4,)
    assert mean[0] == pytest.approx(1.0)
    assert std[0] == pytest.approx(0.01)
