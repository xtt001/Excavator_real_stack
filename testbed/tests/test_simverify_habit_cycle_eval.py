from __future__ import annotations

import numpy as np
import pytest

from testbed.simverify.habit_cycle_eval import (
    condition_swap_metrics,
    delivered_condition_rows,
    sector_condition,
    split_action_metrics,
)


def test_delivered_condition_rows_preserves_causal_gate_and_changes_only_target() -> None:
    recorded = np.zeros((5, 6), dtype=np.float32)
    recorded[2:] = sector_condition("center", "right")
    mask = np.asarray([0, 0, 1, 1, 1], dtype=np.uint8)

    delivered = delivered_condition_rows(
        recorded,
        mask,
        target_override="left",
    )

    np.testing.assert_array_equal(delivered[:2], np.zeros((2, 6)))
    np.testing.assert_array_equal(
        delivered[2:],
        np.repeat(sector_condition("center", "left")[None], 3, axis=0),
    )


def test_delivered_condition_rows_rejects_rearming_or_pre_dump_leakage() -> None:
    recorded = np.zeros((5, 6), dtype=np.float32)
    recorded[2:] = sector_condition("center", "right")
    with pytest.raises(ValueError, match="single false-to-true"):
        delivered_condition_rows(
            recorded,
            np.asarray([0, 0, 1, 0, 1], dtype=np.uint8),
        )
    leaked = recorded.copy()
    leaked[1] = sector_condition("center", "right")
    with pytest.raises(ValueError, match="inactive zeros"):
        delivered_condition_rows(
            leaked,
            np.asarray([0, 0, 1, 1, 1], dtype=np.uint8),
        )


def test_split_action_metrics_keeps_pre_and_post_windows_separate() -> None:
    expert = np.zeros((4, 4), dtype=np.float32)
    policy = np.zeros_like(expert)
    policy[2:] = 2.0
    metrics = split_action_metrics(
        expert,
        policy,
        np.asarray([0, 0, 1, 1], dtype=np.uint8),
    )
    assert metrics["pre_dump"]["overall"]["mae"] == 0.0
    assert metrics["post_commit"]["overall"]["mae"] == 2.0
    assert metrics["full_cycle"]["overall"]["mae"] == 1.0


def test_condition_swap_metrics_reports_phase_localization() -> None:
    base = np.zeros((4, 4), dtype=np.float32)
    alternate = base.copy()
    alternate[2:, 0] = -0.4
    metrics = condition_swap_metrics(
        base,
        alternate,
        np.asarray([0, 0, 1, 1], dtype=np.uint8),
    )
    assert metrics["pre_dump_effect_l1"] == 0.0
    assert metrics["post_commit_effect_l1"] == pytest.approx(0.1)
    assert metrics["post_commit_swing_delta_mean"] == pytest.approx(-0.4)
    assert metrics["closed_loop_execution"] is False
