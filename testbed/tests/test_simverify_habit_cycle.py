from __future__ import annotations

import numpy as np
import pytest

from testbed.simverify.habit_cycle import (
    DIAGNOSTIC_NONADJACENT,
    HabitCycleLifecycle,
    cycle_action_valid_mask,
    relative_intent,
    resolve_target_sector,
)


@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        ("left", "left", "stay"),
        ("center", "left", "step_left"),
        ("center", "right", "step_right"),
        ("left", "right", DIAGNOSTIC_NONADJACENT),
        ("right", "left", DIAGNOSTIC_NONADJACENT),
    ],
)
def test_relative_intent_mapping(
    current: str,
    target: str,
    expected: str,
) -> None:
    assert relative_intent(current, target) == expected


def test_relative_intent_edges_fail_closed() -> None:
    with pytest.raises(ValueError, match="leaves 3x1 work area"):
        resolve_target_sector("left", "step_left")
    with pytest.raises(ValueError, match="leaves 3x1 work area"):
        resolve_target_sector("right", "step_right")
    with pytest.raises(ValueError, match="unsupported relative intent"):
        resolve_target_sector("center", DIAGNOSTIC_NONADJACENT)


def _observe_complete_dump(lifecycle: HabitCycleLifecycle) -> None:
    for event in (
        "leave_initial_ready",
        "dig_entry",
        "carry",
        "dump_start",
        "dump_end",
    ):
        lifecycle.observe_event(event)


def test_same_sector_cycle_requires_leave_dig_and_dump_before_rearming() -> None:
    lifecycle = HabitCycleLifecycle(cycle_id=4, current_sector="center")
    with pytest.raises(RuntimeError, match="only after dump_end"):
        lifecycle.commit_after_dump("stay")
    _observe_complete_dump(lifecycle)
    command = lifecycle.commit_after_dump("stay")
    assert command["scripted_target_sector"] == "center"
    result = lifecycle.confirm_target_ready("center")
    assert result["observable_cycle_completed"] is True
    assert result["physical_effect_validated"] is None


def test_missing_target_and_invalid_event_order_fail_closed() -> None:
    lifecycle = HabitCycleLifecycle(cycle_id=0, current_sector="left")
    with pytest.raises(ValueError, match="expected 'leave_initial_ready'"):
        lifecycle.observe_event("dig_entry")
    _observe_complete_dump(lifecycle)
    with pytest.raises(RuntimeError, match="missing committed target"):
        lifecycle.confirm_target_ready("center")


def test_wrong_realized_sector_is_not_observable_completion() -> None:
    lifecycle = HabitCycleLifecycle(cycle_id=1, current_sector="center")
    _observe_complete_dump(lifecycle)
    lifecycle.commit_after_dump("step_left")
    result = lifecycle.confirm_target_ready("center")
    assert result["scripted_target_sector"] == "left"
    assert result["realized_target_sector"] == "center"
    assert result["observable_cycle_completed"] is False


def test_normal_stop_does_not_invent_outcome() -> None:
    lifecycle = HabitCycleLifecycle(cycle_id=2, current_sector="right")
    result = lifecycle.stop_without_completion("fixed_scenario_complete")
    assert result["observable_cycle_completed"] is False
    assert result["realized_target_sector"] is None


def test_action_chunk_mask_stops_at_half_open_cycle_boundary() -> None:
    mask = cycle_action_valid_mask(
        observation_tick=8,
        cycle_end_tick=11,
        horizon=6,
    )
    np.testing.assert_array_equal(
        mask,
        np.asarray([True, True, True, False, False, False]),
    )

