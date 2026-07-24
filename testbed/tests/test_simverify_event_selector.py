from __future__ import annotations

import json

import numpy as np
import pytest

from testbed.simverify.annotations import EpisodeSignals
from testbed.simverify.event_selector import (
    EVENT_PHASES,
    bootstrap_event_selected_sector,
    event_rows,
    match_event_interval,
    public_selector,
    select_event_corpus,
    selected_sector_records,
)


def _event(step: int) -> dict[str, object]:
    return {
        "interval": [step, step + 1],
        "representative_step": step,
    }


def _two_cycles() -> dict[int, list[dict[str, object]]]:
    shared = _event(20)
    return {
        7: [
            {
                "cycle_id": 0,
                "observable_events": {
                    "ready_start": _event(10),
                    "dig_entry_proxy": _event(12),
                    "carry_transition_proxy": _event(14),
                    "dump_start_proxy": _event(16),
                    "dump_end_proxy": _event(18),
                    "ready_end": shared,
                },
            },
            {
                "cycle_id": 1,
                "observable_events": {
                    "ready_start": shared,
                    "dig_entry_proxy": _event(22),
                    "carry_transition_proxy": _event(24),
                    "dump_start_proxy": _event(26),
                    "dump_end_proxy": _event(28),
                    "ready_end": _event(30),
                },
            },
        ]
    }


def _permissive_selector() -> dict[str, object]:
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    return {
        "prototypes": {
            phase: {"eye": vector, "stick": vector} for phase in EVENT_PHASES
        },
        "support_thresholds": {
            phase: {"eye": -1.0, "stick": -1.0} for phase in EVENT_PHASES
        },
        "change_thresholds": {
            "ready": {"stick": 2.0},
            "dig_entry_proxy": {"eye": 0.0, "stick": 0.0},
            "carry_transition_proxy": {"stick": 0.0},
            "dump_start_proxy": {"eye": 0.0, "stick": 0.0},
            "dump_end_proxy": {"eye": 0.0, "stick": 0.0},
        },
        "offset_bounds": {
            phase: {
                "minimum_signed_offset_steps": 0,
                "maximum_signed_offset_steps": 0,
            }
            for phase in EVENT_PHASES
        },
    }


def _constant_features(
    *,
    start: int,
    end: int,
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    return {(7, step): {"eye": vector, "stick": vector} for step in range(start, end)}


def test_shared_ready_boundary_is_one_selection_and_one_cycle_boundary() -> None:
    cycles = _two_cycles()
    rows = event_rows(cycles, episode_ids=[7])

    assert len(rows) == 11
    shared_rows = [
        row
        for row in rows
        if row["phase"] == "ready" and row["numeric_representative_step"] == 20
    ]
    assert len(shared_rows) == 1
    assert sorted(
        reference["event_name"] for reference in shared_rows[0]["references"]
    ) == ["ready_end", "ready_start"]

    result = select_event_corpus(
        _permissive_selector(),
        cycles,
        _constant_features(start=9, end=32),
        episode_ids=[7],
    )

    first = result["cycles"]["episode_7:cycle_0"]
    second = result["cycles"]["episode_7:cycle_1"]
    assert first["event_keys"]["ready_end"] == second["event_keys"]["ready_start"]
    assert first["event_order_valid"] is True
    assert second["event_order_valid"] is True
    assert first["confirmed_source_steps"] == [10, 20]
    assert second["confirmed_source_steps"] == [20, 30]


def test_offset_filter_is_applied_before_change_ranking() -> None:
    selector = _permissive_selector()
    selector["offset_bounds"]["carry_transition_proxy"] = {
        "minimum_signed_offset_steps": 0,
        "maximum_signed_offset_steps": 1,
    }
    features = {
        (7, 9): {
            "eye": np.asarray([1.0, 0.0]),
            "stick": np.asarray([1.0, 0.0]),
        },
        (7, 10): {
            "eye": np.asarray([1.0, 0.0]),
            "stick": np.asarray([1.0, 0.0]),
        },
        (7, 11): {
            "eye": np.asarray([1.0, 0.0]),
            "stick": np.asarray([0.9, 0.4358899]),
        },
        (7, 12): {
            "eye": np.asarray([1.0, 0.0]),
            "stick": np.asarray([0.0, 1.0]),
        },
        (7, 13): {
            "eye": np.asarray([1.0, 0.0]),
            "stick": np.asarray([-1.0, 0.0]),
        },
        (7, 14): {
            "eye": np.asarray([1.0, 0.0]),
            "stick": np.asarray([1.0, 0.0]),
        },
    }
    row = {
        "event_key": "episode_7:cycle_0:carry_transition_proxy",
        "episode_id": 7,
        "phase": "carry_transition_proxy",
        "interval": [10, 14],
        "numeric_representative_step": 10,
    }

    selection = match_event_interval(
        row,
        features,
        selector=selector,
    )

    assert selection["status"] == "confirmed"
    assert selection["representative_step"] == 11
    assert selection["absolute_offset_steps"] == 1
    assert selection["candidate_count"] == 4
    assert selection["eligible_candidate_count"] == 2
    assert selection["selected"]["signed_offset_steps"] <= 1


def test_selected_dig_after_carry_invalidates_cycle_order() -> None:
    cycles = _two_cycles()
    selector = _permissive_selector()
    selector["offset_bounds"]["dig_entry_proxy"] = {
        "minimum_signed_offset_steps": 5,
        "maximum_signed_offset_steps": 5,
    }
    cycles[7][0]["observable_events"]["dig_entry_proxy"]["interval"] = [12, 18]

    result = select_event_corpus(
        selector,
        cycles,
        _constant_features(start=9, end=32),
        episode_ids=[7],
    )

    first = result["cycles"]["episode_7:cycle_0"]
    assert first["event_steps"]["dig_entry_proxy"] == 17
    assert first["event_steps"]["carry_transition_proxy"] == 14
    assert first["event_order_valid"] is False
    assert (
        "observable_event_order_invalid_after_visual_selection" in first["reason_codes"]
    )
    assert first["current_sector_order_valid"] is False


def test_missing_unrelated_ready_end_preserves_current_sector_row() -> None:
    cycles = _two_cycles()
    cycles[7][1]["observable_events"]["ready_end"] = None
    result = select_event_corpus(
        _permissive_selector(),
        cycles,
        _constant_features(start=9, end=31),
        episode_ids=[7],
    )
    second = result["cycles"]["episode_7:cycle_1"]
    qpos = np.zeros((32, 4), dtype=np.float32)
    qpos[22, 0] = 0.6
    signal = EpisodeSignals(
        episode_id=7,
        step_id=np.arange(32, dtype=np.int64),
        qpos=qpos,
        qvel=np.zeros((32, 4), dtype=np.float32),
        action=np.zeros((32, 4), dtype=np.float32),
        dt=0.02,
    )

    rows = selected_sector_records(
        result,
        cycles,
        {7: signal},
        episode_draw=[7],
    )

    assert second["event_order_valid"] is False
    assert second["current_sector_order_valid"] is True
    assert rows[1]["sector_validity"]["current"]["valid"] is True
    assert rows[1]["numeric_sector_evidence"]["current_swing_qpos"] == pytest.approx(
        0.6
    )


def test_outer_bootstrap_rejects_non_calibration_episode_scope() -> None:
    signal = EpisodeSignals(
        episode_id=1,
        step_id=np.arange(4, dtype=np.int64),
        qpos=np.zeros((4, 4), dtype=np.float32),
        qvel=np.zeros((4, 4), dtype=np.float32),
        action=np.zeros((4, 4), dtype=np.float32),
        dt=0.02,
    )

    with pytest.raises(ValueError, match="calibration episodes only"):
        bootstrap_event_selected_sector(
            {1: []},
            {1: signal},
            {},
            train_ids=[1],
            validation_ids=[2],
            point_selector={},
            point_selections={},
            samples=1,
            seed=7,
        )


def test_public_selector_replaces_numpy_prototypes_with_npz_references() -> None:
    payload = public_selector(_permissive_selector())

    assert payload["prototypes"]["ready"]["stick"] == {
        "npz_key": "event_stick_ready",
        "dimension": 2,
    }
    json.dumps(payload, allow_nan=False)
