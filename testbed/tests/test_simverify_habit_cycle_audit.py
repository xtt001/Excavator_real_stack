from __future__ import annotations

import numpy as np

from testbed.simverify.annotations import EpisodeSignals
from testbed.simverify.habit_cycle_audit import (
    _full_cycle_source_range,
    _true_runs,
    build_transition_candidates,
    definition_decision,
    enumerate_causal_candidates,
    fit_causal_confirmation_dwell,
)


def _signals() -> dict[int, EpisodeSignals]:
    qpos = np.zeros((20, 4), dtype=np.float32)
    qpos[:, 0] = 0.60
    qpos[3:7, 0] = 0.50
    qpos[8:13, 0] = 0.50
    action = np.zeros((20, 4), dtype=np.float32)
    action[3:5, 0] = -1.0
    episode = EpisodeSignals(
        episode_id=3,
        step_id=np.arange(20, dtype=np.int64),
        qpos=qpos,
        qvel=np.zeros((20, 4), dtype=np.float32),
        action=action,
        dt=0.02,
    )
    episode.validate()
    return {3: episode}


def _candidate() -> dict[str, object]:
    return {
        "episode_id": 3,
        "cycle_id": 0,
        "split": "train",
        "relative_intent": "stay",
        "hindsight_expert_target_sector": "left",
        "dump_end_step": 1,
        "next_dig_entry_step": 12,
        "dig_ready_reference_interval": [8, 12],
        "numeric_causal_candidate_steps": [],
        "causal_confirm_step": None,
        "causal_confirmed": False,
        "causal_confirm_matches_reference": False,
        "reason_codes": [],
    }


def _sector_thresholds() -> dict[str, object]:
    return {
        "boundaries_low_to_high": [0.52, 0.57],
        "cluster_centers_low_to_high": [0.49, 0.545, 0.595],
        "labels_low_to_high": ["left", "center", "right"],
        "boundary_review_margin": 0.0,
    }


def test_true_runs_are_half_open_and_forward_only() -> None:
    mask = np.asarray([False, True, True, False, True, True, True, False])
    assert _true_runs(mask, start=2, end=7) == [(2, 3), (4, 7)]


def test_numeric_candidate_stage_preserves_ready_recall_for_visual_gate() -> None:
    signals = _signals()
    contract = fit_causal_confirmation_dwell(
        [_candidate()],
        signals=signals,
        sector_thresholds=_sector_thresholds(),
        dump_swing_threshold=0.63,
        swing_speed_threshold=0.05,
    )
    assert contract["selected_dwell_steps"] == 1
    rows = enumerate_causal_candidates(
        [_candidate()],
        dwell_steps=int(contract["selected_dwell_steps"]),
        signals=signals,
        sector_thresholds=_sector_thresholds(),
        dump_swing_threshold=0.63,
        swing_speed_threshold=0.05,
    )
    assert rows[0]["numeric_causal_candidate_steps"] == [3, 8]
    assert rows[0]["causal_confirmed"] is False


def test_ready_reference_uses_low_speed_envelope_not_action_deadzone() -> None:
    cycles = {
        3: [
            {
                "cycle_id": 0,
                "numeric_sector_evidence": {
                    "current_swing_qpos": 0.50,
                    "next_swing_qpos": 0.50,
                },
                "observable_events": {
                    "dump_end_proxy": {"representative_step": 1},
                },
                "sector_observations": {
                    "next": {"representative_step": 2},
                },
            },
            {
                "cycle_id": 1,
                "numeric_sector_evidence": {
                    "current_swing_qpos": 0.50,
                    "next_swing_qpos": 0.50,
                },
                "observable_events": {
                    "dump_start_proxy": {"representative_step": 15},
                    "dump_end_proxy": {"representative_step": 18},
                },
                "sector_observations": {
                    "next": {"representative_step": 19},
                },
            },
        ]
    }
    rows = build_transition_candidates(
        cycles,
        signals=_signals(),
        metadata={3: {"controller_epoch": "epoch_a"}},
        split={"splits": {"train": [3]}},
        sector_thresholds=_sector_thresholds(),
        dump_swing_threshold=0.63,
        swing_speed_threshold=0.05,
        ready_envelope_steps=3,
    )
    assert rows[0]["dig_ready_reference_interval"] == [3, 6]
    assert rows[0]["next_dig_entry_step"] == 6
    assert rows[0]["outcome"]["source"] == "observable_ready_capture"


def test_definition_decision_prioritizes_boundary_failure() -> None:
    decision = definition_decision(
        transition_inventory={
            "splits": {
                "train": {
                    "source_episode_support": {
                        "stay": 3,
                        "step_left": 2,
                        "step_right": 2,
                    },
                    "habit_stability": {
                        "stay": {"controller_epochs": ["a", "b"]},
                        "adjacent": {"controller_epochs": ["a", "b"]},
                    },
                },
                "validation": {
                    "source_episode_support": {
                        "stay": 1,
                        "step_left": 1,
                        "step_right": 1,
                    }
                },
            }
        },
        boundary_audit={
            "causal_confirmation_passed": False,
            "visual_boundary_passed": True,
        },
        condition_support={
            "counts": {
                "validation_entry_with_supported_alternative": 1,
            }
        },
    )
    assert decision["decision"] == "revise_boundary"
    assert decision["training_authorized"] is False


def test_full_cycle_range_uses_previous_shared_ready_boundary() -> None:
    previous = {
        "episode_id": 3,
        "cycle_id": 4,
        "causal_confirm_step": 100,
        "causal_confirm_matches_reference": True,
        "hindsight_expert_target_sector": "center",
    }
    current = {
        "episode_id": 3,
        "cycle_id": 5,
        "current_sector": "center",
        "causal_confirm_step": 350,
    }
    result = _full_cycle_source_range(
        current,
        {(3, 4): previous},
    )
    assert result == [100, 350]
    assert current["cycle_ready_start_step"] == 100
    assert current["cycle_ready_end_step"] == 350
