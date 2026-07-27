from __future__ import annotations

from testbed.simverify.habit_gate import derive_habit_gate


def _cycle_row(baseline: str, derived_id: int, source_id: int) -> dict:
    coverage = {"B0": 0.7, "B1": 0.9, "B2": 0.7}[baseline]
    recall = {"B0": 0.5, "B1": 0.8, "B2": 0.4}[baseline]
    mae = {"B0": 0.3, "B1": 0.1, "B2": 0.4}[baseline]
    return {
        "baseline_id": baseline,
        "derived_episode_id": derived_id,
        "source_episode_id": source_id,
        "metrics": {"post_commit": {"overall": {"mae": mae}}},
        "action_grammar": {
            "required_event_coverage": coverage,
            "deadzone_effective_recall": recall,
        },
        "zero_action_grammar": {"required_event_coverage": 0.3},
    }


def _swap_row(baseline: str, derived_id: int, source_id: int) -> dict:
    return {
        "baseline_id": baseline,
        "derived_episode_id": derived_id,
        "source_episode_id": source_id,
        "status": "supported_fixed_observation_intervention",
        "metrics": {
            "semantic_direction_correct": baseline == "B1",
            "pre_dump_effect_l1": 0.0,
        },
    }


def test_derive_habit_gate_uses_matched_nulls_and_source_bootstrap() -> None:
    cycles = [
        _cycle_row(baseline, derived_id, source_id)
        for derived_id, source_id in ((0, 12), (1, 20), (2, 34))
        for baseline in ("B0", "B1", "B2")
    ]
    swaps = [
        _swap_row(baseline, derived_id, source_id)
        for derived_id, source_id in ((0, 12), (1, 20), (2, 34))
        for baseline in ("B1", "B2")
    ]

    thresholds, decision = derive_habit_gate(
        cycles,
        swaps,
        bootstrap_repetitions=10_000,
        bootstrap_seed=7,
    )

    assert decision["decision"] == "offline_condition_evidence_accepted"
    assert decision["basic_capability_established_offline"] is True
    assert decision["condition_understanding_established_offline"] is True
    assert thresholds["frozen_before_held_out_test"] is True
    assert all(
        criterion["passed"] for criterion in thresholds["criteria"].values()
    )
