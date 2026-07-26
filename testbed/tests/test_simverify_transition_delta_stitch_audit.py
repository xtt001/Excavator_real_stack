from __future__ import annotations

from testbed.simverify.m3_transition_delta_stitch_audit import (
    development_prerequisite_criteria,
    nested_train_step_envelope_audit,
)


def _row(episode_id: int, steps: float) -> dict[str, float | int]:
    return {
        "episode_id": episode_id,
        "candidate_completion_rate": 1.0,
        "median_action_null_completion_rate": 0.0,
        "paired_completion_delta": 1.0,
        "candidate_completed_steps_q97_5": steps,
        "candidate_max_retrieval_distance": 0.4,
    }


def test_nested_train_audit_exposes_unstable_step_envelope() -> None:
    result = nested_train_step_envelope_audit(
        [_row(1, 100.0), _row(2, 101.0), _row(3, 150.0)]
    )
    assert result["failure_count"] == 1
    assert result["stable_as_zero_false_rejection_hard_gate"] is False


def test_development_gate_uses_budget_not_speed_similarity() -> None:
    criteria = development_prerequisite_criteria(
        [_row(1, 100.0), _row(2, 110.0)],
        [_row(3, 111.0)],
        support_radius=0.5,
        maximum_steps=200,
    )
    assert all(row["passed"] for row in criteria.values())
