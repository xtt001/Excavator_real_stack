from __future__ import annotations

import numpy as np
import pytest

from testbed.policies.task_sequence_compatibility import (
    activation_motif,
    activation_sequence_similarity,
    evaluate_task_sequence_compatibility,
)


@pytest.fixture
def thresholds() -> dict[str, dict[str, float]]:
    return {
        "swing": {"pos": 0.5, "neg": 0.5},
        "boom": {"pos": 0.5, "neg": 0.5},
        "stick": {"pos": 0.5, "neg": 0.5},
        "bucket": {"pos": 0.5, "neg": 0.5},
    }


def _trajectory(*segments: tuple[int, tuple[float, float, float, float]]) -> np.ndarray:
    return np.concatenate(
        [
            np.tile(np.asarray(action, dtype=np.float32), (ticks, 1))
            for ticks, action in segments
        ],
        axis=0,
    )


def test_activation_motif_ignores_leading_idle_and_groups_simultaneous_onsets(
    thresholds: dict[str, dict[str, float]],
) -> None:
    immediate = _trajectory(
        (2, (0.0, -0.8, 0.8, 0.0)),
        (1, (0.0, 0.0, 0.0, 0.0)),
        (2, (0.7, 0.0, 0.0, 0.0)),
    )
    delayed = np.concatenate([np.zeros((19, 4), dtype=np.float32), immediate])

    assert activation_motif(immediate, thresholds=thresholds) == (
        ("boom-", "stick+"),
        ("swing+",),
    )
    assert activation_motif(delayed, thresholds=thresholds) == activation_motif(
        immediate, thresholds=thresholds
    )


def test_activation_motif_collapses_same_direction_deadzone_chatter(
    thresholds: dict[str, dict[str, float]],
) -> None:
    actions = _trajectory(
        (2, (0.0, 0.0, 0.8, 0.0)),
        (2, (0.0, 0.0, 0.0, 0.0)),
        (2, (0.0, 0.0, 0.8, 0.0)),
        (2, (0.0, 0.0, -0.8, 0.0)),
        (2, (0.0, 0.0, 0.0, 0.0)),
        (2, (0.0, 0.0, 0.8, 0.0)),
    )

    assert activation_motif(actions, thresholds=thresholds) == (
        ("stick+",),
        ("stick-",),
        ("stick+",),
    )


def test_sequence_similarity_distinguishes_order_and_missing_progression() -> None:
    reference = (("boom-",), ("stick+",), ("bucket+",), ("boom+",))

    assert activation_sequence_similarity(reference, reference) == 1.0
    assert activation_sequence_similarity(reference[:1], reference) == pytest.approx(
        0.25
    )
    assert activation_sequence_similarity(tuple(reversed(reference)), reference) < 0.5


def test_evaluator_uses_expert_cohort_and_penalizes_single_event_policy(
    thresholds: dict[str, dict[str, float]],
) -> None:
    sequence_a = _trajectory(
        (2, (0.0, -0.8, 0.0, 0.0)),
        (1, (0.0, 0.0, 0.0, 0.0)),
        (2, (0.0, 0.0, 0.8, 0.0)),
        (1, (0.0, 0.0, 0.0, 0.0)),
        (2, (0.0, 0.0, 0.0, 0.8)),
    )
    sequence_b = _trajectory(
        (2, (0.0, -0.8, 0.8, 0.0)),
        (1, (0.0, 0.0, 0.0, 0.0)),
        (2, (0.0, 0.0, 0.0, 0.8)),
    )
    delayed_validation = np.concatenate(
        [np.zeros((13, 4), dtype=np.float32), sequence_a]
    )
    collapsed_policy = np.tile(
        np.asarray((0.0, -0.8, 0.0, 0.0), dtype=np.float32), (21, 1)
    )

    report = evaluate_task_sequence_compatibility(
        model="collapsed",
        training_expert_actions={1: sequence_a, 2: sequence_b},
        validation_expert_actions={10: delayed_validation},
        policy_actions={10: collapsed_policy},
        thresholds=thresholds,
    )

    assert (
        report["validation_expert_calibration"]["nearest_train_similarity"]["median"]
        == 1.0
    )
    assert report["model_summary"]["nearest_train_similarity"][
        "median"
    ] == pytest.approx(1.0 / 3.0)
    assert report["model_summary"]["core_direction_coverage_rate"][
        "median"
    ] == pytest.approx(1.0 / 3.0)
    assert (
        report["cohort_comparison"][
            "model_episode_rate_at_or_above_validation_expert_q25"
        ]
        == 0.0
    )
    assert report["capability_boundaries"]["leading_idle_semantics"].startswith(
        "ignored"
    )


def test_evaluator_rejects_mismatched_validation_ids(
    thresholds: dict[str, dict[str, float]],
) -> None:
    actions = np.zeros((3, 4), dtype=np.float32)
    actions[0, 1] = -0.8

    with pytest.raises(ValueError, match="exactly match"):
        evaluate_task_sequence_compatibility(
            model="bad",
            training_expert_actions={1: actions},
            validation_expert_actions={10: actions},
            policy_actions={11: actions},
            thresholds=thresholds,
        )
