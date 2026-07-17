from __future__ import annotations

from typing import Any

import numpy as np

from testbed.policies.startup_activation import (
    aggregate_startup_activation_rows,
    capability_boundaries,
    evaluate_startup_activation,
)

THRESHOLDS = {
    axis: {"pos": 0.5, "neg": 0.5} for axis in ("swing", "boom", "stick", "bucket")
}


def _event(*, onset: int = 3) -> dict[str, Any]:
    return {
        "schema_version": "single_demo_intent_events_v2",
        "event_id": f"episode_1000:event_0000:step_{onset}",
        "episode_id": 1000,
        "split": "validation",
        "event_index": 0,
        "onset_step": onset,
        "anchor_intent": ["stick+"],
        "single_demo_event_support_directions": ["stick+", "bucket+"],
    }


def _observations(count: int = 5) -> list[dict[str, np.ndarray]]:
    return [
        {
            "qpos": np.full(4, step, dtype=np.float32),
            "qvel": np.full(4, step + 0.25, dtype=np.float32),
            "previous_final_command": np.full(4, 0.9, dtype=np.float32),
            "image_video4": np.full((2, 2, 3), step, dtype=np.uint8),
        }
        for step in range(count)
    ]


class SequenceSource:
    def __init__(self, actions: list[list[float]]) -> None:
        self.actions = [np.asarray(action, dtype=np.float32) for action in actions]
        self.seen: list[dict[str, Any]] = []
        self.index = 0
        self.reset_count = 0

    def reset(self) -> None:
        self.index = 0
        self.reset_count += 1

    def step(self, observation: dict[str, Any]) -> np.ndarray:
        self.seen.append(observation)
        action = self.actions[self.index]
        self.index += 1
        return action

    def snapshot_state(self) -> Any:
        return self.index

    def restore_state(self, state: Any) -> None:
        self.index = int(state)


def test_warmup_effective_outputs_are_ignored_but_advance_arm_state() -> None:
    source = SequenceSource(
        [
            [0.8, 0.0, 0.0, 0.0],  # effective during suppressed warmup
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],  # arm delay 0
            [0.0, 0.8, 0.0, 0.0],  # arm delay 1, any axis may start
        ]
    )

    row = evaluate_startup_activation(
        episode_id=1000,
        first_event=_event(onset=3),
        observations=_observations(),
        thresholds=THRESHOLDS,
        step_source=source,
        hold_horizon_steps=5,
        sampling_hz=20.0,
    )

    assert source.reset_count == 1
    assert len(source.seen) == 4
    assert row["warmup_ticks"] == 2
    assert row["warmup_effective_output_ticks"] == [0]
    assert row["warmup_any_effective_output"] is True
    assert row["natural_liveness"] is True
    assert row["activation_delay_ticks"] == 1
    assert row["first_direction_set"] == ["boom+"]
    assert row["within_1_ticks"] is False
    assert row["within_3_ticks"] is True


def test_post_arm_observation_freezes_qpos_and_zeros_feedback() -> None:
    source = SequenceSource(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.0],
        ]
    )

    row = evaluate_startup_activation(
        episode_id=1000,
        first_event=_event(onset=3),
        observations=_observations(),
        thresholds=THRESHOLDS,
        step_source=source,
        hold_horizon_steps=4,
        sampling_hz=20.0,
    )

    # Warmup qvel remains recorded, while suppressed previous commands are zero.
    np.testing.assert_array_equal(source.seen[0]["qvel"], np.full(4, 0.25))
    np.testing.assert_array_equal(source.seen[1]["qvel"], np.full(4, 1.25))
    for observation in source.seen:
        np.testing.assert_array_equal(
            observation["previous_final_command"], np.zeros(4)
        )
    # Every post-arm call repeats observation 2 with zero qvel.
    for observation in source.seen[2:]:
        np.testing.assert_array_equal(observation["qpos"], np.full(4, 2.0))
        np.testing.assert_array_equal(observation["qvel"], np.zeros(4))
    assert row["frozen_qpos"] == [2.0, 2.0, 2.0, 2.0]


def test_expert_mismatch_is_descriptive_and_does_not_fail_liveness() -> None:
    source = SequenceSource([[0.0, -0.8, 0.0, 0.0]])
    row = evaluate_startup_activation(
        episode_id=1000,
        first_event=_event(onset=0),
        observations=_observations(),
        thresholds=THRESHOLDS,
        step_source=source,
        hold_horizon_steps=3,
        sampling_hz=20.0,
    )

    assert row["natural_liveness"] is True
    assert row["status"] == "effective_action"
    assert row["activation_delay_ticks"] == 0
    assert row["first_direction_set"] == ["boom-"]
    assert row["single_demo_similarity"]["exact_anchor"] is False
    assert row["single_demo_similarity"]["overlap_anchor"] is False
    assert row["single_demo_similarity"]["outside_local_support_directions"] == [
        "boom-"
    ]
    assert row["startup_axis_requirement"] == "none"
    assert row["single_demo_similarity_only"] is True
    assert "expert_similarity" not in row
    assert "expert_match_only" not in row
    assert row["promotion_gate"] is False
    assert row["safety_gate"] is False


def test_horizon_without_effective_action_is_explicit() -> None:
    source = SequenceSource([[0.0, 0.0, 0.0, 0.0]] * 3)
    row = evaluate_startup_activation(
        episode_id=1000,
        first_event=_event(onset=0),
        observations=_observations(),
        thresholds=THRESHOLDS,
        step_source=source,
        hold_horizon_steps=3,
        sampling_hz=20.0,
    )

    assert row["arm_step"] == 0
    assert row["warmup_ticks"] == 0
    assert row["status"] == "horizon_no_effective_action"
    assert row["activation_delay_ticks"] is None
    assert row["first_action_vector"] is None
    assert all(row[f"within_{ticks}_ticks"] is False for ticks in (1, 3, 5, 10, 20))


def test_aggregate_keeps_liveness_independent_from_expert_similarity() -> None:
    live = evaluate_startup_activation(
        episode_id=1000,
        first_event=_event(onset=0),
        observations=_observations(),
        thresholds=THRESHOLDS,
        step_source=SequenceSource([[0.0, -0.8, 0.0, 0.0]]),
        hold_horizon_steps=1,
        sampling_hz=20.0,
    )
    dead = dict(live)
    dead.update(
        {
            "episode_id": 1001,
            "natural_liveness": False,
            "activation_delay_ticks": None,
            "first_direction_set": [],
            "warmup_any_effective_output": False,
            "single_demo_similarity": {
                "exact_anchor": False,
                "overlap_anchor": False,
                "wholly_within_local_support": False,
                "outside_local_support_directions": [],
                "opposite_to_anchor_directions": [],
            },
            **{f"within_{ticks}_ticks": False for ticks in (1, 3, 5, 10, 20)},
        }
    )

    aggregate = aggregate_startup_activation_rows([live, dead])

    assert aggregate["natural_liveness_count"] == 1
    assert aggregate["natural_liveness_rate"] == 0.5
    assert aggregate["within_1_ticks"]["count"] == 1
    assert aggregate["startup_axis_requirement"] == "none"
    assert aggregate["single_demo_similarity_only"] is True
    assert aggregate["promotion_gate"] is False
    assert aggregate["safety_gate"] is False
    assert "physical machine response" in capability_boundaries()["does_not_measure"]
