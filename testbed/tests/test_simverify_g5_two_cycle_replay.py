from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from testbed.simverify import g5_two_cycle_replay as module
from testbed.simverify.g5_two_cycle_replay import (
    derive_expert_two_cycle_thresholds,
    evaluate_expert_two_cycle_gate,
    replay_two_cycle_arrays,
)


class _Policy:
    def __init__(self) -> None:
        self.reset_count = 0
        self.conditions: list[np.ndarray] = []
        self.condition_route_diagnostics = {
            "route_index": 2,
            "consecutive_pending": 0,
        }

    def reset(self) -> None:
        self.reset_count += 1

    def predict(self, observation: dict[str, object]) -> np.ndarray:
        condition = np.asarray(observation["cycle_condition_v1"], dtype=np.float32)
        self.conditions.append(condition.copy())
        return np.full(4, condition[4], dtype=np.float32)

    def last_raw_action_chunk(self) -> np.ndarray:
        return np.zeros((2, 4), dtype=np.float32)

    def last_raw_action_chunk_direct(self) -> np.ndarray:
        return np.ones((2, 4), dtype=np.float32)


def test_two_cycle_replay_resets_once_and_switches_at_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as episode:
        observations = episode.create_group("observations")
        observations.create_dataset("qpos", data=np.zeros((8, 4), np.float32))
        observations.create_dataset("qvel", data=np.zeros((8, 4), np.float32))
        episode.create_dataset("action", data=np.zeros((8, 4), np.float32))
    monkeypatch.setattr(
        module,
        "_read_camera_image",
        lambda _episode, _camera, _tick: np.zeros((4, 4, 3), np.uint8),
    )
    anchor = {
        "target_steps_20hz": [1, 6],
        "shared_ready_boundary_tick": 3,
        "first_condition": {
            "vector": [1, 0, 0, 1, 0, 0],
        },
        "second_condition": {
            "vector": [1, 0, 0, 0, 1, 0],
        },
    }
    policy = _Policy()
    with h5py.File(path, "r") as episode:
        arrays = replay_two_cycle_arrays(
            policy=policy,
            episode=episode,
            anchor=anchor,
            condition_mode="switched",
        )
    assert policy.reset_count == 1
    assert arrays["shared_ready_boundary_local_index"] == 2
    np.testing.assert_array_equal(arrays["condition"][1], [1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(arrays["condition"][2], [1, 0, 0, 0, 1, 0])
    assert arrays["temporal_aggregation_action"][1, 0] == 0.0
    assert arrays["temporal_aggregation_action"][2, 0] == 1.0


def test_expert_thresholds_are_generated_from_source_episode_rows() -> None:
    train = [
        {
            "phase_coverage_mean": 1.0,
            "event_order_valid_rate": 1.0,
            "ready_boundary_discontinuity_q95": 0.2,
        },
        {
            "phase_coverage_mean": 0.9,
            "event_order_valid_rate": 0.8,
            "ready_boundary_discontinuity_q95": 0.3,
        },
    ]
    thresholds = derive_expert_two_cycle_thresholds(train)
    gate = evaluate_expert_two_cycle_gate(
        [
            {
                "phase_coverage_mean": 0.95,
                "event_order_valid_rate": 0.9,
                "ready_boundary_discontinuity_q95": 0.25,
            }
        ],
        thresholds,
    )
    assert thresholds["two_cycle_phase_coverage_lower"] < 1.0
    assert gate["passed"] is True
