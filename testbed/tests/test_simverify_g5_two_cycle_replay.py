from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from testbed.simverify import g5_two_cycle_replay as module
from testbed.simverify.g5_two_cycle_replay import (
    apply_camera_variant,
    build_two_cycle_condition_support,
    derive_expert_two_cycle_thresholds,
    evaluate_expert_two_cycle_gate,
    replay_two_cycle_arrays,
)


class _Policy:
    def __init__(self) -> None:
        self.reset_count = 0
        self.condition_cycle_reset_count = 0
        self.conditions: list[np.ndarray] = []
        self.condition_route_diagnostics = {
            "route_index": 2,
            "consecutive_pending": 0,
        }

    def reset(self) -> None:
        self.reset_count += 1

    def reset_condition_cycle(self) -> None:
        self.condition_cycle_reset_count += 1

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
            reset_condition_cycle_at_boundary=True,
        )
    assert policy.reset_count == 1
    assert policy.condition_cycle_reset_count == 1
    assert arrays["shared_ready_boundary_local_index"] == 2
    np.testing.assert_array_equal(arrays["condition"][1], [1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(arrays["condition"][2], [1, 0, 0, 0, 1, 0])
    assert arrays["temporal_aggregation_action"][1, 0] == 0.0
    assert arrays["temporal_aggregation_action"][2, 0] == 1.0
    assert arrays["condition_cycle_router_reset_count"] == 1


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


def test_two_cycle_support_uses_second_cycle_counterfactual() -> None:
    anchors = [
        {
            "split": "train",
            "episode_id": 3,
            "first_cycle_id": 1,
            "second_cycle_id": 2,
            "first_condition": {"next_ready_sector": "left"},
            "second_condition": {"next_ready_sector": "right"},
        },
        {
            "split": "validation",
            "episode_id": 12,
            "first_cycle_id": 4,
            "second_cycle_id": 5,
            "first_condition": {"next_ready_sector": "center"},
            "second_condition": {"next_ready_sector": "left"},
        },
    ]
    counterfactual = [
        {
            "split": "train",
            "episode_id": 3,
            "cycle_id": 2,
            "changed_factors": ["next_sector"],
            "target_condition": {"next_sector": "left"},
            "supported": True,
            "status": "supported_counterfactual",
        },
        {
            "split": "validation",
            "episode_id": 12,
            "cycle_id": 5,
            "changed_factors": ["next_sector"],
            "target_condition": {"next_sector": "center"},
            "supported": False,
            "status": "unsupported_counterfactual",
        },
    ]
    support = build_two_cycle_condition_support(anchors, counterfactual)
    assert support["train_source_episode_minimum"] == 1
    assert support["train_supported_changed_pair_count"] == 1
    assert support["validation_supported_changed_pair_count"] == 0


def test_camera_variants_mask_and_swap_only_declared_roles() -> None:
    images = {
        f"video{index}": np.full((2, 3, 3), index, dtype=np.uint8)
        for index in range(4, 8)
    }

    eye_only = apply_camera_variant(images, "eye_only")
    np.testing.assert_array_equal(eye_only["video4"], images["video4"])
    np.testing.assert_array_equal(eye_only["video5"], images["video5"])
    assert not np.any(eye_only["video6"])
    assert not np.any(eye_only["video7"])

    cross = apply_camera_variant(images, "swap_cross_role_pairs")
    np.testing.assert_array_equal(cross["video4"], images["video6"])
    np.testing.assert_array_equal(cross["video6"], images["video4"])
    np.testing.assert_array_equal(cross["video5"], images["video7"])
    np.testing.assert_array_equal(cross["video7"], images["video5"])

    for camera in images:
        np.testing.assert_array_equal(
            images[camera], np.full_like(images[camera], int(camera[-1]))
        )


def test_camera_variant_rejects_missing_role() -> None:
    with pytest.raises(ValueError, match="exactly video4"):
        apply_camera_variant(
            {"video4": np.zeros((1, 1, 3), dtype=np.uint8)},
            "four_camera",
        )
