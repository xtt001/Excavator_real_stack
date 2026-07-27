from __future__ import annotations

import numpy as np

from testbed.simverify.habit_runtime_ready import (
    ObservableHabitReadyBoundaryDetector,
)


class _FeatureExtractor:
    provenance = {
        "schema": "test_feature_extractor",
        "privilege_used": False,
    }

    def extract_rgb_batch(self, images: list[np.ndarray]) -> np.ndarray:
        assert len(images) == 4
        features = np.zeros((4, 512), dtype=np.float32)
        for index in range(4):
            features[index, index] = 1.0
        return features


def _unit(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def _detector() -> ObservableHabitReadyBoundaryDetector:
    features = _FeatureExtractor().extract_rgb_batch(
        [np.zeros((2, 2, 3), dtype=np.uint8)] * 4
    )
    eye = _unit(np.concatenate((features[0], features[1])))
    stick = _unit(np.concatenate((features[2], features[3])))
    ready = _unit(np.concatenate((eye, stick)))
    sector_center = np.roll(eye, 8)
    sector_right = np.roll(eye, 16)
    return ObservableHabitReadyBoundaryDetector(
        feature_extractor=_FeatureExtractor(),
        ready_centroids={"ready": ready, "not_ready": -ready},
        sector_centroids={
            "left": eye,
            "center": sector_center,
            "right": sector_right,
        },
        swing_speed_threshold=0.14,
        dwell_policy_ticks=2,
        dump_swing_threshold=0.63,
        sector_thresholds={
            "boundaries_low_to_high": [0.52, 0.57],
            "cluster_centers_low_to_high": [0.49, 0.54, 0.59],
            "labels_low_to_high": ["left", "center", "right"],
            "boundary_review_margin": 0.005,
        },
        artifact_provenance={"manifest_sha256": "test"},
    )


def _observation(*, swing_qpos: float, swing_qvel: float) -> dict:
    return {
        "qpos": np.asarray([swing_qpos, 0, 0, 0], dtype=np.float32),
        "qvel": np.asarray([swing_qvel, 0, 0, 0], dtype=np.float32),
    }


def _policy_observation() -> dict:
    result = {
        f"image_video{index}": np.zeros((216, 384, 3), dtype=np.uint8)
        for index in range(4, 8)
    }
    result["cycle_condition_v1"] = np.asarray(
        [1, 0, 0, 1, 0, 0],
        dtype=np.float32,
    )
    return result


def test_v11_detector_requires_return_activation_then_low_speed_dwell() -> None:
    detector = _detector()
    route = {"route": "next"}
    low_before_arm = detector.observe(
        policy_tick=0,
        observation=_observation(swing_qpos=0.49, swing_qvel=0.02),
        policy_observation=_policy_observation(),
        held_action=np.zeros(4),
        condition_route=route,
    )
    armed = detector.observe(
        policy_tick=1,
        observation=_observation(swing_qpos=0.60, swing_qvel=-0.3),
        policy_observation=_policy_observation(),
        held_action=np.zeros(4),
        condition_route=route,
    )
    dwell_one = detector.observe(
        policy_tick=2,
        observation=_observation(swing_qpos=0.49, swing_qvel=-0.10),
        policy_observation=_policy_observation(),
        held_action=np.zeros(4),
        condition_route=route,
    )
    confirmed = detector.observe(
        policy_tick=3,
        observation=_observation(swing_qpos=0.49, swing_qvel=-0.08),
        policy_observation=_policy_observation(),
        held_action=np.zeros(4),
        condition_route=route,
    )

    assert low_before_arm["state"] == "searching_return_activation"
    assert low_before_arm["eligible"] is False
    assert armed["state"] == "armed"
    assert dwell_one["eligible_ticks"] == 1
    assert dwell_one["candidate"] is False
    assert confirmed["candidate"] is True
    assert confirmed["ready_visual_prediction"] == "ready"
    assert confirmed["visual_sector_prediction"] == "left"
    assert confirmed["confirmed"] is True


def test_v11_detector_fails_closed_without_committed_next_route() -> None:
    detector = _detector()
    result = detector.observe(
        policy_tick=0,
        observation=_observation(swing_qpos=0.60, swing_qvel=-0.3),
        policy_observation=_policy_observation(),
        held_action=np.zeros(4),
        condition_route={"route": "current"},
    )
    assert result["state"] == "searching_return_activation"
    assert result["return_active"] is False
    assert result["confirmed"] is False
    assert detector.provenance["privilege_used"] is False
    assert detector.provenance["future_observations_used"] is False
