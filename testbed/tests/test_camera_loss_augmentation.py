from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from testbed.data.camera_loss_augmentation import (
    apply_camera_loss,
    camera_loss_manifest,
    camera_loss_selected,
    resolve_camera_loss_augmentation,
)

CAMERAS = ["video4", "video5", "video6", "video7"]


def _resolved(*, probability: float = 0.25) -> dict:
    return resolve_camera_loss_augmentation(
        {
            "enabled": True,
            "scope": "train_only",
            "target_camera": "video7",
            "probability": probability,
            "seed": 20260727,
            "mask_rgb": [0, 0, 0],
            "decision_key": [
                "seed",
                "source_episode_id",
                "source_tick",
            ],
        },
        camera_names=CAMERAS,
    )


def test_resolver_freezes_train_only_video7_contract() -> None:
    resolved = _resolved()

    assert resolved == {
        "schema": "camera_loss_augmentation_v1",
        "enabled": True,
        "scope": "train_only",
        "target_camera": "video7",
        "probability": 0.25,
        "seed": 20260727,
        "mask_rgb": [0, 0, 0],
        "decision_key": [
            "seed",
            "source_episode_id",
            "source_tick",
        ],
        "selection": "sha256_uniform_u64_less_than_probability",
    }


@pytest.mark.parametrize(
    "patch",
    [
        {"scope": "validation"},
        {"target_camera": "missing"},
        {"probability": 0.0},
        {"probability": 1.0},
        {"mask_rgb": [0, 0]},
        {"decision_key": ["seed", "source_tick"]},
    ],
)
def test_resolver_rejects_contract_drift(patch: dict) -> None:
    config = {
        "enabled": True,
        "scope": "train_only",
        "target_camera": "video7",
        "probability": 0.25,
        "seed": 20260727,
        "mask_rgb": [0, 0, 0],
        "decision_key": [
            "seed",
            "source_episode_id",
            "source_tick",
        ],
    }
    config.update(patch)

    with pytest.raises((KeyError, TypeError, ValueError)):
        resolve_camera_loss_augmentation(config, camera_names=CAMERAS)


def test_selection_is_stable_and_bound_to_episode_and_tick() -> None:
    resolved = _resolved()
    decisions = [
        camera_loss_selected(resolved, episode_id=3, source_tick=tick)
        for tick in range(100)
    ]

    assert decisions == [
        camera_loss_selected(resolved, episode_id=3, source_tick=tick)
        for tick in range(100)
    ]
    assert 10 < sum(decisions) < 40
    assert decisions != [
        camera_loss_selected(resolved, episode_id=4, source_tick=tick)
        for tick in range(100)
    ]


def test_apply_masks_only_video7_without_mutating_input() -> None:
    resolved = _resolved(probability=0.999999999)
    images = {
        camera: np.full((2, 3, 3), index + 1, dtype=np.uint8)
        for index, camera in enumerate(CAMERAS)
    }
    original = {camera: image.copy() for camera, image in images.items()}

    augmented = apply_camera_loss(
        images,
        resolved,
        episode_id=3,
        source_tick=0,
    )

    for camera in CAMERAS[:-1]:
        np.testing.assert_array_equal(augmented[camera], original[camera])
    np.testing.assert_array_equal(
        augmented["video7"],
        np.zeros_like(original["video7"]),
    )
    for camera in CAMERAS:
        np.testing.assert_array_equal(images[camera], original[camera])
        assert augmented[camera] is not images[camera]


def test_manifest_binds_sorted_unique_selected_keys() -> None:
    resolved = _resolved()
    sample_keys = [(4, 2), (3, 1), (3, 1), (3, 0), (4, 1)]
    selected = [
        key
        for key in sorted(set(sample_keys))
        if camera_loss_selected(
            resolved,
            episode_id=key[0],
            source_tick=key[1],
        )
    ]
    expected_sha = hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode()
    ).hexdigest()

    manifest = camera_loss_manifest(
        resolved,
        sample_keys=sample_keys,
        source_episode_ids=[4, 3, 4],
    )

    assert manifest["eligible_row_count"] == 4
    assert manifest["selected_row_count"] == len(selected)
    assert manifest["selected_keys_sha256"] == expected_sha
    assert manifest["source_episode_ids"] == [3, 4]
    assert manifest["held_out_test_read"] is False


def test_disabled_manifest_cannot_select_validation_rows() -> None:
    disabled = resolve_camera_loss_augmentation(None, camera_names=CAMERAS)

    manifest = camera_loss_manifest(
        disabled,
        sample_keys=[(100, 0)],
        source_episode_ids=[100],
    )

    assert manifest["enabled"] is False
    assert manifest["eligible_row_count"] == 1
    assert manifest["selected_row_count"] == 0
    assert manifest["selected_fraction"] == 0.0
