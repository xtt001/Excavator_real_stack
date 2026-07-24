from __future__ import annotations

import cv2
import numpy as np

from testbed.data.dig_sector_annotation import (
    AnnotationConfig,
    classify_qpos_sector,
    classify_visual_sector,
    detect_entry_step,
    intersect_candidates,
    register_static_background,
)


def test_sector_boundary_contract() -> None:
    config = AnnotationConfig()

    assert classify_qpos_sector(0.0, config).label == "C"
    assert classify_qpos_sector(0.05, config).label == "L"
    assert classify_qpos_sector(-0.05, config).label == "R"
    qpos_boundary = classify_qpos_sector(0.04, config)
    assert qpos_boundary.label is None
    assert qpos_boundary.candidate_labels == ("C", "L")

    assert classify_visual_sector(0.0, config).label == "C"
    assert classify_visual_sector(-15.0, config).label == "L"
    assert classify_visual_sector(15.0, config).label == "R"
    visual_boundary = classify_visual_sector(-12.0, config)
    assert visual_boundary.label is None
    assert visual_boundary.candidate_labels == ("C", "L")


def test_entry_detector_requires_sustained_bucket_motion() -> None:
    config = AnnotationConfig()
    qpos = np.zeros((30, 4), dtype=np.float64)
    qpos[8, 3] = 0.11
    assert detect_entry_step(qpos, 0.0, config) is None

    qpos[15:, 3] = 0.12
    entry = detect_entry_step(qpos, 0.0, config)
    assert entry is not None
    assert 13 <= entry <= 17


def test_static_background_registration_recovers_translation() -> None:
    config = AnnotationConfig()
    rng = np.random.default_rng(42)
    reference = rng.integers(
        0,
        256,
        size=(config.image_height, config.image_width),
        dtype=np.uint8,
    )
    reference = cv2.GaussianBlur(reference, (3, 3), 0)
    target = cv2.warpAffine(
        reference,
        np.asarray([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0]]),
        (config.image_width, config.image_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    result = register_static_background(
        reference,
        target,
        "video4",
        0,
        20,
        config,
    )

    assert result.status == "ok"
    assert result.robust_match_count >= config.min_robust_matches
    assert result.horizontal_displacement_px is not None
    assert abs(result.horizontal_displacement_px - 20.0) < 0.5
    assert result.sector_label == "R"


def test_candidate_intersection_preserves_only_shared_sector() -> None:
    assert intersect_candidates((("C", "L"), ("L",), ("L",))) == ("L",)
    assert intersect_candidates((("C",), ("L",))) == ()
