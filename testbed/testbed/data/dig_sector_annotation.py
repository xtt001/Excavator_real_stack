"""Automatic 3x1 achieved-dig-sector annotation from real observations.

The contract deliberately separates three different facts:

* ``command`` is unknown for historical recordings unless it was recorded;
* ``entry_step`` is a bucket-motion timing proxy, not measured soil contact;
* ``actual_dig_sector`` is a hindsight L/C/R sector relative to the episode's
  initial swing pose, accepted only when qpos and both eye cameras agree.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

CONTRACT_ID = "dig_sector_annotation_v0_1"
ENTRY_METHOD = "sustained_bucket_qpos_rise_v0_1"
VISION_METHOD = "eye_static_background_sift_median_v0_1"
FUSION_METHOD = "swing_qpos_plus_eye_background_displacement_v0_1"
EYE_CAMERAS = ("video4", "video5")
ALL_CAMERAS = ("video4", "video5", "video6", "video7")
HOME_SCALE_FLOOR_RAD = np.asarray((0.02, 0.03, 0.03, 0.08), dtype=np.float64)


@dataclass(frozen=True)
class AnnotationConfig:
    """Frozen v0.1 thresholds for one fixed eye-camera mounting."""

    calibration_id: str = "real_eye_pair_static_mount_640x360_v0_1"
    initial_window_steps: int = 10
    entry_window_radius_steps: int = 10
    bucket_rise_rad: float = 0.10
    bucket_rise_hold_rad: float = 0.08
    bucket_rise_hold_steps: int = 4
    bucket_rise_lookahead_steps: int = 6
    home_outlier_score: float = 8.0
    qpos_center_max_abs_rad: float = 0.03
    qpos_side_min_abs_rad: float = 0.05
    image_width: int = 640
    image_height: int = 360
    static_mask_top_rows: int = 244
    static_mask_left_end_col: int = 249
    static_mask_right_start_col: int = 390
    sift_ratio: float = 0.75
    robust_mad_multiplier: float = 3.0
    robust_min_residual_px: float = 2.0
    min_good_matches: int = 12
    min_robust_matches: int = 8
    vision_center_max_abs_px: float = 10.0
    vision_side_min_abs_px: float = 15.0

    def validate(self) -> None:
        if self.initial_window_steps <= 0:
            raise ValueError("initial_window_steps must be positive")
        if self.entry_window_radius_steps < 0:
            raise ValueError("entry_window_radius_steps must be non-negative")
        if not 0.0 < self.qpos_center_max_abs_rad < self.qpos_side_min_abs_rad:
            raise ValueError("qpos thresholds must define a non-empty boundary band")
        if not 0.0 < self.vision_center_max_abs_px < self.vision_side_min_abs_px:
            raise ValueError("vision thresholds must define a non-empty boundary band")
        if not 0.0 < self.sift_ratio < 1.0:
            raise ValueError("sift_ratio must be within (0, 1)")
        if self.min_good_matches < 2 or self.min_robust_matches < 2:
            raise ValueError("match thresholds must be at least two")
        if not (
            0 < self.static_mask_top_rows <= self.image_height
            and 0 < self.static_mask_left_end_col < self.static_mask_right_start_col
            and self.static_mask_right_start_col < self.image_width
        ):
            raise ValueError("static background mask is outside the resized image")


@dataclass(frozen=True)
class EpisodeSignals:
    episode_name: str
    episode_id: int
    path: Path
    qpos: np.ndarray
    action: np.ndarray
    initial_qpos_rad: np.ndarray


@dataclass(frozen=True)
class SectorEvidence:
    label: str | None
    candidate_labels: tuple[str, ...]
    boundary: bool


@dataclass(frozen=True)
class CameraRegistration:
    camera: str
    status: str
    reference_step: int
    target_step: int
    good_match_count: int
    robust_match_count: int
    horizontal_displacement_px: float | None
    vertical_displacement_px: float | None
    horizontal_mad_px: float | None
    vertical_mad_px: float | None
    sector_label: str | None
    candidate_sector_labels: tuple[str, ...]
    boundary_flag: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_sector_labels"] = list(self.candidate_sector_labels)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


def episode_id(name: str) -> int:
    prefix = "episode_"
    if not name.startswith(prefix):
        raise ValueError(f"unexpected episode name: {name!r}")
    return int(name[len(prefix) :])


def load_manifest_episode_names(path: Path) -> list[str]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("train_ready_episode_ids")
    if not isinstance(names, list) or not names:
        raise ValueError(f"{path}: train_ready_episode_ids is missing or empty")
    result = [str(name) for name in names]
    if len(set(result)) != len(result):
        raise ValueError(f"{path}: duplicate train-ready episode ids")
    return result


def _metadata_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_episode(
    dataset_dir: Path,
    episode_name: str,
    initial_window_steps: int,
) -> EpisodeSignals:
    path = dataset_dir / f"{episode_name}.hdf5"
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        qpos = np.asarray(handle["observations/qpos"], dtype=np.float64)
        action = np.asarray(handle["action"], dtype=np.float64)
        metadata = handle.get("metadata")
        if metadata is not None:
            order = _metadata_text(
                metadata.attrs.get("qpos_order", "swing,boom,stick,bucket")
            )
            units = _metadata_text(metadata.attrs.get("qpos_units", "rad"))
            if order != "swing,boom,stick,bucket":
                raise ValueError(f"{path}: unexpected qpos_order={order!r}")
            if units != "rad":
                raise ValueError(f"{path}: unexpected qpos_units={units!r}")
    if qpos.ndim != 2 or qpos.shape[1] != 4:
        raise ValueError(f"{path}: qpos shape {qpos.shape}, expected (T, 4)")
    if action.shape != qpos.shape:
        raise ValueError(f"{path}: action shape {action.shape} != qpos shape {qpos.shape}")
    if not np.isfinite(qpos).all() or not np.isfinite(action).all():
        raise ValueError(f"{path}: non-finite qpos/action")
    count = min(initial_window_steps, qpos.shape[0])
    if count <= 0:
        raise ValueError(f"{path}: empty episode")
    return EpisodeSignals(
        episode_name=episode_name,
        episode_id=episode_id(episode_name),
        path=path,
        qpos=qpos,
        action=action,
        initial_qpos_rad=np.median(qpos[:count], axis=0),
    )


def robust_home_reference(
    episodes: Sequence[EpisodeSignals],
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.stack([episode.initial_qpos_rad for episode in episodes], axis=0)
    center = np.median(starts, axis=0)
    mad = np.median(np.abs(starts - center), axis=0)
    scale = np.maximum(1.4826 * mad, HOME_SCALE_FLOOR_RAD)
    return center, scale


def home_score(
    initial_qpos_rad: np.ndarray,
    center_qpos_rad: np.ndarray,
    scale_qpos_rad: np.ndarray,
) -> float:
    return float(
        np.max(np.abs(initial_qpos_rad - center_qpos_rad) / scale_qpos_rad)
    )


def moving_median(values: np.ndarray, radius: int = 2) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for index in range(values.shape[0]):
        lo = max(0, index - radius)
        hi = min(values.shape[0], index + radius + 1)
        result[index] = np.median(values[lo:hi])
    return result


def detect_entry_step(
    qpos: np.ndarray,
    initial_bucket_rad: float,
    config: AnnotationConfig,
) -> int | None:
    """Find the first sustained positive bucket displacement."""

    displacement = moving_median(qpos[:, 3]) - float(initial_bucket_rad)
    for index in np.flatnonzero(displacement >= config.bucket_rise_rad):
        tail = displacement[
            index : index + config.bucket_rise_lookahead_steps
        ]
        required = min(config.bucket_rise_hold_steps, tail.shape[0])
        if (
            required > 0
            and int(np.count_nonzero(tail >= config.bucket_rise_hold_rad))
            >= required
        ):
            return int(index)
    return None


def wrapped_delta(values: np.ndarray, reference: float) -> np.ndarray:
    return np.angle(
        np.exp(1j * (np.asarray(values, dtype=np.float64) - reference))
    )


def classify_qpos_sector(
    relative_swing_rad: float,
    config: AnnotationConfig,
) -> SectorEvidence:
    if relative_swing_rad >= config.qpos_side_min_abs_rad:
        return SectorEvidence("L", ("L",), False)
    if relative_swing_rad <= -config.qpos_side_min_abs_rad:
        return SectorEvidence("R", ("R",), False)
    if abs(relative_swing_rad) <= config.qpos_center_max_abs_rad:
        return SectorEvidence("C", ("C",), False)
    if relative_swing_rad > 0.0:
        return SectorEvidence(None, ("C", "L"), True)
    return SectorEvidence(None, ("R", "C"), True)


def classify_visual_sector(
    horizontal_displacement_px: float,
    config: AnnotationConfig,
) -> SectorEvidence:
    """Map background motion to operator-relative swing.

    When the cab/camera turns operator-left, the static background moves right
    to left in the image, hence the sign reversal.
    """

    if horizontal_displacement_px <= -config.vision_side_min_abs_px:
        return SectorEvidence("L", ("L",), False)
    if horizontal_displacement_px >= config.vision_side_min_abs_px:
        return SectorEvidence("R", ("R",), False)
    if abs(horizontal_displacement_px) <= config.vision_center_max_abs_px:
        return SectorEvidence("C", ("C",), False)
    if horizontal_displacement_px < 0.0:
        return SectorEvidence(None, ("C", "L"), True)
    return SectorEvidence(None, ("R", "C"), True)


def decode_eye_frame(
    handle: h5py.File,
    camera: str,
    step: int,
    config: AnnotationConfig,
) -> np.ndarray:
    dataset_path = f"observations/encoded_images/{camera}"
    if dataset_path not in handle:
        raise KeyError(f"missing {dataset_path}")
    dataset = handle[dataset_path]
    if step < 0 or step >= dataset.shape[0]:
        raise IndexError(f"{camera} step {step} outside [0, {dataset.shape[0]})")
    encoded = np.frombuffer(bytes(dataset[step]), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to decode {camera} step {step}")
    return cv2.resize(
        image,
        (config.image_width, config.image_height),
        interpolation=cv2.INTER_AREA,
    )


def static_background_mask(config: AnnotationConfig) -> np.ndarray:
    mask = np.zeros((config.image_height, config.image_width), dtype=np.uint8)
    mask[
        : config.static_mask_top_rows,
        : config.static_mask_left_end_col,
    ] = 255
    mask[
        : config.static_mask_top_rows,
        config.static_mask_right_start_col :,
    ] = 255
    return mask


def register_static_background(
    reference: np.ndarray,
    target: np.ndarray,
    camera: str,
    reference_step: int,
    target_step: int,
    config: AnnotationConfig,
) -> CameraRegistration:
    """Estimate robust static-background displacement between two eye frames."""

    reasons: list[str] = []
    if reference.shape != (config.image_height, config.image_width):
        raise ValueError(f"unexpected reference shape {reference.shape}")
    if target.shape != reference.shape:
        raise ValueError(f"target shape {target.shape} != {reference.shape}")

    sift = cv2.SIFT_create()
    mask = static_background_mask(config)
    reference_points, reference_desc = sift.detectAndCompute(reference, mask)
    target_points, target_desc = sift.detectAndCompute(target, mask)
    if reference_desc is None or target_desc is None:
        return CameraRegistration(
            camera=camera,
            status="failed",
            reference_step=reference_step,
            target_step=target_step,
            good_match_count=0,
            robust_match_count=0,
            horizontal_displacement_px=None,
            vertical_displacement_px=None,
            horizontal_mad_px=None,
            vertical_mad_px=None,
            sector_label=None,
            candidate_sector_labels=(),
            boundary_flag=False,
            reason_codes=("sift_descriptors_unavailable",),
        )

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        reference_desc,
        target_desc,
        k=2,
    )
    good_matches = [
        pair[0]
        for pair in pairs
        if len(pair) == 2
        and pair[0].distance < config.sift_ratio * pair[1].distance
    ]
    if len(good_matches) < config.min_good_matches:
        return CameraRegistration(
            camera=camera,
            status="failed",
            reference_step=reference_step,
            target_step=target_step,
            good_match_count=len(good_matches),
            robust_match_count=0,
            horizontal_displacement_px=None,
            vertical_displacement_px=None,
            horizontal_mad_px=None,
            vertical_mad_px=None,
            sector_label=None,
            candidate_sector_labels=(),
            boundary_flag=False,
            reason_codes=("insufficient_ratio_test_matches",),
        )

    displacement = np.asarray(
        [
            (
                target_points[match.trainIdx].pt[0]
                - reference_points[match.queryIdx].pt[0],
                target_points[match.trainIdx].pt[1]
                - reference_points[match.queryIdx].pt[1],
            )
            for match in good_matches
        ],
        dtype=np.float64,
    )
    median = np.median(displacement, axis=0)
    mad = np.median(np.abs(displacement - median), axis=0)
    residual_limit = np.maximum(
        config.robust_min_residual_px,
        config.robust_mad_multiplier * 1.4826 * mad,
    )
    robust_mask = np.all(
        np.abs(displacement - median) <= residual_limit,
        axis=1,
    )
    robust_displacement = displacement[robust_mask]
    robust_count = int(robust_displacement.shape[0])
    if robust_count < config.min_robust_matches:
        return CameraRegistration(
            camera=camera,
            status="failed",
            reference_step=reference_step,
            target_step=target_step,
            good_match_count=len(good_matches),
            robust_match_count=robust_count,
            horizontal_displacement_px=None,
            vertical_displacement_px=None,
            horizontal_mad_px=float(mad[0]),
            vertical_mad_px=float(mad[1]),
            sector_label=None,
            candidate_sector_labels=(),
            boundary_flag=False,
            reason_codes=("insufficient_robust_background_matches",),
        )

    robust_median = np.median(robust_displacement, axis=0)
    sector = classify_visual_sector(float(robust_median[0]), config)
    reasons.append("static_background_registration_succeeded")
    if sector.boundary:
        reasons.append("visual_displacement_in_boundary_band")
    return CameraRegistration(
        camera=camera,
        status="ok",
        reference_step=reference_step,
        target_step=target_step,
        good_match_count=len(good_matches),
        robust_match_count=robust_count,
        horizontal_displacement_px=float(robust_median[0]),
        vertical_displacement_px=float(robust_median[1]),
        horizontal_mad_px=float(mad[0]),
        vertical_mad_px=float(mad[1]),
        sector_label=sector.label,
        candidate_sector_labels=sector.candidate_labels,
        boundary_flag=sector.boundary,
        reason_codes=tuple(reasons),
    )


def register_eye_pair(
    path: Path,
    target_step: int,
    config: AnnotationConfig,
) -> list[CameraRegistration]:
    results: list[CameraRegistration] = []
    with h5py.File(path, "r") as handle:
        for camera in EYE_CAMERAS:
            try:
                reference = decode_eye_frame(handle, camera, 0, config)
                target = decode_eye_frame(handle, camera, target_step, config)
                result = register_static_background(
                    reference,
                    target,
                    camera,
                    0,
                    target_step,
                    config,
                )
            except (KeyError, IndexError, ValueError, cv2.error) as exc:
                result = CameraRegistration(
                    camera=camera,
                    status="failed",
                    reference_step=0,
                    target_step=target_step,
                    good_match_count=0,
                    robust_match_count=0,
                    horizontal_displacement_px=None,
                    vertical_displacement_px=None,
                    horizontal_mad_px=None,
                    vertical_mad_px=None,
                    sector_label=None,
                    candidate_sector_labels=(),
                    boundary_flag=False,
                    reason_codes=(
                        f"camera_registration_exception:{type(exc).__name__}",
                    ),
                )
            results.append(result)
    return results


def intersect_candidates(
    candidate_groups: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    if not candidate_groups:
        return ()
    common = set(candidate_groups[0])
    for group in candidate_groups[1:]:
        common.intersection_update(group)
    return tuple(label for label in ("L", "C", "R") if label in common)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(type(value).__name__)
