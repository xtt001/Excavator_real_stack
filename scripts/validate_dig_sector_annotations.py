#!/usr/bin/env python3
"""Validate and optionally recompute v0.1 automatic dig-sector annotations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTBED_ROOT = REPO_ROOT / "testbed"
if str(TESTBED_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTBED_ROOT))

from testbed.data.dig_sector_annotation import (  # noqa: E402
    CONTRACT_ID,
    EYE_CAMERAS,
    AnnotationConfig,
    EpisodeSignals,
    classify_qpos_sector,
    classify_visual_sector,
    detect_entry_step,
    home_score,
    intersect_candidates,
    load_episode,
    register_eye_pair,
    robust_home_reference,
    wrapped_delta,
)


SECTORS = {"L", "C", "R"}
STATUSES = {"accepted", "provisional", "rejected"}


class AnnotationError(ValueError):
    """Raised when a sidecar violates the v0.1 contract."""


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise AnnotationError(f"{context}: missing {key!r}")
    return mapping[key]


def close(
    observed: float | None,
    expected: float | None,
    *,
    context: str,
    tolerance: float = 1e-6,
) -> None:
    if observed is None or expected is None:
        if observed is not expected:
            raise AnnotationError(
                f"{context}: observed={observed!r}, expected={expected!r}"
            )
        return
    if not math.isclose(
        float(observed),
        float(expected),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise AnnotationError(
            f"{context}: observed={observed!r}, expected={expected!r}"
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnnotationError(
                f"{path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise AnnotationError(
                f"{path}:{line_number}: record must be an object"
            )
        records.append(record)
    if not records:
        raise AnnotationError(f"{path}: no annotation records")
    return records


def validate_record(record: dict[str, Any]) -> None:
    annotation_id = str(require(record, "annotation_id", "record"))
    if require(record, "contract_id", annotation_id) != CONTRACT_ID:
        raise AnnotationError(f"{annotation_id}: unsupported contract_id")

    source = require(record, "source", annotation_id)
    if require(source, "domain", annotation_id) != "real":
        raise AnnotationError(f"{annotation_id}: v0.1 annotator is real-only")
    if int(require(source, "episode_id", annotation_id)) < 0:
        raise AnnotationError(f"{annotation_id}: negative episode id")
    if int(require(source, "step_count", annotation_id)) <= 0:
        raise AnnotationError(f"{annotation_id}: non-positive step count")
    camera_names = require(source, "camera_names", annotation_id)
    if not set(EYE_CAMERAS).issubset(set(camera_names)):
        raise AnnotationError(f"{annotation_id}: missing eye cameras")

    semantics = require(record, "semantics", annotation_id)
    if semantics.get("label_kind") != "hindsight_achieved_sector":
        raise AnnotationError(f"{annotation_id}: wrong label_kind")
    if semantics.get("metric_equal_width_grid_claimed") is not False:
        raise AnnotationError(
            f"{annotation_id}: v0.1 cannot claim a metric grid"
        )

    command = require(record, "command", annotation_id)
    if (
        command.get("source") != "unknown_not_recorded"
        or command.get("dig_sector") is not None
    ):
        raise AnnotationError(
            f"{annotation_id}: historical outcome cannot become a command"
        )

    verification = require(record, "verification", annotation_id)
    if verification.get("privilege_used_for_annotation") is not False:
        raise AnnotationError(
            f"{annotation_id}: privilege_used_for_annotation must be false"
        )
    if verification.get("metric_ground_truth_available") is not False:
        raise AnnotationError(
            f"{annotation_id}: metric ground truth is not available"
        )

    entry = require(record, "dig_entry", annotation_id)
    vision = require(record, "vision_evidence", annotation_id)
    calibration_id = str(
        require(vision, "calibration_id", annotation_id)
    )
    config = AnnotationConfig(calibration_id=calibration_id)
    config.validate()
    expected_entry_parameters = {
        "moving_median_window_steps": 5,
        "bucket_rise_rad": config.bucket_rise_rad,
        "bucket_rise_hold_rad": config.bucket_rise_hold_rad,
        "bucket_rise_hold_steps": config.bucket_rise_hold_steps,
        "bucket_rise_lookahead_steps": (
            config.bucket_rise_lookahead_steps
        ),
    }
    if entry.get("parameters") != expected_entry_parameters:
        raise AnnotationError(
            f"{annotation_id}: entry parameters differ from v0.1"
        )
    representative_step = entry.get("representative_step")
    window_start = entry.get("window_start_step")
    window_end = entry.get("window_end_step")
    step_count = int(source["step_count"])
    if representative_step is not None:
        representative_step = int(representative_step)
        if not 0 <= representative_step < step_count:
            raise AnnotationError(f"{annotation_id}: entry outside episode")
        if (
            window_start is None
            or window_end is None
            or not int(window_start)
            <= representative_step
            <= int(window_end)
        ):
            raise AnnotationError(
                f"{annotation_id}: invalid entry window"
            )

    qpos = require(record, "qpos_evidence", annotation_id)
    if qpos.get("initial_home_score_threshold") != config.home_outlier_score:
        raise AnnotationError(
            f"{annotation_id}: home gate differs from v0.1"
        )
    expected_qpos_thresholds = {
        "center_accept_abs_max": config.qpos_center_max_abs_rad,
        "side_accept_abs_min": config.qpos_side_min_abs_rad,
        "boundary_band_abs": [
            config.qpos_center_max_abs_rad,
            config.qpos_side_min_abs_rad,
        ],
    }
    if qpos.get("sector_thresholds_rad") != expected_qpos_thresholds:
        raise AnnotationError(
            f"{annotation_id}: qpos thresholds differ from v0.1"
        )
    if vision.get("image_resolution") != [
        config.image_width,
        config.image_height,
    ]:
        raise AnnotationError(
            f"{annotation_id}: vision resolution differs from v0.1"
        )
    if vision.get("static_background_mask") != {
        "top_rows": config.static_mask_top_rows,
        "left_end_col": config.static_mask_left_end_col,
        "right_start_col": config.static_mask_right_start_col,
    }:
        raise AnnotationError(
            f"{annotation_id}: static background mask differs from v0.1"
        )
    if vision.get("sector_thresholds_px") != {
        "center_accept_abs_max": config.vision_center_max_abs_px,
        "side_accept_abs_min": config.vision_side_min_abs_px,
        "boundary_band_abs": [
            config.vision_center_max_abs_px,
            config.vision_side_min_abs_px,
        ],
    }:
        raise AnnotationError(
            f"{annotation_id}: vision thresholds differ from v0.1"
        )
    outcome = require(record, "outcome", annotation_id)
    quality = require(record, "quality", annotation_id)
    status = quality.get("status")
    if status not in STATUSES:
        raise AnnotationError(f"{annotation_id}: invalid status {status!r}")
    if quality.get("auto_usable") is not (status == "accepted"):
        raise AnnotationError(
            f"{annotation_id}: auto_usable must match accepted status"
        )
    if quality.get("review_required") is not (status != "accepted"):
        raise AnnotationError(
            f"{annotation_id}: review_required must be false only for accepted"
        )

    actual = outcome.get("actual_dig_sector")
    candidates = outcome.get("candidate_sector_labels")
    if not isinstance(candidates, list) or not set(candidates).issubset(
        SECTORS
    ):
        raise AnnotationError(f"{annotation_id}: invalid outcome candidates")

    cameras = vision.get("cameras")
    if not isinstance(cameras, list):
        raise AnnotationError(f"{annotation_id}: cameras must be a list")
    if representative_step is None:
        if cameras or vision.get("status") != "not_run":
            raise AnnotationError(
                f"{annotation_id}: vision must not run without an entry proxy"
            )
    else:
        camera_names_in_record = [
            camera.get("camera") for camera in cameras
        ]
        if (
            len(cameras) != len(EYE_CAMERAS)
            or set(camera_names_in_record) != set(EYE_CAMERAS)
        ):
            raise AnnotationError(
                f"{annotation_id}: entry record needs the exact eye pair"
            )
    relative_swing = qpos.get("relative_swing_rad")
    if relative_swing is not None:
        expected_qpos = classify_qpos_sector(
            float(relative_swing),
            config,
        )
        if (
            qpos.get("sector_label") != expected_qpos.label
            or qpos.get("candidate_sector_labels")
            != list(expected_qpos.candidate_labels)
            or qpos.get("boundary_flag") != expected_qpos.boundary
        ):
            raise AnnotationError(
                f"{annotation_id}: qpos classification is inconsistent"
            )
    successful_cameras = [
        camera for camera in cameras if camera.get("status") == "ok"
    ]
    for camera in successful_cameras:
        displacement = camera.get("horizontal_displacement_px")
        if displacement is None:
            raise AnnotationError(
                f"{annotation_id}: successful camera lacks displacement"
            )
        expected_camera_sector = classify_visual_sector(
            float(displacement),
            config,
        )
        if (
            camera.get("sector_label") != expected_camera_sector.label
            or camera.get("candidate_sector_labels")
            != list(expected_camera_sector.candidate_labels)
            or camera.get("boundary_flag")
            != expected_camera_sector.boundary
        ):
            raise AnnotationError(
                f"{annotation_id}: {camera.get('camera')} "
                "classification is inconsistent"
            )
    expected_vision_status = (
        "ok"
        if len(successful_cameras) == len(EYE_CAMERAS)
        else ("partial" if successful_cameras else "failed")
    )
    if representative_step is not None and (
        vision.get("status") != expected_vision_status
    ):
        raise AnnotationError(
            f"{annotation_id}: vision status is inconsistent"
        )
    vision_groups = [
        camera.get("candidate_sector_labels", [])
        for camera in successful_cameras
    ]
    expected_vision_candidates = (
        intersect_candidates(vision_groups) if vision_groups else ()
    )
    if vision.get("candidate_sector_labels") != list(
        expected_vision_candidates
    ):
        raise AnnotationError(
            f"{annotation_id}: fused vision candidates are inconsistent"
        )
    successful_labels = {
        camera.get("sector_label")
        for camera in successful_cameras
        if camera.get("sector_label") is not None
    }
    expected_vision_label = (
        next(iter(successful_labels))
        if len(successful_labels) == 1
        and len(successful_cameras) == len(EYE_CAMERAS)
        and all(
            not camera.get("boundary_flag")
            for camera in successful_cameras
        )
        else None
    )
    if vision.get("sector_label") != expected_vision_label:
        raise AnnotationError(
            f"{annotation_id}: fused vision label is inconsistent"
        )
    horizontal = [
        float(camera["horizontal_displacement_px"])
        for camera in successful_cameras
    ]
    expected_median_horizontal = (
        float(np.median(horizontal)) if horizontal else None
    )
    close(
        vision.get("median_horizontal_displacement_px"),
        expected_median_horizontal,
        context=f"{annotation_id}.median_eye_displacement",
    )
    clear_three_sensor_agreement = (
        representative_step is not None
        and qpos.get("sector_label") in SECTORS
        and not qpos.get("boundary_flag")
        and len(successful_cameras) == len(EYE_CAMERAS)
        and all(
            camera.get("sector_label") == qpos.get("sector_label")
            and not camera.get("boundary_flag")
            for camera in successful_cameras
        )
    )
    evidence_groups: list[Sequence[str]] = []
    qpos_candidates = qpos.get("candidate_sector_labels")
    if isinstance(qpos_candidates, list) and qpos_candidates:
        evidence_groups.append(qpos_candidates)
    evidence_groups.extend(
        camera.get("candidate_sector_labels", [])
        for camera in successful_cameras
    )
    fused_candidates = (
        intersect_candidates(evidence_groups)
        if representative_step is not None and evidence_groups
        else ()
    )
    expected_status = (
        "accepted"
        if clear_three_sensor_agreement
        else (
            "provisional"
            if representative_step is not None and fused_candidates
            else "rejected"
        )
    )
    if status != expected_status:
        raise AnnotationError(
            f"{annotation_id}: status={status}, evidence requires "
            f"{expected_status}"
        )

    if status == "accepted":
        if actual not in SECTORS or candidates != [actual]:
            raise AnnotationError(
                f"{annotation_id}: accepted record needs one resolved sector"
            )
        if representative_step is None:
            raise AnnotationError(
                f"{annotation_id}: accepted record has no entry proxy"
            )
        if qpos.get("sector_label") != actual or qpos.get("boundary_flag"):
            raise AnnotationError(
                f"{annotation_id}: accepted qpos evidence is not resolved"
            )
        if (
            vision.get("status") != "ok"
            or vision.get("sector_label") != actual
            or vision.get("boundary_flag")
            or not isinstance(cameras, list)
            or len(cameras) != len(EYE_CAMERAS)
        ):
            raise AnnotationError(
                f"{annotation_id}: accepted eye-pair evidence is not resolved"
            )
        by_name = {camera.get("camera"): camera for camera in cameras}
        if set(by_name) != set(EYE_CAMERAS):
            raise AnnotationError(
                f"{annotation_id}: accepted record lacks exact eye pair"
            )
        for camera_name, camera in by_name.items():
            if (
                camera.get("status") != "ok"
                or camera.get("sector_label") != actual
                or camera.get("boundary_flag")
            ):
                raise AnnotationError(
                    f"{annotation_id}: {camera_name} does not support {actual}"
                )
    else:
        if actual is not None:
            raise AnnotationError(
                f"{annotation_id}: non-accepted record cannot carry a label"
            )
        expected_candidates = (
            list(fused_candidates) if status == "provisional" else []
        )
        if candidates != expected_candidates:
            raise AnnotationError(
                f"{annotation_id}: outcome candidates differ from evidence"
            )


def load_sources(
    records: Sequence[dict[str, Any]],
    config: AnnotationConfig,
) -> list[EpisodeSignals]:
    episodes: list[EpisodeSignals] = []
    for record in records:
        source = record["source"]
        path = Path(source["dataset_path"])
        if not path.is_file():
            raise AnnotationError(
                f"{record['annotation_id']}: missing source {path}"
            )
        stat = path.stat()
        if stat.st_size != int(source["source_size_bytes"]):
            raise AnnotationError(
                f"{record['annotation_id']}: source size changed"
            )
        if stat.st_mtime_ns != int(source["source_mtime_ns"]):
            raise AnnotationError(
                f"{record['annotation_id']}: source mtime changed"
            )
        episode = load_episode(
            path.parent,
            str(source["episode_name"]),
            config.initial_window_steps,
        )
        if episode.path.resolve() != path.resolve():
            raise AnnotationError(
                f"{record['annotation_id']}: source path/name mismatch"
            )
        if episode.episode_id != int(source["episode_id"]):
            raise AnnotationError(
                f"{record['annotation_id']}: episode id mismatch"
            )
        if episode.qpos.shape[0] != int(source["step_count"]):
            raise AnnotationError(
                f"{record['annotation_id']}: source length changed"
            )
        episodes.append(episode)
    return episodes


def recompute_record(
    record: dict[str, Any],
    episode: EpisodeSignals,
    *,
    home_center: np.ndarray,
    home_scale: np.ndarray,
    config: AnnotationConfig,
) -> None:
    context = record["annotation_id"]
    qpos_record = record["qpos_evidence"]
    expected_score = home_score(
        episode.initial_qpos_rad,
        home_center,
        home_scale,
    )
    close(
        qpos_record["initial_home_score"],
        expected_score,
        context=f"{context}.initial_home_score",
    )
    if not np.allclose(
        np.asarray(qpos_record["initial_qpos_rad"], dtype=np.float64),
        episode.initial_qpos_rad,
        rtol=0.0,
        atol=1e-8,
    ):
        raise AnnotationError(f"{context}: initial_qpos_rad changed")

    if expected_score >= config.home_outlier_score:
        expected_entry = None
    else:
        expected_entry = detect_entry_step(
            episode.qpos,
            float(episode.initial_qpos_rad[3]),
            config,
        )
    observed_entry = record["dig_entry"]["representative_step"]
    if observed_entry != expected_entry:
        raise AnnotationError(
            f"{context}: entry={observed_entry}, recomputed={expected_entry}"
        )
    if expected_entry is None:
        return

    local_start = max(0, expected_entry - 2)
    local_end = min(episode.qpos.shape[0], expected_entry + 3)
    expected_relative = float(
        np.median(
            wrapped_delta(
                episode.qpos[local_start:local_end, 0],
                float(episode.initial_qpos_rad[0]),
            )
        )
    )
    close(
        qpos_record["relative_swing_rad"],
        expected_relative,
        context=f"{context}.relative_swing_rad",
    )
    expected_qpos_sector = classify_qpos_sector(expected_relative, config)
    if (
        qpos_record["sector_label"] != expected_qpos_sector.label
        or qpos_record["candidate_sector_labels"]
        != list(expected_qpos_sector.candidate_labels)
        or qpos_record["boundary_flag"] != expected_qpos_sector.boundary
    ):
        raise AnnotationError(f"{context}: recomputed qpos sector differs")

    expected_cameras = register_eye_pair(
        episode.path,
        expected_entry,
        config,
    )
    observed_cameras = {
        camera["camera"]: camera
        for camera in record["vision_evidence"]["cameras"]
    }
    if set(observed_cameras) != set(EYE_CAMERAS):
        raise AnnotationError(
            f"{context}: source-check needs both eye-camera records"
        )
    for expected in expected_cameras:
        observed = observed_cameras[expected.camera]
        if (
            observed["status"] != expected.status
            or observed["sector_label"] != expected.sector_label
            or observed["candidate_sector_labels"]
            != list(expected.candidate_sector_labels)
            or observed["boundary_flag"] != expected.boundary_flag
            or int(observed["good_match_count"])
            != expected.good_match_count
            or int(observed["robust_match_count"])
            != expected.robust_match_count
        ):
            raise AnnotationError(
                f"{context}: {expected.camera} recomputation differs"
            )
        close(
            observed["horizontal_displacement_px"],
            expected.horizontal_displacement_px,
            context=(
                f"{context}.{expected.camera}."
                "horizontal_displacement_px"
            ),
        )
        close(
            observed["vertical_displacement_px"],
            expected.vertical_displacement_px,
            context=(
                f"{context}.{expected.camera}."
                "vertical_displacement_px"
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument(
        "--check-source",
        action="store_true",
        help="re-read HDF5 and recompute entry, qpos, and eye evidence",
    )
    args = parser.parse_args()

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    records = load_jsonl(args.annotations)
    annotation_ids = [record.get("annotation_id") for record in records]
    if len(set(annotation_ids)) != len(annotation_ids):
        raise AnnotationError("duplicate annotation ids")
    generated = {record.get("generated_at") for record in records}
    if len(generated) != 1:
        raise AnnotationError("one sidecar batch must share generated_at")

    for record in records:
        validate_record(record)

    if args.check_source:
        calibration_ids = {
            record["vision_evidence"]["calibration_id"]
            for record in records
        }
        if len(calibration_ids) != 1:
            raise AnnotationError(
                "one sidecar batch must share one calibration id"
            )
        config = AnnotationConfig(
            calibration_id=next(iter(calibration_ids))
        )
        config.validate()
        episodes = load_sources(records, config)
        home_center, home_scale = robust_home_reference(episodes)
        for record, episode in zip(records, episodes, strict=True):
            recompute_record(
                record,
                episode,
                home_center=home_center,
                home_scale=home_scale,
                config=config,
            )

    statuses = Counter(
        record["quality"]["status"] for record in records
    )
    labels = Counter(
        record["outcome"]["actual_dig_sector"]
        for record in records
        if record["quality"]["status"] == "accepted"
    )
    print(
        json.dumps(
            {
                "validation": "PASS",
                "source_recomputed": args.check_source,
                "record_count": len(records),
                "status_counts": dict(sorted(statuses.items())),
                "accepted_label_counts": {
                    label: int(labels.get(label, 0))
                    for label in ("L", "C", "R")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
