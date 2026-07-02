#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INTRINSICS_MANIFEST = Path("configs/camera_intrinsics/gmsl_h190ta/manifest.json")


@dataclass(frozen=True)
class ImagePair:
    pair_id: str
    left_path: Path
    right_path: Path


@dataclass(frozen=True)
class CameraIntrinsics:
    camera_key: str
    device_hint: str
    serial: str
    orientation: str
    K: Any
    D: Any


def parse_inner_corners(raw: str) -> tuple[int, int]:
    if "x" not in raw:
        raise ValueError("--inner-corners must use COLSxROWS format, e.g. 8x6")
    cols_raw, rows_raw = raw.lower().split("x", 1)
    cols = int(cols_raw)
    rows = int(rows_raw)
    if cols <= 0 or rows <= 0:
        raise ValueError("--inner-corners values must be positive")
    return cols, rows


def _load_cv2_numpy():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV Python and NumPy are required for stereo calibration.") from exc
    return cv2, np


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _progress(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"[gmsl-stereo-calibrate] {message}", file=sys.stderr, flush=True)


def load_camera_intrinsics(manifest_path: str | Path, camera_key: str) -> CameraIntrinsics:
    cv2, np = _load_cv2_numpy()
    del cv2
    manifest_file = Path(manifest_path)
    manifest = _read_json(manifest_file)
    if manifest.get("distortion_model") != "opencv_fisheye":
        raise ValueError(
            f"{manifest_file}: expected distortion_model='opencv_fisheye', got {manifest.get('distortion_model')!r}"
        )
    cameras = {str(camera.get("camera_key")): camera for camera in manifest.get("cameras", [])}
    camera = cameras.get(camera_key)
    if camera is None:
        raise ValueError(f"{manifest_file}: missing camera_key={camera_key!r}")
    return CameraIntrinsics(
        camera_key=camera_key,
        device_hint=str(camera.get("device_hint", "")),
        serial=str(camera.get("serial", "")),
        orientation=str(camera.get("orientation", "normal")),
        K=np.asarray(camera["K"], dtype=np.float64),
        D=np.asarray(camera["D"], dtype=np.float64).reshape(4, 1),
    )


def _image_sort_key(path: Path) -> tuple[str, str]:
    return (path.stem, path.name)


def collect_image_pairs(
    *,
    left_dir: str | Path,
    right_dir: str | Path,
    pattern: str = "*.png",
    pair_by_order: bool = False,
) -> list[ImagePair]:
    left_path = Path(left_dir)
    right_path = Path(right_dir)
    left_images = sorted(left_path.glob(pattern), key=_image_sort_key)
    right_images = sorted(right_path.glob(pattern), key=_image_sort_key)
    if not left_images:
        raise ValueError(f"no left images match {left_path / pattern}")
    if not right_images:
        raise ValueError(f"no right images match {right_path / pattern}")

    if pair_by_order:
        if len(left_images) != len(right_images):
            raise ValueError(f"pair-by-order requires same image count: left={len(left_images)} right={len(right_images)}")
        return [
            ImagePair(pair_id=f"{index:06d}", left_path=left_image, right_path=right_image)
            for index, (left_image, right_image) in enumerate(zip(left_images, right_images, strict=True))
        ]

    right_by_stem = {image.stem: image for image in right_images}
    pairs: list[ImagePair] = []
    missing: list[str] = []
    for left_image in left_images:
        right_image = right_by_stem.get(left_image.stem)
        if right_image is None:
            missing.append(left_image.name)
            continue
        pairs.append(ImagePair(pair_id=left_image.stem, left_path=left_image, right_path=right_image))
    if missing:
        raise ValueError(f"right directory is missing {len(missing)} matching stems; first missing={missing[0]}")
    if not pairs:
        raise ValueError("no paired images found")
    return pairs


def collect_image_pairs_from_manifest(
    *,
    pairs_json: str | Path,
    left_camera: str,
    right_camera: str,
) -> list[ImagePair]:
    manifest_path = Path(pairs_json)
    manifest = _read_json(manifest_path)
    left = manifest.get("left", {})
    right = manifest.get("right", {})
    if left.get("camera_key") != left_camera or right.get("camera_key") != right_camera:
        raise ValueError(
            f"{manifest_path}: expected left/right {left_camera}/{right_camera}, "
            f"got {left.get('camera_key')}/{right.get('camera_key')}"
        )
    base_dir = manifest_path.parent
    pairs = [
        ImagePair(
            pair_id=f"{int(frame['index']):06d}",
            left_path=base_dir / str(frame["left_path"]),
            right_path=base_dir / str(frame["right_path"]),
        )
        for frame in manifest.get("frames", [])
    ]
    if not pairs:
        raise ValueError(f"{manifest_path}: no frames in pairs manifest")
    return pairs


def make_object_points(cols: int, rows: int, square_size_m: float) -> Any:
    _, np = _load_cv2_numpy()
    object_points = np.zeros((cols * rows, 1, 3), np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points[:, 0, :2] = grid * float(square_size_m)
    return object_points


def _detect_checkerboard(cv2: Any, np: Any, image: Any, pattern_size: tuple[int, int]) -> tuple[bool, Any, str]:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            return True, np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2), "findChessboardCornersSB"

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not ok:
        return False, None, "findChessboardCorners"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2), "findChessboardCorners"


def _draw_corners(
    *,
    cv2: Any,
    annotated_dir: Path,
    pair: ImagePair,
    pattern_size: tuple[int, int],
    left_image: Any,
    right_image: Any,
    left_corners: Any,
    right_corners: Any,
    left_found: bool,
    right_found: bool,
) -> None:
    annotated_dir.mkdir(parents=True, exist_ok=True)
    left_draw = left_image.copy()
    right_draw = right_image.copy()
    if left_found:
        cv2.drawChessboardCorners(left_draw, pattern_size, left_corners.astype("float32"), True)
    else:
        cv2.putText(left_draw, "checkerboard not found", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    if right_found:
        cv2.drawChessboardCorners(right_draw, pattern_size, right_corners.astype("float32"), True)
    else:
        cv2.putText(right_draw, "checkerboard not found", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(str(annotated_dir / f"{pair.pair_id}_left.jpg"), left_draw)
    cv2.imwrite(str(annotated_dir / f"{pair.pair_id}_right.jpg"), right_draw)


def _matrix_to_list(matrix: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _vector_to_list(vector: Any) -> list[float]:
    return [float(value) for value in vector.reshape(-1)]


def _write_result(output_json: str | Path, result: dict[str, Any]) -> None:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def calibrate_stereo_pair(
    *,
    intrinsics_manifest: str | Path,
    left_camera: str,
    right_camera: str,
    pairs: list[ImagePair],
    output_json: str | Path,
    inner_corners: tuple[int, int] = (8, 6),
    square_size_m: float = 0.025,
    min_valid_pairs: int = 12,
    max_rms_px: float | None = None,
    annotated_dir: str | Path | None = None,
    allow_image_size_mismatch: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    quiet = not progress
    _progress(
        "prepare calibration "
        f"left={left_camera} right={right_camera} pairs={len(pairs)} "
        f"target={inner_corners[0]}x{inner_corners[1]} square_size_m={square_size_m} "
        f"min_valid_pairs={min_valid_pairs} output={output_json}",
        quiet=quiet,
    )
    cv2, np = _load_cv2_numpy()
    left_intrinsics = load_camera_intrinsics(intrinsics_manifest, left_camera)
    right_intrinsics = load_camera_intrinsics(intrinsics_manifest, right_camera)
    _progress(
        f"intrinsics loaded: {left_camera} serial={left_intrinsics.serial}, "
        f"{right_camera} serial={right_intrinsics.serial}",
        quiet=quiet,
    )
    object_template = make_object_points(inner_corners[0], inner_corners[1], square_size_m)

    object_points: list[Any] = []
    left_image_points: list[Any] = []
    right_image_points: list[Any] = []
    detections: list[dict[str, Any]] = []
    image_size: tuple[int, int] | None = None
    annotated_path = Path(annotated_dir) if annotated_dir is not None else None

    for pair_index, pair in enumerate(pairs, start=1):
        _progress(f"detecting pair {pair_index}/{len(pairs)} id={pair.pair_id}", quiet=quiet)
        left_image = cv2.imread(str(pair.left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(pair.right_path), cv2.IMREAD_COLOR)
        if left_image is None or right_image is None:
            detections.append(
                {
                    "pair_id": pair.pair_id,
                    "left_path": str(pair.left_path),
                    "right_path": str(pair.right_path),
                    "used": False,
                    "reason": "image_read_failed",
                    "left_read_ok": left_image is not None,
                    "right_read_ok": right_image is not None,
                }
            )
            _progress(
                f"pair {pair_index}/{len(pairs)} id={pair.pair_id} skipped: image_read_failed "
                f"left_read_ok={left_image is not None} right_read_ok={right_image is not None}",
                quiet=quiet,
            )
            continue

        current_size = (int(left_image.shape[1]), int(left_image.shape[0]))
        right_size = (int(right_image.shape[1]), int(right_image.shape[0]))
        if current_size != right_size:
            detections.append(
                {
                    "pair_id": pair.pair_id,
                    "left_path": str(pair.left_path),
                    "right_path": str(pair.right_path),
                    "used": False,
                    "reason": f"image_size_mismatch left={current_size} right={right_size}",
                }
            )
            _progress(
                f"pair {pair_index}/{len(pairs)} id={pair.pair_id} skipped: "
                f"image_size_mismatch left={current_size} right={right_size}",
                quiet=quiet,
            )
            continue
        if image_size is None:
            image_size = current_size
            _progress(f"image size set to {image_size[0]}x{image_size[1]}", quiet=quiet)
        elif current_size != image_size:
            detections.append(
                {
                    "pair_id": pair.pair_id,
                    "left_path": str(pair.left_path),
                    "right_path": str(pair.right_path),
                    "used": False,
                    "reason": f"image_size_changed first={image_size} current={current_size}",
                }
            )
            _progress(
                f"pair {pair_index}/{len(pairs)} id={pair.pair_id} skipped: "
                f"image_size_changed first={image_size} current={current_size}",
                quiet=quiet,
            )
            continue

        left_found, left_corners, left_method = _detect_checkerboard(cv2, np, left_image, inner_corners)
        right_found, right_corners, right_method = _detect_checkerboard(cv2, np, right_image, inner_corners)
        used = bool(left_found and right_found)
        if used:
            object_points.append(object_template.copy())
            left_image_points.append(left_corners)
            right_image_points.append(right_corners)
        if annotated_path is not None:
            _draw_corners(
                cv2=cv2,
                annotated_dir=annotated_path,
                pair=pair,
                pattern_size=inner_corners,
                left_image=left_image,
                right_image=right_image,
                left_corners=left_corners,
                right_corners=right_corners,
                left_found=left_found,
                right_found=right_found,
            )
        detections.append(
            {
                "pair_id": pair.pair_id,
                "left_path": str(pair.left_path),
                "right_path": str(pair.right_path),
                "left_found": bool(left_found),
                "right_found": bool(right_found),
                "left_method": left_method,
                "right_method": right_method,
                "used": used,
            }
        )
        if used:
            _progress(
                f"pair {pair_index}/{len(pairs)} id={pair.pair_id} ok: "
                f"valid={len(object_points)} left_method={left_method} right_method={right_method}",
                quiet=quiet,
            )
        else:
            _progress(
                f"pair {pair_index}/{len(pairs)} id={pair.pair_id} skipped: "
                f"left_found={left_found} right_found={right_found}",
                quiet=quiet,
            )

    manifest = _read_json(Path(intrinsics_manifest))
    manifest_size = (int(manifest["image_width"]), int(manifest["image_height"]))
    if image_size is not None and image_size != manifest_size and not allow_image_size_mismatch:
        raise ValueError(
            f"image size {image_size} does not match intrinsics manifest size {manifest_size}; "
            "use raw 1920x1536 frames or pass --allow-image-size-mismatch for diagnostics only"
        )

    base_result: dict[str, Any] = {
        "schema_version": 1,
        "calibration_type": "opencv_fisheye_stereo_pair",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "intrinsics_manifest": str(intrinsics_manifest),
        "left_camera": {
            "camera_key": left_intrinsics.camera_key,
            "device_hint": left_intrinsics.device_hint,
            "serial": left_intrinsics.serial,
            "orientation": left_intrinsics.orientation,
        },
        "right_camera": {
            "camera_key": right_intrinsics.camera_key,
            "device_hint": right_intrinsics.device_hint,
            "serial": right_intrinsics.serial,
            "orientation": right_intrinsics.orientation,
        },
        "target": {
            "type": "checkerboard",
            "inner_corners": [inner_corners[0], inner_corners[1]],
            "square_size_m": float(square_size_m),
        },
        "image_size_px": list(image_size) if image_size is not None else None,
        "total_pair_count": len(pairs),
        "valid_pair_count": len(object_points),
        "min_valid_pairs": min_valid_pairs,
        "detection_results": detections,
        "image_orientation_used": "raw_unrotated_frames_matching_vendor_intrinsics",
        "coordinate_convention": "OpenCV camera frame: x right, y down, z forward",
    }

    if len(object_points) < min_valid_pairs:
        result = {
            **base_result,
            "status": "failed_insufficient_valid_pairs",
            "error": f"valid pairs {len(object_points)} < min_valid_pairs {min_valid_pairs}",
        }
        _write_result(output_json, result)
        _progress(
            f"failed: valid pairs {len(object_points)} < min_valid_pairs {min_valid_pairs}; "
            f"partial result={output_json}",
            quiet=quiet,
        )
        raise RuntimeError(result["error"])

    _progress(f"solving stereo calibration with {len(object_points)} valid pairs", quiet=quiet)
    flags = cv2.fisheye.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    R_init = np.eye(3, dtype=np.float64)
    T_init = np.zeros((3, 1), dtype=np.float64)
    rms, _, _, _, _, R, T = cv2.fisheye.stereoCalibrate(
        object_points,
        left_image_points,
        right_image_points,
        left_intrinsics.K.copy(),
        left_intrinsics.D.copy(),
        right_intrinsics.K.copy(),
        right_intrinsics.D.copy(),
        image_size,
        R_init,
        T_init,
        flags,
        criteria,
    )
    rms_ok = None if max_rms_px is None else bool(float(rms) <= max_rms_px)
    result = {
        **base_result,
        "status": "ok" if rms_ok is not False else "failed_rms_threshold",
        "rms_px": float(rms),
        "max_rms_px": max_rms_px,
        "acceptance": {
            "valid_pair_count_ok": len(object_points) >= min_valid_pairs,
            "rms_px_ok": rms_ok,
        },
        "right_T_left": {
            "R": _matrix_to_list(R),
            "T_m": _vector_to_list(T),
            "transform_direction": "right_T_left",
            "opencv_stereo_convention": "X_right = R * X_left + T",
        },
    }
    _write_result(output_json, result)
    _progress(
        f"success: status={result['status']} valid={len(object_points)}/{len(pairs)} "
        f"rms_px={float(rms):.6f} output={output_json}",
        quiet=quiet,
    )
    if rms_ok is False:
        raise RuntimeError(f"rms_px {float(rms):.6f} > max_rms_px {max_rms_px:.6f}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve pairwise GMSL fisheye stereo extrinsics with fixed intrinsics.")
    parser.add_argument("--intrinsics-manifest", type=Path, default=DEFAULT_INTRINSICS_MANIFEST)
    parser.add_argument("--left", required=True, help="Left camera key in intrinsics manifest, e.g. video4.")
    parser.add_argument("--right", required=True, help="Right camera key in intrinsics manifest, e.g. video5.")
    parser.add_argument("--pairs-json", type=Path, default=None, help="pairs.json written by capture_gmsl_stereo_pairs.py.")
    parser.add_argument("--left-dir", type=Path, default=None)
    parser.add_argument("--right-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--pair-by-order", action="store_true", help="Pair sorted images by index instead of matching stems.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--inner-corners", default="8x6")
    parser.add_argument("--square-size-m", type=float, default=0.025)
    parser.add_argument("--min-valid-pairs", type=int, default=12)
    parser.add_argument("--max-rms-px", type=float, default=None)
    parser.add_argument("--annotated-dir", type=Path, default=None)
    parser.add_argument("--allow-image-size-mismatch", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    parser.add_argument("--quiet-progress", action="store_true", help="Suppress human-readable progress on stderr.")
    return parser.parse_args()


def _collect_pairs_from_args(args: argparse.Namespace) -> list[ImagePair]:
    if args.pairs_json is not None:
        return collect_image_pairs_from_manifest(
            pairs_json=args.pairs_json,
            left_camera=args.left,
            right_camera=args.right,
        )
    if args.left_dir is None or args.right_dir is None:
        raise ValueError("provide either --pairs-json or both --left-dir and --right-dir")
    return collect_image_pairs(
        left_dir=args.left_dir,
        right_dir=args.right_dir,
        pattern=args.pattern,
        pair_by_order=args.pair_by_order,
    )


def main() -> int:
    args = parse_args()
    try:
        pairs = _collect_pairs_from_args(args)
        result = calibrate_stereo_pair(
            intrinsics_manifest=args.intrinsics_manifest,
            left_camera=args.left,
            right_camera=args.right,
            pairs=pairs,
            output_json=args.output_json,
            inner_corners=parse_inner_corners(args.inner_corners),
            square_size_m=args.square_size_m,
            min_valid_pairs=args.min_valid_pairs,
            max_rms_px=args.max_rms_px,
            annotated_dir=args.annotated_dir,
            allow_image_size_mismatch=args.allow_image_size_mismatch,
            progress=not args.quiet_progress,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"stereo calibration: {args.output_json}")
        print(f"valid pairs: {result['valid_pair_count']}/{result['total_pair_count']}")
        print(f"rms_px: {result['rms_px']:.6f}")
        print("transform: right_T_left, X_right = R * X_left + T")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
