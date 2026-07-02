#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple


class CameraSpec(NamedTuple):
    camera_key: str
    device: str

    def to_dict(self) -> dict[str, str]:
        return {"camera_key": self.camera_key, "device": self.device}


def parse_camera_spec(raw: str) -> CameraSpec:
    if "=" not in raw:
        raise ValueError(f"camera spec must be KEY=/dev/videoN, got {raw!r}")
    camera_key, device = raw.split("=", 1)
    camera_key = camera_key.strip()
    device = device.strip()
    if not camera_key:
        raise ValueError(f"camera spec has empty key: {raw!r}")
    if not device.startswith("/dev/video"):
        raise ValueError(f"camera spec has invalid device: {raw!r}")
    return CameraSpec(camera_key=camera_key, device=device)


def _load_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV Python is required for stereo pair capture.") from exc
    return cv2


def _progress(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"[gmsl-stereo-capture] {message}", file=sys.stderr, flush=True)


def _open_capture(cv2: Any, spec: CameraSpec, *, width: int, height: int, fourcc: str) -> Any:
    capture = cv2.VideoCapture(spec.device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"{spec.camera_key}: failed to open {spec.device}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fourcc:
        if len(fourcc) != 4:
            raise ValueError("--fourcc must contain exactly four characters")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    return capture


def _imwrite_params(cv2: Any, image_format: str, *, jpeg_quality: int, png_compression: int) -> list[int]:
    if image_format == "jpg":
        return [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    if image_format == "png":
        return [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]
    raise ValueError(f"unsupported image format: {image_format}")


def _warmup_pair(left_capture: Any, right_capture: Any, frames: int) -> None:
    for _ in range(frames):
        left_capture.grab()
        right_capture.grab()
        left_capture.retrieve()
        right_capture.retrieve()


def capture_stereo_pairs(
    *,
    left: CameraSpec,
    right: CameraSpec,
    output_dir: str | Path,
    count: int = 30,
    interval_s: float = 1.0,
    width: int = 1920,
    height: int = 1536,
    fourcc: str = "UYVY",
    warmup_frames: int = 10,
    image_format: str = "png",
    jpeg_quality: int = 95,
    png_compression: int = 3,
    progress: bool = True,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("--count must be positive")
    if interval_s < 0:
        raise ValueError("--interval-s must be non-negative")
    quiet = not progress
    _progress(
        "prepare capture "
        f"left={left.camera_key}:{left.device} right={right.camera_key}:{right.device} "
        f"count={count} interval_s={interval_s} size={width}x{height} fourcc={fourcc or 'default'} "
        f"format={image_format} output={output_dir}",
        quiet=quiet,
    )
    cv2 = _load_cv2()
    output_path = Path(output_dir)
    left_dir = output_path / left.camera_key
    right_dir = output_path / right.camera_key
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    _progress(f"output directories ready: {left_dir} , {right_dir}", quiet=quiet)

    params = _imwrite_params(cv2, image_format, jpeg_quality=jpeg_quality, png_compression=png_compression)
    _progress(f"opening cameras: {left.device} and {right.device}", quiet=quiet)
    left_capture = _open_capture(cv2, left, width=width, height=height, fourcc=fourcc)
    right_capture = _open_capture(cv2, right, width=width, height=height, fourcc=fourcc)
    _progress("camera open success", quiet=quiet)
    frames: list[dict[str, Any]] = []
    try:
        if warmup_frames > 0:
            _progress(f"warming up cameras: {warmup_frames} frame pairs", quiet=quiet)
        _warmup_pair(left_capture, right_capture, warmup_frames)
        _progress("warmup complete; start capture", quiet=quiet)
        next_capture_time = time.monotonic()
        for index in range(count):
            now = time.monotonic()
            if now < next_capture_time:
                time.sleep(next_capture_time - now)
            next_capture_time = time.monotonic() + interval_s

            _progress(f"capturing pair {index + 1}/{count}", quiet=quiet)
            grab_started_mono_ns = time.monotonic_ns()
            left_grab_ok = bool(left_capture.grab())
            right_grab_ok = bool(right_capture.grab())
            if not left_grab_ok or not right_grab_ok:
                raise RuntimeError(
                    f"grab failed at pair {index}: left_ok={left_grab_ok} right_ok={right_grab_ok}"
                )

            left_ok, left_frame = left_capture.retrieve()
            left_retrieve_mono_ns = time.monotonic_ns()
            right_ok, right_frame = right_capture.retrieve()
            right_retrieve_mono_ns = time.monotonic_ns()
            if not left_ok or left_frame is None or not right_ok or right_frame is None:
                raise RuntimeError(
                    f"retrieve failed at pair {index}: left_ok={left_ok} right_ok={right_ok}"
                )

            suffix = f".{image_format}"
            left_rel = Path(left.camera_key) / f"{index:06d}{suffix}"
            right_rel = Path(right.camera_key) / f"{index:06d}{suffix}"
            left_path = output_path / left_rel
            right_path = output_path / right_rel
            if not cv2.imwrite(str(left_path), left_frame, params):
                raise RuntimeError(f"failed to write {left_path}")
            if not cv2.imwrite(str(right_path), right_frame, params):
                raise RuntimeError(f"failed to write {right_path}")

            host_retrieve_skew_ms = (right_retrieve_mono_ns - left_retrieve_mono_ns) / 1_000_000.0
            frames.append(
                {
                    "index": index,
                    "left_path": str(left_rel),
                    "right_path": str(right_rel),
                    "grab_started_mono_ns": grab_started_mono_ns,
                    "left_retrieve_mono_ns": left_retrieve_mono_ns,
                    "right_retrieve_mono_ns": right_retrieve_mono_ns,
                    "host_retrieve_skew_ms": host_retrieve_skew_ms,
                }
            )
            _progress(
                f"captured pair {index + 1}/{count}: left={left_rel} right={right_rel} "
                f"host_retrieve_skew_ms={host_retrieve_skew_ms:.3f}",
                quiet=quiet,
            )
    finally:
        left_capture.release()
        right_capture.release()
        _progress("camera handles released", quiet=quiet)

    manifest = {
        "schema_version": 1,
        "capture_type": "gmsl_stereo_pair_checkerboard",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_path),
        "left": left.to_dict(),
        "right": right.to_dict(),
        "width": width,
        "height": height,
        "fourcc": fourcc,
        "image_format": image_format,
        "warmup_frames": warmup_frames,
        "interval_s": interval_s,
        "synchronization_note": "OpenCV grab() is issued for left then right to reduce host-side skew; keep the checkerboard still for each pair.",
        "frames": frames,
    }
    manifest_path = output_path / "pairs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _progress(f"success: captured {len(frames)}/{count} pairs; manifest={manifest_path}", quiet=quiet)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture paired checkerboard frames from two GMSL V4L2 cameras.")
    parser.add_argument("--left", required=True, help="Left camera spec, e.g. video4=/dev/video4.")
    parser.add_argument("--right", required=True, help="Right camera spec, e.g. video5=/dev/video5.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--fourcc", default="UYVY", help="V4L2 fourcc to request. Use an empty string to leave default.")
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--dry-run-plan", action="store_true", help="Print the capture plan without opening cameras.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    parser.add_argument("--quiet-progress", action="store_true", help="Suppress human-readable progress on stderr.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        left = parse_camera_spec(args.left)
        right = parse_camera_spec(args.right)
        if args.dry_run_plan:
            output = {
                "left": left.to_dict(),
                "right": right.to_dict(),
                "output_dir": str(args.output_dir),
                "count": args.count,
                "width": args.width,
                "height": args.height,
                "fourcc": args.fourcc,
            }
        else:
            output = capture_stereo_pairs(
                left=left,
                right=right,
                output_dir=args.output_dir,
                count=args.count,
                interval_s=args.interval_s,
                width=args.width,
                height=args.height,
                fourcc=args.fourcc,
                warmup_frames=args.warmup_frames,
                image_format=args.image_format,
                jpeg_quality=args.jpeg_quality,
                png_compression=args.png_compression,
                progress=not args.quiet_progress,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(output, indent=2))
    elif args.dry_run_plan:
        print(f"left: {output['left']['camera_key']} {output['left']['device']}")
        print(f"right: {output['right']['camera_key']} {output['right']['device']}")
        print(f"output: {output['output_dir']}")
    else:
        print(f"pairs manifest: {Path(output['output_dir']) / 'pairs.json'}")
        print(f"frames: {len(output['frames'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
