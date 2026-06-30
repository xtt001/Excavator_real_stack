#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple


DEFAULT_MAPPING = Path("configs/camera_calibration/gmsl_h190ta_four_camera/camera_mount_mapping.json")
VALID_ORIENTATIONS = {"normal", "rotate_180"}


class CaptureTarget(NamedTuple):
    mount_position: str
    camera_key: str
    device: str
    serial: str
    orientation: str
    intrinsics_file: str

    def to_dict(self) -> dict[str, str]:
        return {
            "mount_position": self.mount_position,
            "camera_key": self.camera_key,
            "device": self.device,
            "serial": self.serial,
            "orientation": self.orientation,
            "intrinsics_file": self.intrinsics_file,
        }


class SkippedTarget(NamedTuple):
    mount_position: str
    camera_key: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "mount_position": self.mount_position,
            "camera_key": self.camera_key,
            "reason": self.reason,
        }


class CapturePlan(NamedTuple):
    targets: list[CaptureTarget]
    skipped: list[SkippedTarget]

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": [target.to_dict() for target in self.targets],
            "skipped": [target.to_dict() for target in self.skipped],
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_positions(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    positions = [item.strip() for item in raw.split(",") if item.strip()]
    if not positions:
        raise ValueError("--positions must contain at least one mount position")
    return positions


def build_capture_plan(
    mapping_path: str | Path,
    *,
    positions: list[str] | None = None,
    require_all: bool = False,
) -> CapturePlan:
    mapping_file = Path(mapping_path)
    mapping = _read_json(mapping_file)
    mounts = mapping.get("mounts", [])
    mounts_by_position = {mount["mount_position"]: mount for mount in mounts}
    requested_positions = positions or mapping.get("training_camera_order") or mapping.get("position_order") or []

    targets: list[CaptureTarget] = []
    skipped: list[SkippedTarget] = []
    errors: list[str] = []
    for mount_position in requested_positions:
        mount = mounts_by_position.get(mount_position)
        if mount is None:
            errors.append(f"{mount_position}: not found in {mapping_file}")
            continue

        camera_key = str(mount.get("camera_key", ""))
        status = mount.get("intrinsics_status")
        device = str(mount.get("device_hint", ""))
        orientation = str(mount.get("orientation", ""))
        if status != "available":
            skipped.append(
                SkippedTarget(
                    mount_position=mount_position,
                    camera_key=camera_key,
                    reason=f"intrinsics_status={status!r}",
                )
            )
            continue
        if not device.startswith("/dev/video"):
            errors.append(f"{mount_position}: invalid device_hint={device!r}")
            continue
        if orientation not in VALID_ORIENTATIONS:
            errors.append(f"{mount_position}: invalid orientation={orientation!r}")
            continue

        targets.append(
            CaptureTarget(
                mount_position=mount_position,
                camera_key=camera_key,
                device=device,
                serial=str(mount.get("serial", "")),
                orientation=orientation,
                intrinsics_file=str(mount.get("intrinsics_file", "")),
            )
        )

    if require_all and skipped:
        errors.extend(f"{target.mount_position}: {target.reason}" for target in skipped)
    if errors:
        raise ValueError("; ".join(errors))
    return CapturePlan(targets=targets, skipped=skipped)


def _load_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV Python and NumPy are required for real frame capture.") from exc
    return cv2, np


def _apply_orientation(cv2: Any, frame: Any, orientation: str) -> Any:
    if orientation == "normal":
        return frame
    if orientation == "rotate_180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise ValueError(f"unsupported orientation: {orientation}")


def _capture_one_frame(
    *,
    cv2: Any,
    target: CaptureTarget,
    width: int | None,
    height: int | None,
    frames: int,
    settle_frames: int,
) -> Any:
    capture = cv2.VideoCapture(target.device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"{target.mount_position}: failed to open {target.device}")
    try:
        if width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        frame = None
        total_reads = settle_frames + frames
        for _ in range(total_reads):
            ok, candidate = capture.read()
            if ok:
                frame = candidate
        if frame is None:
            raise RuntimeError(f"{target.mount_position}: no frame captured from {target.device}")
        return frame
    finally:
        capture.release()


def _make_preview(cv2: Any, frame: Any, target: CaptureTarget, preview_width: int, preview_height: int) -> Any:
    preview = cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    label = f"{target.mount_position} {target.camera_key} {target.serial}"
    cv2.rectangle(preview, (0, 0), (preview_width, 28), (0, 0, 0), thickness=-1)
    cv2.putText(preview, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def _make_contact_sheet(np: Any, previews: list[Any], *, columns: int = 2) -> Any:
    if not previews:
        raise ValueError("no previews available for contact sheet")
    height, width = previews[0].shape[:2]
    rows = (len(previews) + columns - 1) // columns
    sheet = np.zeros((rows * height, columns * width, previews[0].shape[2]), dtype=previews[0].dtype)
    for index, preview in enumerate(previews):
        row = index // columns
        col = index % columns
        sheet[row * height : (row + 1) * height, col * width : (col + 1) * width] = preview
    return sheet


def capture_contact_sheet(
    *,
    plan: CapturePlan,
    output_dir: str | Path,
    width: int | None = 1920,
    height: int | None = 1536,
    frames: int = 3,
    settle_frames: int = 3,
    preview_width: int = 480,
    preview_height: int = 384,
) -> dict[str, Any]:
    if not plan.targets:
        raise ValueError("capture plan has no available camera targets")
    cv2, np = _load_cv2()
    output_path = Path(output_dir)
    raw_dir = output_path / "raw"
    preview_dir = output_path / "preview"
    raw_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    captures: list[dict[str, Any]] = []
    previews: list[Any] = []
    for target in plan.targets:
        frame = _capture_one_frame(
            cv2=cv2,
            target=target,
            width=width,
            height=height,
            frames=frames,
            settle_frames=settle_frames,
        )
        raw_path = raw_dir / f"{target.mount_position}_{target.camera_key}_raw.jpg"
        oriented = _apply_orientation(cv2, frame, target.orientation)
        preview = _make_preview(cv2, oriented, target, preview_width, preview_height)
        preview_path = preview_dir / f"{target.mount_position}_{target.camera_key}_preview.jpg"
        if not cv2.imwrite(str(raw_path), frame):
            raise RuntimeError(f"{target.mount_position}: failed to write {raw_path}")
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"{target.mount_position}: failed to write {preview_path}")
        previews.append(preview)
        captures.append(
            {
                **target.to_dict(),
                "raw_path": str(raw_path),
                "preview_path": str(preview_path),
                "status": "captured",
            }
        )

    contact_sheet = _make_contact_sheet(np, previews)
    contact_sheet_path = output_path / "contact_sheet.jpg"
    if not cv2.imwrite(str(contact_sheet_path), contact_sheet):
        raise RuntimeError(f"failed to write {contact_sheet_path}")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_path),
        "contact_sheet": str(contact_sheet_path),
        "captures": captures,
        "skipped": [target.to_dict() for target in plan.skipped],
    }
    manifest_path = output_path / "capture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture GMSL camera evidence frames and a labeled contact sheet.")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--positions", default=None, help="Comma-separated mount positions. Defaults to training order.")
    parser.add_argument("--require-all", action="store_true", help="Fail if any requested camera is still pending.")
    parser.add_argument("--dry-run-plan", action="store_true", help="Print the capture plan without opening cameras.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--settle-frames", type=int, default=3)
    parser.add_argument("--preview-width", type=int, default=480)
    parser.add_argument("--preview-height", type=int, default=384)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_capture_plan(
        args.mapping,
        positions=_split_positions(args.positions),
        require_all=args.require_all,
    )
    if args.dry_run_plan:
        output = plan.to_dict()
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print("capture targets:")
            for target in plan.targets:
                print(f"  {target.mount_position}: {target.camera_key} {target.device} {target.orientation}")
            for target in plan.skipped:
                print(f"  skipped {target.mount_position}: {target.reason}")
        return 0

    output_dir = args.output_dir or Path("artifacts") / "gmsl_contact_sheet" / datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = capture_contact_sheet(
        plan=plan,
        output_dir=output_dir,
        width=args.width,
        height=args.height,
        frames=args.frames,
        settle_frames=args.settle_frames,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
    )
    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"contact sheet: {manifest['contact_sheet']}")
        print(f"manifest: {Path(manifest['output_dir']) / 'capture_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
