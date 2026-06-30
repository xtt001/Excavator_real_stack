#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


RAW_VALUE_PATTERN = re.compile(r"^([A-Za-z0-9_]+)=\s*([^[]+?)(?:\s|$)")
VALID_ORIENTATIONS = {"normal", "rotate_180"}


def parse_vendor_intrinsics(path: str | Path) -> dict[str, float | int | str]:
    intrinsics_path = Path(path)
    values: dict[str, float | int | str] = {}
    for line in intrinsics_path.read_text(encoding="utf-8").splitlines():
        match = RAW_VALUE_PATTERN.match(line.strip())
        if match is None:
            continue
        key, raw_value = match.groups()
        raw_value = raw_value.strip()
        if key == "type":
            values[key] = raw_value
        elif key in {"imageWidth", "imageHeight"}:
            values[key] = int(raw_value)
        else:
            values[key] = float(raw_value)

    required = {"type", "imageWidth", "imageHeight", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"{intrinsics_path} missing required keys: {', '.join(missing)}")
    if values["type"] != "fisheye":
        raise ValueError(f"{intrinsics_path} type must be fisheye, got {values['type']!r}")
    return values


def build_camera_entry(
    *,
    camera_key: str,
    device_hint: str,
    serial: str,
    intrinsics_file_name: str,
    orientation: str,
    raw: dict[str, float | int | str],
) -> dict[str, Any]:
    if orientation not in VALID_ORIENTATIONS:
        raise ValueError(f"orientation must be one of {sorted(VALID_ORIENTATIONS)}, got {orientation!r}")

    return {
        "camera_key": camera_key,
        "device_hint": device_hint,
        "serial": serial,
        "intrinsics_file": intrinsics_file_name,
        "orientation": orientation,
        "upside_down": orientation == "rotate_180",
        "K": [
            [float(raw["fx"]), 0.0, float(raw["cx"])],
            [0.0, float(raw["fy"]), float(raw["cy"])],
            [0.0, 0.0, 1.0],
        ],
        "D": [float(raw["k1"]), float(raw["k2"]), float(raw["k3"]), float(raw["k4"])],
    }


def import_intrinsics(
    *,
    manifest_path: str | Path,
    intrinsics_file: str | Path,
    camera_key: str,
    device_hint: str,
    serial: str,
    orientation: str,
    copy_intrinsics: bool,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    output_file = Path(output_path) if output_path is not None else manifest_file
    raw_file = Path(intrinsics_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    raw = parse_vendor_intrinsics(raw_file)

    if int(raw["imageWidth"]) != int(manifest["image_width"]):
        raise ValueError(f"imageWidth mismatch: raw={raw['imageWidth']} manifest={manifest['image_width']}")
    if int(raw["imageHeight"]) != int(manifest["image_height"]):
        raise ValueError(f"imageHeight mismatch: raw={raw['imageHeight']} manifest={manifest['image_height']}")

    cameras = manifest.setdefault("cameras", [])
    if any(camera.get("camera_key") == camera_key for camera in cameras):
        raise ValueError(f"camera_key already exists in manifest: {camera_key}")

    intrinsics_file_name = raw_file.name
    if copy_intrinsics:
        destination = output_file.parent / intrinsics_file_name
        if raw_file.resolve() != destination.resolve():
            shutil.copy2(raw_file, destination)

    cameras.append(
        build_camera_entry(
            camera_key=camera_key,
            device_hint=device_hint,
            serial=serial,
            intrinsics_file_name=intrinsics_file_name,
            orientation=orientation,
            raw=raw,
        )
    )
    output_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import one H190TA vendor intrinsics txt into the GMSL manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("configs/camera_intrinsics/gmsl_h190ta/manifest.json"))
    parser.add_argument("--intrinsics-file", type=Path, required=True)
    parser.add_argument("--camera", required=True, help="Camera key to add, for example video0.")
    parser.add_argument("--device", required=True, help="Device hint, for example /dev/video0.")
    parser.add_argument("--serial", required=True, help="Camera serial, for example H190TA-I06031461.")
    parser.add_argument("--orientation", choices=sorted(VALID_ORIENTATIONS), default="normal")
    parser.add_argument("--output", type=Path, default=None, help="Output manifest path. Defaults to --manifest.")
    parser.add_argument(
        "--no-copy-intrinsics",
        action="store_true",
        help="Do not copy the raw txt next to the output manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    updated = import_intrinsics(
        manifest_path=args.manifest,
        intrinsics_file=args.intrinsics_file,
        camera_key=args.camera,
        device_hint=args.device,
        serial=args.serial,
        orientation=args.orientation,
        copy_intrinsics=not args.no_copy_intrinsics,
        output_path=args.output,
    )
    print(f"updated {args.output or args.manifest}: {len(updated['cameras'])} cameras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
