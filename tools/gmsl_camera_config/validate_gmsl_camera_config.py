#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple


DEFAULT_CONFIG_DIR = Path("configs/camera_calibration/gmsl_h190ta_four_camera")
VALID_ORIENTATIONS = {"normal", "rotate_180"}


class ValidationReport(NamedTuple):
    ok: bool
    errors: list[str]
    warnings: list[str]
    available_cameras: list[str]
    pending_cameras: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "available_cameras": self.available_cameras,
            "pending_cameras": self.pending_cameras,
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_path(repo_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _index_by(items: list[dict[str, Any]], key: str, errors: list[str], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"{label} entry missing non-empty {key}")
            continue
        if value in indexed:
            errors.append(f"{label} has duplicate {key}: {value}")
            continue
        indexed[value] = item
    return indexed


def _compare_field(
    *,
    errors: list[str],
    mount_position: str,
    source_label: str,
    source: dict[str, Any],
    expected: dict[str, Any],
    field: str,
) -> None:
    if source.get(field) != expected.get(field):
        errors.append(
            f"{mount_position}: {source_label}.{field}={source.get(field)!r} "
            f"does not match mapping.{field}={expected.get(field)!r}"
        )


def validate_config(
    *,
    repo_root: str | Path,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> ValidationReport:
    root = Path(repo_root)
    config_root = _repo_path(root, config_dir)
    mapping_path = config_root / "camera_mount_mapping.json"
    mapping = _read_json(mapping_path)

    errors: list[str] = []
    warnings: list[str] = []

    intrinsics_manifest_path = _repo_path(root, mapping["intrinsics_manifest"])
    preprocess_path = _repo_path(root, mapping["preprocess_manifest"])
    extrinsics_path = _repo_path(root, mapping["extrinsics_template"])
    intrinsics_manifest = _read_json(intrinsics_manifest_path)
    preprocess = _read_json(preprocess_path)
    extrinsics = _read_json(extrinsics_path)

    mounts = mapping.get("mounts", [])
    if not isinstance(mounts, list):
        errors.append("camera_mount_mapping.json field mounts must be a list")
        mounts = []
    mounts_by_position = _index_by(mounts, "mount_position", errors, "mapping.mounts")
    mounts_by_camera_key = _index_by(mounts, "camera_key", errors, "mapping.mounts")

    position_order = mapping.get("position_order", [])
    training_order = mapping.get("training_camera_order", [])
    if position_order != list(mounts_by_position):
        errors.append(
            "mapping.position_order must match mount entries in order: "
            f"position_order={position_order!r} mounts={list(mounts_by_position)!r}"
        )
    if training_order != position_order:
        errors.append(
            "mapping.training_camera_order must match position_order for current ACT input ordering: "
            f"training_camera_order={training_order!r} position_order={position_order!r}"
        )

    preprocess_cameras = _index_by(preprocess.get("cameras", []), "mount_position", errors, "preprocess.cameras")
    extrinsics_cameras = _index_by(extrinsics.get("cameras", []), "mount_position", errors, "extrinsics.cameras")
    intrinsics_cameras = _index_by(intrinsics_manifest.get("cameras", []), "camera_key", errors, "intrinsics.cameras")

    expected_camera_order = [
        mounts_by_position[position]["camera_key"]
        for position in training_order
        if position in mounts_by_position
    ]
    if preprocess.get("position_order") != training_order:
        errors.append(
            "preprocess.position_order must match mapping.training_camera_order: "
            f"preprocess={preprocess.get('position_order')!r} mapping={training_order!r}"
        )
    if preprocess.get("camera_order") != expected_camera_order:
        errors.append(
            "preprocess.camera_order must follow mapping.training_camera_order camera keys: "
            f"preprocess={preprocess.get('camera_order')!r} expected={expected_camera_order!r}"
        )

    available_cameras: list[str] = []
    pending_cameras: list[str] = []
    for mount_position in training_order:
        mount = mounts_by_position.get(mount_position)
        if mount is None:
            errors.append(f"training_camera_order references missing mount: {mount_position}")
            continue

        status = mount.get("intrinsics_status")
        if status == "available":
            available_cameras.append(mount_position)
        elif status == "pending_import":
            pending_cameras.append(mount_position)
            warnings.append(f"{mount_position}: intrinsics are pending import")
        else:
            errors.append(f"{mount_position}: unsupported intrinsics_status={status!r}")

        orientation = mount.get("orientation")
        if status == "available" and orientation not in VALID_ORIENTATIONS:
            errors.append(f"{mount_position}: available camera has invalid orientation={orientation!r}")

        preprocess_camera = preprocess_cameras.get(mount_position)
        if preprocess_camera is None:
            errors.append(f"{mount_position}: missing preprocess camera entry")
        else:
            for field in ("camera_key", "serial", "device_hint", "intrinsics_status", "orientation"):
                _compare_field(
                    errors=errors,
                    mount_position=mount_position,
                    source_label="preprocess",
                    source=preprocess_camera,
                    expected=mount,
                    field=field,
                )

        extrinsics_camera = extrinsics_cameras.get(mount_position)
        if extrinsics_camera is None:
            errors.append(f"{mount_position}: missing extrinsics camera entry")
        else:
            for field in ("camera_key", "serial", "device_hint"):
                _compare_field(
                    errors=errors,
                    mount_position=mount_position,
                    source_label="extrinsics",
                    source=extrinsics_camera,
                    expected=mount,
                    field=field,
                )

        if status != "available":
            continue

        camera_key = mount.get("camera_key")
        intrinsics_camera = intrinsics_cameras.get(camera_key)
        if intrinsics_camera is None:
            errors.append(f"{mount_position}: available camera_key={camera_key!r} missing from intrinsics manifest")
            continue

        for field in ("serial", "device_hint", "intrinsics_file", "orientation"):
            _compare_field(
                errors=errors,
                mount_position=mount_position,
                source_label="intrinsics",
                source=intrinsics_camera,
                expected=mount,
                field=field,
            )

        intrinsics_file = intrinsics_manifest_path.parent / str(mount.get("intrinsics_file"))
        if not intrinsics_file.is_file():
            errors.append(f"{mount_position}: intrinsics file does not exist: {intrinsics_file}")

    unexpected_preprocess = sorted(set(preprocess_cameras) - set(mounts_by_position))
    unexpected_extrinsics = sorted(set(extrinsics_cameras) - set(mounts_by_position))
    unexpected_intrinsics = sorted(set(intrinsics_cameras) - set(mounts_by_camera_key))
    if unexpected_preprocess:
        errors.append(f"preprocess has mount positions not in mapping: {unexpected_preprocess}")
    if unexpected_extrinsics:
        errors.append(f"extrinsics has mount positions not in mapping: {unexpected_extrinsics}")
    if unexpected_intrinsics:
        warnings.append(f"intrinsics manifest has camera keys not in mount mapping: {unexpected_intrinsics}")

    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        available_cameras=available_cameras,
        pending_cameras=pending_cameras,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the four-camera GMSL mapping/preprocess/extrinsics config.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_config(repo_root=args.repo_root, config_dir=args.config_dir)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        state = "OK" if report.ok else "FAILED"
        print(f"GMSL camera config validation: {state}")
        if report.available_cameras:
            print("available: " + ", ".join(report.available_cameras))
        if report.pending_cameras:
            print("pending: " + ", ".join(report.pending_cameras))
        for warning in report.warnings:
            print(f"warning: {warning}")
        for error in report.errors:
            print(f"error: {error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
