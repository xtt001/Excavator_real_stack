from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = REPO_ROOT / "configs" / "camera_calibration" / "gmsl_h190ta_four_camera"
INTRINSICS_DIR = REPO_ROOT / "configs" / "camera_intrinsics" / "gmsl_h190ta"
VALIDATE_TOOL = REPO_ROOT / "tools" / "gmsl_camera_config" / "validate_gmsl_camera_config.py"
CAPTURE_TOOL = REPO_ROOT / "tools" / "gmsl_camera_config" / "capture_gmsl_contact_sheet.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_config_tree(tmp_path: Path) -> Path:
    repo_copy = tmp_path / "repo"
    shutil.copytree(CALIB_DIR, repo_copy / "configs" / "camera_calibration" / "gmsl_h190ta_four_camera")
    shutil.copytree(INTRINSICS_DIR, repo_copy / "configs" / "camera_intrinsics" / "gmsl_h190ta")
    return repo_copy


def test_validate_gmsl_camera_config_accepts_current_contract() -> None:
    module = _load_module(VALIDATE_TOOL, "validate_gmsl_camera_config")

    report = module.validate_config(repo_root=REPO_ROOT)

    assert report.ok is True
    assert report.errors == []
    assert report.available_cameras == ["stick_top", "stick_bottom"]
    assert report.pending_cameras == ["eye_left", "eye_right"]


def test_validate_gmsl_camera_config_catches_manifest_mismatch(tmp_path: Path) -> None:
    module = _load_module(VALIDATE_TOOL, "validate_gmsl_camera_config")
    repo_copy = _copy_config_tree(tmp_path)
    preprocess_path = (
        repo_copy / "configs" / "camera_calibration" / "gmsl_h190ta_four_camera" / "preprocess_manifest.json"
    )
    preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
    preprocess["cameras"][0]["serial"] = "H190TA-WRONG"
    preprocess_path.write_text(json.dumps(preprocess, indent=2) + "\n", encoding="utf-8")

    report = module.validate_config(repo_root=repo_copy)

    assert report.ok is False
    assert any("stick_bottom" in error and "serial" in error for error in report.errors)


def test_capture_plan_uses_training_order_and_skips_pending_by_default() -> None:
    module = _load_module(CAPTURE_TOOL, "capture_gmsl_contact_sheet")

    plan = module.build_capture_plan(CALIB_DIR / "camera_mount_mapping.json")

    assert [target.mount_position for target in plan.targets] == ["stick_top", "stick_bottom"]
    assert [target.device for target in plan.targets] == ["/dev/video7", "/dev/video6"]
    assert [target.orientation for target in plan.targets] == ["rotate_180", "normal"]
    assert [skipped.mount_position for skipped in plan.skipped] == ["eye_left", "eye_right"]


def test_capture_plan_require_all_rejects_pending_entries() -> None:
    module = _load_module(CAPTURE_TOOL, "capture_gmsl_contact_sheet")

    try:
        module.build_capture_plan(CALIB_DIR / "camera_mount_mapping.json", require_all=True)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("require_all should reject pending entries")

    assert "eye_left" in message
    assert "eye_right" in message


def test_capture_contact_sheet_dry_run_outputs_json_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CAPTURE_TOOL),
            "--mapping",
            str(CALIB_DIR / "camera_mount_mapping.json"),
            "--dry-run-plan",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    plan = json.loads(result.stdout)

    assert [target["mount_position"] for target in plan["targets"]] == ["stick_top", "stick_bottom"]
    assert [target["camera_key"] for target in plan["targets"]] == ["video7", "video6"]
    assert [skipped["mount_position"] for skipped in plan["skipped"]] == ["eye_left", "eye_right"]
