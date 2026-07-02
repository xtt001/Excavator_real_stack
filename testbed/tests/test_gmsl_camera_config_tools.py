from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = REPO_ROOT / "configs" / "camera_calibration" / "gmsl_h190ta_four_camera"
INTRINSICS_DIR = REPO_ROOT / "configs" / "camera_intrinsics" / "gmsl_h190ta"
VALIDATE_TOOL = REPO_ROOT / "tools" / "gmsl_camera_config" / "validate_gmsl_camera_config.py"
CAPTURE_TOOL = REPO_ROOT / "tools" / "gmsl_camera_config" / "capture_gmsl_contact_sheet.py"
STEREO_CAPTURE_TOOL = REPO_ROOT / "tools" / "gmsl_camera_config" / "capture_gmsl_stereo_pairs.py"
STEREO_CALIBRATE_TOOL = REPO_ROOT / "tools" / "gmsl_camera_config" / "calibrate_gmsl_stereo_pair.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
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
    assert report.available_cameras == ["stick_top", "stick_bottom", "eye_left", "eye_right"]
    assert report.pending_cameras == []


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


def test_capture_plan_uses_training_order_for_all_available_cameras() -> None:
    module = _load_module(CAPTURE_TOOL, "capture_gmsl_contact_sheet")

    plan = module.build_capture_plan(CALIB_DIR / "camera_mount_mapping.json")

    assert [target.mount_position for target in plan.targets] == [
        "stick_top",
        "stick_bottom",
        "eye_left",
        "eye_right",
    ]
    assert [target.device for target in plan.targets] == ["/dev/video7", "/dev/video6", "/dev/video4", "/dev/video5"]
    assert [target.orientation for target in plan.targets] == ["normal", "normal", "normal", "normal"]
    assert plan.skipped == []


def test_capture_plan_require_all_accepts_current_contract() -> None:
    module = _load_module(CAPTURE_TOOL, "capture_gmsl_contact_sheet")

    plan = module.build_capture_plan(CALIB_DIR / "camera_mount_mapping.json", require_all=True)

    assert [target.mount_position for target in plan.targets] == [
        "stick_top",
        "stick_bottom",
        "eye_left",
        "eye_right",
    ]
    assert plan.skipped == []


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

    assert [target["mount_position"] for target in plan["targets"]] == [
        "stick_top",
        "stick_bottom",
        "eye_left",
        "eye_right",
    ]
    assert [target["camera_key"] for target in plan["targets"]] == ["video7", "video6", "video4", "video5"]
    assert plan["skipped"] == []


def test_stereo_capture_tool_parses_camera_specs() -> None:
    module = _load_module(STEREO_CAPTURE_TOOL, "capture_gmsl_stereo_pairs")

    spec = module.parse_camera_spec("video4=/dev/video4")

    assert spec.camera_key == "video4"
    assert spec.device == "/dev/video4"
    with pytest.raises(ValueError):
        module.parse_camera_spec("video4:/dev/video4")
    with pytest.raises(ValueError):
        module.parse_camera_spec("video4=/tmp/video4")


def test_stereo_calibrate_tool_collects_matching_image_stems(tmp_path: Path) -> None:
    module = _load_module(STEREO_CALIBRATE_TOOL, "calibrate_gmsl_stereo_pair")
    left_dir = tmp_path / "video4"
    right_dir = tmp_path / "video5"
    left_dir.mkdir()
    right_dir.mkdir()
    for stem in ["000000", "000001", "000002"]:
        (left_dir / f"{stem}.png").write_bytes(b"")
        (right_dir / f"{stem}.png").write_bytes(b"")

    pairs = module.collect_image_pairs(left_dir=left_dir, right_dir=right_dir, pattern="*.png")

    assert [pair.pair_id for pair in pairs] == ["000000", "000001", "000002"]
    assert pairs[0].left_path == left_dir / "000000.png"
    assert pairs[0].right_path == right_dir / "000000.png"
    assert module.parse_inner_corners("8x6") == (8, 6)
