from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INTRINSICS_DIR = REPO_ROOT / "configs" / "camera_intrinsics" / "gmsl_h190ta"
CALIB_DIR = REPO_ROOT / "configs" / "camera_calibration" / "gmsl_h190ta_four_camera"
MOUNT_MAPPING = CALIB_DIR / "camera_mount_mapping.json"
PREPROCESS_MANIFEST = CALIB_DIR / "preprocess_manifest.json"
EXTRINSICS_TEMPLATE = CALIB_DIR / "extrinsics_template.json"
BOARD_SVG = REPO_ROOT / "docs" / "assets" / "gmsl_chessboard_8x6_25mm_a4.svg"
GUIDE = REPO_ROOT / "docs" / "gmsl_four_camera_calibration_guide_20260630.md"
IMPORT_TOOL = REPO_ROOT / "tools" / "gmsl_camera_config" / "import_h190ta_intrinsics.py"
STEREO_CAPTURE_TOOL = REPO_ROOT / "tools" / "gmsl_camera_config" / "capture_gmsl_stereo_pairs.py"
STEREO_CALIBRATE_TOOL = REPO_ROOT / "tools" / "gmsl_camera_config" / "calibrate_gmsl_stereo_pair.py"


def test_four_camera_preprocess_manifest_tracks_110hfov_policy() -> None:
    manifest = json.loads(PREPROCESS_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["version"] == "gmsl_h190ta_110hfov_virtual_rectilinear_v1"
    assert manifest["intrinsics_manifest"] == "configs/camera_intrinsics/gmsl_h190ta/manifest.json"
    assert manifest["output"]["width"] == 384
    assert manifest["output"]["height"] == 216
    assert manifest["default_transform"]["projection"] == "virtual_rectilinear"
    assert manifest["default_transform"]["hfov_deg"] == 110.0

    cameras = {camera["camera_key"]: camera for camera in manifest["cameras"]}
    assert len(cameras) == 4
    assert cameras["video6"]["intrinsics_status"] == "available"
    assert cameras["video6"]["mount_position"] == "stick_bottom"
    assert cameras["video6"]["transform"]["pitch_down_deg"] == 20.0
    assert cameras["video7"]["intrinsics_status"] == "available"
    assert cameras["video7"]["mount_position"] == "stick_top"
    assert cameras["video7"]["transform"]["pitch_down_deg"] == 0.0
    assert cameras["video4"]["intrinsics_status"] == "available"
    assert cameras["video4"]["mount_position"] == "eye_left"
    assert cameras["video4"]["serial"] == "H190TA-I06031461"
    assert cameras["video4"]["orientation"] == "normal"
    assert cameras["video4"]["transform"]["pitch_down_deg"] == 10.0
    assert cameras["video5"]["intrinsics_status"] == "available"
    assert cameras["video5"]["mount_position"] == "eye_right"
    assert cameras["video5"]["serial"] == "H190TA-I06031462"
    assert cameras["video5"]["orientation"] == "normal"
    assert cameras["video5"]["transform"]["pitch_down_deg"] == 10.0

    pending = [camera for camera in cameras.values() if camera["intrinsics_status"] == "pending_import"]
    assert pending == []
    for camera in cameras.values():
        assert camera["model"] == "H190TA"
        assert camera["transform"]["hfov_deg"] == 110.0
        assert camera["extrinsics_status"] == "pending_calibration"


def test_camera_mount_mapping_tracks_physical_positions_and_serials() -> None:
    mapping = json.loads(MOUNT_MAPPING.read_text(encoding="utf-8"))

    assert mapping["version"] == "gmsl_h190ta_mount_mapping_20260630"
    assert mapping["intrinsics_manifest"] == "configs/camera_intrinsics/gmsl_h190ta/manifest.json"
    assert mapping["preprocess_manifest"] == (
        "configs/camera_calibration/gmsl_h190ta_four_camera/preprocess_manifest.json"
    )
    assert mapping["position_order"] == ["stick_top", "stick_bottom", "eye_left", "eye_right"]

    by_position = {entry["mount_position"]: entry for entry in mapping["mounts"]}
    assert set(by_position) == {"stick_top", "stick_bottom", "eye_left", "eye_right"}

    assert by_position["stick_top"]["serial"] == "H190TA-I06031460"
    assert by_position["stick_top"]["camera_key"] == "video7"
    assert by_position["stick_top"]["intrinsics_file"] == "H190TA-I06031460.txt"
    assert by_position["stick_top"]["intrinsics_status"] == "available"
    assert by_position["stick_top"]["orientation"] == "normal"

    assert by_position["stick_bottom"]["serial"] == "H190TA-I06031459"
    assert by_position["stick_bottom"]["camera_key"] == "video6"
    assert by_position["stick_bottom"]["intrinsics_file"] == "H190TA-I06031459.txt"
    assert by_position["stick_bottom"]["intrinsics_status"] == "available"

    assert by_position["eye_left"]["serial"] == "H190TA-I06031461"
    assert by_position["eye_left"]["camera_key"] == "video4"
    assert by_position["eye_left"]["device_hint"] == "/dev/video4"
    assert by_position["eye_left"]["intrinsics_file"] == "H190TA-I06031461.txt"
    assert by_position["eye_left"]["intrinsics_status"] == "available"
    assert by_position["eye_left"]["orientation"] == "normal"

    assert by_position["eye_right"]["serial"] == "H190TA-I06031462"
    assert by_position["eye_right"]["camera_key"] == "video5"
    assert by_position["eye_right"]["device_hint"] == "/dev/video5"
    assert by_position["eye_right"]["intrinsics_file"] == "H190TA-I06031462.txt"
    assert by_position["eye_right"]["intrinsics_status"] == "available"
    assert by_position["eye_right"]["orientation"] == "normal"

    assert mapping["training_camera_order"] == ["stick_top", "stick_bottom", "eye_left", "eye_right"]


def test_extrinsics_template_and_printable_board_contract() -> None:
    extrinsics = json.loads(EXTRINSICS_TEMPLATE.read_text(encoding="utf-8"))
    board_svg = BOARD_SVG.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")

    assert extrinsics["status"] == "template_pending_field_calibration"
    assert extrinsics["calibration_target"]["inner_corners"] == [8, 6]
    assert extrinsics["calibration_target"]["square_size_mm"] == 25.0
    assert extrinsics["calibration_target"]["target_svg"] == "docs/assets/gmsl_chessboard_8x6_25mm_a4.svg"
    assert len(extrinsics["cameras"]) == 4
    assert all(camera["extrinsics_status"] == "pending_calibration" for camera in extrinsics["cameras"])
    assert {camera["mount_position"] for camera in extrinsics["cameras"]} == {
        "stick_top",
        "stick_bottom",
        "eye_left",
        "eye_right",
    }
    stereo_pairs = {pair["pair_name"]: pair for pair in extrinsics["stereo_pairs"]}
    assert stereo_pairs["eye_left_eye_right"]["left_camera_key"] == "video4"
    assert stereo_pairs["eye_left_eye_right"]["right_camera_key"] == "video5"
    assert stereo_pairs["eye_left_eye_right"]["right_T_left"] is None
    assert stereo_pairs["eye_left_eye_right"]["opencv_stereo_convention"] == "X_right = R * X_left + T"
    assert stereo_pairs["stick_bottom_stick_top"]["left_camera_key"] == "video6"
    assert stereo_pairs["stick_bottom_stick_top"]["right_camera_key"] == "video7"
    assert stereo_pairs["stick_bottom_stick_top"]["right_T_left"] is None

    assert 'width="297mm"' in board_svg
    assert 'height="210mm"' in board_svg
    assert "inner_corners_cols=8" in board_svg
    assert "square_size_mm=25" in board_svg
    assert guide.count("docs/assets/gmsl_chessboard_8x6_25mm_a4.svg") >= 1
    assert "8x6" in guide
    assert "25 mm" in guide
    assert "不要缩放" in guide


def test_pairwise_stereo_calibration_tools_and_workflow_contract() -> None:
    capture_text = STEREO_CAPTURE_TOOL.read_text(encoding="utf-8")
    calibrate_text = STEREO_CALIBRATE_TOOL.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")

    assert STEREO_CAPTURE_TOOL.exists()
    assert STEREO_CALIBRATE_TOOL.exists()
    assert "pairs.json" in capture_text
    assert "right_T_left" in calibrate_text
    assert "X_right = R * X_left + T" in calibrate_text
    assert "cv2.fisheye.CALIB_FIX_INTRINSIC" in calibrate_text
    assert "video4=/dev/video4" in guide
    assert "video5=/dev/video5" in guide
    assert "video6=/dev/video6" in guide
    assert "video7=/dev/video7" in guide
    assert "video4_video5/pairs.json" in guide
    assert "video6_video7/pairs.json" in guide
    assert "video5_T_video4" in guide
    assert "video7_T_video6" in guide

    spec = importlib.util.spec_from_file_location("calibrate_gmsl_stereo_pair", STEREO_CALIBRATE_TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.parse_inner_corners("8x6") == (8, 6)


def test_import_h190ta_intrinsics_tool_updates_manifest_copy(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("import_h190ta_intrinsics", IMPORT_TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest_path = tmp_path / "manifest.json"
    shutil.copy(INTRINSICS_DIR / "manifest.json", manifest_path)
    input_intrinsics = INTRINSICS_DIR / "H190TA-I06031459.txt"

    updated = module.import_intrinsics(
        manifest_path=manifest_path,
        intrinsics_file=input_intrinsics,
        camera_key="video0",
        device_hint="/dev/video0",
        serial="H190TA-I06031459-CLONE",
        orientation="normal",
        copy_intrinsics=True,
        output_path=manifest_path,
    )

    cameras = {camera["camera_key"]: camera for camera in updated["cameras"]}
    assert "video0" in cameras
    assert cameras["video0"]["device_hint"] == "/dev/video0"
    assert cameras["video0"]["serial"] == "H190TA-I06031459-CLONE"
    assert cameras["video0"]["intrinsics_file"] == "H190TA-I06031459.txt"
    assert cameras["video0"]["orientation"] == "normal"
    assert cameras["video0"]["K"][0][0] == 564.52909345
    assert cameras["video0"]["D"] == [0.0339082869, -0.0158314506, -0.0000680984, 0.0004041576]
