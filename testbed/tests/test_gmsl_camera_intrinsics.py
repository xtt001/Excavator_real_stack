from __future__ import annotations

import json
import math
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INTRINSICS_DIR = REPO_ROOT / "configs" / "camera_intrinsics" / "gmsl_h190ta"


def _parse_intrinsics_txt(path: Path) -> dict[str, float | int | str]:
    values: dict[str, float | int | str] = {}
    pattern = re.compile(r"^([A-Za-z0-9_]+)=\s*([^[]+?)(?:\s|$)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        key, raw_value = match.groups()
        raw_value = raw_value.strip()
        if key in {"type"}:
            values[key] = raw_value
        elif key in {"imageWidth", "imageHeight"}:
            values[key] = int(raw_value)
        else:
            values[key] = float(raw_value)
    return values


def _assert_close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)


def test_gmsl_h190ta_manifest_matches_raw_intrinsics() -> None:
    manifest = json.loads((INTRINSICS_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["distortion_model"] == "opencv_fisheye"
    assert manifest["image_width"] == 1920
    assert manifest["image_height"] == 1536

    cameras = {camera["camera_key"]: camera for camera in manifest["cameras"]}
    assert set(cameras) == {"video4", "video5", "video6", "video7"}

    expected = {
        "video4": {
            "serial": "H190TA-I06031461",
            "orientation": "normal",
            "upside_down": False,
        },
        "video5": {
            "serial": "H190TA-I06031462",
            "orientation": "normal",
            "upside_down": False,
        },
        "video6": {
            "serial": "H190TA-I06031459",
            "orientation": "normal",
            "upside_down": False,
        },
        "video7": {
            "serial": "H190TA-I06031460",
            "orientation": "normal",
            "upside_down": False,
        },
    }
    for camera_key, expected_camera in expected.items():
        camera = cameras[camera_key]
        assert camera["serial"] == expected_camera["serial"]
        assert camera["orientation"] == expected_camera["orientation"]
        assert camera["upside_down"] is expected_camera["upside_down"]

        raw = _parse_intrinsics_txt(INTRINSICS_DIR / camera["intrinsics_file"])
        assert raw["type"] == "fisheye"
        assert raw["imageWidth"] == manifest["image_width"]
        assert raw["imageHeight"] == manifest["image_height"]

        expected_k = [
            [raw["fx"], 0.0, raw["cx"]],
            [0.0, raw["fy"], raw["cy"]],
            [0.0, 0.0, 1.0],
        ]
        expected_d = [raw["k1"], raw["k2"], raw["k3"], raw["k4"]]

        for actual_row, expected_row in zip(camera["K"], expected_k, strict=True):
            for actual, expected_value in zip(actual_row, expected_row, strict=True):
                _assert_close(float(actual), float(expected_value))
        for actual, expected_value in zip(camera["D"], expected_d, strict=True):
            _assert_close(float(actual), float(expected_value))
