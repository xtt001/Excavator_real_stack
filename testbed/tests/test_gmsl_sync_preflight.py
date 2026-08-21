from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_gmsl_sync.py"
SPEC = importlib.util.spec_from_file_location("check_gmsl_sync", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sync_gate_accepts_coherent_advancing_groups() -> None:
    rows = [
        {"group_id": index, "valid": True, "skew_ms": 0.04}
        for index in range(1, 41)
    ]
    report = MODULE.evaluate_samples(
        rows,
        min_valid_fraction=0.98,
        max_skew_ms=5.0,
        min_distinct_groups=30,
    )
    assert report["status"] == "PASS"


def test_sync_gate_rejects_episode2_style_video7_phase_error() -> None:
    rows = [
        {"group_id": index, "valid": index > 34, "skew_ms": 8.0}
        for index in range(1, 41)
    ]
    report = MODULE.evaluate_samples(
        rows,
        min_valid_fraction=0.98,
        max_skew_ms=5.0,
        min_distinct_groups=30,
    )
    assert report["status"] == "FAIL"


def test_camera_group_sample_requires_one_four_camera_group() -> None:
    payload = {
        "images": {
            camera: {
                "metadata": {
                    "group_id": 123,
                    "group_valid": 1,
                    "group_camera_count": 4,
                    "group_skew_ms": 0.03,
                    "v4l2_timestamp_ns": 1000 + index,
                    "v4l2_error": 0,
                }
            }
            for index, camera in enumerate(MODULE.EXPECTED_CAMERAS)
        }
    }
    sample = MODULE.camera_group_sample(payload)
    assert sample["valid"] is True
    assert sample["group_id"] == 123
