from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = REPO_ROOT / "tools" / "gmsl_latency_benchmark"
SOURCE = TOOL_DIR / "src" / "gmsl_latency_benchmark.cpp"
SUMMARY_TOOL = TOOL_DIR / "summarize_latency_json.py"
README = TOOL_DIR / "README.md"
GUIDE = REPO_ROOT / "docs" / "gmsl_latency_benchmark_guide_20260630.md"
OPTIMIZATION_PLAN = REPO_ROOT / "docs" / "gmsl_realtime_preprocessing_optimization_plan_20260630.md"
CMAKE = TOOL_DIR / "CMakeLists.txt"


def test_gmsl_latency_benchmark_files_exist() -> None:
    assert SOURCE.is_file()
    assert SUMMARY_TOOL.is_file()
    assert README.is_file()
    assert GUIDE.is_file()
    assert OPTIMIZATION_PLAN.is_file()
    assert CMAKE.is_file()


def test_gmsl_latency_benchmark_cli_contract() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    for flag in (
        "--camera",
        "--raw-camera",
        "--frames",
        "--warmup",
        "--output-json",
        "--capture-only",
        "--cpu",
        "--no-rotate",
    ):
        assert flag in source

    for metric in (
        "read_ms",
        "color_ms",
        "upload_ms",
        "remap_ms",
        "download_ms",
        "rotate_ms",
        "process_ms",
        "frame_ms",
        "p50",
        "p95",
        "p99",
        "max",
    ):
        assert metric in source


def test_gmsl_latency_benchmark_docs_include_field_runbook() -> None:
    readme = README.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")

    assert "cmake -S tools/gmsl_latency_benchmark" in readme
    assert "--camera video6=/dev/video6" in readme
    assert "--camera video7=/dev/video7" in readme
    assert "--output-json artifacts/gmsl_latency" in readme

    assert "GMSL_PRINT_CONFIG_ONLY=1" in guide
    assert "GMSL_VIDEO_DEVICES=\"6 7\"" in guide
    assert "capture-only" in guide
    assert "p50/p95/p99/max" in guide
    assert "summarize_latency_json.py" in guide

    assert "GStreamer/NVMM" in OPTIMIZATION_PLAN.read_text(encoding="utf-8")
    assert "fused CUDA" in OPTIMIZATION_PLAN.read_text(encoding="utf-8")


def test_gmsl_latency_summary_tool_writes_markdown_and_csv(tmp_path: Path) -> None:
    report_path = tmp_path / "opencv_remap.json"
    markdown_path = tmp_path / "summary.md"
    csv_path = tmp_path / "summary.csv"
    report_path.write_text(
        json.dumps(
            {
                "version": "gmsl_latency_benchmark_20260630",
                "capture_only": False,
                "prefer_gpu": True,
                "frames": 300,
                "warmup": 30,
                "cameras": [
                    {
                        "camera_key": "video7",
                        "device": "/dev/video7",
                        "serial": "H190TA-I06031460",
                        "raw_only": False,
                        "used_gpu": True,
                        "measured_frames": 300,
                        "read_failures": 0,
                        "error": "",
                        "metrics": {
                            "read_ms": {"p50": 0.7, "p95": 0.9, "p99": 1.2, "max": 1.4},
                            "upload_ms": {"p50": 0.3, "p95": 0.4, "p99": 0.5, "max": 0.6},
                            "remap_ms": {"p50": 1.0, "p95": 1.3, "p99": 1.6, "max": 1.8},
                            "download_ms": {"p50": 0.4, "p95": 0.6, "p99": 0.7, "max": 0.8},
                            "process_ms": {"p50": 2.2, "p95": 3.5, "p99": 3.8, "max": 4.2},
                            "frame_ms": {"p50": 3.0, "p95": 4.1, "p99": 4.5, "max": 5.0},
                        },
                    },
                    {
                        "camera_key": "video6",
                        "device": "/dev/video6",
                        "serial": "H190TA-I06031459",
                        "raw_only": False,
                        "used_gpu": True,
                        "measured_frames": 300,
                        "read_failures": 0,
                        "error": "",
                        "metrics": {
                            "read_ms": {"p50": 0.8, "p95": 1.0, "p99": 1.3, "max": 1.6},
                            "upload_ms": {"p50": 0.5, "p95": 0.8, "p99": 1.0, "max": 1.2},
                            "remap_ms": {"p50": 1.9, "p95": 2.4, "p99": 2.7, "max": 3.0},
                            "download_ms": {"p50": 0.7, "p95": 1.1, "p99": 1.3, "max": 1.5},
                            "process_ms": {"p50": 4.2, "p95": 5.1, "p99": 5.4, "max": 5.8},
                            "frame_ms": {"p50": 5.0, "p95": 6.2, "p99": 6.6, "max": 7.0},
                        },
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_TOOL),
            str(report_path),
            "--output-markdown",
            str(markdown_path),
            "--output-csv",
            str(csv_path),
            "--process-p95-budget-ms",
            "4.0",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "GMSL Latency Summary" in markdown
    assert "video7" in markdown
    assert "PASS" in markdown
    assert "video6" in markdown
    assert "FAIL" in markdown
    assert "upload_ms p95" in markdown
    assert "download_ms p95" in markdown

    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["camera_key"] == "video7"
    assert rows[0]["decision"] == "PASS"
    assert rows[0]["process_ms_p95"] == "3.500"
    assert rows[1]["camera_key"] == "video6"
    assert rows[1]["decision"] == "FAIL"
    assert rows[1]["download_ms_p99"] == "1.300"
