from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = REPO_ROOT / "tools" / "gmsl_latency_benchmark"
SOURCE = TOOL_DIR / "src" / "gmsl_latency_benchmark.cpp"
README = TOOL_DIR / "README.md"
GUIDE = REPO_ROOT / "docs" / "gmsl_latency_benchmark_guide_20260630.md"
CMAKE = TOOL_DIR / "CMakeLists.txt"


def test_gmsl_latency_benchmark_files_exist() -> None:
    assert SOURCE.is_file()
    assert README.is_file()
    assert GUIDE.is_file()
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
