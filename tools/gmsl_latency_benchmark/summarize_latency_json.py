#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


METRICS = (
    "read_ms",
    "color_ms",
    "upload_ms",
    "remap_ms",
    "download_ms",
    "rotate_ms",
    "process_ms",
    "frame_ms",
)
QUANTILES = ("p50", "p95", "p99")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _metric_value(camera: dict[str, Any], metric: str, quantile: str) -> float | None:
    raw = camera.get("metrics", {}).get(metric, {}).get(quantile)
    if raw is None:
        return None
    return float(raw)


def _decision(
    *,
    report: dict[str, Any],
    camera: dict[str, Any],
    process_p95_budget_ms: float,
    frame_p95_budget_ms: float | None,
) -> tuple[str, str, float | None]:
    error = str(camera.get("error", ""))
    read_failures = int(camera.get("read_failures", 0))
    if error:
        return "ERROR", "error", None
    if read_failures > 0:
        return "WARN", "read_failures", float(read_failures)

    capture_only = bool(report.get("capture_only")) or bool(camera.get("raw_only"))
    if capture_only:
        value = _metric_value(camera, "frame_ms", "p95")
        if frame_p95_budget_ms is None or value is None:
            return "INFO", "frame_ms_p95", value
        return ("PASS" if value <= frame_p95_budget_ms else "FAIL", "frame_ms_p95", value)

    value = _metric_value(camera, "process_ms", "p95")
    if value is None:
        return "ERROR", "missing_process_ms_p95", None
    return ("PASS" if value <= process_p95_budget_ms else "FAIL", "process_ms_p95", value)


def rows_from_reports(
    input_paths: list[Path],
    *,
    process_p95_budget_ms: float,
    frame_p95_budget_ms: float | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for input_path in input_paths:
        report = json.loads(input_path.read_text(encoding="utf-8"))
        for camera in report.get("cameras", []):
            decision, decision_metric, decision_value = _decision(
                report=report,
                camera=camera,
                process_p95_budget_ms=process_p95_budget_ms,
                frame_p95_budget_ms=frame_p95_budget_ms,
            )
            row: dict[str, str] = {
                "report": str(input_path),
                "capture_only": str(bool(report.get("capture_only"))).lower(),
                "prefer_gpu": str(bool(report.get("prefer_gpu"))).lower(),
                "camera_key": str(camera.get("camera_key", "")),
                "device": str(camera.get("device", "")),
                "serial": str(camera.get("serial", "")),
                "used_gpu": str(bool(camera.get("used_gpu"))).lower(),
                "measured_frames": str(camera.get("measured_frames", "")),
                "read_failures": str(camera.get("read_failures", "")),
                "decision": decision,
                "decision_metric": decision_metric,
                "decision_value_ms": _fmt(decision_value),
            }
            for metric in METRICS:
                for quantile in QUANTILES:
                    row[f"{metric}_{quantile}"] = _fmt(_metric_value(camera, metric, quantile))
            rows.append(row)
    return rows


def csv_fieldnames() -> list[str]:
    fields = [
        "report",
        "capture_only",
        "prefer_gpu",
        "camera_key",
        "device",
        "serial",
        "used_gpu",
        "measured_frames",
        "read_failures",
        "decision",
        "decision_metric",
        "decision_value_ms",
    ]
    for metric in METRICS:
        for quantile in QUANTILES:
            fields.append(f"{metric}_{quantile}")
    return fields


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=csv_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    columns = [
        ("report", "report"),
        ("camera_key", "camera"),
        ("device", "device"),
        ("used_gpu", "gpu"),
        ("decision", "decision"),
        ("read_ms_p95", "capture/read p95"),
        ("upload_ms_p95", "upload_ms p95"),
        ("remap_ms_p95", "remap_ms p95"),
        ("download_ms_p95", "download_ms p95"),
        ("process_ms_p95", "process_ms p95"),
        ("frame_ms_p95", "frame_ms p95"),
        ("frame_ms_p99", "frame_ms p99"),
    ]
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for field, _ in columns:
            value = row.get(field, "")
            if field == "report":
                value = Path(value).name
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_markdown(
    rows: list[dict[str, str]],
    *,
    process_p95_budget_ms: float,
    frame_p95_budget_ms: float | None,
) -> str:
    decisions = {"PASS": 0, "FAIL": 0, "WARN": 0, "ERROR": 0, "INFO": 0}
    for row in rows:
        decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
    frame_budget = f"{frame_p95_budget_ms:.3f} ms" if frame_p95_budget_ms is not None else "not set"

    lines = [
        "# GMSL Latency Summary",
        "",
        f"- process_ms p95 budget: {process_p95_budget_ms:.3f} ms",
        f"- frame_ms p95 budget: {frame_budget}",
        f"- rows: {len(rows)}",
        "- decisions: "
        + ", ".join(f"{name}={count}" for name, count in decisions.items() if count > 0),
        "",
        markdown_table(rows),
        "",
        "## Decision Notes",
        "",
        "- FAIL on non-capture benchmarks means `process_ms p95` exceeds the budget.",
        "- INFO on capture-only benchmarks means no `frame_ms p95` budget was supplied.",
        "- Large `upload_ms` or `download_ms` points toward zero-copy/NVMM or fused CUDA preprocessing.",
        "- Large `read_ms` or `frame_ms` in capture-only runs points toward input-path or driver queue work.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one or more GMSL latency benchmark JSON reports.")
    parser.add_argument("inputs", type=Path, nargs="+", help="Benchmark JSON files from gmsl_latency_benchmark.")
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--process-p95-budget-ms", type=float, default=4.0)
    parser.add_argument("--frame-p95-budget-ms", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = rows_from_reports(
        args.inputs,
        process_p95_budget_ms=args.process_p95_budget_ms,
        frame_p95_budget_ms=args.frame_p95_budget_ms,
    )
    markdown = render_markdown(
        rows,
        process_p95_budget_ms=args.process_p95_budget_ms,
        frame_p95_budget_ms=args.frame_p95_budget_ms,
    )
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    if args.output_csv is not None:
        write_csv(rows, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
