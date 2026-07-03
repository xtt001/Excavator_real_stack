#!/usr/bin/env python3
"""Test-only summary for policy receiver JSONL logs.

This script is for preflight/shadow/control test logs. It does not command the
machine and does not mark data as training-ready.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


REQUIRED_BUNDLE_FILES = (
    "policy_best.ckpt",
    "dataset_stats.pkl",
    "resolved_config.yaml",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print key metrics from a policy receiver test log."
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run-dir", type=Path, help="Specific receiver test run dir.")
    target.add_argument(
        "--latest",
        type=Path,
        help="Root directory containing receiver test run dirs; use newest run.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Optional policy bundle directory to check before log summary.",
    )
    parser.add_argument(
        "--expect-camera-names",
        type=str,
        default=None,
        help=(
            "Optional comma-separated camera_names contract expected in "
            "bundle/resolved_config.yaml, for example video4,video5,video6,video7."
        ),
    )
    parser.add_argument("--expect-output-mode", type=str, default=None)
    parser.add_argument(
        "--allow-stop-reason",
        action="append",
        default=["complete"],
        help=(
            "Allowed summary stop_reason. Repeatable. Shadow preflight should keep "
            "the default complete; control logs may add aborted for intentional Ctrl+C."
        ),
    )
    parser.add_argument("--require-shadow-zero", action="store_true")
    parser.add_argument("--expect-policy-remote", action="store_true")
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--max-shadow-command-abs",
        type=float,
        default=1e-6,
        help="Allowed command abs max when --require-shadow-zero is set.",
    )
    args = parser.parse_args()

    ok = True
    if args.bundle_dir is not None:
        ok = _print_bundle_check(
            args.bundle_dir,
            expected_camera_names=_parse_expected_camera_names(args.expect_camera_names),
        ) and ok

    run_dir = _resolve_run_dir(args.run_dir, args.latest)
    if run_dir is None:
        return 0 if ok else 2

    summary = _load_summary(run_dir)
    steps = _load_steps(run_dir / "steps.jsonl")
    metrics = _compute_metrics(steps, warmup_steps=max(0, int(args.warmup_steps)))
    verdict_ok, reasons = _verdict(
        summary=summary,
        metrics=metrics,
        expect_output_mode=args.expect_output_mode,
        allow_stop_reasons=set(str(reason) for reason in args.allow_stop_reason),
        require_shadow_zero=bool(args.require_shadow_zero),
        expect_policy_remote=bool(args.expect_policy_remote),
        min_steps=int(args.min_steps),
        max_shadow_command_abs=float(args.max_shadow_command_abs),
    )
    ok = ok and verdict_ok
    _print_log_summary(run_dir, summary=summary, metrics=metrics, ok=verdict_ok, reasons=reasons)
    return 0 if ok else 2


def _parse_expected_camera_names(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _print_bundle_check(
    bundle_dir: Path,
    *,
    expected_camera_names: list[str] | None = None,
) -> bool:
    bundle = Path(bundle_dir)
    print(f"Bundle: {bundle}")
    ok = True
    for name in REQUIRED_BUNDLE_FILES:
        path = bundle / name
        if not path.exists():
            print(f"  MISSING {name}")
            ok = False
            continue
        print(f"  OK {name} {_format_bytes(path.stat().st_size)}")
    optional = bundle / "run_metadata.json"
    if optional.exists():
        print(f"  OK run_metadata.json {_format_bytes(optional.stat().st_size)}")
    else:
        print("  WARN run_metadata.json missing")
    if expected_camera_names is not None:
        ok = _print_camera_contract_check(
            bundle / "resolved_config.yaml",
            expected_camera_names=expected_camera_names,
        ) and ok
    print(f"Bundle verdict: {'OK' if ok else 'NOT OK'}")
    return ok


def _print_camera_contract_check(
    resolved_config_path: Path,
    *,
    expected_camera_names: list[str],
) -> bool:
    if not resolved_config_path.exists():
        return False
    try:
        resolved = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"  ERROR resolved_config.yaml unreadable: {type(exc).__name__}: {exc}")
        return False
    task_cfg = resolved.get("task", {}) or {}
    actual_camera_names = [str(name) for name in task_cfg.get("camera_names", ["fpv"])]
    if actual_camera_names != expected_camera_names:
        print(
            "  MISMATCH camera_names "
            f"expected={expected_camera_names!r} actual={actual_camera_names!r}"
        )
        return False
    print(f"  OK camera_names {actual_camera_names!r}")
    return True


def _resolve_run_dir(run_dir: Path | None, latest_root: Path | None) -> Path | None:
    if run_dir is not None:
        return Path(run_dir)
    if latest_root is None:
        return None
    root = Path(latest_root)
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "steps.jsonl").exists()
    ]
    if not candidates:
        raise SystemExit(f"No receiver test runs with steps.jsonl under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_steps(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing steps.jsonl: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def _compute_metrics(steps: list[dict[str, Any]], *, warmup_steps: int) -> dict[str, Any]:
    checked = steps[min(len(steps), warmup_steps) :]
    latencies = [
        _float(step.get("policy_inference_latency_ms"))
        for step in checked
        if _float(step.get("policy_inference_latency_ms")) > 0.0
    ]
    wall_times = [_int(step.get("wall_time_ns")) for step in steps if _int(step.get("wall_time_ns")) > 0]
    duration_s = (
        (wall_times[-1] - wall_times[0]) / 1_000_000_000.0
        if len(wall_times) >= 2
        else 0.0
    )
    effective_hz = (len(steps) - 1) / duration_s if duration_s > 0.0 else 0.0
    policy_actions = [_vec(step.get("policy_action")) for step in checked]
    policy_actions = [vec for vec in policy_actions if vec]
    assist_steps = [
        step
        for step in checked
        if int(step.get("policy_deadzone_assist_active", 0) or 0)
    ]
    assist_axes = _counts(
        axis
        for step in assist_steps
        for axis in str(step.get("policy_deadzone_assist_axes", "")).split(",")
        if axis
    )
    modes = _counts(str(step.get("policy_output_mode", "")) for step in steps)
    remote_modes = _counts(str(step.get("policy_remote_mode", "")) for step in steps)
    activated_steps = [
        int(step.get("local_step", idx))
        for idx, step in enumerate(steps)
        if int(step.get("policy_remote_activated", 0) or 0)
    ]
    return {
        "steps": len(steps),
        "checked_steps": len(checked),
        "effective_hz": effective_hz,
        "latency_mean_ms": _mean(latencies),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_max_ms": max(latencies) if latencies else 0.0,
        "policy_error_count": sum(
            1 for step in checked if str(step.get("policy_error", "")).strip()
        ),
        "health_bad_count": sum(
            1 for step in checked if int(step.get("receiver_health_ok", 0) or 0) == 0
        ),
        "health_errors": _counts(
            str(step.get("receiver_health_error_code", ""))
            for step in checked
            if str(step.get("receiver_health_error_code", "")).strip()
        ),
        "ack_bad_count": sum(
            1 for step in checked if int(step.get("controller_ack", 0) or 0) == 0
        ),
        "fault_codes": _counts(
            str(step.get("controller_fault_code", ""))
            for step in checked
            if str(step.get("controller_fault_code", "")).strip()
        ),
        "output_modes": modes,
        "policy_remote_modes": remote_modes,
        "policy_remote_activated_steps": activated_steps,
        "policy_action_mean": _vector_mean(policy_actions),
        "policy_action_max_abs": _vectors_max_abs(policy_actions),
        "deadzone_assist_enabled_count": sum(
            1
            for step in checked
            if int(step.get("policy_deadzone_assist_enabled", 0) or 0)
        ),
        "deadzone_assist_active_count": len(assist_steps),
        "deadzone_assist_axes": assist_axes,
        "deadzone_assist_active_pct": (
            len(assist_steps) / len(checked) * 100.0 if checked else 0.0
        ),
        "returned_action_max_abs": _steps_vec_max_abs(checked, "policy_returned_action"),
        "raw_action_max_abs": _steps_vec_max_abs(checked, "raw_action"),
        "safe_action_max_abs": _steps_vec_max_abs(checked, "safe_action"),
        "commanded_action_max_abs": _steps_vec_max_abs(checked, "commanded_action"),
    }


def _verdict(
    *,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    expect_output_mode: str | None,
    allow_stop_reasons: set[str],
    require_shadow_zero: bool,
    expect_policy_remote: bool,
    min_steps: int,
    max_shadow_command_abs: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    steps = int(metrics["steps"])
    if steps < min_steps:
        reasons.append(f"steps {steps} < min_steps {min_steps}")
    stop_reason = str(summary.get("stop_reason", "complete") or "complete")
    if stop_reason not in allow_stop_reasons:
        reasons.append(
            f"stop_reason={stop_reason} not allowed={sorted(allow_stop_reasons)}"
        )
    if int(metrics["policy_error_count"]):
        reasons.append(f"policy_error_count={metrics['policy_error_count']}")
    if int(metrics["health_bad_count"]):
        reasons.append(f"receiver_health_bad_count={metrics['health_bad_count']}")
    if int(metrics["ack_bad_count"]):
        reasons.append(f"controller_ack_bad_count={metrics['ack_bad_count']}")
    if metrics["fault_codes"]:
        reasons.append(f"controller_fault_codes={metrics['fault_codes']}")
    if expect_output_mode:
        modes = set(metrics["output_modes"].keys())
        if modes != {expect_output_mode}:
            reasons.append(f"policy_output_modes={sorted(modes)} expected={expect_output_mode}")
    if require_shadow_zero:
        for key in ("returned_action_max_abs", "raw_action_max_abs", "safe_action_max_abs", "commanded_action_max_abs"):
            if float(metrics[key]) > max_shadow_command_abs:
                reasons.append(f"{key}={metrics[key]:.6g} > {max_shadow_command_abs:g}")
    if expect_policy_remote:
        if "policy" not in metrics["policy_remote_modes"]:
            reasons.append("policy_remote never entered policy mode")
        if not metrics["policy_remote_activated_steps"]:
            reasons.append("policy_remote_activated was never observed")
    return (not reasons), reasons


def _print_log_summary(
    run_dir: Path,
    *,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    ok: bool,
    reasons: list[str],
) -> None:
    print(f"Run: {run_dir}")
    print(
        "Summary: "
        f"steps={metrics['steps']} stop_reason={summary.get('stop_reason', 'complete')} "
        f"effective_hz={metrics['effective_hz']:.2f}"
    )
    print(
        "Policy latency ms: "
        f"mean={metrics['latency_mean_ms']:.2f} "
        f"p50={metrics['latency_p50_ms']:.2f} "
        f"p95={metrics['latency_p95_ms']:.2f} "
        f"max={metrics['latency_max_ms']:.2f}"
    )
    print(
        "Policy action: "
        f"mean={_format_vec(metrics['policy_action_mean'])} "
        f"max_abs={metrics['policy_action_max_abs']:.4f}"
    )
    print(
        "Command max_abs: "
        f"returned={metrics['returned_action_max_abs']:.4f} "
        f"raw={metrics['raw_action_max_abs']:.4f} "
        f"safe={metrics['safe_action_max_abs']:.4f} "
        f"commanded={metrics['commanded_action_max_abs']:.4f}"
    )
    print(
        "Deadzone assist: "
        f"enabled_steps={metrics['deadzone_assist_enabled_count']} "
        f"active_steps={metrics['deadzone_assist_active_count']} "
        f"active_pct={metrics['deadzone_assist_active_pct']:.1f}% "
        f"axes={metrics['deadzone_assist_axes'] or '-'}"
    )
    print(
        "Health/control: "
        f"policy_errors={metrics['policy_error_count']} "
        f"health_bad={metrics['health_bad_count']} "
        f"ack_bad={metrics['ack_bad_count']} "
        f"fault_codes={metrics['fault_codes'] or '-'}"
    )
    print(
        "Modes: "
        f"policy_output={metrics['output_modes'] or '-'} "
        f"policy_remote={metrics['policy_remote_modes'] or '-'} "
        f"activated_steps={metrics['policy_remote_activated_steps'] or '-'}"
    )
    print(f"Verdict: {'OK' if ok else 'NOT OK'}")
    if reasons:
        for reason in reasons:
            print(f"  - {reason}")


def _vec(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list | tuple):
        return []
    out: list[float] = []
    for item in value:
        try:
            val = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(val):
            return []
        out.append(val)
    return out


def _steps_vec_max_abs(steps: Iterable[dict[str, Any]], key: str) -> float:
    return _vectors_max_abs([_vec(step.get(key)) for step in steps])


def _vectors_max_abs(vectors: Iterable[list[float]]) -> float:
    max_abs = 0.0
    for vec in vectors:
        for value in vec:
            max_abs = max(max_abs, abs(float(value)))
    return max_abs


def _vector_mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = min(len(vec) for vec in vectors)
    if width == 0:
        return []
    return [sum(vec[i] for vec in vectors) / len(vectors) for i in range(width)]


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        if value == "":
            continue
        out[value] = out.get(value, 0) + 1
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (float(pct) / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _format_vec(values: list[float]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(f"{value:+.4f}" for value in values) + "]"


def _format_bytes(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or suffix == "GiB":
            return f"{value:.1f}{suffix}"
        value /= 1024.0
    return f"{size}B"


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    sys.exit(main())
