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
        "--checkpoint-name",
        default="policy_best.ckpt",
        help="Checkpoint filename required inside --bundle-dir.",
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
    parser.add_argument("--expect-scripted-cycle", action="store_true")
    task_state_mode = parser.add_mutually_exclusive_group()
    task_state_mode.add_argument(
        "--expect-task-state-v2",
        action="store_true",
        help="Compatibility alias for --expect-task-state-v2-auto-progress.",
    )
    task_state_mode.add_argument(
        "--expect-task-state-v2-auto-progress",
        action="store_true",
        help="Require a complete automatic WORK-to-RETURN progression.",
    )
    task_state_mode.add_argument(
        "--expect-task-state-v2-stationary-shadow",
        action="store_true",
        help="Require stationary shadow_zero to remain safely in WORK.",
    )
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
        ok = (
            _print_bundle_check(
                args.bundle_dir,
                checkpoint_name=str(args.checkpoint_name),
                expected_camera_names=_parse_expected_camera_names(
                    args.expect_camera_names
                ),
            )
            and ok
        )

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
        expect_scripted_cycle=bool(args.expect_scripted_cycle),
        expect_task_state_v2=bool(args.expect_task_state_v2),
        expect_task_state_v2_auto_progress=bool(
            args.expect_task_state_v2_auto_progress
        ),
        expect_task_state_v2_stationary_shadow=bool(
            args.expect_task_state_v2_stationary_shadow
        ),
        min_steps=int(args.min_steps),
        max_shadow_command_abs=float(args.max_shadow_command_abs),
    )
    ok = ok and verdict_ok
    _print_log_summary(
        run_dir, summary=summary, metrics=metrics, ok=verdict_ok, reasons=reasons
    )
    return 0 if ok else 2


def _parse_expected_camera_names(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _print_bundle_check(
    bundle_dir: Path,
    *,
    checkpoint_name: str = "policy_best.ckpt",
    expected_camera_names: list[str] | None = None,
) -> bool:
    bundle = Path(bundle_dir)
    print(f"Bundle: {bundle}")
    ok = True
    for name in (str(checkpoint_name), *REQUIRED_BUNDLE_FILES):
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
        ok = (
            _print_camera_contract_check(
                bundle / "resolved_config.yaml",
                expected_camera_names=expected_camera_names,
            )
            and ok
        )
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
        resolved = (
            yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
        )
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
    candidates = sorted(
        {
            steps_path.parent
            for steps_path in root.rglob("steps.jsonl")
            if steps_path.is_file()
        }
    )
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


def _compute_metrics(
    steps: list[dict[str, Any]], *, warmup_steps: int
) -> dict[str, Any]:
    checked = steps[min(len(steps), warmup_steps) :]
    policy_steps = [
        step
        for step in checked
        if str(step.get("policy_remote_mode", "")) == "policy"
        or _float(step.get("policy_inference_latency_ms")) > 0.0
    ]
    latencies = [
        _float(step.get("policy_inference_latency_ms"))
        for step in policy_steps
        if _float(step.get("policy_inference_latency_ms")) > 0.0
    ]
    wall_times = [
        _int(step.get("wall_time_ns"))
        for step in steps
        if _int(step.get("wall_time_ns")) > 0
    ]
    duration_s = (
        (wall_times[-1] - wall_times[0]) / 1_000_000_000.0
        if len(wall_times) >= 2
        else 0.0
    )
    effective_hz = (len(steps) - 1) / duration_s if duration_s > 0.0 else 0.0
    policy_actions = [_vec(step.get("policy_action")) for step in policy_steps]
    policy_actions = [vec for vec in policy_actions if len(vec) == 4]
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
    modes = _counts(
        mode
        for step in steps
        for mode in [str(step.get("policy_output_mode", ""))]
        if mode
    )
    remote_modes = _counts(str(step.get("policy_remote_mode", "")) for step in steps)
    activated_steps = [
        int(step.get("local_step", idx))
        for idx, step in enumerate(steps)
        if int(step.get("policy_remote_activated", 0) or 0)
    ]
    pump_alignment = [
        int(step.get("action_pump_command_current", -1))
        for step in policy_steps
        if int(step.get("action_pump_command_current", -1)) >= 0
    ]
    policy_loop_ms = [
        (_int(current.get("wall_time_ns")) - _int(previous.get("wall_time_ns")))
        / 1_000_000.0
        for previous, current in zip(policy_steps, policy_steps[1:])
        if _int(current.get("local_step")) == _int(previous.get("local_step")) + 1
        and _int(current.get("wall_time_ns")) > _int(previous.get("wall_time_ns"))
    ]
    sample_to_update_ms = _timestamp_deltas_ms(
        policy_steps,
        start_key="action_sample_timestamp_ns",
        end_key="action_update_timestamp_ns",
    )
    update_to_send_ms = _timestamp_deltas_ms(
        policy_steps,
        start_key="action_update_timestamp_ns",
        end_key="action_send_timestamp_ns",
    )
    sample_to_send_ms = _timestamp_deltas_ms(
        policy_steps,
        start_key="action_sample_timestamp_ns",
        end_key="action_send_timestamp_ns",
    )
    image_to_send_ms = [
        (_int(step.get("action_send_timestamp_ns")) - _newest_image_timestamp_ns(step))
        / 1_000_000.0
        for step in policy_steps
        if _newest_image_timestamp_ns(step) > 0
        and _int(step.get("action_send_timestamp_ns"))
        >= _newest_image_timestamp_ns(step)
    ]
    image_to_sample_ms = [
        (
            _int(step.get("action_sample_timestamp_ns"))
            - _newest_image_timestamp_ns(step)
        )
        / 1_000_000.0
        for step in policy_steps
        if _newest_image_timestamp_ns(step) > 0
        and _int(step.get("action_sample_timestamp_ns"))
        >= _newest_image_timestamp_ns(step)
    ]
    return {
        "steps": len(steps),
        "checked_steps": len(checked),
        "effective_hz": effective_hz,
        "latency_mean_ms": _mean(latencies),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_max_ms": max(latencies) if latencies else 0.0,
        "policy_loop_p50_ms": _percentile(policy_loop_ms, 50),
        "policy_loop_p95_ms": _percentile(policy_loop_ms, 95),
        "sample_to_update_p50_ms": _percentile(sample_to_update_ms, 50),
        "sample_to_update_p95_ms": _percentile(sample_to_update_ms, 95),
        "update_to_send_p50_ms": _percentile(update_to_send_ms, 50),
        "update_to_send_p95_ms": _percentile(update_to_send_ms, 95),
        "sample_to_send_p50_ms": _percentile(sample_to_send_ms, 50),
        "sample_to_send_p95_ms": _percentile(sample_to_send_ms, 95),
        "image_to_send_p50_ms": _percentile(image_to_send_ms, 50),
        "image_to_send_p95_ms": _percentile(image_to_send_ms, 95),
        "image_to_sample_p50_ms": _percentile(image_to_sample_ms, 50),
        "image_to_sample_p95_ms": _percentile(image_to_sample_ms, 95),
        "frame_alignment_enabled_count": sum(
            1
            for step in policy_steps
            if int(step.get("policy_frame_alignment_enabled", 0) or 0)
        ),
        "frame_reused_count": sum(
            1 for step in policy_steps if int(step.get("policy_frame_reused", 0) or 0)
        ),
        "pump_alignment_known_count": len(pump_alignment),
        "pump_current_count": sum(value == 1 for value in pump_alignment),
        "pump_stale_count": sum(value == 0 for value in pump_alignment),
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
        "scripted_cycle_enabled_count": sum(
            int(step.get("scripted_cycle_enabled", 0) or 0) for step in checked
        ),
        "scripted_cycle_active_count": sum(
            int(step.get("scripted_cycle_active", 0) or 0) for step in checked
        ),
        "scripted_cycle_goal_changed_count": sum(
            int(step.get("scripted_cycle_goal_changed", 0) or 0) for step in checked
        ),
        "scripted_cycle_faults": _counts(
            str(step.get("scripted_cycle_fault", ""))
            for step in checked
            if str(step.get("scripted_cycle_fault", "")).strip()
        ),
        "scripted_cycle_activation_rejections": _counts(
            str(step.get("scripted_cycle_activation_rejected_reason", ""))
            for step in checked
            if str(step.get("scripted_cycle_activation_rejected_reason", "")).strip()
        ),
        "scripted_cycle_targets": _counts(
            str(step.get("planner_target_side", ""))
            for step in checked
            if str(step.get("planner_target_side", "")).strip()
        ),
        "task_state_enabled_count": sum(
            int(step.get("scripted_cycle_task_state_v2_enabled", 0) or 0)
            for step in checked
        ),
        "task_state_stages": _counts(
            str(step.get("scripted_cycle_task_state_stage", ""))
            for step in checked
            if str(step.get("scripted_cycle_task_state_stage", "")).strip()
        ),
        "task_state_changed_count": sum(
            int(step.get("scripted_cycle_task_state_changed", 0) or 0)
            for step in checked
        ),
        "task_state_advance_requested_count": sum(
            int(step.get("scripted_cycle_task_state_advance_requested", 0) or 0)
            for step in checked
        ),
        "task_state_advance_rejections": _counts(
            str(step.get("scripted_cycle_task_state_advance_rejected_reason", ""))
            for step in checked
            if str(
                step.get("scripted_cycle_task_state_advance_rejected_reason", "")
            ).strip()
        ),
        "task_auto_progress_enabled_count": sum(
            int(step.get("scripted_cycle_task_auto_progress_enabled", 0) or 0)
            for step in checked
        ),
        "task_auto_work_liveness_count": sum(
            int(step.get("scripted_cycle_task_auto_work_liveness", 0) or 0)
            for step in checked
        ),
        "task_auto_bucket_effective_count": sum(
            int(step.get("scripted_cycle_task_auto_bucket_effective_observed", 0) or 0)
            for step in checked
        ),
        "task_auto_applied_events": _counts(
            str(step.get("scripted_cycle_task_state_applied_event", ""))
            for step in checked
            if str(step.get("scripted_cycle_task_state_applied_event", "")).strip()
        ),
        "task_auto_pending_events": _counts(
            str(step.get("scripted_cycle_task_auto_pending_event", ""))
            for step in checked
            if str(step.get("scripted_cycle_task_auto_pending_event", "")).strip()
        ),
        "task_state_invalid_count": sum(
            1
            for step in policy_steps
            if not _valid_task_state_v2(step.get("policy_task_state_v2"))
        ),
        "task_state_planner_mismatch_count": sum(
            1
            for step in policy_steps
            if _vec(step.get("policy_task_state_v2"))
            != _vec(step.get("planner_task_state_v2"))
        ),
        "policy_action_mean": _vector_mean(policy_actions),
        "policy_action_vector_count": len(policy_actions),
        "policy_step_count": len(policy_steps),
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
        "returned_action_max_abs": _steps_vec_max_abs(
            checked, "policy_returned_action"
        ),
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
    expect_scripted_cycle: bool,
    min_steps: int,
    max_shadow_command_abs: float,
    expect_task_state_v2: bool = False,
    expect_task_state_v2_auto_progress: bool = False,
    expect_task_state_v2_stationary_shadow: bool = False,
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
    if int(metrics["pump_stale_count"]):
        reasons.append(
            "action_pump_stale_command_count="
            f"{metrics['pump_stale_count']}/{metrics['pump_alignment_known_count']}"
        )
    if expect_output_mode:
        modes = set(metrics["output_modes"].keys())
        allowed_modes = {expect_output_mode, "script_stop_zero"}
        if expect_output_mode not in modes or not modes.issubset(allowed_modes):
            reasons.append(
                f"policy_output_modes={sorted(modes)} expected={expect_output_mode}"
            )
    if require_shadow_zero:
        for key in (
            "returned_action_max_abs",
            "raw_action_max_abs",
            "safe_action_max_abs",
            "commanded_action_max_abs",
        ):
            if float(metrics[key]) > max_shadow_command_abs:
                reasons.append(f"{key}={metrics[key]:.6g} > {max_shadow_command_abs:g}")
    if expect_policy_remote:
        if "policy" not in metrics["policy_remote_modes"]:
            reasons.append("policy_remote never entered policy mode")
        if not metrics["policy_remote_activated_steps"]:
            reasons.append("policy_remote_activated was never observed")
    if expect_scripted_cycle:
        if not int(metrics["scripted_cycle_enabled_count"]):
            reasons.append("scripted-cycle runtime was never observed")
        if not int(metrics["scripted_cycle_active_count"]):
            reasons.append("scripted-cycle runtime never became active")
        if metrics["scripted_cycle_faults"]:
            reasons.append(f"scripted_cycle_faults={metrics['scripted_cycle_faults']}")
        if metrics["scripted_cycle_activation_rejections"]:
            reasons.append(
                "scripted_cycle_activation_rejections="
                f"{metrics['scripted_cycle_activation_rejections']}"
            )
        if not metrics["scripted_cycle_targets"]:
            reasons.append("scripted-cycle planner target was never logged")
    expect_task_common = bool(
        expect_task_state_v2
        or expect_task_state_v2_auto_progress
        or expect_task_state_v2_stationary_shadow
    )
    if expect_task_common:
        if not int(metrics["task_state_enabled_count"]):
            reasons.append("task-state-v2 runtime was never observed")
        if not int(metrics["task_auto_progress_enabled_count"]):
            reasons.append("task-state-v2 automatic progress was never observed")
        if int(metrics["task_state_advance_requested_count"]):
            reasons.append("task-state-v2 received an unexpected manual mark")
        if metrics["task_state_advance_rejections"]:
            reasons.append(
                "task-state-v2 mark rejections="
                f"{metrics['task_state_advance_rejections']}"
            )
        if int(metrics["task_state_invalid_count"]):
            reasons.append(
                f"task-state-v2 invalid policy vectors={metrics['task_state_invalid_count']}"
            )
        if int(metrics["task_state_planner_mismatch_count"]):
            reasons.append(
                "task-state-v2 planner/policy mismatch count="
                f"{metrics['task_state_planner_mismatch_count']}"
            )
        if int(metrics["policy_action_vector_count"]) != int(
            metrics["policy_step_count"]
        ):
            reasons.append(
                "policy action vectors missing or malformed="
                f"{metrics['policy_action_vector_count']}/"
                f"{metrics['policy_step_count']}"
            )
    if expect_task_state_v2 or expect_task_state_v2_auto_progress:
        expected_stages = {"work", "work_complete", "return_committed"}
        missing_stages = expected_stages - set(metrics["task_state_stages"])
        if missing_stages:
            reasons.append(f"task-state-v2 missing stages={sorted(missing_stages)}")
        if int(metrics["task_state_changed_count"]) < 2:
            reasons.append(
                "task-state-v2 did not log both work-complete and return-commit changes"
            )
        if not int(metrics["task_auto_work_liveness_count"]):
            reasons.append("task-state-v2 never confirmed boom/bucket work liveness")
        if not int(metrics["task_auto_bucket_effective_count"]):
            reasons.append("task-state-v2 never confirmed effective bucket work")
        required_auto_events = {"work_complete", "return_commit"}
        missing_auto_events = required_auto_events - set(
            metrics["task_auto_applied_events"]
        )
        if missing_auto_events:
            reasons.append(
                f"task-state-v2 missing automatic events={sorted(missing_auto_events)}"
            )
    if expect_task_state_v2_stationary_shadow:
        if expect_output_mode != "shadow_zero" or not require_shadow_zero:
            reasons.append(
                "stationary task-state shadow requires shadow_zero and locked outputs"
            )
        stages = set(metrics["task_state_stages"])
        if "work" not in stages:
            reasons.append("stationary shadow never reached the WORK stage")
        forbidden_stages = stages & {"work_complete", "return_committed"}
        if forbidden_stages:
            reasons.append(
                f"stationary shadow advanced task state={sorted(forbidden_stages)}"
            )
        if int(metrics["task_state_changed_count"]):
            reasons.append("stationary shadow changed task-state bits")
        if int(metrics["task_auto_work_liveness_count"]):
            reasons.append("stationary shadow falsely confirmed work liveness")
        if int(metrics["task_auto_bucket_effective_count"]):
            reasons.append("stationary shadow falsely confirmed bucket work")
        if metrics["task_auto_pending_events"]:
            reasons.append(
                "stationary shadow created pending automatic events="
                f"{metrics['task_auto_pending_events']}"
            )
        if metrics["task_auto_applied_events"]:
            reasons.append(
                "stationary shadow applied automatic events="
                f"{metrics['task_auto_applied_events']}"
            )
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
        "Policy chain ms: "
        f"loop_p50/p95={metrics['policy_loop_p50_ms']:.2f}/"
        f"{metrics['policy_loop_p95_ms']:.2f} "
        f"sample_update={metrics['sample_to_update_p50_ms']:.2f}/"
        f"{metrics['sample_to_update_p95_ms']:.2f} "
        f"update_send={metrics['update_to_send_p50_ms']:.2f}/"
        f"{metrics['update_to_send_p95_ms']:.2f} "
        f"image_send={metrics['image_to_send_p50_ms']:.2f}/"
        f"{metrics['image_to_send_p95_ms']:.2f}"
    )
    print(
        "Frame alignment: "
        f"enabled_steps={metrics['frame_alignment_enabled_count']} "
        f"reused_steps={metrics['frame_reused_count']} "
        f"image_sample={metrics['image_to_sample_p50_ms']:.2f}/"
        f"{metrics['image_to_sample_p95_ms']:.2f}"
    )
    print(
        "Pump alignment: "
        f"current={metrics['pump_current_count']} "
        f"stale={metrics['pump_stale_count']} "
        f"known={metrics['pump_alignment_known_count']}"
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
    print(
        "Scripted cycle: "
        f"enabled_steps={metrics['scripted_cycle_enabled_count']} "
        f"active_steps={metrics['scripted_cycle_active_count']} "
        f"goal_changes={metrics['scripted_cycle_goal_changed_count']} "
        f"targets={metrics['scripted_cycle_targets'] or '-'} "
        f"faults={metrics['scripted_cycle_faults'] or '-'} "
        f"activation_rejections="
        f"{metrics['scripted_cycle_activation_rejections'] or '-'}"
    )
    print(
        "Task state v2: "
        f"enabled_steps={metrics['task_state_enabled_count']} "
        f"stages={metrics['task_state_stages'] or '-'} "
        f"changes={metrics['task_state_changed_count']} "
        f"marks={metrics['task_state_advance_requested_count']} "
        f"invalid={metrics['task_state_invalid_count']} "
        f"mismatch={metrics['task_state_planner_mismatch_count']} "
        f"rejections={metrics['task_state_advance_rejections'] or '-'}"
    )
    print(
        "Automatic progress: "
        f"enabled_steps={metrics['task_auto_progress_enabled_count']} "
        f"work_liveness_steps={metrics['task_auto_work_liveness_count']} "
        f"bucket_effective_steps={metrics['task_auto_bucket_effective_count']} "
        f"pending={metrics['task_auto_pending_events'] or '-'} "
        f"applied={metrics['task_auto_applied_events'] or '-'}"
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


def _valid_task_state_v2(value: Any) -> bool:
    vector = _vec(value)
    if len(vector) != 5:
        return False
    current, dig_target, complete, commit, next_target = vector
    if current not in {-1.0, 1.0} or dig_target != current:
        return False
    if complete not in {0.0, 1.0} or commit not in {0.0, 1.0}:
        return False
    if commit == 0.0:
        return next_target == 0.0
    return next_target in {-1.0, 1.0}


def _steps_vec_max_abs(steps: Iterable[dict[str, Any]], key: str) -> float:
    return _vectors_max_abs([_vec(step.get(key)) for step in steps])


def _timestamp_deltas_ms(
    steps: Iterable[dict[str, Any]],
    *,
    start_key: str,
    end_key: str,
) -> list[float]:
    values: list[float] = []
    for step in steps:
        start = _int(step.get(start_key))
        end = _int(step.get(end_key))
        if start > 0 and end >= start:
            values.append((end - start) / 1_000_000.0)
    return values


def _newest_image_timestamp_ns(step: dict[str, Any]) -> int:
    raw = step.get("image_timestamp_ns")
    if isinstance(raw, dict):
        return max((_int(value) for value in raw.values()), default=0)
    return _int(raw)


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
