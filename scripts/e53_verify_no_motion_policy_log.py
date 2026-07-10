#!/usr/bin/env python3
"""Verify a shadow-zero policy dry-run log before any live motion test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_POLICY_ACTION_KEYS = (
    "policy_action",
    "policy_scaled_action",
    "policy_assisted_action",
    "policy_returned_action",
)
NO_MOTION_ACTION_KEYS = (
    "safe_action",
    "commanded_action",
    "policy_returned_action",
)
RUNTIME_GATE_VECTOR_SHAPES = {
    "policy_intent_probabilities": (8,),
    "policy_phase_gated_action": (4,),
    "policy_snap_active_mask": (4,),
    "policy_snap_action": (4,),
    "temporal_direction_gate_probabilities": (8,),
    "temporal_direction_gate_active_mask": (8,),
    "policy_temporal_direction_action": (4,),
}
RUNTIME_GATE_SCALAR_KEYS = (
    "phase_gate_prob",
    "phase_gate_threshold",
    "phase_gate_inactive_scale",
    "phase_gate_active",
    "policy_snap_margin",
    "policy_snap_intent_threshold",
    "temporal_direction_gate_threshold",
    "temporal_direction_gate_inactive_scale",
    "gohome_candidate_probability",
    "gohome_candidate_threshold",
    "gohome_candidate_required_steps",
    "gohome_candidate_consecutive_steps",
    "gohome_eligibility_probability",
    "gohome_eligibility_threshold",
    "gohome_eligibility_required_steps",
    "gohome_eligibility_consecutive_steps",
    "gohome_raw_active",
    "gohome_request_active",
    "gohome_request_suppressed",
    "go_home_requested",
)
RUNTIME_GATE_STRING_KEYS = (
    "policy_gate_stack_id",
    "gohome_request_suppression_reason",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps_jsonl", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--expected-output-mode", default="shadow_zero")
    parser.add_argument("--motion-tolerance", type=float, default=0.0)
    parser.add_argument("--policy-nonzero-threshold", type=float, default=1e-6)
    parser.add_argument("--min-policy-nonzero-steps", type=int, default=1)
    parser.add_argument(
        "--require-runtime-gate-diagnostics",
        action="store_true",
        help="Require the complete E52 runtime gate diagnostics on every step.",
    )
    parser.add_argument(
        "--required-policy-action-key",
        action="append",
        default=[],
        help="Additional per-step action-like policy diagnostic field required in the JSONL.",
    )
    args = parser.parse_args()

    report = verify_no_motion_policy_log(
        args.steps_jsonl,
        expected_output_mode=str(args.expected_output_mode),
        motion_tolerance=float(args.motion_tolerance),
        policy_nonzero_threshold=float(args.policy_nonzero_threshold),
        min_policy_nonzero_steps=int(args.min_policy_nonzero_steps),
        required_policy_action_keys=tuple(args.required_policy_action_key),
        require_runtime_gate_diagnostics=bool(
            args.require_runtime_gate_diagnostics
        ),
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["ok"]:
        raise SystemExit(1)


def verify_no_motion_policy_log(
    steps_jsonl: Path,
    *,
    expected_output_mode: str = "shadow_zero",
    motion_tolerance: float = 0.0,
    policy_nonzero_threshold: float = 1e-6,
    min_policy_nonzero_steps: int = 1,
    required_policy_action_keys: tuple[str, ...] = (),
    require_runtime_gate_diagnostics: bool = False,
) -> dict[str, Any]:
    rows, read_errors = _read_jsonl(Path(steps_jsonl))
    errors = list(read_errors)
    required_policy_keys = tuple(DEFAULT_POLICY_ACTION_KEYS) + tuple(required_policy_action_keys)
    max_abs: dict[str, float] = {}
    missing_counts = {key: 0 for key in required_policy_keys + NO_MOTION_ACTION_KEYS}
    policy_nonzero_steps = 0
    policy_error_steps = 0
    wrong_mode_steps = 0
    nonfinite_steps = 0
    malformed_action_steps = 0
    runtime_gate_missing_counts = {
        key: 0
        for key in (
            tuple(RUNTIME_GATE_VECTOR_SHAPES)
            + RUNTIME_GATE_SCALAR_KEYS
            + RUNTIME_GATE_STRING_KEYS
        )
    }
    malformed_runtime_gate_fields = 0

    for index, row in enumerate(rows):
        mode = str(row.get("policy_output_mode", ""))
        if mode != expected_output_mode:
            wrong_mode_steps += 1

        policy_error = str(row.get("policy_error", "") or "")
        if policy_error:
            policy_error_steps += 1

        for key in required_policy_keys + NO_MOTION_ACTION_KEYS:
            if key not in row or row.get(key) is None:
                missing_counts[key] += 1
                continue
            action = _as_action(row.get(key))
            if action is None:
                malformed_action_steps += 1
                continue
            if not np.all(np.isfinite(action)):
                nonfinite_steps += 1
                continue
            value = float(np.max(np.abs(action))) if action.size else 0.0
            max_abs[key] = max(max_abs.get(key, 0.0), value)

        policy_action = _as_action(row.get("policy_action"))
        if policy_action is not None and np.all(np.isfinite(policy_action)):
            if float(np.max(np.abs(policy_action))) > float(policy_nonzero_threshold):
                policy_nonzero_steps += 1

        for key in NO_MOTION_ACTION_KEYS:
            action = _as_action(row.get(key))
            if action is not None and np.all(np.isfinite(action)):
                if float(np.max(np.abs(action))) > float(motion_tolerance):
                    errors.append(
                        f"step {index} {key} exceeds no-motion tolerance: "
                        f"{float(np.max(np.abs(action))):.9g} > {float(motion_tolerance):.9g}"
                    )

        if require_runtime_gate_diagnostics:
            for key, shape in RUNTIME_GATE_VECTOR_SHAPES.items():
                if key not in row or row.get(key) is None:
                    runtime_gate_missing_counts[key] += 1
                    continue
                value = _as_finite_array(row.get(key), shape=shape)
                if value is None:
                    malformed_runtime_gate_fields += 1
                    errors.append(
                        f"step {index} {key} must be a finite vector with shape {shape}"
                    )
            for key in RUNTIME_GATE_SCALAR_KEYS:
                if key not in row or row.get(key) is None:
                    runtime_gate_missing_counts[key] += 1
                    continue
                if _as_finite_scalar(row.get(key)) is None:
                    malformed_runtime_gate_fields += 1
                    errors.append(f"step {index} {key} must be a finite scalar")
            for key in RUNTIME_GATE_STRING_KEYS:
                if key not in row or row.get(key) is None:
                    runtime_gate_missing_counts[key] += 1
            if not str(row.get("policy_gate_stack_id", "")):
                errors.append(f"step {index} policy_gate_stack_id is empty")
            if bool(row.get("go_home_requested", False)):
                errors.append(
                    f"step {index} go_home_requested must remain false in shadow_zero"
                )
            if bool(row.get("gohome_request_active", False)):
                if not bool(row.get("gohome_request_suppressed", False)):
                    errors.append(
                        f"step {index} active gohome request was not suppressed in shadow_zero"
                    )
                if not str(row.get("gohome_request_suppression_reason", "")):
                    errors.append(
                        f"step {index} suppressed gohome request has no suppression reason"
                    )

    if not rows:
        errors.append("steps_jsonl is empty")
    for key, count in sorted(missing_counts.items()):
        if count:
            errors.append(f"missing {key} on {count} step(s)")
    if wrong_mode_steps:
        errors.append(
            f"policy_output_mode mismatch on {wrong_mode_steps} step(s), expected {expected_output_mode!r}"
        )
    if policy_error_steps:
        errors.append(f"policy_error is non-empty on {policy_error_steps} step(s)")
    if nonfinite_steps:
        errors.append(f"non-finite action values on {nonfinite_steps} field occurrence(s)")
    if malformed_action_steps:
        errors.append(f"malformed action values on {malformed_action_steps} field occurrence(s)")
    if require_runtime_gate_diagnostics:
        for key, count in sorted(runtime_gate_missing_counts.items()):
            if count:
                errors.append(f"missing {key} on {count} step(s)")
    if policy_nonzero_steps < int(min_policy_nonzero_steps):
        errors.append(
            "policy_action did not prove policy inference ran: "
            f"{policy_nonzero_steps} nonzero step(s) < {int(min_policy_nonzero_steps)}"
        )

    return {
        "ok": not errors,
        "steps_jsonl": str(steps_jsonl),
        "steps": len(rows),
        "expected_output_mode": str(expected_output_mode),
        "motion_tolerance": float(motion_tolerance),
        "policy_nonzero_threshold": float(policy_nonzero_threshold),
        "min_policy_nonzero_steps": int(min_policy_nonzero_steps),
        "policy_nonzero_steps": int(policy_nonzero_steps),
        "wrong_output_mode_steps": int(wrong_mode_steps),
        "policy_error_steps": int(policy_error_steps),
        "max_abs_policy_action": float(max_abs.get("policy_action", 0.0)),
        "max_abs_safe_action": float(max_abs.get("safe_action", 0.0)),
        "max_abs_commanded_action": float(max_abs.get("commanded_action", 0.0)),
        "max_abs_by_key": {key: float(value) for key, value in sorted(max_abs.items())},
        "missing_counts": {key: int(value) for key, value in sorted(missing_counts.items())},
        "required_policy_action_keys": list(required_policy_keys),
        "no_motion_action_keys": list(NO_MOTION_ACTION_KEYS),
        "require_runtime_gate_diagnostics": bool(
            require_runtime_gate_diagnostics
        ),
        "runtime_gate_missing_counts": {
            key: int(value)
            for key, value in sorted(runtime_gate_missing_counts.items())
        },
        "malformed_runtime_gate_fields": int(malformed_runtime_gate_fields),
        "errors": errors,
    }


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [], [f"steps_jsonl does not exist: {path}"]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number} is not valid JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"line {line_number} is not a JSON object")
            continue
        rows.append(payload)
    return rows, errors


def _as_action(value: Any) -> np.ndarray | None:
    try:
        action = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if action.shape != (4,):
        return None
    return action


def _as_finite_array(value: Any, *, shape: tuple[int, ...]) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if result.shape != shape or not np.all(np.isfinite(result)):
        return None
    return result


def _as_finite_scalar(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


if __name__ == "__main__":
    main()
