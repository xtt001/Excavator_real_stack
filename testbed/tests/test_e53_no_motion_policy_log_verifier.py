import json
from pathlib import Path

from scripts.e53_verify_no_motion_policy_log import verify_no_motion_policy_log


def _write_steps(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _gated_shadow_row() -> dict:
    return {
        "local_step": 0,
        "policy_output_mode": "shadow_zero",
        "policy_action": [0.2, 0.0, 0.0, 0.0],
        "policy_scaled_action": [0.1, 0.0, 0.0, 0.0],
        "policy_assisted_action": [0.1, 0.0, 0.0, 0.0],
        "policy_returned_action": [0.0, 0.0, 0.0, 0.0],
        "safe_action": [0.0, 0.0, 0.0, 0.0],
        "commanded_action": [0.0, 0.0, 0.0, 0.0],
        "policy_error": "",
        "go_home_requested": 0,
        "policy_gate_stack_id": "E52",
        "policy_intent_probabilities": [0.8, 0.1, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1],
        "phase_gate_prob": 0.9,
        "phase_gate_threshold": 0.15,
        "phase_gate_inactive_scale": 0.5,
        "phase_gate_active": 1,
        "policy_phase_gated_action": [0.2, 0.0, 0.0, 0.0],
        "policy_snap_active_mask": [1, 0, 0, 0],
        "policy_snap_action": [0.21, 0.0, 0.0, 0.0],
        "policy_snap_margin": 0.02,
        "policy_snap_intent_threshold": 0.7,
        "temporal_direction_gate_probabilities": [
            0.9,
            0.1,
            0.2,
            0.8,
            0.1,
            0.1,
            0.1,
            0.1,
        ],
        "temporal_direction_gate_threshold": 0.5,
        "temporal_direction_gate_inactive_scale": 0.75,
        "temporal_direction_gate_active_mask": [1, 0, 0, 1, 0, 0, 0, 0],
        "policy_temporal_direction_action": [0.21, 0.0, 0.0, 0.0],
        "gohome_candidate_probability": 0.99,
        "gohome_candidate_threshold": 0.97,
        "gohome_candidate_required_steps": 10,
        "gohome_candidate_consecutive_steps": 10,
        "gohome_eligibility_probability": 0.85,
        "gohome_eligibility_threshold": 0.8,
        "gohome_eligibility_required_steps": 3,
        "gohome_eligibility_consecutive_steps": 3,
        "gohome_raw_active": 1,
        "gohome_request_active": 1,
        "gohome_request_suppressed": 1,
        "gohome_request_suppression_reason": "policy_output_mode_shadow_zero",
    }


def test_verify_no_motion_policy_log_passes_shadow_zero_with_nonzero_policy_action(tmp_path: Path) -> None:
    steps = tmp_path / "steps.jsonl"
    _write_steps(
        steps,
        [
            {
                "local_step": 0,
                "policy_output_mode": "shadow_zero",
                "policy_action": [0.2, 0.0, 0.0, 0.0],
                "policy_scaled_action": [0.1, 0.0, 0.0, 0.0],
                "policy_assisted_action": [0.1, 0.0, 0.0, 0.0],
                "policy_returned_action": [0.0, 0.0, 0.0, 0.0],
                "safe_action": [0.0, 0.0, 0.0, 0.0],
                "commanded_action": [0.0, 0.0, 0.0, 0.0],
                "policy_error": "",
            },
            {
                "local_step": 1,
                "policy_output_mode": "shadow_zero",
                "policy_action": [0.0, -0.3, 0.0, 0.0],
                "policy_scaled_action": [0.0, -0.15, 0.0, 0.0],
                "policy_assisted_action": [0.0, -0.15, 0.0, 0.0],
                "policy_returned_action": [0.0, 0.0, 0.0, 0.0],
                "safe_action": [0.0, 0.0, 0.0, 0.0],
                "commanded_action": [0.0, 0.0, 0.0, 0.0],
                "policy_error": "",
            },
        ],
    )

    report = verify_no_motion_policy_log(steps)

    assert report["ok"] is True
    assert report["steps"] == 2
    assert report["policy_nonzero_steps"] == 2
    assert report["max_abs_commanded_action"] == 0.0
    assert report["max_abs_safe_action"] == 0.0
    assert report["errors"] == []


def test_verify_no_motion_policy_log_fails_when_motion_commanded(tmp_path: Path) -> None:
    steps = tmp_path / "steps.jsonl"
    _write_steps(
        steps,
        [
            {
                "local_step": 0,
                "policy_output_mode": "shadow_zero",
                "policy_action": [0.2, 0.0, 0.0, 0.0],
                "policy_scaled_action": [0.1, 0.0, 0.0, 0.0],
                "policy_assisted_action": [0.1, 0.0, 0.0, 0.0],
                "policy_returned_action": [0.0, 0.0, 0.0, 0.0],
                "safe_action": [0.0, 0.02, 0.0, 0.0],
                "commanded_action": [0.0, 0.02, 0.0, 0.0],
                "policy_error": "",
            }
        ],
    )

    report = verify_no_motion_policy_log(steps)

    assert report["ok"] is False
    assert any("commanded_action" in error for error in report["errors"])
    assert any("safe_action" in error for error in report["errors"])


def test_verify_no_motion_policy_log_can_require_runtime_gate_diagnostics(
    tmp_path: Path,
) -> None:
    steps = tmp_path / "steps.jsonl"
    _write_steps(steps, [_gated_shadow_row()])

    report = verify_no_motion_policy_log(
        steps,
        require_runtime_gate_diagnostics=True,
    )

    assert report["ok"] is True
    assert report["require_runtime_gate_diagnostics"] is True
    assert report["malformed_runtime_gate_fields"] == 0
    assert not any(report["runtime_gate_missing_counts"].values())


def test_runtime_gate_verification_requires_all_fields_without_changing_legacy_default(
    tmp_path: Path,
) -> None:
    steps = tmp_path / "steps.jsonl"
    row = _gated_shadow_row()
    del row["phase_gate_prob"]
    _write_steps(steps, [row])

    legacy_report = verify_no_motion_policy_log(steps)
    gated_report = verify_no_motion_policy_log(
        steps,
        require_runtime_gate_diagnostics=True,
    )

    assert legacy_report["ok"] is True
    assert gated_report["ok"] is False
    assert gated_report["runtime_gate_missing_counts"]["phase_gate_prob"] == 1
    assert any("missing phase_gate_prob" in error for error in gated_report["errors"])


def test_runtime_gate_verification_rejects_unsuppressed_shadow_gohome_request(
    tmp_path: Path,
) -> None:
    steps = tmp_path / "steps.jsonl"
    row = _gated_shadow_row()
    row["gohome_request_suppressed"] = 0
    row["gohome_request_suppression_reason"] = ""
    _write_steps(steps, [row])

    report = verify_no_motion_policy_log(
        steps,
        require_runtime_gate_diagnostics=True,
    )

    assert report["ok"] is False
    assert any("was not suppressed" in error for error in report["errors"])
    assert any("no suppression reason" in error for error in report["errors"])
