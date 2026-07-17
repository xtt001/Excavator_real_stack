from __future__ import annotations

import json

import pytest

from testbed.cli.audit_short_horizon_rollout import (
    run_short_horizon_rollout_audit,
)
from testbed.policies.short_horizon_rollout import (
    SCHEMA_VERSION,
    ShortHorizonRolloutContract,
    ShortHorizonRolloutStep,
    evaluate_short_horizon_rollout,
)


def _contract_payload(
    *,
    state_origin: str = "teacher_forced",
    authority: str = "observe_only",
    horizon: int = 3,
) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "test_id": "short-rollout-test",
        "state_origin": state_origin,
        "control_authority": authority,
        "policy_id": "fixture-policy",
        "checkpoint_sha256": "a" * 64,
        "resolved_config_sha256": "b" * 64,
        "sampling_hz": 20.0,
        "horizon_ticks": horizon,
        "camera_names": ["video4", "video5", "video6", "video7"],
        "max_observation_gap_ms": 60.0,
        "max_camera_age_ms": 10.0,
        "deadzone_positive": [0.661, 0.259, 0.5, 0.408],
        "deadzone_negative": [0.721, 0.357, 0.5, 0.508],
        "require_deadman": True,
        "require_controller_ack": True,
    }
    if authority == "bounded_control":
        payload.update(
            {
                "command_abs_limit": [0.8, 0.8, 0.8, 0.8],
                "command_delta_limit": [0.4, 0.4, 0.4, 0.4],
                "qvel_abort_limit": [1.0, 1.0, 1.0, 1.0],
                "qpos_lower_limit": [-2.0, -2.0, -2.0, -2.0],
                "qpos_upper_limit": [2.0, 2.0, 2.0, 2.0],
                "allowed_direction_mask": [[True, True]] * 4,
            }
        )
    return payload


def _step_payload(
    tick: int,
    *,
    origin: str,
    policy_action: list[float] | None = None,
    commanded_action: list[float] | None = None,
    sent: bool = False,
    generated_by: int | None = None,
    qvel: list[float] | None = None,
) -> dict:
    observation_ns = 1_000_000_000 + tick * 50_000_000
    command = commanded_action or [0.0, 0.0, 0.0, 0.0]
    return {
        "tick": tick,
        "observation_timestamp_ns": observation_ns,
        "camera_timestamps_ns": {
            camera: observation_ns - 1_000_000
            for camera in ("video4", "video5", "video6", "video7")
        },
        "camera_frame_ids": {
            camera: f"{camera}:{tick}"
            for camera in ("video4", "video5", "video6", "video7")
        },
        "observation_origin": origin,
        "generated_by_command_id": generated_by,
        "qpos": [0.0, 0.0, 0.0, 0.0],
        "qvel": qvel or [0.0, 0.0, 0.0, 0.0],
        "policy_action": policy_action or [0.0, 0.3, 0.0, 0.0],
        "policy_returned_action": command,
        "safe_action": command,
        "commanded_action": command,
        "command_sent": sent,
        "command_id": tick if sent else None,
        "send_timestamp_ns": observation_ns + 1_000_000 if sent else None,
        "controller_ack": sent,
        "deadman_pressed": True,
        "estop_active": False,
        "manual_override_active": False,
        "sensor_stale": False,
        "safety_reasons": [],
        "policy_error": "",
    }


def test_teacher_forced_trace_is_valid_but_not_self_generated() -> None:
    contract = ShortHorizonRolloutContract.from_mapping(_contract_payload())
    steps = [
        ShortHorizonRolloutStep.from_mapping(
            _step_payload(index, origin="teacher_forced")
        )
        for index in range(3)
    ]

    report = evaluate_short_horizon_rollout(
        contract=contract,
        steps=steps,
        termination_reason="horizon_complete",
    )

    assert report["trace_integrity_valid"] is True
    assert report["contract_compliant"] is True
    assert (
        report["causal_state_progression"]["self_generated_state_evidence"]
        == "none_noncausal_observation_source"
    )
    assert report["action_chain"]["policy_activation_motif"] == [["boom+"]]
    assert report["action_chain"]["commanded_activation_motif"] == []
    assert report["capability_boundaries"]["physical_response_estimable"] is False


def test_live_policy_on_requires_complete_causal_command_links() -> None:
    contract = ShortHorizonRolloutContract.from_mapping(
        _contract_payload(state_origin="live_policy_on", authority="bounded_control")
    )
    steps = [
        ShortHorizonRolloutStep.from_mapping(
            _step_payload(
                index,
                origin="live_policy_on",
                commanded_action=[0.0, 0.3, 0.0, 0.0],
                sent=True,
                generated_by=None if index == 0 else index - 1,
                qvel=[0.0, 0.2, 0.0, 0.0],
            )
        )
        for index in range(3)
    ]

    report = evaluate_short_horizon_rollout(
        contract=contract,
        steps=steps,
        termination_reason="horizon_complete",
    )

    assert report["trace_integrity_valid"] is True
    assert report["contract_compliant"] is True
    assert report["causal_state_progression"] == {
        "eligible_transition_count": 2,
        "causally_linked_transition_count": 2,
        "causal_link_fraction": 1.0,
        "nonzero_command_linked_transition_count": 2,
        "self_generated_state_evidence": "direct_physical_short_horizon",
    }
    assert report["capability_boundaries"]["physical_response_estimable"] is True


def test_missing_parent_command_prevents_self_generated_claim() -> None:
    contract = ShortHorizonRolloutContract.from_mapping(
        _contract_payload(state_origin="live_policy_on", authority="bounded_control")
    )
    steps = [
        ShortHorizonRolloutStep.from_mapping(
            _step_payload(
                index,
                origin="live_policy_on",
                commanded_action=[0.0, 0.3, 0.0, 0.0],
                sent=True,
                generated_by=None,
            )
        )
        for index in range(3)
    ]

    report = evaluate_short_horizon_rollout(
        contract=contract,
        steps=steps,
        termination_reason="horizon_complete",
    )

    assert report["trace_integrity_valid"] is False
    assert (
        report["causal_state_progression"]["self_generated_state_evidence"]
        == "not_estimable_invalid_trace"
    )


def test_observe_only_rejects_nonzero_command() -> None:
    contract = ShortHorizonRolloutContract.from_mapping(_contract_payload(horizon=1))
    step = ShortHorizonRolloutStep.from_mapping(
        _step_payload(
            0,
            origin="teacher_forced",
            commanded_action=[0.0, 0.3, 0.0, 0.0],
        )
    )

    report = evaluate_short_horizon_rollout(
        contract=contract,
        steps=[step],
        termination_reason="horizon_complete",
    )

    assert report["trace_integrity_valid"] is True
    assert report["contract_compliant"] is False
    assert "observe_only" in report["contract_breaches"][0]


def test_safety_limit_trigger_must_end_trace_immediately() -> None:
    contract = ShortHorizonRolloutContract.from_mapping(
        _contract_payload(
            state_origin="live_policy_on",
            authority="bounded_control",
            horizon=3,
        )
    )
    steps = [
        ShortHorizonRolloutStep.from_mapping(
            _step_payload(
                0,
                origin="live_policy_on",
                commanded_action=[0.0, 0.3, 0.0, 0.0],
                sent=True,
            )
        ),
        ShortHorizonRolloutStep.from_mapping(
            _step_payload(
                1,
                origin="live_policy_on",
                commanded_action=[0.0, 0.0, 0.0, 0.0],
                sent=True,
                generated_by=0,
                qvel=[0.0, 1.1, 0.0, 0.0],
            )
        ),
    ]

    report = evaluate_short_horizon_rollout(
        contract=contract,
        steps=steps,
        termination_reason="safety_abort",
    )

    assert report["trace_integrity_valid"] is True
    assert report["contract_compliant"] is True
    assert report["abort_triggers"] == [{"tick": 1, "reason": "qvel_abort_limit"}]


def test_contract_rejects_unbounded_live_control_and_long_horizon() -> None:
    with pytest.raises(ValueError, match="bounded_control"):
        ShortHorizonRolloutContract.from_mapping(
            _contract_payload(state_origin="live_policy_on")
        )
    payload = _contract_payload(horizon=41)
    with pytest.raises(ValueError, match="exceeds"):
        ShortHorizonRolloutContract.from_mapping(payload)


def test_step_accepts_nanosecond_timestamps_beyond_float_exact_range() -> None:
    payload = _step_payload(0, origin="teacher_forced")
    payload["observation_timestamp_ns"] = 1_783_939_147_882_339_302
    payload["camera_timestamps_ns"] = {
        camera: 1_783_939_147_881_339_302
        for camera in ("video4", "video5", "video6", "video7")
    }

    step = ShortHorizonRolloutStep.from_mapping(payload)

    assert step.observation_timestamp_ns == 1_783_939_147_882_339_302


def test_cli_writes_audited_negative_control(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": _contract_payload(horizon=2),
        "termination_reason": "horizon_complete",
        "steps": [_step_payload(index, origin="teacher_forced") for index in range(2)],
    }
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_short_horizon_rollout_audit(
        trace_json=trace_path,
        output_dir=tmp_path / "report",
    )

    assert result["trace_integrity_valid"] is True
    assert result["contract_compliant"] is True
    assert (
        result["self_generated_state_evidence"] == "none_noncausal_observation_source"
    )
    report = json.loads(
        (tmp_path / "report/short_horizon_rollout_report.json").read_text()
    )
    assert report["source_manifest"] == "short_horizon_rollout_source_manifest.json"
