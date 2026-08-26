from __future__ import annotations

import copy
from pathlib import Path

from testbed.cli.real_transition_acceptance import evaluate_acceptance


def _contract() -> dict:
    return {
        "schema": "real_transition_target_release_acceptance_contract_v1",
        "authoritative_runtime_mode": "per_goal_reset",
        "planner_open_loop_gates": {
            "all_policy_actions_finite": True,
            "all_safe_actions_finite": True,
            "policy_error_count_max": 0,
            "guard_trigger_count_max": 0,
            "safe_action_mae_max": 0.13,
            "policy_action_sign_agreement_rate_min": 0.9,
            "supported_target_release_pair_hit_rate_min": 0.8,
            "B_release_idle_rate_min": 0.95,
            "validation_dig_positive_effective_rate_min": 0.2,
            "validation_return_negative_effective_rate_min": 0.5,
            "locked_test_dig_positive_effective_rate_min": 0.2,
            "locked_test_return_negative_effective_rate_min": 0.5,
        },
        "state_hold_gates": {
            "same_direction_within_5_rate_min": 0.8,
            "same_direction_within_20_rate_min": 0.8,
            "startup_same_direction_within_20_rate_min": 1.0,
            "query0_wrong_effective_count_max": 0,
        },
        "evidence_boundary": "offline only",
    }


def _cycle() -> dict:
    return {
        "status": "REFERENCE_CYCLE_COMPLETE",
        "safe_action_mae": 0.1,
        "policy_action_sign_agreement_rate": 0.95,
        "policy_dig_positive_effective_rate": 0.25,
        "policy_return_negative_effective_rate": 0.6,
        "policy_release_idle_rate": 1.0,
        "phase_steps": {"dig": 10, "return_approach": 10, "release": 5},
        "steps": 25,
        "supported_target_release_probe": {
            "sample_count": 2,
            "pair_hit_count": 2,
            "B_signs": [0, 0],
        },
        "policy_error_count": 0,
        "guard_trigger_count": 0,
        "all_policy_actions_finite": True,
        "all_safe_actions_finite": True,
    }


def _planner(bundle: Path) -> dict:
    return {
        "reports": [
            {
                "bundle_dir": str(bundle),
                "modes": [
                    {
                        "mode": "per_goal_reset",
                        "runs": [
                            {"split": split, "cycles": [_cycle()]}
                            for split in ("validation", "locked_test")
                        ],
                    }
                ],
            }
        ]
    }


def _state_hold(bundle: Path) -> dict:
    return {
        "bundle_dir": str(bundle),
        "summary": [
            {
                "split": split,
                "group": group,
                "same_direction_within_5_rate": 0.9,
                "same_direction_within_20_rate": 1.0,
                "query0_wrong_effective_count": 0,
            }
            for split in ("validation", "locked_test")
            for group in ("overall", "startup")
        ],
    }


def test_acceptance_passes_only_when_every_frozen_gate_passes(
    tmp_path: Path,
) -> None:
    result = evaluate_acceptance(
        contract=_contract(),
        bundle_dir=tmp_path,
        planner_report=_planner(tmp_path),
        state_hold_report=_state_hold(tmp_path),
    )

    assert result["status"] == "PASS"
    assert result["failed_checks"] == []


def test_acceptance_reports_exact_failed_gate(tmp_path: Path) -> None:
    planner = _planner(tmp_path)
    planner["reports"][0]["modes"][0]["runs"][1]["cycles"][0][
        "policy_return_negative_effective_rate"
    ] = 0.1

    result = evaluate_acceptance(
        contract=_contract(),
        bundle_dir=tmp_path,
        planner_report=planner,
        state_hold_report=_state_hold(tmp_path),
    )

    assert result["status"] == "FAIL"
    assert result["failed_checks"] == [
        "locked_test.return_negative_effective_rate"
    ]


def test_v2_state_hold_uses_same_data_reference_not_window5(tmp_path: Path) -> None:
    contract = _contract()
    contract["schema"] = "real_transition_target_release_acceptance_contract_v2"
    contract["state_hold_gates"] = {
        "validation_same_direction_within_20_rate_min": 0.5,
        "validation_startup_same_direction_within_20_rate_min": 0.5,
        "validation_query0_wrong_effective_count_max": 2,
        "locked_test_same_direction_within_20_rate_min": 0.5,
        "locked_test_startup_same_direction_within_20_rate_min": 0.5,
        "locked_test_query0_wrong_effective_count_max": 2,
    }
    state_hold = copy.deepcopy(_state_hold(tmp_path))
    for row in state_hold["summary"]:
        row["same_direction_within_5_rate"] = 0.0

    result = evaluate_acceptance(
        contract=contract,
        bundle_dir=tmp_path,
        planner_report=_planner(tmp_path),
        state_hold_report=state_hold,
    )

    assert result["status"] == "PASS"
    assert not any("within_5" in check["name"] for check in result["checks"])
