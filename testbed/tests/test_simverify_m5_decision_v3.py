from __future__ import annotations

import pytest

from testbed.simverify.m5_decision_v3 import (
    _validate_chain,
    decision_payload_v3,
)


def _chain() -> dict[str, dict]:
    checkpoint = "c" * 64
    g5_v1_sha = "1" * 64
    g5_1_sha = "2" * 64
    identity = {
        "path": "/artifact",
        "manifest_sha256": "a" * 64,
        "gate_sha256": "b" * 64,
    }
    return {
        "prior_m5": {
            "gate": {
                "decision": "revise_condition",
                "terminal_for_experiment_version": True,
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "manifest": {},
            "identity": identity,
        },
        "g4": {
            "gate": {
                "decision": "next_condition_understanding_established_offline",
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "manifest": {
                "candidate_baseline_id": "B1.4",
                "decision": "next_condition_understanding_established_offline",
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "identity": identity,
        },
        "g5_v1": {
            "gate": {
                "decision": "g5_core_two_cycle_condition_continuity_not_established",
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "manifest": {
                "authorizes_remaining_g5_robustness": False,
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "identity": {**identity, "manifest_sha256": g5_v1_sha},
        },
        "g5_1": {
            "gate": {
                "decision": (
                    "g5_core_two_cycle_condition_continuity_established_development"
                ),
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "manifest": {
                "authorizes_remaining_g5_robustness": True,
                "previous_g5_core": {"manifest_sha256": g5_v1_sha},
                "bundles": {"B1.4": {"checkpoint_sha256": checkpoint}},
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "identity": {**identity, "manifest_sha256": g5_1_sha},
        },
        "e04": {
            "gate": {
                "decision": "e04_camera_counterfactual_robustness_not_established",
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "manifest": {
                "authorizes_e05": False,
                "previous_g5": {"manifest_sha256": g5_1_sha},
                "bundle": {"checkpoint_sha256": checkpoint},
                "source_episode_ids": [12, 34],
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "identity": identity,
        },
    }


def test_m5_v3_closes_after_e04_without_downstream_authorization() -> None:
    chain = _chain()
    _validate_chain(**chain)
    decision = decision_payload_v3(**chain)
    assert decision["decision"] == "revise_condition"
    assert decision["terminal_for_experiment_version"] is True
    assert decision["held_out_test_authorized"] is False
    assert decision["sim_observable_only"] is False
    assert decision["real_finetune_candidate"] is False
    assert decision["control_candidate"] is False
    gate_path = {row["gate"]: row["result"] for row in decision["gate_path"]}
    assert gate_path["G4"].startswith("next_condition_understanding_established")
    assert gate_path["G5_1"].startswith("core_two_cycle_continuity_established")
    assert gate_path["E04"] == "camera_counterfactual_robustness_not_established"
    assert gate_path["E05"] == "not_entered"
    assert gate_path["G6"] == "not_entered"


def test_m5_v3_rejects_e04_held_out_overlap() -> None:
    chain = _chain()
    chain["e04"]["manifest"]["source_episode_ids"] = [12, 13]
    with pytest.raises(ValueError, match="held-out episode"):
        _validate_chain(**chain)
