from __future__ import annotations

import pytest

from testbed.simverify.m5_decision_v4 import (
    _camera_summary,
    decision_payload_v4,
    validate_chain_v4,
)


def _identity(manifest_sha: str = "a" * 64) -> dict[str, object]:
    return {
        "path": "/artifact",
        "manifest_sha256": manifest_sha,
        "gate_sha256": "b" * 64,
        "checksums_sha256": "c" * 64,
    }


def _chain() -> dict[str, dict]:
    candidate_checkpoint = "d" * 64
    null_checkpoint = "e" * 64
    g5_v1_sha = "1" * 64
    g5_1_sha = "2" * 64
    prior_manifest_sha = "3" * 64
    prior_checksums_sha = "4" * 64
    identity = _identity()
    checkpoint_contract = {
        "checkpoint_semantics": {"real_control_allowed": False},
        "experiment_contract": {
            "m5_v3_manifest_sha256": prior_manifest_sha,
            "m5_v3_checksums_sha256": prior_checksums_sha,
        },
    }
    criteria = {
        "four_camera": {
            "passed": True,
            "semantic_margin_min_source_mean": 0.03,
        },
        "drop_video7": {
            "passed": False,
            "semantic_margin_min_source_mean": 0.02,
        },
    }
    return {
        "prior_m5": {
            "gate": {
                "schema": "simverify_m5_decision_v3",
                "experiment_version": "B1.4_G5.1_E04",
                "decision": "revise_condition",
                "terminal_for_experiment_version": True,
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "manifest": {},
            "identity": {
                **identity,
                "manifest_sha256": prior_manifest_sha,
                "checksums_sha256": prior_checksums_sha,
            },
        },
        "g4": {
            "gate": {
                "decision": "next_condition_understanding_established_offline",
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "manifest": {
                "candidate_baseline_id": "B1.5",
                "null_baseline_id": "B2.5",
                "decision": "next_condition_understanding_established_offline",
                "condition_replay_packages": [
                    {
                        "baseline_id": "B1.5",
                        "checkpoint_sha256": candidate_checkpoint,
                    },
                    {
                        "baseline_id": "B2.5",
                        "checkpoint_sha256": null_checkpoint,
                    },
                ],
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
                "candidate_baseline_id": "B1.5",
                "null_baseline_id": "B2.5",
                "authorizes_remaining_g5_robustness": True,
                "previous_g5_core": {"manifest_sha256": g5_v1_sha},
                "bundles": {
                    "B1.5": {
                        "checkpoint_sha256": candidate_checkpoint,
                        "checkpoint_contract": checkpoint_contract,
                    },
                    "B2.5": {
                        "checkpoint_sha256": null_checkpoint,
                        "checkpoint_contract": {
                            "checkpoint_semantics": {
                                "real_control_allowed": False
                            },
                            "experiment_contract": {},
                        },
                    },
                },
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "identity": {**identity, "manifest_sha256": g5_1_sha},
        },
        "e04": {
            "gate": {
                "decision": "e04_camera_counterfactual_robustness_not_established",
                "criteria": criteria,
                "source_episode_metrics": [
                    {
                        "camera_variant": "drop_video7",
                        "episode_id": 12,
                        "semantic_margin_mean": 0.04,
                        "condition_effect_mean": 0.003,
                        "phase_coverage_mean": 1.0,
                        "failure_rate": 0.0,
                    }
                ],
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "manifest": {
                "candidate_baseline_id": "B1.5",
                "decision": "e04_camera_counterfactual_robustness_not_established",
                "authorizes_e05": False,
                "previous_g5": {"manifest_sha256": g5_1_sha},
                "bundle": {"checkpoint_sha256": candidate_checkpoint},
                "source_episode_ids": [12, 34],
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
            "identity": identity,
        },
    }


def test_m5_v4_closes_b1_5_after_e04_stop() -> None:
    chain = _chain()
    checkpoints = validate_chain_v4(**chain)
    decision = decision_payload_v4(**chain, checkpoints=checkpoints)
    assert decision["decision"] == "revise_condition"
    assert decision["terminal_for_experiment_version"] is True
    assert decision["held_out_test_authorized"] is False
    assert decision["sim_observable_only"] is False
    assert decision["real_finetune_candidate"] is False
    assert decision["control_candidate"] is False
    summary = decision["camera_counterfactual_summary"]
    assert summary["semantic_direction_positive_for_all_variants"] is True
    assert summary["drop_video7_frozen_criterion_passed"] is False
    assert summary["passing_variant_count"] == 1
    assert summary["failing_variant_count"] == 1


def test_m5_v4_rejects_candidate_checkpoint_mismatch() -> None:
    chain = _chain()
    chain["e04"]["manifest"]["bundle"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="B1.5 checkpoint differs"):
        validate_chain_v4(**chain)


def test_m5_v4_rejects_held_out_e04_source() -> None:
    chain = _chain()
    chain["e04"]["manifest"]["source_episode_ids"] = [12, 13]
    with pytest.raises(ValueError, match="held-out episode"):
        validate_chain_v4(**chain)


def test_camera_summary_keeps_positive_semantics_separate_from_gate() -> None:
    chain = _chain()
    summary = _camera_summary(chain["e04"]["gate"])
    assert summary["complete_e04_passed"] is False
    assert summary["semantic_direction_positive_for_all_variants"] is True
    assert summary["failing_variants"] == ["drop_video7"]
