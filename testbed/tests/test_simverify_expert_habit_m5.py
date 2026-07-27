from __future__ import annotations

import copy

import pytest

from testbed.simverify.habit_m5_decision import _validate_evidence


def _scenario(*, targets: tuple[str, str], checkpoint_sha: str) -> dict:
    return {
        "closed_loop_execution": True,
        "physical_effect_validated": False,
        "real_control_candidate": False,
        "privilege_policy_input_scan": {
            "env_state": False,
            "bucket_mass": False,
        },
        "bundle_contract": {
            "artifacts": {
                "policy_best.ckpt": {"sha256": checkpoint_sha},
            }
        },
        "observable_cycle_contract": {
            "observable_cycle_completed": True,
            "requested_cycle_count": 2,
            "completed_cycle_count": 2,
            "cycle_completions": [
                {"realized_target_sector": target} for target in targets
            ],
        },
    }


def _evidence() -> tuple[dict, dict]:
    checkpoint_sha = "b1-checkpoint"
    bundles = {
        baseline: {
            "baseline_id": baseline,
            "artifacts": {
                "policy_best.ckpt": {
                    "sha256": checkpoint_sha
                    if baseline == "B1"
                    else f"{baseline}-checkpoint"
                }
            },
        }
        for baseline in ("B0", "B1", "B2")
    }
    packages = {
        "definition": {
            "payload": {
                "decision": "accept",
                "provenance": {"held_out_observation_read_count": 0},
            }
        },
        "dataset": {
            "payload": {
                "provenance": {
                    "held_out_observation_read_count": 0,
                    "privilege_used": False,
                }
            }
        },
        "validation": {"payload": {"held_out_test_read": False}},
        "offline_gate": {
            "payload": {
                "basic_capability_established_offline": True,
                "condition_understanding_established_offline": False,
                "decision": "condition_understanding_not_established_offline",
                "held_out_test_read": False,
            }
        },
        "paired_branch": {
            "payload": {
                "closed_loop_execution_after_shared_prefix": True,
                "condition_response": {
                    "condition_changes_rollout_above_repeat_variability": True
                },
                "observable_cycle_completion": {
                    "all_branches_completed": True,
                    "branches": {
                        "reference": {
                            "scripted_target_sector": "left",
                            "realized_target_sector": "left",
                        },
                        "repeat": {
                            "scripted_target_sector": "left",
                            "realized_target_sector": "left",
                        },
                        "treatment": {
                            "scripted_target_sector": "center",
                            "realized_target_sector": "center",
                        },
                    },
                },
                "held_out_test_read": False,
                "physical_effect_validated": False,
                "real_control_candidate": False,
            }
        },
        "repeat_same": {
            "payload": _scenario(
                targets=("left", "left"),
                checkpoint_sha=checkpoint_sha,
            )
        },
        "move_adjacent": {
            "payload": _scenario(
                targets=("center", "center"),
                checkpoint_sha=checkpoint_sha,
            )
        },
    }
    return packages, bundles


def test_expert_habit_m5_accepts_closed_loop_observable_evidence() -> None:
    packages, bundles = _evidence()
    _validate_evidence(packages=packages, bundles=bundles)


def test_expert_habit_m5_rejects_wrong_adjacent_endpoint() -> None:
    packages, bundles = _evidence()
    invalid = copy.deepcopy(packages)
    invalid["move_adjacent"]["payload"]["observable_cycle_contract"][
        "cycle_completions"
    ][0]["realized_target_sector"] = "left"
    with pytest.raises(ValueError, match="continuous fixed-scenario"):
        _validate_evidence(packages=invalid, bundles=bundles)
