from __future__ import annotations

from pathlib import Path

import pytest

from testbed.simverify.artifacts import write_checksums, write_json
from testbed.simverify.m5_decision import (
    _condition_summary,
    _decision_payload,
    _validate_replay_reference,
)


def test_m5_decision_keeps_downstream_and_deployment_locked() -> None:
    identity = {"path": "/evidence", "manifest_sha256": "a" * 64}
    condition_gates = [
        {
            "baseline_id": baseline,
            "decision": "condition_understanding_not_established",
            "factor_pass": {
                "current_sector": baseline == "B1",
                "next_sector": False,
            },
            "source_episode_summary": {},
            "identity": {**identity, "baseline_id": baseline},
        }
        for baseline in ("B1", "B1.1", "B1.2")
    ]

    decision = _decision_payload(
        m0={"identity": identity},
        m1={"identity": identity},
        m2={"identity": identity},
        g3={"identity": identity},
        condition_gates=condition_gates,
    )

    assert decision["decision"] == "revise_condition"
    assert decision["terminal_for_experiment_version"] is True
    assert decision["held_out_test_read"] is False
    assert decision["held_out_test_authorized"] is False
    assert decision["control_candidate"] is False
    assert decision["sim_observable_only"] is False
    assert decision["real_finetune_candidate"] is False
    path = {row["gate"]: row["result"] for row in decision["gate_path"]}
    assert path["G4"] == "revise_condition"
    assert path["G5"] == "not_entered"
    assert path["G6"] == "not_entered"


def test_condition_summary_uses_source_episode_means_and_permutations() -> None:
    def factor(values: list[float], passed: bool) -> dict:
        permutations = {
            f"p{index}": {"passed": index < 3} for index in range(5)
        }
        return {
            "action_sensitivity_vs_masked": {
                "b1_source_episode_values": values,
            },
            "signed_semantic_margin_vs_b2": {
                "b1_source_episode_values": [value * 2 for value in values],
            },
            "phase_specificity": {
                "positive_vs_masked": {
                    "b1_source_episode_values": [value * 3 for value in values],
                }
            },
            "semantic_identifiability": {
                "permutation_results": permutations,
            },
            "passed": passed,
        }

    gate = {
        "factor_pass": {
            "current_sector": True,
            "next_sector": False,
        },
        "criteria": {
            "current_sector": factor([1.0, 3.0], True),
            "next_sector": factor([2.0, 4.0], False),
        },
    }
    summary = _condition_summary(gate)

    assert summary["current_sector"]["action_effect_mean"] == 2.0
    assert summary["current_sector"]["signed_semantic_margin_mean"] == 4.0
    assert summary["current_sector"]["phase_specificity_mean"] == 6.0
    assert summary["current_sector"]["semantic_permutations_rejected"] == 3
    assert summary["next_sector"]["factor_pass"] is False


def test_replay_reference_accepts_g3_nested_provenance_and_rejects_heldout(
    tmp_path: Path,
) -> None:
    m0_sha = "1" * 64
    m2_sha = "2" * 64

    def package(name: str, episode_id: int) -> tuple[Path, dict]:
        root = tmp_path / name
        manifest_identity = write_json(
            root / "replay_manifest.json",
            {
                "schema": "simverify_b0_replay_manifest_v1",
                "episode_ids": [episode_id],
                "held_out_test_read": False,
                "closed_loop_execution": False,
                "provenance": {
                    "evidence_scope": "recorded-observation/offline",
                    "m0_dataset_manifest_sha256": m0_sha,
                    "m2_manifest_sha256": m2_sha,
                },
            },
        )
        checksums_identity = write_checksums(
            root,
            [manifest_identity],
            path=root / "checksums.sha256",
        )
        record = {
            "path": str(root),
            "manifest_sha256": manifest_identity["sha256"],
            "checksums_sha256": checksums_identity["sha256"],
        }
        return root, record

    _root, valid = package("valid", 12)
    result = _validate_replay_reference(
        valid,
        manifest_name="replay_manifest.json",
        m0_manifest_sha256=m0_sha,
        m2_manifest_sha256=m2_sha,
        replay_cache={},
    )
    assert result["manifest_sha256"] == valid["manifest_sha256"]

    _root, heldout = package("heldout", 1)
    with pytest.raises(ValueError, match="evidence boundary"):
        _validate_replay_reference(
            heldout,
            manifest_name="replay_manifest.json",
            m0_manifest_sha256=m0_sha,
            m2_manifest_sha256=m2_sha,
            replay_cache={},
        )
