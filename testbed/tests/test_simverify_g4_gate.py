from __future__ import annotations

import numpy as np

from testbed.simverify.m3_condition_causal_v2 import (
    _semantic_permutations,
    phase_specificity,
    signed_semantic_margin,
)
from testbed.simverify.m3_condition_gate import (
    _formula_audit,
    first_effect_latency,
    paired_metric_result,
    symmetric_trace_consistency,
)
from testbed.simverify.m3_condition_replay import (
    _most_frequent_train_condition,
)


def test_g4_formula_audit_corrects_double_null_and_rate_units() -> None:
    audit = _formula_audit()
    assert audit["result_independent"] is True
    assert audit["held_out_test_read"] is False
    assert "counts the null twice" in audit["findings"][0]["old_formula_issue"]
    assert "dimensionless rate" in audit["findings"][1]["old_formula_issue"]
    assert audit["findings"][0]["b2_null_still_used"] is True


def test_paired_metric_uses_repeat_noise_as_zero_margin() -> None:
    result = paired_metric_result(
        np.asarray([0.30, 0.40, 0.35]),
        np.asarray([0.10, 0.10, 0.10]),
        repeat_noise=0.01,
        lower_is_better=False,
        repetitions=10_000,
        seed=7,
    )
    assert result["passed"] is True
    assert result["paired_bootstrap"]["p02_5"] > 0.01


def test_lower_is_better_paired_metric_requires_negative_margin() -> None:
    result = paired_metric_result(
        np.asarray([0.0, 0.0, 0.1]),
        np.asarray([0.8, 0.7, 0.9]),
        repeat_noise=0.02,
        lower_is_better=True,
        repetitions=10_000,
        seed=8,
    )
    assert result["passed"] is True
    assert result["paired_bootstrap"]["p97_5"] < -0.02


def test_latency_censors_after_window_when_no_effect_exceeds_noise() -> None:
    assert first_effect_latency([0.0, 0.01, 0.03], noise_floor=0.02) == 2
    assert first_effect_latency([0.0, 0.01], noise_floor=0.02) == 3


def test_symmetric_trace_consistency_is_bounded_and_exact() -> None:
    reference = np.asarray([[1.0, -1.0], [0.0, 0.0]])
    assert symmetric_trace_consistency(reference, reference.copy()) == 1.0
    assert symmetric_trace_consistency(reference, -reference) == 0.0
    assert symmetric_trace_consistency(np.zeros(2), np.zeros(2)) == 1.0


def test_masked_condition_is_fit_from_train_frequency_only() -> None:
    def row(split: str, current: str, next_sector: str, vector: list[int]):
        return {
            "split": split,
            "quality": {"status": "accepted"},
            "policy_condition": {
                "current_sector": current,
                "next_ready_sector": next_sector,
                "vector": vector,
            },
        }

    selected = _most_frequent_train_condition(
        [
            row("train", "left", "center", [1, 0, 0, 0, 1, 0]),
            row("train", "right", "left", [0, 0, 1, 1, 0, 0]),
            row("train", "right", "left", [0, 0, 1, 1, 0, 0]),
            row("validation", "center", "right", [0, 1, 0, 0, 0, 1]),
        ]
    )
    assert selected["current_sector"] == "right"
    assert selected["next_sector"] == "left"
    assert selected["selected_count"] == 2


def test_signed_semantic_margin_changes_under_wrong_label_mapping() -> None:
    row = {
        "changed_factor": "current_sector",
        "base_condition": {"current_sector": "left"},
        "target_condition": {"current_sector": "right"},
        "metrics": {"swing_action_delta_mean": 0.2},
    }
    centers = {"left": -1.0, "center": 0.0, "right": 1.0}
    identity = {"left": "left", "center": "center", "right": "right"}
    reversed_mapping = {"left": "right", "center": "center", "right": "left"}
    assert (
        signed_semantic_margin(
            row,
            semantic_mapping=identity,
            sector_centers=centers,
            action_direction_sign=1,
        )
        == 0.2
    )
    assert (
        signed_semantic_margin(
            row,
            semantic_mapping=reversed_mapping,
            sector_centers=centers,
            action_direction_sign=1,
        )
        == -0.2
    )


def test_phase_specificity_separates_intended_and_off_window() -> None:
    row = {
        "metrics": {
            "per_tick_effect_l1": [0.1, 0.5, 0.5, 0.1],
            "relevant_window_local": [1, 3],
        }
    }
    assert phase_specificity(row) == 0.4


def test_exactly_five_non_identity_sector_permutations() -> None:
    mappings = _semantic_permutations()
    assert len(mappings) == 5
    assert all(set(mapping) == {"left", "center", "right"} for mapping in mappings)
    assert all(len(set(mapping.values())) == 3 for mapping in mappings)
