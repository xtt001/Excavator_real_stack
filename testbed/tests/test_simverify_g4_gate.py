from __future__ import annotations

import numpy as np

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
