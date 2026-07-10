import numpy as np
import pytest

from scripts.e42_startup_intent_overlap_diagnostic import summarize_startup_intent_overlap


def test_summarize_startup_intent_overlap_reports_high_intent_extra() -> None:
    expert_eff = np.zeros((3, 2, 2), dtype=bool)
    policy_eff = np.zeros((3, 2, 2), dtype=bool)
    intent_prob = np.zeros((3, 4), dtype=np.float32)

    expert_eff[0, 0, 0] = True
    policy_eff[0, 0, 0] = True
    policy_eff[1, 0, 0] = True
    expert_eff[2, 1, 1] = True

    intent_prob[0, 0] = 0.85
    intent_prob[1, 0] = 0.82
    intent_prob[2, 3] = 0.91

    summary = summarize_startup_intent_overlap(
        expert_eff,
        policy_eff,
        intent_prob,
        high_intent_thresholds=(0.7, 0.9),
    )

    assert summary["same_count"] == 1
    assert summary["extra_count"] == 1
    assert summary["missing_count"] == 1
    assert summary["same_intent_mean"] == pytest.approx(0.85)
    assert summary["extra_intent_mean"] == pytest.approx(0.82)
    assert summary["missing_intent_mean"] == pytest.approx(0.91)
    assert summary["extra_intent_ge_0.70_pct"] == 100.0
    assert summary["extra_intent_ge_0.90_pct"] == 0.0
