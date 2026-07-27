from __future__ import annotations

import numpy as np

from testbed.simverify.habit_agx_branch_eval import (
    compute_branch_effects,
    summarize_observable_completions,
)


def test_branch_effect_separates_condition_from_repeat_variability() -> None:
    reference = np.zeros((5, 4), dtype=np.float64)
    repeat = reference.copy()
    treatment = reference.copy()
    repeat[2:] = 0.01
    treatment[2:] = np.asarray([0.1, 0.2, 0.3, 0.4])

    result = compute_branch_effects(
        reference,
        repeat,
        treatment,
        takeover_tick=2,
    )

    np.testing.assert_allclose(result["repeat_mean_abs_delta"], [0.01] * 4)
    np.testing.assert_allclose(
        result["treatment_mean_abs_delta"],
        [0.1, 0.2, 0.3, 0.4],
    )
    np.testing.assert_allclose(
        result["treatment_to_repeat_mean_abs_ratio"],
        [10.0, 20.0, 30.0, 40.0],
    )
    assert result["treatment_exceeds_repeat_variability_all_axes"] is True


def test_v11_observable_completion_requires_target_match_in_all_branches() -> None:
    records = {
        role: {
            "target_sector": target,
            "observable_cycle": {
                "ready_detection_enabled": True,
                "observable_cycle_completed": True,
                "completion_policy_tick": 300,
                "scripted_target_sector": target,
                "realized_target_sector": target,
                "physical_effect_validated": False,
            },
        }
        for role, target in {
            "reference": "left",
            "repeat": "left",
            "treatment": "center",
        }.items()
    }
    result = summarize_observable_completions(records)
    assert result["all_branches_completed"] is True
    assert result["physical_effect_validated"] is False

    records["treatment"]["observable_cycle"]["realized_target_sector"] = "left"
    failed = summarize_observable_completions(records)
    assert failed["all_branches_completed"] is False
    assert failed["branches"]["treatment"]["status"] == "not_completed"
