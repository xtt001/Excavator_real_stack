from __future__ import annotations

import numpy as np

from testbed.simverify.habit_agx_branch_eval import compute_branch_effects


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
