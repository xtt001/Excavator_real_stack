import numpy as np
import pytest

from scripts.e39_startup_failure_diagnostics import summarize_window_effectiveness


def test_summarize_window_effectiveness_counts_same_dir_and_extra_wrong() -> None:
    expert_eff = np.zeros((3, 2, 2), dtype=bool)
    policy_eff = np.zeros((3, 2, 2), dtype=bool)
    expert_eff[0, 0, 0] = True
    expert_eff[1, 1, 1] = True
    policy_eff[0, 0, 0] = True
    policy_eff[1, 0, 1] = True
    policy_eff[2, 1, 0] = True

    summary = summarize_window_effectiveness(expert_eff, policy_eff)

    assert summary["expert_effective_frames"] == 2
    assert summary["policy_effective_frames"] == 3
    assert summary["same_dir_frames"] == 1
    assert summary["extra_or_wrong_frames"] == 2
    assert summary["policy_any_effective_pct"] == 100.0
    assert summary["same_dir_pct_of_expert_effective"] == 50.0
    assert summary["extra_or_wrong_pct_of_policy_effective"] == pytest.approx(66.66666666666667)


def test_summarize_window_effectiveness_handles_no_policy_motion() -> None:
    expert_eff = np.zeros((2, 2, 2), dtype=bool)
    policy_eff = np.zeros((2, 2, 2), dtype=bool)
    expert_eff[:, 0, 0] = True

    summary = summarize_window_effectiveness(expert_eff, policy_eff)

    assert summary["expert_effective_frames"] == 2
    assert summary["policy_effective_frames"] == 0
    assert summary["same_dir_frames"] == 0
    assert summary["extra_or_wrong_frames"] == 0
    assert summary["policy_any_effective_pct"] == 0.0
    assert summary["same_dir_pct_of_expert_effective"] == 0.0
    assert summary["extra_or_wrong_pct_of_policy_effective"] == 0.0
