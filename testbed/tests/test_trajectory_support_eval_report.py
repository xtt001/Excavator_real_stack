from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.trajectory_support_eval import EvalSpec, build_control_actions, run_level1_report


def _write_actions(path: Path, *, expert: np.ndarray, policy: np.ndarray, dt: float = 0.05) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_s=np.arange(expert.shape[0], dtype=np.float64) * dt,
        expert_action=expert.astype(np.float32),
        policy_action=policy.astype(np.float32),
    )


def _thresholds() -> dict[str, object]:
    return {
        "deadzone_action": {
            axis: {"pos": 0.2, "neg": 0.2}
            for axis in ("swing", "boom", "stick", "bucket")
        }
    }


def test_build_control_actions_is_deterministic_and_complete() -> None:
    expert = np.arange(24, dtype=np.float64).reshape(6, 4) / 24.0

    controls = build_control_actions(expert)

    assert list(controls) == [
        "expert",
        "zero",
        "expert_delay1",
        "expert_delay5",
        "expert_sign_flipped",
        "expert_axis_shuffled",
        "expert_scale_0.5",
        "expert_scale_1.5",
    ]
    np.testing.assert_array_equal(controls["zero"], 0.0)
    np.testing.assert_array_equal(controls["expert_delay1"][1:], expert[:-1])
    np.testing.assert_array_equal(controls["expert_delay5"][5:], expert[:-5])
    np.testing.assert_array_equal(controls["expert_axis_shuffled"], expert[:, [1, 2, 3, 0]])


def test_run_level1_report_writes_provenance_aggregate_and_plots(tmp_path: Path) -> None:
    episode_ids = ["episode_1", "episode_2"]
    manifest = tmp_path / "train_ready_manifest.json"
    manifest.write_text(json.dumps({"train_ready_episode_ids": episode_ids}), encoding="utf-8")
    deadzone = tmp_path / "deadzone.json"
    deadzone.write_text(json.dumps(_thresholds()), encoding="utf-8")
    raw_dir = tmp_path / "raw"
    final_dir = tmp_path / "final"
    expert = np.zeros((8, 4), dtype=np.float32)
    expert[1:6, 0] = 0.8
    raw = expert.copy()
    raw[6:, 0] = 0.8
    final = expert.copy()
    for index, episode_id in enumerate(episode_ids):
        shifted = np.roll(expert, index, axis=0)
        _write_actions(
            raw_dir / "episodes" / episode_id / "actions.npz",
            expert=shifted,
            policy=np.roll(raw, index, axis=0),
        )
        _write_actions(
            final_dir / "episodes" / episode_id / "actions.npz",
            expert=shifted,
            policy=np.roll(final, index, axis=0),
        )

    output_dir = tmp_path / "report"
    result = run_level1_report(
        eval_specs=[EvalSpec("raw_act", raw_dir), EvalSpec("candidate", final_dir)],
        manifest_path=manifest,
        deadzone_path=deadzone,
        output_dir=output_dir,
        horizons=(2, 4),
        stride=2,
        bootstrap_samples=100,
        bootstrap_seed=7,
        argv=["trajectory_support_eval.py", "--test-fixture"],
    )

    expected = {
        "run_manifest.json",
        "data_split.json",
        "intent_integral_by_episode.csv",
        "intent_integral_aggregate.csv",
        "candidate_comparison_to_baseline.csv",
        "full_episode_intent_by_episode.csv",
        "summary.json",
        "plots/cumulative_intent_by_axis.png",
        "plots/intent_error_by_horizon.png",
    }
    assert expected == {str(path.relative_to(output_dir)) for path in result.values()}
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["claim_boundary"] == "teacher_forced_level1_only"
    assert summary["episodes"] == 2
    assert summary["models"] == [
        "expert",
        "zero",
        "expert_delay1",
        "expert_delay5",
        "expert_sign_flipped",
        "expert_axis_shuffled",
        "expert_scale_0.5",
        "expert_scale_1.5",
        "raw_act",
        "candidate",
    ]
    aggregate = summary["aggregate"]
    expert_h2 = next(row for row in aggregate if row["model"] == "expert" and row["horizon_steps"] == 2)
    sign_h2 = next(
        row for row in aggregate if row["model"] == "expert_sign_flipped" and row["horizon_steps"] == 2
    )
    assert expert_h2["channel_l1_error_mean"] == 0.0
    assert sign_h2["channel_l1_error_mean"] > 0.0
    assert expert_h2["channel_l1_error_ci95_low"] == 0.0
    assert expert_h2["channel_l1_error_ci95_high"] == 0.0
    comparison = summary["paired_comparison_to_baseline"]
    candidate_h2 = next(
        row
        for row in comparison
        if row["candidate_model"] == "candidate"
        and row["horizon_steps"] == 2
        and row["metric"] == "channel_l1_error"
    )
    assert candidate_h2["baseline_model"] == "raw_act"
    assert candidate_h2["delta_mean"] < 0.0
    assert candidate_h2["episodes_improved_rate"] == 1.0


def test_run_level1_report_rejects_mismatched_expert_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train_ready_episode_ids": ["episode_1"]}), encoding="utf-8")
    deadzone = tmp_path / "deadzone.json"
    deadzone.write_text(json.dumps(_thresholds()), encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"
    expert = np.zeros((5, 4), dtype=np.float32)
    mismatched = expert.copy()
    mismatched[0, 0] = 0.8
    _write_actions(first / "episodes/episode_1/actions.npz", expert=expert, policy=expert)
    _write_actions(second / "episodes/episode_1/actions.npz", expert=mismatched, policy=mismatched)

    with pytest.raises(ValueError, match="expert_action mismatch"):
        run_level1_report(
            eval_specs=[EvalSpec("first", first), EvalSpec("second", second)],
            manifest_path=manifest,
            deadzone_path=deadzone,
            output_dir=tmp_path / "out",
            horizons=(2,),
            stride=1,
            bootstrap_samples=10,
            bootstrap_seed=1,
            argv=["trajectory_support_eval.py", "--test-fixture"],
        )
