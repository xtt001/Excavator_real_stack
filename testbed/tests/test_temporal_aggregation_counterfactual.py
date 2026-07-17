from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from testbed.cli.audit_temporal_aggregation_counterfactual import (
    run_temporal_aggregation_counterfactual_audit,
)
from testbed.policies.action_start_distribution import sha256_file
from testbed.policies.state_hold_demo_relation import (
    evaluate_state_hold_trace_demo_relation,
)
from testbed.policies.temporal_aggregation_counterfactual import (
    evaluate_temporal_aggregation_counterfactual,
)

THRESHOLDS = {
    axis: {"pos": 0.5, "neg": 0.5}
    for axis in ("swing", "boom", "stick", "bucket")
}
ZERO = [0.0, 0.0, 0.0, 0.0]
BOOM_POS = [0.0, 0.8, 0.0, 0.0]


def test_evaluator_counts_rescue_regression_and_delay_changes() -> None:
    rows = [
        _row(
            anchor_step=10,
            legacy=[ZERO, ZERO, ZERO],
            newest=[ZERO, BOOM_POS, BOOM_POS],
            recency=[ZERO, ZERO, ZERO],
        ),
        _row(
            anchor_step=20,
            legacy=[ZERO, BOOM_POS, BOOM_POS],
            newest=[ZERO, ZERO, ZERO],
            recency=[BOOM_POS, BOOM_POS, BOOM_POS],
        ),
        _row(
            anchor_step=30,
            legacy=[BOOM_POS, BOOM_POS, BOOM_POS],
            newest=[BOOM_POS, BOOM_POS, BOOM_POS],
            recency=[ZERO, BOOM_POS, BOOM_POS],
        ),
    ]

    report = evaluate_temporal_aggregation_counterfactual(
        rows_by_model={"candidate": rows}, thresholds=THRESHOLDS
    )
    aggregate = report["aggregate"]["candidate"]

    assert aggregate["modes"]["legacy"]["demo_target_reproduced_anchors"] == 2
    assert aggregate["modes"]["newest"]["demo_target_reproduced_anchors"] == 2
    assert aggregate["modes"]["recency"]["demo_target_reproduced_anchors"] == 2
    newest = aggregate["comparisons"]["newest_vs_legacy"]
    assert newest[
        "legacy_nonreproduction_changed_to_reproduction_anchor_ids"
    ] == [
        "episode_74:10:boom+"
    ]
    assert newest[
        "legacy_reproduction_changed_to_nonreproduction_anchor_ids"
    ] == [
        "episode_74:20:boom+"
    ]
    assert newest["same_reproduction_delay_anchor_ids"] == [
        "episode_74:30:boom+"
    ]
    recency = aggregate["comparisons"]["recency_vs_legacy"]
    assert recency["alternative_faster_anchor_ids"] == [
        "episode_74:20:boom+"
    ]
    assert recency["alternative_slower_anchor_ids"] == [
        "episode_74:30:boom+"
    ]
    assert recency["reproduction_delay_delta_ticks"]["mean"] == 0.0


def test_evaluator_tracks_hidden_teacher_forcing_per_mode() -> None:
    row = _row(
        legacy=[ZERO, ZERO, ZERO],
        newest=[ZERO, BOOM_POS, BOOM_POS],
        recency=[ZERO, ZERO, ZERO],
        teacher_status="demo_target_reproduced",
    )

    report = evaluate_temporal_aggregation_counterfactual(
        rows_by_model={"candidate": [row]}, thresholds=THRESHOLDS
    )
    modes = report["aggregate"]["candidate"]["modes"]

    field = "demo_target_reproduction_hidden_by_stored_teacher_forcing_anchors"
    assert modes["legacy"][field] == 1
    assert modes["newest"][field] == 0
    assert modes["recency"][field] == 1


def test_evaluator_reuses_full_horizon_safety_metrics() -> None:
    unsafe_newest = [
        [0.8, 0.8, 0.0, 0.0],
        [0.0, -0.8, 0.0, 0.0],
        [0.0, 0.8, 0.0, 0.0],
    ]
    row = _row(
        legacy=[BOOM_POS, BOOM_POS, BOOM_POS],
        newest=unsafe_newest,
        recency=[BOOM_POS, BOOM_POS, BOOM_POS],
    )

    report = evaluate_temporal_aggregation_counterfactual(
        rows_by_model={"candidate": [row]}, thresholds=THRESHOLDS
    )
    newest = report["per_anchor"][0]["modes"]["newest"]

    assert newest["anchor_extra_effective_tick_count"] == 2
    assert newest["anchor_extra_effective_direction_activation_count"] == 2
    assert newest["anchor_extra_effective_direction_tick_counts"]["swing+"] == 1
    assert newest["anchor_extra_effective_direction_tick_counts"]["boom-"] == 1
    assert newest["anchor_extra_effective_axis_tick_counts"]["swing"] == 1
    assert newest["anchor_extra_effective_axis_tick_counts"]["boom"] == 1
    assert newest["opposite_to_demo_target_tick_count"] == 1
    assert newest["direction_flip_count"] == 2
    assert newest["max_effective_axes"] == 2


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda row: row["state_hold_diagnostics_trace"][0].pop(
                "policy_temporal_aggregation_newest_action"
            ),
            "missing field",
        ),
        (
            lambda row: row.__setitem__(
                "state_hold_temporal_aggregation_decomposition_complete", False
            ),
            "decomposition is incomplete",
        ),
        (
            lambda row: row["state_hold_diagnostics_trace"].pop(),
            "diagnostics_trace is incomplete",
        ),
        (
            lambda row: row["state_hold_diagnostics_trace"][0].__setitem__(
                "policy_temporal_aggregation_source_steps",
                [
                    row["state_hold_diagnostics_trace"][0][
                        "policy_temporal_aggregation_query_step"
                    ]
                    + 1
                ],
            ),
            "noncausal source steps",
        ),
    ],
)
def test_evaluator_rejects_missing_incomplete_or_noncausal_traces(
    mutation: Any, match: str
) -> None:
    row = _row()
    mutation(row)

    with pytest.raises(ValueError, match=match):
        evaluate_temporal_aggregation_counterfactual(
            rows_by_model={"candidate": [row]}, thresholds=THRESHOLDS
        )


def test_evaluator_rejects_heldout_episode() -> None:
    row = _row(episode_id="episode_105")

    with pytest.raises(ValueError, match="held-out episode is forbidden"):
        evaluate_temporal_aggregation_counterfactual(
            rows_by_model={"candidate": [row]}, thresholds=THRESHOLDS
        )


def test_cli_writes_hashed_trace_and_checkpoint_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "source_run"
    trace_dir = run_dir / "assist_disabled"
    bundle_dir = tmp_path / "bundle"
    trace_dir.mkdir(parents=True)
    bundle_dir.mkdir()
    trace_path = trace_dir / "state_hold_anchors.jsonl"
    trace_path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    deadzone_path = run_dir / "resolved_direct_output_deadzone.json"
    deadzone_path.write_text(
        json.dumps({"deadzone_action": THRESHOLDS}) + "\n", encoding="utf-8"
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("policy: {}\n", encoding="utf-8")
    for name in (
        "policy_best.ckpt",
        "dataset_stats.pkl",
        "resolved_config.yaml",
        "run_metadata.json",
    ):
        (bundle_dir / name).write_text(name + "\n", encoding="utf-8")
    run_summary = {
        "candidate_id": "candidate",
        "pipeline_mode": "raw",
        "config_path": str(config_path),
        "action_bundle_dir": str(bundle_dir),
        "episode_ids": ["episode_74"],
        "hold_horizon_steps": 3,
        "trace_full_horizon_after_reproduction": True,
        "temporal_aggregation_decomposition": True,
        "assist_mode": "disabled",
        "resolved_direct_output_deadzone": str(deadzone_path),
        "deadzone_provenance": {
            "action_domain": "direct_policy_output",
            "policy_action_scale": [1.0, 1.0, 1.0, 1.0],
            "legacy_raw_scaled_deadzone_reused": False,
        },
        "reports": [
            {
                "mode": "assist_disabled",
                "pipeline_mode": "raw",
                "assist_enabled": False,
                "anchor_rows": 1,
                "paths": {"rows_jsonl": str(trace_path)},
            }
        ],
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(run_summary) + "\n", encoding="utf-8"
    )

    result = run_temporal_aggregation_counterfactual_audit(
        trace_paths={"candidate": trace_path},
        deadzone_json=deadzone_path,
        expected_anchors=1,
        output_dir=tmp_path / "output",
    )

    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    source = report["inputs"]["models"]["candidate"]
    assert source["trace"]["sha256"] == sha256_file(trace_path)
    assert source["bundle_artifacts"]["policy_best.ckpt"]["sha256"] == (
        sha256_file(bundle_dir / "policy_best.ckpt")
    )
    assert report["heldout_evaluated"] is False
    assert report["mechanical_assist"]["estimated"] is False
    assert report["limitations"]["scope"].startswith(
        "this is an exact command counterfactual"
    )
    for key in (
        "report",
        "source_manifest",
        "per_anchor_jsonl",
        "per_anchor_csv",
        "mode_aggregate_csv",
        "pairwise_aggregate_csv",
    ):
        assert Path(result[key]).is_file()
        assert result[f"{key}_sha256"] == sha256_file(result[key])


def _row(
    *,
    episode_id: str = "episode_74",
    anchor_step: int = 10,
    legacy: list[list[float]] | None = None,
    newest: list[list[float]] | None = None,
    recency: list[list[float]] | None = None,
    teacher_status: str = "demo_target_reproduced",
) -> dict[str, Any]:
    legacy = copy.deepcopy(legacy or [ZERO, ZERO, ZERO])
    newest = copy.deepcopy(newest or legacy)
    recency = copy.deepcopy(recency or legacy)
    horizon = len(legacy)
    assert len(newest) == horizon and len(recency) == horizon
    effective_ticks = [
        tick for tick, action in enumerate(legacy) if action[1] > 0.5
    ]
    delay = effective_ticks[0] if effective_ticks else None
    demo_relation = evaluate_state_hold_trace_demo_relation(
        expert_action=np.asarray(BOOM_POS, dtype=np.float32),
        action_trace=legacy,
        thresholds=THRESHOLDS,
        target_axis_index=1,
        target_direction="pos",
    )
    diagnostics = []
    for tick in range(horizon):
        query_step = anchor_step + tick
        diagnostics.append(
            {
                "policy_temporal_aggregation_action_domain": (
                    "direct_policy_output"
                ),
                "policy_temporal_aggregation_exponential_k": 0.01,
                "policy_action_scale": [1.0, 1.0, 1.0, 1.0],
                "policy_deadzone_assist_enabled": 0,
                "policy_error": "",
                "policy_temporal_aggregation_query_step": query_step,
                "policy_temporal_aggregation_source_steps": [query_step],
                "policy_temporal_aggregation_query_offsets": [0],
                "policy_temporal_aggregation_population": 1,
                "policy_temporal_aggregation_legacy_action": legacy[tick],
                "policy_temporal_aggregation_newest_action": newest[tick],
                "policy_temporal_aggregation_recency_action": recency[tick],
            }
        )
    return {
        "episode_id": episode_id,
        "anchor_step": anchor_step,
        "anchor_group": "test",
        "axis_index": 1,
        "axis": "boom",
        "direction": "pos",
        "deadzone_threshold": 0.5,
        "expert_action_vector": BOOM_POS,
        "hold_horizon_steps": horizon,
        "trace_full_horizon_after_reproduction": True,
        "state_hold_qvel_zero": True,
        "teacher_forced_status": teacher_status,
        "state_hold_status": (
            "demo_target_reproduced"
            if delay is not None
            else "demo_target_not_reproduced"
        ),
        "state_hold_demo_target_not_reproduced": delay is None,
        "state_hold_demo_target_reproduction_delay_ticks": delay,
        "demo_target_reproduction_hidden_by_teacher_forcing": (
            teacher_status == "demo_target_reproduced" and delay is None
        ),
        "state_hold_ticks_evaluated": horizon,
        "state_hold_full_horizon_complete": True,
        "state_hold_action_trace": legacy,
        "state_hold_diagnostics_trace": diagnostics,
        "state_hold_temporal_aggregation_decomposition_complete": True,
        "state_hold_temporal_aggregation_decomposition_ticks_recorded": horizon,
        **demo_relation,
    }
