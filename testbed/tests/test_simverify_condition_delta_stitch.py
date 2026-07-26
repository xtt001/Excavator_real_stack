from __future__ import annotations

from testbed.simverify.m3_condition_delta_stitch import (
    aggregate_condition_metrics_by_episode,
    build_paired_condition_metrics,
    evaluate_condition_delta_stitch_gate,
)


def _result(
    baseline: str,
    request: str,
    similarity: dict[str, float],
) -> dict[str, object]:
    return {
        "anchor_index": 0,
        "episode_id": 12,
        "cycle_id": 4,
        "baseline_id": baseline,
        "request_name": request,
        "base_condition": {"next_sector": "left"},
        "target_condition": {"next_sector": "right"},
        "endpoint_sector_similarity": similarity,
        "trace_sha256": f"{baseline}-{request}",
        "selected_transition_index_sha256": f"{baseline}-{request}",
        "completed": True,
        "selected_transition_count": 5,
        "unique_selected_transition_count": 5,
    }


def test_endpoint_semantic_gate_prefers_b1_over_shuffled_b2() -> None:
    rows = [
        _result("B1.4", "base", {"left": 1.0, "center": 0.0, "right": 0.0}),
        _result("B1.4", "target", {"left": 0.0, "center": 0.0, "right": 1.0}),
        _result("B2.4", "base", {"left": 0.6, "center": 0.0, "right": 0.4}),
        _result("B2.4", "target", {"left": 0.4, "center": 0.0, "right": 0.6}),
    ]
    paired = build_paired_condition_metrics(rows)
    source = aggregate_condition_metrics_by_episode(paired)
    gate = evaluate_condition_delta_stitch_gate(rows, source)
    assert gate["supported_path_effect_established"] is True
    assert source[0]["b1_4_endpoint_semantic_score_mean"] == 1.0
    assert source[0]["b1_4_minus_b2_4_semantic_score"] > 0.0
