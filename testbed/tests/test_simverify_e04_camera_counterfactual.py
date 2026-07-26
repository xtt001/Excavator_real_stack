from __future__ import annotations

from pathlib import Path

import numpy as np

from testbed.simverify.e04_camera_counterfactual import (
    aggregate_e04_by_source,
    camera_pair_failed,
    cross_process_replay_noise,
    derive_e04_thresholds,
    evaluate_e04_gate,
)
from testbed.simverify.g5_two_cycle_replay import CAMERA_VARIANTS


def _pair(
    episode_id: int,
    anchor_index: int,
    variant: str,
    *,
    repeat: int = 0,
    effect: float = 0.03,
    coverage: float = 0.95,
    semantic: float = 0.02,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "anchor_index": anchor_index,
        "camera_variant": variant,
        "repeat": repeat,
        "condition_switch_action_effect": effect,
        "two_cycle_phase_coverage": coverage,
        "route2_semantic_margin": semantic,
        "event_order_valid": True,
        "ready_boundary_discontinuity": 0.02,
        "second_cycle_route0_tick_count": 10,
        "second_cycle_route2_tick_count": 5,
        "shared_ready_boundary_route_index": 0,
        "condition_cycle_router_reset_count": 1,
    }


def _threshold_rows() -> list[dict[str, object]]:
    rows = []
    for episode_id, effect, coverage in ((12, 0.03, 0.95), (34, 0.04, 0.98)):
        rows.append(
            _pair(episode_id, 0, "four_camera", effect=effect, coverage=coverage)
        )
        rows.append(
            _pair(
                episode_id,
                0,
                "four_camera",
                repeat=1,
                effect=effect,
                coverage=coverage,
            )
        )
    return rows


def test_e04_thresholds_use_matched_four_camera_repeats() -> None:
    thresholds = derive_e04_thresholds(_threshold_rows())
    assert 0.03 <= thresholds["condition_effect_lower"] <= 0.04
    assert 0.95 <= thresholds["phase_coverage_lower"] <= 0.98
    assert thresholds["failure_rate_upper"] == 0.0


def test_camera_pair_failure_checks_semantic_and_lifecycle() -> None:
    thresholds = {
        "condition_effect_lower": 0.02,
        "phase_coverage_lower": 0.9,
    }
    assert not camera_pair_failed(
        _pair(12, 0, "eye_only"),
        thresholds=thresholds,
    )
    assert camera_pair_failed(
        _pair(12, 0, "eye_only", semantic=-0.01),
        thresholds=thresholds,
    )


def test_e04_gate_requires_every_frozen_variant() -> None:
    thresholds = derive_e04_thresholds(_threshold_rows())
    rows = []
    for variant in CAMERA_VARIANTS:
        for episode_id in (12, 34):
            rows.append(
                _pair(
                    episode_id,
                    0,
                    variant,
                    effect=0.04,
                    coverage=0.98,
                )
            )
    source_rows = aggregate_e04_by_source(rows, thresholds=thresholds)
    gate = evaluate_e04_gate(
        source_rows,
        thresholds=thresholds,
        ready_upper=0.15,
    )
    assert gate["authorizes_e05"] is True

    failing_rows = [
        {
            **row,
            "semantic_margin_mean": (
                -0.01
                if row["camera_variant"] == "stick_only" and row["episode_id"] == 12
                else row["semantic_margin_mean"]
            ),
        }
        for row in source_rows
    ]
    gate = evaluate_e04_gate(
        failing_rows,
        thresholds=thresholds,
        ready_upper=0.15,
    )
    assert gate["authorizes_e05"] is False
    assert gate["criteria"]["stick_only"]["passed"] is False


def test_cross_process_noise_uses_immutable_trace_maximum(tmp_path: Path) -> None:
    roots = [tmp_path / f"repeat{index}" for index in range(3)]
    for index, root in enumerate(roots):
        trace = root / "base_traces" / "episode.npz"
        trace.parent.mkdir(parents=True)
        np.savez_compressed(
            trace,
            future_runtime_safe_action=np.asarray(
                [[0.0, float(index) * 0.01]],
                dtype=np.float32,
            ),
        )
    result = cross_process_replay_noise(roots)
    assert np.isclose(result["max_abs_delta"], 0.02)
    assert result["shared_trace_count"] == 1
    assert result["comparison_count"] == 2
