from __future__ import annotations

import numpy as np

from testbed.cli.state_hold_transition_replay import _summaries, _trace_metrics


def _thresholds() -> dict[str, dict[str, float]]:
    return {
        "swing": {"pos": 0.6, "neg": 0.7},
        "boom": {"pos": 0.2, "neg": 0.3},
        "stick": {"pos": 0.4, "neg": 0.4},
        "bucket": {"pos": 0.5, "neg": 0.5},
    }


def test_trace_metrics_requires_same_axis_and_direction() -> None:
    trace = np.zeros((6, 4), dtype=np.float32)
    trace[1, 1] = -0.31
    trace[2, 0] = 0.61

    metrics = _trace_metrics(
        trace=trace,
        thresholds=_thresholds(),
        target_axis=0,
        target_direction=0,
        short_window_steps=5,
    )

    assert metrics["target_reproduced_within_short_window"] is True
    assert metrics["target_reproduction_delay_ticks"] == 2
    assert metrics["query0_opposite_effective"] is False
    assert metrics["query0_other_effective"] is False
    assert metrics["horizon_other_effective_ticks"] == 1


def test_trace_metrics_reports_query0_wrong_effective() -> None:
    trace = np.zeros((5, 4), dtype=np.float32)
    trace[0, 0] = -0.71

    metrics = _trace_metrics(
        trace=trace,
        thresholds=_thresholds(),
        target_axis=0,
        target_direction=0,
        short_window_steps=5,
    )

    assert metrics["target_reproduced_within_horizon"] is False
    assert metrics["query0_opposite_effective"] is True
    assert metrics["query0_other_effective"] is True


def test_summary_keeps_validation_and_locked_separate() -> None:
    rows = [
        {
            "split": "validation",
            "anchor_group": "startup",
            "episode_id": 1,
            "target_reproduced_within_short_window": True,
            "target_reproduced_within_horizon": True,
            "query0_opposite_effective": False,
            "query0_other_effective": False,
        },
        {
            "split": "locked_test",
            "anchor_group": "startup",
            "episode_id": 2,
            "target_reproduced_within_short_window": False,
            "target_reproduced_within_horizon": True,
            "query0_opposite_effective": False,
            "query0_other_effective": False,
        },
    ]

    summaries = _summaries(rows)
    validation = next(
        row
        for row in summaries
        if row["split"] == "validation" and row["group"] == "overall"
    )
    locked = next(
        row
        for row in summaries
        if row["split"] == "locked_test" and row["group"] == "overall"
    )

    assert validation["same_direction_within_5_rate"] == 1.0
    assert locked["same_direction_within_5_rate"] == 0.0
    assert locked["same_direction_within_20_rate"] == 1.0
