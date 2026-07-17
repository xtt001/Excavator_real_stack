import numpy as np

from testbed.data.execution_response import (
    classify_effective_commands,
    response_label,
    summarize_response_latency,
)


def test_classify_effective_commands_keeps_task_zero_stick_unsupported() -> None:
    commands = np.asarray(
        [[0.7, -0.4, 0.6, 0.41], [0.1, 0.1, 0.0, 0.1]],
        dtype=np.float32,
    )
    effective, direction = classify_effective_commands(
        commands,
        positive_threshold=[0.661, 0.259, 0.5, 0.408],
        negative_threshold=[0.721, 0.357, 0.5, 0.508],
        supported_axes=("swing", "boom", "bucket"),
    )
    np.testing.assert_array_equal(effective[0], [True, True, False, True])
    np.testing.assert_array_equal(direction[0], [1, -1, 0, 1])
    np.testing.assert_array_equal(effective[1], [False, False, False, False])


def test_response_label_is_conservative_for_truncated_windows() -> None:
    label, peak = response_label(
        np.asarray([0.0, 0.2], dtype=np.float32),
        direction=1,
        qvel_noise=0.1,
        complete=True,
    )
    assert label == 1
    np.testing.assert_allclose(peak, 0.2)
    label, peak = response_label(
        np.asarray([0.0], dtype=np.float32),
        direction=1,
        qvel_noise=0.1,
        complete=False,
    )
    assert label == -1
    assert np.isnan(peak)


def test_summarize_response_latency_counts_first_response_only() -> None:
    summary = summarize_response_latency(
        [
            {"axis": "boom", "direction": "pos", "response_1t": 0, "response_4t": 1},
            {"axis": "boom", "direction": "pos", "response_1t": 1, "response_4t": 1},
            {"axis": "boom", "direction": "pos", "response_1t": 0, "response_4t": 0},
        ],
        response_horizons=(1, 4),
    )
    assert summary["same_direction_response_rows"] == 2
    assert summary["no_same_direction_response_rows"] == 1
    assert summary["groups"]["boom:pos"]["first_response_tick_counts"] == {
        "1": 1,
        "4": 1,
    }
