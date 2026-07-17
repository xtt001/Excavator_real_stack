from __future__ import annotations

import numpy as np

from testbed.policies.execution_monitor_eval import (
    aggregate_monitor_summaries,
    evaluate_response_sidecar,
)

POSITIVE = [0.661, 0.259, 0.5, 0.408]
NEGATIVE = [0.721, 0.357, 0.5, 0.508]
QVEL_NOISE = [0.1, 0.1, 0.1, 0.1]


def _write_sidecar(path, *, qvel: np.ndarray, label: int) -> None:
    steps = qvel.shape[0]
    response_mask = np.full((1, steps, 4), -1, dtype=np.int8)
    response_mask[0, 0, 1] = label
    event_mask = np.zeros((steps, 4), dtype=bool)
    event_mask[0, 1] = True
    np.savez_compressed(
        path,
        qpos=np.zeros((steps, 4), dtype=np.float32),
        qvel=qvel.astype(np.float32),
        previous_final_command=np.tile(
            np.asarray([0.0, 0.3, 0.0, 0.0], dtype=np.float32),
            (steps, 1),
        ),
        command_send_timestamp_ns=np.arange(steps, dtype=np.int64) * 10 + 1,
        observation_timestamp_ns=np.arange(steps, dtype=np.int64) * 10 + 5,
        event_mask=event_mask,
        valid_mask=np.ones(steps, dtype=bool),
        reset_mask=np.zeros(steps, dtype=bool),
        response_mask=response_mask,
    )


def test_evaluate_response_sidecar_replays_response_and_matches_label(tmp_path) -> None:
    qvel = np.zeros((4, 4), dtype=np.float32)
    qvel[1, 1] = 0.2
    path = tmp_path / "episode_73.execution_response.npz"
    _write_sidecar(path, qvel=qvel, label=1)

    summary = evaluate_response_sidecar(
        sidecar_path=path,
        episode_id=73,
        split="train",
        positive_threshold=POSITIVE,
        negative_threshold=NEGATIVE,
        qvel_response_threshold=QVEL_NOISE,
        response_window_ticks=2,
        response_horizon_index=0,
    )

    assert summary.event_count == 1
    assert summary.responded_count == 1
    assert summary.stalled_count == 0
    assert summary.unknown_count == 0
    assert summary.sidecar_response_count == 1
    assert summary.response_mismatch_count == 0


def test_evaluate_response_sidecar_keeps_no_response_as_stalled_candidate(tmp_path) -> None:
    path = tmp_path / "episode_82.execution_response.npz"
    _write_sidecar(path, qvel=np.zeros((4, 4), dtype=np.float32), label=0)

    summary = evaluate_response_sidecar(
        sidecar_path=path,
        episode_id=82,
        split="train",
        positive_threshold=POSITIVE,
        negative_threshold=NEGATIVE,
        qvel_response_threshold=QVEL_NOISE,
        response_window_ticks=2,
        response_horizon_index=0,
    )

    assert summary.stalled_count == 1
    assert summary.sidecar_stalled_candidate_count == 1
    assert summary.response_mismatch_count == 0


def test_aggregate_marks_retry_precision_not_estimable() -> None:
    summary = aggregate_monitor_summaries([])

    assert summary["episode_count"] == 0
    assert summary["retry_precision_estimable"] is False
    assert "operator correction" in summary["retry_precision_reason"]
