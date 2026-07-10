import numpy as np

from scripts.e47_temporal_direction_gate_probe import build_temporal_context_features, build_temporal_gate_summary


def test_build_temporal_context_features_edge_pads_offsets() -> None:
    features = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=np.float32,
    )

    out, names = build_temporal_context_features(features, ["a", "b"], offsets=[-1, 0, 1])

    np.testing.assert_allclose(
        out,
        [
            [1.0, 10.0, 1.0, 10.0, 2.0, 20.0],
            [1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            [2.0, 20.0, 3.0, 30.0, 3.0, 30.0],
        ],
    )
    assert names == ["a_t-1", "b_t-1", "a_t0", "b_t0", "a_t+1", "b_t+1"]


def test_build_temporal_gate_summary_matches_e34_gate_contract() -> None:
    row = {
        "label": "tdir_t50_s75",
        "threshold": 0.5,
        "inactive_scale": 0.75,
        "scaled_frames": 12,
        "mae": 0.04,
        "rmse": 0.08,
        "replay_dir": "/tmp/replay",
    }

    summary = build_temporal_gate_summary(row)

    assert summary == {
        "selection_mode": "all_train_ready_oof_temporal_direction_gate",
        "gate_name": "tdir_t50_s75",
        "scan_row": row,
    }
