import numpy as np
import pytest

from scripts.e51_full_act_temporal_gate_smoke import (
    build_causal_temporal_step_features,
    resolve_temporal_feature_names,
)


def test_build_causal_temporal_step_features_pads_history_only() -> None:
    features = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=np.float32,
    )

    out, names = build_causal_temporal_step_features(
        features,
        ["a", "b"],
        step=1,
        offsets=[-2, -1, 0],
    )

    np.testing.assert_allclose(out, [[1.0, 10.0, 1.0, 10.0, 2.0, 20.0]])
    assert names == ["a_t-2", "b_t-2", "a_t-1", "b_t-1", "a_t0", "b_t0"]


def test_build_causal_temporal_step_features_rejects_future_offsets() -> None:
    with pytest.raises(ValueError, match="future offsets"):
        build_causal_temporal_step_features(
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            ["a", "b"],
            step=0,
            offsets=[0, 1],
        )


def test_resolve_temporal_feature_names_uses_metadata_when_payload_names_are_stale(tmp_path) -> None:
    model_path = tmp_path / "temporal_direction_gate_model.pt"
    model_path.write_bytes(b"unused")
    metadata_path = tmp_path / "temporal_direction_gate_model_metadata.json"
    metadata_path.write_text(
        '{"feature_names": ["a_t-1", "b_t-1", "a_t0", "b_t0"], "context_offsets": [-1, 0]}',
        encoding="utf-8",
    )

    names, offsets = resolve_temporal_feature_names(
        model_path,
        {"feature_names": ["a", "b"], "feature_mean": np.zeros(4, dtype=np.float32)},
    )

    assert names == ["a_t-1", "b_t-1", "a_t0", "b_t0"]
    assert offsets == [-1, 0]
