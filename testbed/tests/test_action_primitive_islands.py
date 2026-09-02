from __future__ import annotations

import json

import numpy as np
import pytest

from testbed.data.action_primitive_islands import (
    ACTION_PRIMITIVE_KEY,
    PRIMITIVE_NAMES,
    derive_action_primitive_islands,
    primitive_one_hot,
    resolve_action_primitive_config,
)

POSITIVE = np.asarray([0.661, 0.259, 0.5, 0.408], dtype=np.float32)
NEGATIVE = np.asarray([0.721, 0.357, 0.5, 0.508], dtype=np.float32)


def _actions() -> np.ndarray:
    action = np.zeros((90, 4), dtype=np.float32)
    action[5:25, 1] = -0.5
    action[30:50, 0] = 0.8
    action[55:75, 3] = 0.6
    action[75:90, 0] = -0.8
    return action


def test_derives_non_exhaustive_factual_islands() -> None:
    result = derive_action_primitive_islands(
        _actions(),
        positive_thresholds=POSITIVE,
        negative_thresholds=NEGATIVE,
        action_window_steps=10,
    )

    assert result.evaluable
    assert result.reasons == ()
    assert result.segments == {
        "tool_pre": ((5, 24),),
        "swing_out": ((30, 49),),
        "bucket_out": ((55, 74),),
        "swing_return": ((75, 89),),
    }
    assert {
        key: values.tolist() for key, values in result.candidate_starts.items()
    } == {
        "tool_pre": list(range(5, 16)),
        "swing_out": list(range(30, 41)),
        "bucket_out": list(range(55, 66)),
        "swing_return": list(range(75, 81)),
    }


def test_intersects_candidates_with_valid_starts() -> None:
    action = _actions()
    valid = np.asarray([5, 10, 15, 30, 35, 40, 55, 60, 65, 75, 80], dtype=np.int64)
    result = derive_action_primitive_islands(
        action,
        positive_thresholds=POSITIVE,
        negative_thresholds=NEGATIVE,
        action_window_steps=10,
        valid_starts=valid,
    )

    assert result.evaluable
    assert result.candidate_starts["tool_pre"].tolist() == [5, 10, 15]
    assert result.candidate_starts["swing_out"].tolist() == [30, 35, 40]
    assert result.candidate_starts["bucket_out"].tolist() == [55, 60, 65]
    assert result.candidate_starts["swing_return"].tolist() == [75, 80]


def test_short_bucket_island_is_retained_but_not_evaluable() -> None:
    action = _actions()
    action[55:75, 3] = 0.0
    action[55:63, 3] = 0.6
    result = derive_action_primitive_islands(
        action,
        positive_thresholds=POSITIVE,
        negative_thresholds=NEGATIVE,
        action_window_steps=10,
    )

    assert result.segments["bucket_out"] == ((55, 62),)
    assert result.candidate_starts["bucket_out"].size == 0
    assert not result.evaluable
    assert result.reasons == ("missing_full_window_bucket_out",)


def test_primitive_encoding_is_fixed_one_hot() -> None:
    for index, name in enumerate(PRIMITIVE_NAMES):
        value = primitive_one_hot(name)
        assert value.shape == (4,)
        assert value[index] == 1.0
        assert float(value.sum()) == 1.0
    with pytest.raises(ValueError, match="unknown action primitive"):
        primitive_one_hot("hold")


def test_resolve_config_requires_frozen_contract_files(tmp_path) -> None:
    threshold = tmp_path / "deadzone.json"
    threshold.write_text(
        json.dumps(
            {
                "deadzone_action": {
                    "swing": {"pos": 0.661, "neg": 0.721},
                    "boom": {"pos": 0.259, "neg": 0.357},
                    "stick": {"pos": 0.5, "neg": 0.5},
                    "bucket": {"pos": 0.408, "neg": 0.508},
                }
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    cfg = resolve_action_primitive_config(
        {
            "enabled": True,
            "condition_key": ACTION_PRIMITIVE_KEY,
            "primitive_names": list(PRIMITIVE_NAMES),
            "threshold_json": str(threshold),
            "manifest_path": str(manifest),
            "action_window_steps": 20,
            "append_samples_per_episode": 3,
        }
    )

    assert cfg["enabled"]
    assert cfg["condition_dim"] == 4
    np.testing.assert_allclose(cfg["positive_thresholds"], POSITIVE)
    np.testing.assert_allclose(cfg["negative_thresholds"], NEGATIVE)
