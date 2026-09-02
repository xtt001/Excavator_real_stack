from __future__ import annotations

import numpy as np
import pytest

from testbed.actions.policy import (
    _act_policy_config_from_resolved,
    _policy_obs_from_real_obs,
)
from testbed.data.task_state_v2 import TASK_STATE_V2_KEY
from testbed.runtime._train import _resolve_low_dim_state_dim


def _observation(task_state: list[float]) -> dict:
    return {
        "qpos": np.zeros(4, dtype=np.float32),
        "qvel": np.zeros(4, dtype=np.float32),
        "images": {"fpv": np.zeros((8, 10, 3), dtype=np.uint8)},
        TASK_STATE_V2_KEY: np.asarray(task_state, dtype=np.float32),
    }


def test_task_state_v2_resolves_to_thirteen_low_dim_inputs() -> None:
    keys = ["qpos", "qvel", TASK_STATE_V2_KEY]
    assert _resolve_low_dim_state_dim(keys, "real_excavator") == 13
    config = _act_policy_config_from_resolved(
        {
            "task": {"camera_names": ["fpv"]},
            "policy": {"low_dim_keys": keys, "act_params": {"chunk_size": 20}},
            "train": {},
        }
    )
    assert config["state_dim"] == 13
    assert config["backbone_pretrained"] is False


def test_policy_observation_preserves_valid_task_state() -> None:
    converted = _policy_obs_from_real_obs(
        _observation([1.0, 1.0, 1.0, 1.0, -1.0]), camera_name="fpv"
    )

    np.testing.assert_allclose(converted[TASK_STATE_V2_KEY], [1.0, 1.0, 1.0, 1.0, -1.0])


@pytest.mark.parametrize(
    "value",
    (
        [1.0, -1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, -1.0],
        [1.0, 1.0, 0.0, 1.0, 0.0],
    ),
)
def test_policy_observation_rejects_invalid_task_state(value: list[float]) -> None:
    with pytest.raises(ValueError, match="next target gated"):
        _policy_obs_from_real_obs(_observation(value), camera_name="fpv")
