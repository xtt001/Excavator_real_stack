from __future__ import annotations

import numpy as np
import pytest

from testbed.actions.policy import _policy_obs_from_real_obs
from testbed.data.action_primitive_islands import ACTION_PRIMITIVE_KEY
from testbed.data.work_return_context import WORK_CONTEXT_KEY


def _observation(primitive: np.ndarray) -> dict:
    return {
        "qpos": np.zeros(4, dtype=np.float32),
        "qvel": np.zeros(4, dtype=np.float32),
        "images": {"fpv": np.zeros((8, 10, 3), dtype=np.uint8)},
        ACTION_PRIMITIVE_KEY: primitive,
    }


def test_policy_observation_preserves_oracle_action_primitive() -> None:
    converted = _policy_obs_from_real_obs(
        _observation(np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
        camera_name="fpv",
    )

    np.testing.assert_allclose(
        converted[ACTION_PRIMITIVE_KEY], [0.0, 1.0, 0.0, 0.0]
    )


def test_policy_observation_rejects_non_one_hot_primitive() -> None:
    with pytest.raises(ValueError, match="finite one-hot"):
        _policy_obs_from_real_obs(
            _observation(np.asarray([0.0, 0.5, 0.5, 0.0], dtype=np.float32)),
            camera_name="fpv",
        )


def test_policy_observation_preserves_work_return_context() -> None:
    observation = _observation(
        np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    )
    observation.pop(ACTION_PRIMITIVE_KEY)
    observation[WORK_CONTEXT_KEY] = np.asarray(
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32
    )

    converted = _policy_obs_from_real_obs(observation, camera_name="fpv")

    np.testing.assert_allclose(
        converted[WORK_CONTEXT_KEY], [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    )
