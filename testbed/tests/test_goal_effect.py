from __future__ import annotations

import numpy as np

from testbed.policies.act.goal_effect import (
    build_goal_effect_targets,
    future_delta_scale,
    resolve_goal_effect_config,
)


def test_goal_effect_uses_shortest_swing_angle_delta() -> None:
    config = resolve_goal_effect_config(
        {"enabled": True, "horizons": [1], "unsupported_axes": []}
    )
    qpos = np.zeros((2, 4), dtype=np.float32)
    qpos[0, 0] = 3.13
    qpos[1, 0] = -3.13
    qvel = np.zeros_like(qpos)
    action = np.zeros_like(qpos)

    targets = build_goal_effect_targets(
        qpos=qpos,
        qvel=qvel,
        action=action,
        timestep=0,
        config=config,
    )

    assert abs(float(targets["goal_future_delta"][0, 0])) < 0.05
    assert int(targets["goal_future_direction"][0, 0]) == 2
    scale = future_delta_scale([qpos], [1])
    assert float(scale[0]) < 0.05
