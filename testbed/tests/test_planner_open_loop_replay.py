from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from testbed.cli.planner_open_loop_replay import (
    _complete_run_groups,
    _effective_signs,
    _supported_target_release_probe,
    _target_geometry_sign,
)


def _calibration() -> SimpleNamespace:
    return SimpleNamespace(
        deadzone_positive=np.asarray([0.6, 0.2, 0.3, 0.4]),
        deadzone_negative=np.asarray([0.7, 0.3, 0.4, 0.5]),
        target_ranges={"A": (-0.4, -0.1), "B": (0.1, 0.4)},
    )


def test_replay_effective_signs_use_direct_policy_deadzone() -> None:
    signs = _effective_signs(
        np.asarray([[0.6, -0.3, 0.29, -0.5], [0.1, 0.2, 0.4, 0.0]]),
        _calibration(),
    )

    np.testing.assert_array_equal(signs, [[1, -1, 0, -1], [0, 1, 1, 0]])


def test_target_geometry_requires_return_towards_or_stop_inside_band() -> None:
    calibration = _calibration()

    assert _target_geometry_sign(0.8, "A", calibration) == -1
    assert _target_geometry_sign(-0.8, "B", calibration) == 1
    assert _target_geometry_sign(0.2, "B", calibration) == 0


def test_complete_run_groups_reject_gaps_and_keep_planner_order() -> None:
    rows = [
        {
            "split": "validation",
            "source_run_id": "run_ok",
            "cycle_index": 0,
            "current_ready_side": "A",
            "scripted_target_side": "B",
        },
        {
            "split": "validation",
            "source_run_id": "run_ok",
            "cycle_index": 1,
            "current_ready_side": "B",
            "scripted_target_side": "A",
        },
        {
            "split": "validation",
            "source_run_id": "run_gap",
            "cycle_index": 1,
            "current_ready_side": "B",
            "scripted_target_side": "A",
        },
    ]

    groups = _complete_run_groups(rows, "validation")

    assert len(groups) == 1
    assert [row["scripted_target_side"] for row in groups[0]] == ["B", "A"]


def test_supported_release_probe_requires_A_negative_and_B_idle() -> None:
    class _Policy:
        def reset(self) -> None:
            return None

        def predict(self, observation):
            side_code = float(observation["real_transition_condition_v1"][0])
            return np.asarray(
                [-0.71 if side_code < 0.0 else 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )

    images = {
        name: np.zeros((2, 2, 3), dtype=np.uint8)
        for name in ("video4", "video5", "video6", "video7")
    }
    steps = [
        {
            "observation": {
                "qpos": np.asarray([value, 0.0, 0.0, 0.0], dtype=np.float32),
                "qvel": np.zeros(4, dtype=np.float32),
                "images": images,
            }
        }
        for value in (1.0, 0.35, 0.2, 0.0)
    ]

    result = _supported_target_release_probe(
        policy=_Policy(),
        steps=steps,
        calibration=_calibration(),
        apex_index=0,
        decision_range=(0.1, 0.4),
        qvel_input=False,
    )

    assert result["sample_count"] == 2
    assert result["pair_hit_count"] == 2
    assert result["pair_hit_rate"] == 1.0
