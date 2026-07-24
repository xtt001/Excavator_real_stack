from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "collect_fixed_tip_fk_samples.py"
)
SPEC = importlib.util.spec_from_file_location("collect_fixed_tip_fk_samples", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_episode_ids_rejects_duplicates() -> None:
    assert MODULE.parse_episode_ids("1, 2,3") == (1, 2, 3)
    try:
        MODULE.parse_episode_ids("1,1")
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("duplicate episode ids must be rejected")


def test_target_tracking_action_respects_axis_directions() -> None:
    current = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    target = np.asarray([0.6, 0.6, 0.4, 0.6], dtype=np.float32)
    action = MODULE.target_tracking_action(
        current,
        target,
        gain=2.0,
        max_command=0.8,
        deadband=0.0,
    )
    np.testing.assert_allclose(action, [0.2, -0.2, -0.2, 0.2], atol=1.0e-6)


def test_target_tracking_action_applies_deadband_and_clipping() -> None:
    current = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    target = np.asarray([0.501, 1.0, 0.0, 0.499], dtype=np.float32)
    action = MODULE.target_tracking_action(
        current,
        target,
        gain=10.0,
        max_command=0.7,
        deadband=0.01,
    )
    np.testing.assert_allclose(action, [0.0, -0.7, -0.7, 0.0], atol=1.0e-6)


def test_diverse_target_selection_is_deterministic() -> None:
    candidates = [
        MODULE.TargetCandidate(
            episode_id=index % 2,
            step=index,
            qpos=np.asarray(
                [index / 20.0, (index % 3) / 2.0, (index % 5) / 4.0, 0.5],
                dtype=np.float32,
            ),
        )
        for index in range(20)
    ]
    first = MODULE.select_diverse_targets(candidates, 6, seed=9)
    second = MODULE.select_diverse_targets(candidates, 6, seed=9)
    assert [value.step for value in first] == [value.step for value in second]
    assert len({value.step for value in first}) == 6
