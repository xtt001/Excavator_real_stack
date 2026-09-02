from __future__ import annotations

import numpy as np

from testbed.data.work_return_context import (
    derive_work_return_context,
    work_context_vector,
)


def _actions() -> np.ndarray:
    action = np.zeros((100, 4), dtype=np.float32)
    action[5:30, 1] = -0.5
    action[35:55, 0] = 0.8
    action[60:80, 3] = 0.6
    action[85:100, 0] = -0.8
    return action


def test_derives_single_complete_work_boundary_and_full_chunks() -> None:
    result = derive_work_return_context(
        _actions(),
        positive_thresholds=[0.661, 0.259, 0.5, 0.408],
        negative_thresholds=[0.721, 0.357, 0.5, 0.508],
        action_window_steps=10,
    )

    assert result.outbound_segment == (35, 54)
    assert result.bucket_segment == (60, 79)
    assert result.boundary_row == 80
    assert result.return_segment == (85, 99)
    assert result.work_starts[0] == 0
    assert result.work_starts[-1] == 70
    assert result.return_starts[0] == 80
    assert result.return_starts[-1] == 90


def test_context_keeps_anchor_and_target_while_hard_route_changes() -> None:
    work = work_context_vector(
        current_anchor="B", dig_target="B", next_target="A", route="work"
    )
    returning = work_context_vector(
        current_anchor="B", dig_target="B", next_target="A", route="return"
    )

    np.testing.assert_allclose(work, [1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(returning, [1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
