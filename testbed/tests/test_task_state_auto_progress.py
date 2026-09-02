from __future__ import annotations

import numpy as np

from testbed.tasks.task_state_auto_progress import TaskStateAutoProgress


def _contract() -> dict:
    return {
        "schema": "real_transition_task_state_v2_auto_progress_contract_v1",
        "status": "DATA_CONTRACT_PASS",
        "runtime_config": {
            "advance_source": "automatic_policy_state",
            "required_liveness_axes": ["boom", "bucket"],
            "min_liveness_qpos_delta_rad": 0.05,
            "require_positive_swing_excursion": True,
            "bucket_positive_action_threshold": 0.408,
            "min_bucket_effective_steps": 5,
            "bucket_release_steps": 2,
            "return_idle_steps": 2,
            "positive_action_thresholds": [0.661, 0.259, 0.5, 0.408],
            "negative_action_thresholds": [0.721, 0.357, 0.5, 0.508],
        },
    }


class _PolicySource:
    def __init__(self) -> None:
        self.work_complete = False
        self.return_commit = False

    def set_task_dig_complete(self, *, completed: bool) -> bool:
        changed = self.work_complete != bool(completed)
        self.work_complete = bool(completed)
        return changed

    def set_task_return_commit(self, *, committed: bool) -> bool:
        changed = self.return_commit != bool(committed)
        self.return_commit = bool(committed)
        return changed


def test_automatic_progress_requires_physical_work_and_ordered_action_evidence() -> (
    None
):
    progress = TaskStateAutoProgress(_contract())
    source = _PolicySource()
    progress.reset_goal(np.zeros(4, dtype=np.float32))

    for _ in range(8):
        progress.observe_policy_action(
            [0.0, 0.0, 0.0, 0.7],
            excursion_observed=True,
            task_dig_complete=False,
            task_return_commit=False,
        )
    assert progress.pending_event == ""

    progress.observe_qpos([0.1, 0.06, 0.0, 0.08])
    assert progress.work_liveness_observed is True
    for _ in range(5):
        progress.observe_policy_action(
            [0.0, 0.0, 0.0, 0.6],
            excursion_observed=True,
            task_dig_complete=False,
            task_return_commit=False,
        )
    for _ in range(2):
        progress.observe_policy_action(
            [0.0, 0.0, 0.0, 0.2],
            excursion_observed=True,
            task_dig_complete=False,
            task_return_commit=False,
        )
    assert progress.pending_event == "work_complete"

    event, changed = progress.apply_pending(source)
    assert (event, changed) == ("work_complete", True)
    assert source.work_complete is True
    progress.observe_policy_action(
        [0.0, 0.4, 0.0, 0.0],
        excursion_observed=True,
        task_dig_complete=True,
        task_return_commit=False,
    )
    for _ in range(2):
        progress.observe_policy_action(
            [0.0, 0.0, 0.0, 0.0],
            excursion_observed=True,
            task_dig_complete=True,
            task_return_commit=False,
        )
    assert progress.pending_event == "return_commit"

    event, changed = progress.apply_pending(source)
    assert (event, changed) == ("return_commit", True)
    assert source.return_commit is True


def test_automatic_progress_requires_positive_excursion_before_bucket_count() -> None:
    progress = TaskStateAutoProgress(_contract())
    progress.reset_goal(np.zeros(4, dtype=np.float32))
    progress.observe_qpos([0.0, 0.06, 0.0, 0.08])

    for _ in range(7):
        progress.observe_policy_action(
            [0.0, 0.0, 0.0, 0.6],
            excursion_observed=False,
            task_dig_complete=False,
            task_return_commit=False,
        )

    assert progress.status()["bucket_effective_count"] == 0
    assert progress.pending_event == ""
