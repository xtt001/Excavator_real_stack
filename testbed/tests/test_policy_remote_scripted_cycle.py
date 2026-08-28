from __future__ import annotations

import numpy as np

from testbed.actions.base import ActionInfo, ActionSource
from testbed.actions.policy_remote import RemoteArmedPolicyActionSource
from testbed.tasks.act_cycle_planner import ScriptCyclePlanner
from testbed.tasks.home_side_contract import build_rule_ready_contract
from testbed.tasks.scripted_cycle_runtime import ScriptedCycleRuntime


class _Remote(ActionSource):
    def __init__(self) -> None:
        self.start_requested = False

    def reset(self) -> None:
        self.start_requested = False

    def next_action(self, obs):
        requested = bool(self.start_requested)
        self.start_requested = False
        return np.zeros(4, dtype=np.float32), ActionInfo(
            source_type="remote",
            source_id="remote",
            latency_ms=0.0,
            extras={"policy_start_requested": requested},
        )

    def close(self) -> None:
        return None


class _Policy(ActionSource):
    def __init__(self) -> None:
        self.cycle_planner = ScriptCyclePlanner(
            initial_side="B",
            steps=[{"target_side": "B"}, {"target_side": "A"}],
            loop=False,
        )
        self.reset_count = 0
        self.seen_targets: list[str] = []

    def reset(self) -> None:
        self.reset_count += 1
        self.cycle_planner.reset()

    def next_action(self, obs):
        goal = self.cycle_planner.committed_goal
        assert goal is not None
        self.seen_targets.append(str(goal.target_side))
        action = np.asarray(
            [-0.8 if goal.target_side == "A" else 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        return action, ActionInfo(
            source_type="policy",
            source_id="policy",
            latency_ms=1.0,
            extras={
                "policy_action": action.copy(),
                "policy_scaled_action": action.copy(),
                "policy_assisted_action": action.copy(),
                "policy_returned_action": action.copy(),
                "policy_output_mode": "control",
                "policy_error": "",
            },
        )

    def close(self) -> None:
        return None

    def commit_cycle_goal(self):
        return self.cycle_planner.commit_goal()

    def mark_cycle_target_ready(self, realized_side: str):
        return self.cycle_planner.mark_target_ready(realized_side)

    def cycle_planner_status(self) -> dict:
        goal = self.cycle_planner.committed_goal
        return {
            "enabled": True,
            "cycle_index": self.cycle_planner.cycle_index,
            "goal_epoch": self.cycle_planner.goal_epoch,
            "done": self.cycle_planner.done,
            "committed": goal is not None,
            "target_side": None if goal is None else goal.target_side,
            "condition": None if goal is None else list(goal.condition),
        }


def _source() -> tuple[RemoteArmedPolicyActionSource, _Remote, _Policy]:
    remote = _Remote()
    policy = _Policy()
    runtime = ScriptedCycleRuntime(
        policy_source=policy,
        ready_contract=build_rule_ready_contract(),
    )
    source = RemoteArmedPolicyActionSource(
        remote=remote,
        policy=policy,
        infer_on_new_frame=False,
        scripted_cycle_runtime=runtime,
    )
    return source, remote, policy


def _obs(*, timestamp_ns: int, swing_qpos: float, swing_qvel: float = 0.0):
    return {
        "timestamp_ns": int(timestamp_ns),
        "qpos": np.asarray([swing_qpos, 0.0, 0.0, 0.0], dtype=np.float32),
        "qvel": np.asarray([swing_qvel, 0.0, 0.0, 0.0], dtype=np.float32),
        "images": {},
    }


def _step_window(
    source: RemoteArmedPolicyActionSource,
    *,
    start_ns: int,
    swing_qpos: float,
    swing_qvel: float = 0.0,
    count: int = 12,
):
    timestamp_ns = int(start_ns)
    last = None
    for _ in range(count):
        last = source.next_action(
            _obs(
                timestamp_ns=timestamp_ns,
                swing_qpos=swing_qpos,
                swing_qvel=swing_qvel,
            )
        )
        timestamp_ns += 50_000_000
    return timestamp_ns, last


def test_remote_activation_commits_first_script_goal() -> None:
    source, remote, policy = _source()
    timestamp_ns, _ = _step_window(source, start_ns=1_000_000_000, swing_qpos=0.2)
    remote.start_requested = True

    action, info = source.next_action(_obs(timestamp_ns=timestamp_ns, swing_qpos=0.2))

    np.testing.assert_allclose(action, np.zeros(4, dtype=np.float32))
    assert info.extras["policy_remote_mode"] == "policy"
    assert info.extras["planner_target_side"] == "B"
    assert info.extras["planner_goal_epoch"] == 1
    assert info.extras["scripted_cycle_active"] == 1
    assert policy.seen_targets[-1] == "B"


def test_remote_activation_is_rejected_until_initial_ready() -> None:
    source, remote, policy = _source()
    remote.start_requested = True

    action, info = source.next_action(_obs(timestamp_ns=1_000_000_000, swing_qpos=0.2))

    np.testing.assert_allclose(action, np.zeros(4, dtype=np.float32))
    assert info.extras["policy_remote_mode"] == "manual"
    assert info.extras["scripted_cycle_activation_rejected_reason"].startswith(
        "initial_ready:"
    )
    assert policy.seen_targets == []


def test_script_completion_latches_zero_until_operator_acknowledges() -> None:
    source, remote, _policy = _source()
    timestamp_ns, _ = _step_window(source, start_ns=1_000_000_000, swing_qpos=0.2)
    remote.start_requested = True
    source.next_action(_obs(timestamp_ns=timestamp_ns, swing_qpos=0.2))
    timestamp_ns += 50_000_000

    timestamp_ns, _ = _step_window(
        source,
        start_ns=timestamp_ns,
        swing_qpos=1.2,
        swing_qvel=0.2,
        count=3,
    )
    timestamp_ns, _ = _step_window(
        source, start_ns=timestamp_ns, swing_qpos=0.2, count=12
    )
    timestamp_ns, _ = _step_window(
        source,
        start_ns=timestamp_ns,
        swing_qpos=1.2,
        swing_qvel=0.2,
        count=3,
    )
    timestamp_ns, last = _step_window(
        source, start_ns=timestamp_ns, swing_qpos=-0.2, count=12
    )

    assert last is not None
    action, info = last
    np.testing.assert_allclose(action, np.zeros(4, dtype=np.float32))
    assert info.extras["policy_remote_mode"] == "script_stop"
    assert info.extras["scripted_cycle_completed"] == 1
    assert source.policy_status()["scripted_cycle_stop_latched"] == 1

    remote.start_requested = True
    action, info = source.next_action(_obs(timestamp_ns=timestamp_ns, swing_qpos=-0.2))
    np.testing.assert_allclose(action, np.zeros(4, dtype=np.float32))
    assert info.extras["policy_remote_mode"] == "manual"
    assert source.policy_status()["scripted_cycle_stop_latched"] == 0
