from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from testbed.tasks.act_cycle_planner import ScriptCyclePlanner
from testbed.tasks.home_side_contract import build_rule_ready_contract
from testbed.tasks.scripted_cycle_runtime import ReadySideWindow, ScriptedCycleRuntime


@dataclass
class _Clock:
    value: float = 0.0

    def __call__(self) -> float:
        return float(self.value)


class _PlannerPolicySource:
    def __init__(self, *, initial_side: str, targets: list[str]) -> None:
        self.cycle_planner = ScriptCyclePlanner(
            initial_side=initial_side,
            steps=[{"target_side": value} for value in targets],
            loop=False,
        )
        self.commit_count = 0
        self.ready_count = 0

    def commit_cycle_goal(self):
        self.commit_count += 1
        return self.cycle_planner.commit_goal()

    def mark_cycle_target_ready(self, realized_side: str):
        self.ready_count += 1
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


def _runtime(
    *, initial_side: str = "B", targets: list[str] | None = None
) -> tuple[ScriptedCycleRuntime, _PlannerPolicySource, _Clock]:
    source = _PlannerPolicySource(
        initial_side=initial_side,
        targets=list(targets or ["B", "A"]),
    )
    clock = _Clock()
    runtime = ScriptedCycleRuntime(
        policy_source=source,
        ready_contract=build_rule_ready_contract(),
        cycle_review_s=45.0,
        cycle_stop_s=60.0,
        run_stop_s=240.0,
        clock=clock,
    )
    return runtime, source, clock


def _observe(
    runtime: ScriptedCycleRuntime,
    *,
    timestamp_ns: int,
    swing_qpos: float,
    swing_qvel: float = 0.0,
) -> dict:
    return runtime.observe(
        {
            "timestamp_ns": int(timestamp_ns),
            "qpos": np.asarray([swing_qpos, 0.0, 0.0, 0.0], dtype=np.float32),
            "qvel": np.asarray([swing_qvel, 0.0, 0.0, 0.0], dtype=np.float32),
        }
    )


def _stable_side(
    runtime: ScriptedCycleRuntime, *, start_ns: int, swing_qpos: float
) -> int:
    timestamp_ns = int(start_ns)
    for _ in range(12):
        _observe(
            runtime,
            timestamp_ns=timestamp_ns,
            swing_qpos=swing_qpos,
            swing_qvel=0.0,
        )
        timestamp_ns += 50_000_000
    return timestamp_ns


def _confirm_excursion(
    runtime: ScriptedCycleRuntime, *, start_ns: int, swing_qpos: float = 1.2
) -> int:
    timestamp_ns = int(start_ns)
    for _ in range(3):
        _observe(
            runtime,
            timestamp_ns=timestamp_ns,
            swing_qpos=swing_qpos,
            swing_qvel=0.2,
        )
        runtime.evaluate()
        timestamp_ns += 50_000_000
    assert runtime.status()["excursion_observed"] is True
    return timestamp_ns


def test_same_side_goal_requires_excursion_before_advancing() -> None:
    runtime, source, _clock = _runtime(initial_side="B", targets=["B", "A"])
    timestamp_ns = _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=0.2)

    started = runtime.activate()

    assert started["planner"]["target_side"] == "B"
    assert runtime.evaluate()["goal_changed"] is False
    assert source.ready_count == 0

    timestamp_ns = _confirm_excursion(runtime, start_ns=timestamp_ns)
    advanced = None
    for _ in range(12):
        _observe(
            runtime,
            timestamp_ns=timestamp_ns,
            swing_qpos=0.2,
            swing_qvel=0.0,
        )
        advanced = runtime.evaluate()
        timestamp_ns += 50_000_000
        if advanced["goal_changed"]:
            break

    assert advanced is not None
    assert advanced["goal_changed"] is True
    assert advanced["planner"]["target_side"] == "A"
    assert source.ready_count == 1
    assert source.commit_count == 2


def test_script_completes_only_after_each_target_ready() -> None:
    runtime, source, _clock = _runtime(initial_side="B", targets=["B", "A"])
    timestamp_ns = _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=0.2)
    runtime.activate()
    timestamp_ns = _confirm_excursion(runtime, start_ns=timestamp_ns)
    for _ in range(12):
        _observe(
            runtime,
            timestamp_ns=timestamp_ns,
            swing_qpos=0.2,
            swing_qvel=0.0,
        )
        runtime.evaluate()
        timestamp_ns += 50_000_000
        if runtime.status()["planner"]["target_side"] == "A":
            break

    timestamp_ns = _confirm_excursion(runtime, start_ns=timestamp_ns)
    completed = None
    for _ in range(12):
        _observe(
            runtime,
            timestamp_ns=timestamp_ns,
            swing_qpos=-0.2,
            swing_qvel=0.0,
        )
        completed = runtime.evaluate()
        timestamp_ns += 50_000_000
        if completed["completed_now"]:
            break

    assert completed is not None
    assert completed["completed"] is True
    assert completed["completed_now"] is True
    assert completed["stop_policy"] is True
    assert source.ready_count == 2
    assert source.cycle_planner.done is True


def test_stable_wrong_side_fails_closed_after_excursion() -> None:
    runtime, _source, _clock = _runtime(initial_side="B", targets=["A"])
    timestamp_ns = _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=0.2)
    runtime.activate()
    timestamp_ns = _confirm_excursion(runtime, start_ns=timestamp_ns)
    result = None
    for _ in range(12):
        _observe(
            runtime,
            timestamp_ns=timestamp_ns,
            swing_qpos=0.2,
            swing_qvel=0.0,
        )
        result = runtime.evaluate()
        timestamp_ns += 50_000_000
        if result["stop_policy"]:
            break

    assert result is not None
    assert result["stop_policy"] is True
    assert result["fault"] == "stable_wrong_side:expected_A:actual_B"


def test_activation_requires_stable_expected_initial_side() -> None:
    runtime, _source, _clock = _runtime(initial_side="B", targets=["A"])
    _observe(
        runtime,
        timestamp_ns=1_000_000_000,
        swing_qpos=0.2,
        swing_qvel=0.0,
    )
    assert runtime.activation_blocker().startswith("initial_ready:")

    _stable_side(runtime, start_ns=2_000_000_000, swing_qpos=-0.2)
    assert runtime.activation_blocker() == ("initial_side_mismatch:expected_B:actual_A")


def test_cycle_timeout_stops_policy() -> None:
    runtime, _source, clock = _runtime(initial_side="B", targets=["A"])
    _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=0.2)
    runtime.activate()
    clock.value = 60.0

    result = runtime.evaluate()

    assert result["stop_policy"] is True
    assert result["fault"] == "cycle_timeout"


def test_ready_window_rejects_side_position_outside_training_support() -> None:
    ready = ReadySideWindow(
        build_rule_ready_contract(),
        target_ranges={"A": (-0.38, -0.09), "B": (0.11, 0.39)},
    )
    timestamp_ns = 1_000_000_000
    result = None
    for _ in range(12):
        result = ready.update(
            timestamp_ns=timestamp_ns,
            qpos=np.asarray([0.09, 0.0, 0.0, 0.0]),
            qvel=np.zeros(4),
        )
        timestamp_ns += 50_000_000

    assert result is not None
    assert result["actual_side"] == "B"
    assert "swing_outside_B_training_support" in result["blockers"]
