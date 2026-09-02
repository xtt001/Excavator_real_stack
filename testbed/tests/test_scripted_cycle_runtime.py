from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from testbed.tasks.act_cycle_planner import (
    ScriptCyclePlanner,
    SideMatchedScriptCyclePlanner,
)
from testbed.tasks.home_side_contract import build_rule_ready_contract
from testbed.tasks.scripted_cycle_runtime import ReadySideWindow, ScriptedCycleRuntime
from testbed.tasks.task_state_auto_progress import TaskStateAutoProgress


@dataclass
class _Clock:
    value: float = 0.0

    def __call__(self) -> float:
        return float(self.value)


class _PlannerPolicySource:
    def __init__(
        self, *, initial_side: str, targets: list[str], task_state_v2: bool = False
    ) -> None:
        self.cycle_planner = ScriptCyclePlanner(
            initial_side=initial_side,
            steps=[{"target_side": value} for value in targets],
            loop=False,
        )
        self.commit_count = 0
        self.ready_count = 0
        self.excursion_set_count = 0
        self.task_state_v2_enabled = bool(task_state_v2)
        self.task_dig_complete = False
        self.task_return_commit = False
        self.task_state_epoch = 0

    def commit_cycle_goal(self):
        self.commit_count += 1
        goal = self.cycle_planner.commit_goal()
        self.task_dig_complete = False
        self.task_return_commit = False
        self.task_state_epoch = int(goal.goal_epoch) * 3
        return goal

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
            "task_state_v2_enabled": self.task_state_v2_enabled,
            "task_dig_complete": self.task_dig_complete,
            "task_return_commit": self.task_return_commit,
            "task_state_epoch": self.task_state_epoch,
        }

    def set_cycle_excursion_observed(self, *, observed: bool) -> bool:
        assert observed is True
        self.excursion_set_count += 1
        return True

    def set_task_dig_complete(self, *, completed: bool) -> bool:
        changed = self.task_dig_complete != bool(completed)
        self.task_dig_complete = bool(completed)
        self.task_state_epoch += int(changed)
        return changed

    def set_task_return_commit(self, *, committed: bool) -> bool:
        changed = self.task_return_commit != bool(committed)
        self.task_return_commit = bool(committed)
        self.task_state_epoch += int(changed)
        return changed


def _runtime(
    *,
    initial_side: str = "B",
    targets: list[str] | None = None,
    target_ranges: dict[str, tuple[float, float]] | None = None,
    swing_landing: dict | None = None,
    task_state_v2: bool = False,
) -> tuple[ScriptedCycleRuntime, _PlannerPolicySource, _Clock]:
    source = _PlannerPolicySource(
        initial_side=initial_side,
        targets=list(targets or ["B", "A"]),
        task_state_v2=task_state_v2,
    )
    clock = _Clock()
    runtime = ScriptedCycleRuntime(
        policy_source=source,
        ready_contract=build_rule_ready_contract(),
        target_ranges=target_ranges,
        swing_landing=swing_landing,
        cycle_review_s=45.0,
        cycle_stop_s=60.0,
        run_stop_s=240.0,
        task_state_v2=(
            {
                "enabled": True,
                "advance_source": "operator_mark",
                "require_excursion_before_work_complete": True,
            }
            if task_state_v2
            else None
        ),
        clock=clock,
    )
    return runtime, source, clock


def _landing_cfg() -> dict:
    return {
        "enabled": True,
        "coast_stop_time_s": 0.50,
        "edge_margin_rad": 0.03,
        "p_gain": 0.60,
        "d_gain": 0.12,
        "return_confirm_drop_rad": 0.05,
        "return_min_qvel_rad_s": 0.05,
        "pd_blend_width_rad": 0.03,
        "pd_blend_time_s": 0.25,
        "policy_gain_time_s": 0.25,
        "min_action_positive": 0.661,
        "min_action_negative": 0.721,
        "max_action_positive": 0.72,
        "max_action_negative": 0.78,
        "qvel_stable_rad_s": 0.015,
    }


def _auto_progress_contract() -> dict:
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
    assert runtime.status()["event"] == "cycle_excursion_confirmed"
    return timestamp_ns


def test_same_side_goal_requires_excursion_before_advancing() -> None:
    runtime, source, _clock = _runtime(initial_side="B", targets=["B", "A"])
    timestamp_ns = _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=0.2)

    started = runtime.activate()

    assert started["planner"]["target_side"] == "B"
    assert runtime.evaluate()["goal_changed"] is False
    assert source.ready_count == 0

    timestamp_ns = _confirm_excursion(runtime, start_ns=timestamp_ns)
    assert source.excursion_set_count == 1
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


def test_direct_motion_towards_a_does_not_confirm_excursion() -> None:
    runtime, source, _clock = _runtime(initial_side="B", targets=["A"])
    timestamp_ns = _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=0.3)
    runtime.activate()

    for swing_qpos in (0.2, 0.1, 0.0, -0.1):
        _observe(
            runtime,
            timestamp_ns=timestamp_ns,
            swing_qpos=swing_qpos,
            swing_qvel=-0.2,
        )
        runtime.evaluate()
        timestamp_ns += 50_000_000

    assert runtime.status()["excursion_observed"] is False
    assert source.ready_count == 0


def test_task_state_v2_requires_excursion_then_advances_in_two_marks() -> None:
    runtime, source, _clock = _runtime(
        initial_side="B", targets=["A"], task_state_v2=True
    )
    timestamp_ns = _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=0.2)
    started = runtime.activate()
    assert started["task_state_stage"] == "work"

    with pytest.raises(
        RuntimeError, match="work_complete_requires_confirmed_positive_excursion"
    ):
        runtime.advance_task_state()

    _confirm_excursion(runtime, start_ns=timestamp_ns)
    completed = runtime.advance_task_state()
    assert completed["task_state_changed"] is True
    assert completed["task_state_stage"] == "work_complete"
    assert source.task_dig_complete is True
    assert source.task_return_commit is False

    committed = runtime.advance_task_state()
    assert committed["task_state_changed"] is True
    assert committed["task_state_stage"] == "return_committed"
    assert source.task_return_commit is True

    ignored = runtime.advance_task_state()
    assert ignored["task_state_changed"] is False
    assert ignored["task_state_advance_ignored"] is True


def test_task_state_v2_never_accepts_target_ready_before_return_commit() -> None:
    runtime, source, _clock = _runtime(
        initial_side="B", targets=["A"], task_state_v2=True
    )
    timestamp_ns = _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=0.2)
    runtime.activate()
    timestamp_ns = _confirm_excursion(runtime, start_ns=timestamp_ns)

    for _ in range(12):
        _observe(
            runtime,
            timestamp_ns=timestamp_ns,
            swing_qpos=-0.2,
            swing_qvel=0.0,
        )
        runtime.evaluate()
        timestamp_ns += 50_000_000
    assert source.ready_count == 0
    assert runtime.status()["completed"] is False

    runtime.advance_task_state()
    runtime.advance_task_state()
    completed = runtime.evaluate()
    assert completed["completed"] is True
    assert source.ready_count == 1


def test_task_state_v2_can_advance_automatically_without_operator_marks() -> None:
    source = _PlannerPolicySource(initial_side="B", targets=["A"], task_state_v2=True)
    runtime = ScriptedCycleRuntime(
        policy_source=source,
        ready_contract=build_rule_ready_contract(),
        task_state_v2={
            "enabled": True,
            "advance_source": "automatic_policy_state",
        },
        task_state_auto_progress=TaskStateAutoProgress(_auto_progress_contract()),
    )
    timestamp_ns = _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=0.2)
    runtime.activate()
    runtime.observe(
        {
            "timestamp_ns": timestamp_ns,
            "qpos": np.asarray([0.2, 0.06, 0.0, 0.08], dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
    )
    timestamp_ns = _confirm_excursion(runtime, start_ns=timestamp_ns + 50_000_000)

    for _ in range(5):
        runtime.observe_policy_action([0.0, 0.0, 0.0, 0.6])
    for _ in range(2):
        runtime.observe_policy_action([0.0, 0.0, 0.0, 0.2])
    assert runtime.status()["task_state_auto_progress"]["pending_event"] == (
        "work_complete"
    )
    completed = runtime.evaluate()
    assert completed["task_state_changed"] is True
    assert source.task_dig_complete is True

    runtime.observe_policy_action([0.0, 0.0, 0.0, 0.0])
    runtime.observe_policy_action([0.0, 0.0, 0.0, 0.0])
    committed = runtime.evaluate()
    assert committed["task_state_changed"] is True
    assert committed["task_state_stage"] == "return_committed"
    assert source.task_return_commit is True


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


def test_target_b_outbound_crossing_cannot_complete_cycle_before_left_return() -> None:
    runtime, source, _clock = _runtime(
        initial_side="A",
        targets=["B"],
        target_ranges={"A": (-0.3788, -0.0931), "B": (0.1112, 0.3928)},
        swing_landing=_landing_cfg(),
    )
    timestamp_ns = _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=-0.2)
    runtime.activate()
    timestamp_ns = _confirm_excursion(
        runtime,
        start_ns=timestamp_ns,
        swing_qpos=0.2,
    )

    for _ in range(12):
        _observe(
            runtime,
            timestamp_ns=timestamp_ns,
            swing_qpos=0.2,
            swing_qvel=0.0,
        )
        result = runtime.evaluate()
        timestamp_ns += 50_000_000

    assert result["return_phase_latched"] is False
    assert result["completed"] is False
    assert source.ready_count == 0


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


def test_activation_selects_script_matching_stable_ready_side() -> None:
    source = _PlannerPolicySource(initial_side="B", targets=["A"])
    source.cycle_planner = SideMatchedScriptCyclePlanner(
        {
            "A": ScriptCyclePlanner(initial_side="A", steps=["B", "A"]),
            "B": ScriptCyclePlanner(initial_side="B", steps=["A", "B"]),
        }
    )
    runtime = ScriptedCycleRuntime(
        policy_source=source,
        ready_contract=build_rule_ready_contract(),
    )
    _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=-0.2)

    started = runtime.activate()

    assert source.cycle_planner.selected_initial_side == "A"
    assert started["planner"]["target_side"] == "B"


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


def _latch_left_return_phase(runtime: ScriptedCycleRuntime, *, start_ns: int) -> int:
    timestamp_ns = _confirm_excursion(
        runtime,
        start_ns=start_ns,
        swing_qpos=1.2,
    )
    _shaped, diagnostics = runtime.shape_policy_action(
        np.asarray([-0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        {
            "timestamp_ns": timestamp_ns,
            "qpos": np.asarray([1.1, 0.0, 0.0, 0.0]),
            "qvel": np.asarray([-0.2, 0.0, 0.0, 0.0]),
        },
    )
    assert diagnostics["swing_landing_return_phase"] == 1
    return timestamp_ns + 50_000_000


def test_swing_landing_does_not_intervene_during_rightward_excavation_motion() -> None:
    runtime, _source, _clock = _runtime(
        initial_side="A",
        targets=["B"],
        target_ranges={"A": (-0.3788, -0.0931), "B": (0.1112, 0.3928)},
        swing_landing=_landing_cfg(),
    )
    timestamp_ns = _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=-0.2)
    runtime.activate()
    action = np.asarray([0.81, 0.2, -0.3, 0.4], dtype=np.float32)

    shaped, diagnostics = runtime.shape_policy_action(
        action,
        {
            "timestamp_ns": timestamp_ns,
            "qpos": np.asarray([0.20, 0.0, 0.0, 0.0]),
            "qvel": np.asarray([0.30, 0.0, 0.0, 0.0]),
        },
    )

    np.testing.assert_allclose(shaped, action)
    assert diagnostics["swing_landing_mode"] == "waiting_return_phase"
    assert diagnostics["swing_landing_return_phase"] == 0


def test_swing_landing_releases_after_measured_entry_toward_a_without_changing_other_axes() -> (
    None
):
    runtime, _source, _clock = _runtime(
        initial_side="B",
        targets=["A"],
        target_ranges={"A": (-0.3788, -0.0931), "B": (0.1112, 0.3928)},
        swing_landing=_landing_cfg(),
    )
    _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=0.2)
    runtime.activate()
    timestamp_ns = _latch_left_return_phase(runtime, start_ns=2_000_000_000)
    action = np.asarray([-0.85, 0.2, -0.3, 0.4], dtype=np.float32)

    shaped, diagnostics = runtime.shape_policy_action(
        action,
        {
            "timestamp_ns": timestamp_ns,
            "qpos": np.asarray([-0.20, 0.0, 0.0, 0.0]),
            "qvel": np.asarray([-0.56, 0.0, 0.0, 0.0]),
        },
    )

    assert action[0] < shaped[0] < 0.0
    np.testing.assert_allclose(shaped[1:], action[1:])
    assert diagnostics["swing_landing_mode"] == "policy_gain"
    assert 0.0 < diagnostics["swing_landing_policy_gain"] < 1.0
    assert diagnostics["swing_landing_projected_qpos_rad"] < -0.37


def test_swing_landing_releases_after_measured_entry_toward_b() -> None:
    runtime, _source, _clock = _runtime(
        initial_side="A",
        targets=["B"],
        target_ranges={"A": (-0.3788, -0.0931), "B": (0.1112, 0.3928)},
        swing_landing=_landing_cfg(),
    )
    _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=-0.2)
    runtime.activate()
    timestamp_ns = _latch_left_return_phase(runtime, start_ns=2_000_000_000)

    shaped, diagnostics = runtime.shape_policy_action(
        np.asarray([-0.85, 0.2, -0.3, 0.4], dtype=np.float32),
        {
            "timestamp_ns": timestamp_ns,
            "qpos": np.asarray([0.30, 0.0, 0.0, 0.0]),
            "qvel": np.asarray([-0.56, 0.0, 0.0, 0.0]),
        },
    )

    assert -0.85 < shaped[0] < 0.0
    assert diagnostics["swing_landing_mode"] == "policy_gain"
    assert 0.0 < diagnostics["swing_landing_policy_gain"] < 1.0
    assert diagnostics["swing_landing_projected_qpos_rad"] < 0.13


def test_swing_landing_does_not_stall_outside_b_from_coast_projection() -> None:
    runtime, _source, _clock = _runtime(
        initial_side="A",
        targets=["B"],
        target_ranges={"A": (-0.3788, -0.0931), "B": (0.1112, 0.3928)},
        swing_landing=_landing_cfg(),
    )
    _stable_side(runtime, start_ns=1_000_000_000, swing_qpos=-0.2)
    runtime.activate()
    timestamp_ns = _latch_left_return_phase(runtime, start_ns=2_000_000_000)
    action = np.asarray([-0.69, 0.2, -0.3, 0.4], dtype=np.float32)

    shaped, diagnostics = runtime.shape_policy_action(
        action,
        {
            "timestamp_ns": timestamp_ns,
            "qpos": np.asarray([0.452, 0.0, 0.0, 0.0]),
            "qvel": np.asarray([-0.24, 0.0, 0.0, 0.0]),
        },
    )

    np.testing.assert_allclose(shaped, action)
    assert diagnostics["swing_landing_mode"] == "model"
    assert diagnostics["swing_landing_policy_gain"] == 1.0
    assert diagnostics["swing_landing_projected_qpos_rad"] < 0.3928


def test_swing_landing_blends_positive_pd_only_after_each_left_boundary() -> None:
    ranges = {"A": (-0.3788, -0.0931), "B": (0.1112, 0.3928)}
    runtime_a, _source, _clock = _runtime(
        initial_side="B",
        targets=["A"],
        target_ranges=ranges,
        swing_landing=_landing_cfg(),
    )
    _stable_side(runtime_a, start_ns=1_000_000_000, swing_qpos=0.2)
    runtime_a.activate()
    timestamp_a = _latch_left_return_phase(runtime_a, start_ns=2_000_000_000)
    inside_a, inside_diag_a = runtime_a.shape_policy_action(
        np.asarray([-0.6, 0.1, 0.2, 0.3], dtype=np.float32),
        {
            "timestamp_ns": timestamp_a,
            "qpos": np.asarray([-0.37, 0.0, 0.0, 0.0]),
            "qvel": np.asarray([-0.20, 0.0, 0.0, 0.0]),
        },
    )
    shaped_a, diagnostics_a = runtime_a.shape_policy_action(
        np.asarray([-0.6, 0.1, 0.2, 0.3], dtype=np.float32),
        {
            "timestamp_ns": timestamp_a + 50_000_000,
            "qpos": np.asarray([-0.3838, 0.0, 0.0, 0.0]),
            "qvel": np.asarray([-0.20, 0.0, 0.0, 0.0]),
        },
    )

    runtime_b, _source, _clock = _runtime(
        initial_side="A",
        targets=["B"],
        target_ranges=ranges,
        swing_landing=_landing_cfg(),
    )
    _stable_side(runtime_b, start_ns=2_000_000_000, swing_qpos=-0.2)
    runtime_b.activate()
    timestamp_b = _latch_left_return_phase(runtime_b, start_ns=3_000_000_000)
    shaped_b, diagnostics_b = runtime_b.shape_policy_action(
        np.asarray([-0.6, 0.1, 0.2, 0.3], dtype=np.float32),
        {
            "timestamp_ns": timestamp_b + 50_000_000,
            "qpos": np.asarray([0.1062, 0.0, 0.0, 0.0]),
            "qvel": np.asarray([-0.20, 0.0, 0.0, 0.0]),
        },
    )

    assert -0.6 < inside_a[0] < 0.0
    assert inside_diag_a["swing_landing_pd_blend"] == 0.0
    assert inside_a[0] < shaped_a[0] < 0.661
    assert -0.6 < shaped_b[0] < 0.661
    assert diagnostics_a["swing_landing_mode"] == "pd_blend"
    assert diagnostics_b["swing_landing_mode"] == "pd_blend"
    assert 0.0 < diagnostics_a["swing_landing_pd_blend"] < 1.0
    assert 0.0 < diagnostics_b["swing_landing_pd_blend"] < 1.0
    np.testing.assert_allclose(shaped_a[1:], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(shaped_b[1:], [0.1, 0.2, 0.3])
