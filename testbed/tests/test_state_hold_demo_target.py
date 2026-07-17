from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from testbed.policies.state_hold_demo_target import (
    StepOutput,
    aggregate_state_hold_demo_target_rows,
    evaluate_state_hold_demo_target,
    extract_should_move_anchors,
    write_state_hold_demo_target_report,
)


def _thresholds(value: float = 0.5) -> dict[str, dict[str, float]]:
    return {
        axis: {"pos": value, "neg": value}
        for axis in ("swing", "boom", "stick", "bucket")
    }


def _observations(qpos_values: list[float]) -> list[dict[str, np.ndarray]]:
    return [
        {
            "qpos": np.array([value, 0.0, 0.0, 0.0], dtype=np.float32),
            "qvel": np.full(4, value + 0.25, dtype=np.float32),
            "image_fpv": np.full((2, 2, 3), int(value), dtype=np.uint8),
        }
        for value in qpos_values
    ]


class QposProgressSource:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.runs: list[list[tuple[float, np.ndarray]]] = []

    def reset(self) -> None:
        self.reset_calls += 1
        self.runs.append([])

    def step(self, observation: Mapping[str, Any]) -> StepOutput:
        qpos = float(np.asarray(observation["qpos"])[0])
        qvel = np.asarray(observation["qvel"], dtype=np.float32).copy()
        self.runs[-1].append((qpos, qvel))
        action = np.zeros(4, dtype=np.float32)
        if qpos >= 1.0:
            action[0] = 0.8
        return StepOutput(action=action, diagnostics={"raw_action": action.copy()})


class FrozenTickRecoverySource:
    def __init__(self, recovery_call: int, action: np.ndarray) -> None:
        self.recovery_call = recovery_call
        self.action = np.asarray(action, dtype=np.float32)
        self.call = 0

    def reset(self) -> None:
        self.call = 0

    def step(self, observation: Mapping[str, Any]) -> np.ndarray:
        del observation
        current = self.call
        self.call += 1
        if current >= self.recovery_call:
            return self.action.copy()
        return np.zeros(4, dtype=np.float32)


class PreviousCommandRetrySource:
    def __init__(self) -> None:
        self.inputs: list[list[np.ndarray]] = []

    def reset(self) -> None:
        self.inputs.append([])

    def step(self, observation: Mapping[str, Any]) -> np.ndarray:
        previous = np.asarray(
            observation["previous_final_command"], dtype=np.float32
        ).copy()
        self.inputs[-1].append(previous)
        action = np.zeros(4, dtype=np.float32)
        action[0] = previous[0] + 0.2
        return action


def test_teacher_forced_progress_can_hide_state_hold_deadlock() -> None:
    expert = np.zeros((4, 4), dtype=np.float32)
    expert[0:2, 0] = 0.8
    source = QposProgressSource()

    rows = evaluate_state_hold_demo_target(
        episode_id="episode_001",
        observations=_observations([0.0, 1.0, 2.0, 3.0]),
        expert_action=expert,
        thresholds=_thresholds(),
        step_source=source,
        hold_horizon_steps=3,
    )

    assert len(rows) == 1
    assert rows[0]["teacher_forced_status"] == "demo_target_reproduced"
    assert rows[0]["teacher_forced_demo_target_reproduction_delay_ticks"] == 1
    assert rows[0]["state_hold_status"] == "demo_target_not_reproduced"
    assert rows[0]["demo_target_reproduction_hidden_by_teacher_forcing"] is True
    assert "state_hold_deadlocked" not in rows[0]
    assert "hidden_by_teacher_forcing" not in rows[0]
    assert all(np.allclose(qvel, 0.0) for _, qvel in source.runs[1])


def test_state_hold_reports_delayed_same_direction_recovery() -> None:
    expert = np.zeros((5, 4), dtype=np.float32)
    expert[0:2, 1] = -0.8
    action = np.zeros(4, dtype=np.float32)
    action[1] = -0.7

    rows = evaluate_state_hold_demo_target(
        episode_id="episode_002",
        observations=_observations([0.0] * 5),
        expert_action=expert,
        thresholds=_thresholds(),
        step_source=FrozenTickRecoverySource(recovery_call=2, action=action),
        hold_horizon_steps=4,
    )

    assert rows[0]["state_hold_status"] == "demo_target_reproduced"
    assert rows[0]["state_hold_demo_target_reproduction_delay_ticks"] == 2
    aggregate = aggregate_state_hold_demo_target_rows(rows)[0]
    assert aggregate["state_hold_demo_target_reproduction_delay_ticks"] == [2]
    assert aggregate["state_hold_demo_target_not_reproduced_anchors"] == 0
    assert "state_hold_deadlocked_anchors" not in aggregate


def test_full_horizon_trace_keeps_later_ticks_after_first_recovery() -> None:
    expert = np.zeros((5, 4), dtype=np.float32)
    expert[0, 0] = 0.8
    action = np.array([0.8, 0.0, 0.0, 0.0], dtype=np.float32)

    rows = evaluate_state_hold_demo_target(
        episode_id="episode_full_horizon",
        observations=_observations([0.0] * 5),
        expert_action=expert,
        thresholds=_thresholds(),
        step_source=FrozenTickRecoverySource(recovery_call=0, action=action),
        hold_horizon_steps=4,
        trace_full_horizon_after_reproduction=True,
    )

    row = rows[0]
    assert row["state_hold_status"] == "demo_target_reproduced"
    assert row["state_hold_demo_target_reproduction_delay_ticks"] == 0
    assert row["state_hold_ticks_evaluated"] == 4
    assert len(row["state_hold_action_trace"]) == 4
    assert row["state_hold_full_horizon_complete"] is True
    assert row["state_hold_trace_termination"] == (
        "horizon_complete_after_demo_target_reproduction"
    )
    assert row["state_hold_anchor_extra_effective"] is False
    assert row["state_hold_max_effective_axes"] == 1


def test_wrong_axis_and_wrong_direction_do_not_count_as_recovery() -> None:
    expert = np.zeros((3, 4), dtype=np.float32)
    expert[0, 2] = 0.8
    wrong = np.array([0.9, 0.0, -0.9, 0.0], dtype=np.float32)

    rows = evaluate_state_hold_demo_target(
        episode_id="episode_003",
        observations=_observations([0.0] * 3),
        expert_action=expert,
        thresholds=_thresholds(),
        step_source=FrozenTickRecoverySource(recovery_call=0, action=wrong),
        hold_horizon_steps=3,
    )

    assert rows[0]["axis"] == "stick"
    assert rows[0]["direction"] == "pos"
    assert rows[0]["state_hold_demo_target_not_reproduced"] is True


def test_anchor_extraction_discovers_multiple_mid_cycle_transitions() -> None:
    expert = np.zeros((8, 4), dtype=np.float32)
    expert[0:2, 0] = 0.8
    expert[3, 0] = 0.8
    expert[5, 1] = -0.8
    expert[7, 3] = 0.9

    anchors = extract_should_move_anchors(expert, _thresholds())

    assert [(a.step, a.axis, a.direction, a.group) for a in anchors] == [
        (0, "swing", "pos", "startup"),
        (3, "swing", "pos", "mid_cycle"),
        (5, "boom", "neg", "mid_cycle"),
        (7, "bucket", "pos", "mid_cycle"),
    ]


def test_each_branch_resets_and_replays_identical_warmup_deterministically() -> None:
    expert = np.zeros((5, 4), dtype=np.float32)
    expert[2:4, 0] = 0.8
    source = QposProgressSource()

    evaluate_state_hold_demo_target(
        episode_id="episode_004",
        observations=_observations([0.0, 0.5, 0.5, 1.0, 1.5]),
        expert_action=expert,
        thresholds=_thresholds(),
        step_source=source,
        hold_horizon_steps=2,
    )

    assert source.reset_calls == 2
    assert [item[0] for item in source.runs[0][:2]] == [0.0, 0.5]
    assert [item[0] for item in source.runs[1][:2]] == [0.0, 0.5]
    np.testing.assert_allclose(source.runs[0][0][1], source.runs[1][0][1])
    np.testing.assert_allclose(source.runs[0][1][1], source.runs[1][1][1])


def test_state_hold_recursively_feeds_back_previous_final_command() -> None:
    expert = np.zeros((3, 4), dtype=np.float32)
    expert[0, 0] = 0.8
    observations = _observations([0.0] * 3)
    for observation in observations:
        observation["previous_final_command"] = np.array(
            [0.1, 0.0, 0.0, 0.0], dtype=np.float32
        )
    source = PreviousCommandRetrySource()

    rows = evaluate_state_hold_demo_target(
        episode_id="episode_feedback",
        observations=observations,
        expert_action=expert,
        thresholds=_thresholds(),
        step_source=source,
        hold_horizon_steps=3,
    )

    assert rows[0]["execution_feedback_recursive"] is True
    assert rows[0]["teacher_forced_status"] == "demo_target_not_reproduced"
    assert rows[0]["state_hold_status"] == "demo_target_reproduced"
    assert rows[0]["state_hold_demo_target_reproduction_delay_ticks"] == 1
    np.testing.assert_allclose(source.inputs[1][0], [0.1, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(source.inputs[1][1], [0.3, 0.0, 0.0, 0.0])


def test_report_writes_machine_readable_rows_and_group_aggregates(
    tmp_path: Any,
) -> None:
    expert = np.zeros((4, 4), dtype=np.float32)
    expert[0, 0] = 0.8
    expert[2, 1] = -0.8
    rows = evaluate_state_hold_demo_target(
        episode_id="episode_005",
        observations=_observations([0.0] * 4),
        expert_action=expert,
        thresholds=_thresholds(),
        step_source=FrozenTickRecoverySource(
            recovery_call=10,
            action=np.zeros(4, dtype=np.float32),
        ),
        hold_horizon_steps=2,
    )

    paths = write_state_hold_demo_target_report(
        output_dir=tmp_path,
        rows=rows,
        metadata={"pipeline": "synthetic"},
    )

    jsonl_rows = [
        json.loads(line) for line in paths["rows_jsonl"].read_text().splitlines()
    ]
    summary = json.loads(paths["summary"].read_text())
    assert len(jsonl_rows) == 2
    assert paths["rows_csv"].exists()
    assert [row["group"] for row in summary["aggregate"]] == [
        "overall",
        "startup",
        "mid_cycle",
    ]
    assert summary["aggregate"][0]["state_hold_demo_target_not_reproduced_anchors"] == 2
