from __future__ import annotations

from typing import Any

import numpy as np
import torch

from testbed.actions.policy import PolicyActionSource
from testbed.data.causal_visual_history import CausalVisualHistory
from testbed.policies.act.adapter import ACTAdapter
from testbed.policies.act.factorized_action import FactorizedTemporalAggregator
from testbed.policies.runtime_gate_stack import RuntimeGateStack
from testbed.policies.state_hold_demo_target import evaluate_state_hold_demo_target


def _thresholds() -> dict[str, dict[str, float]]:
    return {
        axis: {"pos": 0.5, "neg": 0.5} for axis in ("swing", "boom", "stick", "bucket")
    }


def _observations(length: int) -> list[dict[str, np.ndarray]]:
    return [
        {
            "qpos": np.array([step / 10.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "qvel": np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
            "image_fpv": np.full((2, 2, 3), step, dtype=np.uint8),
        }
        for step in range(length)
    ]


class _SnapshotAccumulatorSource:
    def __init__(self) -> None:
        self.model_calls = 0
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.step_index = 0

    def step(self, observation: dict[str, Any]) -> np.ndarray:
        self.model_calls += 1
        qpos = float(np.asarray(observation["qpos"])[0])
        qvel = float(np.asarray(observation["qvel"])[0])
        self.total += qpos + qvel
        self.step_index += 1
        return np.array(
            [0.05 * self.total, -0.03 * self.step_index, 0.0, 0.0],
            dtype=np.float32,
        )

    def snapshot_state(self) -> dict[str, float | int]:
        return {"total": float(self.total), "step_index": int(self.step_index)}

    def restore_state(self, state: dict[str, float | int]) -> None:
        self.total = float(state["total"])
        self.step_index = int(state["step_index"])


class _LegacyView:
    def __init__(self, source: _SnapshotAccumulatorSource) -> None:
        self.source = source

    def reset(self) -> None:
        self.source.reset()

    def step(self, observation: dict[str, Any]) -> np.ndarray:
        return self.source.step(observation)


def test_shared_prefix_is_exactly_equal_to_legacy_replay_with_fewer_calls() -> None:
    expert = np.zeros((12, 4), dtype=np.float32)
    expert[2, 0] = 0.8
    expert[5, 1] = -0.8
    expert[8, 3] = 0.8
    observations = _observations(12)

    optimized_source = _SnapshotAccumulatorSource()
    optimized_counters: dict[str, Any] = {}
    optimized_rows = evaluate_state_hold_demo_target(
        episode_id="episode_optimized",
        observations=observations,
        expert_action=expert,
        thresholds=_thresholds(),
        step_source=optimized_source,
        hold_horizon_steps=3,
        trace_full_horizon_after_reproduction=True,
        instrumentation=optimized_counters,
    )

    legacy_source = _SnapshotAccumulatorSource()
    legacy_counters: dict[str, Any] = {}
    legacy_rows = evaluate_state_hold_demo_target(
        episode_id="episode_optimized",
        observations=observations,
        expert_action=expert,
        thresholds=_thresholds(),
        step_source=_LegacyView(legacy_source),
        hold_horizon_steps=3,
        trace_full_horizon_after_reproduction=True,
        instrumentation=legacy_counters,
    )

    assert optimized_rows == legacy_rows
    assert optimized_counters["evaluation_path"] == "shared_teacher_prefix"
    assert legacy_counters["evaluation_path"] == "legacy_replay"
    assert optimized_counters["source_step_calls"] == 20
    assert legacy_counters["source_step_calls"] == 48
    assert optimized_source.model_calls == 20
    assert legacy_source.model_calls == 48


def test_act_adapter_snapshot_restores_all_nested_state_without_aliasing() -> None:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.device = torch.device("cpu")
    adapter._t = 3
    adapter._all_time_actions = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    adapter._temporal_weight_cache = {2: torch.tensor([[0.4], [0.6]])}
    adapter._cached_actions = torch.ones(2, 4)
    adapter._last_temporal_aggregation_diagnostics = {"trace": [1, 2]}
    adapter._last_factorized_diagnostics = {"factor": np.array([1.0])}
    adapter._last_goal_effect_diagnostics = {"goal": [3.0]}
    adapter._last_temporal_input_diagnostics = {"accepted": {"video4": True}}
    adapter._temporal_last_timestamps = {"video4": 11}
    adapter._temporal_fallback_timestamp = 12
    adapter._visual_history = CausalVisualHistory(["video4"], history_length=2)
    adapter._visual_history.append(
        {"video4": np.ones((3, 2, 2), dtype=np.float32)},
        {"video4": 11},
    )
    adapter._factorized_aggregator = FactorizedTemporalAggregator(
        num_queries=2,
        device=torch.device("cpu"),
        max_episode_len=4,
        exponential_k=0.01,
    )
    adapter._factorized_aggregator.t = 1
    adapter._factorized_aggregator._values = torch.ones(4, 6, 20)
    adapter._factorized_aggregator._occupied = torch.ones(4, 6, dtype=torch.bool)
    adapter._factorized_aggregator._weight_cache = {1: torch.ones(1)}

    state = adapter.snapshot_state()
    adapter._all_time_actions.fill_(99.0)
    adapter._temporal_weight_cache[2].fill_(99.0)
    adapter._cached_actions.fill_(99.0)
    adapter._last_temporal_aggregation_diagnostics["trace"].append(99)
    adapter._temporal_last_timestamps["video4"] = 99
    adapter._visual_history.append(
        {"video4": np.full((3, 2, 2), 2.0, dtype=np.float32)},
        {"video4": 12},
    )
    adapter._factorized_aggregator._values.fill_(99.0)

    adapter.restore_state(state)

    assert adapter._t == 3
    torch.testing.assert_close(
        adapter._all_time_actions,
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
    )
    torch.testing.assert_close(
        adapter._temporal_weight_cache[2], torch.tensor([[0.4], [0.6]])
    )
    assert adapter._last_temporal_aggregation_diagnostics == {"trace": [1, 2]}
    assert adapter._temporal_last_timestamps == {"video4": 11}
    assert adapter._visual_history.snapshot().timestamps_ns["video4"].tolist() == [
        11,
        11,
    ]
    assert torch.all(adapter._factorized_aggregator._values == 1.0)

    adapter._all_time_actions.fill_(7.0)
    assert not torch.all(state.all_time_actions == 7.0)
    adapter.reset()
    assert adapter._t == 0
    assert adapter._all_time_actions is None
    assert adapter._cached_actions is None
    assert adapter._factorized_aggregator.t == 0


def test_runtime_gate_snapshot_restore_has_no_array_aliases() -> None:
    gate = RuntimeGateStack.__new__(RuntimeGateStack)
    gate._feature_history = [np.arange(20, dtype=np.float32)]
    gate._candidate_run = 3
    gate._eligibility_run = 2
    gate._gohome_emitted = True

    state = gate.snapshot_state()
    gate._feature_history[0].fill(99.0)
    gate._candidate_run = 0
    gate.restore_state(state)

    np.testing.assert_array_equal(gate._feature_history[0], np.arange(20))
    assert gate._candidate_run == 3
    assert gate._eligibility_run == 2
    assert gate._gohome_emitted is True
    gate._feature_history[0].fill(7.0)
    assert not np.all(state.feature_history[0] == 7.0)


def test_policy_action_source_snapshot_captures_qvel_assist_policy_and_gate() -> None:
    class _StateOwner:
        def __init__(self, value: int) -> None:
            self.value = value

        def snapshot_state(self) -> int:
            return int(self.value)

        def restore_state(self, state: int) -> None:
            self.value = int(state)

        def reset(self) -> None:
            self.value = 0

    policy = _StateOwner(5)
    gate = _StateOwner(6)
    source = PolicyActionSource(
        policy=policy,
        source_id="test",
        output_mode="control",
        qvel_mode="raw",
        fail_safe_zero=False,
        runtime_gate_stack=gate,
    )
    source._step = 7
    source._record_start_pending = True
    source._last_qpos = np.arange(4, dtype=np.float32)
    source._last_obs_time_ns = 123
    source._filtered_qvel[:] = 0.25
    source._assist_last_sign[:] = [1, -1, 0, 1]
    source._assist_consecutive_steps[:] = [2, 3, 0, 4]

    state = source.snapshot_state()
    source._last_qpos.fill(99.0)
    source._filtered_qvel.fill(99.0)
    source._assist_last_sign.fill(0)
    source._assist_consecutive_steps.fill(0)
    policy.value = 50
    gate.value = 60
    source.restore_state(state)

    np.testing.assert_array_equal(source._last_qpos, np.arange(4))
    np.testing.assert_array_equal(source._filtered_qvel, np.full(4, 0.25))
    np.testing.assert_array_equal(source._assist_last_sign, [1, -1, 0, 1])
    np.testing.assert_array_equal(source._assist_consecutive_steps, [2, 3, 0, 4])
    assert source._step == 7
    assert source._last_obs_time_ns == 123
    assert policy.value == 5
    assert gate.value == 6
    source._last_qpos.fill(8.0)
    assert not np.all(state.last_qpos == 8.0)
