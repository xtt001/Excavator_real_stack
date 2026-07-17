from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from testbed.actions.policy import PolicyActionSource
from testbed.policies.act.adapter import ACTAdapter
from testbed.policies.state_hold_demo_target import (
    StepOutput,
    evaluate_state_hold_demo_target,
)


def _adapter(*, diagnostics_enabled: bool) -> ACTAdapter:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.device = torch.device("cpu")
    adapter.norm_stats = {
        "action_mean": np.array([0.5, -1.0, 2.0, -0.5], dtype=np.float32),
        "action_std": np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
    }
    adapter._num_queries = 3
    adapter._t = 0
    adapter._all_time_actions = None
    adapter._temporal_weight_cache = {}
    adapter._max_episode_len = 8
    adapter._temporal_aggregation_diagnostics_enabled = diagnostics_enabled
    adapter._last_temporal_aggregation_diagnostics = None
    return adapter


def _chunk(start: float) -> torch.Tensor:
    return torch.tensor(
        [
            [
                [start, start + 1.0, start + 2.0, start + 3.0],
                [start + 4.0, start + 5.0, start + 6.0, start + 7.0],
                [start + 8.0, start + 9.0, start + 10.0, start + 11.0],
            ]
        ],
        dtype=torch.float32,
    )


def test_opt_in_diagnostics_leave_legacy_aggregate_bit_exact() -> None:
    baseline = _adapter(diagnostics_enabled=False)
    diagnostic = _adapter(diagnostics_enabled=True)

    for start in (1.0, 11.0, 21.0, 31.0):
        baseline_action = baseline._aggregate(_chunk(start))
        diagnostic_action = diagnostic._aggregate(_chunk(start))
        np.testing.assert_array_equal(diagnostic_action, baseline_action)
        baseline._t += 1
        diagnostic._t += 1

    assert baseline.temporal_aggregation_diagnostics is None
    assert diagnostic.temporal_aggregation_diagnostics is not None


def test_decomposition_uses_current_query_zero_and_direct_action_units() -> None:
    adapter = _adapter(diagnostics_enabled=True)
    adapter._aggregate(_chunk(1.0))
    adapter._t = 1
    assert adapter._all_time_actions is not None
    adapter._all_time_actions[5, 1] = torch.full((4,), 99.0)

    normalized_legacy = adapter._aggregate(_chunk(11.0))
    diagnostics = adapter.temporal_aggregation_diagnostics

    assert diagnostics is not None
    assert diagnostics["policy_temporal_aggregation_query_step"] == 1
    assert diagnostics["policy_temporal_aggregation_source_steps"] == [0, 1]
    assert diagnostics["policy_temporal_aggregation_query_offsets"] == [1, 0]
    assert diagnostics["policy_temporal_aggregation_population"] == 2
    assert max(diagnostics["policy_temporal_aggregation_source_steps"]) <= 1

    mean = adapter.norm_stats["action_mean"]
    std = adapter.norm_stats["action_std"]
    newest_normalized = _chunk(11.0)[0, 0].numpy()
    np.testing.assert_allclose(
        diagnostics["policy_temporal_aggregation_newest_action"],
        newest_normalized * std + mean,
    )
    np.testing.assert_allclose(
        diagnostics["policy_temporal_aggregation_legacy_action"],
        normalized_legacy * std + mean,
    )

    old_query_one = _chunk(1.0)[0, 1]
    newest_query_zero = _chunk(11.0)[0, 0]
    legacy_weights = torch.exp(-0.01 * torch.arange(2, dtype=torch.float32))
    legacy_weights = legacy_weights / legacy_weights.sum()
    expected_recency = torch.stack([old_query_one, newest_query_zero])
    expected_recency = (expected_recency * legacy_weights.flip(0)[:, None]).sum(0)
    np.testing.assert_allclose(
        diagnostics["policy_temporal_aggregation_recency_action"],
        expected_recency.numpy() * std + mean,
    )


def test_reset_clears_decomposition_trace() -> None:
    adapter = _adapter(diagnostics_enabled=True)
    adapter._aggregate(_chunk(1.0))
    diagnostics = adapter.temporal_aggregation_diagnostics
    assert diagnostics is not None
    diagnostics["policy_temporal_aggregation_source_steps"].append(99)
    assert adapter.temporal_aggregation_diagnostics[
        "policy_temporal_aggregation_source_steps"
    ] == [0]

    adapter._visual_history = None
    adapter._temporal_last_timestamps = {}
    adapter._temporal_fallback_timestamp = 4
    adapter._last_temporal_input_diagnostics = {"enabled": False}
    adapter._factorized_aggregator = None
    adapter.reset()

    assert adapter._t == 0
    assert adapter._all_time_actions is None
    assert adapter.temporal_aggregation_diagnostics is None


class _PolicyWithAggregationDiagnostics:
    def reset(self) -> None:
        pass

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        del obs
        return np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)

    @property
    def temporal_aggregation_diagnostics(self) -> dict[str, Any]:
        return {
            "policy_temporal_aggregation_action_domain": "direct_policy_output",
            "policy_temporal_aggregation_legacy_action": [0.2, 0.3, 0.4, 0.5],
            "policy_temporal_aggregation_newest_action": [0.2, 0.3, 0.4, 0.8],
            "policy_temporal_aggregation_recency_action": [0.2, 0.3, 0.4, 0.6],
        }


def test_policy_action_source_passes_decomposition_without_changing_action() -> None:
    source = PolicyActionSource(
        policy=_PolicyWithAggregationDiagnostics(),
        source_id="unit",
        camera_name="fpv",
        action_scale=[1.0, 1.0, 1.0, 1.0],
        output_mode="control",
        fail_safe_zero=False,
    )
    observation = {
        "qpos": np.zeros(4, dtype=np.float32),
        "qvel": np.zeros(4, dtype=np.float32),
        "images": {"fpv": np.zeros((2, 3, 3), dtype=np.uint8)},
    }

    action, info = source.next_action(observation)

    np.testing.assert_array_equal(
        action,
        np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
    )
    assert info.extras["policy_temporal_aggregation_newest_action"] == [
        0.2,
        0.3,
        0.4,
        0.8,
    ]


def test_policy_action_source_config_enables_diagnostics_only_when_requested() -> None:
    with patch(
        "testbed.actions.policy.load_act_policy_from_bundle",
        return_value=_PolicyWithAggregationDiagnostics(),
    ) as loader:
        PolicyActionSource.from_config(
            {
                "bundle_dir": "/tmp/unit_bundle",
                "temporal_aggregation_diagnostics": True,
            }
        )

    assert loader.call_args.kwargs["temporal_aggregation_diagnostics"] is True

    with patch(
        "testbed.actions.policy.load_act_policy_from_bundle",
        return_value=_PolicyWithAggregationDiagnostics(),
    ) as default_loader:
        PolicyActionSource.from_config({"bundle_dir": "/tmp/unit_bundle"})

    assert default_loader.call_args.kwargs["temporal_aggregation_diagnostics"] is False


class _DecompositionStepSource:
    def __init__(self) -> None:
        self.call = 0

    def reset(self) -> None:
        self.call = 0

    def step(self, observation: Mapping[str, Any]) -> StepOutput:
        del observation
        query_step = self.call
        self.call += 1
        legacy = [0.40, 0.0, 0.0, 0.0]
        newest = [0.70, 0.0, 0.0, 0.0]
        recency = [0.60, 0.0, 0.0, 0.0]
        return StepOutput(
            action=np.asarray(legacy, dtype=np.float32),
            diagnostics={
                "policy_temporal_aggregation_action_domain": ("direct_policy_output"),
                "policy_temporal_aggregation_query_step": query_step,
                "policy_temporal_aggregation_source_steps": list(range(query_step + 1)),
                "policy_temporal_aggregation_legacy_action": legacy,
                "policy_temporal_aggregation_newest_action": newest,
                "policy_temporal_aggregation_recency_action": recency,
            },
        )


def test_state_hold_reports_newest_crossing_when_legacy_misses() -> None:
    expert = np.zeros((3, 4), dtype=np.float32)
    expert[0, 0] = 0.8
    observations = [
        {
            "qpos": np.zeros(4, dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
        }
        for _ in range(3)
    ]
    thresholds = {
        axis: {"pos": 0.5, "neg": 0.5} for axis in ("swing", "boom", "stick", "bucket")
    }

    rows = evaluate_state_hold_demo_target(
        episode_id="episode_unit",
        observations=observations,
        expert_action=expert,
        thresholds=thresholds,
        step_source=_DecompositionStepSource(),
        hold_horizon_steps=2,
    )

    row = rows[0]
    assert row["state_hold_status"] == "demo_target_not_reproduced"
    assert row["state_hold_temporal_aggregation_decomposition_complete"] is True
    assert row["state_hold_newest_crosses_legacy_misses_ticks"] == [0, 1]
    assert row["state_hold_newest_crosses_legacy_misses_tick_count"] == 2
    assert row["state_hold_recency_crosses_legacy_misses_ticks"] == [0, 1]
    assert (
        row["state_hold_temporal_aggregation_decomposition_trace"][0][
            "newest_crosses_legacy_misses"
        ]
        is True
    )
