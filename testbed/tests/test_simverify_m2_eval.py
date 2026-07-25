from __future__ import annotations

import numpy as np
import pytest
import torch

from testbed.policies.act.adapter import ACTAdapter
from testbed.simverify.m2_eval import (
    EVENT_ORDER,
    _counterfactual_anchors,
    extract_ordered_task_events,
    simulate_latest_wins,
    validate_replay_trace_arrays,
)
from testbed.simverify.m2_eval import (
    test_intent_registry as build_test_intent_registry,
)


def _trace_arrays(*, steps: int = 3, queries: int = 2) -> dict[str, np.ndarray]:
    return {
        "raw_policy_chunk_normalized": np.zeros((steps, queries, 4), dtype=np.float32),
        "raw_policy_chunk_direct": np.zeros((steps, queries, 4), dtype=np.float32),
        "temporal_aggregation_action": np.zeros((steps, 4), dtype=np.float32),
        "future_runtime_safe_action": np.zeros((steps, 4), dtype=np.float32),
        "expert_action": np.zeros((steps, 4), dtype=np.float32),
        "condition": np.zeros((steps, 6), dtype=np.float32),
        "target_tick": np.arange(steps, dtype=np.int64),
        "source_observation_index": np.arange(steps, dtype=np.int64),
        "condition_cycle_id": np.zeros(steps, dtype=np.int64),
        "condition_valid_mask": np.ones(steps, dtype=np.uint8),
        "observation_age_ticks": np.zeros(steps, dtype=np.int64),
        "action_age_ticks": np.zeros(steps, dtype=np.int64),
    }


def test_intent_registry_freezes_hr12_for_e00_through_e07() -> None:
    registry = build_test_intent_registry()

    assert set(registry["intents"]) == {f"E{index:02d}" for index in range(8)}
    for intent in registry["intents"].values():
        assert intent["question"]
        assert intent["observable_inputs"]
        assert intent["intervention"]
        assert intent["metrics"]
        assert intent["can_prove"]
        assert intent["cannot_prove"]
        assert intent["stop_conditions"]
        assert intent["evidence_scope"] == "recorded-observation/offline"
        assert intent["closed_loop_claim_allowed"] is False
        assert intent["privilege_inputs_allowed"] is False


def test_replay_trace_requires_separate_three_stage_actions() -> None:
    arrays = _trace_arrays()
    validate_replay_trace_arrays(arrays, chunk_size=2)

    arrays["future_runtime_safe_action"] = arrays["temporal_aggregation_action"]
    with pytest.raises(ValueError, match="must not alias"):
        validate_replay_trace_arrays(arrays, chunk_size=2)


def test_event_extractor_matches_ordered_data_generated_templates() -> None:
    action = np.zeros((60, 4), dtype=np.float32)
    templates = {}
    for index, event_name in enumerate(EVENT_ORDER):
        tick = 5 + index * 9
        signature = [0, 0, 0, 0]
        signature[index % 4] = 1 if index < 4 else -1
        action[tick] = np.asarray(signature, dtype=np.float32)
        fraction = tick / (action.shape[0] - 1)
        templates[event_name] = {
            "mode_effective_signature": signature,
            "required_axis_signs": signature,
            "relative_position": {
                "p02_5": fraction,
                "p97_5": fraction,
            },
        }

    result = extract_ordered_task_events(
        action,
        templates,
        deadzone=[0.05] * 4,
    )

    assert result["required_event_coverage"] == 1.0
    assert result["event_order_valid"] is True
    assert result["missing_events"] == []
    assert list(result["event_ticks"].values()) == [0, 14, 23, 32, 41, 59]
    assert result["event_match_source"]["ready_start"] == "observable_cycle_boundary"
    assert result["physical_event_claimed"] is False


def test_latest_wins_uses_issue_age_and_times_out_to_zero() -> None:
    old = np.stack([np.full(4, float(index), dtype=np.float32) for index in range(4)])
    newer = np.stack([np.full(4, 10.0 + index, dtype=np.float32) for index in range(3)])

    result = simulate_latest_wins(
        [
            {"issue_tick": 0, "ready_tick": 1, "raw_chunk_direct": old},
            {"issue_tick": 2, "ready_tick": 4, "raw_chunk_direct": newer},
        ],
        control_ticks=8,
        timeout_ticks=4,
    )

    np.testing.assert_array_equal(
        result["source_issue_tick"], [-1, 0, 0, 0, 2, 2, 2, 2]
    )
    np.testing.assert_array_equal(result["action_age_ticks"], [-1, 1, 2, 3, 2, 3, 4, 5])
    np.testing.assert_array_equal(
        result["future_runtime_safe_action"][:, 0],
        [0, 1, 2, 3, 12, 12, 12, 0],
    )
    np.testing.assert_array_equal(
        result["timed_out"],
        [True, False, False, False, False, False, False, True],
    )


def test_counterfactual_anchors_change_exactly_one_condition_field() -> None:
    accepted = [
        {
            "episode_id": 3,
            "cycle_id": 7,
            "split": "train",
            "target_steps_20hz": [10, 20],
            "policy_condition": {
                "current_sector": "left",
                "next_ready_sector": "center",
            },
        }
    ]
    sector_support = {
        sector: {
            "supported": sector != "right",
            "distance_threshold": 0.2,
            "nearest_neighbors": [],
        }
        for sector in ("left", "center", "right")
    }
    support = {
        "entries": [
            {
                "episode_id": 3,
                "cycle_id": 7,
                "counterfactuals": {
                    "current": sector_support,
                    "next": sector_support,
                },
            }
        ]
    }

    rows = _counterfactual_anchors(accepted, support)

    assert len(rows) == 4
    assert all(row["primary_factor_count"] == 1 for row in rows)
    assert all(len(row["changed_factors"]) == 1 for row in rows)
    assert {tuple(row["changed_factors"]) for row in rows} == {
        ("current_sector",),
        ("next_sector",),
    }
    assert sum(row["supported"] for row in rows) == 2


def test_act_adapter_exposes_direct_raw_chunk_as_independent_copy() -> None:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter._last_raw_action_chunk = torch.tensor(
        [[0.0, 1.0, -1.0, 2.0]],
        dtype=torch.float32,
    )
    adapter.norm_stats = {
        "action_mean": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "action_std": np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
    }

    normalized = adapter.last_raw_action_chunk()
    direct = adapter.last_raw_action_chunk_direct()

    np.testing.assert_array_equal(direct, [[1.0, 5.0, -1.0, 14.0]])
    assert direct.dtype == np.float32
    assert not np.shares_memory(normalized, direct)
    direct[0, 0] = 999.0
    assert adapter.last_raw_action_chunk_direct()[0, 0] == 1.0
