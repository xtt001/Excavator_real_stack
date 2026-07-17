from __future__ import annotations

import pytest
import torch

from testbed.policies.act.adapter import (
    ACTAdapter,
    _held_temporal_prefix_actions,
    _resolve_demo_target_hold_loss_config,
)


def _config(**overrides):
    config = {
        "enabled": True,
        "thresholds": {
            "swing": {"pos": 0.5, "neg": 0.6},
            "boom": {"pos": 0.4, "neg": 0.4},
            "stick": {"pos": 0.3, "neg": 0.3},
            "bucket": {"pos": 0.2, "neg": 0.2},
        },
        "weight": 2.0,
        "assist_trigger_fraction": 0.5,
        "margin": 0.05,
        "hold_horizon_steps": 3,
        "min_consecutive_steps": 2,
    }
    config.update(overrides)
    return config


def _adapter(**overrides) -> ACTAdapter:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.norm_stats = {
        "action_mean": torch.zeros(4, dtype=torch.float32).numpy(),
        "action_std": torch.ones(4, dtype=torch.float32).numpy(),
    }
    adapter._demo_target_hold_loss = _resolve_demo_target_hold_loss_config(
        _config(**overrides)
    )
    return adapter


def test_disabled_demo_target_hold_loss_needs_no_thresholds() -> None:
    resolved = _resolve_demo_target_hold_loss_config({"enabled": False})

    assert resolved["enabled"] is False
    assert resolved["hold_horizon_steps"] == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"weight": -0.1}, "weight"),
        ({"assist_trigger_fraction": 0.0}, "assist_trigger_fraction"),
        ({"margin": -0.01}, "margin"),
        ({"hold_horizon_steps": 0}, "hold_horizon_steps"),
        (
            {"hold_horizon_steps": 2, "min_consecutive_steps": 3},
            "min_consecutive_steps",
        ),
    ],
)
def test_demo_target_hold_config_rejects_invalid_values(
    overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_demo_target_hold_loss_config(_config(**overrides))


def test_held_temporal_prefix_actions_matches_runtime_query_order() -> None:
    policy = torch.tensor(
        [[[0.1, 0.0], [0.3, 0.0], [0.5, 0.0]]],
        dtype=torch.float32,
    )

    held = _held_temporal_prefix_actions(policy, hold_horizon_steps=3)

    decay = torch.exp(torch.tensor(-0.01))
    expected_tick_1 = (torch.tensor(0.3) + decay * torch.tensor(0.1)) / (1.0 + decay)
    weights_2 = torch.exp(-0.01 * torch.arange(3, dtype=torch.float32))
    expected_tick_2 = (
        torch.tensor([0.5, 0.3, 0.1]) * weights_2
    ).sum() / weights_2.sum()
    assert torch.isclose(held[0, 0, 0], torch.tensor(0.1))
    assert torch.isclose(held[0, 1, 0], expected_tick_1)
    assert torch.isclose(held[0, 2, 0], expected_tick_2)


def test_state_hold_loss_targets_last_consecutive_assist_trigger_ticks() -> None:
    adapter = _adapter()
    policy = torch.zeros((1, 3, 4), dtype=torch.float32, requires_grad=True)
    transition = torch.zeros((1, 4, 2), dtype=torch.bool)
    transition[0, 0, 0] = True
    transition[0, 1, 1] = True

    terms = adapter._demo_target_hold_loss_terms(
        policy_normalized=policy,
        transition_mask=transition,
    )

    # swing+ target = 0.5 * 0.5 + 0.05 = 0.30;
    # boom- target = 0.5 * 0.4 + 0.05 = 0.25.
    assert torch.isclose(terms["state_hold_pos_shortfall_loss"], torch.tensor(0.30))
    assert torch.isclose(terms["state_hold_neg_shortfall_loss"], torch.tensor(0.25))
    assert torch.isclose(terms["demo_target_hold_loss"], torch.tensor(1.10))

    terms["demo_target_hold_loss"].backward()
    assert policy.grad is not None
    assert policy.grad[0, :, 0].sum() < 0.0
    assert policy.grad[0, :, 1].sum() > 0.0


def test_state_hold_loss_is_zero_without_a_transition_anchor() -> None:
    adapter = _adapter()

    terms = adapter._demo_target_hold_loss_terms(
        policy_normalized=torch.zeros((2, 3, 4), dtype=torch.float32),
        transition_mask=torch.zeros((2, 4, 2), dtype=torch.bool),
    )

    assert torch.isclose(terms["demo_target_hold_loss"], torch.tensor(0.0))


def test_state_hold_loss_rejects_missing_or_ambiguous_transition_mask() -> None:
    adapter = _adapter()
    policy = torch.zeros((1, 3, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="requires state_hold_transition_mask"):
        adapter._demo_target_hold_loss_terms(
            policy_normalized=policy,
            transition_mask=None,
        )

    ambiguous = torch.zeros((1, 4, 2), dtype=torch.bool)
    ambiguous[0, 0] = True
    with pytest.raises(ValueError, match="cannot select both directions"):
        adapter._demo_target_hold_loss_terms(
            policy_normalized=policy,
            transition_mask=ambiguous,
        )
