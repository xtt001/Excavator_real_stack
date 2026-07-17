import numpy as np
import torch

from testbed.policies.act.goal_effect import (
    GoalEffectHead,
    build_goal_effect_targets,
    future_delta_scale,
    goal_effect_loss_terms,
    resolve_goal_effect_config,
)


def test_goal_effect_targets_are_causal_and_mask_structural_stick() -> None:
    qpos = np.zeros((30, 4), dtype=np.float32)
    qpos[5:, 0] = np.linspace(0.0, 0.2, 25)
    qpos[5:, 3] = np.linspace(0.0, -0.4, 25)
    qvel = np.gradient(qpos, axis=0).astype(np.float32)
    action = np.zeros_like(qpos)
    config = resolve_goal_effect_config({"enabled": True, "horizons": [4, 8, 20]})

    targets = build_goal_effect_targets(
        qpos=qpos,
        qvel=qvel,
        action=action,
        timestep=5,
        config=config,
    )

    assert targets["goal_future_delta"].shape == (3, 4)
    assert targets["goal_future_valid"].all(axis=0).tolist() == [True, True, False, True]
    assert not targets["goal_effect_valid"][:, 2].any()
    assert targets["goal_future_direction"][:, 0].tolist() == [2, 2, 2]
    assert targets["goal_future_direction"][:, 3].tolist() == [0, 0, 0]


def test_future_delta_scale_never_crosses_episode_boundaries() -> None:
    scale = future_delta_scale(
        [np.zeros((5, 4), dtype=np.float32), np.ones((5, 4), dtype=np.float32)],
        [4],
    )
    assert scale.shape == (4,)
    assert np.allclose(scale, 1.0e-3)


def test_goal_effect_head_and_masked_loss_contract() -> None:
    config = resolve_goal_effect_config(
        {"enabled": True, "horizons": [4, 8], "delta_scale": [0.1] * 4}
    )
    head = GoalEffectHead(
        hidden_dim=16,
        action_dim=4,
        num_queries=5,
        horizons=config.horizons,
    )
    context = torch.randn(2, 16)
    proposal = torch.randn(2, 5, 4)
    outputs = head(context, proposal)
    assert outputs["goal_delta"].shape == (2, 2, 4)
    assert outputs["goal_direction_logits"].shape == (2, 2, 4, 3)
    targets = {
        "goal_future_delta": torch.zeros(2, 2, 4),
        "goal_future_valid": torch.ones(2, 2, 4, dtype=torch.bool),
        "goal_future_direction": torch.ones(2, 2, 4, dtype=torch.long),
        "goal_effect_delta": torch.zeros(2, 2, 4),
        "goal_effect_valid": torch.ones(2, 2, 4, dtype=torch.bool),
    }
    losses = goal_effect_loss_terms(
        outputs=outputs,
        targets=targets,
        config=config,
        device_like=context,
    )
    assert torch.isfinite(losses["goal_effect_loss"])
    losses["goal_effect_loss"].backward()
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_disabled_goal_effect_has_zero_loss_without_targets() -> None:
    config = resolve_goal_effect_config({"enabled": False})
    losses = goal_effect_loss_terms(
        outputs=None,
        targets=None,
        config=config,
        device_like=torch.ones(2, 3),
    )
    assert all(value.item() == 0.0 for value in losses.values())
