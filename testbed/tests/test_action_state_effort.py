from __future__ import annotations

import numpy as np
import torch

from testbed.policies.act.action_state_effort import (
    IDLE,
    NEG_NEAR,
    POS_NEAR,
    POS_SAFE,
    action_state_loss_terms,
    compute_action_state_labels,
    resolve_action_state_effort_config,
    summarize_action_state_labels,
)


def _thresholds() -> dict[str, dict[str, float]]:
    return {
        "swing": {"pos": 0.6, "neg": 0.7},
        "boom": {"pos": 0.25, "neg": 0.35},
        "stick": {"pos": 0.5, "neg": 0.5},
        "bucket": {"pos": 0.4, "neg": 0.5},
    }


def test_action_state_labels_separate_idle_near_and_safe() -> None:
    actions = np.zeros((4, 4), dtype=np.float32)
    actions[0, 1] = 0.10
    actions[1, 1] = 0.25
    actions[2, 1] = 0.27
    actions[3, 1] = 0.40
    labels = compute_action_state_labels(
        actions=actions,
        thresholds=_thresholds(),
        safe_margin=0.02,
        persistence_steps=2,
    )
    assert labels.state[:, 1].tolist() == [IDLE, POS_NEAR, POS_SAFE, POS_SAFE]
    assert labels.signed_margin[1, 1, 0] == 0.0
    assert bool(labels.persistent_effective[2, 1, 0]) is True
    assert bool(labels.persistent_effective[3, 1, 0]) is False


def test_action_state_labels_preserve_negative_direction() -> None:
    actions = np.zeros((2, 4), dtype=np.float32)
    actions[:, 1] = [-0.36, -0.40]
    labels = compute_action_state_labels(
        actions=actions,
        thresholds=_thresholds(),
        safe_margin=0.02,
        persistence_steps=1,
    )
    assert labels.state[:, 1].tolist() == [NEG_NEAR, 4]


def test_action_state_loss_penalizes_sub_deadzone_policy_for_active_target() -> None:
    config = resolve_action_state_effort_config(
        {
            "enabled": True,
            "thresholds": _thresholds(),
            "safe_margin": 0.02,
            "required_margin": 0.02,
            "class_weights": [1, 2, 2, 2, 2],
        }
    )
    policy = torch.zeros((1, 2, 4), dtype=torch.float32)
    policy[:, :, 1] = 0.10
    labels = torch.full((1, 2, 4), IDLE, dtype=torch.long)
    labels[:, :, 1] = POS_SAFE
    valid = torch.ones((1, 2, 4), dtype=torch.bool)
    logits = torch.zeros((1, 2, 20), dtype=torch.float32)
    persistent = torch.zeros((1, 2, 4, 2), dtype=torch.bool)
    result = action_state_loss_terms(
        policy_direct=policy,
        state_logits=logits,
        state_labels=labels,
        state_valid=valid,
        persistent_effective=persistent,
        config=config,
    )
    assert float(result["action_state_margin_loss"]) > 0.0
    assert float(result["action_state_loss"]) > 0.0


def test_action_state_census_is_reviewable() -> None:
    actions = np.zeros((2, 4), dtype=np.float32)
    actions[0, 3] = 0.41
    labels = compute_action_state_labels(
        actions=actions,
        thresholds=_thresholds(),
        safe_margin=0.02,
        persistence_steps=1,
    )
    summary = summarize_action_state_labels(labels)
    assert summary["state_order"] == ["idle", "pos_near", "pos_safe", "neg_near", "neg_safe"]
    assert summary["counts"]["bucket"]["pos_near"] == 1


def test_action_state_current_steps_restricts_auxiliary_supervision() -> None:
    config = resolve_action_state_effort_config(
        {
            "enabled": True,
            "thresholds": _thresholds(),
            "current_steps": 1,
            "classification_weight": 0.0,
            "margin_weight": 1.0,
            "idle_weight": 0.0,
            "wrong_weight": 0.0,
        }
    )
    policy = torch.zeros((1, 2, 4), dtype=torch.float32)
    policy[:, 1, 1] = 0.10
    labels = torch.full((1, 2, 4), IDLE, dtype=torch.long)
    labels[:, 1, 1] = POS_SAFE
    valid = torch.ones((1, 2, 4), dtype=torch.bool)
    logits = torch.zeros((1, 2, 20), dtype=torch.float32)
    result = action_state_loss_terms(
        policy_direct=policy,
        state_logits=logits,
        state_labels=labels,
        state_valid=valid,
        persistent_effective=None,
        config=config,
    )
    assert float(result["action_state_margin_loss"]) == 0.0
