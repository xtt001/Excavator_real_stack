import numpy as np
import torch

from testbed.policies.act.effective_action import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    compute_effective_action_labels,
    effective_action_loss_terms,
    resolve_effective_action_config,
    weighted_action_l1,
)

THRESHOLDS = {
    "swing": {"pos": 0.6, "neg": 0.7},
    "boom": {"pos": 0.25, "neg": 0.35},
    "stick": {"pos": 0.5, "neg": 0.5},
    "bucket": {"pos": 0.4, "neg": 0.5},
}


def test_effective_target_separates_neutral_and_signed_active_actions() -> None:
    actions = np.asarray(
        [
            [0.10, 0.24, 0.0, 0.39],
            [0.61, 0.25, -0.5, -0.5],
            [0.62, 0.25, -0.5, -0.5],
        ],
        dtype=np.float32,
    )
    labels = compute_effective_action_labels(
        actions=actions,
        thresholds=THRESHOLDS,
        transition_window_steps=2,
        persistence_steps=2,
    )
    assert np.allclose(labels.action[0], 0.0)
    assert labels.phase[0, 0] == NEUTRAL
    assert labels.phase[1, 0] == POSITIVE
    assert labels.phase[1, 3] == NEGATIVE
    assert labels.transition[1, 0]
    assert labels.persistent[1, 0]
    assert labels.phase[1, 2] == NEGATIVE


def test_active_margin_pushes_boundary_outward_but_preserves_saturation() -> None:
    actions = np.asarray(
        [[0.661, 0.259, 0.5, 0.408], [1.0, 1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    labels = compute_effective_action_labels(
        actions=actions,
        thresholds=THRESHOLDS,
        active_margin=0.02,
    )
    assert labels.action[0, 0] > THRESHOLDS["swing"]["pos"]
    assert labels.action[0, 1] > THRESHOLDS["boom"]["pos"]
    assert np.allclose(labels.action[1], 1.0)


def test_effective_action_requires_thresholds_when_enabled() -> None:
    try:
        resolve_effective_action_config({"enabled": True})
    except ValueError as exc:
        assert "threshold_json" in str(exc)
    else:
        raise AssertionError("missing thresholds must be rejected")


def test_effective_action_auxiliary_loss_is_finite_and_masked() -> None:
    config = resolve_effective_action_config(
        {
            "enabled": True,
            "thresholds": THRESHOLDS,
            "classification_weight": 0.2,
            "magnitude_weight": 0.1,
        }
    )
    target = torch.zeros((1, 2, 4), dtype=torch.float32)
    policy = torch.zeros_like(target)
    phase = torch.tensor([[[NEUTRAL, POSITIVE, NEUTRAL, NEGATIVE]] * 2])
    valid = torch.ones((1, 2, 4), dtype=torch.bool)
    logits = torch.zeros((1, 2, 12), dtype=torch.float32)
    result = effective_action_loss_terms(
        target_normalized=target,
        policy_normalized=policy,
        phase_logits=logits,
        phase_labels=phase,
        phase_valid=valid,
        valid_mask=torch.ones((1, 2, 1), dtype=torch.bool),
        loss_weight=torch.ones_like(target),
        action_mean=np.zeros(4, dtype=np.float32),
        action_std=np.ones(4, dtype=np.float32),
        config=config,
    )
    assert torch.isfinite(result["effective_action_loss"])
    assert float(result["effective_action_active_l1"]) == 0.0


def test_weighted_action_l1_emphasizes_transition_axis_without_gating() -> None:
    target = torch.zeros((1, 1, 4), dtype=torch.float32)
    policy = torch.ones_like(target)
    weights = torch.tensor([[[4.0, 1.0, 1.0, 1.0]]])
    loss = weighted_action_l1(
        expert=target,
        policy=policy,
        valid_mask=torch.ones((1, 1, 1), dtype=torch.bool),
        loss_weight=weights,
    )
    assert torch.isfinite(loss)
    assert float(loss) == 1.0
