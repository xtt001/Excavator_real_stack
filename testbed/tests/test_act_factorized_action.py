from __future__ import annotations

import math

import pytest
import torch

from testbed.policies.act.factorized_action import (
    FactorizedTemporalAggregator,
    direct_tri_state_labels,
    factorized_training_loss_terms,
    intent_logits_to_tri_logits,
    project_factorized_action,
    query_factorized_values,
    resolve_factorized_config,
)


def _raw_config(*, chunk_size: int = 3, **overrides) -> dict:
    config = {
        "enabled": True,
        "intent_dim": 8,
        "intent_layout": "axis_major_pos_neg",
        "tri_state_order": ["neg", "idle", "pos"],
        "idle_logit": 0.0,
        "classification": {
            "weight": 0.05,
            "class_weights": [8.0, 1.0, 8.0],
            "label_domain": "direct_policy_output",
        },
        "effort": {
            "transform": "abs_after_unnormalize",
            "loss": "direct_magnitude_l1",
            "weight": 1.0,
        },
        "temporal": {
            "enabled": True,
            "chunk_size": chunk_size,
            "exponential_k": 0.01,
            "source_order": "oldest_to_newest",
            "aggregate": "probabilities_and_effort",
        },
        "held_prefix": {
            "enabled": True,
            "weight": 0.1,
            "hold_horizon_steps": chunk_size,
            "target_delays": [chunk_size - 2, chunk_size - 1],
        },
        "selection": {
            "mode": "strict_argmax",
            "tie_break": "idle",
            "nonfinite": "error",
        },
        "projection": {
            "thresholds": {
                "swing": {"pos": 0.5, "neg": 0.6},
                "boom": {"pos": 0.4, "neg": 0.45},
                "stick": {"pos": 0.3, "neg": 0.35},
                "bucket": {"pos": 0.2, "neg": 0.25},
            },
            "margin": 0.02,
            "clip": 1.0,
            "action_scale": [1.0, 1.0, 1.0, 1.0],
        },
    }
    config.update(overrides)
    return config


def _config(*, chunk_size: int = 3) -> dict:
    return resolve_factorized_config(
        _raw_config(chunk_size=chunk_size), num_queries=chunk_size
    )


def _stats(*, mean=(0.0, 0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0, 1.0)):
    return {
        "action_mean": torch.tensor(mean, dtype=torch.float32).numpy(),
        "action_std": torch.tensor(std, dtype=torch.float32).numpy(),
    }


def test_intent_layout_maps_pos_neg_checkpoint_logits_to_neg_idle_pos() -> None:
    logits = torch.tensor([[[1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]]])

    tri = intent_logits_to_tri_logits(logits)

    assert tri.shape == (1, 1, 4, 3)
    assert torch.equal(tri[0, 0, 0], torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.equal(tri[0, 0, 3], torch.tensor([-4.0, 0.0, 4.0]))


def test_direct_labels_use_asymmetric_thresholds_and_idle_interval() -> None:
    cfg = _config()
    actions = torch.tensor([[[-0.60, -0.44, 0.00, 0.20], [-0.59, 0.40, -0.35, 0.19]]])

    labels = direct_tri_state_labels(actions, pos=cfg["pos"], neg=cfg["neg"])

    assert labels.tolist() == [[[0, 1, 1, 2], [1, 2, 0, 1]]]


def test_magnitude_loss_unnormalizes_before_taking_absolute_value() -> None:
    cfg = _config()
    expert = torch.zeros((1, 3, 4), dtype=torch.float32)
    policy = torch.zeros_like(expert)
    # With mean=0.2 and std=0.5, normalized expert 0 means direct +0.2,
    # while normalized policy -0.8 means direct -0.2.  Direct magnitudes match.
    policy[..., 0] = -0.8
    logits = torch.zeros((1, 3, 8), dtype=torch.float32)
    transition = torch.zeros((1, 4, 2), dtype=torch.bool)

    terms = factorized_training_loss_terms(
        expert_normalized=expert,
        policy_normalized=policy,
        intent_logits=logits,
        valid_mask=torch.ones((1, 3, 1), dtype=torch.bool),
        norm_stats=_stats(mean=(0.2, 0.0, 0.0, 0.0), std=(0.5, 1.0, 1.0, 1.0)),
        config=cfg,
        transition_mask=transition,
    )

    assert torch.isclose(terms["factorized_magnitude_l1"], torch.tensor(0.0))


def test_idle_ce_pushes_both_direction_logits_down() -> None:
    cfg = _config()
    expert = torch.zeros((1, 3, 4), dtype=torch.float32)
    policy = torch.zeros_like(expert, requires_grad=True)
    logits = torch.zeros((1, 3, 8), dtype=torch.float32, requires_grad=True)

    terms = factorized_training_loss_terms(
        expert_normalized=expert,
        policy_normalized=policy,
        intent_logits=logits,
        valid_mask=torch.ones((1, 3, 1), dtype=torch.bool),
        norm_stats=_stats(),
        config=cfg,
        transition_mask=torch.zeros((1, 4, 2), dtype=torch.bool),
    )
    terms["factorized_loss"].backward()

    assert logits.grad is not None
    assert torch.all(logits.grad > 0.0)


def test_held_prefix_nll_has_target_direction_gradient_and_conflict_metric() -> None:
    cfg = _config()
    expert = torch.zeros((1, 3, 4), dtype=torch.float32)
    policy = torch.zeros_like(expert, requires_grad=True)
    logits = torch.zeros((1, 3, 8), dtype=torch.float32, requires_grad=True)
    transition = torch.zeros((1, 4, 2), dtype=torch.bool)
    transition[0, 0, 0] = True

    terms = factorized_training_loss_terms(
        expert_normalized=expert,
        policy_normalized=policy,
        intent_logits=logits,
        valid_mask=torch.ones((1, 3, 1), dtype=torch.bool),
        norm_stats=_stats(),
        config=cfg,
        transition_mask=transition,
    )

    assert terms["factorized_held_nll"] > 0.0
    assert terms["factorized_held_conflict_count"].item() == 2.0
    assert terms["factorized_held_target_count"].item() == 2.0
    terms["factorized_held_loss"].backward()
    assert logits.grad is not None
    assert logits.grad[0, :, 0].sum() < 0.0  # positive checkpoint logit
    assert logits.grad[0, :, 1].sum() > 0.0  # negative checkpoint logit


def test_query_effort_is_absolute_direct_action_not_absolute_normalized() -> None:
    normalized = torch.tensor([[[0.0, -1.0, 0.5, -0.5]] * 3], dtype=torch.float32)
    logits = torch.zeros((1, 3, 8), dtype=torch.float32)

    probabilities, effort, signed = query_factorized_values(
        policy_normalized=normalized,
        intent_logits=logits,
        norm_stats=_stats(mean=(0.2, 0.1, -0.1, 0.0), std=(0.5, 0.5, 0.2, 0.4)),
    )

    assert probabilities.shape == (1, 3, 4, 3)
    assert torch.allclose(signed[0, 0], torch.tensor([0.2, -0.4, 0.0, -0.2]))
    assert torch.allclose(effort[0, 0], torch.tensor([0.2, 0.4, 0.0, 0.2]))


def test_temporal_aggregator_keeps_zero_effort_sources_and_reset_state() -> None:
    aggregator = FactorizedTemporalAggregator(
        num_queries=2,
        device=torch.device("cpu"),
        max_episode_len=1,
        exponential_k=0.01,
    )
    probabilities = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
    probabilities[..., 1] = 1.0
    zero = torch.zeros((1, 2, 4), dtype=torch.float32)

    first = aggregator.aggregate(
        probabilities=probabilities,
        effort=zero,
        legacy_signed_action=zero,
    )
    second = aggregator.aggregate(
        probabilities=probabilities,
        effort=zero,
        legacy_signed_action=zero,
    )

    assert first.diagnostics["temporal_source_count"] == 1
    assert second.diagnostics["temporal_source_count"] == 2
    assert torch.equal(second.effort, torch.zeros(4))
    aggregator.reset()
    reset = aggregator.aggregate(
        probabilities=probabilities,
        effort=zero,
        legacy_signed_action=zero,
    )
    assert reset.diagnostics["temporal_source_count"] == 1


def test_probability_and_effort_aggregate_before_one_projection() -> None:
    cfg = _config(chunk_size=2)
    aggregator = FactorizedTemporalAggregator(
        num_queries=2,
        device=torch.device("cpu"),
        max_episode_len=2,
        exponential_k=0.01,
    )
    first_prob = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
    first_prob[..., 1] = 1.0
    first_prob[0, 1, 0] = torch.tensor([0.10, 0.20, 0.70])
    second_prob = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
    second_prob[..., 1] = 1.0
    second_prob[0, 0, 0] = torch.tensor([0.20, 0.20, 0.60])
    effort = torch.zeros((1, 2, 4), dtype=torch.float32)
    effort[:, :, 0] = 0.1
    legacy_first = torch.zeros_like(effort)
    legacy_first[0, 1, 0] = 0.1
    legacy_second = torch.zeros_like(effort)
    legacy_second[0, 0, 0] = -0.1
    aggregator.aggregate(
        probabilities=first_prob,
        effort=effort,
        legacy_signed_action=legacy_first,
    )
    aggregated = aggregator.aggregate(
        probabilities=second_prob,
        effort=effort,
        legacy_signed_action=legacy_second,
    )

    action, diagnostics = project_factorized_action(
        probabilities=aggregated.probabilities,
        effort=aggregated.effort,
        config=cfg,
    )

    assert abs(float(aggregated.legacy_signed_action[0])) < 0.01
    assert math.isclose(float(action[0]), 0.52, rel_tol=0.0, abs_tol=1e-6)
    assert diagnostics["selected_class"][0] == "pos"


def test_projection_uses_idle_on_exact_tie_and_asymmetric_move_floors() -> None:
    cfg = _config()
    probabilities = torch.tensor(
        [
            [0.4, 0.2, 0.4],  # exact direction tie -> idle
            [0.7, 0.2, 0.1],  # negative
            [0.1, 0.8, 0.1],  # idle
            [0.1, 0.2, 0.7],  # positive
        ],
        dtype=torch.float32,
    )

    action, diagnostics = project_factorized_action(
        probabilities=probabilities,
        effort=torch.tensor([0.9, 0.1, 0.9, 0.1]),
        config=cfg,
    )

    assert torch.allclose(action, torch.tensor([0.0, -0.47, 0.0, 0.22]))
    assert diagnostics["selection_tie"] == [True, False, False, False]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda cfg: cfg["projection"]["thresholds"]["swing"].update(pos=0.0),
            "strictly positive",
        ),
        (lambda cfg: cfg["temporal"].update(chunk_size=4), "chunk_size"),
        (
            lambda cfg: cfg["classification"].update(class_weights=[1.0, 0.0, 1.0]),
            "class_weights",
        ),
        (lambda cfg: cfg["selection"].update(tie_break="negative"), "tie_break"),
    ],
)
def test_config_rejects_contract_violations(mutator, message: str) -> None:
    raw = _raw_config()
    mutator(raw)

    with pytest.raises(ValueError, match=message):
        resolve_factorized_config(raw, num_queries=3)


def test_projection_rejects_nonfinite_probability() -> None:
    probabilities = torch.full((4, 3), 1.0 / 3.0)
    probabilities[0, 0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        project_factorized_action(
            probabilities=probabilities,
            effort=torch.zeros(4),
            config=_config(),
        )
