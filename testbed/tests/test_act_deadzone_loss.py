from __future__ import annotations

import torch

from testbed.policies.act.adapter import (
    ACTAdapter,
    _masked_action_l1,
    _resolve_deadzone_loss_config,
    _resolve_intent_loss_config,
    _resolve_temporal_release_loss_config,
)
from testbed.policies.act.trainer import ACTTrainer


def _adapter_with_deadzone_loss() -> ACTAdapter:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.norm_stats = {
        "action_mean": torch.zeros(4, dtype=torch.float32).numpy(),
        "action_std": torch.ones(4, dtype=torch.float32).numpy(),
    }
    adapter._deadzone_loss = _resolve_deadzone_loss_config(
        {
            "enabled": True,
            "thresholds": {
                "swing": {"pos": 0.5, "neg": 0.5},
                "boom": {"pos": 0.4, "neg": 0.4},
                "stick": {"pos": 0.3, "neg": 0.3},
                "bucket": {"pos": 0.2, "neg": 0.2},
            },
            "same_dir_promote_weight": 2.0,
            "idle_suppression_weight": 3.0,
            "wrong_effective_weight": 5.0,
            "margin": 0.1,
            "effective_target": "threshold_plus_margin",
            "apply_idle_suppression_when": "expert_ineffective",
        }
    )
    return adapter


def _adapter_with_deadzone_loss_config(extra: dict) -> ACTAdapter:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.norm_stats = {
        "action_mean": torch.zeros(4, dtype=torch.float32).numpy(),
        "action_std": torch.ones(4, dtype=torch.float32).numpy(),
    }
    cfg = {
        "enabled": True,
        "thresholds": {
            "swing": {"pos": 0.5, "neg": 0.5},
            "boom": {"pos": 0.4, "neg": 0.4},
            "stick": {"pos": 0.3, "neg": 0.3},
            "bucket": {"pos": 0.2, "neg": 0.2},
        },
        "same_dir_promote_weight": 1.0,
        "idle_suppression_weight": 0.0,
        "wrong_effective_weight": 0.0,
        "margin": 0.1,
    }
    cfg.update(extra)
    adapter._deadzone_loss = _resolve_deadzone_loss_config(cfg)
    return adapter


def _adapter_with_intent_loss(extra: dict | None = None) -> ACTAdapter:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.norm_stats = {
        "action_mean": torch.zeros(4, dtype=torch.float32).numpy(),
        "action_std": torch.ones(4, dtype=torch.float32).numpy(),
    }
    cfg = {
        "enabled": True,
        "thresholds": {
            "swing": {"pos": 0.5, "neg": 0.5},
            "boom": {"pos": 0.4, "neg": 0.4},
            "stick": {"pos": 0.3, "neg": 0.3},
            "bucket": {"pos": 0.2, "neg": 0.2},
        },
        "weight": 2.0,
    }
    if extra:
        cfg.update(extra)
    adapter._intent_loss = _resolve_intent_loss_config(cfg)
    return adapter


def _adapter_with_window_deadzone_loss() -> ACTAdapter:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.norm_stats = {
        "action_mean": torch.zeros(4, dtype=torch.float32).numpy(),
        "action_std": torch.ones(4, dtype=torch.float32).numpy(),
    }
    adapter._window_deadzone_loss = {
        "enabled": True,
        "same_dir_promote_weight": 2.0,
        "stop_suppression_weight": 3.0,
        "wrong_effective_weight": 5.0,
        "margin": torch.full((4,), 0.1, dtype=torch.float32),
        "pos": torch.tensor([0.5, 0.4, 0.3, 0.2], dtype=torch.float32),
        "neg": torch.tensor([0.5, 0.4, 0.3, 0.2], dtype=torch.float32),
    }
    return adapter


def _adapter_with_temporal_release_loss(extra: dict | None = None) -> ACTAdapter:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.norm_stats = {
        "action_mean": torch.zeros(4, dtype=torch.float32).numpy(),
        "action_std": torch.ones(4, dtype=torch.float32).numpy(),
    }
    cfg = {
        "enabled": True,
        "thresholds": {
            "swing": {"pos": 0.5, "neg": 0.5},
            "boom": {"pos": 0.4, "neg": 0.4},
            "stick": {"pos": 0.3, "neg": 0.3},
            "bucket": {"pos": 0.2, "neg": 0.2},
        },
        "weight": 2.0,
        "release_window_steps": 2,
    }
    if extra:
        cfg.update(extra)
    adapter._temporal_release_loss = _resolve_temporal_release_loss_config(cfg)
    return adapter


class _FakeActModel:
    num_queries = 2

    def __init__(self, policy: torch.Tensor):
        self.policy = policy

    def __call__(self, proprio, image, env_state, actions, is_pad):
        batch = actions.shape[0]
        latent = [
            torch.zeros((batch, 1), dtype=actions.dtype, device=actions.device),
            torch.zeros((batch, 1), dtype=actions.dtype, device=actions.device),
        ]
        return self.policy.to(actions.device), None, latent, None


def _adapter_for_forward_loss(policy: torch.Tensor) -> ACTAdapter:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.device = torch.device("cpu")
    adapter.norm_stats = {
        "action_mean": torch.zeros(4, dtype=torch.float32).numpy(),
        "action_std": torch.ones(4, dtype=torch.float32).numpy(),
    }
    adapter.kl_weight = 0.0
    adapter._model = _FakeActModel(policy)
    adapter._normalize = torch.nn.Identity()
    adapter._deadzone_loss = _resolve_deadzone_loss_config({"enabled": False})
    adapter._intent_loss = _resolve_intent_loss_config({"enabled": False})
    adapter._window_deadzone_loss = {"enabled": False}
    adapter._factorized_action = {"enabled": False, "intent_dim": 0}
    return adapter


def test_forward_loss_uses_action_loss_mask_for_l1_imitation() -> None:
    expert = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [-9.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    policy = torch.tensor(
        [
            [
                [0.5, 0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    adapter = _adapter_for_forward_loss(policy)

    loss = adapter.forward_loss(
        torch.zeros((1, 4), dtype=torch.float32),
        torch.zeros((1, 1, 3, 2, 2), dtype=torch.float32),
        expert,
        torch.zeros((1, 2), dtype=torch.bool),
        action_loss_mask=torch.tensor([[True, False]], dtype=torch.bool),
    )

    assert torch.isclose(loss["l1"], torch.tensor(0.125))


class _RecordingAdapter:
    device = torch.device("cpu")

    def __init__(self):
        self.kwargs = None

    def forward_loss(self, proprio, image, action, is_pad, **kwargs):
        self.kwargs = kwargs
        return {"loss": torch.tensor(0.0)}


def test_trainer_forward_passes_extended_deadzone_batch_masks() -> None:
    adapter = _RecordingAdapter()
    data = {
        "image": torch.zeros((1, 1, 3, 2, 2), dtype=torch.float32),
        "proprio": torch.zeros((1, 4), dtype=torch.float32),
        "action": torch.zeros((1, 2, 4), dtype=torch.float32),
        "is_pad": torch.zeros((1, 2), dtype=torch.bool),
        "deadzone_move_mask": torch.zeros((1, 2, 4, 2), dtype=torch.bool),
        "deadzone_stop_mask": torch.zeros((1, 2), dtype=torch.bool),
        "deadzone_wrong_mask": torch.zeros((1, 2, 4, 2), dtype=torch.bool),
        "action_loss_mask": torch.ones((1, 2), dtype=torch.bool),
        "state_hold_transition_mask": torch.zeros((1, 4, 2), dtype=torch.bool),
    }

    ACTTrainer._forward(data, adapter)

    assert adapter.kwargs is not None
    assert set(adapter.kwargs) == {
        "deadzone_move_mask",
        "deadzone_stop_mask",
        "deadzone_wrong_mask",
        "action_loss_mask",
        "state_hold_transition_mask",
    }


class _CounterfactualRecordingAdapter:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.proprios: list[torch.Tensor] = []

    def forward_loss(self, proprio, image, action, is_pad, **kwargs):
        del image, action, is_pad, kwargs
        self.proprios.append(proprio.detach().clone())
        loss = proprio.sum()
        zero = loss * 0.0
        return {
            "loss": loss,
            "l1": zero + 1.0,
            "intent_loss": zero + 2.0,
            "demo_target_hold_loss": zero + 3.0,
        }


def test_trainer_averages_symmetric_counterfactual_pair_as_one_branch() -> None:
    adapter = _CounterfactualRecordingAdapter()
    data = {
        "image": torch.zeros((2, 1, 3, 2, 2), dtype=torch.float32),
        "proprio": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "action": torch.zeros((2, 2, 4), dtype=torch.float32),
        "is_pad": torch.zeros((2, 2), dtype=torch.bool),
        "execution_feedback_counterfactual_proprio": torch.tensor(
            [
                [[10.0, 20.0], [-10.0, -20.0]],
                [[30.0, 40.0], [-30.0, -40.0]],
            ]
        ),
        "execution_feedback_counterfactual_mask": torch.tensor([True, False]),
        "execution_feedback_counterfactual_loss_weight": torch.ones(2),
    }

    result = ACTTrainer._forward(data, adapter)

    assert len(adapter.proprios) == 2
    assert torch.equal(
        adapter.proprios[1],
        torch.tensor([[10.0, 20.0], [-10.0, -20.0]]),
    )
    # The symmetric pair is one model call whose scalar reduction represents
    # one counterfactual branch; the ordinary causal branch remains present.
    assert torch.isclose(
        result["execution_feedback_counterfactual_loss"], torch.tensor(0.0)
    )
    assert torch.isclose(result["loss"], torch.tensor(10.0))
    assert result["execution_feedback_counterfactual_samples"].item() == 1.0


def test_trainer_emits_zero_counterfactual_metrics_without_active_rows() -> None:
    adapter = _CounterfactualRecordingAdapter()
    data = {
        "image": torch.zeros((1, 1, 3, 2, 2), dtype=torch.float32),
        "proprio": torch.tensor([[1.0, 2.0]]),
        "action": torch.zeros((1, 2, 4), dtype=torch.float32),
        "is_pad": torch.zeros((1, 2), dtype=torch.bool),
        "execution_feedback_counterfactual_proprio": torch.zeros((1, 2, 2)),
        "execution_feedback_counterfactual_mask": torch.tensor([False]),
        "execution_feedback_counterfactual_loss_weight": torch.ones(1),
    }

    result = ACTTrainer._forward(data, adapter)

    assert len(adapter.proprios) == 1
    assert result["execution_feedback_counterfactual_samples"].item() == 0.0
    assert result["execution_feedback_counterfactual_loss"].item() == 0.0
    assert result["loss"].item() == 3.0


def test_window_deadzone_loss_uses_explicit_move_stop_and_wrong_masks() -> None:
    adapter = _adapter_with_window_deadzone_loss()
    expert = torch.tensor(
        [
            [
                [0.60, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
                [0.60, 0.00, 0.00, 0.00],
            ]
        ],
        dtype=torch.float32,
    )
    policy = torch.tensor(
        [
            [
                [0.40, 0.00, 0.00, 0.00],
                [0.00, 0.45, 0.00, 0.00],
                [0.70, 0.00, 0.00, 0.30],
            ]
        ],
        dtype=torch.float32,
    )
    move_mask = torch.zeros((1, 3, 4, 2), dtype=torch.bool)
    move_mask[0, 0, 0, 0] = True
    move_mask[0, 2, 0, 0] = True
    stop_mask = torch.tensor([[False, True, False]], dtype=torch.bool)
    wrong_mask = torch.zeros((1, 3, 4, 2), dtype=torch.bool)
    wrong_mask[0, 2, 3, 0] = True

    terms = adapter._window_deadzone_loss_terms(
        expert_normalized=expert,
        policy_normalized=policy,
        valid_mask=torch.ones_like(expert, dtype=torch.bool),
        move_mask=move_mask,
        stop_mask=stop_mask,
        wrong_mask=wrong_mask,
    )

    assert torch.isclose(terms["window_deadzone_same_dir_loss"], torch.tensor(0.1))
    assert torch.isclose(terms["window_deadzone_stop_loss"], torch.tensor(0.05))
    assert torch.isclose(terms["window_deadzone_wrong_loss"], torch.tensor(0.1))
    assert torch.isclose(terms["window_deadzone_loss"], torch.tensor(0.85))


def test_temporal_release_loss_penalizes_same_direction_persistence_after_expert_release() -> (
    None
):
    adapter = _adapter_with_temporal_release_loss()
    expert = torch.tensor(
        [
            [
                [0.60, 0.00, 0.00, 0.00],
                [0.60, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
            ]
        ],
        dtype=torch.float32,
    )
    policy = torch.tensor(
        [
            [
                [0.70, 0.00, 0.00, 0.00],
                [0.70, 0.00, 0.00, 0.00],
                [0.70, 0.00, 0.00, 0.00],
                [0.60, 0.00, 0.00, 0.00],
                [0.70, 0.00, 0.00, 0.00],
            ]
        ],
        dtype=torch.float32,
    )

    terms = adapter._temporal_release_loss_terms(
        expert_normalized=expert,
        policy_normalized=policy,
        valid_mask=torch.ones_like(expert, dtype=torch.bool),
    )

    assert torch.isclose(terms["temporal_release_pos_loss"], torch.tensor(0.15))
    assert torch.isclose(terms["temporal_release_neg_loss"], torch.tensor(0.0))
    assert torch.isclose(terms["temporal_release_loss"], torch.tensor(0.30))


def test_temporal_release_loss_penalizes_negative_direction_release() -> None:
    adapter = _adapter_with_temporal_release_loss(
        {"weight": 1.0, "release_window_steps": 1}
    )
    expert = torch.tensor(
        [[[-0.60, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]]],
        dtype=torch.float32,
    )
    policy = torch.tensor(
        [[[-0.70, 0.00, 0.00, 0.00], [-0.80, 0.00, 0.00, 0.00]]],
        dtype=torch.float32,
    )

    terms = adapter._temporal_release_loss_terms(
        expert_normalized=expert,
        policy_normalized=policy,
        valid_mask=torch.ones_like(expert, dtype=torch.bool),
    )

    assert torch.isclose(terms["temporal_release_pos_loss"], torch.tensor(0.0))
    assert torch.isclose(terms["temporal_release_neg_loss"], torch.tensor(0.30))
    assert torch.isclose(terms["temporal_release_loss"], torch.tensor(0.30))


def test_masked_action_l1_ignores_action_loss_masked_steps() -> None:
    expert = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [-9.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    policy = torch.tensor(
        [
            [
                [0.5, 0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.ones_like(expert, dtype=torch.bool)
    action_loss_mask = torch.tensor([[True, False]], dtype=torch.bool)

    loss = _masked_action_l1(
        expert=expert,
        policy=policy,
        valid_mask=valid_mask,
        action_loss_mask=action_loss_mask,
    )

    assert torch.isclose(loss, torch.tensor(0.125))


def test_deadzone_loss_promotes_expert_motion_and_suppresses_idle_motion() -> None:
    adapter = _adapter_with_deadzone_loss()

    expert = torch.tensor(
        [
            [
                [0.60, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
            ]
        ],
        dtype=torch.float32,
    )
    policy = torch.tensor(
        [
            [
                [0.40, 0.00, 0.00, 0.00],
                [0.00, 0.45, 0.00, 0.00],
            ]
        ],
        dtype=torch.float32,
    )

    terms = adapter._deadzone_loss_terms(
        expert_normalized=expert,
        policy_normalized=policy,
        valid_mask=torch.ones_like(expert, dtype=torch.bool),
    )

    assert torch.isclose(terms["deadzone_same_dir_loss"], torch.tensor(0.2))
    assert torch.isclose(terms["deadzone_idle_loss"], torch.tensor(0.05))
    assert torch.isclose(terms["deadzone_wrong_loss"], torch.tensor(0.0))
    assert torch.isclose(terms["deadzone_loss"], torch.tensor(0.55))


def test_deadzone_loss_penalizes_wrong_direction_separately_from_idle() -> None:
    adapter = _adapter_with_deadzone_loss()

    expert = torch.tensor([[[0.60, 0.00, 0.00, 0.00]]], dtype=torch.float32)
    policy = torch.tensor([[[-0.70, 0.00, 0.00, 0.00]]], dtype=torch.float32)

    terms = adapter._deadzone_loss_terms(
        expert_normalized=expert,
        policy_normalized=policy,
        valid_mask=torch.ones_like(expert, dtype=torch.bool),
    )

    assert terms["deadzone_same_dir_loss"] > 0
    assert torch.isclose(terms["deadzone_idle_loss"], torch.tensor(0.0))
    assert torch.isclose(terms["deadzone_wrong_loss"], torch.tensor(0.2))


def test_deadzone_loss_can_focus_same_direction_promotion_on_effective_transitions() -> (
    None
):
    adapter = _adapter_with_deadzone_loss_config(
        {
            "same_dir_window": "expert_transition_window",
            "same_dir_window_steps": 1,
        }
    )

    expert = torch.tensor(
        [
            [
                [0.60, 0.00, 0.00, 0.00],
                [0.60, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
                [0.60, 0.00, 0.00, 0.00],
                [0.60, 0.00, 0.00, 0.00],
            ]
        ],
        dtype=torch.float32,
    )
    policy = torch.tensor(
        [
            [
                [0.40, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
                [0.40, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
            ]
        ],
        dtype=torch.float32,
    )

    terms = adapter._deadzone_loss_terms(
        expert_normalized=expert,
        policy_normalized=policy,
        valid_mask=torch.ones_like(expert, dtype=torch.bool),
    )

    assert torch.isclose(terms["deadzone_same_dir_loss"], torch.tensor(0.2))


def test_deadzone_loss_can_penalize_idle_crossing_frequency() -> None:
    sparse = _adapter_with_deadzone_loss_config(
        {
            "same_dir_promote_weight": 0.0,
            "idle_suppression_weight": 1.0,
            "wrong_effective_weight": 0.0,
            "idle_denominator": "all_idle_axes",
        }
    )
    frequent = _adapter_with_deadzone_loss_config(
        {
            "same_dir_promote_weight": 0.0,
            "idle_suppression_weight": 1.0,
            "wrong_effective_weight": 0.0,
            "idle_denominator": "all_idle_axes",
        }
    )

    expert = torch.zeros((1, 10, 4), dtype=torch.float32)
    sparse_policy = torch.zeros((1, 10, 4), dtype=torch.float32)
    frequent_policy = torch.zeros((1, 10, 4), dtype=torch.float32)
    sparse_policy[:, 0, 3] = 0.30
    frequent_policy[:, :5, 3] = 0.30
    valid = torch.ones_like(expert, dtype=torch.bool)

    sparse_terms = sparse._deadzone_loss_terms(
        expert_normalized=expert,
        policy_normalized=sparse_policy,
        valid_mask=valid,
    )
    frequent_terms = frequent._deadzone_loss_terms(
        expert_normalized=expert,
        policy_normalized=frequent_policy,
        valid_mask=valid,
    )

    assert (
        frequent_terms["deadzone_idle_loss"] > sparse_terms["deadzone_idle_loss"] * 4.0
    )


def test_deadzone_loss_can_penalize_wrong_crossing_frequency() -> None:
    sparse = _adapter_with_deadzone_loss_config(
        {
            "same_dir_promote_weight": 0.0,
            "idle_suppression_weight": 0.0,
            "wrong_effective_weight": 1.0,
            "wrong_denominator": "all_wrong_candidate_axes",
        }
    )
    frequent = _adapter_with_deadzone_loss_config(
        {
            "same_dir_promote_weight": 0.0,
            "idle_suppression_weight": 0.0,
            "wrong_effective_weight": 1.0,
            "wrong_denominator": "all_wrong_candidate_axes",
        }
    )

    expert = torch.zeros((1, 10, 4), dtype=torch.float32)
    expert[:, :, 0] = 0.60
    sparse_policy = torch.zeros((1, 10, 4), dtype=torch.float32)
    frequent_policy = torch.zeros((1, 10, 4), dtype=torch.float32)
    sparse_policy[:, 0, 3] = 0.30
    frequent_policy[:, :5, 3] = 0.30
    valid = torch.ones_like(expert, dtype=torch.bool)

    sparse_terms = sparse._deadzone_loss_terms(
        expert_normalized=expert,
        policy_normalized=sparse_policy,
        valid_mask=valid,
    )
    frequent_terms = frequent._deadzone_loss_terms(
        expert_normalized=expert,
        policy_normalized=frequent_policy,
        valid_mask=valid,
    )

    assert (
        frequent_terms["deadzone_wrong_loss"]
        > sparse_terms["deadzone_wrong_loss"] * 4.0
    )


def test_intent_loss_derives_axis_direction_targets_from_expert_deadzones() -> None:
    adapter = _adapter_with_intent_loss()

    expert = torch.tensor(
        [
            [
                [0.60, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
            ]
        ],
        dtype=torch.float32,
    )
    intent_logits = torch.zeros((1, 2, 8), dtype=torch.float32)

    terms = adapter._intent_loss_terms(
        expert_normalized=expert,
        intent_logits=intent_logits,
        valid_mask=torch.ones_like(expert, dtype=torch.bool),
    )

    assert torch.isclose(terms["intent_axis_dir_loss"], torch.tensor(0.69314718))
    assert torch.isclose(terms["intent_loss"], torch.tensor(1.38629436))


def test_intent_loss_uses_axis_then_direction_order() -> None:
    adapter = _adapter_with_intent_loss({"weight": 1.0})

    expert = torch.tensor([[[0.00, -0.50, 0.00, 0.00]]], dtype=torch.float32)
    correct_axis_then_dir_logits = torch.tensor(
        [[[-8.0, -8.0, -8.0, 8.0, -8.0, -8.0, -8.0, -8.0]]],
        dtype=torch.float32,
    )

    terms = adapter._intent_loss_terms(
        expert_normalized=expert,
        intent_logits=correct_axis_then_dir_logits,
        valid_mask=torch.ones_like(expert, dtype=torch.bool),
    )

    assert terms["intent_axis_dir_loss"] < 0.001


def test_intent_loss_can_weight_sparse_positive_direction_labels() -> None:
    unweighted = _adapter_with_intent_loss({"weight": 1.0, "positive_weight": 1.0})
    weighted = _adapter_with_intent_loss({"weight": 1.0, "positive_weight": 8.0})

    expert = torch.tensor([[[0.60, 0.00, 0.00, 0.00]]], dtype=torch.float32)
    all_negative_logits = torch.full((1, 1, 8), -4.0, dtype=torch.float32)

    unweighted_terms = unweighted._intent_loss_terms(
        expert_normalized=expert,
        intent_logits=all_negative_logits,
        valid_mask=torch.ones_like(expert, dtype=torch.bool),
    )
    weighted_terms = weighted._intent_loss_terms(
        expert_normalized=expert,
        intent_logits=all_negative_logits,
        valid_mask=torch.ones_like(expert, dtype=torch.bool),
    )

    assert (
        weighted_terms["intent_axis_dir_loss"]
        > unweighted_terms["intent_axis_dir_loss"] * 4.0
    )


def test_intent_loss_current_steps_masks_future_queries() -> None:
    all_queries = _adapter_with_intent_loss({"weight": 1.0})
    current_only = _adapter_with_intent_loss({"weight": 1.0, "current_steps": 1})
    expert = torch.tensor(
        [[[0.00, 0.00, 0.00, 0.00], [0.60, 0.00, 0.00, 0.00]]],
        dtype=torch.float32,
    )
    logits = torch.full((1, 2, 8), -4.0, dtype=torch.float32)
    all_terms = all_queries._intent_loss_terms(
        expert_normalized=expert,
        intent_logits=logits,
        valid_mask=torch.ones_like(expert, dtype=torch.bool),
    )
    current_terms = current_only._intent_loss_terms(
        expert_normalized=expert,
        intent_logits=logits,
        valid_mask=torch.ones_like(expert, dtype=torch.bool),
    )

    assert current_terms["intent_axis_dir_loss"] < all_terms["intent_axis_dir_loss"]


def test_disabled_intent_loss_is_zero_without_logits() -> None:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter._intent_loss = _resolve_intent_loss_config({"enabled": False})

    terms = adapter._intent_loss_terms(
        expert_normalized=torch.zeros((1, 1, 4), dtype=torch.float32),
        intent_logits=None,
        valid_mask=torch.ones((1, 1, 4), dtype=torch.bool),
    )

    assert torch.isclose(terms["intent_axis_dir_loss"], torch.tensor(0.0))
    assert torch.isclose(terms["intent_loss"], torch.tensor(0.0))
