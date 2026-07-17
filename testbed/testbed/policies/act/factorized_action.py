"""Factorized ACT direction/idle intent, effort, aggregation, and projection.

This module is an opt-in experimental owner.  It keeps the discontinuous
mechanical-deadzone projection out of the generic ACT adapter and makes the
probability/effort aggregation contract directly unit-testable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from testbed.policies.deadzone_eval import load_deadzone_thresholds

AXIS_NAMES = ("swing", "boom", "stick", "bucket")
TRI_STATE_NAMES = ("neg", "idle", "pos")
NEGATIVE_CLASS = 0
IDLE_CLASS = 1
POSITIVE_CLASS = 2


def resolve_factorized_config(
    raw: Any,
    *,
    num_queries: int,
) -> dict[str, Any]:
    """Validate and resolve the opt-in factorized inference/training contract."""

    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False, "intent_dim": 0}

    intent_dim = int(cfg.get("intent_dim", 8))
    if intent_dim != len(AXIS_NAMES) * 2:
        raise ValueError("factorized_intent_effort.intent_dim must be 8")
    if str(cfg.get("intent_layout", "axis_major_pos_neg")) != "axis_major_pos_neg":
        raise ValueError(
            "factorized_intent_effort.intent_layout must be 'axis_major_pos_neg'"
        )
    if list(cfg.get("tri_state_order", TRI_STATE_NAMES)) != list(TRI_STATE_NAMES):
        raise ValueError(
            "factorized_intent_effort.tri_state_order must be ['neg', 'idle', 'pos']"
        )
    idle_logit = _finite_float(cfg.get("idle_logit", 0.0), name="idle_logit")
    if idle_logit != 0.0:
        raise ValueError("factorized_intent_effort.idle_logit must be 0.0")

    classification_raw = dict(cfg.get("classification", {}) or {})
    classification_weight = _non_negative_float(
        classification_raw.get("weight", 0.05),
        name="classification.weight",
    )
    if (
        str(classification_raw.get("label_domain", "direct_policy_output"))
        != "direct_policy_output"
    ):
        raise ValueError(
            "factorized_intent_effort.classification.label_domain must be "
            "'direct_policy_output'"
        )
    class_weights = _resolve_class_weights(
        classification_raw.get("class_weights", [8.0, 1.0, 8.0])
    )

    effort_raw = dict(cfg.get("effort", {}) or {})
    if str(effort_raw.get("transform", "abs_after_unnormalize")) != (
        "abs_after_unnormalize"
    ):
        raise ValueError(
            "factorized_intent_effort.effort.transform must be 'abs_after_unnormalize'"
        )
    if str(effort_raw.get("loss", "direct_magnitude_l1")) != "direct_magnitude_l1":
        raise ValueError(
            "factorized_intent_effort.effort.loss must be 'direct_magnitude_l1'"
        )
    effort_weight = _non_negative_float(
        effort_raw.get("weight", 1.0),
        name="effort.weight",
    )

    temporal_raw = dict(cfg.get("temporal", {}) or {})
    if not bool(temporal_raw.get("enabled", True)):
        raise ValueError("factorized_intent_effort.temporal.enabled must be true")
    chunk_size = int(temporal_raw.get("chunk_size", num_queries))
    if chunk_size != int(num_queries):
        raise ValueError(
            "factorized_intent_effort.temporal.chunk_size must equal ACT num_queries"
        )
    exponential_k = _non_negative_float(
        temporal_raw.get("exponential_k", 0.01),
        name="temporal.exponential_k",
    )
    if str(temporal_raw.get("source_order", "oldest_to_newest")) != (
        "oldest_to_newest"
    ):
        raise ValueError(
            "factorized_intent_effort.temporal.source_order must be 'oldest_to_newest'"
        )
    if str(temporal_raw.get("aggregate", "probabilities_and_effort")) != (
        "probabilities_and_effort"
    ):
        raise ValueError(
            "factorized_intent_effort.temporal.aggregate must be "
            "'probabilities_and_effort'"
        )

    held_raw = dict(cfg.get("held_prefix", {}) or {})
    held_enabled = bool(held_raw.get("enabled", True))
    held_weight = _non_negative_float(
        held_raw.get("weight", 0.1),
        name="held_prefix.weight",
    )
    hold_horizon_steps = int(held_raw.get("hold_horizon_steps", chunk_size))
    if not 0 < hold_horizon_steps <= chunk_size:
        raise ValueError(
            "factorized_intent_effort.held_prefix.hold_horizon_steps must be in "
            "[1, chunk_size]"
        )
    target_delays = tuple(
        int(value)
        for value in held_raw.get(
            "target_delays",
            [hold_horizon_steps - 2, hold_horizon_steps - 1],
        )
    )
    if held_enabled and not target_delays:
        raise ValueError(
            "factorized_intent_effort.held_prefix.target_delays must not be empty"
        )
    if len(set(target_delays)) != len(target_delays) or any(
        delay < 0 or delay >= hold_horizon_steps for delay in target_delays
    ):
        raise ValueError(
            "factorized_intent_effort.held_prefix.target_delays must be unique and "
            "inside the held horizon"
        )

    selection_raw = dict(cfg.get("selection", {}) or {})
    if str(selection_raw.get("mode", "strict_argmax")) != "strict_argmax":
        raise ValueError(
            "factorized_intent_effort.selection.mode must be 'strict_argmax'"
        )
    if str(selection_raw.get("tie_break", "idle")) != "idle":
        raise ValueError("factorized_intent_effort.selection.tie_break must be 'idle'")
    if str(selection_raw.get("nonfinite", "error")) != "error":
        raise ValueError("factorized_intent_effort.selection.nonfinite must be 'error'")

    projection_raw = dict(cfg.get("projection", {}) or {})
    thresholds, threshold_path = _resolve_thresholds(projection_raw)
    margin = _non_negative_float(
        projection_raw.get("margin", 0.02), name="projection.margin"
    )
    clip = _finite_float(projection_raw.get("clip", 1.0), name="projection.clip")
    if clip <= 0.0:
        raise ValueError("factorized_intent_effort.projection.clip must be positive")
    pos = torch.as_tensor(
        [thresholds[axis]["pos"] for axis in AXIS_NAMES], dtype=torch.float32
    )
    neg = torch.as_tensor(
        [thresholds[axis]["neg"] for axis in AXIS_NAMES], dtype=torch.float32
    )
    if torch.any(pos + margin > clip) or torch.any(neg + margin > clip):
        raise ValueError(
            "factorized_intent_effort projection floor must not exceed clip"
        )
    action_scale = _axis_values(
        projection_raw.get("action_scale", 1.0),
        name="projection.action_scale",
    )
    if action_scale != [1.0] * len(AXIS_NAMES):
        raise ValueError(
            "factorized_intent_effort.projection.action_scale must be identity"
        )

    return {
        "enabled": True,
        "intent_dim": intent_dim,
        "classification_weight": classification_weight,
        "class_weights": class_weights,
        "effort_weight": effort_weight,
        "chunk_size": chunk_size,
        "exponential_k": exponential_k,
        "held_enabled": held_enabled,
        "held_weight": held_weight,
        "hold_horizon_steps": hold_horizon_steps,
        "target_delays": target_delays,
        "pos": pos,
        "neg": neg,
        "margin": margin,
        "clip": clip,
        "threshold_json": threshold_path,
        "threshold_sha256": (
            _sha256_file(Path(threshold_path)) if threshold_path is not None else None
        ),
        "selection_rule_version": "strict_argmax_idle_on_tie_v1",
        "output_domain": "direct_policy_output",
    }


def intent_logits_to_tri_logits(intent_logits: torch.Tensor) -> torch.Tensor:
    """Map checkpoint-compatible axis-major ``[pos, neg]`` logits to tri-state."""

    if intent_logits.ndim < 1 or intent_logits.shape[-1] != len(AXIS_NAMES) * 2:
        raise ValueError(
            "intent logits must end in 8 axis-major [pos, neg] values, got "
            f"{tuple(intent_logits.shape)}"
        )
    paired = intent_logits.reshape(*intent_logits.shape[:-1], len(AXIS_NAMES), 2)
    positive = paired[..., 0]
    negative = paired[..., 1]
    idle = torch.zeros_like(positive)
    tri_logits = torch.stack([negative, idle, positive], dim=-1)
    if not torch.isfinite(tri_logits).all():
        raise ValueError("factorized intent logits contain non-finite values")
    return tri_logits


def direct_tri_state_labels(
    direct_action: torch.Tensor,
    *,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> torch.Tensor:
    """Return per-axis labels in ``[neg=0, idle=1, pos=2]`` order."""

    if direct_action.shape[-1] != len(AXIS_NAMES):
        raise ValueError(f"direct_action must end in 4 axes, got {direct_action.shape}")
    pos = pos.to(device=direct_action.device, dtype=direct_action.dtype)
    neg = neg.to(device=direct_action.device, dtype=direct_action.dtype)
    labels = torch.full_like(direct_action, IDLE_CLASS, dtype=torch.long)
    labels = torch.where(direct_action <= -neg, NEGATIVE_CLASS, labels)
    labels = torch.where(direct_action >= pos, POSITIVE_CLASS, labels)
    return labels


def factorized_training_loss_terms(
    *,
    expert_normalized: torch.Tensor,
    policy_normalized: torch.Tensor,
    intent_logits: torch.Tensor | None,
    valid_mask: torch.Tensor,
    norm_stats: Mapping[str, Any],
    config: Mapping[str, Any],
    transition_mask: torch.Tensor | None,
    action_loss_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute mutually exclusive direction, direct magnitude, and held NLL."""

    zero = policy_normalized.new_zeros(())
    if not bool(config.get("enabled", False)):
        return {
            "factorized_magnitude_l1": zero,
            "factorized_effort_loss": zero,
            "factorized_class_nll": zero,
            "factorized_class_loss": zero,
            "factorized_held_nll": zero,
            "factorized_held_loss": zero,
            "factorized_held_conflict_count": zero,
            "factorized_held_target_count": zero,
            "factorized_loss": zero,
        }
    if intent_logits is None:
        raise ValueError("factorized_intent_effort requires model intent logits")
    if intent_logits.shape[:-1] != expert_normalized.shape[:-1]:
        raise ValueError(
            "factorized intent logits must match action batch/chunk shape, got "
            f"{tuple(intent_logits.shape)} versus {tuple(expert_normalized.shape)}"
        )

    action_mean, action_std = _norm_tensors(norm_stats, reference=policy_normalized)
    expert_direct = expert_normalized * action_std + action_mean
    policy_direct = policy_normalized * action_std + action_mean
    valid = valid_mask.to(dtype=torch.bool).expand_as(policy_direct)
    if action_loss_mask is not None:
        valid = valid & action_loss_mask.to(
            device=valid.device, dtype=torch.bool
        ).unsqueeze(-1)
    magnitude_error = (
        torch.abs(torch.abs(policy_direct) - torch.abs(expert_direct)) / action_std
    )
    valid_count = valid.to(policy_direct.dtype).sum().clamp_min(1.0)
    magnitude_l1 = (magnitude_error * valid.to(policy_direct.dtype)).sum() / valid_count

    tri_logits = intent_logits_to_tri_logits(intent_logits)
    tri_probabilities = torch.softmax(tri_logits, dim=-1)
    if not torch.isfinite(tri_probabilities).all():
        raise ValueError("factorized tri-state probabilities contain non-finite values")
    pos = config["pos"].to(device=expert_direct.device, dtype=expert_direct.dtype)
    neg = config["neg"].to(device=expert_direct.device, dtype=expert_direct.dtype)
    labels = direct_tri_state_labels(expert_direct, pos=pos, neg=neg)
    log_probabilities = torch.log_softmax(tri_logits, dim=-1)
    target_nll = -torch.gather(
        log_probabilities, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    label_valid = valid_mask.to(dtype=torch.bool).expand_as(expert_direct)
    class_weights = config["class_weights"].to(
        device=expert_direct.device, dtype=expert_direct.dtype
    )
    target_weights = torch.gather(
        class_weights.view(1, 1, len(AXIS_NAMES), 3).expand(*labels.shape, 3),
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    weighted_valid = target_weights * label_valid.to(expert_direct.dtype)
    class_nll = (target_nll * weighted_valid).sum() / weighted_valid.sum().clamp_min(
        1.0
    )
    class_loss = float(config["classification_weight"]) * class_nll

    held_terms = _held_prefix_loss_terms(
        tri_probabilities=tri_probabilities,
        labels=labels,
        transition_mask=transition_mask,
        config=config,
    )
    effort_loss = float(config["effort_weight"]) * magnitude_l1
    total = effort_loss + class_loss + held_terms["factorized_held_loss"]
    return {
        "factorized_magnitude_l1": magnitude_l1,
        "factorized_effort_loss": effort_loss,
        "factorized_class_nll": class_nll,
        "factorized_class_loss": class_loss,
        **held_terms,
        "factorized_loss": total,
    }


def held_temporal_prefix_values(
    values: torch.Tensor,
    *,
    hold_horizon_steps: int,
    exponential_k: float,
) -> torch.Tensor:
    """Replay ACT oldest-to-newest aggregation for one observation held fixed."""

    if values.ndim < 3:
        raise ValueError(
            "held temporal values must have shape (B, C, ...), got "
            f"{tuple(values.shape)}"
        )
    horizon = int(hold_horizon_steps)
    if not 0 < horizon <= int(values.shape[1]):
        raise ValueError(
            "hold_horizon_steps must be in [1, chunk length], got "
            f"{horizon} for {values.shape[1]}"
        )
    prefixes: list[torch.Tensor] = []
    trailing_dims = [1] * (values.ndim - 2)
    for delay in range(horizon):
        current = torch.flip(values[:, : delay + 1], dims=(1,))
        weights = torch.exp(
            -float(exponential_k)
            * torch.arange(delay + 1, dtype=values.dtype, device=values.device)
        )
        weights = weights / weights.sum()
        prefixes.append(
            (current * weights.view(1, delay + 1, *trailing_dims)).sum(dim=1)
        )
    return torch.stack(prefixes, dim=1)


def query_factorized_values(
    *,
    policy_normalized: torch.Tensor,
    intent_logits: torch.Tensor,
    norm_stats: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-query probabilities, direct magnitude, and signed direct action."""

    action_mean, action_std = _norm_tensors(norm_stats, reference=policy_normalized)
    direct_action = policy_normalized * action_std + action_mean
    tri_probabilities = torch.softmax(
        intent_logits_to_tri_logits(intent_logits), dim=-1
    )
    effort = torch.abs(direct_action)
    if not torch.isfinite(tri_probabilities).all() or not torch.isfinite(effort).all():
        raise ValueError("factorized query values contain non-finite values")
    return tri_probabilities, effort, direct_action


@dataclass(frozen=True)
class FactorizedAggregate:
    probabilities: torch.Tensor
    effort: torch.Tensor
    legacy_signed_action: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class FactorizedTemporalState:
    """Device-local no-alias state for one factorized ACT branch point."""

    t: int
    values: torch.Tensor | None
    occupied: torch.Tensor | None
    weight_cache: Mapping[int, torch.Tensor]


class FactorizedTemporalAggregator:
    """Stateful ACT aggregator with an explicit occupancy mask."""

    def __init__(
        self,
        *,
        num_queries: int,
        device: torch.device,
        max_episode_len: int,
        exponential_k: float,
    ) -> None:
        self.num_queries = int(num_queries)
        self.device = device
        self.max_episode_len = int(max_episode_len)
        self.exponential_k = float(exponential_k)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self._values: torch.Tensor | None = None
        self._occupied: torch.Tensor | None = None
        self._weight_cache: dict[int, torch.Tensor] = {}

    def snapshot_state(self) -> FactorizedTemporalState:
        return FactorizedTemporalState(
            t=int(self.t),
            values=None if self._values is None else self._values.detach().clone(),
            occupied=(
                None if self._occupied is None else self._occupied.detach().clone()
            ),
            weight_cache={
                int(key): value.detach().clone()
                for key, value in self._weight_cache.items()
            },
        )

    def restore_state(self, state: FactorizedTemporalState) -> None:
        if not isinstance(state, FactorizedTemporalState):
            raise TypeError("state must be FactorizedTemporalState")
        if int(state.t) < 0:
            raise ValueError("factorized temporal state t must be non-negative")
        self.t = int(state.t)
        self._values = (
            None
            if state.values is None
            else state.values.detach().to(self.device).clone()
        )
        self._occupied = (
            None
            if state.occupied is None
            else state.occupied.detach().to(self.device).clone()
        )
        self._weight_cache = {
            int(key): value.detach().to(self.device).clone()
            for key, value in state.weight_cache.items()
        }

    def aggregate(
        self,
        *,
        probabilities: torch.Tensor,
        effort: torch.Tensor,
        legacy_signed_action: torch.Tensor,
    ) -> FactorizedAggregate:
        if probabilities.shape != (1, self.num_queries, len(AXIS_NAMES), 3):
            raise ValueError(
                "factorized probabilities must have shape "
                f"(1, {self.num_queries}, 4, 3), got {tuple(probabilities.shape)}"
            )
        expected_action_shape = (1, self.num_queries, len(AXIS_NAMES))
        if effort.shape != expected_action_shape or legacy_signed_action.shape != (
            expected_action_shape
        ):
            raise ValueError(
                "factorized effort and signed action must have shape "
                f"{expected_action_shape}"
            )
        values = torch.cat(
            [
                probabilities.reshape(1, self.num_queries, -1),
                effort,
                legacy_signed_action,
            ],
            dim=-1,
        )
        if not torch.isfinite(values).all():
            raise ValueError("factorized temporal input contains non-finite values")
        self._ensure_capacity()
        assert self._values is not None
        assert self._occupied is not None
        self._values[self.t, self.t : self.t + self.num_queries] = values[0]
        self._occupied[self.t, self.t : self.t + self.num_queries] = True

        populated = self._occupied[: self.t + 1, self.t]
        current = self._values[: self.t + 1, self.t][populated]
        source_rows = torch.arange(self.t + 1, device=self.device)[populated]
        source_count = int(current.shape[0])
        if source_count <= 0:
            raise RuntimeError(
                "factorized temporal aggregation has no populated source"
            )
        weights = self._weight_cache.get(source_count)
        if weights is None:
            weights = torch.exp(
                -self.exponential_k
                * torch.arange(source_count, dtype=current.dtype, device=self.device)
            )
            weights = weights / weights.sum()
            self._weight_cache[source_count] = weights
        aggregated = (current * weights.unsqueeze(-1)).sum(dim=0)
        probabilities_bar = aggregated[:12].reshape(len(AXIS_NAMES), 3)
        effort_bar = aggregated[12:16]
        legacy_bar = aggregated[16:20]
        diagnostics = {
            "temporal_source_count": source_count,
            "temporal_source_rows": [int(value) for value in source_rows.cpu()],
            "temporal_query_ages": [
                int(self.t - int(value)) for value in source_rows.cpu()
            ],
            "temporal_weights": [float(value) for value in weights.cpu()],
        }
        self.t += 1
        return FactorizedAggregate(
            probabilities=probabilities_bar,
            effort=effort_bar,
            legacy_signed_action=legacy_bar,
            diagnostics=diagnostics,
        )

    def _ensure_capacity(self) -> None:
        required_rows = self.t + 1
        required_cols = self.t + self.num_queries
        if self._values is None or self._occupied is None:
            rows = max(self.max_episode_len, required_cols)
            self._values = torch.zeros(
                (rows, rows + self.num_queries, 20),
                dtype=torch.float32,
                device=self.device,
            )
            self._occupied = torch.zeros(
                (rows, rows + self.num_queries),
                dtype=torch.bool,
                device=self.device,
            )
            return
        if (
            required_rows <= self._values.shape[0]
            and required_cols <= self._values.shape[1]
        ):
            return
        old_values = self._values
        old_occupied = self._occupied
        rows = max(required_rows, old_values.shape[0] * 2)
        self._values = torch.zeros(
            (rows, rows + self.num_queries, 20),
            dtype=old_values.dtype,
            device=self.device,
        )
        self._occupied = torch.zeros(
            (rows, rows + self.num_queries),
            dtype=torch.bool,
            device=self.device,
        )
        self._values[: old_values.shape[0], : old_values.shape[1]] = old_values
        self._occupied[: old_occupied.shape[0], : old_occupied.shape[1]] = old_occupied


def project_factorized_action(
    *,
    probabilities: torch.Tensor,
    effort: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select one tri-state per axis and apply one deadzone-aware projection."""

    if probabilities.shape != (len(AXIS_NAMES), 3):
        raise ValueError(
            f"probabilities must have shape (4, 3), got {tuple(probabilities.shape)}"
        )
    if effort.shape != (len(AXIS_NAMES),):
        raise ValueError(f"effort must have shape (4,), got {tuple(effort.shape)}")
    if not torch.isfinite(probabilities).all() or not torch.isfinite(effort).all():
        raise ValueError("factorized projection input contains non-finite values")
    if torch.any(effort < 0.0):
        raise ValueError("factorized effort must be non-negative")

    maxima = probabilities.max(dim=-1).values
    winner_mask = probabilities == maxima.unsqueeze(-1)
    tie = winner_mask.sum(dim=-1) != 1
    selected = probabilities.argmax(dim=-1)
    selected = torch.where(
        tie,
        torch.full_like(selected, IDLE_CLASS),
        selected,
    )
    sorted_probabilities = torch.sort(probabilities, dim=-1, descending=True).values
    winner_margin = sorted_probabilities[:, 0] - sorted_probabilities[:, 1]

    pos = config["pos"].to(device=effort.device, dtype=effort.dtype)
    neg = config["neg"].to(device=effort.device, dtype=effort.dtype)
    margin = float(config["margin"])
    clip = float(config["clip"])
    floors = torch.where(selected == POSITIVE_CLASS, pos + margin, neg + margin)
    projected_magnitude = torch.minimum(
        torch.clamp(torch.maximum(effort, floors), max=clip),
        torch.full_like(effort, clip),
    )
    action = torch.zeros_like(effort)
    action = torch.where(selected == POSITIVE_CLASS, projected_magnitude, action)
    action = torch.where(selected == NEGATIVE_CLASS, -projected_magnitude, action)
    class_names = [TRI_STATE_NAMES[int(value)] for value in selected.cpu()]
    diagnostics = {
        "aggregated_tri_state_probability": probabilities.detach().cpu().tolist(),
        "selected_class_index": [int(value) for value in selected.cpu()],
        "selected_class": class_names,
        "winner_runner_up_margin": [float(value) for value in winner_margin.cpu()],
        "selection_tie": [bool(value) for value in tie.cpu()],
        "selection_reason": [
            "idle_exact_tie" if bool(value) else "strict_argmax" for value in tie.cpu()
        ],
        "aggregated_effort": [float(value) for value in effort.cpu()],
        "positive_projection_floor": [float(value + margin) for value in pos.cpu()],
        "negative_projection_floor": [float(value + margin) for value in neg.cpu()],
        "projected_action": [float(value) for value in action.cpu()],
        "output_domain": str(config["output_domain"]),
        "threshold_json": config.get("threshold_json"),
        "threshold_sha256": config.get("threshold_sha256"),
        "projection_margin": margin,
        "projection_clip": clip,
        "selection_rule_version": str(config["selection_rule_version"]),
    }
    return action, diagnostics


def _held_prefix_loss_terms(
    *,
    tri_probabilities: torch.Tensor,
    labels: torch.Tensor,
    transition_mask: torch.Tensor | None,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    zero = tri_probabilities.new_zeros(())
    if not bool(config["held_enabled"]):
        return {
            "factorized_held_nll": zero,
            "factorized_held_loss": zero,
            "factorized_held_conflict_count": zero,
            "factorized_held_target_count": zero,
        }
    if transition_mask is None:
        raise ValueError(
            "factorized held-prefix loss requires state_hold_transition_mask"
        )
    expected = (tri_probabilities.shape[0], len(AXIS_NAMES), 2)
    if tuple(transition_mask.shape) != expected:
        raise ValueError(
            "state_hold_transition_mask must have shape "
            f"{expected}, got {tuple(transition_mask.shape)}"
        )
    transition = transition_mask.to(device=tri_probabilities.device, dtype=torch.bool)
    if torch.any(transition[..., 0] & transition[..., 1]):
        raise ValueError(
            "state_hold_transition_mask cannot select both directions of one axis"
        )
    held = held_temporal_prefix_values(
        tri_probabilities,
        hold_horizon_steps=int(config["hold_horizon_steps"]),
        exponential_k=float(config["exponential_k"]),
    )
    delays = torch.as_tensor(
        config["target_delays"], dtype=torch.long, device=held.device
    )
    tail = held.index_select(1, delays)
    pos_mask = transition[..., 0]
    neg_mask = transition[..., 1]
    target_mask = (pos_mask | neg_mask).unsqueeze(1).expand(-1, len(delays), -1)
    desired_class = torch.where(
        pos_mask,
        torch.full_like(pos_mask, POSITIVE_CLASS, dtype=torch.long),
        torch.full_like(pos_mask, NEGATIVE_CLASS, dtype=torch.long),
    )
    desired_class = desired_class.unsqueeze(1).expand(-1, len(delays), -1)
    selected_probability = torch.gather(
        tail, dim=-1, index=desired_class.unsqueeze(-1)
    ).squeeze(-1)
    target_count = target_mask.to(tail.dtype).sum()
    held_nll = (
        -torch.log(selected_probability.clamp_min(torch.finfo(tail.dtype).tiny))
        * target_mask.to(tail.dtype)
    ).sum() / target_count.clamp_min(1.0)

    recorded_labels = labels.index_select(1, delays)
    conflict = (recorded_labels != desired_class) & target_mask
    conflict_count = conflict.to(tail.dtype).sum()
    return {
        "factorized_held_nll": held_nll,
        "factorized_held_loss": float(config["held_weight"]) * held_nll,
        "factorized_held_conflict_count": conflict_count,
        "factorized_held_target_count": target_count,
    }


def _resolve_thresholds(
    projection: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], str | None]:
    threshold_path_raw = projection.get("threshold_json")
    if threshold_path_raw is not None:
        threshold_path = Path(str(threshold_path_raw)).expanduser().resolve()
        if not threshold_path.is_file():
            raise FileNotFoundError(
                "factorized_intent_effort projection threshold_json does not exist: "
                f"{threshold_path}"
            )
        thresholds = load_deadzone_thresholds(threshold_path)
        resolved_path: str | None = str(threshold_path)
    else:
        raw_thresholds = projection.get("thresholds")
        if not isinstance(raw_thresholds, Mapping):
            raise ValueError(
                "factorized_intent_effort.projection requires threshold_json or "
                "thresholds"
            )
        thresholds = {}
        for axis in AXIS_NAMES:
            axis_raw = raw_thresholds.get(axis)
            if not isinstance(axis_raw, Mapping):
                raise ValueError(
                    f"factorized_intent_effort thresholds missing axis {axis!r}"
                )
            thresholds[axis] = {
                "pos": float(axis_raw["pos"]),
                "neg": float(axis_raw["neg"]),
            }
        resolved_path = None
    for axis in AXIS_NAMES:
        for direction in ("pos", "neg"):
            value = float(thresholds[axis][direction])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "factorized_intent_effort thresholds must be finite and strictly "
                    f"positive: {axis}.{direction}={value}"
                )
            thresholds[axis][direction] = value
    return thresholds, resolved_path


def _norm_tensors(
    norm_stats: Mapping[str, Any],
    *,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_mean = torch.as_tensor(
        norm_stats["action_mean"], dtype=reference.dtype, device=reference.device
    )
    action_std = torch.as_tensor(
        norm_stats["action_std"], dtype=reference.dtype, device=reference.device
    )
    if action_mean.shape != (len(AXIS_NAMES),) or action_std.shape != (
        len(AXIS_NAMES),
    ):
        raise ValueError("action_mean/action_std must each contain four axes")
    if not torch.isfinite(action_mean).all() or not torch.isfinite(action_std).all():
        raise ValueError("action normalization stats must be finite")
    if torch.any(action_std <= 0.0):
        raise ValueError("action_std must be strictly positive")
    return action_mean, action_std


def _resolve_class_weights(raw: Any) -> torch.Tensor:
    values = np.asarray(raw, dtype=np.float32)
    if values.shape == (3,):
        values = np.broadcast_to(values, (len(AXIS_NAMES), 3)).copy()
    if values.shape != (len(AXIS_NAMES), 3):
        raise ValueError(
            "factorized_intent_effort.classification.class_weights must contain "
            "3 values or a 4x3 table"
        )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(
            "factorized_intent_effort.classification.class_weights must be finite "
            "and strictly positive"
        )
    return torch.as_tensor(values.copy(), dtype=torch.float32)


def _axis_values(raw: Any, *, name: str) -> list[float]:
    if isinstance(raw, (int, float)):
        values = [float(raw)] * len(AXIS_NAMES)
    else:
        values = [float(value) for value in raw]
    if len(values) != len(AXIS_NAMES):
        raise ValueError(f"factorized_intent_effort.{name} must contain four axes")
    if any(not np.isfinite(value) for value in values):
        raise ValueError(f"factorized_intent_effort.{name} must be finite")
    return values


def _finite_float(raw: Any, *, name: str) -> float:
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"factorized_intent_effort.{name} must be finite")
    return value


def _non_negative_float(raw: Any, *, name: str) -> float:
    value = _finite_float(raw, name=name)
    if value < 0.0:
        raise ValueError(f"factorized_intent_effort.{name} must be non-negative")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
