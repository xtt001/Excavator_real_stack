"""Goal/progress/effect-conditioned ACT auxiliary contract.

The continuous ACT action remains the only command source.  This module only
defines train-time targets and differentiable losses for two observations:

* ``goal_delta``/``goal_direction`` forecast the locally observed future
  motion from the current observation;
* ``effect_delta`` forecasts the same future motion conditioned on the ACT
  proposal chunk.  At inference this is a confidence/diagnostic signal; it
  never overwrites, clips, or gates the proposal.

All targets are derived from the recorded qpos/qvel/action arrays.  The
response label is deliberately not a hydraulic-failure label: qpos motion is
the observable outcome and unsupported/invalid horizons are masked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

AXIS_COUNT = 4
DEFAULT_HORIZONS = (4, 8, 20)
DEFAULT_DELTA_THRESHOLD = (0.0015, 0.0015, 0.0015, 0.0015)
DEFAULT_UNSUPPORTED_AXES = (2,)


def _broadcast_axis_values(value: Any, *, name: str) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        values = [float(value)] * AXIS_COUNT
    else:
        values = [float(item) for item in value]
    if len(values) != AXIS_COUNT:
        raise ValueError(f"{name} must contain exactly {AXIS_COUNT} values")
    if not all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    return tuple(values)


@dataclass(frozen=True)
class GoalEffectConfig:
    """Resolved semantic and loss settings for the auxiliary heads."""

    enabled: bool
    horizons: tuple[int, ...]
    delta_threshold: tuple[float, ...]
    unsupported_axes: tuple[int, ...]
    goal_delta_weight: float
    goal_direction_weight: float
    effect_delta_weight: float
    consistency_weight: float
    delta_scale: tuple[float, ...]


def resolve_goal_effect_config(
    raw: Mapping[str, Any] | None,
    *,
    target_scale: Sequence[float] | None = None,
) -> GoalEffectConfig:
    """Validate an opt-in goal/effect configuration.

    ``target_scale`` is computed from the training fold by the dataset loader;
    an explicit config value is allowed for reproducible offline fixtures.
    """

    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    horizons_raw = cfg.get("horizons", DEFAULT_HORIZONS)
    horizons = tuple(sorted({int(item) for item in horizons_raw}))
    if not horizons or any(item <= 0 for item in horizons):
        raise ValueError("goal_effect.horizons must contain positive integers")
    threshold = _broadcast_axis_values(
        cfg.get("delta_threshold", DEFAULT_DELTA_THRESHOLD),
        name="goal_effect.delta_threshold",
    )
    if any(item < 0.0 for item in threshold):
        raise ValueError("goal_effect.delta_threshold must be non-negative")
    unsupported = tuple(sorted({int(item) for item in cfg.get("unsupported_axes", DEFAULT_UNSUPPORTED_AXES)}))
    if any(item < 0 or item >= AXIS_COUNT for item in unsupported):
        raise ValueError("goal_effect.unsupported_axes contains an invalid axis")
    scale_raw = cfg.get(
        "delta_scale",
        target_scale if target_scale is not None else (1.0,) * AXIS_COUNT,
    )
    scale = _broadcast_axis_values(scale_raw, name="goal_effect.delta_scale")
    if any(item <= 0.0 for item in scale):
        raise ValueError("goal_effect.delta_scale must be positive")
    weights = {
        "goal_delta_weight": float(cfg.get("goal_delta_weight", 0.10)),
        "goal_direction_weight": float(cfg.get("goal_direction_weight", 0.05)),
        "effect_delta_weight": float(cfg.get("effect_delta_weight", 0.10)),
        "consistency_weight": float(cfg.get("consistency_weight", 0.02)),
    }
    if any(not np.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("goal_effect loss weights must be finite and non-negative")
    return GoalEffectConfig(
        enabled=enabled,
        horizons=horizons,
        delta_threshold=threshold,
        unsupported_axes=unsupported,
        goal_delta_weight=weights["goal_delta_weight"] if enabled else 0.0,
        goal_direction_weight=weights["goal_direction_weight"] if enabled else 0.0,
        effect_delta_weight=weights["effect_delta_weight"] if enabled else 0.0,
        consistency_weight=weights["consistency_weight"] if enabled else 0.0,
        delta_scale=scale,
    )


def future_delta_scale(
    qpos_sequences: Sequence[np.ndarray],
    horizons: Sequence[int],
    *,
    floor: float = 1.0e-3,
) -> np.ndarray:
    """Return robust train-fold scales for future qpos deltas.

    The 90th percentile is used instead of a validation-derived statistic.  A
    floor keeps the structural stick axis numerically well-conditioned while
    its supervision remains masked by ``unsupported_axes``.
    """

    if not qpos_sequences:
        raise ValueError("qpos_sequences must not be empty")
    scales: list[np.ndarray] = []
    for horizon in horizons:
        deltas: list[np.ndarray] = []
        for qpos in qpos_sequences:
            array = np.asarray(qpos, dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != AXIS_COUNT:
                raise ValueError("qpos sequence must have shape (T, 4)")
            if array.shape[0] > int(horizon):
                deltas.append(np.abs(array[int(horizon) :] - array[: -int(horizon)]))
        if not deltas:
            scales.append(np.full(AXIS_COUNT, float(floor), dtype=np.float32))
            continue
        joined = np.concatenate(deltas, axis=0)
        scales.append(np.maximum(np.quantile(joined, 0.90, axis=0), float(floor)))
    # The loss uses one scale per axis; aggregate across the declared horizons.
    return np.maximum(np.stack(scales, axis=0).mean(axis=0), float(floor)).astype(
        np.float32
    )


def build_goal_effect_targets(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    timestep: int,
    config: GoalEffectConfig,
) -> dict[str, np.ndarray]:
    """Build one causal sample's future-goal/effect labels.

    The target uses only observations after ``timestep`` as a training label;
    inference never receives these arrays.  ``valid`` masks the unavailable
    tail and structural unsupported axes.  Direction classes are
    ``0=negative, 1=idle, 2=positive``.
    """

    qpos_arr = np.asarray(qpos, dtype=np.float32)
    qvel_arr = np.asarray(qvel, dtype=np.float32)
    action_arr = np.asarray(action, dtype=np.float32)
    if qpos_arr.ndim != 2 or qpos_arr.shape[1] != AXIS_COUNT:
        raise ValueError("qpos must have shape (T, 4)")
    if qvel_arr.shape != qpos_arr.shape or action_arr.shape != qpos_arr.shape:
        raise ValueError("qvel/action must match qpos shape")
    t0 = int(timestep)
    if t0 < 0 or t0 >= qpos_arr.shape[0]:
        raise ValueError(f"timestep {t0} is outside sequence of length {len(qpos_arr)}")

    horizon_count = len(config.horizons)
    delta = np.zeros((horizon_count, AXIS_COUNT), dtype=np.float32)
    valid = np.zeros((horizon_count, AXIS_COUNT), dtype=bool)
    direction = np.ones((horizon_count, AXIS_COUNT), dtype=np.int64)
    thresholds = np.asarray(config.delta_threshold, dtype=np.float32)
    unsupported = np.asarray(config.unsupported_axes, dtype=np.int64)
    for index, horizon in enumerate(config.horizons):
        end = t0 + int(horizon)
        if end >= qpos_arr.shape[0]:
            continue
        delta[index] = qpos_arr[end] - qpos_arr[t0]
        valid[index] = True
        direction[index] = np.where(
            delta[index] < -thresholds,
            0,
            np.where(delta[index] > thresholds, 2, 1),
        )
        if unsupported.size:
            valid[index, unsupported] = False
            direction[index, unsupported] = 1

    if unsupported.size:
        valid[:, unsupported] = False

    # A response trace is kept as an auxiliary audit label.  It is not used as
    # a hydraulic failure ground truth: the model only receives the future
    # qpos delta above, and the response mask is available for diagnostics.
    effect_valid = np.zeros(AXIS_COUNT, dtype=bool)
    effect_response = np.zeros(AXIS_COUNT, dtype=np.float32)
    effect_horizon = int(max(config.horizons))
    response_end = min(qvel_arr.shape[0], t0 + effect_horizon + 1)
    if response_end > t0 + 1:
        effect_response[:] = np.mean(qvel_arr[t0 + 1 : response_end], axis=0)
        effect_valid[:] = True
    if unsupported.size:
        effect_valid[unsupported] = False
    return {
        "goal_future_delta": delta,
        "goal_future_valid": valid,
        "goal_future_direction": direction,
        "goal_effect_delta": delta.copy(),
        "goal_effect_valid": valid.copy(),
        "goal_effect_response": effect_response,
        "goal_effect_response_valid": effect_valid,
    }


class GoalEffectHead(nn.Module):
    """Observation-context goal head plus action-conditioned effect head."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        action_dim: int,
        num_queries: int,
        horizons: Sequence[int],
    ) -> None:
        super().__init__()
        self.horizons = tuple(int(item) for item in horizons)
        self.num_queries = int(num_queries)
        self.action_dim = int(action_dim)
        trunk_dim = int(hidden_dim)
        self.goal_trunk = nn.Sequential(
            nn.Linear(hidden_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.GELU(),
        )
        self.effect_trunk = nn.Sequential(
            nn.Linear(hidden_dim + num_queries * action_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.GELU(),
        )
        output_dim = len(self.horizons) * AXIS_COUNT
        self.goal_delta_head = nn.Linear(trunk_dim, output_dim)
        self.goal_direction_head = nn.Linear(
            trunk_dim, len(self.horizons) * AXIS_COUNT * 3
        )
        self.effect_delta_head = nn.Linear(trunk_dim, output_dim)

    def forward(
        self,
        context: torch.Tensor,
        proposal_chunk: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if context.ndim != 2:
            raise ValueError(f"context must have shape (B, D), got {tuple(context.shape)}")
        if proposal_chunk.ndim != 3:
            raise ValueError(
                "proposal_chunk must have shape (B, C, A), "
                f"got {tuple(proposal_chunk.shape)}"
            )
        if proposal_chunk.shape[1] != self.num_queries or proposal_chunk.shape[2] != self.action_dim:
            raise ValueError("proposal_chunk shape does not match GoalEffectHead contract")
        batch = context.shape[0]
        goal_hidden = self.goal_trunk(context)
        effect_input = torch.cat([context, proposal_chunk.reshape(batch, -1)], dim=-1)
        effect_hidden = self.effect_trunk(effect_input)
        horizon_count = len(self.horizons)
        return {
            "goal_delta": self.goal_delta_head(goal_hidden).reshape(
                batch, horizon_count, AXIS_COUNT
            ),
            "goal_direction_logits": self.goal_direction_head(goal_hidden).reshape(
                batch, horizon_count, AXIS_COUNT, 3
            ),
            "effect_delta": self.effect_delta_head(effect_hidden).reshape(
                batch, horizon_count, AXIS_COUNT
            ),
        }


def goal_effect_loss_terms(
    *,
    outputs: Mapping[str, torch.Tensor] | None,
    targets: Mapping[str, torch.Tensor] | None,
    config: GoalEffectConfig,
    device_like: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute masked auxiliary losses and expose unweighted diagnostics."""

    zero = device_like.new_zeros(())
    empty = {
        "goal_delta_loss": zero,
        "goal_direction_loss": zero,
        "effect_delta_loss": zero,
        "goal_effect_consistency_loss": zero,
        "goal_effect_loss": zero,
    }
    if not config.enabled:
        return empty
    if outputs is None or targets is None:
        raise ValueError("goal_effect enabled requires model outputs and targets")
    required = (
        "goal_future_delta",
        "goal_future_valid",
        "goal_future_direction",
        "goal_effect_delta",
        "goal_effect_valid",
    )
    missing = [name for name in required if name not in targets]
    if missing:
        raise ValueError(f"goal_effect targets missing keys: {missing}")
    scale = torch.as_tensor(
        config.delta_scale,
        device=device_like.device,
        dtype=device_like.dtype,
    ).reshape(1, 1, AXIS_COUNT)
    goal_target = targets["goal_future_delta"].to(device_like).to(device_like.dtype)
    goal_valid = targets["goal_future_valid"].to(device_like).bool()
    effect_target = targets["goal_effect_delta"].to(device_like).to(device_like.dtype)
    effect_valid = targets["goal_effect_valid"].to(device_like).bool()
    goal_pred = outputs["goal_delta"].to(device_like)
    effect_pred = outputs["effect_delta"].to(device_like)
    goal_mask = goal_valid.expand_as(goal_pred)
    effect_mask = effect_valid.expand_as(effect_pred)

    def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scaled_pred = pred / scale
        scaled_target = target / scale
        values = F.smooth_l1_loss(scaled_pred, scaled_target, reduction="none")
        count = mask.to(values.dtype).sum().clamp_min(1.0)
        return (values * mask.to(values.dtype)).sum() / count

    goal_delta_loss = masked_smooth_l1(goal_pred, goal_target, goal_mask)
    effect_delta_loss = masked_smooth_l1(effect_pred, effect_target, effect_mask)

    direction_target = targets["goal_future_direction"].to(device_like).long()
    direction_valid = goal_valid
    logits = outputs["goal_direction_logits"].to(device_like)
    direction_values = F.cross_entropy(
        logits.reshape(-1, 3), direction_target.reshape(-1), reduction="none"
    ).reshape_as(direction_valid)
    direction_count = direction_valid.to(direction_values.dtype).sum().clamp_min(1.0)
    goal_direction_loss = (
        direction_values * direction_valid.to(direction_values.dtype)
    ).sum() / direction_count
    consistency_mask = goal_mask & effect_mask
    consistency_count = consistency_mask.to(goal_pred.dtype).sum().clamp_min(1.0)
    consistency = ((goal_pred - effect_pred).abs() / scale)
    consistency_loss = (consistency * consistency_mask.to(consistency.dtype)).sum() / consistency_count
    total = (
        config.goal_delta_weight * goal_delta_loss
        + config.goal_direction_weight * goal_direction_loss
        + config.effect_delta_weight * effect_delta_loss
        + config.consistency_weight * consistency_loss
    )
    return {
        "goal_delta_loss": goal_delta_loss,
        "goal_direction_loss": goal_direction_loss,
        "effect_delta_loss": effect_delta_loss,
        "goal_effect_consistency_loss": consistency_loss,
        "goal_effect_loss": total,
    }
