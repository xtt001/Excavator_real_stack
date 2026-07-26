"""
ACT policy adapter.

Wraps the original ACTPolicy (nn.Module) behind the testbed Policy ABC so
that CLI tools and adapter tests can call policy.predict(obs) without knowing
any ACT internals.

Temporal aggregation
--------------------
When `temporal_agg=True`, actions are chunked and averaged using the
scheme from the original paper: at each step we query the model at
frequency 1, accumulate the chunk into a (T, T+C, Na) tensor, then
select the weighted average of all past predictions for the current step.
Live control can run longer than the resolved training `episode_len`, so the
aggregation tensor grows at runtime instead of treating that length as a hard
limit.
"""

from __future__ import annotations

import json
import pickle
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as transforms

from testbed.data.causal_visual_history import (
    CausalVisualHistory,
    CausalVisualHistoryState,
    resolve_temporal_input_config,
)
from testbed.policies.act.action_state_effort import (
    action_state_loss_terms,
    resolve_action_state_effort_config,
)
from testbed.policies.act.effective_action import (
    effective_action_loss_terms,
    resolve_effective_action_config,
    weighted_action_l1,
)
from testbed.policies.act.factorized_action import (
    FactorizedTemporalAggregator,
    FactorizedTemporalState,
    factorized_training_loss_terms,
    project_factorized_action,
    query_factorized_values,
    resolve_factorized_config,
)
from testbed.policies.act.goal_effect import (
    GoalEffectConfig,
    goal_effect_loss_terms,
    resolve_goal_effect_config,
)
from testbed.policies.act.phase_routed_condition import (
    build_runtime_phase_router,
    resolve_phase_routed_condition_config,
)
from testbed.policies.base import Policy, register_policy


def _kl_divergence(mu, logvar):
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise, mean_kld


_AXIS_NAMES = ("swing", "boom", "stick", "bucket")
_INFERENCE_PRECISIONS = ("fp32", "fp16")
_TEMPORAL_AGG_K = 0.01


def _resolve_inference_autocast_dtype(
    value: Any,
    *,
    device: torch.device,
) -> torch.dtype | None:
    precision = str(value or "fp32").strip().lower()
    if precision not in _INFERENCE_PRECISIONS:
        raise ValueError(
            "inference_precision must be one of "
            f"{_INFERENCE_PRECISIONS}, got {value!r}"
        )
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError(
                "inference_precision='fp16' requires a CUDA device; "
                f"resolved device is {device}."
            )
        return torch.float16
    return None


def _camera_image_to_chw_float(raw_image: np.ndarray, *, key: str) -> np.ndarray:
    """Convert one supported camera input to the legacy CHW float contract."""

    raw_image = np.asarray(raw_image)
    image = np.asarray(raw_image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(
            f"ACTAdapter.predict(): expected {key!r} to be rank-3, "
            f"got shape {image.shape}."
        )
    if image.shape[0] == 3:
        return image
    if image.shape[-1] == 3:
        image = np.transpose(image, (2, 0, 1))
        if raw_image.dtype == np.uint8 or image.max() > 1.0:
            image = image / 255.0
        return image
    raise ValueError(
        f"ACTAdapter.predict(): expected {key!r} to have 3 channels, "
        f"got shape {image.shape}."
    )


def _single_frame_image_tensor(
    raw_images: list[np.ndarray],
    *,
    keys: list[str],
    device: torch.device,
    device_uint8_preprocess: bool,
) -> torch.Tensor:
    """Build ``(1, cameras, C, H, W)`` while preserving the legacy fallback."""

    use_device_uint8 = bool(device_uint8_preprocess) and all(
        np.asarray(image).ndim == 3
        and np.asarray(image).shape[-1] == 3
        and np.asarray(image).dtype == np.uint8
        for image in raw_images
    )
    if use_device_uint8:
        stacked = np.ascontiguousarray(np.stack(raw_images, axis=0))
        return (
            torch.from_numpy(stacked)
            .to(device=device, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .div_(255.0)
            .unsqueeze(0)
        )

    images = [
        _camera_image_to_chw_float(image, key=key)
        for image, key in zip(raw_images, keys)
    ]
    stacked = np.ascontiguousarray(np.stack(images, axis=0))
    return torch.from_numpy(stacked).float().to(device).unsqueeze(0)


@dataclass(frozen=True)
class ACTAdapterState:
    """All mutable inference state required for deterministic branch replay."""

    step: int
    all_time_actions: torch.Tensor | None
    temporal_weight_cache: dict[int, torch.Tensor]
    cached_actions: torch.Tensor | None
    temporal_aggregation_diagnostics: dict[str, Any] | None
    factorized_diagnostics: dict[str, Any] | None
    goal_effect_diagnostics: dict[str, Any] | None
    temporal_input_diagnostics: dict[str, Any] | None
    temporal_last_timestamps: dict[str, int]
    temporal_fallback_timestamp: int
    visual_history_state: CausalVisualHistoryState | None
    factorized_aggregator_state: FactorizedTemporalState | None
    condition_route: int | None
    condition_route_consecutive: int | None


def _coerce_timestamp(value: Any) -> int | None:
    """Return an integer timestamp or ``None`` for absent/malformed metadata."""

    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _goal_effect_diagnostics(outputs: Any) -> dict[str, Any] | None:
    """Convert the optional auxiliary forecast to compact JSON-safe values."""

    if outputs is None:
        return None
    result: dict[str, Any] = {}
    for key, value in outputs.items():
        tensor = value.detach().float().cpu()
        if tensor.ndim > 1:
            tensor = tensor[0]
        result[f"policy_{key}"] = tensor.numpy().tolist()
    return result


def _resolve_deadzone_loss_config(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "same_dir_promote_weight": 0.0,
            "idle_suppression_weight": 0.0,
            "wrong_effective_weight": 0.0,
            "margin": torch.zeros(4, dtype=torch.float32),
            "pos": torch.zeros(4, dtype=torch.float32),
            "neg": torch.zeros(4, dtype=torch.float32),
            "penalize_wrong_effective": False,
            "idle_denominator": "crossing_axes",
            "wrong_denominator": "crossing_axes",
        }

    thresholds = _load_deadzone_thresholds(cfg)
    margin = _broadcast_axis_values(cfg.get("margin", 0.0), name="deadzone_loss.margin")
    apply_when = str(cfg.get("apply_when", "expert_effective_only"))
    if apply_when != "expert_effective_only":
        raise ValueError(
            "deadzone_loss.apply_when currently supports only 'expert_effective_only'."
        )
    legacy_weight = float(cfg.get("weight", 0.1))
    legacy_wrong_weight = float(cfg.get("wrong_weight", legacy_weight))
    return {
        "enabled": True,
        "same_dir_promote_weight": float(
            cfg.get("same_dir_promote_weight", legacy_weight)
        ),
        "idle_suppression_weight": float(
            cfg.get("idle_suppression_weight", legacy_wrong_weight)
        ),
        "wrong_effective_weight": float(
            cfg.get("wrong_effective_weight", legacy_wrong_weight)
        ),
        "margin": torch.as_tensor(margin, dtype=torch.float32),
        "pos": torch.as_tensor(
            [thresholds[axis]["pos"] for axis in _AXIS_NAMES], dtype=torch.float32
        ),
        "neg": torch.as_tensor(
            [thresholds[axis]["neg"] for axis in _AXIS_NAMES], dtype=torch.float32
        ),
        "penalize_wrong_effective": bool(cfg.get("penalize_wrong_effective", True)),
        "same_dir_window": _resolve_same_dir_window(cfg.get("same_dir_window", "all")),
        "same_dir_window_steps": int(cfg.get("same_dir_window_steps", 1)),
        "idle_denominator": _resolve_denominator(
            cfg.get("idle_denominator", "crossing_axes"),
            name="deadzone_loss.idle_denominator",
            allowed={"crossing_axes", "all_idle_axes"},
        ),
        "wrong_denominator": _resolve_denominator(
            cfg.get("wrong_denominator", "crossing_axes"),
            name="deadzone_loss.wrong_denominator",
            allowed={"crossing_axes", "all_wrong_candidate_axes"},
        ),
    }


def _resolve_intent_loss_config(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "weight": 0.0,
            "positive_weight": torch.ones(8, dtype=torch.float32),
            "intent_dim": 0,
            "current_steps": 0,
            "pos": torch.zeros(4, dtype=torch.float32),
            "neg": torch.zeros(4, dtype=torch.float32),
        }

    thresholds = _load_deadzone_thresholds(cfg)
    positive_weight = _broadcast_intent_values(
        cfg.get("positive_weight", 1.0),
        name="intent_loss.positive_weight",
    )
    current_steps = int(cfg.get("current_steps", 0))
    if current_steps < 0:
        raise ValueError("intent_loss.current_steps must be >= 0")
    return {
        "enabled": True,
        "weight": float(cfg.get("weight", 0.05)),
        "positive_weight": torch.as_tensor(positive_weight, dtype=torch.float32),
        "intent_dim": 8,
        "current_steps": current_steps,
        "pos": torch.as_tensor(
            [thresholds[axis]["pos"] for axis in _AXIS_NAMES], dtype=torch.float32
        ),
        "neg": torch.as_tensor(
            [thresholds[axis]["neg"] for axis in _AXIS_NAMES], dtype=torch.float32
        ),
    }


def _resolve_window_deadzone_loss_config(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "same_dir_promote_weight": 0.0,
            "stop_suppression_weight": 0.0,
            "wrong_effective_weight": 0.0,
            "margin": torch.zeros(4, dtype=torch.float32),
            "pos": torch.zeros(4, dtype=torch.float32),
            "neg": torch.zeros(4, dtype=torch.float32),
        }
    thresholds = _load_deadzone_thresholds(cfg)
    margin = _broadcast_axis_values(
        cfg.get("margin", 0.0), name="window_deadzone_loss.margin"
    )
    return {
        "enabled": True,
        "same_dir_promote_weight": float(cfg.get("same_dir_promote_weight", 0.1)),
        "stop_suppression_weight": float(cfg.get("stop_suppression_weight", 0.05)),
        "wrong_effective_weight": float(cfg.get("wrong_effective_weight", 0.05)),
        "margin": torch.as_tensor(margin, dtype=torch.float32),
        "pos": torch.as_tensor(
            [thresholds[axis]["pos"] for axis in _AXIS_NAMES], dtype=torch.float32
        ),
        "neg": torch.as_tensor(
            [thresholds[axis]["neg"] for axis in _AXIS_NAMES], dtype=torch.float32
        ),
    }


def _resolve_temporal_release_loss_config(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "weight": 0.0,
            "release_window_steps": 0,
            "pos": torch.zeros(4, dtype=torch.float32),
            "neg": torch.zeros(4, dtype=torch.float32),
        }
    thresholds = _load_deadzone_thresholds(cfg)
    return {
        "enabled": True,
        "weight": float(cfg.get("weight", 0.05)),
        "release_window_steps": int(cfg.get("release_window_steps", 3)),
        "pos": torch.as_tensor(
            [thresholds[axis]["pos"] for axis in _AXIS_NAMES], dtype=torch.float32
        ),
        "neg": torch.as_tensor(
            [thresholds[axis]["neg"] for axis in _AXIS_NAMES], dtype=torch.float32
        ),
    }


def _resolve_condition_counterfactual_consistency_config(
    raw: Any,
) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "coefficient": 0.0,
            "candidate_forward": "inference_zero_latent",
            "loss_domain": "normalized_action",
            "reduction": "mean_abs_all_queries_axes",
        }

    expected = {
        "coefficient": 1.0,
        "candidate_forward": "inference_zero_latent",
        "loss_domain": "normalized_action",
        "reduction": "mean_abs_all_queries_axes",
    }
    for key, expected_value in expected.items():
        actual = cfg.get(key)
        if actual != expected_value:
            raise ValueError(
                "condition_counterfactual_consistency."
                f"{key} must be frozen to {expected_value!r}, got {actual!r}"
            )
    return {"enabled": True, **expected}


def _resolve_demo_target_hold_loss_config(raw: Any) -> dict[str, Any]:
    """Resolve the assist-aware held-sequence training objective."""

    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "weight": 0.0,
            "assist_trigger_fraction": 0.0,
            "margin": torch.zeros(4, dtype=torch.float32),
            "hold_horizon_steps": 0,
            "min_consecutive_steps": 0,
            "pos": torch.zeros(4, dtype=torch.float32),
            "neg": torch.zeros(4, dtype=torch.float32),
        }

    thresholds = _load_deadzone_thresholds(cfg)
    weight = float(cfg.get("weight", 0.1))
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(
            "demo_target_hold_loss.weight must be finite and non-negative."
        )
    trigger_fraction = float(cfg.get("assist_trigger_fraction", 0.5))
    if not np.isfinite(trigger_fraction) or not 0.0 < trigger_fraction <= 1.0:
        raise ValueError(
            "demo_target_hold_loss.assist_trigger_fraction must be in (0, 1]."
        )
    margin = _broadcast_axis_values(
        cfg.get("margin", 0.0),
        name="demo_target_hold_loss.margin",
    )
    if any(not np.isfinite(value) or value < 0.0 for value in margin):
        raise ValueError(
            "demo_target_hold_loss.margin must contain finite non-negative values."
        )
    hold_horizon_steps = int(cfg.get("hold_horizon_steps", 20))
    if hold_horizon_steps <= 0:
        raise ValueError("demo_target_hold_loss.hold_horizon_steps must be positive.")
    min_consecutive_steps = int(cfg.get("min_consecutive_steps", 2))
    if not 0 < min_consecutive_steps <= hold_horizon_steps:
        raise ValueError(
            "demo_target_hold_loss.min_consecutive_steps must be in "
            "[1, hold_horizon_steps]."
        )
    return {
        "enabled": True,
        "weight": weight,
        "assist_trigger_fraction": trigger_fraction,
        "margin": torch.as_tensor(margin, dtype=torch.float32),
        "hold_horizon_steps": hold_horizon_steps,
        "min_consecutive_steps": min_consecutive_steps,
        "pos": torch.as_tensor(
            [thresholds[axis]["pos"] for axis in _AXIS_NAMES],
            dtype=torch.float32,
        ),
        "neg": torch.as_tensor(
            [thresholds[axis]["neg"] for axis in _AXIS_NAMES],
            dtype=torch.float32,
        ),
    }


def _resolve_same_dir_window(value: Any) -> str:
    result = str(value or "all")
    if result not in {"all", "expert_transition_window"}:
        raise ValueError(
            "deadzone_loss.same_dir_window must be 'all' or 'expert_transition_window'."
        )
    return result


def _resolve_denominator(value: Any, *, name: str, allowed: set[str]) -> str:
    result = str(value or "crossing_axes")
    if result not in allowed:
        formatted = ", ".join(repr(item) for item in sorted(allowed))
        raise ValueError(f"{name} must be one of {formatted}.")
    return result


def _load_deadzone_thresholds(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    if "thresholds" in cfg:
        payload = cfg["thresholds"]
    else:
        path_raw = cfg.get("threshold_json")
        if not path_raw:
            raise ValueError(
                "deadzone_loss.enabled requires threshold_json or thresholds."
            )
        path = Path(str(path_raw))
        if not path.exists():
            raise FileNotFoundError(
                f"deadzone_loss threshold_json does not exist: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    raw = (
        payload.get("deadzone_action", payload)
        if isinstance(payload, dict)
        else payload
    )
    thresholds: dict[str, dict[str, float]] = {}
    for axis in _AXIS_NAMES:
        axis_raw = raw.get(axis) if isinstance(raw, dict) else None
        if not isinstance(axis_raw, dict):
            raise ValueError(f"deadzone_loss thresholds missing axis {axis!r}.")
        thresholds[axis] = {
            "pos": _threshold_value(axis_raw.get("pos"), axis=axis, direction="pos"),
            "neg": _threshold_value(axis_raw.get("neg"), axis=axis, direction="neg"),
        }
    return thresholds


def _threshold_value(value: Any, *, axis: str, direction: str) -> float:
    if isinstance(value, dict):
        if "threshold_action_abs" in value:
            value = value["threshold_action_abs"]
        elif "value" in value:
            value = value["value"]
        else:
            raise ValueError(
                f"deadzone_loss threshold for {axis}.{direction} is missing value."
            )
    result = float(value)
    if result < 0.0:
        raise ValueError(
            f"deadzone_loss threshold for {axis}.{direction} must be >= 0."
        )
    return result


def _broadcast_axis_values(value: Any, *, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * len(_AXIS_NAMES)
    values = [float(item) for item in value]
    if len(values) != len(_AXIS_NAMES):
        raise ValueError(f"{name} must be a scalar or {len(_AXIS_NAMES)} values.")
    return values


def _broadcast_intent_values(value: Any, *, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * (len(_AXIS_NAMES) * 2)
    values = [float(item) for item in value]
    if len(values) == len(_AXIS_NAMES):
        expanded: list[float] = []
        for item in values:
            expanded.extend([item, item])
        return expanded
    if len(values) != len(_AXIS_NAMES) * 2:
        raise ValueError(
            f"{name} must be a scalar, {len(_AXIS_NAMES)} values, or {len(_AXIS_NAMES) * 2} values."
        )
    return values


def _expert_transition_window_mask(
    *,
    expert_step_effective: torch.Tensor,
    valid_step: torch.Tensor,
    window_steps: int,
) -> torch.Tensor:
    steps = max(1, int(window_steps))
    effective = expert_step_effective.to(dtype=torch.bool) & valid_step.to(
        dtype=torch.bool
    )
    previous = torch.zeros_like(effective)
    previous[:, 1:] = effective[:, :-1]
    transition = effective & ~previous
    mask = transition.clone()
    for offset in range(1, steps):
        shifted = torch.zeros_like(transition)
        shifted[:, offset:] = transition[:, :-offset]
        mask = mask | shifted
    return mask & effective


def _direction_release_window_mask(
    *,
    expert_effective: torch.Tensor,
    valid: torch.Tensor,
    window_steps: int,
) -> torch.Tensor:
    steps = max(1, int(window_steps))
    effective = expert_effective.to(dtype=torch.bool) & valid.to(dtype=torch.bool)
    previous = torch.zeros_like(effective)
    previous[:, 1:] = effective[:, :-1]
    release_start = previous & ~effective & valid.to(dtype=torch.bool)
    mask = release_start.clone()
    for offset in range(1, steps):
        shifted = torch.zeros_like(release_start)
        shifted[:, offset:] = release_start[:, :-offset]
        mask = mask | shifted
    return mask & valid.to(dtype=torch.bool)


def _masked_action_l1(
    *,
    expert: torch.Tensor,
    policy: torch.Tensor,
    valid_mask: torch.Tensor,
    action_loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    import torch.nn.functional as F

    all_l1 = F.l1_loss(expert, policy, reduction="none")
    valid = valid_mask.to(dtype=torch.bool).expand_as(all_l1)
    if action_loss_mask is None:
        return (all_l1 * valid.to(all_l1.dtype)).mean()
    action_valid = action_loss_mask.to(
        device=expert.device, dtype=torch.bool
    ).unsqueeze(-1)
    mask = valid & action_valid
    count = mask.to(all_l1.dtype).sum().clamp_min(1.0)
    return (all_l1 * mask.to(all_l1.dtype)).sum() / count


def _held_temporal_prefix_actions(
    policy: torch.Tensor,
    *,
    hold_horizon_steps: int,
) -> torch.Tensor:
    """Replay ACT aggregation prefixes for one observation repeated in place."""

    if policy.ndim != 3:
        raise ValueError(f"policy must have shape (B, C, A), got {tuple(policy.shape)}")
    horizon = int(hold_horizon_steps)
    if not 0 < horizon <= int(policy.shape[1]):
        raise ValueError(
            "hold_horizon_steps must be in [1, policy chunk length], "
            f"got {horizon} for chunk length {policy.shape[1]}"
        )
    prefixes: list[torch.Tensor] = []
    for delay in range(horizon):
        # At held tick ``delay``, the oldest repeated chunk contributes query
        # ``delay`` and the newest contributes query zero.  This ordering and
        # exponential weighting exactly match ``_aggregate`` below.
        actions = torch.flip(policy[:, : delay + 1], dims=(1,))
        weights = torch.exp(
            -_TEMPORAL_AGG_K
            * torch.arange(
                delay + 1,
                dtype=policy.dtype,
                device=policy.device,
            )
        )
        weights = weights / weights.sum()
        prefixes.append((actions * weights.view(1, -1, 1)).sum(dim=1))
    return torch.stack(prefixes, dim=1)


@register_policy("act")
class ACTAdapter(Policy):
    """
    Runnable ACT policy for inference.

    Parameters
    ----------
    policy_config  Dict passed to build_ACT_model_and_optimizer.
    norm_stats     Dataset normalisation stats (proprio_mean/std, action_mean/std).
    temporal_agg   Use temporal action aggregation (default False).
    device         Torch device string (default "cuda").
    """

    def __init__(
        self,
        policy_config: dict,
        norm_stats: dict[str, np.ndarray],
        temporal_agg: bool = False,
        device: str = "cuda",
        inference_precision: str = "fp32",
        inference_compile: bool = False,
        inference_compile_mode: str = "reduce-overhead",
        inference_compile_dynamic: bool = False,
        device_uint8_preprocess: bool = False,
        temporal_aggregation_diagnostics: bool = False,
    ):
        from testbed.policies.act.detr.main import build_ACT_model_and_optimizer

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.norm_stats = norm_stats
        self.temporal_agg = temporal_agg
        self._inference_autocast_dtype = _resolve_inference_autocast_dtype(
            inference_precision,
            device=self.device,
        )
        self._inference_precision = str(inference_precision or "fp32").strip().lower()
        self._inference_compile = bool(inference_compile)
        self._inference_compile_mode = str(
            inference_compile_mode or "reduce-overhead"
        ).strip()
        if not self._inference_compile_mode:
            raise ValueError("inference_compile_mode must not be empty")
        self._inference_compile_dynamic = bool(inference_compile_dynamic)
        self._inference_compile_applied = False
        self._device_uint8_preprocess = bool(device_uint8_preprocess)
        self._temporal_aggregation_diagnostics_enabled = bool(
            temporal_aggregation_diagnostics
        )
        self.kl_weight = policy_config.get("kl_weight", 10)
        self._camera_names = list(policy_config.get("camera_names", []))
        self._temporal_input = resolve_temporal_input_config(
            policy_config.get("temporal_input")
        )
        self._low_dim_keys = list(policy_config.get("low_dim_keys", ["qpos"]))
        self._deadzone_loss = _resolve_deadzone_loss_config(
            policy_config.get("deadzone_loss")
        )
        self._intent_loss = _resolve_intent_loss_config(
            policy_config.get("intent_loss")
        )
        self._window_deadzone_loss = _resolve_window_deadzone_loss_config(
            policy_config.get("window_deadzone_loss")
        )
        self._temporal_release_loss = _resolve_temporal_release_loss_config(
            policy_config.get("temporal_release_loss")
        )
        self._condition_counterfactual_consistency = (
            _resolve_condition_counterfactual_consistency_config(
                policy_config.get("condition_counterfactual_consistency")
            )
        )
        self._phase_routed_condition = resolve_phase_routed_condition_config(
            policy_config.get("phase_routed_condition")
        )
        self._demo_target_hold_loss = _resolve_demo_target_hold_loss_config(
            policy_config.get("demo_target_hold_loss")
        )
        self._goal_effect: GoalEffectConfig = resolve_goal_effect_config(
            policy_config.get("goal_effect"),
            target_scale=norm_stats.get("goal_effect_delta_scale"),
        )
        self._action_state_effort = resolve_action_state_effort_config(
            policy_config.get("action_state_effort")
        )
        self._effective_action = resolve_effective_action_config(
            policy_config.get("effective_action")
        )
        policy_config = dict(policy_config)
        policy_config["phase_routed_condition"] = dict(
            self._phase_routed_condition
        )
        goal_effect_policy_config = dict(policy_config.get("goal_effect") or {})
        goal_effect_policy_config["enabled"] = bool(self._goal_effect.enabled)
        goal_effect_policy_config["horizons"] = list(self._goal_effect.horizons)
        policy_config["goal_effect"] = goal_effect_policy_config
        action_state_policy_config = dict(
            policy_config.get("action_state_effort") or {}
        )
        action_state_policy_config["enabled"] = bool(
            self._action_state_effort["enabled"]
        )
        action_state_policy_config["state_count"] = int(
            self._action_state_effort["state_count"]
        )
        policy_config["action_state_effort"] = action_state_policy_config
        effective_action_policy_config = dict(
            policy_config.get("effective_action") or {}
        )
        effective_action_policy_config["enabled"] = bool(
            self._effective_action["enabled"]
        )
        policy_config["effective_action"] = effective_action_policy_config
        self._factorized_action = resolve_factorized_config(
            policy_config.get("factorized_intent_effort"),
            num_queries=int(policy_config["num_queries"]),
        )
        if self._factorized_action["enabled"]:
            incompatible = {
                "deadzone_loss": self._deadzone_loss["enabled"],
                "intent_loss": self._intent_loss["enabled"],
                "window_deadzone_loss": self._window_deadzone_loss["enabled"],
                "temporal_release_loss": self._temporal_release_loss["enabled"],
                "demo_target_hold_loss": self._demo_target_hold_loss["enabled"],
            }
            active = [name for name, enabled in incompatible.items() if enabled]
            if active:
                raise ValueError(
                    "factorized_intent_effort replaces incompatible signed/BCE "
                    f"objectives: {', '.join(active)}"
                )
        policy_config["intent_dim"] = int(
            self._factorized_action["intent_dim"]
            if self._factorized_action["enabled"]
            else self._intent_loss["intent_dim"]
        )

        model, optimizer = build_ACT_model_and_optimizer(policy_config)
        self._model = model.to(self.device)
        self._model.eval()
        self._optimizer = optimizer

        # temporal aggregation state
        self._num_queries: int = policy_config["num_queries"]
        self._t: int = 0
        self._all_time_actions: torch.Tensor | None = None
        self._cached_actions: torch.Tensor | None = None
        self._temporal_weight_cache: dict[int, torch.Tensor] = {}
        self._last_temporal_aggregation_diagnostics: dict[str, Any] | None = None
        self._max_episode_len = int(policy_config.get("max_episode_len", 400))
        self._factorized_aggregator = (
            FactorizedTemporalAggregator(
                num_queries=self._num_queries,
                device=self.device,
                max_episode_len=self._max_episode_len,
                exponential_k=float(self._factorized_action["exponential_k"]),
            )
            if self._factorized_action["enabled"]
            else None
        )
        self._last_factorized_diagnostics: dict[str, Any] | None = None
        self._last_goal_effect_diagnostics: dict[str, Any] | None = None
        self._visual_history = (
            CausalVisualHistory(
                self._camera_names,
                history_length=int(self._temporal_input["history_steps"]),
            )
            if self._temporal_input["enabled"]
            else None
        )
        self._temporal_last_timestamps: dict[str, int] = {}
        self._temporal_fallback_timestamp = 0
        self._last_temporal_input_diagnostics: dict[str, Any] | None = None
        self._last_raw_action_chunk: torch.Tensor | None = None
        self._condition_phase_router = build_runtime_phase_router(
            self._phase_routed_condition
        )
        self._last_condition_route_diagnostics: dict[str, Any] | None = None

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self._proprio_mean, self._proprio_std = self._resolve_proprio_norm_stats()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _compile_model_for_inference(self) -> None:
        """Apply the opt-in compiler after checkpoint weights are loaded."""

        if not self._inference_compile or self._inference_compile_applied:
            return
        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            raise RuntimeError(
                "inference_compile=true requires a PyTorch build with torch.compile"
            )
        self._model = compile_fn(
            self._model,
            mode=self._inference_compile_mode,
            dynamic=self._inference_compile_dynamic,
        )
        self._inference_compile_applied = True

    def reset(self) -> None:
        """Called once per inference episode to clear temporal state."""
        self._t = 0
        self._all_time_actions = None
        self._cached_actions = None
        self._last_temporal_aggregation_diagnostics = None
        self._last_factorized_diagnostics = None
        self._last_goal_effect_diagnostics = None
        visual_history = getattr(self, "_visual_history", None)
        if visual_history is not None:
            visual_history.reset()
        self._temporal_last_timestamps.clear()
        self._temporal_fallback_timestamp = 0
        self._last_temporal_input_diagnostics = None
        self._last_raw_action_chunk = None
        factorized_aggregator = getattr(self, "_factorized_aggregator", None)
        if factorized_aggregator is not None:
            factorized_aggregator.reset()
        condition_router = getattr(self, "_condition_phase_router", None)
        if condition_router is not None:
            condition_router.reset()
        self._last_condition_route_diagnostics = None

    def snapshot_state(self) -> ACTAdapterState:
        """Capture temporal inference state without sharing mutable storage."""

        visual_history = getattr(self, "_visual_history", None)
        factorized_aggregator = getattr(self, "_factorized_aggregator", None)
        return ACTAdapterState(
            step=int(self._t),
            all_time_actions=(
                None
                if self._all_time_actions is None
                else self._all_time_actions.detach().clone()
            ),
            temporal_weight_cache={
                int(key): value.detach().clone()
                for key, value in self._temporal_weight_cache.items()
            },
            cached_actions=(
                None
                if self._cached_actions is None
                else self._cached_actions.detach().clone()
            ),
            temporal_aggregation_diagnostics=deepcopy(
                self._last_temporal_aggregation_diagnostics
            ),
            factorized_diagnostics=deepcopy(self._last_factorized_diagnostics),
            goal_effect_diagnostics=deepcopy(self._last_goal_effect_diagnostics),
            temporal_input_diagnostics=deepcopy(self._last_temporal_input_diagnostics),
            temporal_last_timestamps={
                str(key): int(value)
                for key, value in self._temporal_last_timestamps.items()
            },
            temporal_fallback_timestamp=int(self._temporal_fallback_timestamp),
            visual_history_state=(
                None if visual_history is None else visual_history.snapshot_state()
            ),
            factorized_aggregator_state=(
                None
                if factorized_aggregator is None
                else factorized_aggregator.snapshot_state()
            ),
            condition_route=(
                None
                if getattr(self, "_condition_phase_router", None) is None
                else int(self._condition_phase_router.route)
            ),
            condition_route_consecutive=(
                None
                if getattr(self, "_condition_phase_router", None) is None
                else int(self._condition_phase_router.consecutive)
            ),
        )

    def restore_state(self, state: ACTAdapterState) -> None:
        """Restore a no-alias snapshot on this adapter's inference device."""

        if not isinstance(state, ACTAdapterState):
            raise TypeError("state must be ACTAdapterState")
        if int(state.step) < 0:
            raise ValueError("ACT adapter state step must be non-negative")
        self._t = int(state.step)
        self._all_time_actions = (
            None
            if state.all_time_actions is None
            else state.all_time_actions.detach().to(self.device).clone()
        )
        self._temporal_weight_cache = {
            int(key): value.detach().to(self.device).clone()
            for key, value in state.temporal_weight_cache.items()
        }
        self._cached_actions = (
            None
            if state.cached_actions is None
            else state.cached_actions.detach().to(self.device).clone()
        )
        self._last_temporal_aggregation_diagnostics = deepcopy(
            state.temporal_aggregation_diagnostics
        )
        self._last_factorized_diagnostics = deepcopy(state.factorized_diagnostics)
        self._last_goal_effect_diagnostics = deepcopy(state.goal_effect_diagnostics)
        self._last_temporal_input_diagnostics = deepcopy(
            state.temporal_input_diagnostics
        )
        self._temporal_last_timestamps = {
            str(key): int(value)
            for key, value in state.temporal_last_timestamps.items()
        }
        self._temporal_fallback_timestamp = int(state.temporal_fallback_timestamp)

        visual_history = getattr(self, "_visual_history", None)
        if (visual_history is None) != (state.visual_history_state is None):
            raise ValueError("ACT adapter visual history state/config mismatch")
        if visual_history is not None:
            visual_history.restore_state(state.visual_history_state)

        factorized_aggregator = getattr(self, "_factorized_aggregator", None)
        if (factorized_aggregator is None) != (
            state.factorized_aggregator_state is None
        ):
            raise ValueError("ACT adapter factorized state/config mismatch")
        if factorized_aggregator is not None:
            factorized_aggregator.restore_state(state.factorized_aggregator_state)
        condition_router = getattr(self, "_condition_phase_router", None)
        if (condition_router is None) != (state.condition_route is None):
            raise ValueError("ACT adapter condition router state/config mismatch")
        if condition_router is not None:
            if state.condition_route_consecutive is None:
                raise ValueError("ACT adapter condition router counter is missing")
            condition_router.route = int(state.condition_route)
            condition_router.consecutive = int(state.condition_route_consecutive)

    @property
    def camera_names(self) -> list[str]:
        return list(self._camera_names)

    @property
    def inference_precision(self) -> str:
        return self._inference_precision

    @property
    def inference_compile(self) -> bool:
        return self._inference_compile

    def last_raw_action_chunk(self) -> np.ndarray:
        """Return the latest normalized ACT chunk for parity diagnostics."""
        if self._last_raw_action_chunk is None:
            raise RuntimeError("no ACT inference has run since reset")
        return self._last_raw_action_chunk.cpu().numpy().copy()

    def last_raw_action_chunk_direct(self) -> np.ndarray:
        """Return the latest ACT chunk in frozen source-action units."""

        normalized = self.last_raw_action_chunk()
        action_mean = np.asarray(self.norm_stats["action_mean"], dtype=np.float32)
        action_std = np.asarray(self.norm_stats["action_std"], dtype=np.float32)
        if action_mean.shape != (4,) or action_std.shape != (4,):
            raise ValueError("ACT action normalization stats must have shape (4,)")
        if not np.isfinite(action_mean).all() or not np.isfinite(action_std).all():
            raise ValueError("ACT action normalization stats must be finite")
        return (
            normalized.astype(np.float32, copy=False) * action_std + action_mean
        ).astype(np.float32, copy=True)

    @property
    def factorized_diagnostics(self) -> dict[str, Any] | None:
        """Return JSON-safe diagnostics for the last opt-in factorized action."""

        return deepcopy(getattr(self, "_last_factorized_diagnostics", None))

    @property
    def goal_effect_diagnostics(self) -> dict[str, Any] | None:
        """Return the latest auxiliary forecast without changing the action."""

        return deepcopy(getattr(self, "_last_goal_effect_diagnostics", None))

    @property
    def temporal_input_diagnostics(self) -> dict[str, Any] | None:
        """Return diagnostics for the opt-in causal visual history input."""

        return deepcopy(getattr(self, "_last_temporal_input_diagnostics", None))

    @property
    def temporal_aggregation_diagnostics(self) -> dict[str, Any] | None:
        """Return the last opt-in aggregation decomposition in direct units."""

        return deepcopy(getattr(self, "_last_temporal_aggregation_diagnostics", None))

    @property
    def condition_route_diagnostics(self) -> dict[str, Any] | None:
        return deepcopy(getattr(self, "_last_condition_route_diagnostics", None))

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, obs: dict) -> np.ndarray:
        """
        Parameters
        ----------
        obs   dict with keys:
                "qpos"      : (Nq,) float32
                "qvel"      : (Nv,) float32 when configured in low_dim_keys
                "cycle_condition_v1": (6,) float32 when configured
                "image_<cam>": (C, H, W) float32 [0, 1]   for each camera
                "image_timestamp_ns": optional per-camera timestamp mapping;
                    a local causal step fallback is used when absent
              Camera images should be in channel-first format.

        Returns
        -------
        action : (Na,) float32  in *unnormalised* action space.
        """
        action, _ = self._predict_action_and_optional_intent(obs)
        return action

    def predict_action_and_intent(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return the executed action and query-0 axis-direction intent probabilities.

        This advances temporal aggregation exactly once, matching :meth:`predict`.
        Intent probabilities are ordered axis-major positive/negative.
        """
        if getattr(self, "_factorized_action", {}).get("enabled", False):
            raise ValueError(
                "factorized_intent_effort is a no-gate output and cannot be passed "
                "to runtime_gates"
            )
        action, intent_probabilities = self._predict_action_and_optional_intent(obs)
        if intent_probabilities is None:
            raise ValueError("loaded ACT policy has no intent logits")
        return action, intent_probabilities

    def _predict_action_and_optional_intent(
        self, obs: dict
    ) -> tuple[np.ndarray, np.ndarray | None]:
        condition_route = None
        condition_router = getattr(self, "_condition_phase_router", None)
        if condition_router is not None:
            if "qpos" not in obs or "qvel" not in obs:
                raise ValueError(
                    "phase-routed ACT requires qpos and qvel for causal routing"
                )
            condition_route = int(
                condition_router.step(obs["qpos"], obs["qvel"])
            )
            self._last_condition_route_diagnostics = {
                "schema": "simverify_condition_route_runtime_diagnostics_v1",
                "route": ("current", "neutral", "next")[condition_route],
                "route_index": condition_route,
                "consecutive_pending": int(condition_router.consecutive),
                "runtime_inputs": [
                    "current_qpos",
                    "current_qvel",
                    "past_router_state",
                ],
            }
        proprio = self._build_proprio(obs)

        # normalise low-dimensional robot state
        proprio = (proprio - self._proprio_mean) / self._proprio_std

        # Assemble image tensor in configured camera order. Ignore metadata
        # keys like `image_format` that may appear in live observations.
        raw_cam_images: list[np.ndarray] = []
        camera_keys: list[str] = []
        for cam in self._camera_names:
            key = f"image_{cam}"
            if key not in obs:
                raise ValueError(
                    f"ACTAdapter.predict(): missing required camera input {key!r}."
                )
            raw_cam_img = np.asarray(obs[key])
            if raw_cam_img.ndim != 3:
                raise ValueError(
                    f"ACTAdapter.predict(): expected {key!r} to be rank-3, "
                    f"got shape {raw_cam_img.shape}."
                )
            if raw_cam_img.shape[0] != 3 and raw_cam_img.shape[-1] != 3:
                raise ValueError(
                    f"ACTAdapter.predict(): expected {key!r} to have 3 channels, "
                    f"got shape {raw_cam_img.shape}."
                )
            raw_cam_images.append(raw_cam_img)
            camera_keys.append(key)

        if not raw_cam_images:
            raise ValueError("ACTAdapter.predict(): no camera inputs configured.")

        visual_history = getattr(self, "_visual_history", None)
        if visual_history is not None:
            cam_images = [
                _camera_image_to_chw_float(image, key=key)
                for image, key in zip(raw_cam_images, camera_keys)
            ]
            timestamp_map, timestamp_source = self._resolve_temporal_timestamps(obs)
            snapshot = visual_history.append(
                {
                    camera_name: image
                    for camera_name, image in zip(self._camera_names, cam_images)
                },
                timestamp_map,
            )
            # CausalVisualHistory stores each camera independently.  Reorder
            # its snapshots to (history, cameras, C, H, W), matching the
            # dataset sample contract.  The model owner consumes the batch
            # dimension added below; no previous command enters this path.
            img = np.ascontiguousarray(
                np.stack(
                    [snapshot.images[camera] for camera in self._camera_names],
                    axis=1,
                )
            )
            self._last_temporal_input_diagnostics = {
                "enabled": True,
                "history_steps": int(self._temporal_input["history_steps"]),
                "timestamp_source": timestamp_source,
                "valid_mask": {
                    camera: snapshot.valid_mask[camera].astype(bool).tolist()
                    for camera in self._camera_names
                },
                "timestamps_ns": {
                    camera: snapshot.timestamps_ns[camera].astype(np.int64).tolist()
                    for camera in self._camera_names
                },
                "accepted": dict(snapshot.accepted),
                "duplicate_timestamp": dict(snapshot.duplicate_timestamp),
            }
            image = torch.from_numpy(img).float().to(self.device).unsqueeze(0)
            # (1, history, n_cams, C, H, W)
        else:
            self._last_temporal_input_diagnostics = {
                "enabled": False,
                "history_steps": 1,
            }
            image = _single_frame_image_tensor(
                raw_cam_images,
                keys=camera_keys,
                device=self.device,
                device_uint8_preprocess=self._device_uint8_preprocess,
            )
        image = self._normalize(image)

        if self._model.training:
            self._model.eval()
        a_hat, intent_logits = self._forward_inference(
            proprio,
            image,
            condition_route=condition_route,
        )
        if intent_logits is not None and (
            intent_logits.ndim != 3
            or intent_logits.shape[0] != 1
            or intent_logits.shape[2] != 8
        ):
            raise ValueError(
                "ACT intent logits must have shape (1, num_queries, 8), "
                f"got {tuple(intent_logits.shape)}"
            )

        action_is_direct = False
        if getattr(self, "_factorized_action", {}).get("enabled", False):
            if not self.temporal_agg:
                raise ValueError(
                    "factorized_intent_effort inference requires temporal_agg=true"
                )
            if intent_logits is None:
                raise ValueError(
                    "factorized_intent_effort requires model intent logits"
                )
            if self._factorized_aggregator is None:
                raise RuntimeError("factorized temporal aggregator is not initialized")
            tri_probabilities, effort, signed_direct = query_factorized_values(
                policy_normalized=a_hat,
                intent_logits=intent_logits,
                norm_stats=self.norm_stats,
            )
            aggregated = self._factorized_aggregator.aggregate(
                probabilities=tri_probabilities,
                effort=effort,
                legacy_signed_action=signed_direct,
            )
            projected, projection_diagnostics = project_factorized_action(
                probabilities=aggregated.probabilities,
                effort=aggregated.effort,
                config=self._factorized_action,
            )
            action = projected.detach().cpu().numpy()
            action_is_direct = True
            diagnostics = {
                **aggregated.diagnostics,
                **projection_diagnostics,
                "legacy_signed_act_aggregate": [
                    float(value)
                    for value in aggregated.legacy_signed_action.detach().cpu()
                ],
            }
            self._last_factorized_diagnostics = {
                f"policy_factorized_{key}": value for key, value in diagnostics.items()
            }
        elif self.temporal_agg:
            action = self._aggregate(a_hat)
        else:
            # non-aggregated: execute every num_queries steps
            if self._t % self._num_queries == 0:
                self._cached_actions = a_hat.squeeze(0)  # (C, Na)
            step_in_chunk = self._t % self._num_queries
            action = self._cached_actions[step_in_chunk].cpu().numpy()

        self._t += 1

        # unnormalise
        if not action_is_direct:
            action = (
                action * self.norm_stats["action_std"] + self.norm_stats["action_mean"]
            )
        intent_prob = (
            None
            if intent_logits is None
            else torch.sigmoid(intent_logits[0, 0]).detach().cpu().numpy()
        )
        return (
            action.astype(np.float32),
            None if intent_prob is None else np.asarray(intent_prob, dtype=np.float32),
        )

    def _forward_inference(
        self,
        proprio: torch.Tensor,
        image: torch.Tensor,
        *,
        condition_route: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run only the neural-network forward in the requested precision."""
        inference_autocast_dtype = getattr(
            self, "_inference_autocast_dtype", None
        )
        autocast_context = (
            nullcontext()
            if inference_autocast_dtype is None
            else torch.autocast(
                device_type=self.device.type,
                dtype=inference_autocast_dtype,
                enabled=True,
            )
        )
        with torch.inference_mode(), autocast_context:
            (
                a_hat,
                _,
                _,
                intent_logits,
                goal_effect_outputs,
                _action_state_logits,
                _effective_action_phase_logits,
            ) = self._unpack_model_output(
                (
                    self._model(proprio, image, None)
                    if condition_route is None
                    else self._model(
                        proprio,
                        image,
                        None,
                        condition_route=torch.as_tensor(
                            [condition_route],
                            dtype=torch.int64,
                            device=proprio.device,
                        ),
                    )
                )
            )

        # Aggregation, diagnostics, sigmoid, and CPU conversion stay in FP32.
        a_hat = a_hat.float()
        if intent_logits is not None:
            intent_logits = intent_logits.float()
        self._last_raw_action_chunk = a_hat[0].detach()
        self._last_goal_effect_diagnostics = _goal_effect_diagnostics(
            goal_effect_outputs
        )
        return a_hat, intent_logits

    def _resolve_temporal_timestamps(
        self, obs: dict[str, Any]
    ) -> tuple[dict[str, int], str]:
        """Resolve per-camera causal timestamps without changing old inputs.

        Real observations normally provide ``image_timestamp_ns`` and/or a
        synchronized timestamp.  Offline HDF5 replay does not, so a strictly
        increasing local counter is used only for the temporal experiment.
        Equal real timestamps remain equal (and are handled as duplicates by
        :class:`CausalVisualHistory`); an older timestamp is intentionally
        rejected by the helper rather than silently reordering frames.
        """

        raw = obs.get("image_timestamp_ns")
        sync = _coerce_timestamp(obs.get("sync_timestamp_ns"))
        if sync is None:
            sync = _coerce_timestamp(obs.get("timestamp_ns"))
        raw_mapping = raw if isinstance(raw, dict) else {}
        raw_scalar = _coerce_timestamp(raw) if not isinstance(raw, dict) else None

        self._temporal_fallback_timestamp += 1
        timestamps: dict[str, int] = {}
        used_fallback = False
        used_observation_timestamp = False
        for camera_name in self._camera_names:
            candidate = _coerce_timestamp(raw_mapping.get(camera_name))
            if candidate is None:
                candidate = raw_scalar
            if candidate is None:
                candidate = sync
            if candidate is None:
                candidate = self._temporal_fallback_timestamp
                used_fallback = True
            else:
                used_observation_timestamp = True

            last = self._temporal_last_timestamps.get(camera_name)
            if last is not None and candidate < last:
                # Let CausalVisualHistory raise the same causal-order error,
                # but preserve a useful local diagnostic source.
                timestamps[camera_name] = int(candidate)
            else:
                timestamps[camera_name] = int(candidate)
                self._temporal_last_timestamps[camera_name] = int(candidate)
            self._temporal_fallback_timestamp = max(
                self._temporal_fallback_timestamp, int(candidate)
            )

        if used_observation_timestamp and used_fallback:
            source = "observation_plus_local_fallback"
        elif used_observation_timestamp:
            source = "observation_timestamp"
        else:
            source = "local_step_fallback"
        return timestamps, source

    def _build_proprio(self, obs: dict) -> torch.Tensor:
        parts: list[np.ndarray] = []
        for key in self._low_dim_keys:
            if key not in obs:
                raise ValueError(
                    f"ACTAdapter.predict(): missing required low-dimensional input {key!r}."
                )
            value = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            parts.append(value)
        if not parts:
            raise ValueError("ACTAdapter.predict(): low_dim_keys must not be empty.")
        proprio = np.concatenate(parts, axis=0).astype(np.float32)
        return torch.from_numpy(proprio).float().to(self.device).unsqueeze(0)

    def _resolve_proprio_norm_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        if "proprio_mean" in self.norm_stats and "proprio_std" in self.norm_stats:
            mean = self.norm_stats["proprio_mean"]
            std = self.norm_stats["proprio_std"]
        elif self._low_dim_keys == ["qpos"]:
            # Backward compatibility for older qpos-only checkpoints.
            mean = self.norm_stats["qpos_mean"]
            std = self.norm_stats["qpos_std"]
        else:
            raise KeyError(
                "dataset_stats.pkl does not contain proprio_mean/proprio_std for "
                f"low_dim_keys={self._low_dim_keys}. Recompute stats by retraining "
                "with the updated data pipeline."
            )
        return (
            torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.asarray(std, dtype=np.float32)).to(self.device),
        )

    def _aggregate(self, a_hat: torch.Tensor) -> np.ndarray:
        """
        Temporal aggregation from the ACT paper.
        a_hat shape: (1, C, Na)
        """
        Na = a_hat.shape[-1]

        if self._all_time_actions is None:
            horizon = max(self._max_episode_len, self._t + self._num_queries)
            self._all_time_actions = torch.zeros(
                [horizon, horizon + self._num_queries, Na], device=self.device
            )

        required_rows = self._t + 1
        required_cols = self._t + self._num_queries
        if (
            required_rows > self._all_time_actions.shape[0]
            or required_cols > self._all_time_actions.shape[1]
        ):
            current_rows = self._all_time_actions.shape[0]
            new_rows = max(required_rows, current_rows * 2)
            expanded = torch.zeros(
                [new_rows, new_rows + self._num_queries, Na],
                device=self.device,
            )
            expanded[
                : self._all_time_actions.shape[0], : self._all_time_actions.shape[1]
            ] = self._all_time_actions
            self._all_time_actions = expanded

        t = self._t
        self._all_time_actions[[t], t : t + self._num_queries] = a_hat

        # weighted average of all past chunks that cover step t
        # NOTE: only rows whose chunk actually covers column t are non-zero;
        #       filter them out exactly as in the original ACT repo to avoid
        #       zero-padding contaminating the weighted mean.
        actions_for_curr_step = self._all_time_actions[: t + 1, t]  # (t+1, Na)
        actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        num_actions = int(len(actions_for_curr_step))
        exp_weights = self._temporal_weight_cache.get(num_actions)
        if exp_weights is None:
            exp_weights = torch.exp(
                -_TEMPORAL_AGG_K
                * torch.arange(num_actions, dtype=torch.float32, device=self.device)
            )
            exp_weights = (exp_weights / exp_weights.sum()).unsqueeze(1)
            self._temporal_weight_cache[num_actions] = exp_weights
        action = (actions_for_curr_step * exp_weights).sum(0).cpu().numpy()
        if getattr(self, "_temporal_aggregation_diagnostics_enabled", False):
            recency_weights = torch.flip(exp_weights, dims=(0,))
            recency_action = (
                (actions_for_curr_step * recency_weights).sum(0).cpu().numpy()
            )
            newest_action = a_hat[0, 0].detach().cpu().numpy()
            action_mean = np.asarray(self.norm_stats["action_mean"])
            action_std = np.asarray(self.norm_stats["action_std"])

            def to_direct(normalized: np.ndarray) -> list[float]:
                direct = np.asarray(normalized) * action_std + action_mean
                direct = direct.astype(np.float32)
                return [float(value) for value in direct]

            source_steps = torch.nonzero(actions_populated, as_tuple=False).flatten()
            self._last_temporal_aggregation_diagnostics = {
                "policy_temporal_aggregation_action_domain": "direct_policy_output",
                "policy_temporal_aggregation_exponential_k": float(_TEMPORAL_AGG_K),
                "policy_temporal_aggregation_query_step": int(t),
                "policy_temporal_aggregation_source_steps": [
                    int(value) for value in source_steps.detach().cpu().tolist()
                ],
                "policy_temporal_aggregation_query_offsets": [
                    int(t - value) for value in source_steps.detach().cpu().tolist()
                ],
                "policy_temporal_aggregation_population": num_actions,
                "policy_temporal_aggregation_legacy_action": to_direct(action),
                "policy_temporal_aggregation_newest_action": to_direct(newest_action),
                "policy_temporal_aggregation_recency_action": to_direct(recency_action),
            }
        return action

    # ── training forward ──────────────────────────────────────────────────────

    def forward_loss(
        self,
        proprio: torch.Tensor,
        image: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        *,
        raw_action: torch.Tensor | None = None,
        deadzone_move_mask: torch.Tensor | None = None,
        deadzone_stop_mask: torch.Tensor | None = None,
        deadzone_wrong_mask: torch.Tensor | None = None,
        action_loss_mask: torch.Tensor | None = None,
        state_hold_transition_mask: torch.Tensor | None = None,
        goal_future_delta: torch.Tensor | None = None,
        goal_future_valid: torch.Tensor | None = None,
        goal_future_direction: torch.Tensor | None = None,
        goal_effect_delta: torch.Tensor | None = None,
        goal_effect_valid: torch.Tensor | None = None,
        action_state_labels: torch.Tensor | None = None,
        action_state_valid: torch.Tensor | None = None,
        action_state_persistent_effective: torch.Tensor | None = None,
        effective_action_phase: torch.Tensor | None = None,
        effective_action_valid: torch.Tensor | None = None,
        effective_action_loss_weight: torch.Tensor | None = None,
        counterfactual_proprio: torch.Tensor | None = None,
        counterfactual_consistency_eligible: torch.Tensor | None = None,
        condition_route: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training-time forward pass.

        Parameters
        ----------
        proprio (B, Np)
        image   (B, n_cams, C, H, W)   normalised
        actions (B, C, Na)
        is_pad  (B, C)  bool

        Returns
        -------
        {"l1": ..., "kl": ..., "loss": ...}
        """
        image = self._normalize(image)
        actions = actions[:, : self._model.num_queries]
        if raw_action is not None:
            raw_action = raw_action[:, : self._model.num_queries]
        is_pad = is_pad[:, : self._model.num_queries]
        if action_loss_mask is not None:
            action_loss_mask = action_loss_mask[:, : self._model.num_queries]
        if deadzone_move_mask is not None:
            deadzone_move_mask = deadzone_move_mask[:, : self._model.num_queries]
        if deadzone_stop_mask is not None:
            deadzone_stop_mask = deadzone_stop_mask[:, : self._model.num_queries]
        if deadzone_wrong_mask is not None:
            deadzone_wrong_mask = deadzone_wrong_mask[:, : self._model.num_queries]
        if action_state_labels is not None:
            action_state_labels = action_state_labels[:, : self._model.num_queries]
        if action_state_valid is not None:
            action_state_valid = action_state_valid[:, : self._model.num_queries]
        if action_state_persistent_effective is not None:
            action_state_persistent_effective = action_state_persistent_effective[
                :, : self._model.num_queries
            ]
        if effective_action_phase is not None:
            effective_action_phase = effective_action_phase[
                :, : self._model.num_queries
            ]
        if effective_action_valid is not None:
            effective_action_valid = effective_action_valid[
                :, : self._model.num_queries
            ]
        if effective_action_loss_weight is not None:
            effective_action_loss_weight = effective_action_loss_weight[
                :, : self._model.num_queries
            ]

        (
            a_hat,
            _,
            (mu, logvar),
            intent_logits,
            goal_effect_outputs,
            action_state_logits,
            effective_action_phase_logits,
        ) = self._unpack_model_output(
            (
                self._model(proprio, image, None, actions, is_pad)
                if condition_route is None
                else self._model(
                    proprio,
                    image,
                    None,
                    actions,
                    is_pad,
                    condition_route=condition_route,
                )
            )
        )
        total_kld, _, _ = _kl_divergence(mu, logvar)

        valid_mask = ~is_pad.unsqueeze(-1)
        effective_action_config = getattr(
            self,
            "_effective_action",
            resolve_effective_action_config({"enabled": False}),
        )
        factorized_loss_d = factorized_training_loss_terms(
            expert_normalized=actions,
            policy_normalized=a_hat,
            intent_logits=intent_logits,
            valid_mask=valid_mask,
            norm_stats=self.norm_stats,
            config=self._factorized_action,
            transition_mask=state_hold_transition_mask,
            action_loss_mask=action_loss_mask,
        )
        effective_action_loss_d = effective_action_loss_terms(
            target_normalized=actions,
            policy_normalized=a_hat,
            phase_logits=effective_action_phase_logits,
            phase_labels=effective_action_phase,
            phase_valid=effective_action_valid,
            valid_mask=valid_mask,
            loss_weight=effective_action_loss_weight,
            action_mean=self.norm_stats["action_mean"],
            action_std=self.norm_stats["action_std"],
            config=effective_action_config,
        )
        raw_l1 = a_hat.new_zeros(())
        if self._factorized_action["enabled"]:
            l1 = factorized_loss_d["factorized_magnitude_l1"]
            imitation_loss = factorized_loss_d["factorized_loss"]
        elif effective_action_config["enabled"]:
            l1 = weighted_action_l1(
                expert=actions,
                policy=a_hat,
                valid_mask=valid_mask,
                loss_weight=effective_action_loss_weight,
                action_loss_mask=action_loss_mask,
            )
            if (
                raw_action is None
                and float(effective_action_config["raw_continuity_weight"]) > 0.0
            ):
                raise ValueError(
                    "effective_action raw_continuity_weight requires raw_action targets"
                )
            raw_l1 = (
                _masked_action_l1(
                    expert=raw_action,
                    policy=a_hat,
                    valid_mask=valid_mask,
                    action_loss_mask=action_loss_mask,
                )
                if raw_action is not None
                else a_hat.new_zeros(())
            )
            imitation_loss = (
                l1 + float(effective_action_config["raw_continuity_weight"]) * raw_l1
            )
        else:
            l1 = _masked_action_l1(
                expert=actions,
                policy=a_hat,
                valid_mask=valid_mask,
                action_loss_mask=action_loss_mask,
            )
            imitation_loss = l1
        deadzone_loss_d = self._deadzone_loss_terms(
            expert_normalized=actions,
            policy_normalized=a_hat,
            valid_mask=valid_mask,
        )
        window_deadzone_loss_d = self._window_deadzone_loss_terms(
            expert_normalized=actions,
            policy_normalized=a_hat,
            valid_mask=valid_mask,
            move_mask=deadzone_move_mask,
            stop_mask=deadzone_stop_mask,
            wrong_mask=deadzone_wrong_mask,
        )
        intent_loss_d = self._intent_loss_terms(
            expert_normalized=actions,
            intent_logits=intent_logits,
            valid_mask=valid_mask,
        )
        temporal_release_loss_d = self._temporal_release_loss_terms(
            expert_normalized=actions,
            policy_normalized=a_hat,
            valid_mask=valid_mask,
        )
        demo_target_hold_loss_d = self._demo_target_hold_loss_terms(
            policy_normalized=a_hat,
            transition_mask=state_hold_transition_mask,
        )
        goal_effect_targets = None
        if any(
            value is not None
            for value in (
                goal_future_delta,
                goal_future_valid,
                goal_future_direction,
                goal_effect_delta,
                goal_effect_valid,
            )
        ):
            goal_effect_targets = {
                "goal_future_delta": goal_future_delta,
                "goal_future_valid": goal_future_valid,
                "goal_future_direction": goal_future_direction,
                "goal_effect_delta": goal_effect_delta,
                "goal_effect_valid": goal_effect_valid,
            }
        goal_effect_loss_d = goal_effect_loss_terms(
            outputs=goal_effect_outputs,
            targets=goal_effect_targets,
            config=getattr(
                self,
                "_goal_effect",
                resolve_goal_effect_config({"enabled": False}),
            ),
            device_like=a_hat,
        )
        action_mean = torch.as_tensor(
            self.norm_stats["action_mean"],
            dtype=a_hat.dtype,
            device=a_hat.device,
        )
        action_std = torch.as_tensor(
            self.norm_stats["action_std"],
            dtype=a_hat.dtype,
            device=a_hat.device,
        )
        action_state_loss_d = action_state_loss_terms(
            policy_direct=a_hat * action_std + action_mean,
            state_logits=action_state_logits,
            state_labels=action_state_labels,
            state_valid=action_state_valid,
            persistent_effective=action_state_persistent_effective,
            config=getattr(
                self,
                "_action_state_effort",
                resolve_action_state_effort_config({"enabled": False}),
            ),
        )
        consistency_loss_d = self._condition_counterfactual_consistency_terms(
            proprio=proprio,
            image=image,
            counterfactual_proprio=counterfactual_proprio,
            eligible_mask=counterfactual_consistency_eligible,
            device_like=a_hat,
        )

        return {
            "l1": l1,
            "effective_action_raw_l1": (
                raw_l1 if effective_action_config["enabled"] else a_hat.new_zeros(())
            ),
            "kl": total_kld[0],
            **deadzone_loss_d,
            **window_deadzone_loss_d,
            **intent_loss_d,
            **temporal_release_loss_d,
            **demo_target_hold_loss_d,
            **goal_effect_loss_d,
            **action_state_loss_d,
            **effective_action_loss_d,
            **factorized_loss_d,
            **consistency_loss_d,
            "loss": (
                imitation_loss
                + total_kld[0] * self.kl_weight
                + deadzone_loss_d["deadzone_loss"]
                + window_deadzone_loss_d["window_deadzone_loss"]
                + intent_loss_d["intent_loss"]
                + temporal_release_loss_d["temporal_release_loss"]
                + demo_target_hold_loss_d["demo_target_hold_loss"]
                + goal_effect_loss_d["goal_effect_loss"]
                + action_state_loss_d["action_state_loss"]
                + effective_action_loss_d["effective_action_loss"]
                + consistency_loss_d["condition_counterfactual_consistency_loss"]
            ),
        }

    def _condition_counterfactual_consistency_terms(
        self,
        *,
        proprio: torch.Tensor,
        image: torch.Tensor,
        counterfactual_proprio: torch.Tensor | None,
        eligible_mask: torch.Tensor | None,
        device_like: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = device_like.new_zeros(())
        result = {
            "condition_counterfactual_consistency_raw_l1": zero,
            "condition_counterfactual_consistency_loss": zero,
            "condition_counterfactual_consistency_eligible_count": zero,
        }
        cfg = getattr(
            self,
            "_condition_counterfactual_consistency",
            _resolve_condition_counterfactual_consistency_config(None),
        )
        if not cfg["enabled"]:
            return result
        if (counterfactual_proprio is None) != (eligible_mask is None):
            raise ValueError(
                "counterfactual proprio and eligibility must be supplied together"
            )
        if counterfactual_proprio is None:
            return result

        eligible = eligible_mask.to(
            device=proprio.device,
            dtype=torch.bool,
        ).reshape(-1)
        if eligible.shape[0] != proprio.shape[0]:
            raise ValueError(
                "counterfactual eligibility batch does not match proprio"
            )
        if tuple(counterfactual_proprio.shape) != tuple(proprio.shape):
            raise ValueError(
                "counterfactual proprio must match primary proprio shape"
            )
        result["condition_counterfactual_consistency_eligible_count"] = (
            eligible.sum().to(dtype=device_like.dtype)
        )
        if not bool(eligible.any()):
            return result

        primary_proprio = proprio[eligible]
        paired_proprio = torch.cat(
            (primary_proprio, counterfactual_proprio[eligible]),
            dim=0,
        )
        paired_image = torch.cat((image[eligible], image[eligible]), dim=0)
        model_was_training = self._model.training
        self._model.eval()
        try:
            paired_a_hat = self._unpack_model_output(
                self._model(paired_proprio, paired_image, None, None, None)
            )[0]
        finally:
            self._model.train(model_was_training)
        pair_count = primary_proprio.shape[0]
        raw_l1 = (
            paired_a_hat[:pair_count] - paired_a_hat[pair_count:]
        ).abs().mean()
        result["condition_counterfactual_consistency_raw_l1"] = raw_l1
        result["condition_counterfactual_consistency_loss"] = (
            raw_l1 * cfg["coefficient"]
        )
        return result

    @staticmethod
    def _unpack_model_output(output):
        a_hat, is_pad_hat, latent = output[:3]
        intent_logits = output[3] if len(output) > 3 else None
        goal_effect_outputs = output[4] if len(output) > 4 else None
        action_state_logits = output[5] if len(output) > 5 else None
        effective_action_phase_logits = output[6] if len(output) > 6 else None
        return (
            a_hat,
            is_pad_hat,
            latent,
            intent_logits,
            goal_effect_outputs,
            action_state_logits,
            effective_action_phase_logits,
        )

    def _deadzone_loss_terms(
        self,
        *,
        expert_normalized: torch.Tensor,
        policy_normalized: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self._deadzone_loss
        zero = policy_normalized.new_zeros(())
        if not cfg["enabled"]:
            return {
                "deadzone_same_dir_loss": zero,
                "deadzone_idle_loss": zero,
                "deadzone_wrong_loss": zero,
                "deadzone_loss": zero,
            }

        action_mean = torch.as_tensor(
            self.norm_stats["action_mean"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        action_std = torch.as_tensor(
            self.norm_stats["action_std"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        expert = expert_normalized * action_std + action_mean
        policy = policy_normalized * action_std + action_mean

        pos = cfg["pos"].to(device=policy.device, dtype=policy.dtype)
        neg = cfg["neg"].to(device=policy.device, dtype=policy.dtype)
        margin = cfg["margin"].to(device=policy.device, dtype=policy.dtype)
        valid = valid_mask.to(dtype=torch.bool).expand_as(policy)

        expert_pos = (expert >= pos) & valid
        expert_neg = (expert <= -neg) & valid
        expert_axis_effective = expert_pos | expert_neg
        expert_step_effective = expert_axis_effective.any(dim=-1, keepdim=True)
        if cfg["same_dir_window"] == "expert_transition_window":
            same_dir_step_mask = _expert_transition_window_mask(
                expert_step_effective=expert_step_effective,
                valid_step=valid.any(dim=-1, keepdim=True),
                window_steps=int(cfg["same_dir_window_steps"]),
            )
            expert_pos = expert_pos & same_dir_step_mask
            expert_neg = expert_neg & same_dir_step_mask
        same_pos_shortfall = torch.relu(pos + margin - policy) * expert_pos.to(
            policy.dtype
        )
        same_neg_shortfall = torch.relu(policy + neg + margin) * expert_neg.to(
            policy.dtype
        )
        same_dir_count = (expert_pos | expert_neg).to(policy.dtype).sum().clamp_min(1.0)
        same_dir_loss = (same_pos_shortfall + same_neg_shortfall).sum() / same_dir_count

        policy_pos_effective = policy >= pos
        policy_neg_effective = policy <= -neg

        idle_axes = valid & ~expert_step_effective
        idle_pos_excess = torch.relu(policy - pos) * (
            idle_axes & policy_pos_effective
        ).to(policy.dtype)
        idle_neg_excess = torch.relu(-neg - policy) * (
            idle_axes & policy_neg_effective
        ).to(policy.dtype)
        if cfg["idle_denominator"] == "all_idle_axes":
            idle_count = idle_axes.to(policy.dtype).sum().clamp_min(1.0)
        else:
            idle_count = (
                (idle_axes & (policy_pos_effective | policy_neg_effective))
                .to(policy.dtype)
                .sum()
                .clamp_min(1.0)
            )
        idle_loss = (idle_pos_excess + idle_neg_excess).sum() / idle_count

        wrong_loss = zero
        if cfg["penalize_wrong_effective"]:
            not_expert_pos = (~expert_pos) & valid & expert_step_effective
            not_expert_neg = (~expert_neg) & valid & expert_step_effective
            wrong_pos_mask = not_expert_pos & policy_pos_effective
            wrong_neg_mask = not_expert_neg & policy_neg_effective
            wrong_pos_excess = torch.relu(policy - pos) * wrong_pos_mask.to(
                policy.dtype
            )
            wrong_neg_excess = torch.relu(-neg - policy) * wrong_neg_mask.to(
                policy.dtype
            )
            if cfg["wrong_denominator"] == "all_wrong_candidate_axes":
                wrong_count = (
                    not_expert_pos.to(policy.dtype).sum()
                    + not_expert_neg.to(policy.dtype).sum()
                ).clamp_min(1.0)
            else:
                wrong_count = (
                    (wrong_pos_mask | wrong_neg_mask)
                    .to(policy.dtype)
                    .sum()
                    .clamp_min(1.0)
                )
            wrong_loss = (wrong_pos_excess + wrong_neg_excess).sum() / wrong_count

        total = (
            float(cfg["same_dir_promote_weight"]) * same_dir_loss
            + float(cfg["idle_suppression_weight"]) * idle_loss
            + float(cfg["wrong_effective_weight"]) * wrong_loss
        )
        return {
            "deadzone_same_dir_loss": same_dir_loss,
            "deadzone_idle_loss": idle_loss,
            "deadzone_wrong_loss": wrong_loss,
            "deadzone_loss": total,
        }

    def _window_deadzone_loss_terms(
        self,
        *,
        expert_normalized: torch.Tensor,
        policy_normalized: torch.Tensor,
        valid_mask: torch.Tensor,
        move_mask: torch.Tensor | None,
        stop_mask: torch.Tensor | None,
        wrong_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        cfg = getattr(self, "_window_deadzone_loss", {"enabled": False})
        zero = policy_normalized.new_zeros(())
        if not cfg.get("enabled", False):
            return {
                "window_deadzone_same_dir_loss": zero,
                "window_deadzone_stop_loss": zero,
                "window_deadzone_wrong_loss": zero,
                "window_deadzone_loss": zero,
            }
        if move_mask is None or stop_mask is None or wrong_mask is None:
            raise ValueError(
                "window deadzone loss requires move, stop, and wrong masks"
            )

        action_mean = torch.as_tensor(
            self.norm_stats["action_mean"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        action_std = torch.as_tensor(
            self.norm_stats["action_std"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        policy = policy_normalized * action_std + action_mean

        pos = cfg["pos"].to(device=policy.device, dtype=policy.dtype)
        neg = cfg["neg"].to(device=policy.device, dtype=policy.dtype)
        margin = cfg["margin"].to(device=policy.device, dtype=policy.dtype)
        valid = valid_mask.to(dtype=torch.bool).expand_as(policy)
        valid_dir = torch.stack([valid, valid], dim=-1)

        move = move_mask.to(device=policy.device, dtype=torch.bool) & valid_dir
        move_pos = move[..., 0]
        move_neg = move[..., 1]
        same_pos_shortfall = torch.relu(pos + margin - policy) * move_pos.to(
            policy.dtype
        )
        same_neg_shortfall = torch.relu(policy + neg + margin) * move_neg.to(
            policy.dtype
        )
        same_count = (move_pos | move_neg).to(policy.dtype).sum().clamp_min(1.0)
        same_dir_loss = (same_pos_shortfall + same_neg_shortfall).sum() / same_count

        policy_pos_effective = policy >= pos
        policy_neg_effective = policy <= -neg
        stop_axes = (
            stop_mask.to(device=policy.device, dtype=torch.bool).unsqueeze(-1) & valid
        )
        stop_pos_mask = stop_axes & policy_pos_effective
        stop_neg_mask = stop_axes & policy_neg_effective
        stop_pos_excess = torch.relu(policy - pos) * stop_pos_mask.to(policy.dtype)
        stop_neg_excess = torch.relu(-neg - policy) * stop_neg_mask.to(policy.dtype)
        stop_count = (
            (stop_pos_mask | stop_neg_mask).to(policy.dtype).sum().clamp_min(1.0)
        )
        stop_loss = (stop_pos_excess + stop_neg_excess).sum() / stop_count

        wrong = wrong_mask.to(device=policy.device, dtype=torch.bool) & valid_dir
        wrong_pos_mask = wrong[..., 0] & policy_pos_effective
        wrong_neg_mask = wrong[..., 1] & policy_neg_effective
        wrong_pos_excess = torch.relu(policy - pos) * wrong_pos_mask.to(policy.dtype)
        wrong_neg_excess = torch.relu(-neg - policy) * wrong_neg_mask.to(policy.dtype)
        wrong_count = (
            (wrong_pos_mask | wrong_neg_mask).to(policy.dtype).sum().clamp_min(1.0)
        )
        wrong_loss = (wrong_pos_excess + wrong_neg_excess).sum() / wrong_count

        total = (
            float(cfg["same_dir_promote_weight"]) * same_dir_loss
            + float(cfg["stop_suppression_weight"]) * stop_loss
            + float(cfg["wrong_effective_weight"]) * wrong_loss
        )
        return {
            "window_deadzone_same_dir_loss": same_dir_loss,
            "window_deadzone_stop_loss": stop_loss,
            "window_deadzone_wrong_loss": wrong_loss,
            "window_deadzone_loss": total,
        }

    def _temporal_release_loss_terms(
        self,
        *,
        expert_normalized: torch.Tensor,
        policy_normalized: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = getattr(self, "_temporal_release_loss", {"enabled": False})
        zero = policy_normalized.new_zeros(())
        if not cfg.get("enabled", False):
            return {
                "temporal_release_pos_loss": zero,
                "temporal_release_neg_loss": zero,
                "temporal_release_loss": zero,
            }

        action_mean = torch.as_tensor(
            self.norm_stats["action_mean"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        action_std = torch.as_tensor(
            self.norm_stats["action_std"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        expert = expert_normalized * action_std + action_mean
        policy = policy_normalized * action_std + action_mean
        pos = cfg["pos"].to(device=policy.device, dtype=policy.dtype)
        neg = cfg["neg"].to(device=policy.device, dtype=policy.dtype)
        valid = valid_mask.to(dtype=torch.bool).expand_as(policy)

        expert_pos = (expert >= pos) & valid
        expert_neg = (expert <= -neg) & valid
        release_pos = _direction_release_window_mask(
            expert_effective=expert_pos,
            valid=valid,
            window_steps=int(cfg["release_window_steps"]),
        )
        release_neg = _direction_release_window_mask(
            expert_effective=expert_neg,
            valid=valid,
            window_steps=int(cfg["release_window_steps"]),
        )
        policy_pos_effective = policy >= pos
        policy_neg_effective = policy <= -neg
        pos_excess = torch.relu(policy - pos) * (release_pos & policy_pos_effective).to(
            policy.dtype
        )
        neg_excess = torch.relu(-neg - policy) * (
            release_neg & policy_neg_effective
        ).to(policy.dtype)
        pos_count = (
            (release_pos & policy_pos_effective).to(policy.dtype).sum().clamp_min(1.0)
        )
        neg_count = (
            (release_neg & policy_neg_effective).to(policy.dtype).sum().clamp_min(1.0)
        )
        pos_loss = pos_excess.sum() / pos_count
        neg_loss = neg_excess.sum() / neg_count
        total = float(cfg["weight"]) * (pos_loss + neg_loss)
        return {
            "temporal_release_pos_loss": pos_loss,
            "temporal_release_neg_loss": neg_loss,
            "temporal_release_loss": total,
        }

    def _demo_target_hold_loss_terms(
        self,
        *,
        policy_normalized: torch.Tensor,
        transition_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Penalize held-prefix actions that never reach the assist trigger."""

        cfg = getattr(self, "_demo_target_hold_loss", {"enabled": False})
        zero = policy_normalized.new_zeros(())
        if not cfg.get("enabled", False):
            return {
                "state_hold_pos_shortfall_loss": zero,
                "state_hold_neg_shortfall_loss": zero,
                "demo_target_hold_loss": zero,
            }
        if transition_mask is None:
            raise ValueError(
                "demo_target_hold_loss requires state_hold_transition_mask"
            )
        expected_shape = (policy_normalized.shape[0], len(_AXIS_NAMES), 2)
        if tuple(transition_mask.shape) != expected_shape:
            raise ValueError(
                "state_hold_transition_mask must have shape "
                f"{expected_shape}, got {tuple(transition_mask.shape)}"
            )
        transition = transition_mask.to(
            device=policy_normalized.device,
            dtype=torch.bool,
        )
        if torch.any(transition[..., 0] & transition[..., 1]):
            raise ValueError(
                "state_hold_transition_mask cannot select both directions of one axis"
            )

        action_mean = torch.as_tensor(
            self.norm_stats["action_mean"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        action_std = torch.as_tensor(
            self.norm_stats["action_std"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        policy = policy_normalized * action_std + action_mean
        held = _held_temporal_prefix_actions(
            policy,
            hold_horizon_steps=int(cfg["hold_horizon_steps"]),
        )
        consecutive = int(cfg["min_consecutive_steps"])
        held_tail = held[:, -consecutive:]

        fraction = float(cfg["assist_trigger_fraction"])
        margin = cfg["margin"].to(device=held.device, dtype=held.dtype)
        pos_target = (
            fraction * cfg["pos"].to(device=held.device, dtype=held.dtype) + margin
        )
        neg_target = (
            fraction * cfg["neg"].to(device=held.device, dtype=held.dtype) + margin
        )
        pos_mask = transition[..., 0].unsqueeze(1).expand(-1, consecutive, -1)
        neg_mask = transition[..., 1].unsqueeze(1).expand(-1, consecutive, -1)
        pos_shortfall = torch.relu(pos_target - held_tail) * pos_mask.to(held.dtype)
        neg_shortfall = torch.relu(neg_target + held_tail) * neg_mask.to(held.dtype)
        pos_count = pos_mask.to(held.dtype).sum().clamp_min(1.0)
        neg_count = neg_mask.to(held.dtype).sum().clamp_min(1.0)
        pos_loss = pos_shortfall.sum() / pos_count
        neg_loss = neg_shortfall.sum() / neg_count
        total = float(cfg["weight"]) * (pos_loss + neg_loss)
        return {
            "state_hold_pos_shortfall_loss": pos_loss,
            "state_hold_neg_shortfall_loss": neg_loss,
            "demo_target_hold_loss": total,
        }

    def _intent_loss_terms(
        self,
        *,
        expert_normalized: torch.Tensor,
        intent_logits: torch.Tensor | None,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self._intent_loss
        zero = expert_normalized.new_zeros(())
        if not cfg["enabled"]:
            return {
                "intent_axis_dir_loss": zero,
                "intent_loss": zero,
            }
        if intent_logits is None:
            raise ValueError("intent_loss.enabled requires model intent logits.")

        action_mean = torch.as_tensor(
            self.norm_stats["action_mean"],
            dtype=expert_normalized.dtype,
            device=expert_normalized.device,
        )
        action_std = torch.as_tensor(
            self.norm_stats["action_std"],
            dtype=expert_normalized.dtype,
            device=expert_normalized.device,
        )
        expert = expert_normalized * action_std + action_mean
        pos = cfg["pos"].to(device=expert.device, dtype=expert.dtype)
        neg = cfg["neg"].to(device=expert.device, dtype=expert.dtype)
        valid = valid_mask.to(dtype=torch.bool).expand_as(expert)
        target = torch.stack(
            [
                ((expert >= pos) & valid).to(intent_logits.dtype),
                ((expert <= -neg) & valid).to(intent_logits.dtype),
            ],
            dim=-1,
        ).reshape(*expert.shape[:-1], 8)
        label_valid = (
            torch.stack([valid, valid], dim=-1)
            .reshape(*expert.shape[:-1], 8)
            .to(intent_logits.dtype)
        )
        current_steps = int(cfg.get("current_steps", 0))
        if current_steps > 0:
            query_index = torch.arange(expert.shape[1], device=expert.device).view(
                1, -1, 1
            )
            label_valid = label_valid * (query_index < current_steps).to(
                label_valid.dtype
            )
        if intent_logits.shape != target.shape:
            raise ValueError(
                f"intent logits must have shape {target.shape}, got {intent_logits.shape}"
            )

        import torch.nn.functional as F

        positive_weight = cfg["positive_weight"].to(
            device=intent_logits.device,
            dtype=intent_logits.dtype,
        )
        bce = F.binary_cross_entropy_with_logits(
            intent_logits,
            target,
            reduction="none",
            pos_weight=positive_weight,
        )
        count = label_valid.sum().clamp_min(1.0)
        axis_dir_loss = (bce * label_valid).sum() / count
        total = float(cfg["weight"]) * axis_dir_loss
        return {
            "intent_axis_dir_loss": axis_dir_loss,
            "intent_loss": total,
        }

    def configure_optimizers(self):
        return self._optimizer

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, sd, strict: bool = True):
        return self._model.load_state_dict(sd, strict=strict)

    # ── checkpoint helpers ────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        policy_config: dict,
        norm_stats_path: str | Path,
        temporal_agg: bool = False,
        device: str = "cuda",
        inference_precision: str = "fp32",
        inference_compile: bool = False,
        inference_compile_mode: str = "reduce-overhead",
        inference_compile_dynamic: bool = False,
        device_uint8_preprocess: bool = False,
        temporal_aggregation_diagnostics: bool = False,
    ) -> ACTAdapter:
        """
        Convenience factory: load an ACT policy from a checkpoint file.

        Parameters
        ----------
        ckpt_path       Path to policy_best.ckpt (or policy_epoch_N.ckpt).
        policy_config   Same config dict used during training.
        norm_stats_path Path to dataset_stats.pkl.
        """
        ckpt_path = Path(ckpt_path)
        norm_stats_path = Path(norm_stats_path)

        with open(norm_stats_path, "rb") as f:
            norm_stats = pickle.load(f)

        adapter = cls(
            policy_config=policy_config,
            norm_stats=norm_stats,
            temporal_agg=temporal_agg,
            temporal_aggregation_diagnostics=temporal_aggregation_diagnostics,
            device=device,
            inference_precision=inference_precision,
            inference_compile=inference_compile,
            inference_compile_mode=inference_compile_mode,
            inference_compile_dynamic=inference_compile_dynamic,
            device_uint8_preprocess=device_uint8_preprocess,
        )

        raw = torch.load(ckpt_path, map_location="cpu")
        if isinstance(raw, dict) and "model_state_dict" in raw:
            sd = raw["model_state_dict"]
        elif isinstance(raw, dict):
            sd = raw
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(raw)}")

        adapter.load_state_dict(sd)
        adapter._model.to(adapter.device)
        adapter._model.eval()
        adapter._compile_model_for_inference()
        return adapter
