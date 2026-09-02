"""Direct-action adherence for planner-owned return-commit intent."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from testbed.tasks.real_transition_return_commit import RETURN_COMMIT_KEY


def resolve_return_commit_loss_config(raw: Any) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("return_commit_loss config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "scope": "train_only",
            "condition_key": RETURN_COMMIT_KEY,
            "weight": 0.0,
            "dig_preservation_weight": 0.0,
            "return_preservation_weight": 0.0,
            "return_effective_preservation_weight": 0.0,
            "dig_counterfactual_weight": 0.0,
            "return_counterfactual_weight": 0.0,
            "contrast_weight": 0.0,
            "contrast_margin": 0.1,
            "append_samples_per_episode": 0,
            "action_window_steps": 1,
            "axis_index": 0,
            "positive_deadzone": 0.0,
            "negative_deadzone": 0.0,
            "intent_threshold": 0.05,
            "dig_counterfactual_mode": "positive_intent",
            "dig_candidate_mode": "swing_positive",
            "negative_deadzone_guard_margin": 0.0,
            "threshold_path": None,
        }
    if cfg.get("condition_key") != RETURN_COMMIT_KEY:
        raise ValueError(
            f"return_commit_loss.condition_key must be {RETURN_COMMIT_KEY!r}"
        )
    scope = str(cfg.get("scope", "train_only"))
    if scope not in {"train_only", "train_and_validation"}:
        raise ValueError(
            "return_commit_loss.scope must be train_only or train_and_validation"
        )
    threshold_raw = cfg.get("threshold_json")
    if threshold_raw is None or not str(threshold_raw).strip():
        raise ValueError("return_commit_loss.threshold_json is required")
    threshold_path = Path(str(threshold_raw))
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"return commit threshold_json does not exist: {threshold_path}"
        )
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    deadzone = _mapping(payload.get("deadzone_action"), name="deadzone_action")
    swing = _mapping(deadzone.get("swing"), name="deadzone_action.swing")
    axis_names = ("swing", "boom", "stick", "bucket")
    positive_by_axis = np.asarray(
        [
            _positive(
                _mapping(deadzone.get(axis), name=f"deadzone_action.{axis}").get(
                    "pos"
                ),
                name=f"deadzone_action.{axis}.pos",
            )
            for axis in axis_names
        ],
        dtype=np.float32,
    )
    negative_by_axis = np.asarray(
        [
            _positive(
                _mapping(deadzone.get(axis), name=f"deadzone_action.{axis}").get(
                    "neg"
                ),
                name=f"deadzone_action.{axis}.neg",
            )
            for axis in axis_names
        ],
        dtype=np.float32,
    )
    intent = _positive(cfg.get("intent_threshold", 0.05), name="intent_threshold")
    contrast_margin = _positive(
        cfg.get("contrast_margin", 2.0 * intent), name="contrast_margin"
    )
    append_samples = _positive_integer(
        cfg.get("append_samples_per_episode", 2),
        name="append_samples_per_episode",
    )
    if append_samples not in {2, 3}:
        raise ValueError(
            "return_commit_loss.append_samples_per_episode must be 2 or 3"
        )
    counterfactual_mode = str(
        cfg.get("dig_counterfactual_mode", "positive_intent")
    )
    if counterfactual_mode not in {"positive_intent", "no_negative_effective"}:
        raise ValueError(
            "return_commit_loss.dig_counterfactual_mode must be "
            "positive_intent or no_negative_effective"
        )
    dig_candidate_mode = str(cfg.get("dig_candidate_mode", "swing_positive"))
    if dig_candidate_mode not in {"swing_positive", "non_swing_transition"}:
        raise ValueError(
            "return_commit_loss.dig_candidate_mode must be swing_positive "
            "or non_swing_transition"
        )
    return {
        "enabled": True,
        "scope": scope,
        "condition_key": RETURN_COMMIT_KEY,
        "weight": _nonnegative(cfg.get("weight", 1.0), name="weight"),
        "dig_preservation_weight": _nonnegative(
            cfg.get("dig_preservation_weight", 1.0),
            name="dig_preservation_weight",
        ),
        "return_preservation_weight": _nonnegative(
            cfg.get("return_preservation_weight", 1.0),
            name="return_preservation_weight",
        ),
        "return_effective_preservation_weight": _nonnegative(
            cfg.get("return_effective_preservation_weight", 1.0),
            name="return_effective_preservation_weight",
        ),
        "dig_counterfactual_weight": _nonnegative(
            cfg.get("dig_counterfactual_weight", 1.0),
            name="dig_counterfactual_weight",
        ),
        "return_counterfactual_weight": _nonnegative(
            cfg.get("return_counterfactual_weight", 1.0),
            name="return_counterfactual_weight",
        ),
        "contrast_weight": _nonnegative(
            cfg.get("contrast_weight", 1.0), name="contrast_weight"
        ),
        "contrast_margin": contrast_margin,
        "append_samples_per_episode": append_samples,
        "action_window_steps": _positive_integer(
            cfg.get("action_window_steps", 5), name="action_window_steps"
        ),
        "axis_index": 0,
        "positive_deadzone": _positive(
            swing.get("pos"), name="deadzone_action.swing.pos"
        ),
        "negative_deadzone": _positive(
            swing.get("neg"), name="deadzone_action.swing.neg"
        ),
        "intent_threshold": intent,
        "positive_deadzone_by_axis": positive_by_axis,
        "negative_deadzone_by_axis": negative_by_axis,
        "dig_counterfactual_mode": counterfactual_mode,
        "dig_candidate_mode": dig_candidate_mode,
        "dig_candidate_key": (
            "dig_positive"
            if dig_candidate_mode == "swing_positive"
            else "dig_non_swing_transition"
        ),
        "negative_deadzone_guard_margin": _nonnegative(
            cfg.get("negative_deadzone_guard_margin", 0.0),
            name="negative_deadzone_guard_margin",
        ),
        "threshold_path": str(threshold_path.resolve()),
    }


def return_commit_candidate_indices(
    *,
    actions: np.ndarray,
    return_commit: np.ndarray,
    valid_starts: np.ndarray,
    return_commit_valid_mask: np.ndarray | None,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    empty = {
        "dig_positive": np.zeros(0, dtype=np.int64),
        "dig_non_swing_transition": np.zeros(0, dtype=np.int64),
        "return_onset": np.zeros(0, dtype=np.int64),
        "return_effective": np.zeros(0, dtype=np.int64),
    }
    if not bool(config.get("enabled", False)):
        return empty
    action_array = np.asarray(actions, dtype=np.float32)
    state = np.asarray(return_commit, dtype=np.float32).reshape(-1)
    if action_array.ndim != 2 or len(action_array) != len(state):
        raise ValueError("return commit candidates require aligned action/state")
    if not np.all(np.isin(state, [0.0, 1.0])) or np.any(np.diff(state) < 0.0):
        raise ValueError("return commit state must be monotonic finite 0/1")
    valid_state = None
    if return_commit_valid_mask is not None:
        valid_state = np.asarray(return_commit_valid_mask, dtype=bool)
        if valid_state.ndim != 2 or valid_state.shape[0] != len(state):
            raise ValueError(
                "return_commit_valid_mask must have shape (T, chunk_steps)"
            )
    window = int(config["action_window_steps"])
    axis = int(config["axis_index"])
    positive = float(config["positive_deadzone"])
    negative = float(config["negative_deadzone"])
    intent = float(config["intent_threshold"])
    valid_set = set(int(value) for value in np.asarray(valid_starts).reshape(-1))
    dig: list[int] = []
    dig_non_swing: list[int] = []
    effective: list[int] = []
    positive_by_axis = np.asarray(
        config["positive_deadzone_by_axis"], dtype=np.float32
    )
    negative_by_axis = np.asarray(
        config["negative_deadzone_by_axis"], dtype=np.float32
    )
    if positive_by_axis.shape != (action_array.shape[1],) or (
        negative_by_axis.shape != (action_array.shape[1],)
    ):
        raise ValueError("return commit all-axis deadzones must match action width")
    action_effective = np.stack(
        (
            action_array >= positive_by_axis,
            action_array <= -negative_by_axis,
        ),
        axis=-1,
    )
    previous_effective = np.zeros_like(action_effective, dtype=bool)
    previous_effective[1:] = action_effective[:-1]
    transition = action_effective & ~previous_effective
    for start in sorted(valid_set):
        if start < 0 or start + window > len(state):
            continue
        if valid_state is not None and (
            valid_state.shape[1] < window
            or not bool(valid_state[start, :window].all())
        ):
            continue
        values = action_array[start : start + window, axis]
        if state[start] == 0.0 and bool(np.all(values >= positive)):
            dig.append(start)
        if state[start] == 0.0 and bool(transition[start, 1:, :].any()):
            dig_non_swing.append(start)
        if state[start] == 1.0 and bool(np.all(values <= -negative)):
            effective.append(start)
    committed = np.flatnonzero(state >= 0.5)
    onset: list[int] = []
    if committed.size:
        start = int(committed[0])
        if start in valid_set and start + window <= len(state):
            state_valid = valid_state is None or (
                valid_state.shape[1] >= window
                and bool(valid_state[start, :window].all())
            )
            if state_valid and bool(
                np.all(action_array[start : start + window, axis] <= -intent)
            ):
                onset.append(start)
    return {
        "dig_positive": np.asarray(dig, dtype=np.int64),
        "dig_non_swing_transition": np.asarray(
            dig_non_swing, dtype=np.int64
        ),
        "return_onset": np.asarray(onset, dtype=np.int64),
        "return_effective": np.asarray(effective, dtype=np.int64),
    }


def return_commit_loss_terms(
    *,
    primary_direct: torch.Tensor,
    counterfactual_direct: torch.Tensor,
    return_primary: torch.Tensor | None,
    valid: torch.Tensor | None,
    return_effective_primary: torch.Tensor | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    cfg = (
        dict(config)
        if isinstance(config, Mapping) and "positive_deadzone" in config
        else resolve_return_commit_loss_config(config)
    )
    zero = primary_direct.new_zeros(())
    result = {
        "return_commit_dig_preservation_loss": zero,
        "return_commit_return_preservation_loss": zero,
        "return_commit_return_effective_preservation_loss": zero,
        "return_commit_dig_counterfactual_loss": zero,
        "return_commit_return_counterfactual_loss": zero,
        "return_commit_contrast_loss": zero,
        "return_commit_unweighted_loss": zero,
        "return_commit_loss": zero,
        "return_commit_valid_count": zero,
        "return_commit_dig_primary_effective_rate": zero,
        "return_commit_return_primary_intent_rate": zero,
        "return_commit_return_primary_effective_rate": zero,
        "return_commit_DIG_counterfactual_no_shortcut_rate": zero,
        "return_commit_pair_hit_rate": zero,
    }
    if not bool(cfg["enabled"]):
        return result
    if tuple(primary_direct.shape) != tuple(counterfactual_direct.shape):
        raise ValueError("return commit primary/counterfactual chunks must match")
    if primary_direct.ndim != 3:
        raise ValueError("return commit chunks must have shape (B, Q, A)")
    if return_primary is None or valid is None:
        raise ValueError("return commit loss requires return_primary and valid labels")
    valid_mask = valid.to(device=primary_direct.device, dtype=torch.bool).reshape(-1)
    return_all = return_primary.to(
        device=primary_direct.device, dtype=torch.bool
    ).reshape(-1)
    effective_all = (
        torch.zeros_like(return_all)
        if return_effective_primary is None
        else return_effective_primary.to(
            device=primary_direct.device, dtype=torch.bool
        ).reshape(-1)
    )
    if valid_mask.numel() != primary_direct.shape[0] or return_all.numel() != primary_direct.shape[0]:
        raise ValueError("return commit labels must have one value per batch row")
    if effective_all.numel() != primary_direct.shape[0] or bool(
        torch.any(effective_all & ~return_all)
    ):
        raise ValueError(
            "return effective labels must select only return-primary batch rows"
        )
    result["return_commit_valid_count"] = valid_mask.sum().to(
        dtype=primary_direct.dtype
    )
    if not bool(valid_mask.any()):
        return result

    axis = int(cfg["axis_index"])
    window = int(cfg["action_window_steps"])
    primary = primary_direct[valid_mask, :window, axis]
    counterfactual = counterfactual_direct[valid_mask, :window, axis]
    selected_return = return_all[valid_mask]
    selected_effective = effective_all[valid_mask]
    return_mask = selected_return & ~selected_effective
    effective_mask = selected_effective
    dig_mask = ~return_mask
    return_selector = selected_return.reshape(-1, 1)
    dig_action = torch.where(return_selector, counterfactual, primary)
    return_action = torch.where(return_selector, primary, counterfactual)
    positive = float(cfg["positive_deadzone"])
    negative = float(cfg["negative_deadzone"])
    intent = float(cfg["intent_threshold"])

    dig_preservation = _masked_mean(
        torch.relu(positive - dig_action), dig_mask
    )
    return_preservation = _masked_mean(
        torch.relu(intent + return_action), return_mask
    )
    return_effective_preservation = _masked_mean(
        torch.relu(negative + return_action), effective_mask
    )
    if cfg["dig_counterfactual_mode"] == "positive_intent":
        dig_counterfactual = _masked_mean(
            torch.relu(intent - dig_action), return_mask
        )
        dig_counterfactual_hit = dig_action >= intent
    else:
        guard_floor = -negative + float(cfg["negative_deadzone_guard_margin"])
        dig_counterfactual = _masked_mean(
            torch.relu(guard_floor - dig_action), return_mask
        )
        dig_counterfactual_hit = dig_action > -negative
    return_counterfactual = _masked_mean(
        torch.relu(intent + return_action), dig_mask
    )
    contrast_rows = dig_mask | return_mask
    contrast = _masked_mean(
        torch.relu(
            float(cfg["contrast_margin"]) - (dig_action - return_action)
        ),
        contrast_rows,
    )
    unweighted = (
        float(cfg["dig_preservation_weight"]) * dig_preservation
        + float(cfg["return_preservation_weight"]) * return_preservation
        + float(cfg["return_effective_preservation_weight"])
        * return_effective_preservation
        + float(cfg["dig_counterfactual_weight"]) * dig_counterfactual
        + float(cfg["return_counterfactual_weight"]) * return_counterfactual
        + float(cfg["contrast_weight"]) * contrast
    )
    pair_hit = torch.where(
        return_mask.reshape(-1, 1),
        dig_counterfactual_hit & (return_action <= -intent),
        (dig_action >= intent) & (return_action <= -intent),
    )
    pair_rows = dig_mask | return_mask
    result.update(
        {
            "return_commit_dig_preservation_loss": dig_preservation,
            "return_commit_return_preservation_loss": return_preservation,
            "return_commit_return_effective_preservation_loss": (
                return_effective_preservation
            ),
            "return_commit_dig_counterfactual_loss": dig_counterfactual,
            "return_commit_return_counterfactual_loss": return_counterfactual,
            "return_commit_contrast_loss": contrast,
            "return_commit_unweighted_loss": unweighted,
            "return_commit_loss": float(cfg["weight"]) * unweighted,
            "return_commit_dig_primary_effective_rate": _masked_rate(
                dig_action >= positive, dig_mask
            ),
            "return_commit_return_primary_intent_rate": _masked_rate(
                return_action <= -intent, return_mask
            ),
            "return_commit_return_primary_effective_rate": _masked_rate(
                return_action <= -negative, effective_mask
            ),
            "return_commit_DIG_counterfactual_no_shortcut_rate": _masked_rate(
                dig_action > -negative, return_mask
            ),
            "return_commit_pair_hit_rate": _masked_rate(pair_hit, pair_rows),
        }
    )
    return result


def _masked_mean(values: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    selected = values[row_mask]
    return values.new_zeros(()) if selected.numel() == 0 else selected.mean()


def _masked_rate(values: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    selected = values[row_mask]
    return (
        torch.zeros((), dtype=torch.float32, device=values.device)
        if selected.numel() == 0
        else selected.to(dtype=torch.float32).mean()
    )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"return_commit_loss.{name} must be boolean")
    return value


def _nonnegative(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"return_commit_loss.{name} must be non-negative")
    return number


def _positive(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"return_commit_loss.{name} must be positive")
    return number


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"return_commit_loss.{name} must be a positive integer")
    number = int(value)
    if number <= 0 or float(value) != float(number):
        raise ValueError(f"return_commit_loss.{name} must be a positive integer")
    return number
