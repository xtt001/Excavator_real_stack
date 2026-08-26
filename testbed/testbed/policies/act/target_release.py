"""Direct A/B return-release supervision for Real Transition ACT.

The condition is only allowed to change the action where the train fold
contains both outcomes at the same swing-side decision region:

* target A keeps an effective negative swing command; and
* target B releases swing to zero.

The paired branch uses the same image and proprioception with only
``real_transition_condition_v1`` flipped.  This makes the target affect the
deterministic ACT action chunk instead of merely making a side classifier
accurate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

CONDITION_KEY = "real_transition_condition_v1"
CONTRACT_SCHEMA = "real_transition_target_release_contract_v1"
CONTINUE_SIDE_CODE = -1
STOP_SIDE_CODE = 1


def resolve_target_release_config(raw: Any) -> dict[str, Any]:
    """Validate the frozen train-fold contract and loss weights."""

    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("target_release_loss config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "scope": "train_only",
            "condition_key": CONDITION_KEY,
            "weight": 0.0,
            "continue_weight": 0.0,
            "stop_weight": 0.0,
            "contrast_weight": 0.0,
            "contrast_margin_scale": 1.0,
            "append_samples_per_episode": 0,
            "axis_index": 0,
            "decision_qpos_range": (0.0, 0.0),
            "action_window_steps": 1,
            "stable_window_steps": 1,
            "stable_qvel_abs": 0.0,
            "positive_deadzone": 0.0,
            "negative_deadzone": 0.0,
            "continue_action_target_abs": 0.0,
            "contract_path": None,
            "contract": {},
        }

    if cfg.get("condition_key") != CONDITION_KEY:
        raise ValueError(f"target_release_loss.condition_key must be {CONDITION_KEY!r}")
    scope = str(cfg.get("scope", "train_only"))
    if scope not in {"train_only", "train_and_validation"}:
        raise ValueError(
            "target_release_loss.scope must be 'train_only' or "
            "'train_and_validation'"
        )
    contract_raw = cfg.get("contract_json")
    if contract_raw is None or not str(contract_raw).strip():
        raise ValueError("target_release_loss.contract_json is required")
    contract_path = Path(str(contract_raw))
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"target_release_loss contract_json does not exist: {contract_path}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, Mapping) or contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(
            f"target_release_loss contract schema must be {CONTRACT_SCHEMA!r}"
        )
    if contract.get("condition_schema") != CONDITION_KEY:
        raise ValueError(
            f"target-release contract condition_schema must be {CONDITION_KEY!r}"
        )

    decision = _mapping(contract.get("decision_region"), name="decision_region")
    axis_index = _integer(decision.get("swing_axis_index"), name="swing_axis_index")
    if axis_index != 0:
        raise ValueError("target-release contract swing_axis_index must be 0")
    if decision.get("continue_target_side") != "A":
        raise ValueError("target-release contract continue_target_side must be 'A'")
    if decision.get("stop_target_side") != "B":
        raise ValueError("target-release contract stop_target_side must be 'B'")
    qpos_range = _finite_pair(
        decision.get("swing_qpos_range_rad"), name="swing_qpos_range_rad"
    )
    if qpos_range[0] >= qpos_range[1]:
        raise ValueError(
            "target-release contract swing_qpos_range_rad must be increasing"
        )

    candidate = _mapping(contract.get("candidate_rule"), name="candidate_rule")
    action_window_steps = _positive_integer(
        candidate.get("action_window_steps"), name="action_window_steps"
    )
    stable_window_steps = _positive_integer(
        candidate.get("stable_window_steps"), name="stable_window_steps"
    )
    stable_qvel_abs = _nonnegative(
        candidate.get("stable_qvel_abs_max_rad_s"),
        name="stable_qvel_abs_max_rad_s",
    )
    if not bool(candidate.get("after_swing_apex", False)):
        raise ValueError("target-release contract requires after_swing_apex=true")

    deadzone = _mapping(contract.get("mechanical_deadzone"), name="mechanical_deadzone")
    swing_deadzone = _mapping(deadzone.get("swing"), name="mechanical_deadzone.swing")
    positive_deadzone = _positive(
        swing_deadzone.get("pos"), name="mechanical_deadzone.swing.pos"
    )
    negative_deadzone = _positive(
        swing_deadzone.get("neg"), name="mechanical_deadzone.swing.neg"
    )
    continue_action_target_abs = _positive(
        decision.get("continue_action_target_abs", negative_deadzone),
        name="decision_region.continue_action_target_abs",
    )
    if continue_action_target_abs < negative_deadzone:
        raise ValueError(
            "target-release continue_action_target_abs cannot be below the "
            "negative mechanical deadzone"
        )
    append_samples = _positive_integer(
        cfg.get("append_samples_per_episode", 1),
        name="append_samples_per_episode",
    )

    return {
        "enabled": True,
        "scope": scope,
        "condition_key": CONDITION_KEY,
        "weight": _nonnegative(cfg.get("weight", 1.0), name="weight"),
        "continue_weight": _nonnegative(
            cfg.get("continue_weight", 1.0), name="continue_weight"
        ),
        "stop_weight": _nonnegative(
            cfg.get("stop_weight", 1.0), name="stop_weight"
        ),
        "contrast_weight": _nonnegative(
            cfg.get("contrast_weight", 1.0), name="contrast_weight"
        ),
        "contrast_margin_scale": _nonnegative(
            cfg.get("contrast_margin_scale", 1.0),
            name="contrast_margin_scale",
        ),
        "append_samples_per_episode": append_samples,
        "axis_index": axis_index,
        "decision_qpos_range": qpos_range,
        "action_window_steps": action_window_steps,
        "stable_window_steps": stable_window_steps,
        "stable_qvel_abs": stable_qvel_abs,
        "positive_deadzone": positive_deadzone,
        "negative_deadzone": negative_deadzone,
        "continue_action_target_abs": continue_action_target_abs,
        "contract_path": str(contract_path.resolve()),
        "contract": dict(contract),
    }


def target_release_candidate_indices(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    actions: np.ndarray,
    condition: np.ndarray,
    valid_starts: np.ndarray,
    condition_valid_mask: np.ndarray | None,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Return train-supported return-release starts for one cycle."""

    if not bool(config.get("enabled", False)):
        return np.zeros(0, dtype=np.int64)
    qpos_arr = np.asarray(qpos, dtype=np.float32)
    qvel_arr = np.asarray(qvel, dtype=np.float32)
    action_arr = np.asarray(actions, dtype=np.float32)
    condition_arr = np.asarray(condition, dtype=np.float32)
    if (
        qpos_arr.ndim != 2
        or qvel_arr.shape != qpos_arr.shape
        or action_arr.shape != qpos_arr.shape
        or qpos_arr.shape[1] <= int(config["axis_index"])
    ):
        raise ValueError("target release requires matching (T, action_dim) state/action")
    if condition_arr.shape != (qpos_arr.shape[0], 2):
        raise ValueError("target release condition must have shape (T, 2)")
    if not (
        np.isfinite(qpos_arr).all()
        and np.isfinite(qvel_arr).all()
        and np.isfinite(action_arr).all()
        and np.isfinite(condition_arr).all()
    ):
        raise ValueError("target release state/action/condition contains NaN or Inf")
    side_codes = condition_arr[:, 0]
    active = condition_arr[:, 1]
    if not np.all(active == 1.0) or not np.all(side_codes == side_codes[0]):
        raise ValueError("target release requires one constant active target per cycle")
    side_code = int(side_codes[0])
    if side_code not in {CONTINUE_SIDE_CODE, STOP_SIDE_CODE}:
        raise ValueError("target release target side code must be -1 or +1")

    valid_condition = None
    if condition_valid_mask is not None:
        valid_condition = np.asarray(condition_valid_mask, dtype=bool)
        if valid_condition.ndim != 2 or valid_condition.shape[0] != qpos_arr.shape[0]:
            raise ValueError("conditions/valid_mask must have shape (T, chunk_steps)")

    total_steps = int(qpos_arr.shape[0])
    axis = int(config["axis_index"])
    action_window = int(config["action_window_steps"])
    stable_window = int(config["stable_window_steps"])
    qpos_low, qpos_high = config["decision_qpos_range"]
    positive = float(config["positive_deadzone"])
    negative = float(config["negative_deadzone"])
    stable_qvel = float(config["stable_qvel_abs"])
    apex = int(np.argmax(qpos_arr[:, axis]))
    starts = np.asarray(valid_starts, dtype=np.int64).reshape(-1)
    candidates: list[int] = []
    for raw_start in starts:
        start = int(raw_start)
        if start < apex or start < 0 or start + action_window > total_steps:
            continue
        position = float(qpos_arr[start, axis])
        if not qpos_low <= position <= qpos_high:
            continue
        if valid_condition is not None:
            if valid_condition.shape[1] < action_window or not bool(
                valid_condition[start, :action_window].all()
            ):
                continue
        action_window_values = action_arr[start : start + action_window, axis]
        if side_code == CONTINUE_SIDE_CODE:
            supported = bool(np.all(action_window_values <= -negative))
        else:
            supported = bool(
                start + stable_window <= total_steps
                and np.all(action_window_values < positive)
                and np.all(action_window_values > -negative)
                and np.all(
                    np.abs(qvel_arr[start : start + stable_window, axis])
                    <= stable_qvel
                )
            )
        if supported:
            candidates.append(start)
    return np.asarray(candidates, dtype=np.int64)


def target_release_loss_terms(
    *,
    primary_direct: torch.Tensor,
    counterfactual_direct: torch.Tensor,
    continue_primary: torch.Tensor | None,
    valid: torch.Tensor | None,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Require A to continue negative and B to release on paired observations."""

    zero = primary_direct.new_zeros(())
    result = {
        "target_release_continue_shortfall_loss": zero,
        "target_release_stop_zero_loss": zero,
        "target_release_contrast_loss": zero,
        "target_release_unweighted_loss": zero,
        "target_release_loss": zero,
        "target_release_valid_count": zero,
        "target_release_continue_crossing_rate": zero,
        "target_release_stop_idle_rate": zero,
        "target_release_pair_hit_rate": zero,
    }
    if not bool(config.get("enabled", False)):
        return result
    if tuple(primary_direct.shape) != tuple(counterfactual_direct.shape):
        raise ValueError("target release primary and counterfactual chunks must match")
    if primary_direct.ndim != 3:
        raise ValueError("target release action chunks must have shape (B, Q, A)")
    if continue_primary is None or valid is None:
        raise ValueError("target release requires continue_primary and valid labels")

    valid_mask = valid.to(device=primary_direct.device, dtype=torch.bool).reshape(-1)
    continue_mask = continue_primary.to(
        device=primary_direct.device, dtype=torch.bool
    ).reshape(-1)
    if valid_mask.numel() != primary_direct.shape[0] or continue_mask.numel() != primary_direct.shape[0]:
        raise ValueError("target release labels must have one value per batch row")
    count = valid_mask.sum().to(dtype=primary_direct.dtype)
    result["target_release_valid_count"] = count
    if not bool(valid_mask.any()):
        return result

    axis = int(config["axis_index"])
    if axis < 0 or axis >= primary_direct.shape[2]:
        raise ValueError("target release axis_index is outside the action dimension")
    window = int(config["action_window_steps"])
    if window > primary_direct.shape[1]:
        raise ValueError("target release action_window_steps exceeds ACT chunk size")
    primary = primary_direct[valid_mask, :window, axis]
    counterfactual = counterfactual_direct[valid_mask, :window, axis]
    primary_is_continue = continue_mask[valid_mask].reshape(-1, 1)
    continue_action = torch.where(primary_is_continue, primary, counterfactual)
    stop_action = torch.where(primary_is_continue, counterfactual, primary)

    negative = float(config["negative_deadzone"])
    continue_target = float(config["continue_action_target_abs"])
    positive = float(config["positive_deadzone"])
    contrast_margin = continue_target * float(config["contrast_margin_scale"])
    continue_shortfall = torch.relu(continue_target + continue_action).mean()
    stop_zero = torch.abs(stop_action).mean()
    contrast = torch.relu(contrast_margin - (stop_action - continue_action)).mean()
    unweighted = (
        float(config["continue_weight"]) * continue_shortfall
        + float(config["stop_weight"]) * stop_zero
        + float(config["contrast_weight"]) * contrast
    )
    continue_hit = continue_action <= -negative
    stop_idle = (stop_action < positive) & (stop_action > -negative)
    pair_hit = continue_hit & stop_idle
    return {
        "target_release_continue_shortfall_loss": continue_shortfall,
        "target_release_stop_zero_loss": stop_zero,
        "target_release_contrast_loss": contrast,
        "target_release_unweighted_loss": unweighted,
        "target_release_loss": float(config["weight"]) * unweighted,
        "target_release_valid_count": count,
        "target_release_continue_crossing_rate": continue_hit.to(
            dtype=primary_direct.dtype
        ).mean(),
        "target_release_stop_idle_rate": stop_idle.to(
            dtype=primary_direct.dtype
        ).mean(),
        "target_release_pair_hit_rate": pair_hit.to(
            dtype=primary_direct.dtype
        ).mean(),
    }


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"target-release contract {name} must be a mapping")
    return value


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"target_release_loss.{name} must be a boolean")
    return bool(value)


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"target_release_loss.{name} must be an integer")
    result = int(value)
    if float(value) != float(result):
        raise ValueError(f"target_release_loss.{name} must be an integer")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    result = _integer(value, name=name)
    if result <= 0:
        raise ValueError(f"target_release_loss.{name} must be positive")
    return result


def _nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"target_release_loss.{name} must be non-negative")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"target_release_loss.{name} must be non-negative")
    return result


def _positive(value: Any, *, name: str) -> float:
    result = _nonnegative(value, name=name)
    if result <= 0.0:
        raise ValueError(f"target_release_loss.{name} must be positive")
    return result


def _finite_pair(value: Any, *, name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 2 or not np.isfinite(array).all():
        raise ValueError(f"target_release_loss.{name} must contain two finite values")
    return float(array[0]), float(array[1])
