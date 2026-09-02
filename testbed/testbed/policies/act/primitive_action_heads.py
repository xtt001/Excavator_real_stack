"""Hard-routed factual action heads for oracle primitive ACT experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from testbed.data.action_primitive_islands import (
    ACTION_PRIMITIVE_KEY,
    PRIMITIVE_NAMES,
)
from testbed.data.work_return_context import (
    TASK_HEAD_NAMES,
    WORK_CONTEXT_KEY,
)


def resolve_primitive_action_heads_config(
    raw: Any,
    *,
    robot_state_dim: int,
) -> dict[str, Any]:
    """Validate structural primitive routing against the proprio layout."""

    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("primitive_action_heads config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "condition_key": ACTION_PRIMITIVE_KEY,
            "primitive_names": PRIMITIVE_NAMES,
            "primitive_count": len(PRIMITIVE_NAMES),
            "one_hot_start_index": int(robot_state_dim),
        }
    condition_key = str(cfg.get("condition_key", ""))
    expected_names = {
        ACTION_PRIMITIVE_KEY: PRIMITIVE_NAMES,
        WORK_CONTEXT_KEY: TASK_HEAD_NAMES,
    }.get(condition_key)
    if expected_names is None:
        raise ValueError(
            "primitive_action_heads.condition_key must be "
            f"{ACTION_PRIMITIVE_KEY!r} or {WORK_CONTEXT_KEY!r}"
        )
    names = tuple(str(value) for value in cfg.get("primitive_names", ()))
    if names != expected_names:
        raise ValueError(
            "primitive_action_heads.primitive_names must equal "
            f"{list(expected_names)!r} for {condition_key}"
        )
    start = _nonnegative_integer(
        cfg.get("one_hot_start_index"), name="one_hot_start_index"
    )
    state_dim = _positive_integer(robot_state_dim, name="robot_state_dim")
    if start + len(names) != state_dim:
        raise ValueError(
            "primitive_action_heads one-hot must occupy the final proprio columns: "
            f"start={start}, count={len(names)}, state_dim={state_dim}"
        )
    zero_indices_raw = dict(cfg.get("branch_zero_indices", {}) or {})
    unexpected_masks = sorted(set(zero_indices_raw) - set(names))
    if unexpected_masks:
        raise ValueError(
            "primitive_action_heads.branch_zero_indices has unknown branches "
            f"{unexpected_masks}"
        )
    zero_indices: dict[str, tuple[int, ...]] = {}
    selector_indices = set(range(start, start + len(names)))
    for name in names:
        values = tuple(
            sorted(
                {
                    _nonnegative_integer(index, name=f"branch_zero_indices.{name}")
                    for index in zero_indices_raw.get(name, [])
                }
            )
        )
        if any(index >= state_dim for index in values):
            raise ValueError(
                f"primitive_action_heads mask for {name} exceeds state_dim"
            )
        if selector_indices.intersection(values):
            raise ValueError(
                f"primitive_action_heads mask for {name} cannot erase route one-hot"
            )
        zero_indices[name] = values
    return {
        "enabled": True,
        "condition_key": condition_key,
        "primitive_names": names,
        "primitive_count": len(names),
        "one_hot_start_index": start,
        "branch_zero_indices": zero_indices,
    }


def mask_hard_routed_proprio(
    proprio: torch.Tensor,
    *,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Remove branch-irrelevant task fields before both ACT projections."""

    if not bool(config.get("enabled", False)):
        return proprio
    selector = _validated_selector(proprio, config=config)
    names = tuple(str(name) for name in config["primitive_names"])
    zero_indices = dict(config.get("branch_zero_indices", {}) or {})
    if not any(zero_indices.get(name) for name in names):
        return proprio
    keep = torch.ones_like(proprio)
    for branch_index, name in enumerate(names):
        for column in zero_indices.get(name, ()):
            keep[:, int(column)] = keep[:, int(column)] * (
                1.0 - selector[:, branch_index]
            )
    return proprio * keep


def select_hard_routed_action(
    *,
    shared_head: torch.nn.Module,
    additional_heads: torch.nn.ModuleList,
    decoder_state: torch.Tensor,
    proprio: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Apply exactly one action head per batch row using a validated one-hot."""

    if not bool(config.get("enabled", False)):
        return shared_head(decoder_state)
    count = int(config["primitive_count"])
    if len(additional_heads) != count - 1:
        raise ValueError(
            "primitive action head count mismatch: "
            f"expected {count - 1} additional heads, got {len(additional_heads)}"
        )
    rounded = _validated_selector(proprio, config=config)
    outputs = torch.stack(
        [shared_head(decoder_state)]
        + [head(decoder_state) for head in additional_heads],
        dim=2,
    )
    return torch.sum(
        outputs * rounded.to(dtype=outputs.dtype)[:, None, :, None],
        dim=2,
    )


def _validated_selector(
    proprio: torch.Tensor,
    *,
    config: Mapping[str, Any],
) -> torch.Tensor:
    count = int(config["primitive_count"])
    start = int(config["one_hot_start_index"])
    if proprio.ndim != 2 or proprio.shape[1] < start + count:
        raise ValueError(
            "primitive action routing requires proprio shaped "
            f"(batch, >= {start + count})"
        )
    selector = proprio[:, start : start + count]
    rounded = torch.round(selector)
    valid = (
        torch.isfinite(selector).all(dim=1)
        & torch.isclose(selector, rounded, atol=1e-6, rtol=0.0).all(dim=1)
        & ((rounded == 0.0) | (rounded == 1.0)).all(dim=1)
        & torch.isclose(
            rounded.sum(dim=1),
            torch.ones(selector.shape[0], device=selector.device),
            atol=1e-6,
            rtol=0.0,
        )
    )
    if not bool(valid.all()):
        invalid = torch.nonzero(~valid, as_tuple=False).reshape(-1).tolist()
        raise ValueError(
            "primitive action routing requires one finite one-hot per batch row; "
            f"invalid rows={invalid}"
        )
    return rounded


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"primitive_action_heads.{name} must be boolean")


def _positive_integer(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0 or float(value) != float(parsed):
        raise ValueError(f"primitive_action_heads.{name} must be a positive integer")
    return parsed


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if value is None:
        raise ValueError(f"primitive_action_heads.{name} is required")
    parsed = int(value)
    if parsed < 0 or float(value) != float(parsed):
        raise ValueError(
            f"primitive_action_heads.{name} must be a non-negative integer"
        )
    return parsed
