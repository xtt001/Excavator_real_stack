"""Low-dimensional coarse action plus visual/qpos ACT residual."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_STAGES = (
    {
        "name": "low_only",
        "start_epoch": 0,
        "end_epoch": 40,
        "train_low": True,
        "train_residual": False,
        "residual_scale": 0.0,
    },
    {
        "name": "residual_only",
        "start_epoch": 40,
        "end_epoch": 140,
        "train_low": False,
        "train_residual": True,
        "residual_scale": 1.0,
    },
    {
        "name": "joint",
        "start_epoch": 140,
        "end_epoch": 200,
        "train_low": True,
        "train_residual": True,
        "residual_scale": 1.0,
    },
)


def resolve_state_visual_residual_config(
    raw: Any,
    *,
    robot_state_dim: int,
    num_queries: int,
    action_dim: int,
) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("state_visual_residual config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return {
            "enabled": False,
            "robot_state_dim": int(robot_state_dim),
            "num_queries": int(num_queries),
            "action_dim": int(action_dim),
            "low_hidden_dim": 1,
            "residual_keep_indices": tuple(range(int(robot_state_dim))),
            "stages": tuple(),
        }
    state_dim = _positive_integer(robot_state_dim, name="robot_state_dim")
    query_count = _positive_integer(num_queries, name="num_queries")
    output_dim = _positive_integer(action_dim, name="action_dim")
    low_hidden = _positive_integer(
        cfg.get("low_hidden_dim", 256), name="low_hidden_dim"
    )
    raw_indices = cfg.get("residual_keep_indices", (0, 1, 2, 3))
    if not isinstance(raw_indices, Sequence) or isinstance(raw_indices, (str, bytes)):
        raise ValueError("state_visual_residual.residual_keep_indices must be a list")
    indices = tuple(
        sorted(
            {
                _nonnegative_integer(value, name="residual_keep_indices")
                for value in raw_indices
            }
        )
    )
    if not indices or any(index >= state_dim for index in indices):
        raise ValueError(
            "state_visual_residual residual indices exceed state dimension"
        )
    stages_raw = cfg.get("stages", DEFAULT_STAGES)
    if not isinstance(stages_raw, Sequence) or isinstance(stages_raw, (str, bytes)):
        raise ValueError("state_visual_residual.stages must be a list")
    stages = tuple(_stage(value) for value in stages_raw)
    if not stages or stages[0]["start_epoch"] != 0:
        raise ValueError("state_visual_residual stages must start at epoch 0")
    for previous, current in zip(stages, stages[1:], strict=False):
        if previous["end_epoch"] != current["start_epoch"]:
            raise ValueError("state_visual_residual stages must be contiguous")
    return {
        "enabled": True,
        "robot_state_dim": state_dim,
        "num_queries": query_count,
        "action_dim": output_dim,
        "low_hidden_dim": low_hidden,
        "residual_keep_indices": indices,
        "stages": stages,
    }


def stage_for_epoch(config: Mapping[str, Any], epoch: int) -> dict[str, Any] | None:
    if not bool(config.get("enabled", False)):
        return None
    value = int(epoch)
    for stage in config["stages"]:
        if int(stage["start_epoch"]) <= value < int(stage["end_epoch"]):
            return dict(stage)
    raise ValueError(f"epoch {value} lies outside state_visual_residual stages")


def _stage(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("state_visual_residual stage must be a mapping")
    stage = dict(value)
    name = str(stage.get("name", "")).strip()
    if not name:
        raise ValueError("state_visual_residual stage name is required")
    start = _nonnegative_integer(stage.get("start_epoch"), name=f"{name}.start_epoch")
    end = _positive_integer(stage.get("end_epoch"), name=f"{name}.end_epoch")
    if end <= start:
        raise ValueError(f"state_visual_residual stage {name} has empty range")
    train_low = _strict_bool(stage.get("train_low"), name=f"{name}.train_low")
    train_residual = _strict_bool(
        stage.get("train_residual"), name=f"{name}.train_residual"
    )
    if not train_low and not train_residual:
        raise ValueError(f"state_visual_residual stage {name} trains no parameters")
    scale = float(stage.get("residual_scale", 1.0))
    if scale < 0.0:
        raise ValueError(f"state_visual_residual stage {name} scale is negative")
    return {
        "name": name,
        "start_epoch": start,
        "end_epoch": end,
        "train_low": train_low,
        "train_residual": train_residual,
        "residual_scale": scale,
    }


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"state_visual_residual.{name} must be boolean")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0 or float(value) != float(parsed):
        raise ValueError(f"state_visual_residual.{name} must be a positive integer")
    return parsed


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if value is None:
        raise ValueError(f"state_visual_residual.{name} is required")
    parsed = int(value)
    if parsed < 0 or float(value) != float(parsed):
        raise ValueError(f"state_visual_residual.{name} must be a non-negative integer")
    return parsed
