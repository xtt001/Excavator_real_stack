"""Auditable model-only checkpoint expansion for additional proprio inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

EXPANDABLE_PROPRIO_WEIGHTS = frozenset(
    {
        "input_proj_robot_state.weight",
        "encoder_joint_proj.weight",
    }
)
OPTIONAL_AUXILIARY_PREFIXES = frozenset(
    {
        "intent_head",
        "action_state_head",
        "goal_context_proj",
        "goal_effect_head",
        "action_context_residual",
    }
)


def expand_proprio_state_dict(
    *,
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Copy a smaller proprio projection into a zero-padded larger model."""

    source_keys = set(source)
    target_keys = set(target)
    unexpected = sorted(source_keys - target_keys)
    if unexpected:
        raise ValueError(
            "expanded-proprio checkpoint keys do not match target: "
            f"missing=[], unexpected={unexpected}"
        )
    missing = sorted(target_keys - source_keys)
    required_missing = [
        key
        for key in missing
        if key.split(".", 1)[0] not in OPTIONAL_AUXILIARY_PREFIXES
    ]
    if required_missing:
        raise ValueError(
            "expanded-proprio checkpoint keys do not match target: "
            f"missing={required_missing}, unexpected=[]"
        )

    expanded: dict[str, torch.Tensor] = {}
    report_rows: list[dict[str, Any]] = []
    for key, target_value in target.items():
        if key not in source:
            continue
        source_value = source[key]
        if source_value.shape == target_value.shape:
            expanded[key] = source_value
            continue
        if key not in EXPANDABLE_PROPRIO_WEIGHTS:
            raise ValueError(
                "expanded-proprio checkpoint has unsupported shape mismatch for "
                f"{key}: source={tuple(source_value.shape)} "
                f"target={tuple(target_value.shape)}"
            )
        if source_value.ndim != 2 or target_value.ndim != 2:
            raise ValueError(f"expanded proprio weight {key} must be rank two")
        if source_value.shape[0] != target_value.shape[0]:
            raise ValueError(
                f"expanded proprio weight {key} output dimension changed: "
                f"{source_value.shape[0]} != {target_value.shape[0]}"
            )
        old_width = int(source_value.shape[1])
        new_width = int(target_value.shape[1])
        if not 0 < old_width < new_width:
            raise ValueError(
                f"expanded proprio weight {key} must append input columns: "
                f"{old_width} -> {new_width}"
            )
        value = torch.zeros_like(target_value)
        value[:, :old_width] = source_value.to(
            device=value.device,
            dtype=value.dtype,
        )
        expanded[key] = value
        report_rows.append(
            {
                "key": key,
                "source_shape": list(source_value.shape),
                "target_shape": list(target_value.shape),
                "copied_input_columns": old_width,
                "zero_initialized_input_columns": new_width - old_width,
            }
        )

    if {row["key"] for row in report_rows} != set(EXPANDABLE_PROPRIO_WEIGHTS):
        raise ValueError(
            "expanded-proprio init expected both ACT proprio projections to grow; "
            f"expanded={[row['key'] for row in report_rows]}"
        )
    return expanded, {
        "contract_version": "act_append_zero_proprio_columns_v1",
        "expanded": report_rows,
        "missing_optional_keys": missing,
    }
