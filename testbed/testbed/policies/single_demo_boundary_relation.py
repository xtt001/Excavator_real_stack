"""Single-demo, observed-boundary command relation diagnostics.

This module describes command differences from one demo. A differing command
can be *observed-boundary-exempt* when
the corresponding joint is at a train-only qpos boundary and the command points
outward.  The proxy is not a physical-limit ground truth and must not replace
any correctness, safety, task-goal, or generic-liveness gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.policies.deadzone_eval import AXIS_NAMES, effective_direction_mask

DIRECTION_NAMES = ("pos", "neg")
FORBIDDEN_HELDOUT = frozenset({105, 106, 107, 108, 109})


def fit_observed_boundary_proxy(
    *,
    dataset_dir: str | Path,
    train_episode_ids: Sequence[int],
    thresholds: Mapping[str, Mapping[str, float]],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    progress_horizon: int = 4,
    boundary_margin_fraction: float = 0.02,
) -> dict[str, Any]:
    """Fit qpos boundary and command-to-qpos sign proxies on train episodes."""

    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("boundary quantiles must satisfy 0 <= lower < upper <= 1")
    if int(progress_horizon) < 1:
        raise ValueError("progress_horizon must be positive")
    if float(boundary_margin_fraction) < 0.0:
        raise ValueError("boundary_margin_fraction must be non-negative")
    dataset_path = Path(dataset_dir).expanduser().resolve()
    qpos_values: list[list[np.ndarray]] = [[] for _ in AXIS_NAMES]
    deltas: list[list[list[float]]] = [
        [[] for _ in DIRECTION_NAMES] for _ in AXIS_NAMES
    ]
    source_episodes: list[dict[str, Any]] = []
    for episode_id in sorted({int(value) for value in train_episode_ids}):
        if episode_id in FORBIDDEN_HELDOUT:
            raise ValueError(f"held-out episode {episode_id} is forbidden")
        path = dataset_path / f"episode_{episode_id}.hdf5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float32)
            action = np.asarray(handle["/action"][()], dtype=np.float32)
        if (
            qpos.shape != action.shape
            or qpos.ndim != 2
            or qpos.shape[1] != len(AXIS_NAMES)
        ):
            raise ValueError(
                f"episode {episode_id}: qpos/action shape mismatch {qpos.shape} vs {action.shape}"
            )
        for axis_index in range(len(AXIS_NAMES)):
            qpos_values[axis_index].append(qpos[:, axis_index])
            horizon_end = np.minimum(
                np.arange(qpos.shape[0]) + int(progress_horizon), qpos.shape[0] - 1
            )
            delta = qpos[horizon_end, axis_index] - qpos[:, axis_index]
            axis_name = AXIS_NAMES[axis_index]
            positive = action[:, axis_index] >= float(thresholds[axis_name]["pos"])
            negative = action[:, axis_index] <= -float(thresholds[axis_name]["neg"])
            deltas[axis_index][0].extend(delta[positive].astype(float).tolist())
            deltas[axis_index][1].extend(delta[negative].astype(float).tolist())
        source_episodes.append(
            {
                "episode_id": episode_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "steps": int(qpos.shape[0]),
            }
        )

    lower: list[float] = []
    upper: list[float] = []
    margin: list[float] = []
    command_qpos_sign: list[list[int]] = []
    for axis_index in range(len(AXIS_NAMES)):
        values = np.concatenate(qpos_values[axis_index], axis=0)
        axis_lower = float(np.quantile(values, lower_quantile))
        axis_upper = float(np.quantile(values, upper_quantile))
        span = max(axis_upper - axis_lower, 1.0e-6)
        lower.append(axis_lower)
        upper.append(axis_upper)
        margin.append(float(boundary_margin_fraction) * span)
        signs: list[int] = []
        for direction_values in deltas[axis_index]:
            median = float(np.median(direction_values)) if direction_values else 0.0
            signs.append(1 if median > 1.0e-3 else -1 if median < -1.0e-3 else 0)
        command_qpos_sign.append(signs)
    return {
        "contract": "observed_qpos_boundary_proxy_v1",
        "physical_limit_ground_truth": False,
        "dataset_dir": str(dataset_path),
        "train_episode_ids": sorted({int(value) for value in train_episode_ids}),
        "source_episodes": source_episodes,
        "axis_order": list(AXIS_NAMES),
        "direction_order": list(DIRECTION_NAMES),
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "progress_horizon": int(progress_horizon),
        "boundary_margin_fraction": float(boundary_margin_fraction),
        "lower": lower,
        "upper": upper,
        "margin": margin,
        "command_qpos_sign": command_qpos_sign,
    }


def evaluate_single_demo_boundary_relation(
    *,
    eval_episode_ids: Sequence[int],
    dataset_dir: str | Path,
    eval_dir: str | Path,
    thresholds: Mapping[str, Mapping[str, float]],
    boundary_proxy: Mapping[str, Any],
    model: str,
    variant: str = "raw",
) -> dict[str, Any]:
    """Describe demo differences and observed-boundary exemptions."""

    dataset_path = Path(dataset_dir).expanduser().resolve()
    replay_root = Path(eval_dir).expanduser().resolve() / "episodes"
    rows: list[dict[str, Any]] = []
    for episode_id in sorted({int(value) for value in eval_episode_ids}):
        if episode_id in FORBIDDEN_HELDOUT:
            raise ValueError(f"held-out episode {episode_id} is forbidden")
        action_path = replay_root / f"episode_{episode_id}" / "actions.npz"
        if not action_path.exists():
            raise FileNotFoundError(action_path)
        with np.load(action_path) as payload:
            expert = np.asarray(payload["expert_action"], dtype=np.float32)
            policy = np.asarray(payload["policy_action"], dtype=np.float32)
        with h5py.File(dataset_path / f"episode_{episode_id}.hdf5", "r") as handle:
            qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float32)
        if expert.shape != policy.shape or expert.shape != qpos.shape:
            raise ValueError(
                f"episode {episode_id}: replay/qpos shape mismatch "
                f"{expert.shape}, {policy.shape}, {qpos.shape}"
            )
        expert_effective = effective_direction_mask(expert, dict(thresholds))
        policy_effective = effective_direction_mask(policy, dict(thresholds))
        lower = np.asarray(boundary_proxy["lower"], dtype=np.float32)
        upper = np.asarray(boundary_proxy["upper"], dtype=np.float32)
        margin = np.asarray(boundary_proxy["margin"], dtype=np.float32)
        command_sign = np.asarray(boundary_proxy["command_qpos_sign"], dtype=np.int8)
        for step in range(expert.shape[0]):
            target = expert_effective[step]
            current = policy_effective[step]
            extra = current & ~target
            target_limit_exempt = np.zeros_like(target, dtype=bool)
            extra_limit_exempt = np.zeros_like(extra, dtype=bool)
            for axis_index in range(len(AXIS_NAMES)):
                for direction_index in range(len(DIRECTION_NAMES)):
                    movement_sign = int(command_sign[axis_index, direction_index])
                    if movement_sign > 0:
                        at_outward_boundary = (
                            qpos[step, axis_index]
                            >= upper[axis_index] - margin[axis_index]
                        )
                    elif movement_sign < 0:
                        at_outward_boundary = (
                            qpos[step, axis_index]
                            <= lower[axis_index] + margin[axis_index]
                        )
                    else:
                        at_outward_boundary = False
                    target_limit_exempt[axis_index, direction_index] = bool(
                        target[axis_index, direction_index] and at_outward_boundary
                    )
                    extra_limit_exempt[axis_index, direction_index] = bool(
                        extra[axis_index, direction_index] and at_outward_boundary
                    )
            adjusted_target = target & ~target_limit_exempt
            rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "episode_id": episode_id,
                    "step": step,
                    "single_demo_direction_count": int(np.count_nonzero(target)),
                    "boundary_adjusted_demo_direction_count": int(
                        np.count_nonzero(adjusted_target)
                    ),
                    "demo_direction_boundary_exempt_count": int(
                        np.count_nonzero(target_limit_exempt)
                    ),
                    "outside_single_demo_direction_count": int(np.count_nonzero(extra)),
                    "outside_demo_boundary_exempt_count": int(
                        np.count_nonzero(extra_limit_exempt)
                    ),
                    "outside_demo_nonexempt_count": int(
                        np.count_nonzero(extra & ~extra_limit_exempt)
                    ),
                    "qpos": qpos[step].astype(float).tolist(),
                    "outside_single_demo_directions": _directions(extra),
                    "outside_demo_boundary_exempt_directions": _directions(
                        extra_limit_exempt
                    ),
                }
            )
    target_total = sum(row["single_demo_direction_count"] for row in rows)
    adjusted_total = sum(row["boundary_adjusted_demo_direction_count"] for row in rows)
    extra_total = sum(row["outside_single_demo_direction_count"] for row in rows)
    exempt_extra_total = sum(row["outside_demo_boundary_exempt_count"] for row in rows)
    return {
        "contract": "single_demo_boundary_relation_v2",
        "model": model,
        "variant": variant,
        "episode_ids": sorted({int(value) for value in eval_episode_ids}),
        "episodes": len({row["episode_id"] for row in rows}),
        "steps": len(rows),
        "single_demo_direction_opportunities": target_total,
        "demo_direction_boundary_exempt": target_total - adjusted_total,
        "boundary_adjusted_demo_direction_opportunities": adjusted_total,
        "outside_single_demo_effective": extra_total,
        "outside_demo_boundary_exempt": exempt_extra_total,
        "outside_demo_nonexempt": extra_total - exempt_extra_total,
        "rows": rows,
        "physical_limit_ground_truth": False,
        "correctness_estimable": False,
        "task_goal_estimable": False,
        "generic_liveness_estimable": False,
    }


def write_single_demo_boundary_report(
    *,
    output_dir: str | Path,
    report: Mapping[str, Any],
    boundary_proxy: Mapping[str, Any],
    source_paths: Mapping[str, str | Path],
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "boundary_proxy.json").write_text(
        json.dumps(boundary_proxy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_payload = dict(report)
    report_payload["boundary_proxy_sha256"] = sha256_file(
        output / "boundary_proxy.json"
    )
    report_payload["source_paths"] = {
        key: str(Path(value).expanduser().resolve())
        for key, value in source_paths.items()
    }
    report_payload["source_sha256"] = {
        key: sha256_file(value) for key, value in source_paths.items()
    }
    report_path = output / "single_demo_boundary_relation_report.json"
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "model",
        "variant",
        "episode_id",
        "step",
        "single_demo_direction_count",
        "boundary_adjusted_demo_direction_count",
        "demo_direction_boundary_exempt_count",
        "outside_single_demo_direction_count",
        "outside_demo_boundary_exempt_count",
        "outside_demo_nonexempt_count",
        "outside_single_demo_directions",
        "outside_demo_boundary_exempt_directions",
    ]
    with (output / "single_demo_boundary_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({key: row[key] for key in fields})
    return report_path


def _directions(mask: np.ndarray) -> str:
    return ",".join(
        f"{AXIS_NAMES[axis]}{DIRECTION_NAMES[direction][0]}"
        for axis, direction in np.argwhere(mask)
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FORBIDDEN_HELDOUT",
    "evaluate_single_demo_boundary_relation",
    "fit_observed_boundary_proxy",
    "sha256_file",
    "write_single_demo_boundary_report",
]
