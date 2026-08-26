"""Audit condition-action weighting on a fixed validation protocol.

The audit keeps the data split and model architecture from a completed ACT
bundle, then recomputes the deterministic condition classifier loss on repeated
validation draws.  It reports the unweighted classifier CE separately from its
weighted contribution to the aggregate training objective.  This is an
offline loss audit; it does not claim physical target following.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from testbed.actions.policy import _act_policy_config_from_resolved
from testbed.data.dataset import load_data
from testbed.policies.act.adapter import ACTAdapter
from testbed.policies.act.trainer import ACTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-condition-weight-audit",
        description=(
            "Recompute condition-action CE and its weighted contribution on "
            "repeated validation draws."
        ),
    )
    parser.add_argument("--bundle-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    reports = [
        audit_bundle(
            bundle_dir=path,
            repeats=int(args.repeats),
            seed=int(args.seed),
            device=str(args.device),
        )
        for path in args.bundle_dir
    ]
    action_label_scopes = sorted(
        {str(report["condition_action_label_scope"]) for report in reports}
    )
    comparison_valid = len(action_label_scopes) <= 1
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "condition_weight_audit_v1",
                "repeats": int(args.repeats),
                "seed": int(args.seed),
                "comparison_valid": comparison_valid,
                "comparison_error": (
                    None
                    if comparison_valid
                    else "bundles use different condition action label scopes"
                ),
                "condition_action_label_scopes": action_label_scopes,
                "reports": reports,
                "boundary": (
                    "Validation samples are repeated random starts from the "
                    "manifest validation episodes. The reported classifier is "
                    "the deterministic ACT auxiliary head; the mock closed-loop "
                    "report is required for action executability."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "reports": reports}, ensure_ascii=False))


def audit_bundle(
    *, bundle_dir: Path, repeats: int, seed: int, device: str
) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    resolved_path = bundle_dir / "resolved_config.yaml"
    stats_path = bundle_dir / "dataset_stats.pkl"
    checkpoint_path = bundle_dir / "policy_best.ckpt"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}

    task_cfg = dict(resolved.get("task", {}) or {})
    policy_cfg = dict(resolved.get("policy", {}) or {})
    train_cfg = dict(resolved.get("train", {}) or {})
    camera_names = list(task_cfg.get("camera_names", []))
    low_dim_keys = list(policy_cfg.get("low_dim_keys", ["qpos"]))
    action_chunk_size = int((policy_cfg.get("act_params", {}) or {}).get("chunk_size", 100))
    condition_cfg = copy.deepcopy(
        train_cfg.get(
            "condition_adherence_loss",
            policy_cfg.get("condition_adherence_loss", {}),
        )
        or {}
    )
    goal_effect_cfg = copy.deepcopy(
        train_cfg.get("goal_effect", policy_cfg.get("goal_effect", {})) or {}
    )
    # Disable stochastic state-hold sampling for this loss-only audit. The
    # condition labels and same-observation flip remain exactly as configured.
    _train_loader, val_loader, _stats, _is_real, split_info = load_data(
        dataset_dir=Path(task_cfg["dataset_dir"]),
        num_episodes=int(task_cfg.get("num_episodes", 1_000_000)),
        camera_names=camera_names,
        episode_len=(
            None
            if task_cfg.get("episode_len") is None
            else int(task_cfg["episode_len"])
        ),
        batch_size_train=int(train_cfg.get("batch_size", 4)),
        batch_size_val=int(train_cfg.get("batch_size", 4)),
        num_workers=0,
        prefetch_factor=1,
        persistent_workers=False,
        pin_memory=False,
        split_seed=int(train_cfg.get("seed", 0)),
        train_split_ratio=float(train_cfg.get("train_split_ratio", 0.8)),
        split_path=None,
        split_manifest_path=task_cfg.get("split_manifest_path"),
        reuse_split=bool(train_cfg.get("reuse_split", True)),
        low_dim_keys=low_dim_keys,
        episode_ids=_train_ready_ids(task_cfg),
        action_chunk_size=action_chunk_size,
        image_transform=str(train_cfg.get("image_transform", "none")),
        state_hold_transition={"enabled": False},
        condition_adherence_loss_train=condition_cfg,
        goal_effect=goal_effect_cfg,
    )

    policy_config = _act_policy_config_from_resolved(resolved)
    policy = ACTAdapter.from_checkpoint(
        ckpt_path=checkpoint_path,
        policy_config=policy_config,
        norm_stats_path=stats_path,
        device=device,
    )
    policy._model.eval()
    condition_weight = float(
        (train_cfg.get("condition_action_loss", {}) or {}).get("weight", 0.0)
    )
    accum: dict[str, float] = {}
    class_ce_sum = 0.0
    class_eval_count = 0.0
    correct_count = 0.0
    batch_count = 0
    try:
        for repeat_index in range(repeats):
            np.random.seed(seed + repeat_index)
            torch.manual_seed(seed + repeat_index)
            for data in val_loader:
                with torch.inference_mode():
                    terms = ACTTrainer._forward(data, policy, False, None)
                batch_count += 1
                eval_count = float(terms["condition_action_eval_count"].item())
                class_ce = float(terms["condition_action_class_loss"].item())
                if eval_count > 0.0:
                    class_ce_sum += class_ce * eval_count
                    class_eval_count += eval_count
                    correct_count += float(
                        terms["condition_action_correct_count"].item()
                    )
                for key in (
                    "loss",
                    "l1",
                    "kl",
                    "deadzone_loss",
                    "goal_effect_loss",
                    "condition_action_loss",
                ):
                    accum[key] = accum.get(key, 0.0) + float(terms[key].item())
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    batch_mean = {
        key: value / max(1, batch_count) for key, value in accum.items()
    }
    unweighted_ce = class_ce_sum / max(1.0, class_eval_count)
    weighted_ce = condition_weight * unweighted_ce
    batch_mean["loss_without_condition_action"] = (
        batch_mean.get("loss", 0.0) - batch_mean.get("condition_action_loss", 0.0)
    )
    return {
        "bundle_dir": str(bundle_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": _checkpoint_epoch(checkpoint_path),
        "condition_action_weight": condition_weight,
        "condition_action_label_scope": str(
            condition_cfg.get("action_label_scope", "anchor_only")
        ),
        "validation_episode_ids": [int(value) for value in split_info["val_ids"]],
        "repeats": int(repeats),
        "batch_count": int(batch_count),
        "condition_action_eval_count": int(class_eval_count),
        "condition_action_correct_count": int(correct_count),
        "condition_action_accuracy": correct_count / max(1.0, class_eval_count),
        "condition_action_class_ce": unweighted_ce,
        "condition_action_weighted_ce": weighted_ce,
        "batch_mean_terms": batch_mean,
    }


def _train_ready_ids(task_cfg: dict[str, Any]) -> list[int]:
    path = Path(str(task_cfg["train_ready_manifest_path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sorted(
        int(str(value).split("_", 1)[-1])
        for value in payload["train_ready_episode_ids"]
    )


def _checkpoint_epoch(path: Path) -> int | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = payload.get("epoch") if isinstance(payload, dict) else None
    return None if value is None else int(value)


if __name__ == "__main__":
    main()
