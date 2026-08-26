"""
tb-train  — Train a policy on collected HDF5 demonstrations.

Usage
-----
    tb-train --config testbed/configs/act_real_v1.yaml
    tb-train --config testbed/configs/act_real_v1.yaml --resume runs/ckpts/run1/policy_latest.ckpt
    python -m testbed.cli.train --config testbed/configs/act_real_v1.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-train",
        description="Train a policy on HDF5 demonstrations.",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Training YAML config (e.g. testbed/configs/act_real_v1.yaml).",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help="Optional separate task YAML (merged with --config).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help=(
            "Path to an ACT checkpoint used for explicit low-dimension input "
            "expansion. Existing projection columns are copied and new input "
            "columns are zero-initialized."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.num_epochs.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Override train.ckpt_dir.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Override task.dataset_dir.",
    )
    parser.add_argument(
        "--train-ready-manifest",
        type=Path,
        default=None,
        help="Override task.train_ready_manifest_path.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Override task.split_manifest_path.",
    )
    parser.add_argument(
        "--condition-action-weight",
        type=float,
        default=None,
        help=(
            "Override train.condition_action_loss.weight for an explicit "
            "single-factor condition-loss experiment."
        ),
    )
    parser.add_argument(
        "--target-release-weight",
        type=float,
        default=None,
        help=(
            "Override train.target_release_loss.weight for a calibrated "
            "direct-action release experiment."
        ),
    )
    args = parser.parse_args()
    if args.resume and args.warm_start:
        parser.error("--resume and --warm-start are mutually exclusive")

    config: dict = {}
    if args.task_config:
        with open(args.task_config) as f:
            config.update(yaml.safe_load(f) or {})
    with open(args.config) as f:
        config.update(yaml.safe_load(f) or {})

    # CLI overrides
    train = config.setdefault("train", {})
    task = config.setdefault("task", {})
    if args.resume:
        train["resume_ckpt"] = str(args.resume)
    if args.warm_start:
        train["warm_start_ckpt"] = str(args.warm_start)
    if args.epochs is not None:
        train["num_epochs"] = args.epochs
    if args.ckpt_dir is not None:
        train["ckpt_dir"] = str(args.ckpt_dir)
    if args.dataset_dir is not None:
        task["dataset_dir"] = str(args.dataset_dir)
    if args.train_ready_manifest is not None:
        task["train_ready_manifest_path"] = str(args.train_ready_manifest)
    if args.split_manifest is not None:
        task["split_manifest_path"] = str(args.split_manifest)
    if args.condition_action_weight is not None:
        if args.condition_action_weight < 0.0:
            parser.error("--condition-action-weight must be non-negative")
        config.setdefault("train", {}).setdefault(
            "condition_action_loss", {}
        )["weight"] = float(args.condition_action_weight)
    if args.target_release_weight is not None:
        if args.target_release_weight < 0.0:
            parser.error("--target-release-weight must be non-negative")
        config.setdefault("train", {}).setdefault(
            "target_release_loss", {}
        )["weight"] = float(args.target_release_weight)

    from testbed.runtime.runner import Runner
    Runner(config).train()


if __name__ == "__main__":
    main()
