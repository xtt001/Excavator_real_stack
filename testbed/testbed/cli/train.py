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
    args = parser.parse_args()

    config: dict = {}
    if args.task_config:
        with open(args.task_config) as f:
            config.update(yaml.safe_load(f) or {})
    with open(args.config) as f:
        config.update(yaml.safe_load(f) or {})

    # CLI overrides
    train = config.setdefault("train", {})
    if args.resume:
        train["resume_ckpt"] = str(args.resume)
    if args.epochs is not None:
        train["num_epochs"] = args.epochs
    if args.ckpt_dir is not None:
        train["ckpt_dir"] = str(args.ckpt_dir)

    from testbed.runtime.runner import Runner
    Runner(config).train()


if __name__ == "__main__":
    main()
