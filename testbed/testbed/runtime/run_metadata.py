"""Helpers for writing resolved configs and run metadata artifacts."""

from __future__ import annotations

import copy
import datetime
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def write_resolved_config(path: str | Path, config: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(_stringify_paths(copy.deepcopy(config)), f, sort_keys=False)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    return path


def build_train_run_metadata(
    *,
    dataset_dir: Path,
    ckpt_dir: Path,
    resolved_config_path: Path,
    dataset_stats_path: Path,
    split_info: dict[str, Any],
    policy_class: str,
    task_name: str,
    device: str,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "run_type": "train",
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "task_name": str(task_name),
        "policy_class": str(policy_class),
        "device_requested": str(device),
        "command": shlex.join(sys.argv),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "paths": {
            "dataset_dir": str(dataset_dir.resolve()),
            "ckpt_dir": str(ckpt_dir.resolve()),
            "resolved_config": str(resolved_config_path.resolve()),
            "dataset_stats": str(dataset_stats_path.resolve()),
            "train_val_split": str(split_info.get("split_path", "")),
        },
        "split": _stringify_paths(copy.deepcopy(split_info)),
        "environment": _collect_environment_snapshot(),
        "repo_snapshots": {
            "repo_a": _collect_repo_snapshot(repo_root),
        },
    }


def _collect_environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
    }

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        snapshot["torch"] = {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "device_names": (
                [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                if cuda_available else []
            ),
        }
    except Exception as exc:  # pragma: no cover - best effort metadata
        snapshot["torch"] = {
            "error": f"{type(exc).__name__}: {exc}",
        }

    return snapshot


def _collect_repo_snapshot(repo_path: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "path": str(repo_path.resolve()),
    }
    if not (repo_path / ".git").exists():
        snapshot["git_available"] = False
        return snapshot

    snapshot["git_available"] = True
    snapshot["commit"] = _run_git(repo_path, "rev-parse", "HEAD")
    snapshot["branch"] = _run_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git(repo_path, "status", "--short")
    snapshot["dirty"] = bool(status)
    snapshot["status_short"] = status.splitlines()[:20] if status else []
    return snapshot


def _run_git(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return f"<git-error:{' '.join(args)}:{result.stderr.strip()}>"
    return result.stdout.strip()


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _stringify_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(v) for v in value]
    if isinstance(value, tuple):
        return [_stringify_paths(v) for v in value]
    return value
