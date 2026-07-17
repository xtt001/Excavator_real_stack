"""Storage policy for ACT training and inference checkpoints."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_SCHEMA_VERSION = 2


def _checkpoint_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_name": config.get("task_name", ""),
        "seed": int(config.get("seed", 0)),
        "policy_class": "ACT",
    }


def build_inference_checkpoint(
    *,
    model_state_dict: Mapping[str, Any],
    epoch: int,
    min_val_loss: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a model-only payload accepted by ``ACTAdapter.from_checkpoint``."""

    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": "inference",
        "model_state_dict": model_state_dict,
        "epoch": int(epoch),
        "min_val_loss": float(min_val_loss),
        "config": _checkpoint_metadata(config),
    }


def build_resume_checkpoint(
    *,
    model_state_dict: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any],
    epoch: int,
    min_val_loss: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete state needed to resume at the following epoch."""

    payload = build_inference_checkpoint(
        model_state_dict=model_state_dict,
        epoch=epoch,
        min_val_loss=min_val_loss,
        config=config,
    )
    payload["checkpoint_kind"] = "resume"
    payload["optimizer_state_dict"] = optimizer_state_dict
    return payload


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    """Write a torch payload in the destination directory, then atomically replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            torch.save(payload, temp_file)
            temp_file.flush()
        os.replace(temp_path, destination)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return destination


def load_resume_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a resume-capable checkpoint, including legacy payloads."""

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "resume_ckpt must contain model_state_dict and optimizer_state_dict"
        )
    if "optimizer_state_dict" not in checkpoint:
        kind = checkpoint.get("checkpoint_kind", "model-only")
        raise ValueError(
            f"resume_ckpt is not resume-capable (checkpoint_kind={kind!r}); "
            "use policy_latest.ckpt"
        )
    return checkpoint


class ACTCheckpointPersistence:
    """Own checkpoint schemas, atomic replacement, links, and periodic retention."""

    def __init__(
        self,
        ckpt_dir: str | Path,
        *,
        seed: int,
        periodic_keep_last: int,
    ) -> None:
        if int(periodic_keep_last) < 1:
            raise ValueError("checkpoint_keep_last must be at least 1")
        self.ckpt_dir = Path(ckpt_dir)
        self.seed = int(seed)
        self.periodic_keep_last = int(periodic_keep_last)

    @property
    def latest_path(self) -> Path:
        return self.ckpt_dir / "policy_latest.ckpt"

    @property
    def best_path(self) -> Path:
        return self.ckpt_dir / "policy_best.ckpt"

    @property
    def last_path(self) -> Path:
        return self.ckpt_dir / "policy_last.ckpt"

    def periodic_path(self, epoch: int) -> Path:
        return self.ckpt_dir / f"policy_epoch_{int(epoch)}_seed_{self.seed}.ckpt"

    def save_resume(
        self,
        *,
        model_state_dict: Mapping[str, Any],
        optimizer_state_dict: Mapping[str, Any],
        epoch: int,
        min_val_loss: float,
        config: Mapping[str, Any],
    ) -> Path:
        return atomic_torch_save(
            build_resume_checkpoint(
                model_state_dict=model_state_dict,
                optimizer_state_dict=optimizer_state_dict,
                epoch=epoch,
                min_val_loss=min_val_loss,
                config=config,
            ),
            self.latest_path,
        )

    def save_best(
        self,
        *,
        model_state_dict: Mapping[str, Any],
        epoch: int,
        min_val_loss: float,
        config: Mapping[str, Any],
    ) -> Path:
        return atomic_torch_save(
            build_inference_checkpoint(
                model_state_dict=model_state_dict,
                epoch=epoch,
                min_val_loss=min_val_loss,
                config=config,
            ),
            self.best_path,
        )

    def save_periodic(
        self,
        *,
        model_state_dict: Mapping[str, Any],
        epoch: int,
        min_val_loss: float,
        config: Mapping[str, Any],
    ) -> Path:
        path = atomic_torch_save(
            build_inference_checkpoint(
                model_state_dict=model_state_dict,
                epoch=epoch,
                min_val_loss=min_val_loss,
                config=config,
            ),
            self.periodic_path(epoch),
        )
        self.prune_periodic()
        return path

    def prune_periodic(self) -> list[Path]:
        """Delete only older files in this run's exact periodic naming pattern."""

        pattern = re.compile(
            rf"^policy_epoch_(\d+)_seed_{re.escape(str(self.seed))}\.ckpt$"
        )
        candidates: list[tuple[int, Path]] = []
        for path in self.ckpt_dir.glob("policy_epoch_*_seed_*.ckpt"):
            match = pattern.fullmatch(path.name)
            if match is not None:
                candidates.append((int(match.group(1)), path))
        candidates.sort(key=lambda item: item[0])
        removed: list[Path] = []
        for _, path in candidates[: -self.periodic_keep_last]:
            path.unlink()
            removed.append(path)
        return removed

    def link_last_to_latest(self) -> Path:
        """Atomically make ``policy_last.ckpt`` a hard link to final latest."""

        if not self.latest_path.is_file():
            raise FileNotFoundError(self.latest_path)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        fd, raw_temp_path = tempfile.mkstemp(
            prefix=f".{self.last_path.name}.",
            suffix=".tmp",
            dir=self.ckpt_dir,
        )
        os.close(fd)
        temp_path = Path(raw_temp_path)
        temp_path.unlink()
        try:
            os.link(self.latest_path, temp_path)
            os.replace(temp_path, self.last_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        return self.last_path
