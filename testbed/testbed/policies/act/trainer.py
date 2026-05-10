"""
ACT offline trainer.

Checkpoint format:
  {model_state_dict, optimizer_state_dict, epoch, min_val_loss, config}
"""

from __future__ import annotations

import os
import pickle
import re
from contextlib import nullcontext, redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from testbed.policies.base import Trainer, compute_dict_mean, detach_dict, set_seed
from testbed.policies.act.adapter import ACTAdapter


class ACTTrainer(Trainer):
    """
    Offline behaviour-cloning trainer for ACT.

    Parameters
    ----------
    policy_config  Dict passed to ACTAdapter (and downstream to detr).
    config         Training hyperparameters (see __init__).
    """

    def __init__(self, policy_config: dict, config: dict):
        self.policy_config = policy_config
        self.config = config

    # ── Trainer ABC ───────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader,
        val_loader,
        config: dict | None = None,
    ) -> tuple[int, float, dict]:
        """
        Full training loop.

        Returns
        -------
        (best_epoch, min_val_loss, best_state_dict)
        """
        cfg = config or self.config
        num_epochs = cfg["num_epochs"]
        ckpt_dir   = Path(cfg["ckpt_dir"])
        seed       = cfg["seed"]
        resume     = cfg.get("resume_ckpt")
        device     = str(cfg.get("device", "cuda"))
        val_every  = max(1, int(cfg.get("val_every", 1)))
        save_latest_every = max(1, int(cfg.get("save_latest_every", 1)))
        checkpoint_every = max(1, int(cfg.get("checkpoint_every", 100)))
        plot_every = max(1, int(cfg.get("plot_every", checkpoint_every)))
        amp_enabled = bool(cfg.get("amp", False))
        amp_dtype_name = str(cfg.get("amp_dtype", "auto"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        set_seed(seed)

        # Build adapter (model + optimizer)
        norm_stats_path = ckpt_dir / "dataset_stats.pkl"
        with open(norm_stats_path, "rb") as f:
            norm_stats = pickle.load(f)

        adapter   = ACTAdapter(self.policy_config, norm_stats, device=device)
        optimizer = adapter.configure_optimizers()

        min_val_loss  = float("inf")
        best_ckpt     = None
        start_epoch   = 0
        train_history: list[dict] = []
        val_history:   list[dict] = []
        val_epochs:    list[int] = []

        # ── optional resume ───────────────────────────────────────────────────
        if resume:
            ckpt_obj = torch.load(resume, map_location="cpu")
            sd = ckpt_obj["model_state_dict"] if "model_state_dict" in ckpt_obj else ckpt_obj
            adapter.load_state_dict(sd)
            if "optimizer_state_dict" in ckpt_obj:
                optimizer.load_state_dict(ckpt_obj["optimizer_state_dict"])
            start_epoch = self._infer_start_epoch(resume, ckpt_obj, cfg)
            if "min_val_loss" in ckpt_obj:
                min_val_loss = float(ckpt_obj["min_val_loss"])
            print(f"Resumed from {resume}, starting epoch {start_epoch}")

        use_amp = amp_enabled and device.startswith("cuda") and torch.cuda.is_available()
        amp_dtype = self._resolve_amp_dtype(amp_dtype_name) if use_amp else None
        scaler = self._build_grad_scaler(use_amp, amp_dtype)
        amp_label = self._format_amp_label(use_amp, amp_dtype)
        print(
            "Training settings:"
            f" val_every={val_every},"
            f" save_latest_every={save_latest_every},"
            f" checkpoint_every={checkpoint_every},"
            f" plot_every={plot_every},"
            f" amp={amp_label}"
        )

        # ── training loop ─────────────────────────────────────────────────────
        for epoch in tqdm(range(start_epoch, num_epochs)):
            should_validate = (
                epoch == start_epoch
                or epoch == num_epochs - 1
                or (epoch - start_epoch) % val_every == 0
            )

            if should_validate:
                adapter._model.eval()
                with torch.inference_mode():
                    ep_dicts = []
                    for data in val_loader:
                        loss_d = self._forward(data, adapter, use_amp, amp_dtype)
                        ep_dicts.append(loss_d)
                    ep_summary = compute_dict_mean(ep_dicts)
                    val_history.append(ep_summary)
                    val_epochs.append(epoch)
                    epoch_val_loss = ep_summary["loss"]
                    if epoch_val_loss < min_val_loss:
                        min_val_loss = epoch_val_loss
                        best_ckpt = (epoch, min_val_loss, deepcopy(adapter.state_dict()))

                self._print_summary("Val", epoch, ep_summary)
            else:
                print(f"Epoch {epoch} [Val] skipped (val_every={val_every})")

            # training
            adapter._model.train()
            optimizer.zero_grad(set_to_none=True)
            last_batch_idx = -1
            for batch_idx, data in enumerate(train_loader):
                loss_d = self._forward(data, adapter, use_amp, amp_dtype)
                loss = loss_d["loss"]
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_history.append(detach_dict(loss_d))
                last_batch_idx = batch_idx

            if last_batch_idx < 0:
                raise ValueError("Train dataloader yielded 0 batches.")

            bs = last_batch_idx + 1
            ep_tr = compute_dict_mean(
                train_history[bs * (epoch - start_epoch): bs * (epoch - start_epoch + 1)]
            )
            self._print_summary("Train", epoch, ep_tr)

            # periodic checkpoint
            if epoch == start_epoch or (epoch + 1) % checkpoint_every == 0:
                self._save_ckpt(
                    ckpt_dir / f"policy_epoch_{epoch}_seed_{seed}.ckpt",
                    adapter, optimizer, epoch, min_val_loss, cfg,
                )
            if epoch == start_epoch or (epoch + 1) % plot_every == 0:
                self._plot_history(train_history, val_history, val_epochs, num_epochs, ckpt_dir, seed)

            if epoch == num_epochs - 1 or (epoch + 1) % save_latest_every == 0:
                self._save_ckpt(
                    ckpt_dir / "policy_latest.ckpt",
                    adapter, optimizer, epoch, min_val_loss, cfg,
                )

        # final checkpoints
        self._save_ckpt(
            ckpt_dir / "policy_latest.ckpt",
            adapter, optimizer, num_epochs - 1, min_val_loss, cfg,
        )
        self._save_ckpt(
            ckpt_dir / "policy_last.ckpt",
            adapter, optimizer, num_epochs - 1, min_val_loss, cfg,
        )
        if best_ckpt is None:
            raise RuntimeError("No validation summary was produced; cannot determine best checkpoint.")
        best_epoch, bvl, best_sd = best_ckpt
        self._save_ckpt(
            ckpt_dir / f"policy_epoch_{best_epoch}_seed_{seed}.ckpt",
            adapter, optimizer, best_epoch, bvl, cfg, sd_override=best_sd,
        )
        # also save a stable "best" checkpoint name for downstream inference
        self._save_ckpt(
            ckpt_dir / "policy_best.ckpt",
            adapter, optimizer, best_epoch, bvl, cfg, sd_override=best_sd,
        )
        self._plot_history(train_history, val_history, val_epochs, num_epochs, ckpt_dir, seed)
        print(f"Training done. Best epoch={best_epoch}, val loss={bvl:.6f}")
        return best_epoch, bvl, best_sd

    def save(self, ckpt_dir: Path | str, tag: str = "best") -> Path:
        """No-op here; saving handled inside fit(). Provided for ABC compliance."""
        return Path(ckpt_dir) / f"policy_{tag}.ckpt"

    def load(self, ckpt_path: Path | str) -> ACTAdapter:
        """Load a checkpoint and return a ready-to-use ACTAdapter."""
        ckpt_dir = Path(ckpt_path).parent
        norm_stats_path = ckpt_dir / "dataset_stats.pkl"
        return ACTAdapter.from_checkpoint(
            ckpt_path=ckpt_path,
            policy_config=self.policy_config,
            norm_stats_path=norm_stats_path,
            device=str(self.config.get("device", "cuda")),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _forward(
        data,
        adapter: ACTAdapter,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> dict:
        image_data, proprio_data, action_data, is_pad = data
        image_data  = image_data.to(adapter.device)
        proprio_data = proprio_data.to(adapter.device)
        action_data = action_data.to(adapter.device)
        is_pad      = is_pad.to(adapter.device)
        with ACTTrainer._autocast_context(adapter.device, amp_enabled, amp_dtype):
            return adapter.forward_loss(proprio_data, image_data, action_data, is_pad)

    @staticmethod
    def _save_ckpt(path, adapter, optimizer, epoch, val_loss, config, sd_override=None):
        torch.save(
            {
                "model_state_dict":     sd_override or adapter.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch":                epoch,
                "min_val_loss":         float(val_loss),
                "config": {
                    "task_name":    config.get("task_name", ""),
                    "seed":         config.get("seed", 0),
                    "policy_class": "ACT",
                },
            },
            path,
        )

    @staticmethod
    def _infer_start_epoch(resume_path: str, ckpt_obj: dict, cfg: dict) -> int:
        if cfg.get("start_epoch") is not None:
            return int(cfg["start_epoch"])
        if "epoch" in ckpt_obj:
            return int(ckpt_obj["epoch"]) + 1
        m = re.search(r"policy_epoch_(\d+)", os.path.basename(resume_path))
        return int(m.group(1)) + 1 if m else 0

    @staticmethod
    def _print_summary(tag: str, epoch: int, d: dict) -> None:
        parts = " ".join(f"{k}:{v:.4f}" for k, v in d.items())
        print(f"Epoch {epoch} [{tag}] {parts}")

    @staticmethod
    def _plot_history(train_history, val_history, val_epochs, num_epochs, ckpt_dir, seed):
        if not train_history:
            return
        try:
            with redirect_stderr(StringIO()):
                import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Skipping training plots because matplotlib is unavailable: {exc}")
            return
        for key in train_history[0]:
            plot_path = ckpt_dir / f"train_val_{key}_seed_{seed}.png"
            plt.figure()
            tv = [d[key].item() if hasattr(d[key], "item") else d[key] for d in train_history]
            vv = [d[key].item() if hasattr(d[key], "item") else d[key] for d in val_history]
            plt.plot(np.linspace(0, num_epochs - 1, len(tv)), tv,  label="train")
            if vv:
                val_x = val_epochs if val_epochs else np.linspace(0, num_epochs - 1, len(vv))
                plt.plot(val_x, vv, label="val")
            plt.tight_layout()
            plt.legend()
            plt.title(key)
            plt.savefig(plot_path)
            plt.close()
        print(f"Plots saved to {ckpt_dir}")

    @staticmethod
    def _autocast_context(device, amp_enabled: bool, amp_dtype: torch.dtype | None):
        if not amp_enabled or amp_dtype is None:
            return nullcontext()
        device_str = str(device)
        device_type = "cuda" if device_str.startswith("cuda") else "cpu"
        return torch.autocast(device_type=device_type, dtype=amp_dtype)

    @staticmethod
    def _resolve_amp_dtype(amp_dtype_name: str) -> torch.dtype:
        key = str(amp_dtype_name).strip().lower()
        if key in {"", "auto"}:
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            return torch.bfloat16 if bf16_supported else torch.float16
        if key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if key in {"fp16", "float16", "half"}:
            return torch.float16
        raise ValueError(f"Unsupported amp_dtype={amp_dtype_name!r}. Use auto, bf16, or fp16.")

    @staticmethod
    def _format_amp_label(amp_enabled: bool, amp_dtype: torch.dtype | None) -> str:
        if not amp_enabled or amp_dtype is None:
            return "disabled"
        if amp_dtype == torch.bfloat16:
            return "enabled(bf16)"
        if amp_dtype == torch.float16:
            return "enabled(fp16)"
        return f"enabled({amp_dtype})"

    @staticmethod
    def _build_grad_scaler(amp_enabled: bool, amp_dtype: torch.dtype | None):
        scaler_enabled = amp_enabled and amp_dtype == torch.float16
        grad_scaler_cls = getattr(torch.amp, "GradScaler", None)
        if grad_scaler_cls is not None:
            return grad_scaler_cls("cuda", enabled=scaler_enabled)
        return torch.cuda.amp.GradScaler(enabled=scaler_enabled)
