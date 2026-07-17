"""
ACT offline trainer.

``policy_latest.ckpt`` is resume-capable. Best and periodic candidates are
model-only inference checkpoints.
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

from testbed.policies.act.adapter import ACTAdapter
from testbed.policies.act.checkpoint_init import expand_proprio_state_dict
from testbed.policies.act.checkpoint_persistence import (
    ACTCheckpointPersistence,
    load_resume_checkpoint,
)
from testbed.policies.base import Trainer, compute_dict_mean, detach_dict, set_seed


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
        ckpt_dir = Path(cfg["ckpt_dir"])
        seed = cfg["seed"]
        resume = cfg.get("resume_ckpt")
        init_ckpt = cfg.get("init_ckpt")
        device = str(cfg.get("device", "cuda"))
        val_every = max(1, int(cfg.get("val_every", 1)))
        save_latest_every = max(1, int(cfg.get("save_latest_every", 100)))
        checkpoint_every = max(1, int(cfg.get("checkpoint_every", 500)))
        checkpoint_keep_last = max(1, int(cfg.get("checkpoint_keep_last", 3)))
        plot_every = max(1, int(cfg.get("plot_every", checkpoint_every)))
        amp_enabled = bool(cfg.get("amp", False))
        amp_dtype_name = str(cfg.get("amp_dtype", "auto"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = ACTCheckpointPersistence(
            ckpt_dir,
            seed=seed,
            periodic_keep_last=checkpoint_keep_last,
        )

        set_seed(seed)

        # Build adapter (model + optimizer)
        norm_stats_path = ckpt_dir / "dataset_stats.pkl"
        with open(norm_stats_path, "rb") as f:
            norm_stats = pickle.load(f)

        adapter = ACTAdapter(self.policy_config, norm_stats, device=device)
        optimizer = adapter.configure_optimizers()

        min_val_loss = float("inf")
        best_ckpt = None
        start_epoch = 0
        train_history: list[dict] = []
        val_history: list[dict] = []
        val_epochs: list[int] = []

        if resume and init_ckpt:
            raise ValueError("resume_ckpt and init_ckpt are mutually exclusive")
        if bool(cfg.get("init_expand_proprio", False)) and not init_ckpt:
            raise ValueError("init_expand_proprio requires init_ckpt")

        # ── optional model-only initialization or full resume ─────────────────
        if init_ckpt:
            init_state = self._load_model_state_dict(init_ckpt)
            if bool(cfg.get("init_expand_proprio", False)):
                init_state, expansion_report = expand_proprio_state_dict(
                    source=init_state,
                    target=adapter.state_dict(),
                )
                print(f"Expanded proprio checkpoint input: {expansion_report}")
            init_strict = not bool(cfg.get("init_allow_missing", False))
            incompatible = adapter.load_state_dict(init_state, strict=init_strict)
            if not init_strict:
                print(
                    "Initialized with non-strict checkpoint expansion: "
                    f"missing={list(incompatible.missing_keys)}, "
                    f"unexpected={list(incompatible.unexpected_keys)}"
                )
            print(f"Initialized model weights from {init_ckpt}, starting epoch 0")

        if resume:
            ckpt_obj = load_resume_checkpoint(resume)
            adapter.load_state_dict(ckpt_obj["model_state_dict"])
            optimizer.load_state_dict(ckpt_obj["optimizer_state_dict"])
            start_epoch = self._infer_start_epoch(resume, ckpt_obj, cfg)
            if "min_val_loss" in ckpt_obj:
                min_val_loss = float(ckpt_obj["min_val_loss"])
            print(f"Resumed from {resume}, starting epoch {start_epoch}")

        removed_periodic = checkpoints.prune_periodic()
        if removed_periodic:
            print(
                "Pruned older periodic checkpoints: "
                + ", ".join(path.name for path in removed_periodic)
            )

        use_amp = (
            amp_enabled and device.startswith("cuda") and torch.cuda.is_available()
        )
        amp_dtype = self._resolve_amp_dtype(amp_dtype_name) if use_amp else None
        scaler = self._build_grad_scaler(use_amp, amp_dtype)
        amp_label = self._format_amp_label(use_amp, amp_dtype)
        print(
            "Training settings:"
            f" val_every={val_every},"
            f" save_latest_every={save_latest_every},"
            f" checkpoint_every={checkpoint_every},"
            f" checkpoint_keep_last={checkpoint_keep_last},"
            f" plot_every={plot_every},"
            f" amp={amp_label}"
        )

        # ── training loop ─────────────────────────────────────────────────────
        latest_saved_epoch: int | None = None
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
                        loss_d = self._forward(
                            data,
                            adapter,
                            use_amp,
                            amp_dtype,
                            non_blocking=bool(cfg.get("pin_memory", False)),
                        )
                        ep_dicts.append(loss_d)
                    ep_summary = compute_dict_mean(ep_dicts)
                    val_history.append(ep_summary)
                    val_epochs.append(epoch)
                    epoch_val_loss = ep_summary["loss"]
                    if epoch_val_loss < min_val_loss:
                        min_val_loss = epoch_val_loss
                        best_ckpt = (
                            epoch,
                            min_val_loss,
                            deepcopy(adapter.state_dict()),
                        )

                self._print_summary("Val", epoch, ep_summary)
            else:
                print(f"Epoch {epoch} [Val] skipped (val_every={val_every})")

            # training
            adapter._model.train()
            optimizer.zero_grad(set_to_none=True)
            last_batch_idx = -1
            for batch_idx, data in enumerate(train_loader):
                loss_d = self._forward(
                    data,
                    adapter,
                    use_amp,
                    amp_dtype,
                    non_blocking=bool(cfg.get("pin_memory", False)),
                )
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
                train_history[
                    bs * (epoch - start_epoch) : bs * (epoch - start_epoch + 1)
                ]
            )
            self._print_summary("Train", epoch, ep_tr)

            # periodic checkpoint
            if (epoch + 1) % checkpoint_every == 0:
                checkpoints.save_periodic(
                    model_state_dict=adapter.state_dict(),
                    epoch=epoch,
                    min_val_loss=min_val_loss,
                    config=cfg,
                )
            if epoch == start_epoch or (epoch + 1) % plot_every == 0:
                self._plot_history(
                    train_history, val_history, val_epochs, num_epochs, ckpt_dir, seed
                )

            if epoch == num_epochs - 1 or (epoch + 1) % save_latest_every == 0:
                checkpoints.save_resume(
                    model_state_dict=adapter.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    epoch=epoch,
                    min_val_loss=min_val_loss,
                    config=cfg,
                )
                latest_saved_epoch = epoch

        # final checkpoints
        if latest_saved_epoch != num_epochs - 1:
            checkpoints.save_resume(
                model_state_dict=adapter.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=num_epochs - 1,
                min_val_loss=min_val_loss,
                config=cfg,
            )
        checkpoints.link_last_to_latest()
        if best_ckpt is None:
            raise RuntimeError(
                "No validation summary was produced; cannot determine best checkpoint."
            )
        best_epoch, bvl, best_sd = best_ckpt
        checkpoints.save_best(
            model_state_dict=best_sd,
            epoch=best_epoch,
            min_val_loss=bvl,
            config=cfg,
        )
        self._plot_history(
            train_history, val_history, val_epochs, num_epochs, ckpt_dir, seed
        )
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
        non_blocking: bool = False,
    ) -> dict:
        extra: dict = {}
        if isinstance(data, dict):
            image_data = data["image"]
            proprio_data = data["proprio"]
            action_data = data["action"]
            is_pad = data["is_pad"]
            if "raw_action" in data:
                extra["raw_action"] = data["raw_action"].to(
                    adapter.device, non_blocking=non_blocking
                )
            for key in (
                "deadzone_move_mask",
                "deadzone_stop_mask",
                "deadzone_wrong_mask",
                "action_loss_mask",
                "state_hold_transition_mask",
                "goal_future_delta",
                "goal_future_valid",
                "goal_future_direction",
                "goal_effect_delta",
                "goal_effect_valid",
                "action_state_labels",
                "action_state_valid",
                "action_state_persistent_effective",
                "effective_action_phase",
                "effective_action_valid",
                "effective_action_loss_weight",
            ):
                if key in data:
                    extra[key] = data[key].to(adapter.device, non_blocking=non_blocking)
        else:
            image_data, proprio_data, action_data, is_pad = data
        image_data = image_data.to(adapter.device, non_blocking=non_blocking)
        proprio_data = proprio_data.to(adapter.device, non_blocking=non_blocking)
        action_data = action_data.to(adapter.device, non_blocking=non_blocking)
        is_pad = is_pad.to(adapter.device, non_blocking=non_blocking)
        with ACTTrainer._autocast_context(adapter.device, amp_enabled, amp_dtype):
            result = adapter.forward_loss(
                proprio_data, image_data, action_data, is_pad, **extra
            )
            if not isinstance(data, dict) or (
                "execution_feedback_counterfactual_proprio" not in data
            ):
                return result

            counterfactual_mask = data["execution_feedback_counterfactual_mask"].to(
                adapter.device, dtype=torch.bool, non_blocking=non_blocking
            )
            if (
                counterfactual_mask.ndim != 1
                or counterfactual_mask.shape[0] != (proprio_data.shape[0])
            ):
                raise ValueError(
                    "execution_feedback_counterfactual_mask must have shape (B,)"
                )
            weights = data["execution_feedback_counterfactual_loss_weight"].to(
                adapter.device,
                dtype=result["loss"].dtype,
                non_blocking=non_blocking,
            )
            if weights.ndim != 1 or weights.shape != counterfactual_mask.shape:
                raise ValueError(
                    "execution_feedback_counterfactual_loss_weight must have shape (B,)"
                )
            if not torch.isfinite(weights).all() or torch.any(weights < 0.0):
                raise ValueError(
                    "execution_feedback counterfactual loss weight must be finite "
                    "and non-negative"
                )
            if not torch.allclose(weights, weights[0].expand_as(weights)):
                raise ValueError(
                    "execution_feedback counterfactual loss weight must be "
                    "constant within a batch"
                )
            counterfactual_weight = weights[0]
            zero = result["loss"].new_zeros(())
            counterfactual_result: dict[str, torch.Tensor] | None = None
            active_indices = torch.nonzero(
                counterfactual_mask, as_tuple=False
            ).flatten()
            if active_indices.numel() > 0:
                variants = data["execution_feedback_counterfactual_proprio"].to(
                    adapter.device, non_blocking=non_blocking
                )
                expected_shape = (
                    proprio_data.shape[0],
                    2,
                    proprio_data.shape[1],
                )
                if tuple(variants.shape) != expected_shape:
                    raise ValueError(
                        "execution_feedback_counterfactual_proprio must have "
                        f"shape {expected_shape}, got {tuple(variants.shape)}"
                    )
                counterfactual_proprio = variants.index_select(
                    0, active_indices
                ).reshape(-1, proprio_data.shape[1])

                def repeat_active(value: torch.Tensor) -> torch.Tensor:
                    return value.index_select(0, active_indices).repeat_interleave(
                        2, dim=0
                    )

                counterfactual_extra = {
                    key: repeat_active(value) for key, value in extra.items()
                }
                counterfactual_result = adapter.forward_loss(
                    counterfactual_proprio,
                    repeat_active(image_data),
                    repeat_active(action_data),
                    repeat_active(is_pad),
                    **counterfactual_extra,
                )
                counterfactual_loss = counterfactual_result["loss"]
            else:
                counterfactual_loss = zero
            weighted_counterfactual_loss = counterfactual_weight * counterfactual_loss
            result = dict(result)
            result.update(
                {
                    "execution_feedback_counterfactual_samples": (
                        active_indices.numel() * result["loss"].new_ones(())
                    ),
                    "execution_feedback_counterfactual_loss": (counterfactual_loss),
                    "execution_feedback_counterfactual_weighted_loss": (
                        weighted_counterfactual_loss
                    ),
                    "execution_feedback_counterfactual_l1": (
                        zero
                        if counterfactual_result is None
                        else counterfactual_result["l1"]
                    ),
                    "execution_feedback_counterfactual_intent_loss": (
                        zero
                        if counterfactual_result is None
                        else counterfactual_result["intent_loss"]
                    ),
                    "execution_feedback_counterfactual_state_hold_loss": (
                        zero
                        if counterfactual_result is None
                        else counterfactual_result["demo_target_hold_loss"]
                    ),
                    "loss": result["loss"] + weighted_counterfactual_loss,
                }
            )
            return result

    @staticmethod
    def _infer_start_epoch(resume_path: str, ckpt_obj: dict, cfg: dict) -> int:
        if cfg.get("start_epoch") is not None:
            return int(cfg["start_epoch"])
        if "epoch" in ckpt_obj:
            return int(ckpt_obj["epoch"]) + 1
        m = re.search(r"policy_epoch_(\d+)", os.path.basename(resume_path))
        return int(m.group(1)) + 1 if m else 0

    @staticmethod
    def _load_model_state_dict(checkpoint_path: str | Path) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(
                "init_ckpt must contain a model state mapping or a "
                "model_state_dict entry"
            )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("init_ckpt model state mapping must not be empty")
        return state_dict

    @staticmethod
    def _print_summary(tag: str, epoch: int, d: dict) -> None:
        parts = " ".join(f"{k}:{v:.4f}" for k, v in d.items())
        print(f"Epoch {epoch} [{tag}] {parts}")

    @staticmethod
    def _plot_history(
        train_history, val_history, val_epochs, num_epochs, ckpt_dir, seed
    ):
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
            tv = [
                d[key].item() if hasattr(d[key], "item") else d[key]
                for d in train_history
            ]
            vv = [
                d[key].item() if hasattr(d[key], "item") else d[key]
                for d in val_history
            ]
            plt.plot(np.linspace(0, num_epochs - 1, len(tv)), tv, label="train")
            if vv:
                val_x = (
                    val_epochs
                    if val_epochs
                    else np.linspace(0, num_epochs - 1, len(vv))
                )
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
        raise ValueError(
            f"Unsupported amp_dtype={amp_dtype_name!r}. Use auto, bf16, or fp16."
        )

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
