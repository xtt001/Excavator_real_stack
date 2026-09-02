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

from testbed.policies.act.adapter import ACTAdapter
from testbed.policies.base import Trainer, compute_dict_mean, detach_dict, set_seed

_CONDITION_EXPANDABLE_WEIGHT_SUFFIXES = (
    "input_proj_robot_state.weight",
    "encoder_joint_proj.weight",
)
_OPTIONAL_WARM_START_PREFIXES = (
    "goal_effect_head.",
    "goal_context_proj.",
    "action_context_residual.",
    "condition_action_head.",
)


def _load_conditioned_warm_start(
    adapter: ACTAdapter,
    source_state: dict[str, torch.Tensor],
) -> list[str]:
    """Copy existing ACT input weights and zero newly added state columns."""

    if not isinstance(source_state, dict):
        raise TypeError("warm-start checkpoint state must be a mapping")
    target_state = adapter.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    required_missing = [
        key
        for key in missing
        if not key.startswith(_OPTIONAL_WARM_START_PREFIXES)
        and _primitive_head_source_key(key, source_state) is None
    ]
    required_unexpected = [
        key
        for key in unexpected
        if not key.startswith(_OPTIONAL_WARM_START_PREFIXES)
    ]
    if required_missing or required_unexpected:
        raise ValueError(
            "warm-start checkpoint keys do not match target model; "
            f"missing={required_missing[:10]}, "
            f"unexpected={required_unexpected[:10]}"
        )

    adapted: dict[str, torch.Tensor] = {}
    expanded: list[str] = []
    for key, target_value in target_state.items():
        primitive_source_key = _primitive_head_source_key(key, source_state)
        if key not in source_state and primitive_source_key is not None:
            source_value = source_state[primitive_source_key]
            if tuple(source_value.shape) != tuple(target_value.shape):
                raise ValueError(
                    f"warm-start primitive head shape mismatch for {key}: "
                    f"source={tuple(source_value.shape)} "
                    f"target={tuple(target_value.shape)}"
                )
            adapted[key] = source_value.to(
                dtype=target_value.dtype, device=target_value.device
            )
            continue
        if key not in source_state:
            adapted[key] = target_value
            continue
        source_value = source_state[key]
        if tuple(source_value.shape) == tuple(target_value.shape):
            adapted[key] = source_value.to(
                dtype=target_value.dtype, device=target_value.device
            )
            continue
        compatible = (
            key.endswith(_CONDITION_EXPANDABLE_WEIGHT_SUFFIXES)
            and source_value.ndim == 2
            and target_value.ndim == 2
            and int(source_value.shape[0]) == int(target_value.shape[0])
            and int(source_value.shape[1]) < int(target_value.shape[1])
        )
        if not compatible:
            raise ValueError(
                f"warm-start shape mismatch for {key}: "
                f"source={tuple(source_value.shape)} target={tuple(target_value.shape)}"
            )
        value = torch.zeros_like(target_value)
        value[:, : int(source_value.shape[1])] = source_value.to(
            dtype=value.dtype, device=value.device
        )
        adapted[key] = value
        expanded.append(key)

    if not expanded:
        raise ValueError(
            "warm_start_ckpt did not require condition-column expansion; use "
            "resume_ckpt for an exact-shape continuation"
        )
    adapter.load_state_dict(adapted, strict=True)
    return sorted(expanded)


def _load_vision_backbone_warm_start(
    adapter: ACTAdapter,
    source_state: dict[str, torch.Tensor],
) -> list[str]:
    """Copy only the ResNet body and leave every control/fusion layer fresh."""

    if not isinstance(source_state, dict):
        raise TypeError("warm-start checkpoint state must be a mapping")
    target_state = adapter.state_dict()
    prefix = "backbones.0.0.body."
    copied = []
    adapted = dict(target_state)
    for key, target_value in target_state.items():
        if not key.startswith(prefix) or key not in source_state:
            continue
        source_value = source_state[key]
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise ValueError(
                f"vision warm-start shape mismatch for {key}: "
                f"source={tuple(source_value.shape)} "
                f"target={tuple(target_value.shape)}"
            )
        adapted[key] = source_value.to(
            dtype=target_value.dtype, device=target_value.device
        )
        copied.append(key)
    if not copied:
        raise ValueError("vision-only warm-start found no ResNet body weights")
    adapter.load_state_dict(adapted, strict=True)
    return sorted(copied)


def _load_state_visual_residual_warm_start(
    adapter: ACTAdapter,
    source_state: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    """Reuse the learned visual residual while resetting its low-dimensional MLP."""

    if not isinstance(source_state, dict):
        raise TypeError("warm-start checkpoint state must be a mapping")
    target_state = adapter.state_dict()
    unexpected = sorted(set(source_state) - set(target_state))
    if unexpected:
        raise ValueError(
            "state-visual warm-start checkpoint has unexpected keys: "
            f"{unexpected[:10]}"
        )
    adapted = dict(target_state)
    copied: list[str] = []
    expanded: list[str] = []
    reset: list[str] = []
    for key, target_value in target_state.items():
        if key.startswith("low_dim_action_head.") or key == (
            "state_visual_residual_proprio_mask"
        ):
            reset.append(key)
            continue
        if key not in source_state:
            raise ValueError(f"state-visual warm-start is missing key {key}")
        source_value = source_state[key]
        if tuple(source_value.shape) == tuple(target_value.shape):
            adapted[key] = source_value.to(
                dtype=target_value.dtype, device=target_value.device
            )
            copied.append(key)
            continue
        compatible = (
            key.endswith(_CONDITION_EXPANDABLE_WEIGHT_SUFFIXES)
            and source_value.ndim == 2
            and target_value.ndim == 2
            and int(source_value.shape[0]) == int(target_value.shape[0])
            and int(source_value.shape[1]) < int(target_value.shape[1])
        )
        if not compatible:
            raise ValueError(
                f"state-visual warm-start shape mismatch for {key}: "
                f"source={tuple(source_value.shape)} "
                f"target={tuple(target_value.shape)}"
            )
        value = torch.zeros_like(target_value)
        value[:, : int(source_value.shape[1])] = source_value.to(
            dtype=value.dtype, device=value.device
        )
        adapted[key] = value
        expanded.append(key)
    if not copied or not expanded or not any(
        key.startswith("low_dim_action_head.") for key in reset
    ):
        raise ValueError(
            "state-visual warm-start requires copied residual weights, expanded "
            "proprio projections, and a reset low-dimensional action head"
        )
    adapter.load_state_dict(adapted, strict=True)
    return {
        "copied": sorted(copied),
        "expanded": sorted(expanded),
        "reset": sorted(reset),
    }


def _load_state_visual_task_state_v2_warm_start(
    adapter: ACTAdapter,
    source_state: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    """Remap the proven qpos/qvel low head into the task-state-v2 layout.

    Source order is ``qpos4, old_target2, qvel4``.  Target order is
    ``qpos4, qvel4, task_state5``.  The old always-on goal-active column is
    folded into the first-layer bias, while the old target-side column maps to
    the gated next-target token.  New current/dig/event columns start at zero.
    """

    if not isinstance(source_state, dict):
        raise TypeError("warm-start checkpoint state must be a mapping")
    target_state = adapter.state_dict()
    unexpected = sorted(set(source_state) - set(target_state))
    if unexpected:
        raise ValueError(
            "task-state-v2 warm-start checkpoint has unexpected keys: "
            f"{unexpected[:10]}"
        )
    first_weight_key = "low_dim_action_head.0.weight"
    first_bias_key = "low_dim_action_head.0.bias"
    source_first = source_state.get(first_weight_key)
    target_first = target_state.get(first_weight_key)
    if source_first is None or target_first is None:
        raise ValueError("task-state-v2 warm-start requires a trained low head")
    if source_first.ndim != 2 or target_first.ndim != 2 or (
        tuple(source_first.shape) != (int(target_first.shape[0]), 10)
        or int(target_first.shape[1]) != 13
    ):
        raise ValueError(
            "task-state-v2 warm-start requires source/target low-head input "
            "dimensions 10 and 13"
        )
    source_mask = source_state.get("state_visual_residual_proprio_mask")
    target_mask = target_state.get("state_visual_residual_proprio_mask")
    if (
        source_mask is None
        or target_mask is None
        or tuple(source_mask.shape) != (10,)
        or tuple(target_mask.shape) != (13,)
        or not torch.equal(source_mask[:4], torch.ones_like(source_mask[:4]))
        or bool(torch.any(source_mask[4:] != 0))
        or not torch.equal(target_mask[:4], torch.ones_like(target_mask[:4]))
        or bool(torch.any(target_mask[4:] != 0))
    ):
        raise ValueError(
            "task-state-v2 warm-start requires qpos-only visual residual masks"
        )

    adapted = dict(target_state)
    copied: list[str] = []
    remapped: list[str] = []
    reset: list[str] = ["state_visual_residual_proprio_mask"]
    for key, target_value in target_state.items():
        if key == "state_visual_residual_proprio_mask":
            continue
        if key == first_weight_key:
            value = torch.zeros_like(target_value)
            source_value = source_first.to(
                dtype=value.dtype, device=value.device
            )
            value[:, 0:4] = source_value[:, 0:4]
            value[:, 4:8] = source_value[:, 6:10]
            value[:, 12] = source_value[:, 4]
            adapted[key] = value
            remapped.append(key)
            continue
        if key == first_bias_key:
            source_bias = source_state.get(key)
            if source_bias is None or tuple(source_bias.shape) != tuple(
                target_value.shape
            ):
                raise ValueError("task-state-v2 warm-start low-head bias mismatch")
            adapted[key] = source_bias.to(
                dtype=target_value.dtype, device=target_value.device
            ) + source_first[:, 5].to(
                dtype=target_value.dtype, device=target_value.device
            )
            remapped.append(key)
            continue
        if key in {"input_proj_robot_state.weight", "encoder_joint_proj.weight"}:
            source_value = source_state.get(key)
            if (
                source_value is None
                or source_value.ndim != 2
                or target_value.ndim != 2
                or int(source_value.shape[0]) != int(target_value.shape[0])
                or int(source_value.shape[1]) != 10
                or int(target_value.shape[1]) != 13
            ):
                raise ValueError(f"task-state-v2 projection mismatch for {key}")
            value = torch.zeros_like(target_value)
            value[:, :4] = source_value[:, :4].to(
                dtype=value.dtype, device=value.device
            )
            adapted[key] = value
            remapped.append(key)
            continue
        source_value = source_state.get(key)
        if source_value is None:
            raise ValueError(f"task-state-v2 warm-start is missing key {key}")
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise ValueError(
                f"task-state-v2 warm-start shape mismatch for {key}: "
                f"source={tuple(source_value.shape)} "
                f"target={tuple(target_value.shape)}"
            )
        adapted[key] = source_value.to(
            dtype=target_value.dtype, device=target_value.device
        )
        copied.append(key)
    adapter.load_state_dict(adapted, strict=True)
    return {
        "copied": sorted(copied),
        "remapped": sorted(remapped),
        "reset": sorted(reset),
    }


def _primitive_head_source_key(
    target_key: str,
    source_state: dict[str, torch.Tensor],
) -> str | None:
    marker = "primitive_action_heads."
    if marker not in target_key:
        return None
    prefix, remainder = target_key.split(marker, 1)
    parts = remainder.split(".", 1)
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    source_key = f"{prefix}action_head.{parts[1]}"
    return source_key if source_key in source_state else None


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
        reset_best_on_resume = bool(cfg.get("reset_best_on_resume", False))
        warm_start = cfg.get("warm_start_ckpt")
        warm_start_mode = str(cfg.get("warm_start_mode", "conditioned"))
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
        if resume and warm_start:
            raise ValueError("resume_ckpt and warm_start_ckpt are mutually exclusive")
        if resume:
            ckpt_obj = torch.load(resume, map_location="cpu")
            sd = ckpt_obj["model_state_dict"] if "model_state_dict" in ckpt_obj else ckpt_obj
            adapter.load_state_dict(sd)
            if "optimizer_state_dict" in ckpt_obj:
                optimizer.load_state_dict(ckpt_obj["optimizer_state_dict"])
            start_epoch = self._infer_start_epoch(resume, ckpt_obj, cfg)
            if "min_val_loss" in ckpt_obj:
                min_val_loss = float(ckpt_obj["min_val_loss"])
            if reset_best_on_resume:
                min_val_loss = float("inf")
                print(
                    "Reset resume best-loss baseline because the training "
                    "objective changed"
                )
            print(f"Resumed from {resume}, starting epoch {start_epoch}")
        elif warm_start:
            ckpt_obj = torch.load(warm_start, map_location="cpu")
            source_state = (
                ckpt_obj["model_state_dict"]
                if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj
                else ckpt_obj
            )
            if warm_start_mode == "vision_backbone_only":
                copied = _load_vision_backbone_warm_start(adapter, source_state)
                print(
                    f"Warm-started only {len(copied)} ResNet body tensors from "
                    f"{warm_start}; control, fusion, transformer and action "
                    "parameters remain freshly initialized"
                )
            elif warm_start_mode == "state_visual_residual_only":
                report = _load_state_visual_residual_warm_start(
                    adapter, source_state
                )
                print(
                    "Warm-started the learned visual residual from "
                    f"{warm_start}; copied={len(report['copied'])}, "
                    f"expanded={report['expanded']}, "
                    f"reset={len(report['reset'])} low-dimensional/buffer tensors"
                )
            elif warm_start_mode == "state_visual_task_state_v2":
                report = _load_state_visual_task_state_v2_warm_start(
                    adapter, source_state
                )
                print(
                    "Warm-started the qpos/qvel visual policy with semantic "
                    f"task-state-v2 remapping from {warm_start}; "
                    f"copied={len(report['copied'])}, "
                    f"remapped={report['remapped']}, reset={report['reset']}"
                )
            elif warm_start_mode == "conditioned":
                expanded = _load_conditioned_warm_start(adapter, source_state)
                print(
                    f"Warm-started from {warm_start}; zero-initialized new condition "
                    f"columns in {', '.join(expanded)}"
                )
            else:
                raise ValueError(
                    "warm_start_mode must be conditioned, vision_backbone_only, "
                    "state_visual_residual_only, or state_visual_task_state_v2"
                )

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
        active_training_stage = None
        for epoch in tqdm(range(start_epoch, num_epochs)):
            stage = adapter.configure_training_epoch(epoch)
            stage_name = None if stage is None else str(stage["name"])
            if stage_name != active_training_stage:
                active_training_stage = stage_name
                if stage is not None:
                    print(
                        "State/visual training stage:"
                        f" epoch={epoch}, name={stage_name},"
                        f" train_low={stage['train_low']},"
                        f" train_residual={stage['train_residual']},"
                        f" residual_scale={stage['residual_scale']}"
                    )
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
                        best_ckpt = (
                            epoch,
                            min_val_loss,
                            deepcopy(adapter.state_dict()),
                            getattr(
                                adapter._model,
                                "_state_visual_residual_stage",
                                None,
                            ),
                        )

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
        best_epoch, bvl, best_sd, best_stage = best_ckpt
        self._save_ckpt(
            ckpt_dir / f"policy_epoch_{best_epoch}_seed_{seed}.ckpt",
            adapter,
            optimizer,
            best_epoch,
            bvl,
            cfg,
            sd_override=best_sd,
            stage_override=best_stage,
        )
        # also save a stable "best" checkpoint name for downstream inference
        self._save_ckpt(
            ckpt_dir / "policy_best.ckpt",
            adapter,
            optimizer,
            best_epoch,
            bvl,
            cfg,
            sd_override=best_sd,
            stage_override=best_stage,
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
        extra: dict = {}
        if isinstance(data, dict):
            image_data = data["image"]
            proprio_data = data["proprio"]
            action_data = data["action"]
            is_pad = data["is_pad"]
            for key in (
                "deadzone_move_mask",
                "deadzone_stop_mask",
                "deadzone_wrong_mask",
                "action_loss_mask",
                "state_hold_transition_mask",
                "counterfactual_proprio",
                "condition_adherence_mask",
                "goal_future_delta",
                "goal_future_valid",
                "goal_future_direction",
                "goal_effect_delta",
                "goal_effect_valid",
                "condition_action_target",
                "condition_action_valid",
                "target_release_continue_primary",
                "target_release_valid",
                "phase_counterfactual_proprio",
                "cycle_phase_return_primary",
                "cycle_phase_valid",
                "excursion_counterfactual_proprio",
                "excursion_post_primary",
                "excursion_observed_valid",
                "return_commit_counterfactual_proprio",
                "return_commit_return_primary",
                "return_commit_return_effective_primary",
                "return_commit_valid",
                "qvel_zero_proprio",
                "qvel_authority_counterfactual_proprio",
                "qvel_authority_moving_primary",
                "qvel_authority_stable_tool_mask",
                "qvel_authority_valid",
                "task_state_v2_uncommitted",
            ):
                if key in data:
                    extra[key] = data[key].to(adapter.device)
        else:
            image_data, proprio_data, action_data, is_pad = data
        image_data  = image_data.to(adapter.device)
        proprio_data = proprio_data.to(adapter.device)
        action_data = action_data.to(adapter.device)
        is_pad      = is_pad.to(adapter.device)
        with ACTTrainer._autocast_context(adapter.device, amp_enabled, amp_dtype):
            return adapter.forward_loss(proprio_data, image_data, action_data, is_pad, **extra)

    @staticmethod
    def _save_ckpt(
        path,
        adapter,
        optimizer,
        epoch,
        val_loss,
        config,
        sd_override=None,
        stage_override=None,
    ):
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
                    "training_stage": (
                        stage_override
                        if stage_override is not None
                        else getattr(
                            adapter._model,
                            "_state_visual_residual_stage",
                            None,
                        )
                    ),
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
