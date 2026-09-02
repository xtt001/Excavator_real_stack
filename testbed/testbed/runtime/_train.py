"""Internal train helper called by Runner.train()."""

from __future__ import annotations

import copy
import datetime
import json
import pickle
from pathlib import Path
from typing import Any


def train_policy(config: dict[str, Any]) -> None:
    task_cfg   = config.get("task", {})
    policy_cfg = config.get("policy", {})
    train_cfg  = config.get("train", {})

    policy_class  = str(policy_cfg.get("class", policy_cfg.get("name", "ACT"))).upper()
    task_name     = task_cfg.get("task_name", task_cfg.get("name", config.get("task_name", "")))
    dataset_dir   = Path(task_cfg.get("dataset_dir", config.get("dataset_dir", "data")))
    num_episodes  = task_cfg.get("num_episodes", config.get("num_episodes", 50))
    episode_len_raw = task_cfg.get("episode_len", config.get("episode_len", 400))
    episode_len = None if episode_len_raw is None else int(episode_len_raw)
    camera_names  = task_cfg.get("camera_names", config.get("camera_names", []))
    low_dim_keys  = list(policy_cfg.get("low_dim_keys", ["qpos"]))
    ckpt_dir      = Path(train_cfg.get("ckpt_dir", config.get("ckpt_dir", f"ckpts/{task_name}")))
    equipment_model = task_cfg.get("equipment_model", config.get("equipment_model", "real_excavator"))
    device        = str(train_cfg.get("device", policy_cfg.get("device", "cuda")))
    split_seed_raw = train_cfg.get("split_seed")
    split_seed = int(train_cfg.get("seed", 0) if split_seed_raw is None else split_seed_raw)
    train_split_ratio = float(train_cfg.get("train_split_ratio", 0.8))
    reuse_split = bool(train_cfg.get("reuse_split", True))
    split_path = Path(train_cfg.get("split_path", ckpt_dir / "train_val_split.yaml"))
    episode_ids = _resolve_training_episode_ids(
        task_cfg=task_cfg, train_cfg=train_cfg, low_dim_keys=low_dim_keys
    )
    split_manifest_path = _resolve_split_manifest_path(
        task_cfg=task_cfg, train_cfg=train_cfg
    )
    image_transform = str(train_cfg.get("image_transform", task_cfg.get("image_transform", "none")))
    deadzone_intent = copy.deepcopy(
        train_cfg.get("deadzone_intent", policy_cfg.get("deadzone_intent", {})) or {}
    )
    state_hold_transition = copy.deepcopy(
        train_cfg.get(
            "state_hold_transition",
            policy_cfg.get("state_hold_transition", {}),
        )
        or {}
    )
    condition_adherence_loss = copy.deepcopy(
        train_cfg.get(
            "condition_adherence_loss",
            policy_cfg.get("condition_adherence_loss", {}),
        )
        or {}
    )
    goal_effect = copy.deepcopy(
        train_cfg.get("goal_effect", policy_cfg.get("goal_effect", {})) or {}
    )
    condition_action_loss = copy.deepcopy(
        train_cfg.get(
            "condition_action_loss",
            policy_cfg.get("condition_action_loss", {}),
        )
        or {}
    )
    target_release_loss = copy.deepcopy(
        train_cfg.get(
            "target_release_loss",
            policy_cfg.get("target_release_loss", {}),
        )
        or {}
    )
    cycle_phase_loss = copy.deepcopy(
        train_cfg.get(
            "cycle_phase_loss",
            policy_cfg.get("cycle_phase_loss", {}),
        )
        or {}
    )
    excursion_observed_loss = copy.deepcopy(
        train_cfg.get(
            "excursion_observed_loss",
            policy_cfg.get("excursion_observed_loss", {}),
        )
        or {}
    )
    return_commit_loss = copy.deepcopy(
        train_cfg.get(
            "return_commit_loss",
            policy_cfg.get("return_commit_loss", {}),
        )
        or {}
    )
    action_primitive_islands = copy.deepcopy(
        train_cfg.get(
            "action_primitive_islands",
            policy_cfg.get("action_primitive_islands", {}),
        )
        or {}
    )
    primitive_action_heads = copy.deepcopy(
        train_cfg.get(
            "primitive_action_heads",
            policy_cfg.get("primitive_action_heads", {}),
        )
        or {}
    )
    work_return_context = copy.deepcopy(
        train_cfg.get(
            "work_return_context",
            policy_cfg.get("work_return_context", {}),
        )
        or {}
    )
    task_state_v2 = copy.deepcopy(
        train_cfg.get(
            "task_state_v2",
            policy_cfg.get("task_state_v2", {}),
        )
        or {}
    )
    task_state_v2_adherence_loss = copy.deepcopy(
        train_cfg.get(
            "task_state_v2_adherence_loss",
            policy_cfg.get("task_state_v2_adherence_loss", {}),
        )
        or {}
    )
    qvel_zero_state_hold_loss = copy.deepcopy(
        train_cfg.get(
            "qvel_zero_state_hold_loss",
            policy_cfg.get("qvel_zero_state_hold_loss", {}),
        )
        or {}
    )
    qvel_authority_loss = copy.deepcopy(
        train_cfg.get(
            "qvel_authority_loss",
            policy_cfg.get("qvel_authority_loss", {}),
        )
        or {}
    )
    factual_semantic_sampling = copy.deepcopy(
        train_cfg.get(
            "factual_semantic_sampling",
            policy_cfg.get("factual_semantic_sampling", {}),
        )
        or {}
    )
    state_visual_residual = copy.deepcopy(
        train_cfg.get(
            "state_visual_residual",
            policy_cfg.get("state_visual_residual", {}),
        )
        or {}
    )

    if policy_class != "ACT":
        raise NotImplementedError(f"Trainer for policy class {policy_class!r} not yet implemented.")

    from testbed.data.dataset import load_data
    from testbed.policies.act.trainer import ACTTrainer
    from testbed.runtime.run_metadata import (
        build_train_run_metadata,
        write_json,
        write_resolved_config,
    )

    # build policy_config dict for ACTAdapter / detr
    act_params = policy_cfg.get("act_params", {})
    policy_config = {
        "lr":            float(train_cfg.get("lr", 1e-5)),
        "num_queries":   int(act_params.get("chunk_size", 100)),
        "kl_weight":     float(act_params.get("kl_weight", 10)),
        "hidden_dim":    int(act_params.get("hidden_dim", 512)),
        "dim_feedforward": int(act_params.get("dim_feedforward", 3200)),
        "vision_feature_scale": float(act_params.get("vision_feature_scale", 1.0)),
        "proprio_feature_scale": float(act_params.get("proprio_feature_scale", 1.0)),
        "lr_backbone":   1e-5,
        "backbone":      "resnet18",
        "enc_layers":    4,
        "dec_layers":    7,
        "nheads":        8,
        "camera_names":  camera_names,
        "equipment_model": equipment_model,
        "low_dim_keys":  low_dim_keys,
        "state_dim":     _resolve_low_dim_state_dim(low_dim_keys, equipment_model),
        "device":        device,
        "camera_role_encoding": copy.deepcopy(
            act_params.get("camera_role_encoding", {}) or {}
        ),
        "condition_adherence_loss": copy.deepcopy(condition_adherence_loss),
        "goal_effect": copy.deepcopy(goal_effect),
        "condition_action_loss": copy.deepcopy(condition_action_loss),
        "target_release_loss": copy.deepcopy(target_release_loss),
        "cycle_phase_loss": copy.deepcopy(cycle_phase_loss),
        "excursion_observed_loss": copy.deepcopy(excursion_observed_loss),
        "return_commit_loss": copy.deepcopy(return_commit_loss),
        "action_primitive_islands": copy.deepcopy(action_primitive_islands),
        "primitive_action_heads": copy.deepcopy(primitive_action_heads),
        "work_return_context": copy.deepcopy(work_return_context),
        "task_state_v2": copy.deepcopy(task_state_v2),
        "task_state_v2_adherence_loss": copy.deepcopy(
            task_state_v2_adherence_loss
        ),
        "qvel_zero_state_hold_loss": copy.deepcopy(qvel_zero_state_hold_loss),
        "qvel_authority_loss": copy.deepcopy(qvel_authority_loss),
        "state_visual_residual": copy.deepcopy(state_visual_residual),
        "deadzone_loss": copy.deepcopy(
            train_cfg.get("deadzone_loss", policy_cfg.get("deadzone_loss", {})) or {}
        ),
        "intent_loss": copy.deepcopy(
            train_cfg.get("intent_loss", policy_cfg.get("intent_loss", {})) or {}
        ),
        "window_deadzone_loss": copy.deepcopy(
            train_cfg.get(
                "window_deadzone_loss",
                policy_cfg.get("window_deadzone_loss", {}),
            )
            or {}
        ),
        "temporal_release_loss": copy.deepcopy(
            train_cfg.get(
                "temporal_release_loss",
                policy_cfg.get("temporal_release_loss", {}),
            )
            or {}
        ),
    }

    full_config = {
        "num_epochs":     int(train_cfg.get("num_epochs", 2000)),
        "ckpt_dir":       str(ckpt_dir),
        "seed":           int(train_cfg.get("seed", 0)),
        "task_name":      task_name,
        "device":         device,
        "resume_ckpt":    train_cfg.get("resume_ckpt"),
        "reset_best_on_resume": bool(
            train_cfg.get("reset_best_on_resume", False)
        ),
        "warm_start_ckpt": train_cfg.get("warm_start_ckpt"),
        "warm_start_mode": str(
            train_cfg.get("warm_start_mode", "conditioned")
        ),
        "start_epoch":    train_cfg.get("start_epoch"),
        "val_every":      int(train_cfg.get("val_every", 1)),
        "save_latest_every": int(train_cfg.get("save_latest_every", 1)),
        "checkpoint_every": int(train_cfg.get("checkpoint_every", 100)),
        "plot_every":     int(train_cfg.get("plot_every", train_cfg.get("checkpoint_every", 100))),
        "amp":            bool(train_cfg.get("amp", False)),
        "amp_dtype":      str(train_cfg.get("amp_dtype", "auto")),
        "split_seed":     split_seed,
        "train_split_ratio": train_split_ratio,
        "reuse_split":    reuse_split,
        "split_path":     str(split_path),
        "image_transform": image_transform,
    }

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    batch_size   = int(train_cfg.get("batch_size", 8))
    num_workers  = int(train_cfg.get("num_workers", 4))
    pf_raw       = train_cfg.get("prefetch_factor", 2)
    prefetch_factor = int(pf_raw) if pf_raw is not None and num_workers > 0 else None
    action_chunk_size = int(act_params.get("chunk_size", 100))
    train_loader, val_loader, norm_stats, _, split_info = load_data(
        dataset_dir  = dataset_dir,
        num_episodes = num_episodes,
        camera_names = camera_names,
        episode_len  = episode_len,
        batch_size_train   = batch_size,
        batch_size_val     = batch_size,
        num_workers        = num_workers,
        prefetch_factor    = prefetch_factor,
        persistent_workers = bool(train_cfg.get("persistent_workers", True)) and num_workers > 0,
        pin_memory         = bool(train_cfg.get("pin_memory", True)),
        split_seed         = split_seed,
        train_split_ratio  = train_split_ratio,
        split_path         = split_path,
        split_manifest_path = split_manifest_path,
        reuse_split        = reuse_split,
        low_dim_keys       = low_dim_keys,
        episode_ids        = episode_ids,
        action_chunk_size  = action_chunk_size,
        image_transform    = image_transform,
        deadzone_intent    = deadzone_intent,
        state_hold_transition = state_hold_transition,
        condition_adherence_loss_train = condition_adherence_loss,
        target_release_loss_train = target_release_loss,
        cycle_phase_loss_train = cycle_phase_loss,
        excursion_observed_loss_train = excursion_observed_loss,
        return_commit_loss_train = return_commit_loss,
        action_primitive_islands = action_primitive_islands,
        work_return_context = work_return_context,
        task_state_v2 = task_state_v2,
        qvel_zero_state_hold_loss_train = qvel_zero_state_hold_loss,
        qvel_authority_loss_train = qvel_authority_loss,
        factual_semantic_sampling_train = factual_semantic_sampling,
        goal_effect = goal_effect,
    )

    # save normalisation stats so trainer can load them
    stats_path = ckpt_dir / "dataset_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump(norm_stats, f)
    print(f"Saved normalisation stats to {stats_path}")

    resolved_config = _build_resolved_train_config(
        config=config,
        dataset_dir=dataset_dir,
        ckpt_dir=ckpt_dir,
        split_path=split_path,
        split_manifest_path=split_manifest_path,
        full_config=full_config,
    )
    resolved_config_path = write_resolved_config(ckpt_dir / "resolved_config.yaml", resolved_config)
    run_metadata = build_train_run_metadata(
        dataset_dir=dataset_dir,
        ckpt_dir=ckpt_dir,
        resolved_config_path=resolved_config_path,
        dataset_stats_path=stats_path,
        split_info=split_info,
        policy_class=policy_class,
        task_name=task_name,
        device=device,
    )
    run_metadata["status"] = "started"
    run_metadata_path = write_json(ckpt_dir / "run_metadata.json", run_metadata)
    print(f"Saved resolved config to {resolved_config_path}")
    print(f"Saved run metadata to {run_metadata_path}")

    trainer = ACTTrainer(policy_config=policy_config, config=full_config)
    try:
        best_epoch, best_val_loss, _ = trainer.fit(train_loader, val_loader, full_config)
    except Exception as exc:
        run_metadata["status"] = "failed"
        run_metadata["completed_at"] = datetime.datetime.utcnow().isoformat()
        run_metadata["error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_metadata_path, run_metadata)
        raise

    run_metadata["status"] = "completed"
    run_metadata["completed_at"] = datetime.datetime.utcnow().isoformat()
    run_metadata["training_result"] = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
    }
    write_json(run_metadata_path, run_metadata)


def _build_resolved_train_config(
    *,
    config: dict[str, Any],
    dataset_dir: Path,
    ckpt_dir: Path,
    split_path: Path,
    split_manifest_path: Path | None,
    full_config: dict[str, Any],
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    task_cfg = resolved.setdefault("task", {})
    train_cfg = resolved.setdefault("train", {})

    task_cfg["dataset_dir"] = str(dataset_dir)
    if "train_ready_manifest_path" in config.get("task", {}):
        task_cfg["train_ready_manifest_path"] = str(
            config["task"]["train_ready_manifest_path"]
        )
    if split_manifest_path is not None:
        task_cfg["split_manifest_path"] = str(split_manifest_path)
    train_cfg["ckpt_dir"] = str(ckpt_dir)
    train_cfg["split_path"] = str(split_path)
    train_cfg["image_transform"] = str(full_config["image_transform"])
    train_cfg["split_seed"] = int(full_config["split_seed"])
    train_cfg["train_split_ratio"] = float(full_config["train_split_ratio"])
    train_cfg["reuse_split"] = bool(full_config["reuse_split"])
    train_cfg["val_every"] = int(full_config["val_every"])
    train_cfg["save_latest_every"] = int(full_config["save_latest_every"])
    train_cfg["checkpoint_every"] = int(full_config["checkpoint_every"])
    train_cfg["plot_every"] = int(full_config["plot_every"])
    train_cfg["amp"] = bool(full_config["amp"])
    train_cfg["amp_dtype"] = str(full_config["amp_dtype"])
    train_cfg["reset_best_on_resume"] = bool(
        full_config["reset_best_on_resume"]
    )
    return resolved


def _resolve_low_dim_state_dim(low_dim_keys: list[str], equipment_model: str) -> int:
    dims = {
        "qpos": _resolve_single_low_dim_dim("qpos", equipment_model),
        "qvel": _resolve_single_low_dim_dim("qvel", equipment_model),
        "real_transition_condition_v1": 2,
        "real_transition_excursion_observed_v1": 1,
        "real_transition_cycle_phase_v1": 1,
        "real_transition_return_commit_v1": 1,
        "real_transition_action_primitive_v1": 4,
        "real_transition_work_context_v1": 6,
        "real_transition_task_state_v2": 5,
    }
    return int(sum(dims[key] for key in low_dim_keys))


def _resolve_single_low_dim_dim(key: str, equipment_model: str) -> int:
    if key in ("qpos", "qvel"):
        return 4
    if key == "real_transition_condition_v1":
        return 2
    if key == "real_transition_excursion_observed_v1":
        return 1
    if key == "real_transition_cycle_phase_v1":
        return 1
    if key == "real_transition_return_commit_v1":
        return 1
    if key == "real_transition_action_primitive_v1":
        return 4
    if key == "real_transition_work_context_v1":
        return 6
    if key == "real_transition_task_state_v2":
        return 5
    raise ValueError(f"Unsupported low-dim key {key!r}.")


def _resolve_training_episode_ids(
    *,
    task_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    low_dim_keys: list[str] | None = None,
) -> list[int] | None:
    raw_ids = task_cfg.get("episode_ids")
    if raw_ids is not None:
        return [int(ep_id) for ep_id in raw_ids]
    manifest_raw = train_cfg.get("train_ready_manifest_path") or task_cfg.get(
        "train_ready_manifest_path"
    )
    if not manifest_raw:
        return None
    manifest_path = Path(str(manifest_raw))
    if not manifest_path.exists():
        raise FileNotFoundError(f"train_ready_manifest_path does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if low_dim_keys and "real_transition_condition_v1" in low_dim_keys:
        if payload.get("schema") != "real_transition_train_ready_manifest_v1":
            raise ValueError(
                "goal-conditioned training requires "
                "real_transition_train_ready_manifest_v1"
            )
        if payload.get("condition_schema") != "real_transition_condition_v1":
            raise ValueError(
                "train-ready manifest condition_schema must be "
                "'real_transition_condition_v1'"
            )
    if low_dim_keys and "real_transition_cycle_phase_v1" in low_dim_keys:
        if payload.get("cycle_phase_schema") != "real_transition_cycle_phase_v1":
            raise ValueError(
                "train-ready manifest cycle_phase_schema must be "
                "'real_transition_cycle_phase_v1'"
            )
    if low_dim_keys and "real_transition_return_commit_v1" in low_dim_keys:
        if payload.get("return_commit_schema") != (
            "real_transition_return_commit_v1"
        ):
            raise ValueError(
                "train-ready manifest return_commit_schema must be "
                "'real_transition_return_commit_v1'"
            )
    ids: list[int] = []
    for value in payload.get("train_ready_episode_ids", []):
        text = str(value)
        if text.startswith("episode_"):
            text = text.split("_", 1)[1]
        ids.append(int(text))
    if not ids:
        raise ValueError(f"train_ready_manifest_path contains no train_ready_episode_ids: {manifest_path}")
    return sorted(set(ids))


def _resolve_split_manifest_path(
    *,
    task_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> Path | None:
    raw = train_cfg.get("split_manifest_path") or task_cfg.get("split_manifest_path")
    if raw:
        path = Path(str(raw))
        if not path.exists():
            raise FileNotFoundError(f"split_manifest_path does not exist: {path}")
        return path

    ready_raw = train_cfg.get("train_ready_manifest_path") or task_cfg.get(
        "train_ready_manifest_path"
    )
    if not ready_raw:
        return None
    ready_path = Path(str(ready_raw))
    if not ready_path.exists():
        return None
    payload = json.loads(ready_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "real_transition_train_ready_manifest_v1":
        return None
    sibling = ready_path.parent / "split_manifest.json"
    if not sibling.exists():
        raise FileNotFoundError(
            "real_transition_train_ready_manifest_v1 requires sibling "
            f"split_manifest.json: {sibling}"
        )
    return sibling
