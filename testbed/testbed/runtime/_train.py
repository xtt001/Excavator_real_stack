"""Internal train helper called by Runner.train()."""

from __future__ import annotations

import copy
import datetime
import json
import pickle
from pathlib import Path
from typing import Any


def train_policy(config: dict[str, Any]) -> None:
    task_cfg = config.get("task", {})
    policy_cfg = config.get("policy", {})
    train_cfg = config.get("train", {})

    policy_class = str(policy_cfg.get("class", policy_cfg.get("name", "ACT"))).upper()
    task_name = task_cfg.get(
        "task_name", task_cfg.get("name", config.get("task_name", ""))
    )
    dataset_dir = Path(task_cfg.get("dataset_dir", config.get("dataset_dir", "data")))
    num_episodes = task_cfg.get("num_episodes", config.get("num_episodes", 50))
    episode_len_raw = task_cfg.get("episode_len", config.get("episode_len", 400))
    episode_len = None if episode_len_raw is None else int(episode_len_raw)
    camera_names = task_cfg.get("camera_names", config.get("camera_names", []))
    low_dim_keys = list(policy_cfg.get("low_dim_keys", ["qpos"]))
    ckpt_dir = Path(
        train_cfg.get("ckpt_dir", config.get("ckpt_dir", f"ckpts/{task_name}"))
    )
    equipment_model = task_cfg.get(
        "equipment_model", config.get("equipment_model", "real_excavator")
    )
    device = str(train_cfg.get("device", policy_cfg.get("device", "cuda")))
    split_seed_raw = train_cfg.get("split_seed")
    split_seed = int(
        train_cfg.get("seed", 0) if split_seed_raw is None else split_seed_raw
    )
    train_split_ratio = float(train_cfg.get("train_split_ratio", 0.8))
    reuse_split = bool(train_cfg.get("reuse_split", True))
    split_path = Path(train_cfg.get("split_path", ckpt_dir / "train_val_split.yaml"))
    episode_ids = _resolve_training_episode_ids(task_cfg=task_cfg, train_cfg=train_cfg)
    image_transform = str(
        train_cfg.get("image_transform", task_cfg.get("image_transform", "none"))
    )
    deadzone_intent = copy.deepcopy(
        train_cfg.get("deadzone_intent", policy_cfg.get("deadzone_intent", {})) or {}
    )
    sample_valid_mask_path = str(train_cfg.get("sample_valid_mask_path", "")).strip()
    norm_stats_train_only = bool(train_cfg.get("norm_stats_train_only", False))
    condition_shuffle = copy.deepcopy(train_cfg.get("condition_shuffle", {}) or {})
    condition_shuffle_enabled = bool(condition_shuffle.get("enabled", False))
    if condition_shuffle_enabled:
        if (
            condition_shuffle.get("key") != "cycle_condition_v1"
            or condition_shuffle.get("scope") != "train_only"
            or "cycle_condition_v1" not in low_dim_keys
        ):
            raise ValueError(
                "condition shuffle requires key=cycle_condition_v1, "
                "scope=train_only, and cycle_condition_v1 in low_dim_keys"
            )
        condition_shuffle_seed_train = int(condition_shuffle.get("seed", 0))
    else:
        condition_shuffle_seed_train = None
    condition_phase_randomization = copy.deepcopy(
        train_cfg.get("condition_phase_randomization", {}) or {}
    )
    condition_phase_randomization_enabled = bool(
        condition_phase_randomization.get("enabled", False)
    )
    if condition_phase_randomization_enabled:
        if (
            condition_phase_randomization.get("key") != "cycle_condition_v1.next_sector"
            or condition_phase_randomization.get("scope") != "train_only"
            or "cycle_condition_v1" not in low_dim_keys
        ):
            raise ValueError(
                "condition phase randomization requires "
                "key=cycle_condition_v1.next_sector, scope=train_only, "
                "and cycle_condition_v1 in low_dim_keys"
            )
        if condition_shuffle_enabled:
            raise ValueError(
                "condition shuffle and phase randomization are mutually exclusive"
            )
    condition_counterfactual_consistency = copy.deepcopy(
        train_cfg.get("condition_counterfactual_consistency", {}) or {}
    )
    condition_counterfactual_consistency_enabled = bool(
        condition_counterfactual_consistency.get("enabled", False)
    )
    if condition_counterfactual_consistency_enabled:
        if (
            condition_counterfactual_consistency.get("key")
            != "cycle_condition_v1.next_sector"
            or condition_counterfactual_consistency.get("scope") != "train_only"
            or "cycle_condition_v1" not in low_dim_keys
        ):
            raise ValueError(
                "condition counterfactual consistency requires "
                "key=cycle_condition_v1.next_sector, scope=train_only, "
                "and cycle_condition_v1 in low_dim_keys"
            )
        if condition_shuffle_enabled or condition_phase_randomization_enabled:
            raise ValueError(
                "condition counterfactual consistency is mutually exclusive "
                "with condition shuffle and phase randomization"
            )
    phase_routed_condition = copy.deepcopy(
        (policy_cfg.get("act_params", {}) or {}).get(
            "phase_routed_condition",
            {},
        )
        or {}
    )
    phase_routed_condition_enabled = bool(
        phase_routed_condition.get("enabled", False)
    )
    if phase_routed_condition_enabled:
        if low_dim_keys != ["qpos", "qvel", "cycle_condition_v1"]:
            raise ValueError(
                "phase-routed condition requires low_dim_keys in frozen "
                "qpos,qvel,cycle_condition_v1 order"
            )
        if (
            condition_phase_randomization_enabled
            or condition_counterfactual_consistency_enabled
        ):
            raise ValueError(
                "phase-routed condition is mutually exclusive with B1.1/B1.2 "
                "phase interventions"
            )
    checkpoint_semantics = copy.deepcopy(config.get("checkpoint_semantics", {}) or {})
    experiment_contract = copy.deepcopy(config.get("experiment_contract", {}) or {})

    if policy_class != "ACT":
        raise NotImplementedError(
            f"Trainer for policy class {policy_class!r} not yet implemented."
        )

    from testbed.data.dataset import load_data
    from testbed.policies.act.trainer import ACTTrainer
    from testbed.runtime.run_metadata import (
        build_train_run_metadata,
        write_json,
        write_resolved_config,
    )

    # build policy_config dict for ACTAdapter / detr
    act_params = policy_cfg.get("act_params", {})
    camera_role_encoding = copy.deepcopy(
        act_params.get("camera_role_encoding", {}) or {}
    )
    policy_config = {
        "lr": float(train_cfg.get("lr", 1e-5)),
        "num_queries": int(act_params.get("chunk_size", 100)),
        "kl_weight": float(act_params.get("kl_weight", 10)),
        "hidden_dim": int(act_params.get("hidden_dim", 512)),
        "dim_feedforward": int(act_params.get("dim_feedforward", 3200)),
        "vision_feature_scale": float(act_params.get("vision_feature_scale", 1.0)),
        "proprio_feature_scale": float(act_params.get("proprio_feature_scale", 1.0)),
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
        "equipment_model": equipment_model,
        "low_dim_keys": low_dim_keys,
        "state_dim": _resolve_low_dim_state_dim(low_dim_keys, equipment_model),
        "device": device,
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
        "camera_role_encoding": camera_role_encoding,
        "condition_counterfactual_consistency": copy.deepcopy(
            condition_counterfactual_consistency
        ),
        "phase_routed_condition": copy.deepcopy(phase_routed_condition),
    }

    full_config = {
        "num_epochs": int(train_cfg.get("num_epochs", 2000)),
        "ckpt_dir": str(ckpt_dir),
        "seed": int(train_cfg.get("seed", 0)),
        "task_name": task_name,
        "device": device,
        "resume_ckpt": train_cfg.get("resume_ckpt"),
        "start_epoch": train_cfg.get("start_epoch"),
        "val_every": int(train_cfg.get("val_every", 1)),
        "save_latest_every": int(train_cfg.get("save_latest_every", 1)),
        "checkpoint_every": int(train_cfg.get("checkpoint_every", 100)),
        "plot_every": int(
            train_cfg.get("plot_every", train_cfg.get("checkpoint_every", 100))
        ),
        "amp": bool(train_cfg.get("amp", False)),
        "amp_dtype": str(train_cfg.get("amp_dtype", "auto")),
        "split_seed": split_seed,
        "train_split_ratio": train_split_ratio,
        "reuse_split": reuse_split,
        "split_path": str(split_path),
        "image_transform": image_transform,
        "sample_valid_mask_path": sample_valid_mask_path,
        "norm_stats_train_only": norm_stats_train_only,
        "condition_shuffle": condition_shuffle,
        "condition_phase_randomization": condition_phase_randomization,
        "condition_counterfactual_consistency": (
            condition_counterfactual_consistency
        ),
        "phase_routed_condition": phase_routed_condition,
        "checkpoint_semantics": checkpoint_semantics,
        "experiment_contract": experiment_contract,
    }

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 4))
    pf_raw = train_cfg.get("prefetch_factor", 2)
    prefetch_factor = int(pf_raw) if pf_raw is not None and num_workers > 0 else None
    action_chunk_size = int(act_params.get("chunk_size", 100))
    train_loader, val_loader, norm_stats, _, split_info = load_data(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        episode_len=episode_len,
        batch_size_train=batch_size,
        batch_size_val=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=bool(train_cfg.get("persistent_workers", True))
        and num_workers > 0,
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        split_seed=split_seed,
        train_split_ratio=train_split_ratio,
        split_path=split_path,
        reuse_split=reuse_split,
        low_dim_keys=low_dim_keys,
        episode_ids=episode_ids,
        action_chunk_size=action_chunk_size,
        image_transform=image_transform,
        deadzone_intent=deadzone_intent,
        sample_valid_mask_path=sample_valid_mask_path or None,
        norm_stats_train_only=norm_stats_train_only,
        condition_shuffle_seed_train=condition_shuffle_seed_train,
        condition_phase_randomization_train=(
            condition_phase_randomization
            if condition_phase_randomization_enabled
            else None
        ),
        condition_counterfactual_consistency_train=(
            condition_counterfactual_consistency
            if condition_counterfactual_consistency_enabled
            else None
        ),
        phase_routed_condition=(
            phase_routed_condition
            if phase_routed_condition_enabled
            else None
        ),
    )
    experiment_contract["condition_shuffle_provenance"] = copy.deepcopy(
        split_info["condition_shuffle_train"]
    )
    experiment_contract["condition_phase_randomization_provenance"] = copy.deepcopy(
        split_info["condition_phase_randomization_train"]
    )
    experiment_contract["condition_counterfactual_consistency_provenance"] = (
        copy.deepcopy(split_info["condition_counterfactual_consistency_train"])
    )
    experiment_contract["phase_routed_condition_provenance"] = {
        "train": copy.deepcopy(split_info["phase_routed_condition_train"]),
        "validation": copy.deepcopy(
            split_info["phase_routed_condition_validation"]
        ),
    }
    full_config["experiment_contract"] = copy.deepcopy(experiment_contract)

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
        full_config=full_config,
    )
    resolved_config_path = write_resolved_config(
        ckpt_dir / "resolved_config.yaml", resolved_config
    )
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
    run_metadata["checkpoint_semantics"] = checkpoint_semantics
    run_metadata["experiment_contract"] = experiment_contract
    run_metadata_path = write_json(ckpt_dir / "run_metadata.json", run_metadata)
    print(f"Saved resolved config to {resolved_config_path}")
    print(f"Saved run metadata to {run_metadata_path}")

    trainer = ACTTrainer(policy_config=policy_config, config=full_config)
    try:
        best_epoch, best_val_loss, _ = trainer.fit(
            train_loader, val_loader, full_config
        )
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
    full_config: dict[str, Any],
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    task_cfg = resolved.setdefault("task", {})
    train_cfg = resolved.setdefault("train", {})
    resolved["experiment_contract"] = copy.deepcopy(full_config["experiment_contract"])

    task_cfg["dataset_dir"] = str(dataset_dir)
    if "train_ready_manifest_path" in config.get("task", {}):
        task_cfg["train_ready_manifest_path"] = str(
            config["task"]["train_ready_manifest_path"]
        )
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
    train_cfg["sample_valid_mask_path"] = str(full_config["sample_valid_mask_path"])
    train_cfg["norm_stats_train_only"] = bool(full_config["norm_stats_train_only"])
    train_cfg["condition_shuffle"] = copy.deepcopy(full_config["condition_shuffle"])
    train_cfg["condition_phase_randomization"] = copy.deepcopy(
        full_config["condition_phase_randomization"]
    )
    train_cfg["condition_counterfactual_consistency"] = copy.deepcopy(
        full_config["condition_counterfactual_consistency"]
    )
    train_cfg["phase_routed_condition"] = copy.deepcopy(
        full_config["phase_routed_condition"]
    )
    return resolved


def _resolve_low_dim_state_dim(low_dim_keys: list[str], equipment_model: str) -> int:
    dims = {
        "qpos": _resolve_single_low_dim_dim("qpos", equipment_model),
        "qvel": _resolve_single_low_dim_dim("qvel", equipment_model),
        "cycle_condition_v1": _resolve_single_low_dim_dim(
            "cycle_condition_v1",
            equipment_model,
        ),
    }
    return int(sum(dims[key] for key in low_dim_keys))


def _resolve_single_low_dim_dim(key: str, equipment_model: str) -> int:
    if key in ("qpos", "qvel"):
        return 4
    if key == "cycle_condition_v1":
        return 6
    raise ValueError(f"Unsupported low-dim key {key!r}.")


def _resolve_training_episode_ids(
    *,
    task_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
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
        raise FileNotFoundError(
            f"train_ready_manifest_path does not exist: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text())
    ids: list[int] = []
    for value in payload.get("train_ready_episode_ids", []):
        text = str(value)
        if text.startswith("episode_"):
            text = text.split("_", 1)[1]
        ids.append(int(text))
    if not ids:
        raise ValueError(
            f"train_ready_manifest_path contains no train_ready_episode_ids: {manifest_path}"
        )
    return sorted(set(ids))
