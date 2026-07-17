"""Read-only preflight checks for frozen ACT training experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py

from testbed.data.training_composite import (
    load_mapping,
    sha256_file,
    validate_training_composite,
)

IDENTITY_ACTION_SCALE = [1.0, 1.0, 1.0, 1.0]
EYE2_CAMERA_NAMES = ["video4", "video5"]
FOUR_CAMERA_NAMES = ["video4", "video5", "video6", "video7"]
SUPPORTED_CAMERA_SETS = (EYE2_CAMERA_NAMES, FOUR_CAMERA_NAMES)


def preflight_act_training_config(
    config_path: str | Path,
    *,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Validate one experiment without constructing a model or training."""

    path = Path(config_path).expanduser().resolve()
    config = load_mapping(path)
    task = _mapping(config.get("task"), "task")
    policy = _mapping(config.get("policy"), "policy")
    train = _mapping(config.get("train"), "train")
    contract = _mapping(config.get("experiment_contract"), "experiment_contract")

    if str(policy.get("class", "")).upper() != "ACT":
        raise ValueError("training preflight only accepts policy.class=ACT")
    if list(policy.get("low_dim_keys", [])) != ["qpos"]:
        raise ValueError("G48 contract requires low_dim_keys=[qpos]")
    cameras = list(task.get("camera_names", []))
    expected_cameras = list(
        contract.get("expected_camera_names", EYE2_CAMERA_NAMES)
    )
    if expected_cameras not in SUPPORTED_CAMERA_SETS:
        raise ValueError(
            "experiment contract expected_camera_names must be eye2 or fourcam"
        )
    if cameras != expected_cameras:
        raise ValueError(
            "task.camera_names does not match experiment contract "
            "expected_camera_names"
        )
    act_params = _mapping(policy.get("act_params", {}), "policy.act_params")
    role_config = _mapping(
        act_params.get("camera_role_encoding", {}),
        "policy.act_params.camera_role_encoding",
    )
    role_enabled = bool(role_config.get("enabled", False))
    expected_role_enabled = bool(
        contract.get("camera_role_encoding_enabled", False)
    )
    if role_enabled != expected_role_enabled:
        raise ValueError(
            "camera_role_encoding.enabled does not match experiment contract"
        )
    if role_enabled:
        roles = _mapping(
            role_config.get("roles", {}),
            "policy.act_params.camera_role_encoding.roles",
        )
        if list(roles) != cameras:
            raise ValueError(
                "camera_role_encoding.roles must cover cameras in declared order"
            )
        if any(str(role) not in {"eye", "stick"} for role in roles.values()):
            raise ValueError("camera roles must be eye or stick")
    action_scale = [float(value) for value in contract.get("policy_action_scale", [])]
    if action_scale != IDENTITY_ACTION_SCALE:
        raise ValueError("experiment contract must pin identity policy_action_scale")
    expected_epochs = int(contract.get("expected_num_epochs", 0))
    if expected_epochs <= 0 or int(train.get("num_epochs", 0)) != expected_epochs:
        raise ValueError("train.num_epochs does not match experiment contract")

    dataset_dir = Path(str(task.get("dataset_dir", ""))).expanduser().resolve()
    manifest_path = Path(
        str(task.get("train_ready_manifest_path", ""))
    ).expanduser().resolve()
    split_path = Path(str(train.get("split_path", ""))).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"train_ready_manifest_path missing: {manifest_path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"split_path missing: {split_path}")

    composite = validate_training_composite(
        dataset_dir, verify_hashes=verify_hashes
    )
    manifest = load_mapping(manifest_path)
    split = load_mapping(split_path)
    if manifest_path != Path(composite["manifest_path"]):
        raise ValueError("config manifest is not the dataset view manifest")
    if split_path != Path(composite["split_path"]):
        raise ValueError("config split is not the dataset view split")

    train_ids = [int(value) for value in split.get("train_ids", [])]
    val_ids = [int(value) for value in split.get("val_ids", [])]
    ready_ids = [int(value) for value in manifest.get("train_ready_episode_ids", [])]
    if set(train_ids) & set(val_ids):
        raise ValueError("train and validation episode ids overlap")
    if set(train_ids) | set(val_ids) != set(ready_ids):
        raise ValueError("manifest must contain exactly train+validation ids")
    if manifest.get("test_ids") not in ([], None):
        raise ValueError("training view must not contain test ids")

    forbidden = {
        int(value) for value in contract.get("forbidden_source_episode_ids", [])
    }
    legacy_root = Path(
        str(contract.get("forbidden_legacy_heldout_dataset", "/nonexistent"))
    ).expanduser().resolve()
    records = manifest.get("episodes", [])
    if not isinstance(records, list) or len(records) != len(ready_ids):
        raise ValueError("training view manifest episode records are incomplete")

    for record in records:
        source_id = int(record["source_episode_id"])
        source_path = Path(str(record["source_path"])).resolve()
        if source_id in forbidden:
            raise ValueError(f"forbidden source episode included: {source_id}")
        if source_path == legacy_root or legacy_root in source_path.parents:
            raise ValueError(f"legacy held-out source included: {source_path}")
        with h5py.File(source_path, "r") as h5_file:
            for key in ("/action", "/observations/qpos", "/observations/qvel"):
                if key not in h5_file:
                    raise ValueError(f"{source_path} missing {key}")
            action = h5_file["/action"]
            qpos = h5_file["/observations/qpos"]
            qvel = h5_file["/observations/qvel"]
            if action.ndim != 2 or qpos.ndim != 2 or qvel.ndim != 2:
                raise ValueError(f"{source_path} has invalid action/qpos/qvel rank")
            if action.shape[1] != 4 or qpos.shape[1] != 4 or qvel.shape[1] != 4:
                raise ValueError(f"{source_path} must contain four-axis signals")
            for camera in cameras:
                raw_camera = f"observations/images/{camera}"
                encoded_camera = f"observations/encoded_images/{camera}"
                if raw_camera not in h5_file and encoded_camera not in h5_file:
                    raise ValueError(f"{source_path} missing camera {camera}")

    effective = _mapping(train.get("effective_action", {}), "train.effective_action")
    expected_effective = bool(contract.get("effective_action_enabled", False))
    if bool(effective.get("enabled", False)) != expected_effective:
        raise ValueError("effective_action.enabled does not match experiment contract")
    deadzone_loss = _mapping(train.get("deadzone_loss", {}), "train.deadzone_loss")
    expected_deadzone_loss = bool(contract.get("deadzone_loss_enabled", False))
    if bool(deadzone_loss.get("enabled", False)) != expected_deadzone_loss:
        raise ValueError("deadzone_loss.enabled does not match experiment contract")
    state_hold_transition = _mapping(
        train.get("state_hold_transition", {}), "train.state_hold_transition"
    )
    expected_state_hold_transition = bool(
        contract.get("state_hold_transition_enabled", False)
    )
    if (
        bool(state_hold_transition.get("enabled", False))
        != expected_state_hold_transition
    ):
        raise ValueError(
            "state_hold_transition.enabled does not match experiment contract"
        )
    deadzone_sha256 = None
    if expected_effective:
        threshold_path = Path(str(effective.get("threshold_json", ""))).resolve()
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        metadata = threshold_payload.get("metadata", {})
        if metadata.get("policy_action_scale") != IDENTITY_ACTION_SCALE:
            raise ValueError("deadzone threshold provenance is not identity action scale")
        if metadata.get("action_domain") != "direct_policy_output":
            raise ValueError("deadzone threshold is not in direct policy output domain")
        deadzone_sha256 = sha256_file(threshold_path)

    pending: list[str] = []
    init_ckpt_raw = train.get("init_ckpt")
    if init_ckpt_raw:
        init_ckpt = Path(str(init_ckpt_raw)).expanduser().resolve()
        if not init_ckpt.is_file():
            if bool(contract.get("allow_pending_init_checkpoint", False)):
                pending.append(f"init checkpoint not produced yet: {init_ckpt}")
            else:
                raise FileNotFoundError(f"init_ckpt missing: {init_ckpt}")

    ckpt_dir = Path(str(train.get("ckpt_dir", ""))).expanduser().resolve()
    if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        raise FileExistsError(f"checkpoint directory is not empty: {ckpt_dir}")

    return {
        "status": "pending_dependency" if pending else "ready",
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "task_name": str(task.get("task_name", "")),
        "dataset_dir": str(dataset_dir),
        "manifest_sha256": composite["manifest_sha256"],
        "split_sha256": composite["split_sha256"],
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "camera_names": cameras,
        "camera_role_encoding_enabled": role_enabled,
        "low_dim_keys": list(policy["low_dim_keys"]),
        "num_epochs": int(train["num_epochs"]),
        "policy_action_scale": action_scale,
        "effective_action_enabled": expected_effective,
        "deadzone_loss_enabled": expected_deadzone_loss,
        "state_hold_transition_enabled": expected_state_hold_transition,
        "deadzone_sha256": deadzone_sha256,
        "checkpoint_dir": str(ckpt_dir),
        "pending": pending,
        "training_started": False,
    }


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value
