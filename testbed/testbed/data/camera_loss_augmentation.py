"""Deterministic train-only camera-loss augmentation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def resolve_camera_loss_augmentation(
    config: Mapping[str, Any] | None,
    *,
    camera_names: Sequence[str],
) -> dict[str, Any]:
    """Validate and normalize the single-camera loss contract."""

    raw = dict(config or {})
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return {
            "schema": "camera_loss_augmentation_v1",
            "enabled": False,
            "scope": "none",
            "target_camera": None,
            "probability": 0.0,
            "seed": None,
            "mask_rgb": [0, 0, 0],
            "decision_key": [],
        }
    if raw.get("scope") != "train_only":
        raise ValueError("camera loss augmentation requires scope=train_only")
    target = str(raw.get("target_camera", ""))
    names = [str(name) for name in camera_names]
    if target not in names:
        raise ValueError(
            f"camera loss target {target!r} is not in camera_names={names}"
        )
    probability = float(raw.get("probability", 0.0))
    if not 0.0 < probability < 1.0:
        raise ValueError("camera loss probability must be strictly between 0 and 1")
    seed = int(raw["seed"])
    mask = [int(value) for value in raw.get("mask_rgb", [0, 0, 0])]
    if len(mask) != 3 or any(not 0 <= value <= 255 for value in mask):
        raise ValueError("camera loss mask_rgb must contain three uint8 values")
    decision_key = list(
        raw.get(
            "decision_key",
            ["seed", "source_episode_id", "source_tick"],
        )
    )
    if decision_key != ["seed", "source_episode_id", "source_tick"]:
        raise ValueError(
            "camera loss decision_key must be [seed, source_episode_id, source_tick]"
        )
    return {
        "schema": "camera_loss_augmentation_v1",
        "enabled": True,
        "scope": "train_only",
        "target_camera": target,
        "probability": probability,
        "seed": seed,
        "mask_rgb": mask,
        "decision_key": decision_key,
        "selection": "sha256_uniform_u64_less_than_probability",
    }


def camera_loss_selected(
    config: Mapping[str, Any],
    *,
    episode_id: int,
    source_tick: int,
) -> bool:
    """Return the stable augmentation decision for one source row."""

    if not config.get("enabled", False):
        return False
    payload = f"{int(config['seed'])}:{int(episode_id)}:{int(source_tick)}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    uniform = value / float(1 << 64)
    return uniform < float(config["probability"])


def apply_camera_loss(
    images: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    episode_id: int,
    source_tick: int,
) -> dict[str, np.ndarray]:
    """Copy the camera map and mask only the declared selected camera."""

    result = {str(name): np.asarray(image).copy() for name, image in images.items()}
    if not camera_loss_selected(
        config,
        episode_id=episode_id,
        source_tick=source_tick,
    ):
        return result
    target = str(config["target_camera"])
    if target not in result:
        raise ValueError(f"camera loss target {target!r} is absent from image map")
    mask = np.asarray(config["mask_rgb"], dtype=result[target].dtype)
    result[target][...] = mask
    return result


def camera_loss_manifest(
    config: Mapping[str, Any],
    *,
    sample_keys: Sequence[tuple[int, int]],
    source_episode_ids: Sequence[int],
) -> dict[str, Any]:
    """Bind the exact eligible and selected source rows."""

    ordered = sorted({(int(episode_id), int(tick)) for episode_id, tick in sample_keys})
    selected = [
        key
        for key in ordered
        if camera_loss_selected(
            config,
            episode_id=key[0],
            source_tick=key[1],
        )
    ]
    payload = json.dumps(selected, separators=(",", ":")).encode()
    return {
        **dict(config),
        "eligible_row_count": len(ordered),
        "selected_row_count": len(selected),
        "selected_fraction": (float(len(selected) / len(ordered)) if ordered else 0.0),
        "selected_keys_sha256": hashlib.sha256(payload).hexdigest(),
        "source_episode_ids": sorted(set(map(int, source_episode_ids))),
        "held_out_test_read": False,
    }
