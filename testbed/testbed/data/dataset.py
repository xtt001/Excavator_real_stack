"""
EpisodicDataset, get_norm_stats, load_data.

PyTorch data loading utilities for real-excavator HDF5 episodes.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from testbed.data.deadzone_intent_labels import compute_deadzone_intent_labels
from testbed.data.hdf5_io import list_episodes
from testbed.data.image_transforms import build_image_transform
from testbed.data.schema import ATTR_IS_REAL, GRP_ENCODED_IMAGES

SUPPORTED_LOW_DIM_KEYS = ("qpos", "qvel", "cycle_condition_v1")


def _normalize_low_dim_keys(
    low_dim_keys: list[str] | tuple[str, ...] | None,
) -> list[str]:
    keys = ["qpos"] if not low_dim_keys else [str(key) for key in low_dim_keys]
    invalid = [key for key in keys if key not in SUPPORTED_LOW_DIM_KEYS]
    if invalid:
        raise ValueError(
            f"Unsupported low_dim_keys {invalid}. "
            f"Supported keys: {SUPPORTED_LOW_DIM_KEYS}."
        )
    return keys


def _assemble_low_dim_observation(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    cycle_condition_v1: np.ndarray | None = None,
    low_dim_keys: list[str],
) -> np.ndarray:
    qpos_arr = np.asarray(qpos, dtype=np.float32)
    qvel_arr = np.asarray(qvel, dtype=np.float32)
    condition_arr = (
        None
        if cycle_condition_v1 is None
        else np.asarray(cycle_condition_v1, dtype=np.float32)
    )
    sequence_mode = (
        qpos_arr.ndim > 1
        or qvel_arr.ndim > 1
        or (condition_arr is not None and condition_arr.ndim > 1)
    )
    parts: list[np.ndarray] = []
    for key in low_dim_keys:
        if key == "qpos":
            part = qpos_arr
        elif key == "qvel":
            part = qvel_arr
        elif key == "cycle_condition_v1":
            if condition_arr is None:
                raise ValueError("cycle_condition_v1 is configured but unavailable.")
            part = condition_arr
        else:
            continue
        if sequence_mode:
            part = part.reshape(part.shape[0], -1)
        else:
            part = part.reshape(-1)
        parts.append(part)
    if not parts:
        raise ValueError("low_dim_keys must contain at least one supported key.")
    axis = 1 if sequence_mode else 0
    return np.concatenate(parts, axis=axis).astype(np.float32)


# ─── Normalization stats ──────────────────────────────────────────────────────


def get_norm_stats(
    dataset_dir: str | Path,
    num_episodes: int,
    episode_ids: list[int] | None = None,
    low_dim_keys: list[str] | tuple[str, ...] | None = None,
    deadzone_intent: dict[str, Any] | None = None,
    valid_mask_path: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute mean/std normalization statistics from a set of episodes.

    Parameters
    ----------
    dataset_dir   Directory containing episode_N.hdf5 files.
    num_episodes  Maximum number of episodes to consider.
    episode_ids   Explicit list of episode indices to use.  If None, uses
                  range(num_episodes) and skips missing files.

    Returns
    -------
    {
      "action_mean":  (Na,)  float32
      "action_std":   (Na,)  float32
      "proprio_mean": (Np,)  float32
      "proprio_std":  (Np,)  float32
      "example_proprio": (T, Np) float32
      "qpos_mean":    (Nq,)  float32    qpos-only alias when low_dim_keys=['qpos']
      "qpos_std":     (Nq,)  float32    qpos-only alias when low_dim_keys=['qpos']
      "example_qpos": (T, Nq) float32   qpos-only alias when low_dim_keys=['qpos']
    }
    """
    import h5py

    dataset_dir = Path(dataset_dir)
    selected_low_dim_keys = _normalize_low_dim_keys(low_dim_keys)
    deadzone_intent_cfg = _resolve_deadzone_intent_config(deadzone_intent)
    all_proprio_data: list[torch.Tensor] = []
    all_qpos_data: list[torch.Tensor] = []
    all_action_data: list[torch.Tensor] = []
    example_qpos = None
    example_proprio = None

    ids = episode_ids if episode_ids is not None else list(range(num_episodes))
    for ep_idx in ids:
        p = dataset_dir / f"episode_{ep_idx}.hdf5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            qpos = f["/observations/qpos"][()]
            qvel = f["/observations/qvel"][()]
            cycle_condition = (
                f["/conditions/cycle_condition_v1"][()]
                if "cycle_condition_v1" in selected_low_dim_keys
                else None
            )
            action = f["/action"][()]
            valid_mask = np.ones(int(action.shape[0]), dtype=bool)
            if valid_mask_path:
                valid_mask = _read_required_valid_mask(
                    f,
                    valid_mask_path,
                    int(action.shape[0]),
                )
            action_mask = valid_mask.copy()
            if deadzone_intent_cfg["use_action_loss_mask_for_stats"]:
                mask = _read_optional_handoff_mask(
                    f,
                    "handoff/action_loss_mask",
                    int(action.shape[0]),
                    enabled=True,
                )
                if mask is None:
                    raise ValueError(
                        "deadzone_intent.use_action_loss_mask_for_stats requires "
                        "handoff/action_loss_mask"
                    )
                action_mask &= mask
            qpos = np.asarray(qpos, dtype=np.float32)[valid_mask]
            qvel = np.asarray(qvel, dtype=np.float32)[valid_mask]
            if cycle_condition is not None:
                cycle_condition = np.asarray(
                    cycle_condition,
                    dtype=np.float32,
                )[valid_mask]
            action = np.asarray(action, dtype=np.float32)[action_mask]
            if qpos.shape[0] == 0 or action.shape[0] == 0:
                raise ValueError(
                    f"Episode {ep_idx} has no rows for normalization after masks"
                )
        proprio = _assemble_low_dim_observation(
            qpos=qpos,
            qvel=qvel,
            cycle_condition_v1=cycle_condition,
            low_dim_keys=selected_low_dim_keys,
        )
        all_proprio_data.append(torch.from_numpy(proprio))
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        example_qpos = qpos
        example_proprio = proprio

    if not all_proprio_data:
        raise FileNotFoundError(
            f"No episodes found under {dataset_dir}. "
            "Expected files like episode_0.hdf5."
        )

    # Do not assume all episodes share the same timestep length. Operator
    # controlled recordings can stop at different lengths, so stats are
    # computed over the concatenated time axis.
    proprio_tensor = torch.cat(all_proprio_data, dim=0)  # (sum_T, Np)
    qpos_tensor = torch.cat(all_qpos_data, dim=0)  # (sum_T, Nq)
    action_tensor = torch.cat(all_action_data, dim=0)  # (sum_T, Na)

    action_mean = action_tensor.mean(dim=0, keepdim=True)
    action_std = action_tensor.std(dim=0, keepdim=True).clamp(min=1e-2)
    proprio_mean = proprio_tensor.mean(dim=0, keepdim=True)
    proprio_std = proprio_tensor.std(dim=0, keepdim=True).clamp(min=1e-2)
    qpos_mean = qpos_tensor.mean(dim=0, keepdim=True)
    qpos_std = qpos_tensor.std(dim=0, keepdim=True).clamp(min=1e-2)

    stats = {
        "action_mean": action_mean.numpy().squeeze().astype(np.float32),
        "action_std": action_std.numpy().squeeze().astype(np.float32),
        "proprio_mean": proprio_mean.numpy().squeeze().astype(np.float32),
        "proprio_std": proprio_std.numpy().squeeze().astype(np.float32),
        "example_proprio": example_proprio,
        "proprio_keys": np.asarray(selected_low_dim_keys, dtype=object),
        "proprio_dim": int(proprio_tensor.shape[1]),
        "qpos_only_dim": int(qpos_tensor.shape[1]),
    }
    if selected_low_dim_keys == ["qpos"]:
        stats.update(
            {
                "qpos_mean": qpos_mean.numpy().squeeze().astype(np.float32),
                "qpos_std": qpos_std.numpy().squeeze().astype(np.float32),
                "example_qpos": example_qpos,
            }
        )
    return stats


# ─── Dataset ─────────────────────────────────────────────────────────────────


class EpisodicDataset(Dataset):
    """
    PyTorch Dataset over a set of HDF5 episode files.

    Each __getitem__ samples a random start timestep t0 from episode_i,
    then returns:
      image_data  : (n_cams, C, H, W)   float32 [0, 1]
      proprio_data: (Np,)               float32 normalised
      action_data : (T - t0, Na)        float32 normalised + zero-padded to T
      is_pad      : (T,)                bool    True where zero-padded

    Parameters
    ----------
    episode_ids   List of integer episode indices.
    dataset_dir   Directory with episode_N.hdf5 files.
    camera_names  Cameras to include (in order).
    norm_stats    Dict returned by get_norm_stats.
    """

    def __init__(
        self,
        episode_ids: list[int],
        dataset_dir: str | Path,
        camera_names: list[str],
        norm_stats: dict[str, np.ndarray],
        episode_len: int | None = None,
        low_dim_keys: list[str] | tuple[str, ...] | None = None,
        action_chunk_size: int | None = None,
        image_transform: str = "none",
        deadzone_intent: dict[str, Any] | None = None,
        sample_valid_mask_path: str | None = None,
        condition_shuffle_seed: int | None = None,
        condition_phase_randomization: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = Path(dataset_dir)
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = int(episode_len) if episode_len is not None else None
        self.low_dim_keys = _normalize_low_dim_keys(low_dim_keys)
        self.action_chunk_size = (
            int(action_chunk_size) if action_chunk_size is not None else None
        )
        self.image_transform_name = str(image_transform or "none")
        self.image_transform = build_image_transform(self.image_transform_name)
        self.deadzone_intent = _resolve_deadzone_intent_config(deadzone_intent)
        self.sample_valid_mask_path = (
            str(sample_valid_mask_path) if sample_valid_mask_path else None
        )
        self.condition_shuffle_seed = (
            None if condition_shuffle_seed is None else int(condition_shuffle_seed)
        )
        if (
            self.condition_shuffle_seed is not None
            and "cycle_condition_v1" not in self.low_dim_keys
        ):
            raise ValueError(
                "condition shuffle requires cycle_condition_v1 in low_dim_keys"
            )
        self.condition_phase_randomization = dict(condition_phase_randomization or {})
        phase_randomization_enabled = bool(
            self.condition_phase_randomization.get("enabled", False)
        )
        if (
            phase_randomization_enabled
            and "cycle_condition_v1" not in self.low_dim_keys
        ):
            raise ValueError(
                "condition phase randomization requires "
                "cycle_condition_v1 in low_dim_keys"
            )
        if phase_randomization_enabled and self.condition_shuffle_seed is not None:
            raise ValueError(
                "condition shuffle and phase randomization are mutually exclusive"
            )
        (
            self.condition_shuffle_mapping,
            self.condition_shuffle_manifest,
        ) = _build_condition_shuffle_mapping(
            dataset_dir=self.dataset_dir,
            episode_ids=self.episode_ids,
            action_chunk_size=self.action_chunk_size,
            deadzone_intent=self.deadzone_intent,
            sample_valid_mask_path=self.sample_valid_mask_path,
            seed=self.condition_shuffle_seed,
        )
        (
            self.condition_phase_randomization_mapping,
            self.condition_phase_randomization_manifest,
        ) = _build_condition_phase_randomization_mapping(
            dataset_dir=self.dataset_dir,
            episode_ids=self.episode_ids,
            action_chunk_size=self.action_chunk_size,
            deadzone_intent=self.deadzone_intent,
            sample_valid_mask_path=self.sample_valid_mask_path,
            config=self.condition_phase_randomization,
        )
        self.is_real: bool | None = None
        # Warm-up to populate self.is_real
        self.__getitem__(0)

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, index: int):
        import h5py

        ep_id = self.episode_ids[index]
        path = self.dataset_dir / f"episode_{ep_id}.hdf5"

        with h5py.File(path, "r") as f:
            is_real: bool = bool(f.attrs.get(ATTR_IS_REAL, True))
            metadata = dict(f["metadata"].attrs) if "metadata" in f else {}
            action_prealigned = _bool_attr(metadata.get("action_prealigned", False))
            original_action_shape = f["/action"].shape
            T = original_action_shape[0]

            # ── sample start timestep ─────────────────────────────────────
            train_exclude_mask = _combined_training_exclude_mask(
                f,
                T,
                sample_valid_mask_path=self.sample_valid_mask_path,
            )
            action_loss_start_mask = _read_optional_handoff_mask(
                f,
                "handoff/action_loss_mask",
                T,
                enabled=bool(self.deadzone_intent["require_action_loss_in_chunk"]),
            )
            valid_starts = _valid_start_indices(
                total_steps=T,
                train_exclude_mask=train_exclude_mask,
                action_chunk_size=self.action_chunk_size,
                action_loss_mask=action_loss_start_mask,
                require_action_loss_in_chunk=bool(
                    self.deadzone_intent["require_action_loss_in_chunk"]
                ),
            )
            if valid_starts.size == 0:
                raise ValueError(
                    f"Episode {ep_id} has no valid training start after train_exclude_mask."
                )
            t0 = int(np.random.choice(valid_starts))

            # ── observation at t0 ─────────────────────────────────────────
            qpos = f["/observations/qpos"][t0]
            qvel = f["/observations/qvel"][t0]
            cycle_condition = (
                np.asarray(
                    f["/conditions/cycle_condition_v1"][t0],
                    dtype=np.float32,
                )
                if "cycle_condition_v1" in self.low_dim_keys
                else None
            )
            if self.condition_shuffle_mapping is not None:
                cycle_condition = self.condition_shuffle_mapping[(ep_id, t0)]
            if self.condition_phase_randomization_mapping is not None:
                cycle_condition = self.condition_phase_randomization_mapping.get(
                    (ep_id, t0),
                    cycle_condition,
                )
            proprio = _assemble_low_dim_observation(
                qpos=qpos,
                qvel=qvel,
                cycle_condition_v1=cycle_condition,
                low_dim_keys=self.low_dim_keys,
            )
            image_dict = {}
            for cam in self.camera_names:
                image = _read_camera_image(f, cam, t0)
                if self.image_transform is not None:
                    image = self.image_transform(image)
                image_dict[cam] = image

            # ── action from t0 onward ────────────────────────────────────
            start = t0 if (not is_real or action_prealigned) else max(0, t0 - 1)
            action = f["/action"][start:]
            action_len = T - start
            deadzone_labels = None
            if self.deadzone_intent["enabled"]:
                full_action = np.asarray(f["/action"][()], dtype=np.float32)
                action_loss_mask = _read_optional_handoff_mask(
                    f,
                    "handoff/action_loss_mask",
                    T,
                    enabled=bool(self.deadzone_intent["use_handoff_masks"]),
                )
                tail_idle_mask = _read_optional_handoff_mask(
                    f,
                    "handoff/tail_idle_mask",
                    T,
                    enabled=bool(self.deadzone_intent["use_handoff_masks"]),
                )
                owner_automation = _read_optional_handoff_mask(
                    f,
                    "handoff/owner_automation",
                    T,
                    enabled=bool(self.deadzone_intent["use_handoff_masks"]),
                )
                deadzone_labels = compute_deadzone_intent_labels(
                    actions=full_action,
                    thresholds=self.deadzone_intent["thresholds"],
                    action_loss_mask=action_loss_mask,
                    tail_idle_mask=tail_idle_mask,
                    owner_automation=owner_automation,
                )

        self.is_real = is_real

        # ── pad action to fixed dataset length for batching ────────────────
        target_len = self.episode_len if self.episode_len is not None else T
        if T > target_len:
            raise ValueError(
                f"Episode {ep_id} has length {T}, which exceeds configured "
                f"episode_len {target_len}. Increase task.episode_len or re-record."
            )

        padded_action = np.zeros(
            (target_len, original_action_shape[1]), dtype=np.float32
        )
        padded_action[:action_len] = action
        is_pad = np.ones(target_len, dtype=bool)
        is_pad[:action_len] = False
        padded_move_mask = None
        padded_stop_mask = None
        padded_wrong_mask = None
        padded_action_loss_mask = None
        if deadzone_labels is not None:
            padded_move_mask = np.zeros(
                (target_len, original_action_shape[1], 2), dtype=bool
            )
            padded_stop_mask = np.zeros(target_len, dtype=bool)
            padded_wrong_mask = np.zeros(
                (target_len, original_action_shape[1], 2), dtype=bool
            )
            padded_action_loss_mask = np.zeros(target_len, dtype=bool)
            padded_move_mask[:action_len] = deadzone_labels.move_mask[start:]
            padded_stop_mask[:action_len] = deadzone_labels.stop_mask[start:]
            padded_wrong_mask[:action_len] = deadzone_labels.wrong_mask[start:]
            padded_action_loss_mask[:action_len] = deadzone_labels.action_loss_mask[
                start:
            ]

        # ── assemble camera tensor ─────────────────────────────────────────
        all_cam_images = np.stack(
            [image_dict[c] for c in self.camera_names], axis=0
        )  # (n_cams, H, W, 3)

        # ── convert to tensors ────────────────────────────────────────────
        image_data = torch.from_numpy(all_cam_images)
        proprio_data = torch.from_numpy(proprio).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad_t = torch.from_numpy(is_pad)

        # channel-last → channel-first + normalize to [0, 1]
        image_data = torch.einsum("k h w c -> k c h w", image_data).float() / 255.0

        # normalise proprio and actions
        action_data = (
            action_data - torch.from_numpy(self.norm_stats["action_mean"])
        ) / torch.from_numpy(self.norm_stats["action_std"])
        proprio_data = (
            proprio_data - torch.from_numpy(self.norm_stats["proprio_mean"])
        ) / torch.from_numpy(self.norm_stats["proprio_std"])

        if deadzone_labels is None:
            return image_data, proprio_data, action_data, is_pad_t

        return {
            "image": image_data,
            "proprio": proprio_data,
            "action": action_data,
            "is_pad": is_pad_t,
            "deadzone_move_mask": torch.from_numpy(padded_move_mask),
            "deadzone_stop_mask": torch.from_numpy(padded_stop_mask),
            "deadzone_wrong_mask": torch.from_numpy(padded_wrong_mask),
            "action_loss_mask": torch.from_numpy(padded_action_loss_mask),
        }


def _read_camera_image(h5_file: Any, camera_name: str, timestep: int) -> np.ndarray:
    raw_path = f"observations/images/{camera_name}"
    if raw_path in h5_file:
        return np.asarray(h5_file[raw_path][timestep], dtype=np.uint8)
    encoded_path = f"{GRP_ENCODED_IMAGES}/{camera_name}"
    if encoded_path not in h5_file:
        raise KeyError(
            f"Camera {camera_name!r} not found as raw or encoded image data."
        )
    encoded = np.asarray(h5_file[encoded_path][timestep], dtype=np.uint8).reshape(-1)
    return _decode_jpeg_image(encoded)


def _bool_attr(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return bool(int(value))
    except Exception:
        return str(value).strip().lower() in {"true", "yes", "1"}


def _build_condition_shuffle_mapping(
    *,
    dataset_dir: Path,
    episode_ids: list[int],
    action_chunk_size: int | None,
    deadzone_intent: dict[str, Any],
    sample_valid_mask_path: str | None,
    seed: int | None,
) -> tuple[
    dict[tuple[int, int], np.ndarray] | None,
    dict[str, Any],
]:
    if seed is None:
        return None, {
            "enabled": False,
            "scope": "none",
        }

    import h5py

    keys: list[tuple[int, int]] = []
    values: list[np.ndarray] = []
    for episode_id in episode_ids:
        path = dataset_dir / f"episode_{episode_id}.hdf5"
        with h5py.File(path, "r") as handle:
            total_steps = int(handle["/action"].shape[0])
            condition_path = "/conditions/cycle_condition_v1"
            if condition_path not in handle:
                raise KeyError(f"condition shuffle requires {condition_path}: {path}")
            condition = np.asarray(
                handle[condition_path][()],
                dtype=np.float32,
            )
            if condition.shape != (total_steps, 6):
                raise ValueError("cycle_condition_v1 must have shape (T, 6)")
            action_loss_start_mask = _read_optional_handoff_mask(
                handle,
                "handoff/action_loss_mask",
                total_steps,
                enabled=bool(deadzone_intent["require_action_loss_in_chunk"]),
            )
            valid_starts = _valid_start_indices(
                total_steps=total_steps,
                train_exclude_mask=_combined_training_exclude_mask(
                    handle,
                    total_steps,
                    sample_valid_mask_path=sample_valid_mask_path,
                ),
                action_chunk_size=action_chunk_size,
                action_loss_mask=action_loss_start_mask,
                require_action_loss_in_chunk=bool(
                    deadzone_intent["require_action_loss_in_chunk"]
                ),
            )
            for tick in valid_starts.tolist():
                vector = condition[int(tick)]
                _validate_cycle_condition_vector(vector)
                keys.append((int(episode_id), int(tick)))
                values.append(vector.copy())
    if not keys:
        raise ValueError("condition shuffle has no valid training starts")

    matrix = np.stack(values).astype(np.float32)
    permutation = np.random.default_rng(seed).permutation(len(keys))
    shuffled = matrix[permutation].copy()
    source_counts = Counter(_condition_key(row) for row in matrix)
    shuffled_counts = Counter(_condition_key(row) for row in shuffled)
    if source_counts != shuffled_counts:
        raise AssertionError("condition shuffle changed token marginals")
    mapping = {key: shuffled[index].copy() for index, key in enumerate(keys)}
    digest = hashlib.sha256()
    for key in keys:
        digest.update(np.asarray(key, dtype=np.int64).tobytes())
        digest.update(mapping[key].astype(np.float32).tobytes())
    changed = int(np.sum(np.any(matrix != shuffled, axis=1)))
    return mapping, {
        "enabled": True,
        "scope": "train_valid_starts_only",
        "key": "cycle_condition_v1",
        "seed": int(seed),
        "row_count": len(keys),
        "changed_row_count": changed,
        "unchanged_row_count": len(keys) - changed,
        "changed_row_fraction": float(changed / len(keys)),
        "source_token_counts": {
            key: int(count) for key, count in sorted(source_counts.items())
        },
        "shuffled_token_counts": {
            key: int(count) for key, count in sorted(shuffled_counts.items())
        },
        "mapping_sha256": digest.hexdigest(),
    }


def _build_condition_phase_randomization_mapping(
    *,
    dataset_dir: Path,
    episode_ids: list[int],
    action_chunk_size: int | None,
    deadzone_intent: dict[str, Any],
    sample_valid_mask_path: str | None,
    config: dict[str, Any],
) -> tuple[
    dict[tuple[int, int], np.ndarray] | None,
    dict[str, Any],
]:
    if not bool(config.get("enabled", False)):
        return None, {
            "enabled": False,
            "scope": "none",
        }
    if (
        config.get("key") != "cycle_condition_v1.next_sector"
        or config.get("scope") != "train_only"
        or config.get("phase_boundary") != "dump_end_proxy.representative_target_tick"
        or config.get("eligibility_rule") != "t0_plus_chunk_le_dump_end"
    ):
        raise ValueError(
            "condition phase randomization requires the frozen B1.1 key, "
            "scope, phase boundary, and eligibility rule"
        )
    if action_chunk_size is None or int(action_chunk_size) <= 0:
        raise ValueError(
            "condition phase randomization requires a positive action chunk size"
        )
    annotation_path = Path(str(config.get("annotation_path", "")))
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"condition phase annotation file does not exist: {annotation_path}"
        )
    expected_sha = str(config.get("annotation_sha256", "")).strip()
    observed_sha = _sha256_path(annotation_path)
    if not expected_sha or observed_sha != expected_sha:
        raise ValueError(
            "condition phase annotation SHA mismatch: "
            f"expected={expected_sha} observed={observed_sha}"
        )
    seed = int(config.get("seed", 0))
    boundaries = _load_train_dump_end_boundaries(
        annotation_path,
        episode_ids=episode_ids,
    )

    import h5py

    keys: list[tuple[int, int]] = []
    values: list[np.ndarray] = []
    valid_start_total = 0
    preserved_crossing_or_post = 0
    for episode_id in episode_ids:
        path = dataset_dir / f"episode_{episode_id}.hdf5"
        with h5py.File(path, "r") as handle:
            total_steps = int(handle["/action"].shape[0])
            condition_path = "/conditions/cycle_condition_v1"
            cycle_id_path = "/conditions/cycle_id"
            if condition_path not in handle or cycle_id_path not in handle:
                raise KeyError(
                    "condition phase randomization requires "
                    f"{condition_path} and {cycle_id_path}: {path}"
                )
            condition = np.asarray(handle[condition_path][()], dtype=np.float32)
            cycle_ids = np.asarray(handle[cycle_id_path][()], dtype=np.int64)
            if condition.shape != (total_steps, 6) or cycle_ids.shape != (total_steps,):
                raise ValueError(
                    "condition phase arrays must have shapes (T, 6) and (T,)"
                )
            action_loss_start_mask = _read_optional_handoff_mask(
                handle,
                "handoff/action_loss_mask",
                total_steps,
                enabled=bool(deadzone_intent["require_action_loss_in_chunk"]),
            )
            valid_starts = _valid_start_indices(
                total_steps=total_steps,
                train_exclude_mask=_combined_training_exclude_mask(
                    handle,
                    total_steps,
                    sample_valid_mask_path=sample_valid_mask_path,
                ),
                action_chunk_size=action_chunk_size,
                action_loss_mask=action_loss_start_mask,
                require_action_loss_in_chunk=bool(
                    deadzone_intent["require_action_loss_in_chunk"]
                ),
            )
            valid_start_total += int(valid_starts.size)
            for tick_raw in valid_starts.tolist():
                tick = int(tick_raw)
                cycle_id = int(cycle_ids[tick])
                boundary_key = (int(episode_id), cycle_id)
                if boundary_key not in boundaries:
                    raise ValueError(
                        "valid training start lacks an accepted train dump-end "
                        f"annotation: episode={episode_id} tick={tick} "
                        f"cycle_id={cycle_id}"
                    )
                vector = condition[tick]
                _validate_cycle_condition_vector(vector)
                dump_end = int(boundaries[boundary_key]["dump_end_target_tick"])
                if tick + int(action_chunk_size) <= dump_end:
                    if not np.array_equal(
                        vector,
                        boundaries[boundary_key]["condition"],
                    ):
                        raise ValueError(
                            "episode condition disagrees with annotation at "
                            f"episode={episode_id} cycle={cycle_id}"
                        )
                    keys.append((int(episode_id), tick))
                    values.append(vector.copy())
                else:
                    preserved_crossing_or_post += 1
    if not keys:
        raise ValueError("condition phase randomization has no eligible train starts")

    matrix = np.stack(values).astype(np.float32)
    source_labels = np.argmax(matrix[:, 3:], axis=1).astype(np.int64)
    randomized_labels = _exact_marginal_label_derangement(
        source_labels,
        seed=seed,
    )
    randomized = matrix.copy()
    randomized[:, 3:] = 0.0
    randomized[np.arange(randomized.shape[0]), 3 + randomized_labels] = 1.0
    source_counts = Counter(int(value) for value in source_labels.tolist())
    randomized_counts = Counter(int(value) for value in randomized_labels.tolist())
    if source_counts != randomized_counts:
        raise AssertionError("phase randomization changed next-sector marginals")
    mapping = {key: randomized[index].copy() for index, key in enumerate(keys)}
    digest = hashlib.sha256()
    for key in keys:
        digest.update(np.asarray(key, dtype=np.int64).tobytes())
        digest.update(mapping[key].astype(np.float32).tobytes())
    changed = int(np.sum(source_labels != randomized_labels))
    sector_names = ("left", "center", "right")
    return mapping, {
        "schema": "condition_next_phase_randomization_manifest_v1",
        "enabled": True,
        "scope": "train_chunk_safe_pre_dump_valid_starts_only",
        "key": "cycle_condition_v1.next_sector",
        "seed": seed,
        "action_chunk_size": int(action_chunk_size),
        "phase_boundary": "dump_end_proxy.representative_target_tick",
        "eligibility_rule": "t0_plus_chunk_le_dump_end",
        "annotation_path": str(annotation_path.resolve()),
        "annotation_sha256": observed_sha,
        "annotation_episode_ids": sorted({int(key[0]) for key in boundaries}),
        "annotation_cycle_count": len(boundaries),
        "valid_start_count": valid_start_total,
        "eligible_randomized_start_count": len(keys),
        "preserved_crossing_or_post_start_count": preserved_crossing_or_post,
        "changed_row_count": changed,
        "unchanged_eligible_row_count": len(keys) - changed,
        "changed_eligible_fraction": float(changed / len(keys)),
        "source_next_sector_counts": {
            sector_names[key]: int(value)
            for key, value in sorted(source_counts.items())
        },
        "randomized_next_sector_counts": {
            sector_names[key]: int(value)
            for key, value in sorted(randomized_counts.items())
        },
        "current_sector_unchanged": True,
        "post_boundary_condition_unchanged": True,
        "normalization_stats_unchanged": True,
        "mapping_sha256": digest.hexdigest(),
    }


def _load_train_dump_end_boundaries(
    path: Path,
    *,
    episode_ids: list[int],
) -> dict[tuple[int, int], dict[str, Any]]:
    selected_ids = {int(value) for value in episode_ids}
    boundaries: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = int(row["episode_id"])
            if episode_id not in selected_ids:
                continue
            if (
                row.get("split") != "train"
                or row.get("quality", {}).get("status") != "accepted"
            ):
                continue
            cycle_id = int(row["cycle_id"])
            vector = np.asarray(
                row["policy_condition"]["vector"],
                dtype=np.float32,
            )
            _validate_cycle_condition_vector(vector)
            key = (episode_id, cycle_id)
            if key in boundaries:
                raise ValueError(f"duplicate accepted annotation for {key}")
            boundaries[key] = {
                "dump_end_target_tick": int(
                    row["observable_events"]["dump_end_proxy"][
                        "representative_target_tick"
                    ]
                ),
                "condition": vector,
            }
    if not boundaries:
        raise ValueError("condition phase annotation selected no train cycles")
    return boundaries


def _exact_marginal_label_derangement(
    labels: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    source = np.asarray(labels, dtype=np.int64).reshape(-1)
    if source.size < 2 or np.any((source < 0) | (source > 2)):
        raise ValueError("next-sector labels must be a non-empty 0..2 vector")
    counts = Counter(int(value) for value in source.tolist())
    maximum = max(counts.values())
    if maximum > source.size - maximum:
        raise ValueError(
            "exact-marginal next-sector derangement is impossible for the "
            f"observed counts: {dict(sorted(counts.items()))}"
        )
    rng = np.random.default_rng(seed)
    ordered_indices: list[int] = []
    for label, _count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        indices = np.flatnonzero(source == label)
        ordered_indices.extend(rng.permutation(indices).tolist())
    receiver = np.asarray(ordered_indices, dtype=np.int64)
    ordered_labels = source[receiver]
    donor_labels = np.roll(ordered_labels, -maximum)
    randomized = np.empty_like(source)
    randomized[receiver] = donor_labels
    if np.any(randomized == source):
        raise AssertionError("constructed next-sector assignment is not a derangement")
    if Counter(randomized.tolist()) != counts:
        raise AssertionError("constructed derangement changed label marginals")
    return randomized


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cycle_condition_vector(vector: np.ndarray) -> None:
    value = np.asarray(vector, dtype=np.float32)
    if (
        value.shape != (6,)
        or not np.isfinite(value).all()
        or not np.all(np.isin(value, (0.0, 1.0)))
        or float(np.sum(value[:3])) != 1.0
        or float(np.sum(value[3:])) != 1.0
    ):
        raise ValueError("cycle_condition_v1 must contain two one-hot[3] fields")


def _condition_key(vector: np.ndarray) -> str:
    return ",".join(str(int(value)) for value in vector.tolist())


def _resolve_deadzone_intent_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "thresholds": {},
            "use_handoff_masks": False,
            "use_action_loss_mask_for_stats": False,
            "require_action_loss_in_chunk": False,
        }
    thresholds = cfg.get("thresholds")
    if thresholds is None:
        threshold_json = cfg.get("threshold_json")
        if not threshold_json:
            raise ValueError(
                "deadzone_intent.enabled requires thresholds or threshold_json"
            )
        path = Path(str(threshold_json))
        if not path.exists():
            raise FileNotFoundError(
                f"deadzone_intent threshold_json does not exist: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        thresholds = (
            payload.get("deadzone_action", payload)
            if isinstance(payload, dict)
            else payload
        )
    if not isinstance(thresholds, dict):
        raise ValueError("deadzone_intent thresholds must be a mapping")
    return {
        "enabled": True,
        "thresholds": thresholds,
        "use_handoff_masks": bool(cfg.get("use_handoff_masks", False)),
        "use_action_loss_mask_for_stats": bool(
            cfg.get("use_action_loss_mask_for_stats", False)
        ),
        "require_action_loss_in_chunk": bool(
            cfg.get("require_action_loss_in_chunk", False)
        ),
    }


def _read_optional_handoff_mask(
    h5_file: Any,
    path: str,
    total_steps: int,
    *,
    enabled: bool,
) -> np.ndarray | None:
    if not enabled:
        return None
    if path not in h5_file:
        return None
    arr = np.asarray(h5_file[path][()], dtype=bool).reshape(-1)
    if arr.size != int(total_steps):
        raise ValueError(f"{path} length must be {total_steps}, got {arr.size}")
    return arr


def _read_train_exclude_mask(h5_file: Any, total_steps: int) -> np.ndarray | None:
    path = "diagnostics/train_exclude_mask"
    if path not in h5_file:
        return None
    mask = np.asarray(h5_file[path][()], dtype=bool).reshape(-1)
    if mask.size != int(total_steps):
        return None
    return mask


def _read_required_valid_mask(
    h5_file: Any,
    path: str,
    total_steps: int,
) -> np.ndarray:
    normalized = str(path).strip().lstrip("/")
    if not normalized:
        raise ValueError("sample valid-mask path must not be empty")
    if normalized not in h5_file:
        raise KeyError(f"required sample valid-mask is missing: {normalized}")
    mask = np.asarray(h5_file[normalized][()], dtype=bool).reshape(-1)
    if mask.size != int(total_steps):
        raise ValueError(f"{normalized} length must be {total_steps}, got {mask.size}")
    return mask


def _combined_training_exclude_mask(
    h5_file: Any,
    total_steps: int,
    *,
    sample_valid_mask_path: str | None,
) -> np.ndarray | None:
    excluded = _read_train_exclude_mask(h5_file, total_steps)
    if not sample_valid_mask_path:
        return excluded
    derived_exclude = ~_read_required_valid_mask(
        h5_file,
        sample_valid_mask_path,
        total_steps,
    )
    if excluded is None:
        return derived_exclude
    return np.asarray(excluded, dtype=bool) | derived_exclude


def _valid_start_indices(
    *,
    total_steps: int,
    train_exclude_mask: np.ndarray | None,
    action_chunk_size: int | None,
    action_loss_mask: np.ndarray | None = None,
    require_action_loss_in_chunk: bool = False,
) -> np.ndarray:
    """Return t0 values whose sampled action window does not cross masked samples."""

    T = int(total_steps)
    if T <= 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(T, dtype=np.int64)
    horizon = max(1, int(action_chunk_size) if action_chunk_size is not None else 1)
    ends = np.minimum(starts + horizon, T)
    valid = np.ones(T, dtype=bool)

    if train_exclude_mask is not None:
        mask = np.asarray(train_exclude_mask, dtype=bool).reshape(-1)
        if mask.size == T and np.any(mask):
            prefix = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
            window_has_mask = (prefix[ends] - prefix[starts]) > 0
            valid &= ~window_has_mask

    if require_action_loss_in_chunk:
        if action_loss_mask is None:
            raise ValueError(
                "deadzone_intent.require_action_loss_in_chunk requires "
                "handoff/action_loss_mask"
            )
        action_mask = np.asarray(action_loss_mask, dtype=bool).reshape(-1)
        if action_mask.size != T:
            raise ValueError(
                f"handoff/action_loss_mask length must be {T}, got {action_mask.size}"
            )
        prefix = np.concatenate(([0], np.cumsum(action_mask.astype(np.int64))))
        window_has_action_loss = (prefix[ends] - prefix[starts]) > 0
        valid &= window_has_action_loss

    return starts[valid]


def _decode_jpeg_image(encoded: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required to decode JPEG training images"
        ) from exc
    bgr = cv2.imdecode(np.asarray(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("failed to decode JPEG training image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ─── load_data ────────────────────────────────────────────────────────────────


def load_data(
    dataset_dir: str | Path,
    num_episodes: int,
    camera_names: list[str],
    episode_len: int | None,
    batch_size_train: int,
    batch_size_val: int,
    num_workers: int = 1,
    prefetch_factor: int = 1,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    *,
    split_seed: int = 0,
    train_split_ratio: float = 0.8,
    split_path: str | Path | None = None,
    reuse_split: bool = True,
    low_dim_keys: list[str] | tuple[str, ...] | None = None,
    episode_ids: list[int] | None = None,
    action_chunk_size: int | None = None,
    image_transform: str = "none",
    deadzone_intent: dict[str, Any] | None = None,
    sample_valid_mask_path: str | None = None,
    norm_stats_train_only: bool = False,
    condition_shuffle_seed_train: int | None = None,
    condition_phase_randomization_train: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, dict, bool, dict[str, Any]]:
    """
    Build train/val DataLoaders from an HDF5 dataset directory.

    Returns
    -------
    train_loader, val_loader, norm_stats, is_real, split_info
    """
    dataset_dir = Path(dataset_dir)
    print(f"\nData from: {dataset_dir}\n")
    deadzone_intent_cfg = _resolve_deadzone_intent_config(deadzone_intent)

    # discover available episode files
    discovered = [int(p.stem.split("_", 1)[1]) for p in list_episodes(dataset_dir)]
    if episode_ids is None:
        available = [i for i in discovered if i < num_episodes]
    else:
        requested_ids = [int(i) for i in episode_ids]
        discovered_set = set(discovered)
        available = [i for i in requested_ids if i in discovered_set]

    if not available:
        raise FileNotFoundError(
            f"No episodes found under {dataset_dir}. "
            "Expected files like episode_0.hdf5."
        )
    if episode_ids is None and len(available) < num_episodes:
        print(
            f"Warning: requested {num_episodes} episodes "
            f"but found {len(available)}. Using available episodes."
        )
    if episode_ids is not None and len(available) < len(episode_ids):
        missing = sorted(set(int(i) for i in episode_ids) - set(available))
        print(
            f"Warning: requested explicit episode_ids but {len(missing)} are missing: "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    # Filter to episodes where action_dim matches qpos_dim.
    # Real v1 expects one normalized command per recorded joint axis.
    import h5py

    dim_info = {}
    length_info = {}
    valid_start_count = {}
    for ep_id in available:
        p = dataset_dir / f"episode_{ep_id}.hdf5"
        with h5py.File(p, "r") as f:
            dim_info[ep_id] = (
                f["/action"].shape[1],
                f["/observations/qpos"].shape[1],
            )
            length_info[ep_id] = int(f["/action"].shape[0])
            action_loss_start_mask = _read_optional_handoff_mask(
                f,
                "handoff/action_loss_mask",
                length_info[ep_id],
                enabled=bool(deadzone_intent_cfg["require_action_loss_in_chunk"]),
            )
            valid_start_count[ep_id] = int(
                _valid_start_indices(
                    total_steps=length_info[ep_id],
                    train_exclude_mask=_combined_training_exclude_mask(
                        f,
                        length_info[ep_id],
                        sample_valid_mask_path=sample_valid_mask_path,
                    ),
                    action_chunk_size=action_chunk_size,
                    action_loss_mask=action_loss_start_mask,
                    require_action_loss_in_chunk=bool(
                        deadzone_intent_cfg["require_action_loss_in_chunk"]
                    ),
                ).size
            )
    filtered = [i for i in available if dim_info[i][0] == dim_info[i][1]]
    dropped = len(available) - len(filtered)
    if dropped:
        act_dims = set(d[0] for d in dim_info.values())
        qpos_dim = dim_info[available[0]][1]
        print(
            f"Warning: skipped {dropped} episode(s) where action_dim != qpos_dim ({qpos_dim}). "
            f"Found action dims: {act_dims}. Re-collect data with `tb-record-real` to fix."
        )
    available = filtered

    if not available:
        raise FileNotFoundError(
            f"No valid episodes found under {dataset_dir} (action_dim != qpos_dim for all). "
            "Re-collect data with `tb-record-real`."
        )

    mask_filtered = [i for i in available if valid_start_count.get(i, 0) > 0]
    dropped_masked = len(available) - len(mask_filtered)
    if dropped_masked:
        print(
            f"Warning: skipped {dropped_masked} episode(s) with no valid training "
            "starts after diagnostics/train_exclude_mask."
        )
    available = mask_filtered

    if not available:
        raise FileNotFoundError(
            f"No valid episodes found under {dataset_dir} after train_exclude_mask filtering."
        )

    max_episode_len = max(length_info[ep_id] for ep_id in available)
    target_episode_len = (
        int(episode_len) if episode_len is not None else max_episode_len
    )
    if max_episode_len > target_episode_len:
        raise ValueError(
            f"Dataset contains an episode of length {max_episode_len}, but configured "
            f"episode_len is only {target_episode_len}. Increase task.episode_len."
        )

    train_ids, val_ids, split_info = _resolve_episode_split(
        dataset_dir=dataset_dir,
        available_episode_ids=available,
        requested_num_episodes=int(num_episodes),
        split_seed=int(split_seed),
        train_split_ratio=float(train_split_ratio),
        split_path=None if split_path is None else Path(split_path),
        reuse_split=bool(reuse_split),
    )

    selected_low_dim_keys = _normalize_low_dim_keys(low_dim_keys)
    norm_stats_episode_ids = train_ids if norm_stats_train_only else available
    norm_stats = get_norm_stats(
        dataset_dir,
        num_episodes,
        episode_ids=norm_stats_episode_ids,
        low_dim_keys=selected_low_dim_keys,
        deadzone_intent=deadzone_intent,
        valid_mask_path=sample_valid_mask_path,
    )

    train_ds = EpisodicDataset(
        train_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        episode_len=target_episode_len,
        low_dim_keys=selected_low_dim_keys,
        action_chunk_size=action_chunk_size,
        image_transform=image_transform,
        deadzone_intent=deadzone_intent,
        sample_valid_mask_path=sample_valid_mask_path,
        condition_shuffle_seed=condition_shuffle_seed_train,
        condition_phase_randomization=condition_phase_randomization_train,
    )
    val_ds = EpisodicDataset(
        val_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        episode_len=target_episode_len,
        low_dim_keys=selected_low_dim_keys,
        action_chunk_size=action_chunk_size,
        image_transform=image_transform,
        deadzone_intent=deadzone_intent,
        sample_valid_mask_path=sample_valid_mask_path,
        condition_shuffle_seed=None,
        condition_phase_randomization=None,
    )

    split_info["dataset_max_episode_len"] = int(max_episode_len)
    split_info["loader_episode_len"] = int(target_episode_len)
    split_info["low_dim_keys"] = list(selected_low_dim_keys)
    split_info["low_dim_dim"] = int(norm_stats["proprio_dim"])
    split_info["action_chunk_size"] = (
        None if action_chunk_size is None else int(action_chunk_size)
    )
    split_info["image_transform"] = str(image_transform or "none")
    split_info["deadzone_intent_enabled"] = bool(deadzone_intent_cfg["enabled"])
    split_info["sample_valid_mask_path"] = str(sample_valid_mask_path or "")
    split_info["norm_stats_train_only"] = bool(norm_stats_train_only)
    split_info["condition_shuffle_train"] = dict(train_ds.condition_shuffle_manifest)
    split_info["condition_shuffle_validation"] = dict(val_ds.condition_shuffle_manifest)
    split_info["condition_phase_randomization_train"] = dict(
        train_ds.condition_phase_randomization_manifest
    )
    split_info["condition_phase_randomization_validation"] = dict(
        val_ds.condition_phase_randomization_manifest
    )
    split_info["norm_stats_episode_ids"] = [
        int(ep_id) for ep_id in norm_stats_episode_ids
    ]
    split_info["gap_mask_valid_start_count"] = {
        int(ep_id): int(valid_start_count.get(ep_id, 0)) for ep_id in available
    }
    split_info["explicit_episode_ids"] = (
        [] if episode_ids is None else [int(ep_id) for ep_id in episode_ids]
    )

    loader_kw: dict = {"pin_memory": pin_memory, "num_workers": num_workers}
    if num_workers > 0:
        loader_kw["prefetch_factor"] = prefetch_factor
        loader_kw["persistent_workers"] = bool(persistent_workers)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size_train, shuffle=True, **loader_kw
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size_val, shuffle=True, **loader_kw
    )

    return train_loader, val_loader, norm_stats, train_ds.is_real, split_info


def _resolve_episode_split(
    *,
    dataset_dir: Path,
    available_episode_ids: list[int],
    requested_num_episodes: int,
    split_seed: int,
    train_split_ratio: float,
    split_path: Path | None,
    reuse_split: bool,
) -> tuple[list[int], list[int], dict[str, Any]]:
    if split_path is not None and split_path.exists() and reuse_split:
        split_info = _load_split_file(split_path)
        _validate_saved_split(
            split_info=split_info,
            dataset_dir=dataset_dir,
            available_episode_ids=available_episode_ids,
        )
        train_ids = [int(ep_id) for ep_id in split_info["train_ids"]]
        val_ids = [int(ep_id) for ep_id in split_info["val_ids"]]
        split_info["reused_existing_split"] = True
        return train_ids, val_ids, split_info

    train_ids, val_ids = _generate_episode_split(
        available_episode_ids=available_episode_ids,
        split_seed=split_seed,
        train_split_ratio=train_split_ratio,
    )
    split_info = {
        "schema_version": 1,
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "dataset_dir": str(dataset_dir.resolve()),
        "requested_num_episodes": int(requested_num_episodes),
        "available_episode_ids": [int(ep_id) for ep_id in available_episode_ids],
        "split_seed": int(split_seed),
        "train_split_ratio": float(train_split_ratio),
        "train_ids": [int(ep_id) for ep_id in train_ids],
        "val_ids": [int(ep_id) for ep_id in val_ids],
        "reused_existing_split": False,
    }

    if split_path is not None:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w") as f:
            yaml.safe_dump(split_info, f, sort_keys=False)
        split_info["split_path"] = str(split_path)
    else:
        split_info["split_path"] = ""

    return train_ids, val_ids, split_info


def _generate_episode_split(
    *,
    available_episode_ids: list[int],
    split_seed: int,
    train_split_ratio: float,
) -> tuple[list[int], list[int]]:
    available = [int(ep_id) for ep_id in available_episode_ids]
    if not available:
        raise ValueError("Cannot generate split from an empty episode list.")

    ratio = float(np.clip(train_split_ratio, 0.0, 1.0))
    shuffled = list(np.random.default_rng(split_seed).permutation(available))

    if len(shuffled) == 1:
        # For tiny smoke/overfit runs, share the only episode across train/val.
        single = [int(shuffled[0])]
        return single, single

    split = int(round(ratio * len(shuffled)))
    split = min(max(split, 1), len(shuffled) - 1)
    return shuffled[:split], shuffled[split:]


def _load_split_file(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid split file format at {path}. Expected a mapping.")
    data["split_path"] = str(path)
    return data


def _validate_saved_split(
    *,
    split_info: dict[str, Any],
    dataset_dir: Path,
    available_episode_ids: list[int],
) -> None:
    expected_dataset_dir = str(dataset_dir.resolve())
    saved_dataset_dir = str(split_info.get("dataset_dir", ""))
    if saved_dataset_dir and saved_dataset_dir != expected_dataset_dir:
        raise ValueError(
            "Saved split file dataset_dir does not match current dataset_dir: "
            f"{saved_dataset_dir} != {expected_dataset_dir}"
        )

    available = {int(ep_id) for ep_id in available_episode_ids}
    train_ids = [int(ep_id) for ep_id in split_info.get("train_ids", [])]
    val_ids = [int(ep_id) for ep_id in split_info.get("val_ids", [])]
    if not train_ids or not val_ids:
        raise ValueError(
            "Saved split file must contain non-empty train_ids and val_ids."
        )

    split_ids = set(train_ids) | set(val_ids)
    missing = sorted(split_ids - available)
    if missing:
        raise ValueError(
            "Saved split file references episode ids not available in the current dataset: "
            + ", ".join(str(ep_id) for ep_id in missing)
        )
