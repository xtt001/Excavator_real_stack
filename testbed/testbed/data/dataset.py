"""
EpisodicDataset, get_norm_stats, load_data.

PyTorch data loading utilities for real-excavator HDF5 episodes.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from testbed.data.causal_visual_history import (
    causal_window_indices,
    resolve_temporal_input_config,
)
from testbed.data.deadzone_intent_labels import compute_deadzone_intent_labels
from testbed.data.execution_feedback import (
    ExecutionFeedbackSidecar,
    build_execution_feedback_norm_stats,
    generate_symmetric_weak_command_variants,
    load_execution_feedback_sidecar,
    resolve_execution_feedback_config,
    validate_execution_feedback_manifest,
)
from testbed.data.hdf5_io import list_episodes
from testbed.data.image_transforms import build_image_transform
from testbed.data.schema import ATTR_IS_REAL, GRP_ENCODED_IMAGES
from testbed.data.state_hold_transition import (
    anchor_transition_direction_mask,
    compute_transition_direction_mask,
    intersect_transition_starts,
    resolve_state_hold_transition_config,
    sample_state_hold_start,
)
from testbed.policies.act.action_state_effort import (
    compute_action_state_labels,
    resolve_action_state_effort_config,
)
from testbed.policies.act.effective_action import (
    compute_effective_action_labels,
    resolve_effective_action_config,
)
from testbed.policies.act.goal_effect import (
    GoalEffectConfig,
    build_goal_effect_targets,
    future_delta_scale,
    resolve_goal_effect_config,
)

SUPPORTED_LOW_DIM_KEYS = ("qpos", "qvel", "previous_final_command")


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
    previous_final_command: np.ndarray | None = None,
    low_dim_keys: list[str],
) -> np.ndarray:
    qpos_arr = np.asarray(qpos, dtype=np.float32)
    qvel_arr = np.asarray(qvel, dtype=np.float32)
    previous_arr = (
        None
        if previous_final_command is None
        else np.asarray(previous_final_command, dtype=np.float32)
    )
    sequence_mode = (
        qpos_arr.ndim > 1
        or qvel_arr.ndim > 1
        or (previous_arr is not None and previous_arr.ndim > 1)
    )
    parts: list[np.ndarray] = []
    for key in low_dim_keys:
        if key == "qpos":
            part = qpos_arr
        elif key == "qvel":
            part = qvel_arr
        elif key == "previous_final_command":
            if previous_arr is None:
                raise ValueError(
                    "low_dim_keys includes previous_final_command but no causal "
                    "execution-feedback command was provided"
                )
            part = previous_arr
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
    goal_effect: dict[str, Any] | None = None,
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
    qpos_sequences: list[np.ndarray] = []
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
            action = f["/action"][()]
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
                action = np.asarray(action, dtype=np.float32)[mask]
        proprio = _assemble_low_dim_observation(
            qpos=qpos,
            qvel=qvel,
            low_dim_keys=selected_low_dim_keys,
        )
        all_proprio_data.append(torch.from_numpy(proprio))
        all_qpos_data.append(torch.from_numpy(qpos))
        qpos_sequences.append(np.asarray(qpos, dtype=np.float32))
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
    goal_effect_cfg = resolve_goal_effect_config(goal_effect)
    if goal_effect_cfg.enabled:
        stats["goal_effect_delta_scale"] = future_delta_scale(
            qpos_sequences,
            goal_effect_cfg.horizons,
        )
        stats["goal_effect_horizons"] = np.asarray(
            goal_effect_cfg.horizons, dtype=np.int64
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
        state_hold_transition: dict[str, Any] | None = None,
        execution_feedback: dict[str, Any] | None = None,
        goal_effect: dict[str, Any] | None = None,
        action_state_effort: dict[str, Any] | None = None,
        effective_action: dict[str, Any] | None = None,
        temporal_input: dict[str, Any] | None = None,
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
        self.state_hold_transition = resolve_state_hold_transition_config(
            state_hold_transition
        )
        self.execution_feedback = resolve_execution_feedback_config(execution_feedback)
        self.goal_effect: GoalEffectConfig = resolve_goal_effect_config(
            goal_effect,
            target_scale=norm_stats.get("goal_effect_delta_scale"),
        )
        self.action_state_effort = resolve_action_state_effort_config(
            action_state_effort
        )
        self.effective_action = resolve_effective_action_config(effective_action)
        self.temporal_input = resolve_temporal_input_config(temporal_input)
        expected_feedback_keys = [
            "qpos",
            "qvel",
            "previous_final_command",
        ]
        if self.execution_feedback["enabled"]:
            if self.low_dim_keys != expected_feedback_keys:
                raise ValueError(
                    "execution_feedback requires low_dim_keys in exact order "
                    f"{expected_feedback_keys}, got {self.low_dim_keys}"
                )
            manifest = validate_execution_feedback_manifest(
                self.execution_feedback["manifest_path"],
                verify_hashes=False,
                expected_dataset_dir=self.dataset_dir,
            )
            self._execution_feedback_records = {
                int(record["episode_id"]): record for record in manifest["episodes"]
            }
        else:
            if "previous_final_command" in self.low_dim_keys:
                raise ValueError(
                    "previous_final_command low-dimensional input requires "
                    "execution_feedback.enabled=true"
                )
            self._execution_feedback_records = {}
        self._execution_feedback_cache: dict[int, ExecutionFeedbackSidecar] = {}
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
            train_exclude_mask = _read_train_exclude_mask(f, T)
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
            transition_mask = None
            transition_starts = np.zeros(0, dtype=np.int64)
            if self.state_hold_transition["enabled"]:
                if is_real and not action_prealigned:
                    raise ValueError(
                        "state_hold_transition requires action_prealigned=true "
                        f"for real episode {ep_id}"
                    )
                transition_mask = _state_hold_transition_mask(
                    f,
                    total_steps=T,
                    config=self.state_hold_transition,
                )
                transition_starts = intersect_transition_starts(
                    transition_mask,
                    valid_starts,
                    total_steps=T,
                    hold_horizon_steps=int(
                        self.state_hold_transition["hold_horizon_steps"]
                    ),
                )
                t0 = sample_state_hold_start(
                    valid_starts=valid_starts,
                    transition_starts=transition_starts,
                    probability=float(self.state_hold_transition["probability"]),
                )
            else:
                t0 = int(np.random.choice(valid_starts))

            # ── observation at t0 ─────────────────────────────────────────
            qpos = f["/observations/qpos"][t0]
            qvel = f["/observations/qvel"][t0]
            previous_final_command = None
            if self.execution_feedback["enabled"]:
                feedback_sidecar = self._feedback_sidecar(ep_id, expected_length=T)
                previous_final_command = feedback_sidecar.previous_final_command[t0]
            proprio = _assemble_low_dim_observation(
                qpos=qpos,
                qvel=qvel,
                previous_final_command=previous_final_command,
                low_dim_keys=self.low_dim_keys,
            )
            if self.temporal_input["enabled"]:
                history_indices = causal_window_indices(
                    total_steps=T,
                    target_step=t0,
                    history_length=int(self.temporal_input["history_steps"]),
                )
            else:
                history_indices = np.asarray([t0], dtype=np.int64)
            image_dict = {cam: [] for cam in self.camera_names}
            for history_step in history_indices:
                for cam in self.camera_names:
                    image = _read_camera_image(f, cam, int(history_step))
                    if self.image_transform is not None:
                        image = self.image_transform(image)
                    image_dict[cam].append(image)
            if not self.temporal_input["enabled"]:
                image_dict = {
                    cam: images[0] for cam, images in image_dict.items()
                }
            else:
                image_dict = {
                    cam: np.stack(images, axis=0)
                    for cam, images in image_dict.items()
                }

            # ── action from t0 onward ────────────────────────────────────
            start = t0 if (not is_real or action_prealigned) else max(0, t0 - 1)
            raw_action = np.asarray(f["/action"][start:], dtype=np.float32)
            action = raw_action
            action_len = T - start
            goal_effect_targets = None
            if self.goal_effect.enabled:
                goal_effect_targets = build_goal_effect_targets(
                    qpos=np.asarray(f["/observations/qpos"][()], dtype=np.float32),
                    qvel=np.asarray(f["/observations/qvel"][()], dtype=np.float32),
                    action=np.asarray(f["/action"][()], dtype=np.float32),
                    timestep=t0,
                    config=self.goal_effect,
                )
            action_state_labels = None
            full_action = None
            if (
                self.action_state_effort["enabled"]
                or self.deadzone_intent["enabled"]
                or self.effective_action["enabled"]
            ):
                full_action = np.asarray(f["/action"][()], dtype=np.float32)
            if self.action_state_effort["enabled"]:
                action_state_labels = compute_action_state_labels(
                    actions=full_action,
                    thresholds=self.action_state_effort["thresholds"],
                    safe_margin=float(self.action_state_effort["safe_margin"]),
                    persistence_steps=int(
                        self.action_state_effort["persistence_steps"]
                    ),
                )
            deadzone_labels = None
            if self.deadzone_intent["enabled"]:
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

            effective_action_labels = None
            if self.effective_action["enabled"]:
                if full_action is None:
                    raise RuntimeError(
                        "effective_action requires the full action sequence"
                    )
                effective_action_labels = compute_effective_action_labels(
                    actions=full_action,
                    thresholds=self.effective_action["thresholds"],
                    transition_window_steps=int(
                        self.effective_action["transition_window_steps"]
                    ),
                    persistence_steps=int(self.effective_action["persistence_steps"]),
                    neutral_weight=float(self.effective_action["neutral_weight"]),
                    active_weight=float(self.effective_action["active_weight"]),
                    persistent_weight=float(
                        self.effective_action["persistent_weight"]
                    ),
                    transition_weight=float(self.effective_action["transition_weight"]),
                    active_margin=float(self.effective_action["active_margin"]),
                )
                action = effective_action_labels.action[start:]

            state_hold_transition_mask = None
            if transition_mask is not None:
                state_hold_transition_mask = np.zeros(
                    (original_action_shape[1], 2), dtype=bool
                )
                if np.any(transition_starts == t0):
                    state_hold_transition_mask = anchor_transition_direction_mask(
                        transition_mask,
                        t0,
                    )

            counterfactual_proprio = None
            counterfactual_axis_index = -1
            counterfactual_magnitude_fraction = 0.0
            counterfactual_cfg = self.execution_feedback["counterfactual"]
            if (
                counterfactual_cfg["enabled"]
                and state_hold_transition_mask is not None
                and np.any(state_hold_transition_mask)
            ):
                variants = generate_symmetric_weak_command_variants(
                    episode_id=ep_id,
                    timestep=t0,
                    seed=int(counterfactual_cfg["seed"]),
                    thresholds=counterfactual_cfg["thresholds"],
                )
                counterfactual_proprio = np.stack(
                    [
                        _assemble_low_dim_observation(
                            qpos=qpos,
                            qvel=variants.qvel[variant_index],
                            previous_final_command=(
                                variants.previous_final_command[variant_index]
                            ),
                            low_dim_keys=self.low_dim_keys,
                        )
                        for variant_index in range(
                            variants.previous_final_command.shape[0]
                        )
                    ],
                    axis=0,
                ).astype(np.float32)
                counterfactual_axis_index = int(variants.axis_index)
                counterfactual_magnitude_fraction = float(variants.magnitude_fraction)

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
        padded_raw_action = None
        if effective_action_labels is not None:
            padded_raw_action = np.zeros(
                (target_len, original_action_shape[1]), dtype=np.float32
            )
            padded_raw_action[:action_len] = raw_action
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
        padded_action_state = None
        padded_action_state_valid = None
        padded_action_state_persistent = None
        if action_state_labels is not None:
            padded_action_state = np.zeros(
                (target_len, original_action_shape[1]), dtype=np.int64
            )
            padded_action_state_valid = np.zeros(
                (target_len, original_action_shape[1]), dtype=bool
            )
            padded_action_state_persistent = np.zeros(
                (target_len, original_action_shape[1], 2), dtype=bool
            )
            padded_action_state[:action_len] = action_state_labels.state[start:]
            padded_action_state_valid[:action_len] = action_state_labels.valid[start:]
            padded_action_state_persistent[:action_len] = (
                action_state_labels.persistent_effective[start:]
            )
        padded_effective_phase = None
        padded_effective_valid = None
        padded_effective_weight = None
        padded_effective_transition = None
        padded_effective_persistent = None
        if effective_action_labels is not None:
            padded_effective_phase = np.zeros(
                (target_len, original_action_shape[1]), dtype=np.int64
            )
            padded_effective_valid = np.zeros(
                (target_len, original_action_shape[1]), dtype=bool
            )
            padded_effective_weight = np.zeros(
                (target_len, original_action_shape[1]), dtype=np.float32
            )
            padded_effective_transition = np.zeros(
                (target_len, original_action_shape[1]), dtype=bool
            )
            padded_effective_persistent = np.zeros(
                (target_len, original_action_shape[1]), dtype=bool
            )
            padded_effective_phase[:action_len] = effective_action_labels.phase[start:]
            padded_effective_valid[:action_len] = effective_action_labels.valid[start:]
            padded_effective_weight[:action_len] = effective_action_labels.loss_weight[
                start:
            ]
            padded_effective_transition[:action_len] = (
                effective_action_labels.transition[start:]
            )
            padded_effective_persistent[:action_len] = (
                effective_action_labels.persistent[start:]
            )

        # ── assemble camera tensor ─────────────────────────────────────────
        all_cam_images = np.stack(
            [image_dict[c] for c in self.camera_names], axis=0
        )
        if self.temporal_input["enabled"]:
            # Dataset samples are (history_steps, n_cams, H, W, 3), oldest
            # to newest.  The DataLoader adds the batch dimension later.
            all_cam_images = np.transpose(all_cam_images, (1, 0, 2, 3, 4))
        else:
            all_cam_images = all_cam_images  # (n_cams, H, W, 3)

        # ── convert to tensors ────────────────────────────────────────────
        image_data = torch.from_numpy(all_cam_images)
        proprio_data = torch.from_numpy(proprio).float()
        action_data = torch.from_numpy(padded_action).float()
        raw_action_data = None
        if padded_raw_action is not None:
            raw_action_data = torch.from_numpy(padded_raw_action).float()
        is_pad_t = torch.from_numpy(is_pad)

        # channel-last → channel-first + normalize to [0, 1]
        if self.temporal_input["enabled"]:
            image_data = torch.einsum(
                "t k h w c -> t k c h w", image_data
            ).float() / 255.0
        else:
            image_data = torch.einsum("k h w c -> k c h w", image_data).float() / 255.0

        # normalise proprio and actions
        action_data = (
            action_data - torch.from_numpy(self.norm_stats["action_mean"])
        ) / torch.from_numpy(self.norm_stats["action_std"])
        if raw_action_data is not None:
            raw_action_data = (
                raw_action_data - torch.from_numpy(self.norm_stats["action_mean"])
            ) / torch.from_numpy(self.norm_stats["action_std"])
        proprio_data = (
            proprio_data - torch.from_numpy(self.norm_stats["proprio_mean"])
        ) / torch.from_numpy(self.norm_stats["proprio_std"])

        counterfactual_proprio_data = torch.zeros(
            (2, int(proprio_data.shape[0])), dtype=torch.float32
        )
        counterfactual_active = counterfactual_proprio is not None
        if counterfactual_proprio is not None:
            counterfactual_proprio_data = (
                torch.from_numpy(counterfactual_proprio).float()
                - torch.from_numpy(self.norm_stats["proprio_mean"])
            ) / torch.from_numpy(self.norm_stats["proprio_std"])

        if (
            deadzone_labels is None
            and state_hold_transition_mask is None
            and not self.execution_feedback["enabled"]
            and goal_effect_targets is None
            and action_state_labels is None
            and effective_action_labels is None
        ):
            return image_data, proprio_data, action_data, is_pad_t

        payload = {
            "image": image_data,
            "proprio": proprio_data,
            "action": action_data,
            "is_pad": is_pad_t,
        }
        if raw_action_data is not None:
            payload["raw_action"] = raw_action_data
        if deadzone_labels is not None:
            payload.update(
                {
                    "deadzone_move_mask": torch.from_numpy(padded_move_mask),
                    "deadzone_stop_mask": torch.from_numpy(padded_stop_mask),
                    "deadzone_wrong_mask": torch.from_numpy(padded_wrong_mask),
                    "action_loss_mask": torch.from_numpy(padded_action_loss_mask),
                }
            )
        if action_state_labels is not None:
            payload.update(
                {
                    "action_state_labels": torch.from_numpy(padded_action_state),
                    "action_state_valid": torch.from_numpy(
                        padded_action_state_valid
                    ),
                    "action_state_persistent_effective": torch.from_numpy(
                        padded_action_state_persistent
                    ),
                }
            )
        if effective_action_labels is not None:
            payload.update(
                {
                    "effective_action_phase": torch.from_numpy(
                        padded_effective_phase
                    ),
                    "effective_action_valid": torch.from_numpy(
                        padded_effective_valid
                    ),
                    "effective_action_loss_weight": torch.from_numpy(
                        padded_effective_weight
                    ),
                    "effective_action_transition": torch.from_numpy(
                        padded_effective_transition
                    ),
                    "effective_action_persistent": torch.from_numpy(
                        padded_effective_persistent
                    ),
                }
            )
        if state_hold_transition_mask is not None:
            payload["state_hold_transition_mask"] = torch.from_numpy(
                state_hold_transition_mask
            )
        if self.execution_feedback["enabled"]:
            counterfactual_cfg = self.execution_feedback["counterfactual"]
            payload.update(
                {
                    "execution_feedback_counterfactual_proprio": (
                        counterfactual_proprio_data
                    ),
                    "execution_feedback_counterfactual_mask": torch.tensor(
                        counterfactual_active, dtype=torch.bool
                    ),
                    "execution_feedback_counterfactual_loss_weight": torch.tensor(
                        float(counterfactual_cfg["loss_weight"]),
                        dtype=torch.float32,
                    ),
                    "execution_feedback_counterfactual_axis_index": torch.tensor(
                        counterfactual_axis_index, dtype=torch.int64
                    ),
                    "execution_feedback_counterfactual_magnitude_fraction": (
                        torch.tensor(
                            counterfactual_magnitude_fraction,
                            dtype=torch.float32,
                        )
                    ),
                }
            )
        if goal_effect_targets is not None:
            payload.update(
                {
                    key: torch.from_numpy(value)
                    for key, value in goal_effect_targets.items()
                }
            )
        return payload

    def _feedback_sidecar(
        self,
        episode_id: int,
        *,
        expected_length: int,
    ) -> ExecutionFeedbackSidecar:
        cached = self._execution_feedback_cache.get(int(episode_id))
        if cached is not None:
            if len(cached) != int(expected_length):
                raise ValueError(
                    f"execution-feedback cached length mismatch for episode {episode_id}"
                )
            return cached
        record = self._execution_feedback_records.get(int(episode_id))
        if record is None:
            raise ValueError(f"execution-feedback manifest has no episode {episode_id}")
        sidecar = load_execution_feedback_sidecar(
            record["sidecar_path"],
            expected_episode_id=episode_id,
            expected_length=expected_length,
        )
        self._execution_feedback_cache[int(episode_id)] = sidecar
        return sidecar


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


def _state_hold_transition_mask(
    h5_file: Any,
    *,
    total_steps: int,
    config: dict[str, Any],
) -> np.ndarray:
    """Build transition evidence while respecting any handoff stop ownership."""

    return compute_transition_direction_mask(
        actions=np.asarray(h5_file["/action"][()], dtype=np.float32),
        thresholds=config["thresholds"],
        action_loss_mask=_read_optional_handoff_mask(
            h5_file,
            "handoff/action_loss_mask",
            total_steps,
            enabled=True,
        ),
        tail_idle_mask=_read_optional_handoff_mask(
            h5_file,
            "handoff/tail_idle_mask",
            total_steps,
            enabled=True,
        ),
        owner_automation=_read_optional_handoff_mask(
            h5_file,
            "handoff/owner_automation",
            total_steps,
            enabled=True,
        ),
    )


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
    prefetch_factor: int | None = 1,
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
    state_hold_transition: dict[str, Any] | None = None,
    execution_feedback: dict[str, Any] | None = None,
    goal_effect: dict[str, Any] | None = None,
    action_state_effort: dict[str, Any] | None = None,
    effective_action: dict[str, Any] | None = None,
    temporal_input: dict[str, Any] | None = None,
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
    state_hold_transition_cfg = resolve_state_hold_transition_config(
        state_hold_transition
    )
    execution_feedback_cfg = resolve_execution_feedback_config(execution_feedback)
    goal_effect_cfg = resolve_goal_effect_config(goal_effect)
    action_state_effort_cfg = resolve_action_state_effort_config(action_state_effort)
    effective_action_cfg = resolve_effective_action_config(effective_action)
    temporal_input_cfg = resolve_temporal_input_config(temporal_input)

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
    state_hold_transition_start_count = {}
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
            valid_starts = _valid_start_indices(
                total_steps=length_info[ep_id],
                train_exclude_mask=_read_train_exclude_mask(f, length_info[ep_id]),
                action_chunk_size=action_chunk_size,
                action_loss_mask=action_loss_start_mask,
                require_action_loss_in_chunk=bool(
                    deadzone_intent_cfg["require_action_loss_in_chunk"]
                ),
            )
            valid_start_count[ep_id] = int(valid_starts.size)
            state_hold_transition_start_count[ep_id] = 0
            if (
                state_hold_transition_cfg["enabled"]
                and dim_info[ep_id][0] == dim_info[ep_id][1] == 4
            ):
                metadata = dict(f["metadata"].attrs) if "metadata" in f else {}
                action_prealigned = _bool_attr(metadata.get("action_prealigned", False))
                if bool(f.attrs.get(ATTR_IS_REAL, True)) and not action_prealigned:
                    raise ValueError(
                        "state_hold_transition requires action_prealigned=true "
                        f"for real episode {ep_id}"
                    )
                transition_mask = _state_hold_transition_mask(
                    f,
                    total_steps=length_info[ep_id],
                    config=state_hold_transition_cfg,
                )
                state_hold_transition_start_count[ep_id] = int(
                    intersect_transition_starts(
                        transition_mask,
                        valid_starts,
                        total_steps=length_info[ep_id],
                        hold_horizon_steps=int(
                            state_hold_transition_cfg["hold_horizon_steps"]
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
    if execution_feedback_cfg["enabled"]:
        norm_stats = build_execution_feedback_norm_stats(
            dataset_dir=dataset_dir,
            train_ids=train_ids,
            config=execution_feedback_cfg,
            verify_manifest_hashes=True,
        )
    else:
        norm_stats = get_norm_stats(
            dataset_dir,
            num_episodes,
            # Normalisation and auxiliary target scales are calibration.  Use
            # train IDs only so the formal validation fold cannot influence
            # model selection.
            episode_ids=train_ids,
            low_dim_keys=selected_low_dim_keys,
            deadzone_intent=deadzone_intent,
            goal_effect=goal_effect,
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
        state_hold_transition=state_hold_transition,
        execution_feedback=execution_feedback_cfg,
        goal_effect=goal_effect,
        action_state_effort=action_state_effort,
        effective_action=effective_action,
        temporal_input=temporal_input_cfg,
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
        state_hold_transition=state_hold_transition,
        execution_feedback=execution_feedback_cfg,
        goal_effect=goal_effect,
        action_state_effort=action_state_effort,
        effective_action=effective_action,
        temporal_input=temporal_input_cfg,
    )

    split_info["dataset_max_episode_len"] = int(max_episode_len)
    split_info["loader_episode_len"] = int(target_episode_len)
    split_info["low_dim_keys"] = list(selected_low_dim_keys)
    split_info["low_dim_dim"] = int(norm_stats["proprio_dim"])
    split_info["action_chunk_size"] = (
        None if action_chunk_size is None else int(action_chunk_size)
    )
    split_info["image_transform"] = str(image_transform or "none")
    split_info["temporal_input"] = {
        "enabled": bool(temporal_input_cfg["enabled"]),
        "history_steps": int(temporal_input_cfg["history_steps"]),
    }
    split_info["deadzone_intent_enabled"] = bool(deadzone_intent_cfg["enabled"])
    split_info["action_state_effort"] = {
        "enabled": bool(action_state_effort_cfg["enabled"]),
        "state_order": ["idle", "pos_near", "pos_safe", "neg_near", "neg_safe"],
        "safe_margin": float(action_state_effort_cfg["safe_margin"]),
        "required_margin": float(action_state_effort_cfg["required_margin"]),
        "persistence_steps": int(action_state_effort_cfg["persistence_steps"]),
    }
    split_info["effective_action"] = {
        "enabled": bool(effective_action_cfg["enabled"]),
        "thresholds": effective_action_cfg["thresholds"],
        "transition_window_steps": int(
            effective_action_cfg["transition_window_steps"]
        ),
        "persistence_steps": int(effective_action_cfg["persistence_steps"]),
        "neutral_weight": float(effective_action_cfg["neutral_weight"]),
        "active_weight": float(effective_action_cfg["active_weight"]),
        "persistent_weight": float(effective_action_cfg["persistent_weight"]),
        "transition_weight": float(effective_action_cfg["transition_weight"]),
        "active_margin": float(effective_action_cfg["active_margin"]),
    }
    split_info["state_hold_transition"] = {
        "enabled": bool(state_hold_transition_cfg["enabled"]),
        "probability": float(state_hold_transition_cfg["probability"]),
        "hold_horizon_steps": int(state_hold_transition_cfg["hold_horizon_steps"]),
        "eligible_start_count": {
            int(ep_id): int(state_hold_transition_start_count.get(ep_id, 0))
            for ep_id in available
        },
    }
    split_info["execution_feedback"] = {
        "enabled": bool(execution_feedback_cfg["enabled"]),
        "manifest_path": execution_feedback_cfg["manifest_path"],
        "base_norm_stats_path": execution_feedback_cfg["base_norm_stats_path"],
        "counterfactual": {
            "enabled": bool(execution_feedback_cfg["counterfactual"]["enabled"]),
            "seed": int(execution_feedback_cfg["counterfactual"]["seed"]),
            "loss_weight": float(
                execution_feedback_cfg["counterfactual"]["loss_weight"]
            ),
        },
        "norm_provenance": norm_stats.get("execution_feedback_norm_provenance"),
    }
    split_info["goal_effect"] = {
        "enabled": bool(goal_effect_cfg.enabled),
        "horizons": list(goal_effect_cfg.horizons),
        "unsupported_axes": list(goal_effect_cfg.unsupported_axes),
        "delta_scale": (
            np.asarray(norm_stats.get("goal_effect_delta_scale"), dtype=np.float32).tolist()
            if "goal_effect_delta_scale" in norm_stats
            else None
        ),
    }
    split_info["gap_mask_valid_start_count"] = {
        int(ep_id): int(valid_start_count.get(ep_id, 0)) for ep_id in available
    }
    split_info["explicit_episode_ids"] = (
        [] if episode_ids is None else [int(ep_id) for ep_id in episode_ids]
    )

    loader_kw: dict = {"pin_memory": pin_memory, "num_workers": num_workers}
    if num_workers > 0:
        if prefetch_factor is not None:
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
