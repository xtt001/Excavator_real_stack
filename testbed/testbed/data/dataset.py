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

from testbed.data.action_primitive_islands import (
    ACTION_PRIMITIVE_KEY,
    PRIMITIVE_NAMES,
    derive_action_primitive_islands,
    primitive_one_hot,
    resolve_action_primitive_config,
)
from testbed.data.deadzone_intent_labels import compute_deadzone_intent_labels
from testbed.data.factual_semantic_sampling import (
    MANIFEST_SCHEMA as FACTUAL_SEMANTIC_MANIFEST_SCHEMA,
)
from testbed.data.factual_semantic_sampling import (
    TIER_NAMES as FACTUAL_SEMANTIC_TIER_NAMES,
)
from testbed.data.factual_semantic_sampling import (
    resolve_factual_semantic_sampling_config,
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
from testbed.data.task_state_v2 import (
    TASK_STATE_V2_KEY,
    TASK_STATE_V2_SCHEMA,
    TASK_STATE_V2_TIERS,
    build_task_state_sequence,
    load_task_state_v2_manifest,
    resolve_task_state_v2_config,
    task_state_candidate_starts,
    task_state_chunk_valid_mask,
)
from testbed.data.task_state_v2 import (
    task_state_manifest_by_episode as task_state_manifest_by_episode_fn,
)
from testbed.data.work_return_context import (
    ROUTE_NAMES,
    WORK_CONTEXT_KEY,
    derive_work_return_context,
    resolve_work_return_context_config,
    work_context_vector,
)
from testbed.policies.act.condition_adherence import (
    resolve_condition_adherence_config,
)
from testbed.policies.act.cycle_phase import (
    cycle_phase_candidate_indices,
    resolve_cycle_phase_loss_config,
)
from testbed.policies.act.excursion_observed import (
    excursion_observed_candidate_indices,
    resolve_excursion_observed_loss_config,
)
from testbed.policies.act.goal_effect import (
    build_goal_effect_targets,
    future_delta_scale,
    resolve_goal_effect_config,
)
from testbed.policies.act.qvel_authority import (
    qvel_authority_candidate_starts,
    resolve_qvel_authority_config,
    stable_tool_direction_mask,
)
from testbed.policies.act.qvel_state_hold import (
    resolve_qvel_zero_state_hold_config,
)
from testbed.policies.act.return_commit import (
    resolve_return_commit_loss_config,
    return_commit_candidate_indices,
)
from testbed.policies.act.target_release import (
    resolve_target_release_config,
    target_release_candidate_indices,
)
from testbed.tasks.real_transition_excursion import EXCURSION_OBSERVED_KEY
from testbed.tasks.real_transition_phase import CYCLE_PHASE_KEY
from testbed.tasks.real_transition_return_commit import RETURN_COMMIT_KEY

REAL_TRANSITION_CONDITION_KEY = "real_transition_condition_v1"
SUPPORTED_LOW_DIM_KEYS = (
    "qpos",
    "qvel",
    REAL_TRANSITION_CONDITION_KEY,
    EXCURSION_OBSERVED_KEY,
    CYCLE_PHASE_KEY,
    RETURN_COMMIT_KEY,
    ACTION_PRIMITIVE_KEY,
    WORK_CONTEXT_KEY,
    TASK_STATE_V2_KEY,
)


def _normalize_low_dim_keys(low_dim_keys: list[str] | tuple[str, ...] | None) -> list[str]:
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
    real_transition_condition_v1: np.ndarray | None = None,
    real_transition_excursion_observed_v1: np.ndarray | None = None,
    real_transition_cycle_phase_v1: np.ndarray | None = None,
    real_transition_return_commit_v1: np.ndarray | None = None,
    real_transition_action_primitive_v1: np.ndarray | None = None,
    real_transition_work_context_v1: np.ndarray | None = None,
    real_transition_task_state_v2: np.ndarray | None = None,
    low_dim_keys: list[str],
) -> np.ndarray:
    qpos_arr = np.asarray(qpos, dtype=np.float32)
    qvel_arr = np.asarray(qvel, dtype=np.float32)
    condition_arr = (
        None
        if real_transition_condition_v1 is None
        else np.asarray(real_transition_condition_v1, dtype=np.float32)
    )
    phase_arr = (
        None
        if real_transition_cycle_phase_v1 is None
        else np.asarray(real_transition_cycle_phase_v1, dtype=np.float32)
    )
    excursion_arr = (
        None
        if real_transition_excursion_observed_v1 is None
        else np.asarray(real_transition_excursion_observed_v1, dtype=np.float32)
    )
    return_commit_arr = (
        None
        if real_transition_return_commit_v1 is None
        else np.asarray(real_transition_return_commit_v1, dtype=np.float32)
    )
    action_primitive_arr = (
        None
        if real_transition_action_primitive_v1 is None
        else np.asarray(real_transition_action_primitive_v1, dtype=np.float32)
    )
    work_context_arr = (
        None
        if real_transition_work_context_v1 is None
        else np.asarray(real_transition_work_context_v1, dtype=np.float32)
    )
    task_state_v2_arr = (
        None
        if real_transition_task_state_v2 is None
        else np.asarray(real_transition_task_state_v2, dtype=np.float32)
    )
    sequence_mode = (
        qpos_arr.ndim > 1
        or qvel_arr.ndim > 1
        or (condition_arr is not None and condition_arr.ndim > 1)
        or (excursion_arr is not None and excursion_arr.ndim > 1)
        or (phase_arr is not None and phase_arr.ndim > 1)
        or (return_commit_arr is not None and return_commit_arr.ndim > 1)
        or (action_primitive_arr is not None and action_primitive_arr.ndim > 1)
        or (work_context_arr is not None and work_context_arr.ndim > 1)
        or (task_state_v2_arr is not None and task_state_v2_arr.ndim > 1)
    )
    parts: list[np.ndarray] = []
    for key in low_dim_keys:
        if key == "qpos":
            part = qpos_arr
        elif key == "qvel":
            part = qvel_arr
        elif key == REAL_TRANSITION_CONDITION_KEY:
            if condition_arr is None:
                raise KeyError(
                    f"Requested low-dimensional input {REAL_TRANSITION_CONDITION_KEY!r}, "
                    f"but HDF5 dataset conditions/{REAL_TRANSITION_CONDITION_KEY} is missing."
                )
            part = condition_arr
        elif key == EXCURSION_OBSERVED_KEY:
            if excursion_arr is None:
                raise KeyError(
                    f"Requested low-dimensional input {EXCURSION_OBSERVED_KEY!r}, "
                    f"but HDF5 dataset conditions/{EXCURSION_OBSERVED_KEY} is missing."
                )
            part = excursion_arr
        elif key == CYCLE_PHASE_KEY:
            if phase_arr is None:
                raise KeyError(
                    f"Requested low-dimensional input {CYCLE_PHASE_KEY!r}, "
                    f"but HDF5 dataset conditions/{CYCLE_PHASE_KEY} is missing."
                )
            part = phase_arr
        elif key == RETURN_COMMIT_KEY:
            if return_commit_arr is None:
                raise KeyError(
                    f"Requested low-dimensional input {RETURN_COMMIT_KEY!r}, "
                    f"but HDF5 dataset conditions/{RETURN_COMMIT_KEY} is missing."
                )
            part = return_commit_arr
        elif key == ACTION_PRIMITIVE_KEY:
            if action_primitive_arr is None:
                raise KeyError(
                    f"Requested low-dimensional input {ACTION_PRIMITIVE_KEY!r}, "
                    "but no oracle action primitive was supplied."
                )
            part = action_primitive_arr
        elif key == WORK_CONTEXT_KEY:
            if work_context_arr is None:
                raise KeyError(
                    f"Requested low-dimensional input {WORK_CONTEXT_KEY!r}, "
                    "but no WORK/RETURN task context was supplied."
                )
            part = work_context_arr
        elif key == TASK_STATE_V2_KEY:
            if task_state_v2_arr is None:
                raise KeyError(
                    f"Requested low-dimensional input {TASK_STATE_V2_KEY!r}, "
                    "but no task-state sidecar value was supplied."
                )
            part = task_state_v2_arr
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
    action_primitive_islands: dict[str, Any] | None = None,
    work_return_context: dict[str, Any] | None = None,
    task_state_v2: dict[str, Any] | None = None,
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
    action_primitive_cfg = resolve_action_primitive_config(
        action_primitive_islands
    )
    work_return_cfg = resolve_work_return_context_config(work_return_context)
    task_state_v2_cfg = resolve_task_state_v2_config(task_state_v2)
    if (
        ACTION_PRIMITIVE_KEY in selected_low_dim_keys
        and not action_primitive_cfg["enabled"]
    ):
        raise ValueError(
            f"low_dim_keys contains {ACTION_PRIMITIVE_KEY!r} but "
            "action_primitive_islands is disabled"
        )
    if WORK_CONTEXT_KEY in selected_low_dim_keys and not work_return_cfg["enabled"]:
        raise ValueError(
            f"low_dim_keys contains {WORK_CONTEXT_KEY!r} but "
            "work_return_context is disabled"
        )
    if TASK_STATE_V2_KEY in selected_low_dim_keys and not task_state_v2_cfg["enabled"]:
        raise ValueError(
            f"low_dim_keys contains {TASK_STATE_V2_KEY!r} but "
            "task_state_v2 is disabled"
        )
    task_state_manifest_by_episode: dict[int, dict[str, Any]] = {}
    if task_state_v2_cfg["enabled"]:
        task_state_manifest = load_task_state_v2_manifest(
            str(task_state_v2_cfg["manifest_path"])
        )
        manifest_episodes = (
            Path(str(task_state_manifest.get("dataset_root", ""))) / "episodes"
        )
        if manifest_episodes.resolve() != dataset_dir.resolve():
            raise ValueError("task_state_v2 manifest dataset root mismatch")
        task_state_manifest_by_episode = task_state_manifest_by_episode_fn(
            task_state_manifest
        )
    all_proprio_data: list[torch.Tensor] = []
    all_qpos_data:    list[torch.Tensor] = []
    all_action_data:  list[torch.Tensor] = []
    qpos_sequences: list[np.ndarray] = []
    example_qpos = None
    example_proprio = None
    low_dim_slices: dict[str, tuple[int, int]] | None = None

    ids = episode_ids if episode_ids is not None else list(range(num_episodes))
    for ep_idx in ids:
        p = dataset_dir / f"episode_{ep_idx}.hdf5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            qpos   = f["/observations/qpos"][()]
            qvel   = f["/observations/qvel"][()]
            condition = _read_real_transition_condition(
                f,
                enabled=REAL_TRANSITION_CONDITION_KEY in selected_low_dim_keys,
            )
            excursion_observed = _read_real_transition_excursion_observed(
                f,
                enabled=EXCURSION_OBSERVED_KEY in selected_low_dim_keys,
            )
            cycle_phase = _read_real_transition_cycle_phase(
                f,
                enabled=CYCLE_PHASE_KEY in selected_low_dim_keys,
            )
            return_commit = _read_real_transition_return_commit(
                f,
                enabled=RETURN_COMMIT_KEY in selected_low_dim_keys,
            )
            action = f["/action"][()]
            action_primitive = (
                np.zeros(
                    (len(action), len(PRIMITIVE_NAMES)), dtype=np.float32
                )
                if ACTION_PRIMITIVE_KEY in selected_low_dim_keys
                else None
            )
            work_context = (
                np.zeros((len(action), 6), dtype=np.float32)
                if WORK_CONTEXT_KEY in selected_low_dim_keys
                else None
            )
            task_state_v2_sequence = None
            if TASK_STATE_V2_KEY in selected_low_dim_keys:
                if ep_idx not in task_state_manifest_by_episode:
                    raise ValueError(
                        f"task_state_v2 manifest is missing episode {ep_idx}"
                    )
                task_row = task_state_manifest_by_episode[ep_idx]
                task_state_v2_sequence = build_task_state_sequence(
                    total_steps=len(action),
                    current_side=str(task_row["current_side"]),
                    dig_target=str(task_row["dig_target"]),
                    next_target=str(task_row["next_target"]),
                    work_complete_row=int(task_row["work_complete_row"]),
                    return_commit_row=int(task_row["return_commit_row"]),
                )
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
            real_transition_condition_v1=condition,
            real_transition_excursion_observed_v1=excursion_observed,
            real_transition_cycle_phase_v1=cycle_phase,
            real_transition_return_commit_v1=return_commit,
            real_transition_action_primitive_v1=action_primitive,
            real_transition_work_context_v1=work_context,
            real_transition_task_state_v2=task_state_v2_sequence,
            low_dim_keys=selected_low_dim_keys,
        )
        episode_slices = _low_dim_slices(
            qpos=qpos,
            qvel=qvel,
            real_transition_condition_v1=condition,
            real_transition_excursion_observed_v1=excursion_observed,
            real_transition_cycle_phase_v1=cycle_phase,
            real_transition_return_commit_v1=return_commit,
            real_transition_action_primitive_v1=action_primitive,
            real_transition_work_context_v1=work_context,
            real_transition_task_state_v2=task_state_v2_sequence,
            low_dim_keys=selected_low_dim_keys,
        )
        if low_dim_slices is None:
            low_dim_slices = episode_slices
        elif low_dim_slices != episode_slices:
            raise ValueError(
                "Low-dimensional input dimensions differ across training episodes."
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
    qpos_tensor    = torch.cat(all_qpos_data, dim=0)     # (sum_T, Nq)
    action_tensor  = torch.cat(all_action_data, dim=0)   # (sum_T, Na)

    action_mean = action_tensor.mean(dim=0, keepdim=True)
    action_std  = action_tensor.std(dim=0,  keepdim=True).clamp(min=1e-2)
    proprio_mean = proprio_tensor.mean(dim=0, keepdim=True)
    proprio_std  = proprio_tensor.std(dim=0,  keepdim=True).clamp(min=1e-2)
    for fixed_key in (
        REAL_TRANSITION_CONDITION_KEY,
        EXCURSION_OBSERVED_KEY,
        CYCLE_PHASE_KEY,
        RETURN_COMMIT_KEY,
        ACTION_PRIMITIVE_KEY,
        WORK_CONTEXT_KEY,
        TASK_STATE_V2_KEY,
    ):
        if low_dim_slices and fixed_key in low_dim_slices:
            start, end = low_dim_slices[fixed_key]
            # Preserve the explicit condition/phase scale used by runtime.
            proprio_mean[:, start:end] = 0.0
            proprio_std[:, start:end] = 1.0
    qpos_mean    = qpos_tensor.mean(dim=0,    keepdim=True)
    qpos_std     = qpos_tensor.std(dim=0,     keepdim=True).clamp(min=1e-2)

    stats = {
        "action_mean":  action_mean.numpy().squeeze().astype(np.float32),
        "action_std":   action_std.numpy().squeeze().astype(np.float32),
        "proprio_mean": proprio_mean.numpy().squeeze().astype(np.float32),
        "proprio_std":  proprio_std.numpy().squeeze().astype(np.float32),
        "example_proprio": example_proprio,
        "proprio_keys": np.asarray(selected_low_dim_keys, dtype=object),
        "proprio_dim": int(proprio_tensor.shape[1]),
        "proprio_slices": dict(low_dim_slices or {}),
        "fixed_scale_keys": np.asarray(
            [
                key
                for key in selected_low_dim_keys
                if key
                in {
                    REAL_TRANSITION_CONDITION_KEY,
                    EXCURSION_OBSERVED_KEY,
                    CYCLE_PHASE_KEY,
                    RETURN_COMMIT_KEY,
                    ACTION_PRIMITIVE_KEY,
                    WORK_CONTEXT_KEY,
                    TASK_STATE_V2_KEY,
                }
            ],
            dtype=object,
        ),
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
            qpos_sequences, goal_effect_cfg.horizons
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
        condition_adherence_loss: dict[str, Any] | None = None,
        target_release_loss: dict[str, Any] | None = None,
        cycle_phase_loss: dict[str, Any] | None = None,
        excursion_observed_loss: dict[str, Any] | None = None,
        return_commit_loss: dict[str, Any] | None = None,
        action_primitive_islands: dict[str, Any] | None = None,
        work_return_context: dict[str, Any] | None = None,
        task_state_v2: dict[str, Any] | None = None,
        qvel_zero_state_hold_loss: dict[str, Any] | None = None,
        qvel_authority_loss: dict[str, Any] | None = None,
        factual_semantic_sampling: dict[str, Any] | None = None,
        goal_effect: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.episode_ids  = episode_ids
        self.dataset_dir  = Path(dataset_dir)
        self.camera_names = camera_names
        self.norm_stats   = norm_stats
        self.episode_len  = int(episode_len) if episode_len is not None else None
        self.low_dim_keys = _normalize_low_dim_keys(low_dim_keys)
        self.action_chunk_size = int(action_chunk_size) if action_chunk_size is not None else None
        self.image_transform_name = str(image_transform or "none")
        self.image_transform = build_image_transform(self.image_transform_name)
        self.deadzone_intent = _resolve_deadzone_intent_config(deadzone_intent)
        self.state_hold_transition = resolve_state_hold_transition_config(
            state_hold_transition
        )
        self.condition_adherence = resolve_condition_adherence_config(
            condition_adherence_loss
        )
        self.target_release = resolve_target_release_config(target_release_loss)
        self.cycle_phase_loss = resolve_cycle_phase_loss_config(cycle_phase_loss)
        self.excursion_observed_loss = resolve_excursion_observed_loss_config(
            excursion_observed_loss
        )
        self.return_commit_loss = resolve_return_commit_loss_config(
            return_commit_loss
        )
        self.action_primitive_islands = resolve_action_primitive_config(
            action_primitive_islands
        )
        self.work_return_context = resolve_work_return_context_config(
            work_return_context
        )
        self.task_state_v2 = resolve_task_state_v2_config(task_state_v2)
        self.qvel_zero_state_hold_loss = resolve_qvel_zero_state_hold_config(
            qvel_zero_state_hold_loss
        )
        self.qvel_authority_loss = resolve_qvel_authority_config(
            qvel_authority_loss
        )
        self.factual_semantic_sampling = resolve_factual_semantic_sampling_config(
            factual_semantic_sampling
        )
        self.goal_effect = resolve_goal_effect_config(
            goal_effect,
            target_scale=norm_stats.get("goal_effect_delta_scale"),
        )
        if self.condition_adherence["enabled"]:
            if REAL_TRANSITION_CONDITION_KEY not in self.low_dim_keys:
                raise ValueError(
                    "condition_adherence_loss requires real_transition_condition_v1"
                )
            if self.action_chunk_size is None:
                raise ValueError("condition_adherence_loss requires action_chunk_size")
        if self.target_release["enabled"]:
            if REAL_TRANSITION_CONDITION_KEY not in self.low_dim_keys:
                raise ValueError(
                    "target_release_loss requires real_transition_condition_v1"
                )
            if self.action_chunk_size is None:
                raise ValueError("target_release_loss requires action_chunk_size")
            if int(self.target_release["action_window_steps"]) > int(
                self.action_chunk_size
            ):
                raise ValueError(
                    "target_release_loss action_window_steps exceeds action_chunk_size"
                )
        if self.cycle_phase_loss["enabled"]:
            if CYCLE_PHASE_KEY not in self.low_dim_keys:
                raise ValueError(
                    f"cycle_phase_loss requires {CYCLE_PHASE_KEY}"
                )
        if self.excursion_observed_loss["enabled"]:
            if EXCURSION_OBSERVED_KEY not in self.low_dim_keys:
                raise ValueError(
                    f"excursion_observed_loss requires {EXCURSION_OBSERVED_KEY}"
                )
            if CYCLE_PHASE_KEY not in self.low_dim_keys:
                raise ValueError(
                    f"excursion_observed_loss requires {CYCLE_PHASE_KEY}"
                )
            if "qvel" not in self.low_dim_keys:
                raise ValueError("excursion_observed_loss requires qvel input")
            if self.action_chunk_size is None:
                raise ValueError(
                    "excursion_observed_loss requires action_chunk_size"
                )
            if int(self.excursion_observed_loss["action_window_steps"]) > int(
                self.action_chunk_size
            ):
                raise ValueError(
                    "excursion_observed_loss action_window_steps exceeds "
                    "action_chunk_size"
                )
        if self.qvel_zero_state_hold_loss["enabled"]:
            if "qvel" not in self.low_dim_keys:
                raise ValueError("qvel_zero_state_hold_loss requires qvel input")
            if not self.state_hold_transition["enabled"]:
                raise ValueError(
                    "qvel_zero_state_hold_loss requires state_hold_transition"
                )
            if self.action_chunk_size is None:
                raise ValueError("cycle_phase_loss requires action_chunk_size")
            if int(self.cycle_phase_loss["action_window_steps"]) > int(
                self.action_chunk_size
            ):
                raise ValueError(
                    "cycle_phase_loss action_window_steps exceeds action_chunk_size"
                )
        self._qvel_authority_manifest_by_episode: dict[int, dict[str, Any]] = {}
        if self.qvel_authority_loss["enabled"]:
            if "qvel" not in self.low_dim_keys:
                raise ValueError("qvel_authority_loss requires qvel input")
            if self.action_chunk_size is None:
                raise ValueError("qvel_authority_loss requires action_chunk_size")
            if int(self.qvel_authority_loss["action_window_steps"]) > int(
                self.action_chunk_size
            ):
                raise ValueError(
                    "qvel_authority_loss action_window_steps exceeds action_chunk_size"
                )
            manifest_path = Path(str(self.qvel_authority_loss["manifest_path"]))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != "real_transition_action_primitive_islands_v1":
                raise ValueError("qvel authority primitive manifest schema mismatch")
            if int(manifest.get("chunk_steps", -1)) != int(self.action_chunk_size):
                raise ValueError("qvel authority manifest chunk size mismatch")
            manifest_episodes = Path(str(manifest.get("dataset_root", ""))) / "episodes"
            if manifest_episodes.resolve() != self.dataset_dir.resolve():
                raise ValueError("qvel authority manifest dataset root mismatch")
            self._qvel_authority_manifest_by_episode = {
                int(row["episode_id"]): dict(row)
                for row in manifest.get("episodes", [])
            }
            missing_ids = sorted(
                set(int(value) for value in self.episode_ids)
                - set(self._qvel_authority_manifest_by_episode)
            )
            if missing_ids:
                raise ValueError(
                    f"qvel authority manifest is missing episodes {missing_ids[:20]}"
                )
        self._factual_semantic_manifest_by_episode: dict[int, dict[str, Any]] = {}
        if self.factual_semantic_sampling["enabled"]:
            if "qvel" not in self.low_dim_keys:
                raise ValueError("factual_semantic_sampling requires qvel input")
            conflicts = {
                "target_release_loss": self.target_release["enabled"],
                "cycle_phase_loss": self.cycle_phase_loss["enabled"],
                "excursion_observed_loss": self.excursion_observed_loss["enabled"],
                "return_commit_loss": self.return_commit_loss["enabled"],
                "qvel_zero_state_hold_loss": self.qvel_zero_state_hold_loss["enabled"],
                "qvel_authority_loss": self.qvel_authority_loss["enabled"],
                "action_primitive_islands": self.action_primitive_islands["enabled"],
                "work_return_context": self.work_return_context["enabled"],
            }
            active = [name for name, enabled in conflicts.items() if enabled]
            if active:
                raise ValueError(
                    "factual_semantic_sampling is sampling-only and cannot combine "
                    "with " + ", ".join(active)
                )
            manifest_path = Path(
                str(self.factual_semantic_sampling["manifest_path"])
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != FACTUAL_SEMANTIC_MANIFEST_SCHEMA:
                raise ValueError("factual semantic sampling manifest schema mismatch")
            manifest_root = Path(str(manifest.get("dataset_root", ""))) / "episodes"
            if manifest_root.resolve() != self.dataset_dir.resolve():
                raise ValueError("factual semantic sampling dataset root mismatch")
            self._factual_semantic_manifest_by_episode = {
                int(row["episode_id"]): dict(row)
                for row in manifest.get("episodes", [])
            }
            missing_ids = sorted(
                set(int(value) for value in self.episode_ids)
                - set(self._factual_semantic_manifest_by_episode)
            )
            if missing_ids:
                raise ValueError(
                    "factual semantic manifest is missing episodes "
                    f"{missing_ids[:20]}"
                )
        if self.return_commit_loss["enabled"]:
            if RETURN_COMMIT_KEY not in self.low_dim_keys:
                raise ValueError(
                    f"return_commit_loss requires {RETURN_COMMIT_KEY}"
                )
            if self.action_chunk_size is None:
                raise ValueError("return_commit_loss requires action_chunk_size")
            if int(self.return_commit_loss["action_window_steps"]) > int(
                self.action_chunk_size
            ):
                raise ValueError(
                    "return_commit_loss action_window_steps exceeds action_chunk_size"
                )
        self._action_primitive_manifest_by_episode: dict[int, dict[str, Any]] = {}
        if self.action_primitive_islands["enabled"]:
            if ACTION_PRIMITIVE_KEY not in self.low_dim_keys:
                raise ValueError(
                    f"action_primitive_islands requires {ACTION_PRIMITIVE_KEY}"
                )
            if self.action_chunk_size is None:
                raise ValueError(
                    "action_primitive_islands requires action_chunk_size"
                )
            if int(self.action_primitive_islands["action_window_steps"]) != int(
                self.action_chunk_size
            ):
                raise ValueError(
                    "action_primitive_islands.action_window_steps must equal "
                    "ACT action_chunk_size"
                )
            conflicting = {
                "state_hold_transition": self.state_hold_transition["enabled"],
                "condition_adherence_loss": self.condition_adherence["enabled"],
                "target_release_loss": self.target_release["enabled"],
                "cycle_phase_loss": self.cycle_phase_loss["enabled"],
                "excursion_observed_loss": self.excursion_observed_loss["enabled"],
                "return_commit_loss": self.return_commit_loss["enabled"],
                "qvel_zero_state_hold_loss": self.qvel_zero_state_hold_loss["enabled"],
            }
            active_conflicts = [name for name, enabled in conflicting.items() if enabled]
            if active_conflicts:
                raise ValueError(
                    "oracle action primitive training is factual-only and cannot "
                    "combine with " + ", ".join(active_conflicts)
                )
            manifest_path = Path(
                str(self.action_primitive_islands["manifest_path"])
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != (
                "real_transition_action_primitive_islands_v1"
            ):
                raise ValueError("action primitive manifest schema mismatch")
            if tuple(manifest.get("primitive_names", ())) != PRIMITIVE_NAMES:
                raise ValueError("action primitive manifest names mismatch")
            if int(manifest.get("chunk_steps", -1)) != int(self.action_chunk_size):
                raise ValueError("action primitive manifest chunk size mismatch")
            manifest_episodes = Path(str(manifest.get("dataset_root", ""))) / "episodes"
            if manifest_episodes.resolve() != self.dataset_dir.resolve():
                raise ValueError("action primitive manifest dataset root mismatch")
            self._action_primitive_manifest_by_episode = {
                int(row["episode_id"]): dict(row)
                for row in manifest.get("episodes", [])
            }
            missing_manifest_ids = sorted(
                set(int(value) for value in self.episode_ids)
                - set(self._action_primitive_manifest_by_episode)
            )
            if missing_manifest_ids:
                raise ValueError(
                    "action primitive manifest is missing episodes "
                    f"{missing_manifest_ids[:20]}"
                )
        elif ACTION_PRIMITIVE_KEY in self.low_dim_keys:
            raise ValueError(
                f"low_dim_keys contains {ACTION_PRIMITIVE_KEY!r} but "
                "action_primitive_islands is disabled"
            )
        self._work_return_manifest_by_episode: dict[int, dict[str, Any]] = {}
        if self.work_return_context["enabled"]:
            if WORK_CONTEXT_KEY not in self.low_dim_keys:
                raise ValueError(
                    f"work_return_context requires {WORK_CONTEXT_KEY}"
                )
            if self.action_chunk_size is None:
                raise ValueError("work_return_context requires action_chunk_size")
            if int(self.work_return_context["action_window_steps"]) != int(
                self.action_chunk_size
            ):
                raise ValueError(
                    "work_return_context.action_window_steps must equal ACT "
                    "action_chunk_size"
                )
            conflicting = {
                "action_primitive_islands": self.action_primitive_islands["enabled"],
                "state_hold_transition": self.state_hold_transition["enabled"],
                "condition_adherence_loss": self.condition_adherence["enabled"],
                "target_release_loss": self.target_release["enabled"],
                "cycle_phase_loss": self.cycle_phase_loss["enabled"],
                "excursion_observed_loss": self.excursion_observed_loss["enabled"],
                "return_commit_loss": self.return_commit_loss["enabled"],
                "qvel_zero_state_hold_loss": self.qvel_zero_state_hold_loss["enabled"],
            }
            active_conflicts = [name for name, enabled in conflicting.items() if enabled]
            if active_conflicts:
                raise ValueError(
                    "WORK/RETURN context training is factual-only and cannot combine "
                    "with " + ", ".join(active_conflicts)
                )
            manifest_path = Path(str(self.work_return_context["manifest_path"]))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != (
                "real_transition_work_return_context_manifest_v1"
            ):
                raise ValueError("work return context manifest schema mismatch")
            if int(manifest.get("chunk_steps", -1)) != int(self.action_chunk_size):
                raise ValueError("work return context manifest chunk size mismatch")
            manifest_episodes = Path(str(manifest.get("dataset_root", ""))) / "episodes"
            if manifest_episodes.resolve() != self.dataset_dir.resolve():
                raise ValueError("work return context manifest dataset root mismatch")
            self._work_return_manifest_by_episode = {
                int(row["episode_id"]): dict(row)
                for row in manifest.get("episodes", [])
            }
            missing_ids = sorted(
                set(int(value) for value in self.episode_ids)
                - set(self._work_return_manifest_by_episode)
            )
            if missing_ids:
                raise ValueError(
                    f"work return context manifest is missing episodes {missing_ids[:20]}"
                )
        elif WORK_CONTEXT_KEY in self.low_dim_keys:
            raise ValueError(
                f"low_dim_keys contains {WORK_CONTEXT_KEY!r} but "
                "work_return_context is disabled"
            )
        self._task_state_v2_manifest_by_episode: dict[int, dict[str, Any]] = {}
        if self.task_state_v2["enabled"]:
            expected_keys = ["qpos", "qvel", TASK_STATE_V2_KEY]
            if self.low_dim_keys != expected_keys:
                raise ValueError(
                    "task_state_v2 requires the exact low_dim_keys order "
                    f"{expected_keys!r}"
                )
            if self.action_chunk_size is None:
                raise ValueError("task_state_v2 requires action_chunk_size")
            if int(self.task_state_v2["action_window_steps"]) != int(
                self.action_chunk_size
            ):
                raise ValueError(
                    "task_state_v2.action_window_steps must equal ACT "
                    "action_chunk_size"
                )
            conflicts = {
                "state_hold_transition": self.state_hold_transition["enabled"],
                "condition_adherence_loss": self.condition_adherence["enabled"],
                "target_release_loss": self.target_release["enabled"],
                "cycle_phase_loss": self.cycle_phase_loss["enabled"],
                "excursion_observed_loss": self.excursion_observed_loss["enabled"],
                "return_commit_loss": self.return_commit_loss["enabled"],
                "qvel_zero_state_hold_loss": self.qvel_zero_state_hold_loss[
                    "enabled"
                ],
                "qvel_authority_loss": self.qvel_authority_loss["enabled"],
                "action_primitive_islands": self.action_primitive_islands["enabled"],
                "work_return_context": self.work_return_context["enabled"],
                "factual_semantic_sampling": self.factual_semantic_sampling[
                    "enabled"
                ],
            }
            active = [name for name, enabled in conflicts.items() if enabled]
            if active:
                raise ValueError(
                    "task_state_v2 is a standalone factual sampling contract and "
                    "cannot combine with " + ", ".join(active)
                )
            manifest = load_task_state_v2_manifest(
                str(self.task_state_v2["manifest_path"])
            )
            if manifest.get("schema") != TASK_STATE_V2_SCHEMA:
                raise ValueError("task_state_v2 manifest schema mismatch")
            if int(manifest.get("chunk_steps", -1)) != int(self.action_chunk_size):
                raise ValueError("task_state_v2 manifest chunk size mismatch")
            manifest_episodes = (
                Path(str(manifest.get("dataset_root", ""))) / "episodes"
            )
            if manifest_episodes.resolve() != self.dataset_dir.resolve():
                raise ValueError("task_state_v2 manifest dataset root mismatch")
            self._task_state_v2_manifest_by_episode = (
                task_state_manifest_by_episode_fn(manifest)
            )
            missing_ids = sorted(
                set(int(value) for value in self.episode_ids)
                - set(self._task_state_v2_manifest_by_episode)
            )
            if missing_ids:
                raise ValueError(
                    f"task_state_v2 manifest is missing episodes {missing_ids[:20]}"
                )
        elif TASK_STATE_V2_KEY in self.low_dim_keys:
            raise ValueError(
                f"low_dim_keys contains {TASK_STATE_V2_KEY!r} but "
                "task_state_v2 is disabled"
            )
        self.is_real: bool | None = None
        # Warm-up to populate self.is_real
        self.__getitem__(0)

    def __len__(self) -> int:
        multiplier = (
            1
            + int(self.state_hold_transition["append_samples_per_episode"])
            + int(self.target_release["append_samples_per_episode"])
            + int(self.cycle_phase_loss["append_samples_per_episode"])
            + int(self.excursion_observed_loss["append_samples_per_episode"])
            + int(self.return_commit_loss["append_samples_per_episode"])
            + int(
                self.action_primitive_islands["append_samples_per_episode"]
            )
            + int(self.work_return_context["append_samples_per_episode"])
            + int(self.task_state_v2["append_samples_per_episode"])
            + int(self.qvel_authority_loss["append_samples_per_episode"])
            + int(self.factual_semantic_sampling["append_samples_per_episode"])
        )
        return len(self.episode_ids) * multiplier

    def __getitem__(self, index: int):
        import h5py

        base_length = len(self.episode_ids)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = int(index) % base_length
        sample_tier = int(index) // base_length
        target_release_tiers = int(
            self.target_release["append_samples_per_episode"]
        )
        cycle_phase_tiers = int(
            self.cycle_phase_loss["append_samples_per_episode"]
        )
        excursion_tiers = int(
            self.excursion_observed_loss["append_samples_per_episode"]
        )
        return_commit_tiers = int(
            self.return_commit_loss["append_samples_per_episode"]
        )
        state_hold_tiers = int(
            self.state_hold_transition["append_samples_per_episode"]
        )
        primitive_tiers = int(
            self.action_primitive_islands["append_samples_per_episode"]
        )
        work_return_tiers = int(
            self.work_return_context["append_samples_per_episode"]
        )
        task_state_v2_tiers = int(
            self.task_state_v2["append_samples_per_episode"]
        )
        qvel_authority_tiers = int(
            self.qvel_authority_loss["append_samples_per_episode"]
        )
        factual_semantic_tiers = int(
            self.factual_semantic_sampling["append_samples_per_episode"]
        )
        force_target_release = 0 < sample_tier <= target_release_tiers
        force_cycle_phase = (
            target_release_tiers
            < sample_tier
            <= target_release_tiers + cycle_phase_tiers
        )
        force_excursion_observed = (
            target_release_tiers + cycle_phase_tiers
            < sample_tier
            <= target_release_tiers + cycle_phase_tiers + excursion_tiers
        )
        state_hold_tier_start = (
            target_release_tiers + cycle_phase_tiers + excursion_tiers
        )
        force_state_hold = (
            state_hold_tier_start
            < sample_tier
            <= state_hold_tier_start + state_hold_tiers
        )
        return_commit_tier_start = state_hold_tier_start + state_hold_tiers
        force_return_commit = (
            return_commit_tier_start
            < sample_tier
            <= return_commit_tier_start + return_commit_tiers
        )
        forced_return_commit_kind = (
            None
            if not force_return_commit
            else {
                1: self.return_commit_loss.get(
                    "dig_candidate_key", "dig_positive"
                ),
                2: "return_onset",
                3: "return_effective",
            }[sample_tier - return_commit_tier_start]
        )
        force_action_primitive = self.action_primitive_islands["enabled"]
        forced_action_primitive_kind = (
            PRIMITIVE_NAMES[sample_tier]
            if force_action_primitive and sample_tier <= primitive_tiers
            else None
        )
        forced_work_return_route = (
            ROUTE_NAMES[sample_tier]
            if self.work_return_context["enabled"]
            and sample_tier <= work_return_tiers
            else None
        )
        forced_task_state_v2_tier = (
            TASK_STATE_V2_TIERS[sample_tier]
            if self.task_state_v2["enabled"]
            and sample_tier <= task_state_v2_tiers
            else None
        )
        qvel_authority_tier_start = return_commit_tier_start + return_commit_tiers
        force_qvel_authority = (
            qvel_authority_tier_start
            < sample_tier
            <= qvel_authority_tier_start + qvel_authority_tiers
        )
        forced_qvel_authority_kind = (
            None
            if not force_qvel_authority
            else ("stable_tool" if sample_tier - qvel_authority_tier_start == 1 else "moving_return")
        )
        factual_semantic_tier_start = (
            qvel_authority_tier_start + qvel_authority_tiers
        )
        force_factual_semantic = (
            factual_semantic_tier_start
            < sample_tier
            <= factual_semantic_tier_start + factual_semantic_tiers
        )
        forced_factual_semantic_kind = (
            None
            if not force_factual_semantic
            else FACTUAL_SEMANTIC_TIER_NAMES[
                sample_tier - factual_semantic_tier_start - 1
            ]
        )
        ep_id  = self.episode_ids[episode_index]
        path   = self.dataset_dir / f"episode_{ep_id}.hdf5"

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
            full_action = (
                np.asarray(f["/action"][()], dtype=np.float32)
                if self.action_primitive_islands["enabled"]
                or self.work_return_context["enabled"]
                or self.qvel_authority_loss["enabled"]
                else None
            )
            state_transition_mask = None
            state_transition_starts = np.zeros(0, dtype=np.int64)
            if self.state_hold_transition["enabled"]:
                if not action_prealigned:
                    raise ValueError(
                        "state_hold_transition requires action_prealigned=true"
                    )
                full_action = np.asarray(f["/action"][()], dtype=np.float32)
                state_transition_mask = _state_hold_transition_mask(
                    f,
                    actions=full_action,
                    config=self.state_hold_transition,
                )
                state_transition_starts = intersect_transition_starts(
                    state_transition_mask,
                    valid_starts,
                    total_steps=T,
                    hold_horizon_steps=int(
                        self.state_hold_transition["hold_horizon_steps"]
                    ),
                )
                t0 = sample_state_hold_start(
                    valid_starts=valid_starts,
                    transition_starts=state_transition_starts,
                    probability=float(self.state_hold_transition["probability"]),
                )
                if force_state_hold:
                    if state_transition_starts.size == 0:
                        raise ValueError(
                            f"Episode {ep_id} has no forced state-hold transition"
                        )
                    t0 = int(np.random.choice(state_transition_starts))
            else:
                t0 = int(np.random.choice(valid_starts))

            task_state_v2_value = None
            task_state_v2_chunk_mask = None
            if self.task_state_v2["enabled"]:
                if not action_prealigned:
                    raise ValueError("task_state_v2 requires action_prealigned=true")
                if forced_task_state_v2_tier is None:
                    raise AssertionError("missing forced task_state_v2 tier")
                task_row = self._task_state_v2_manifest_by_episode[ep_id]
                if int(task_row.get("n_rows", -1)) != T:
                    raise ValueError(f"episode {ep_id} task_state_v2 row count changed")
                candidates = task_state_candidate_starts(
                    total_steps=T,
                    work_complete_row=int(task_row["work_complete_row"]),
                    return_commit_row=int(task_row["return_commit_row"]),
                    action_window_steps=int(
                        self.task_state_v2["action_window_steps"]
                    ),
                    valid_starts=valid_starts,
                ).by_name()
                expected_starts = {
                    str(name): np.asarray(values, dtype=np.int64)
                    for name, values in dict(
                        task_row.get("candidate_starts", {})
                    ).items()
                }
                if set(expected_starts) != set(TASK_STATE_V2_TIERS) or any(
                    not np.array_equal(expected_starts[name], candidates[name])
                    for name in TASK_STATE_V2_TIERS
                ):
                    raise ValueError(
                        f"episode {ep_id} task_state_v2 population changed"
                    )
                selected = candidates[forced_task_state_v2_tier]
                if selected.size == 0:
                    raise ValueError(
                        f"episode {ep_id} has no task_state_v2 candidates for "
                        f"{forced_task_state_v2_tier}"
                    )
                t0 = int(np.random.choice(selected))
                task_sequence = build_task_state_sequence(
                    total_steps=T,
                    current_side=str(task_row["current_side"]),
                    dig_target=str(task_row["dig_target"]),
                    next_target=str(task_row["next_target"]),
                    work_complete_row=int(task_row["work_complete_row"]),
                    return_commit_row=int(task_row["return_commit_row"]),
                )
                task_state_v2_value = task_sequence[t0]
                task_state_v2_chunk_mask = task_state_chunk_valid_mask(
                    timestep=t0,
                    total_steps=T,
                    action_chunk_size=int(self.action_chunk_size),
                    work_complete_row=int(task_row["work_complete_row"]),
                    return_commit_row=int(task_row["return_commit_row"]),
                )

            action_primitive = None
            if self.action_primitive_islands["enabled"]:
                if full_action is None:
                    raise AssertionError("action primitive sampling requires actions")
                islands = derive_action_primitive_islands(
                    full_action,
                    positive_thresholds=self.action_primitive_islands[
                        "positive_thresholds"
                    ],
                    negative_thresholds=self.action_primitive_islands[
                        "negative_thresholds"
                    ],
                    action_window_steps=int(
                        self.action_primitive_islands["action_window_steps"]
                    ),
                    valid_starts=valid_starts,
                )
                manifest_row = self._action_primitive_manifest_by_episode[ep_id]
                expected_counts = {
                    str(name): int(value)
                    for name, value in dict(
                        manifest_row.get("candidate_counts", {})
                    ).items()
                }
                actual_counts = {
                    name: int(islands.candidate_starts[name].size)
                    for name in PRIMITIVE_NAMES
                }
                if expected_counts != actual_counts:
                    raise ValueError(
                        f"episode {ep_id} action primitive population changed: "
                        f"expected={expected_counts}, actual={actual_counts}"
                    )
                if forced_action_primitive_kind is None:
                    raise AssertionError("missing forced action primitive tier")
                candidate_starts = islands.candidate_starts[
                    forced_action_primitive_kind
                ]
                if candidate_starts.size == 0:
                    raise ValueError(
                        f"episode {ep_id} has no full-window candidates for "
                        f"{forced_action_primitive_kind}"
                    )
                t0 = int(np.random.choice(candidate_starts))
                action_primitive = primitive_one_hot(
                    forced_action_primitive_kind
                )

            work_context = None
            if self.work_return_context["enabled"]:
                if full_action is None:
                    raise AssertionError("WORK/RETURN sampling requires actions")
                route_context = derive_work_return_context(
                    full_action,
                    positive_thresholds=self.work_return_context[
                        "positive_thresholds"
                    ],
                    negative_thresholds=self.work_return_context[
                        "negative_thresholds"
                    ],
                    action_window_steps=int(
                        self.work_return_context["action_window_steps"]
                    ),
                    valid_starts=valid_starts,
                )
                manifest_row = self._work_return_manifest_by_episode[ep_id]
                expected_counts = {
                    str(name): int(value)
                    for name, value in dict(
                        manifest_row.get("candidate_counts", {})
                    ).items()
                }
                actual_counts = {
                    "work": int(route_context.work_starts.size),
                    "return": int(route_context.return_starts.size),
                }
                if expected_counts != actual_counts or int(
                    manifest_row.get("work_complete_boundary_row", -1)
                ) != int(route_context.boundary_row):
                    raise ValueError(
                        f"episode {ep_id} WORK/RETURN population changed"
                    )
                if forced_work_return_route is None:
                    raise AssertionError("missing forced WORK/RETURN tier")
                starts = (
                    route_context.work_starts
                    if forced_work_return_route == "work"
                    else route_context.return_starts
                )
                t0 = int(np.random.choice(starts))
                work_context = work_context_vector(
                    current_anchor=str(manifest_row["current_anchor"]),
                    dig_target=str(manifest_row["dig_target"]),
                    next_target=str(manifest_row["next_target"]),
                    route=forced_work_return_route,
                )

            qvel_authority_candidates = {
                "stable_tool": np.zeros(0, dtype=np.int64),
                "moving_return": np.zeros(0, dtype=np.int64),
            }
            qvel_authority_moving_primary = False
            qvel_authority_tool_mask = np.zeros((4, 2), dtype=bool)
            if self.qvel_authority_loss["enabled"]:
                if full_action is None:
                    raise AssertionError("qvel authority sampling requires actions")
                manifest_row = self._qvel_authority_manifest_by_episode[ep_id]
                qvel_authority_candidates = qvel_authority_candidate_starts(
                    qvel=np.asarray(f["/observations/qvel"][()], dtype=np.float32),
                    valid_starts=valid_starts,
                    segments=dict(manifest_row.get("segments", {})),
                    chunk_steps=int(self.action_chunk_size),
                    config=self.qvel_authority_loss,
                )
                missing = [
                    name
                    for name, values in qvel_authority_candidates.items()
                    if values.size == 0
                ]
                if missing:
                    raise ValueError(
                        f"episode {ep_id} lacks qvel authority candidates {missing}"
                    )
                if forced_qvel_authority_kind is not None:
                    t0 = int(
                        np.random.choice(
                            qvel_authority_candidates[forced_qvel_authority_kind]
                        )
                    )
                    qvel_authority_moving_primary = (
                        forced_qvel_authority_kind == "moving_return"
                    )
                    if not qvel_authority_moving_primary:
                        qvel_authority_tool_mask = stable_tool_direction_mask(
                            full_action[t0], config=self.qvel_authority_loss
                        )

            if self.factual_semantic_sampling["enabled"]:
                if forced_factual_semantic_kind is not None:
                    manifest_row = self._factual_semantic_manifest_by_episode[ep_id]
                    field = f"{forced_factual_semantic_kind}_starts"
                    candidates = np.asarray(manifest_row.get(field, ()), dtype=np.int64)
                    candidates = candidates[
                        np.isin(candidates, valid_starts)
                    ]
                    if candidates.size == 0:
                        raise ValueError(
                            f"episode {ep_id} has no factual semantic candidates "
                            f"for {forced_factual_semantic_kind}"
                        )
                    t0 = int(np.random.choice(candidates))

            target_release_candidates = np.zeros(0, dtype=np.int64)
            if self.target_release["enabled"]:
                if not action_prealigned:
                    raise ValueError(
                        "target_release_loss requires action_prealigned=true"
                    )
                if full_action is None:
                    full_action = np.asarray(f["/action"][()], dtype=np.float32)
                target_release_candidates = target_release_candidate_indices(
                    qpos=np.asarray(f["/observations/qpos"][()], dtype=np.float32),
                    qvel=np.asarray(f["/observations/qvel"][()], dtype=np.float32),
                    actions=full_action,
                    condition=np.asarray(
                        f[f"conditions/{REAL_TRANSITION_CONDITION_KEY}"][()],
                        dtype=np.float32,
                    ),
                    valid_starts=valid_starts,
                    condition_valid_mask=np.asarray(
                        f["conditions/valid_mask"][()], dtype=bool
                    ),
                    config=self.target_release,
                )
                if target_release_candidates.size == 0:
                    raise ValueError(
                        f"Episode {ep_id} has no train-supported target-release start"
                    )
                if force_target_release:
                    t0 = int(np.random.choice(target_release_candidates))

            cycle_phase_candidates = {
                "pre_positive": np.zeros(0, dtype=np.int64),
                "return_negative": np.zeros(0, dtype=np.int64),
            }
            if self.cycle_phase_loss["enabled"]:
                if full_action is None:
                    full_action = np.asarray(f["/action"][()], dtype=np.float32)
                cycle_phase_candidates = cycle_phase_candidate_indices(
                    actions=full_action,
                    phase=np.asarray(
                        f[f"conditions/{CYCLE_PHASE_KEY}"][()], dtype=np.float32
                    ),
                    valid_starts=valid_starts,
                    phase_valid_mask=np.asarray(
                        f["conditions/cycle_phase_valid_mask"][()], dtype=bool
                    ),
                    config=self.cycle_phase_loss,
                )
                missing_kinds = [
                    key
                    for key, values in cycle_phase_candidates.items()
                    if values.size == 0
                ]
                if missing_kinds:
                    raise ValueError(
                        f"Episode {ep_id} has no cycle-phase candidates for "
                        + ", ".join(missing_kinds)
                    )
                if force_cycle_phase:
                    kind = (
                        "pre_positive"
                        if int(np.random.randint(0, 2)) == 0
                        else "return_negative"
                    )
                    t0 = int(np.random.choice(cycle_phase_candidates[kind]))

            excursion_candidates = {
                "pre_positive": np.zeros(0, dtype=np.int64),
                "apex_negative": np.zeros(0, dtype=np.int64),
                "moving_negative": np.zeros(0, dtype=np.int64),
            }
            if self.excursion_observed_loss["enabled"]:
                if full_action is None:
                    full_action = np.asarray(f["/action"][()], dtype=np.float32)
                excursion_candidates = excursion_observed_candidate_indices(
                    actions=full_action,
                    qvel=np.asarray(
                        f["/observations/qvel"][()], dtype=np.float32
                    ),
                    excursion_observed=np.asarray(
                        f[f"conditions/{EXCURSION_OBSERVED_KEY}"][()],
                        dtype=np.float32,
                    ),
                    return_phase=np.asarray(
                        f[f"conditions/{CYCLE_PHASE_KEY}"][()], dtype=np.float32
                    ),
                    valid_starts=valid_starts,
                    excursion_valid_mask=np.asarray(
                        f["conditions/excursion_observed_valid_mask"][()],
                        dtype=bool,
                    ),
                    config=self.excursion_observed_loss,
                )
                post_candidates = (
                    excursion_candidates["apex_negative"]
                    if excursion_candidates["apex_negative"].size
                    else excursion_candidates["moving_negative"]
                )
                if (
                    excursion_candidates["pre_positive"].size == 0
                    or post_candidates.size == 0
                ):
                    raise ValueError(
                        f"Episode {ep_id} has no excursion-observed pre/post "
                        "candidate"
                    )
                if force_excursion_observed:
                    selected = (
                        excursion_candidates["pre_positive"]
                        if int(np.random.randint(0, 2)) == 0
                        else post_candidates
                    )
                    t0 = int(np.random.choice(selected))

            return_commit_candidates = {
                "dig_positive": np.zeros(0, dtype=np.int64),
                "dig_non_swing_transition": np.zeros(0, dtype=np.int64),
                "return_onset": np.zeros(0, dtype=np.int64),
                "return_effective": np.zeros(0, dtype=np.int64),
            }
            if self.return_commit_loss["enabled"]:
                if full_action is None:
                    full_action = np.asarray(f["/action"][()], dtype=np.float32)
                return_commit_candidates = return_commit_candidate_indices(
                    actions=full_action,
                    return_commit=np.asarray(
                        f[f"conditions/{RETURN_COMMIT_KEY}"][()], dtype=np.float32
                    ),
                    valid_starts=valid_starts,
                    return_commit_valid_mask=np.asarray(
                        f["conditions/return_commit_valid_mask"][()], dtype=bool
                    ),
                    config=self.return_commit_loss,
                )
                required = (
                    self.return_commit_loss["dig_candidate_key"],
                    "return_onset",
                )
                missing = [
                    key
                    for key in required
                    if return_commit_candidates[key].size == 0
                ]
                if missing:
                    raise ValueError(
                        f"Episode {ep_id} has no return-commit candidates for "
                        + ", ".join(missing)
                    )
                if forced_return_commit_kind is not None:
                    t0 = int(
                        np.random.choice(
                            return_commit_candidates[forced_return_commit_kind]
                        )
                    )

            condition_anchor_start = None
            condition_transition_mask = None
            if self.condition_adherence["enabled"]:
                if not action_prealigned:
                    raise ValueError(
                        "condition_adherence_loss requires action_prealigned=true"
                    )
                if full_action is None:
                    full_action = np.asarray(f["/action"][()], dtype=np.float32)
                condition_transition_mask = _state_hold_transition_mask(
                    f,
                    actions=full_action,
                    config=self.condition_adherence,
                )
                condition_anchor_start = _terminal_swing_transition_anchor(
                    transition_mask=condition_transition_mask,
                    valid_starts=valid_starts,
                    total_steps=T,
                    chunk_steps=int(self.action_chunk_size),
                )

            # ── observation at t0 ─────────────────────────────────────────
            qpos = f["/observations/qpos"][t0]
            qvel = f["/observations/qvel"][t0]
            condition = _read_real_transition_condition(
                f,
                timestep=t0,
                enabled=REAL_TRANSITION_CONDITION_KEY in self.low_dim_keys,
            )
            excursion_observed = _read_real_transition_excursion_observed(
                f,
                timestep=t0,
                enabled=EXCURSION_OBSERVED_KEY in self.low_dim_keys,
            )
            cycle_phase = _read_real_transition_cycle_phase(
                f,
                timestep=t0,
                enabled=CYCLE_PHASE_KEY in self.low_dim_keys,
            )
            return_commit = _read_real_transition_return_commit(
                f,
                timestep=t0,
                enabled=RETURN_COMMIT_KEY in self.low_dim_keys,
            )
            proprio = _assemble_low_dim_observation(
                qpos=qpos,
                qvel=qvel,
                real_transition_condition_v1=condition,
                real_transition_excursion_observed_v1=excursion_observed,
                real_transition_cycle_phase_v1=cycle_phase,
                real_transition_return_commit_v1=return_commit,
                real_transition_action_primitive_v1=action_primitive,
                real_transition_work_context_v1=work_context,
                real_transition_task_state_v2=task_state_v2_value,
                low_dim_keys=self.low_dim_keys,
            )
            counterfactual_proprio = None
            if self.condition_adherence["enabled"] or self.target_release["enabled"]:
                if condition is None:
                    raise ValueError("condition intervention requires an active condition")
                counterfactual_condition = np.asarray(condition, dtype=np.float32).copy()
                counterfactual_condition[0] *= -1.0
                counterfactual_proprio = _assemble_low_dim_observation(
                    qpos=qpos,
                    qvel=qvel,
                    real_transition_condition_v1=counterfactual_condition,
                    real_transition_excursion_observed_v1=excursion_observed,
                    real_transition_cycle_phase_v1=cycle_phase,
                    real_transition_return_commit_v1=return_commit,
                    low_dim_keys=self.low_dim_keys,
                )
            phase_counterfactual_proprio = None
            if self.cycle_phase_loss["enabled"]:
                if cycle_phase is None:
                    raise ValueError("cycle phase intervention requires an active phase")
                counterfactual_phase = np.asarray(
                    cycle_phase, dtype=np.float32
                ).copy()
                counterfactual_phase[0] = 1.0 - counterfactual_phase[0]
                phase_counterfactual_proprio = _assemble_low_dim_observation(
                    qpos=qpos,
                    qvel=qvel,
                    real_transition_condition_v1=condition,
                    real_transition_excursion_observed_v1=excursion_observed,
                    real_transition_cycle_phase_v1=counterfactual_phase,
                    real_transition_return_commit_v1=return_commit,
                    low_dim_keys=self.low_dim_keys,
                )
            excursion_counterfactual_proprio = None
            if self.excursion_observed_loss["enabled"]:
                if excursion_observed is None:
                    raise ValueError(
                        "excursion observed intervention requires active state"
                    )
                counterfactual_excursion = np.asarray(
                    excursion_observed, dtype=np.float32
                ).copy()
                counterfactual_excursion[0] = 1.0 - counterfactual_excursion[0]
                excursion_counterfactual_proprio = _assemble_low_dim_observation(
                    qpos=qpos,
                    qvel=qvel,
                    real_transition_condition_v1=condition,
                    real_transition_excursion_observed_v1=(
                        counterfactual_excursion
                    ),
                    real_transition_cycle_phase_v1=cycle_phase,
                    real_transition_return_commit_v1=return_commit,
                    low_dim_keys=self.low_dim_keys,
                )
            return_commit_counterfactual_proprio = None
            if self.return_commit_loss["enabled"]:
                if return_commit is None:
                    raise ValueError(
                        "return commit intervention requires active planner state"
                    )
                counterfactual_return_commit = np.asarray(
                    return_commit, dtype=np.float32
                ).copy()
                counterfactual_return_commit[0] = (
                    1.0 - counterfactual_return_commit[0]
                )
                return_commit_counterfactual_proprio = (
                    _assemble_low_dim_observation(
                        qpos=qpos,
                        qvel=qvel,
                        real_transition_condition_v1=condition,
                        real_transition_excursion_observed_v1=(
                            excursion_observed
                        ),
                        real_transition_cycle_phase_v1=cycle_phase,
                        real_transition_return_commit_v1=(
                            counterfactual_return_commit
                        ),
                        low_dim_keys=self.low_dim_keys,
                    )
                )
            qvel_zero_proprio = None
            if self.qvel_zero_state_hold_loss["enabled"]:
                qvel_zero_proprio = _assemble_low_dim_observation(
                    qpos=qpos,
                    qvel=np.zeros_like(qvel, dtype=np.float32),
                    real_transition_condition_v1=condition,
                    real_transition_excursion_observed_v1=excursion_observed,
                    real_transition_cycle_phase_v1=cycle_phase,
                    real_transition_return_commit_v1=return_commit,
                    low_dim_keys=self.low_dim_keys,
                )
            qvel_authority_counterfactual_proprio = None
            if self.qvel_authority_loss["enabled"]:
                counterfactual_qvel = np.asarray(qvel, dtype=np.float32).copy()
                counterfactual_qvel[0] = (
                    0.0
                    if qvel_authority_moving_primary
                    else float(
                        self.qvel_authority_loss[
                            "counterfactual_moving_swing_qvel"
                        ]
                    )
                )
                qvel_authority_counterfactual_proprio = _assemble_low_dim_observation(
                    qpos=qpos,
                    qvel=counterfactual_qvel,
                    real_transition_condition_v1=condition,
                    real_transition_excursion_observed_v1=excursion_observed,
                    real_transition_cycle_phase_v1=cycle_phase,
                    real_transition_return_commit_v1=return_commit,
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
            action     = f["/action"][start:]
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
            chunk_valid_mask = _read_condition_chunk_valid_mask(
                f,
                timestep=t0,
                action_chunk_size=self.action_chunk_size,
                enabled=REAL_TRANSITION_CONDITION_KEY in self.low_dim_keys,
            )
            if task_state_v2_chunk_mask is not None:
                chunk_valid_mask = (
                    task_state_v2_chunk_mask
                    if chunk_valid_mask is None
                    else chunk_valid_mask & task_state_v2_chunk_mask
                )
            deadzone_labels = None
            if self.deadzone_intent["enabled"]:
                if full_action is None:
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

            state_hold_transition_mask = None
            if state_transition_mask is not None:
                state_hold_transition_mask = np.zeros((4, 2), dtype=bool)
                if np.any(state_transition_starts == t0):
                    state_hold_transition_mask = anchor_transition_direction_mask(
                        state_transition_mask, t0
                    )

            condition_adherence_mask = None
            condition_action_target = None
            condition_action_valid = False
            if condition_transition_mask is not None:
                condition_adherence_mask = np.zeros(
                    (int(self.action_chunk_size), 4, 2), dtype=bool
                )
                if (
                    condition_anchor_start is not None
                    and t0 <= condition_anchor_start < t0 + int(self.action_chunk_size)
                ):
                    relative = int(condition_anchor_start - t0)
                    anchor_direction = anchor_transition_direction_mask(
                        condition_transition_mask, condition_anchor_start
                    )
                    condition_adherence_mask[relative, 0] = anchor_direction[0]
                condition_action_target = np.int64(1 if float(condition[0]) > 0.0 else 0)
                # The terminal transition remains the anchor for the
                # reward-shaped adherence terms.  The action-class target can
                # optionally be emitted at every active observation so the
                # condition branch cannot be learned from a sparse terminal
                # window only.  This is still hindsight-labelled: the side
                # code is recorded by the materializer and is constant for
                # the cycle.
                condition_action_valid = bool(
                    condition_adherence_mask.any()
                    or (
                        self.condition_adherence.get("action_label_scope")
                        == "all_active_steps"
                        and float(condition[1]) == 1.0
                    )
                )
            target_release_valid = bool(
                self.target_release["enabled"]
                and np.any(target_release_candidates == int(t0))
            )
            target_release_continue_primary = bool(
                target_release_valid and condition is not None and float(condition[0]) < 0.0
            )
            cycle_phase_valid = bool(
                self.cycle_phase_loss["enabled"]
                and any(
                    np.any(values == int(t0))
                    for values in cycle_phase_candidates.values()
                )
            )
            cycle_phase_return_primary = bool(
                cycle_phase_valid
                and cycle_phase is not None
                and float(cycle_phase[0]) > 0.0
            )
            excursion_observed_valid = bool(
                self.excursion_observed_loss["enabled"]
                and any(
                    np.any(values == int(t0))
                    for values in excursion_candidates.values()
                )
            )
            excursion_post_primary = bool(
                excursion_observed_valid
                and excursion_observed is not None
                and float(excursion_observed[0]) > 0.0
            )
            return_commit_valid = bool(
                self.return_commit_loss["enabled"]
                and any(
                    np.any(values == int(t0))
                    for key, values in return_commit_candidates.items()
                    if key
                    in {
                        self.return_commit_loss.get(
                            "dig_candidate_key", "dig_positive"
                        ),
                        "return_onset",
                        "return_effective",
                    }
                )
            )
            return_commit_return_primary = bool(
                return_commit_valid
                and return_commit is not None
                and float(return_commit[0]) > 0.0
            )
            return_commit_return_effective_primary = bool(
                return_commit_valid
                and np.any(
                    return_commit_candidates["return_effective"] == int(t0)
                )
            )

        self.is_real = is_real

        # ── pad action to fixed dataset length for batching ────────────────
        target_len = self.episode_len if self.episode_len is not None else T
        if T > target_len:
            raise ValueError(
                f"Episode {ep_id} has length {T}, which exceeds configured "
                f"episode_len {target_len}. Increase task.episode_len or re-record."
            )

        padded_action = np.zeros((target_len, original_action_shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.ones(target_len, dtype=bool)
        is_pad[:action_len] = False
        if chunk_valid_mask is not None:
            mask_len = min(int(chunk_valid_mask.size), target_len)
            is_pad[:mask_len] |= ~chunk_valid_mask[:mask_len]
        padded_move_mask = None
        padded_stop_mask = None
        padded_wrong_mask = None
        padded_action_loss_mask = None
        if deadzone_labels is not None:
            padded_move_mask = np.zeros((target_len, original_action_shape[1], 2), dtype=bool)
            padded_stop_mask = np.zeros(target_len, dtype=bool)
            padded_wrong_mask = np.zeros((target_len, original_action_shape[1], 2), dtype=bool)
            padded_action_loss_mask = np.zeros(target_len, dtype=bool)
            padded_move_mask[:action_len] = deadzone_labels.move_mask[start:]
            padded_stop_mask[:action_len] = deadzone_labels.stop_mask[start:]
            padded_wrong_mask[:action_len] = deadzone_labels.wrong_mask[start:]
            padded_action_loss_mask[:action_len] = deadzone_labels.action_loss_mask[start:]

        # ── assemble camera tensor ─────────────────────────────────────────
        all_cam_images = np.stack(
            [image_dict[c] for c in self.camera_names], axis=0
        )  # (n_cams, H, W, 3)

        # ── convert to tensors ────────────────────────────────────────────
        image_data   = torch.from_numpy(all_cam_images)
        proprio_data = torch.from_numpy(proprio).float()
        action_data  = torch.from_numpy(padded_action).float()
        is_pad_t     = torch.from_numpy(is_pad)

        # channel-last → channel-first + normalize to [0, 1]
        image_data = torch.einsum("k h w c -> k c h w", image_data).float() / 255.0

        # normalise proprio and actions
        action_data = (
            action_data
            - torch.from_numpy(self.norm_stats["action_mean"])
        ) / torch.from_numpy(self.norm_stats["action_std"])
        proprio_data = (
            proprio_data
            - torch.from_numpy(self.norm_stats["proprio_mean"])
        ) / torch.from_numpy(self.norm_stats["proprio_std"])
        counterfactual_proprio_data = None
        if counterfactual_proprio is not None:
            counterfactual_proprio_data = (
                torch.from_numpy(counterfactual_proprio).float()
                - torch.from_numpy(self.norm_stats["proprio_mean"])
            ) / torch.from_numpy(self.norm_stats["proprio_std"])
        phase_counterfactual_proprio_data = None
        if phase_counterfactual_proprio is not None:
            phase_counterfactual_proprio_data = (
                torch.from_numpy(phase_counterfactual_proprio).float()
                - torch.from_numpy(self.norm_stats["proprio_mean"])
            ) / torch.from_numpy(self.norm_stats["proprio_std"])
        excursion_counterfactual_proprio_data = None
        if excursion_counterfactual_proprio is not None:
            excursion_counterfactual_proprio_data = (
                torch.from_numpy(excursion_counterfactual_proprio).float()
                - torch.from_numpy(self.norm_stats["proprio_mean"])
            ) / torch.from_numpy(self.norm_stats["proprio_std"])
        return_commit_counterfactual_proprio_data = None
        if return_commit_counterfactual_proprio is not None:
            return_commit_counterfactual_proprio_data = (
                torch.from_numpy(return_commit_counterfactual_proprio).float()
                - torch.from_numpy(self.norm_stats["proprio_mean"])
            ) / torch.from_numpy(self.norm_stats["proprio_std"])
        qvel_zero_proprio_data = None
        if qvel_zero_proprio is not None:
            qvel_zero_proprio_data = (
                torch.from_numpy(qvel_zero_proprio).float()
                - torch.from_numpy(self.norm_stats["proprio_mean"])
            ) / torch.from_numpy(self.norm_stats["proprio_std"])
        qvel_authority_counterfactual_proprio_data = None
        if qvel_authority_counterfactual_proprio is not None:
            qvel_authority_counterfactual_proprio_data = (
                torch.from_numpy(qvel_authority_counterfactual_proprio).float()
                - torch.from_numpy(self.norm_stats["proprio_mean"])
            ) / torch.from_numpy(self.norm_stats["proprio_std"])

        if (
            deadzone_labels is None
            and state_hold_transition_mask is None
            and condition_adherence_mask is None
            and not self.target_release["enabled"]
            and not self.cycle_phase_loss["enabled"]
            and not self.excursion_observed_loss["enabled"]
            and not self.return_commit_loss["enabled"]
            and not self.qvel_zero_state_hold_loss["enabled"]
            and not self.qvel_authority_loss["enabled"]
            and not self.task_state_v2["enabled"]
            and goal_effect_targets is None
        ):
            return image_data, proprio_data, action_data, is_pad_t

        payload = {
            "image": image_data,
            "proprio": proprio_data,
            "action": action_data,
            "is_pad": is_pad_t,
        }
        if self.task_state_v2["enabled"]:
            if task_state_v2_value is None:
                raise AssertionError("task_state_v2 sample is missing its state")
            payload["task_state_v2_uncommitted"] = torch.tensor(
                bool(
                    float(task_state_v2_value[2]) == 1.0
                    and float(task_state_v2_value[3]) == 0.0
                ),
                dtype=torch.bool,
            )
        if deadzone_labels is not None:
            payload.update(
                {
                    "deadzone_move_mask": torch.from_numpy(padded_move_mask),
                    "deadzone_stop_mask": torch.from_numpy(padded_stop_mask),
                    "deadzone_wrong_mask": torch.from_numpy(padded_wrong_mask),
                    "action_loss_mask": torch.from_numpy(padded_action_loss_mask),
                }
            )
        if state_hold_transition_mask is not None:
            payload["state_hold_transition_mask"] = torch.from_numpy(
                state_hold_transition_mask
            )
        if condition_adherence_mask is not None:
            payload["condition_adherence_mask"] = torch.from_numpy(
                condition_adherence_mask
            )
            payload["condition_action_target"] = torch.tensor(
                int(condition_action_target), dtype=torch.long
            )
            payload["condition_action_valid"] = torch.tensor(
                condition_action_valid, dtype=torch.bool
            )
        if counterfactual_proprio_data is not None:
            payload["counterfactual_proprio"] = counterfactual_proprio_data
        if phase_counterfactual_proprio_data is not None:
            payload["phase_counterfactual_proprio"] = (
                phase_counterfactual_proprio_data
            )
        if excursion_counterfactual_proprio_data is not None:
            payload["excursion_counterfactual_proprio"] = (
                excursion_counterfactual_proprio_data
            )
        if return_commit_counterfactual_proprio_data is not None:
            payload["return_commit_counterfactual_proprio"] = (
                return_commit_counterfactual_proprio_data
            )
        if qvel_zero_proprio_data is not None:
            payload["qvel_zero_proprio"] = qvel_zero_proprio_data
        if qvel_authority_counterfactual_proprio_data is not None:
            payload["qvel_authority_counterfactual_proprio"] = (
                qvel_authority_counterfactual_proprio_data
            )
            payload["qvel_authority_moving_primary"] = torch.tensor(
                qvel_authority_moving_primary, dtype=torch.bool
            )
            payload["qvel_authority_stable_tool_mask"] = torch.from_numpy(
                qvel_authority_tool_mask
            )
            payload["qvel_authority_valid"] = torch.tensor(
                force_qvel_authority, dtype=torch.bool
            )
        if self.target_release["enabled"]:
            payload["target_release_continue_primary"] = torch.tensor(
                target_release_continue_primary, dtype=torch.bool
            )
            payload["target_release_valid"] = torch.tensor(
                target_release_valid, dtype=torch.bool
            )
        if self.cycle_phase_loss["enabled"]:
            payload["cycle_phase_return_primary"] = torch.tensor(
                cycle_phase_return_primary, dtype=torch.bool
            )
            payload["cycle_phase_valid"] = torch.tensor(
                cycle_phase_valid, dtype=torch.bool
            )
        if self.excursion_observed_loss["enabled"]:
            payload["excursion_post_primary"] = torch.tensor(
                excursion_post_primary, dtype=torch.bool
            )
            payload["excursion_observed_valid"] = torch.tensor(
                excursion_observed_valid, dtype=torch.bool
            )
        if self.return_commit_loss["enabled"]:
            payload["return_commit_return_primary"] = torch.tensor(
                return_commit_return_primary, dtype=torch.bool
            )
            payload["return_commit_return_effective_primary"] = torch.tensor(
                return_commit_return_effective_primary, dtype=torch.bool
            )
            payload["return_commit_valid"] = torch.tensor(
                return_commit_valid, dtype=torch.bool
            )
        if goal_effect_targets is not None:
            payload.update(
                {
                    key: torch.from_numpy(value)
                    for key, value in goal_effect_targets.items()
                }
            )
        return payload


def _read_camera_image(h5_file: Any, camera_name: str, timestep: int) -> np.ndarray:
    raw_path = f"observations/images/{camera_name}"
    if raw_path in h5_file:
        return np.asarray(h5_file[raw_path][timestep], dtype=np.uint8)
    encoded_path = f"{GRP_ENCODED_IMAGES}/{camera_name}"
    if encoded_path not in h5_file:
        raise KeyError(f"Camera {camera_name!r} not found as raw or encoded image data.")
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
            raise ValueError("deadzone_intent.enabled requires thresholds or threshold_json")
        path = Path(str(threshold_json))
        if not path.exists():
            raise FileNotFoundError(f"deadzone_intent threshold_json does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        thresholds = payload.get("deadzone_action", payload) if isinstance(payload, dict) else payload
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
    actions: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Build transition evidence while respecting handoff stop ownership."""

    total_steps = int(actions.shape[0])
    return compute_transition_direction_mask(
        actions=actions,
        thresholds=config["thresholds"],
        action_loss_mask=_read_optional_handoff_mask(
            h5_file, "handoff/action_loss_mask", total_steps, enabled=True
        ),
        tail_idle_mask=_read_optional_handoff_mask(
            h5_file, "handoff/tail_idle_mask", total_steps, enabled=True
        ),
        owner_automation=_read_optional_handoff_mask(
            h5_file, "handoff/owner_automation", total_steps, enabled=True
        ),
    )


def _terminal_swing_transition_anchor(
    *,
    transition_mask: np.ndarray,
    valid_starts: np.ndarray,
    total_steps: int,
    chunk_steps: int,
) -> int | None:
    """Return the last chunk-safe inactive-to-effective swing transition."""

    candidates = intersect_transition_starts(
        transition_mask,
        valid_starts,
        total_steps=total_steps,
        hold_horizon_steps=chunk_steps,
    )
    if not candidates.size:
        return None
    swing_candidates = candidates[transition_mask[candidates, 0].any(axis=1)]
    if not swing_candidates.size:
        return None
    return int(swing_candidates[-1])


def _read_real_transition_condition(
    h5_file: Any,
    *,
    timestep: int | None = None,
    enabled: bool,
) -> np.ndarray | None:
    if not enabled:
        return None
    path = f"conditions/{REAL_TRANSITION_CONDITION_KEY}"
    if path not in h5_file:
        raise KeyError(f"HDF5 episode is missing required condition dataset {path!r}.")
    dataset = h5_file[path]
    if dataset.ndim != 2 or int(dataset.shape[1]) != 2:
        raise ValueError(f"{path} must have shape (T, 2), got {tuple(dataset.shape)}")
    values = dataset[()] if timestep is None else dataset[int(timestep)]
    result = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{path} contains non-finite values")
    rows = result.reshape(-1, 2)
    target_code = rows[:, 0]
    goal_active = rows[:, 1]
    if not np.all(np.isin(goal_active, [0.0, 1.0])):
        raise ValueError(f"{path} goal_active must be 0 or 1")
    active = goal_active == 1.0
    if not np.all(np.isin(target_code[active], [-1.0, 1.0])):
        raise ValueError(f"{path} active target_side_code must be -1 or +1")
    if not np.all(target_code[~active] == 0.0):
        raise ValueError(f"{path} inactive target_side_code must be 0")
    return result


def _read_real_transition_cycle_phase(
    h5_file: Any,
    *,
    timestep: int | None = None,
    enabled: bool,
) -> np.ndarray | None:
    if not enabled:
        return None
    path = f"conditions/{CYCLE_PHASE_KEY}"
    if path not in h5_file:
        raise KeyError(f"HDF5 episode is missing required phase dataset {path!r}.")
    dataset = h5_file[path]
    if dataset.ndim != 2 or int(dataset.shape[1]) != 1:
        raise ValueError(f"{path} must have shape (T, 1), got {tuple(dataset.shape)}")
    values = dataset[()] if timestep is None else dataset[int(timestep)]
    result = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(result)) or not np.all(np.isin(result, [0.0, 1.0])):
        raise ValueError(f"{path} must contain only finite 0/1 values")
    return result


def _read_real_transition_excursion_observed(
    h5_file: Any,
    *,
    timestep: int | None = None,
    enabled: bool,
) -> np.ndarray | None:
    if not enabled:
        return None
    path = f"conditions/{EXCURSION_OBSERVED_KEY}"
    if path not in h5_file:
        raise KeyError(f"HDF5 episode is missing required state dataset {path!r}.")
    dataset = h5_file[path]
    if dataset.ndim != 2 or int(dataset.shape[1]) != 1:
        raise ValueError(f"{path} must have shape (T, 1), got {tuple(dataset.shape)}")
    values = dataset[()] if timestep is None else dataset[int(timestep)]
    result = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(result)) or not np.all(np.isin(result, [0.0, 1.0])):
        raise ValueError(f"{path} must contain only finite 0/1 values")
    return result


def _read_real_transition_return_commit(
    h5_file: Any,
    *,
    timestep: int | None = None,
    enabled: bool,
) -> np.ndarray | None:
    if not enabled:
        return None
    path = f"conditions/{RETURN_COMMIT_KEY}"
    if path not in h5_file:
        raise KeyError(
            f"HDF5 episode is missing required return-commit dataset {path!r}."
        )
    dataset = h5_file[path]
    if dataset.ndim != 2 or int(dataset.shape[1]) != 1:
        raise ValueError(f"{path} must have shape (T, 1), got {tuple(dataset.shape)}")
    values = dataset[()] if timestep is None else dataset[int(timestep)]
    result = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(result)) or not np.all(np.isin(result, [0.0, 1.0])):
        raise ValueError(f"{path} must contain only finite 0/1 values")
    return result


def _read_condition_chunk_valid_mask(
    h5_file: Any,
    *,
    timestep: int,
    action_chunk_size: int | None,
    enabled: bool,
) -> np.ndarray | None:
    if not enabled:
        return None
    path = "conditions/valid_mask"
    if path not in h5_file:
        raise KeyError(
            "Goal-conditioned transition training requires conditions/valid_mask."
        )
    dataset = h5_file[path]
    if dataset.ndim != 2 or int(dataset.shape[0]) != int(h5_file["action"].shape[0]):
        raise ValueError(
            f"{path} must have shape (T, chunk_steps), got {tuple(dataset.shape)}"
        )
    mask = np.asarray(dataset[int(timestep)], dtype=bool).reshape(-1)
    if action_chunk_size is not None and mask.size != int(action_chunk_size):
        raise ValueError(
            f"{path} width {mask.size} does not match configured ACT chunk_size "
            f"{int(action_chunk_size)}"
        )
    return mask


def _low_dim_slices(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    real_transition_condition_v1: np.ndarray | None,
    real_transition_excursion_observed_v1: np.ndarray | None,
    real_transition_cycle_phase_v1: np.ndarray | None,
    real_transition_return_commit_v1: np.ndarray | None,
    real_transition_action_primitive_v1: np.ndarray | None,
    real_transition_work_context_v1: np.ndarray | None,
    real_transition_task_state_v2: np.ndarray | None,
    low_dim_keys: list[str],
) -> dict[str, tuple[int, int]]:
    arrays = {
        "qpos": np.asarray(qpos),
        "qvel": np.asarray(qvel),
        REAL_TRANSITION_CONDITION_KEY: (
            None
            if real_transition_condition_v1 is None
            else np.asarray(real_transition_condition_v1)
        ),
        EXCURSION_OBSERVED_KEY: (
            None
            if real_transition_excursion_observed_v1 is None
            else np.asarray(real_transition_excursion_observed_v1)
        ),
        CYCLE_PHASE_KEY: (
            None
            if real_transition_cycle_phase_v1 is None
            else np.asarray(real_transition_cycle_phase_v1)
        ),
        RETURN_COMMIT_KEY: (
            None
            if real_transition_return_commit_v1 is None
            else np.asarray(real_transition_return_commit_v1)
        ),
        ACTION_PRIMITIVE_KEY: (
            None
            if real_transition_action_primitive_v1 is None
            else np.asarray(real_transition_action_primitive_v1)
        ),
        WORK_CONTEXT_KEY: (
            None
            if real_transition_work_context_v1 is None
            else np.asarray(real_transition_work_context_v1)
        ),
        TASK_STATE_V2_KEY: (
            None
            if real_transition_task_state_v2 is None
            else np.asarray(real_transition_task_state_v2)
        ),
    }
    result: dict[str, tuple[int, int]] = {}
    offset = 0
    for key in low_dim_keys:
        value = arrays[key]
        if value is None:
            raise KeyError(f"Missing low-dimensional input {key!r}")
        width = int(value.shape[-1]) if value.ndim > 0 else 1
        result[key] = (offset, offset + width)
        offset += width
    return result


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
        raise RuntimeError("opencv-python is required to decode JPEG training images") from exc
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
    split_manifest_path: str | Path | None = None,
    reuse_split: bool = True,
    low_dim_keys: list[str] | tuple[str, ...] | None = None,
    episode_ids: list[int] | None = None,
    action_chunk_size: int | None = None,
    image_transform: str = "none",
    deadzone_intent: dict[str, Any] | None = None,
    state_hold_transition: dict[str, Any] | None = None,
    condition_adherence_loss_train: dict[str, Any] | None = None,
    target_release_loss_train: dict[str, Any] | None = None,
    cycle_phase_loss_train: dict[str, Any] | None = None,
    excursion_observed_loss_train: dict[str, Any] | None = None,
    return_commit_loss_train: dict[str, Any] | None = None,
    action_primitive_islands: dict[str, Any] | None = None,
    work_return_context: dict[str, Any] | None = None,
    task_state_v2: dict[str, Any] | None = None,
    qvel_zero_state_hold_loss_train: dict[str, Any] | None = None,
    qvel_authority_loss_train: dict[str, Any] | None = None,
    factual_semantic_sampling_train: dict[str, Any] | None = None,
    goal_effect: dict[str, Any] | None = None,
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
    condition_adherence_cfg = resolve_condition_adherence_config(
        condition_adherence_loss_train
    )
    target_release_cfg = resolve_target_release_config(target_release_loss_train)
    cycle_phase_cfg = resolve_cycle_phase_loss_config(cycle_phase_loss_train)
    excursion_observed_cfg = resolve_excursion_observed_loss_config(
        excursion_observed_loss_train
    )
    return_commit_cfg = resolve_return_commit_loss_config(
        return_commit_loss_train
    )
    action_primitive_cfg = resolve_action_primitive_config(
        action_primitive_islands
    )
    work_return_cfg = resolve_work_return_context_config(work_return_context)
    task_state_v2_cfg = resolve_task_state_v2_config(task_state_v2)
    qvel_zero_state_hold_cfg = resolve_qvel_zero_state_hold_config(
        qvel_zero_state_hold_loss_train
    )
    qvel_authority_cfg = resolve_qvel_authority_config(
        qvel_authority_loss_train
    )
    factual_semantic_cfg = resolve_factual_semantic_sampling_config(
        factual_semantic_sampling_train
    )
    goal_effect_cfg = resolve_goal_effect_config(goal_effect)

    task_state_v2_manifest_rows: dict[int, dict[str, Any]] = {}
    if task_state_v2_cfg["enabled"]:
        if action_chunk_size is None:
            raise ValueError("task_state_v2 requires action_chunk_size")
        if int(task_state_v2_cfg["action_window_steps"]) != int(
            action_chunk_size
        ):
            raise ValueError(
                "task_state_v2.action_window_steps must equal action_chunk_size"
            )
        task_manifest = load_task_state_v2_manifest(
            str(task_state_v2_cfg["manifest_path"])
        )
        if int(task_manifest.get("chunk_steps", -1)) != int(
            task_state_v2_cfg["action_window_steps"]
        ):
            raise ValueError("task_state_v2 manifest chunk size mismatch")
        manifest_episodes = (
            Path(str(task_manifest.get("dataset_root", ""))) / "episodes"
        )
        if manifest_episodes.resolve() != dataset_dir.resolve():
            raise ValueError("task_state_v2 manifest dataset root mismatch")
        task_state_v2_manifest_rows = task_state_manifest_by_episode_fn(
            task_manifest
        )

    # discover available episode files
    discovered = [
        int(p.stem.split("_", 1)[1])
        for p in list_episodes(dataset_dir)
    ]
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
    condition_adherence_anchor_count = {}
    condition_adherence_anchor_index = {}
    target_release_start_count = {}
    cycle_phase_pre_start_count = {}
    cycle_phase_return_start_count = {}
    excursion_pre_start_count = {}
    excursion_apex_start_count = {}
    excursion_moving_start_count = {}
    return_commit_dig_start_count = {}
    return_commit_dig_non_swing_start_count = {}
    return_commit_onset_start_count = {}
    return_commit_effective_start_count = {}
    action_primitive_start_count: dict[int, dict[str, int]] = {}
    work_return_start_count: dict[int, dict[str, int]] = {}
    task_state_v2_start_count: dict[int, dict[str, int]] = {}
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
                    train_exclude_mask=_read_train_exclude_mask(f, length_info[ep_id]),
                    action_chunk_size=action_chunk_size,
                    action_loss_mask=action_loss_start_mask,
                    require_action_loss_in_chunk=bool(
                        deadzone_intent_cfg["require_action_loss_in_chunk"]
                    ),
                ).size
            )
            state_hold_transition_start_count[ep_id] = 0
            condition_adherence_anchor_count[ep_id] = 0
            condition_adherence_anchor_index[ep_id] = None
            target_release_start_count[ep_id] = 0
            cycle_phase_pre_start_count[ep_id] = 0
            cycle_phase_return_start_count[ep_id] = 0
            excursion_pre_start_count[ep_id] = 0
            excursion_apex_start_count[ep_id] = 0
            excursion_moving_start_count[ep_id] = 0
            return_commit_dig_start_count[ep_id] = 0
            return_commit_dig_non_swing_start_count[ep_id] = 0
            return_commit_onset_start_count[ep_id] = 0
            return_commit_effective_start_count[ep_id] = 0
            action_primitive_start_count[ep_id] = {
                name: 0 for name in PRIMITIVE_NAMES
            }
            work_return_start_count[ep_id] = {name: 0 for name in ROUTE_NAMES}
            task_state_v2_start_count[ep_id] = {
                name: 0 for name in TASK_STATE_V2_TIERS
            }
            if (
                state_hold_transition_cfg["enabled"]
                or condition_adherence_cfg["enabled"]
                or target_release_cfg["enabled"]
                or cycle_phase_cfg["enabled"]
                or excursion_observed_cfg["enabled"]
                or return_commit_cfg["enabled"]
                or action_primitive_cfg["enabled"]
                or work_return_cfg["enabled"]
                or task_state_v2_cfg["enabled"]
            ):
                metadata = dict(f["metadata"].attrs) if "metadata" in f else {}
                if not _bool_attr(metadata.get("action_prealigned", False)):
                    raise ValueError(
                        "transition supervision requires action_prealigned=true "
                        f"for episode {ep_id}"
                    )
                actions = np.asarray(f["action"][()], dtype=np.float32)
                valid_starts = _valid_start_indices(
                    total_steps=length_info[ep_id],
                    train_exclude_mask=_read_train_exclude_mask(
                        f, length_info[ep_id]
                    ),
                    action_chunk_size=action_chunk_size,
                )
                if state_hold_transition_cfg["enabled"]:
                    mask = _state_hold_transition_mask(
                        f, actions=actions, config=state_hold_transition_cfg
                    )
                    starts = intersect_transition_starts(
                        mask,
                        valid_starts,
                        total_steps=length_info[ep_id],
                        hold_horizon_steps=int(
                            state_hold_transition_cfg["hold_horizon_steps"]
                        ),
                    )
                    state_hold_transition_start_count[ep_id] = int(starts.size)
                if condition_adherence_cfg["enabled"]:
                    mask = _state_hold_transition_mask(
                        f, actions=actions, config=condition_adherence_cfg
                    )
                    anchor = _terminal_swing_transition_anchor(
                        transition_mask=mask,
                        valid_starts=valid_starts,
                        total_steps=length_info[ep_id],
                        chunk_steps=int(action_chunk_size or 1),
                    )
                    condition_adherence_anchor_count[ep_id] = int(anchor is not None)
                    condition_adherence_anchor_index[ep_id] = anchor
                if target_release_cfg["enabled"]:
                    target_valid_starts = _valid_start_indices(
                        total_steps=length_info[ep_id],
                        train_exclude_mask=_read_train_exclude_mask(
                            f, length_info[ep_id]
                        ),
                        action_chunk_size=action_chunk_size,
                        action_loss_mask=action_loss_start_mask,
                        require_action_loss_in_chunk=bool(
                            deadzone_intent_cfg["require_action_loss_in_chunk"]
                        ),
                    )
                    starts = target_release_candidate_indices(
                        qpos=np.asarray(
                            f["observations/qpos"][()], dtype=np.float32
                        ),
                        qvel=np.asarray(
                            f["observations/qvel"][()], dtype=np.float32
                        ),
                        actions=actions,
                        condition=np.asarray(
                            f[f"conditions/{REAL_TRANSITION_CONDITION_KEY}"][()],
                            dtype=np.float32,
                        ),
                        valid_starts=target_valid_starts,
                        condition_valid_mask=np.asarray(
                            f["conditions/valid_mask"][()], dtype=bool
                        ),
                        config=target_release_cfg,
                    )
                    target_release_start_count[ep_id] = int(starts.size)
                if cycle_phase_cfg["enabled"]:
                    if f"conditions/{CYCLE_PHASE_KEY}" not in f:
                        raise ValueError(
                            f"episode {ep_id} is missing conditions/{CYCLE_PHASE_KEY}"
                        )
                    if "conditions/cycle_phase_valid_mask" not in f:
                        raise ValueError(
                            "cycle phase loss requires "
                            "conditions/cycle_phase_valid_mask"
                        )
                    phase_starts = cycle_phase_candidate_indices(
                        actions=actions,
                        phase=np.asarray(
                            f[f"conditions/{CYCLE_PHASE_KEY}"][()],
                            dtype=np.float32,
                        ),
                        valid_starts=valid_starts,
                        phase_valid_mask=np.asarray(
                            f["conditions/cycle_phase_valid_mask"][()], dtype=bool
                        ),
                        config=cycle_phase_cfg,
                    )
                    cycle_phase_pre_start_count[ep_id] = int(
                        phase_starts["pre_positive"].size
                    )
                    cycle_phase_return_start_count[ep_id] = int(
                        phase_starts["return_negative"].size
                    )
                if excursion_observed_cfg["enabled"]:
                    for required_path in (
                        f"conditions/{EXCURSION_OBSERVED_KEY}",
                        f"conditions/{CYCLE_PHASE_KEY}",
                        "conditions/excursion_observed_valid_mask",
                    ):
                        if required_path not in f:
                            raise ValueError(
                                f"episode {ep_id} is missing {required_path}"
                            )
                    excursion_starts = excursion_observed_candidate_indices(
                        actions=actions,
                        qvel=np.asarray(
                            f["observations/qvel"][()], dtype=np.float32
                        ),
                        excursion_observed=np.asarray(
                            f[f"conditions/{EXCURSION_OBSERVED_KEY}"][()],
                            dtype=np.float32,
                        ),
                        return_phase=np.asarray(
                            f[f"conditions/{CYCLE_PHASE_KEY}"][()],
                            dtype=np.float32,
                        ),
                        valid_starts=valid_starts,
                        excursion_valid_mask=np.asarray(
                            f["conditions/excursion_observed_valid_mask"][()],
                            dtype=bool,
                        ),
                        config=excursion_observed_cfg,
                    )
                    excursion_pre_start_count[ep_id] = int(
                        excursion_starts["pre_positive"].size
                    )
                    excursion_apex_start_count[ep_id] = int(
                        excursion_starts["apex_negative"].size
                    )
                    excursion_moving_start_count[ep_id] = int(
                        excursion_starts["moving_negative"].size
                    )
                if return_commit_cfg["enabled"]:
                    for required_path in (
                        f"conditions/{RETURN_COMMIT_KEY}",
                        "conditions/return_commit_valid_mask",
                    ):
                        if required_path not in f:
                            raise ValueError(
                                f"episode {ep_id} is missing {required_path}"
                            )
                    commit_starts = return_commit_candidate_indices(
                        actions=actions,
                        return_commit=np.asarray(
                            f[f"conditions/{RETURN_COMMIT_KEY}"][()],
                            dtype=np.float32,
                        ),
                        valid_starts=valid_starts,
                        return_commit_valid_mask=np.asarray(
                            f["conditions/return_commit_valid_mask"][()],
                            dtype=bool,
                        ),
                        config=return_commit_cfg,
                    )
                    return_commit_dig_start_count[ep_id] = int(
                        commit_starts["dig_positive"].size
                    )
                    return_commit_dig_non_swing_start_count[ep_id] = int(
                        commit_starts["dig_non_swing_transition"].size
                    )
                    return_commit_onset_start_count[ep_id] = int(
                        commit_starts["return_onset"].size
                    )
                    return_commit_effective_start_count[ep_id] = int(
                        commit_starts["return_effective"].size
                    )
                if action_primitive_cfg["enabled"]:
                    primitive_starts = derive_action_primitive_islands(
                        actions,
                        positive_thresholds=action_primitive_cfg[
                            "positive_thresholds"
                        ],
                        negative_thresholds=action_primitive_cfg[
                            "negative_thresholds"
                        ],
                        action_window_steps=int(
                            action_primitive_cfg["action_window_steps"]
                        ),
                        valid_starts=valid_starts,
                    )
                    action_primitive_start_count[ep_id] = {
                        name: int(primitive_starts.candidate_starts[name].size)
                        for name in PRIMITIVE_NAMES
                    }
                if work_return_cfg["enabled"]:
                    route_starts = derive_work_return_context(
                        actions,
                        positive_thresholds=work_return_cfg[
                            "positive_thresholds"
                        ],
                        negative_thresholds=work_return_cfg[
                            "negative_thresholds"
                        ],
                        action_window_steps=int(
                            work_return_cfg["action_window_steps"]
                        ),
                        valid_starts=valid_starts,
                    )
                    work_return_start_count[ep_id] = {
                        "work": int(route_starts.work_starts.size),
                        "return": int(route_starts.return_starts.size),
                    }
                if task_state_v2_cfg["enabled"]:
                    if ep_id not in task_state_v2_manifest_rows:
                        raise ValueError(
                            f"task_state_v2 manifest is missing episode {ep_id}"
                        )
                    task_row = task_state_v2_manifest_rows[ep_id]
                    if int(task_row.get("n_rows", -1)) != length_info[ep_id]:
                        raise ValueError(
                            f"episode {ep_id} task_state_v2 row count changed"
                        )
                    task_starts = task_state_candidate_starts(
                        total_steps=length_info[ep_id],
                        work_complete_row=int(task_row["work_complete_row"]),
                        return_commit_row=int(task_row["return_commit_row"]),
                        action_window_steps=int(
                            task_state_v2_cfg["action_window_steps"]
                        ),
                        valid_starts=valid_starts,
                    ).by_name()
                    expected_starts = {
                        str(name): np.asarray(values, dtype=np.int64)
                        for name, values in dict(
                            task_row.get("candidate_starts", {})
                        ).items()
                    }
                    if set(expected_starts) != set(TASK_STATE_V2_TIERS) or any(
                        not np.array_equal(expected_starts[name], task_starts[name])
                        for name in TASK_STATE_V2_TIERS
                    ):
                        raise ValueError(
                            f"episode {ep_id} task_state_v2 population changed"
                        )
                    task_state_v2_start_count[ep_id] = {
                        name: int(values.size)
                        for name, values in task_starts.items()
                    }
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

    if split_manifest_path is not None:
        train_ids, val_ids, split_info = _resolve_manifest_episode_split(
            manifest_path=Path(split_manifest_path),
            available_episode_ids=available,
        )
    else:
        train_ids, val_ids, split_info = _resolve_episode_split(
            dataset_dir=dataset_dir,
            available_episode_ids=available,
            requested_num_episodes=int(num_episodes),
            split_seed=int(split_seed),
            train_split_ratio=float(train_split_ratio),
            split_path=None if split_path is None else Path(split_path),
            reuse_split=bool(reuse_split),
        )

    training_episode_ids = sorted(set(train_ids) | set(val_ids))
    max_episode_len = max(length_info[ep_id] for ep_id in training_episode_ids)
    target_episode_len = int(episode_len) if episode_len is not None else max_episode_len
    if max_episode_len > target_episode_len:
        raise ValueError(
            f"Dataset contains a train/validation episode of length {max_episode_len}, "
            f"but configured episode_len is only {target_episode_len}. "
            "Increase task.episode_len."
        )

    selected_low_dim_keys = _normalize_low_dim_keys(low_dim_keys)
    # v2 manifest-owned splits require train-only statistics. Keep the legacy
    # random-split path unchanged for checkpoint reproducibility.
    normalization_episode_ids = (
        train_ids if split_manifest_path is not None else available
    )
    if condition_adherence_cfg["enabled"]:
        required_ids = list(train_ids)
        if condition_adherence_cfg.get("scope") == "train_and_validation":
            required_ids.extend(val_ids)
        missing_condition_anchors = [
            ep_id for ep_id in required_ids
            if condition_adherence_anchor_count.get(ep_id, 0) != 1
        ]
        if missing_condition_anchors:
            raise ValueError(
                "condition adherence requires exactly one terminal swing anchor "
                "in every train episode; missing episode ids: "
                + ", ".join(str(ep_id) for ep_id in missing_condition_anchors[:20])
            )
    if target_release_cfg["enabled"]:
        required_ids = list(train_ids)
        if target_release_cfg.get("scope") == "train_and_validation":
            required_ids.extend(val_ids)
        missing_target_release = [
            ep_id
            for ep_id in required_ids
            if target_release_start_count.get(ep_id, 0) <= 0
        ]
        if missing_target_release:
            raise ValueError(
                "target release requires at least one train-supported decision "
                "start in every configured episode; missing episode ids: "
                + ", ".join(str(ep_id) for ep_id in missing_target_release[:20])
            )
    if cycle_phase_cfg["enabled"]:
        required_ids = list(train_ids)
        if cycle_phase_cfg.get("scope") == "train_and_validation":
            required_ids.extend(val_ids)
        missing_pre = [
            ep_id
            for ep_id in required_ids
            if cycle_phase_pre_start_count.get(ep_id, 0) <= 0
        ]
        missing_return = [
            ep_id
            for ep_id in required_ids
            if cycle_phase_return_start_count.get(ep_id, 0) <= 0
        ]
        if missing_pre or missing_return:
            raise ValueError(
                "cycle phase loss requires both positive pre-return and negative "
                "return starts in every configured episode; "
                f"missing_pre={missing_pre[:20]}, missing_return={missing_return[:20]}"
            )
    if excursion_observed_cfg["enabled"]:
        required_ids = list(train_ids)
        if excursion_observed_cfg.get("scope") == "train_and_validation":
            required_ids.extend(val_ids)
        missing_pre = [
            ep_id
            for ep_id in required_ids
            if excursion_pre_start_count.get(ep_id, 0) <= 0
        ]
        missing_post = [
            ep_id
            for ep_id in required_ids
            if (
                excursion_apex_start_count.get(ep_id, 0)
                + excursion_moving_start_count.get(ep_id, 0)
            )
            <= 0
        ]
        if missing_pre or missing_post:
            raise ValueError(
                "excursion observed loss requires positive pre-excursion and "
                "negative post-excursion starts in every configured episode; "
                f"missing_pre={missing_pre[:20]}, missing_post={missing_post[:20]}"
            )
    if return_commit_cfg["enabled"]:
        required_ids = list(train_ids)
        if return_commit_cfg.get("scope") == "train_and_validation":
            required_ids.extend(val_ids)
        missing_dig = [
            ep_id
            for ep_id in required_ids
            if (
                return_commit_dig_start_count.get(ep_id, 0)
                if return_commit_cfg["dig_candidate_key"] == "dig_positive"
                else return_commit_dig_non_swing_start_count.get(ep_id, 0)
            )
            <= 0
        ]
        missing_onset = [
            ep_id
            for ep_id in required_ids
            if return_commit_onset_start_count.get(ep_id, 0) != 1
        ]
        if missing_dig or missing_onset:
            raise ValueError(
                "return commit loss requires executable dig and exactly one "
                "return-onset start in every configured episode; "
                f"missing_dig={missing_dig[:20]}, missing_onset={missing_onset[:20]}"
            )
    if action_primitive_cfg["enabled"]:
        missing_primitives = {
            name: [
                ep_id
                for ep_id in [*train_ids, *val_ids]
                if action_primitive_start_count.get(ep_id, {}).get(name, 0) <= 0
            ]
            for name in PRIMITIVE_NAMES
        }
        missing_primitives = {
            name: values for name, values in missing_primitives.items() if values
        }
        if missing_primitives:
            raise ValueError(
                "oracle action primitive training requires a full chunk for every "
                f"primitive in every train/validation episode: {missing_primitives}"
            )
    if work_return_cfg["enabled"]:
        missing_routes = {
            route: [
                ep_id
                for ep_id in [*train_ids, *val_ids]
                if work_return_start_count.get(ep_id, {}).get(route, 0) <= 0
            ]
            for route in ROUTE_NAMES
        }
        missing_routes = {
            route: values for route, values in missing_routes.items() if values
        }
        if missing_routes:
            raise ValueError(
                "WORK/RETURN training requires both branches in every "
                f"train/validation episode: {missing_routes}"
            )
    if task_state_v2_cfg["enabled"]:
        missing_tiers = {
            tier: [
                ep_id
                for ep_id in [*train_ids, *val_ids]
                if task_state_v2_start_count.get(ep_id, {}).get(tier, 0) <= 0
            ]
            for tier in TASK_STATE_V2_TIERS
        }
        missing_tiers = {
            tier: values for tier, values in missing_tiers.items() if values
        }
        if missing_tiers:
            raise ValueError(
                "task_state_v2 requires all four factual tiers in every "
                f"train/validation episode: {missing_tiers}"
            )
    norm_stats = get_norm_stats(
        dataset_dir,
        num_episodes,
        episode_ids=normalization_episode_ids,
        low_dim_keys=selected_low_dim_keys,
        deadzone_intent=deadzone_intent,
        action_primitive_islands=action_primitive_islands,
        work_return_context=work_return_context,
        task_state_v2=task_state_v2,
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
        condition_adherence_loss=condition_adherence_loss_train,
        target_release_loss=target_release_loss_train,
        cycle_phase_loss=cycle_phase_loss_train,
        excursion_observed_loss=excursion_observed_loss_train,
        return_commit_loss=return_commit_loss_train,
        action_primitive_islands=action_primitive_islands,
        work_return_context=work_return_context,
        task_state_v2=task_state_v2,
        qvel_zero_state_hold_loss=qvel_zero_state_hold_loss_train,
        qvel_authority_loss=qvel_authority_loss_train,
        factual_semantic_sampling=factual_semantic_sampling_train,
        goal_effect=goal_effect,
    )
    condition_adherence_loss_validation = (
        condition_adherence_loss_train
        if condition_adherence_cfg.get("scope") == "train_and_validation"
        else None
    )
    target_release_loss_validation = (
        target_release_loss_train
        if target_release_cfg.get("scope") == "train_and_validation"
        else None
    )
    cycle_phase_loss_validation = (
        cycle_phase_loss_train
        if cycle_phase_cfg.get("scope") == "train_and_validation"
        else None
    )
    excursion_observed_loss_validation = (
        excursion_observed_loss_train
        if excursion_observed_cfg.get("scope") == "train_and_validation"
        else None
    )
    return_commit_loss_validation = (
        return_commit_loss_train
        if return_commit_cfg.get("scope") == "train_and_validation"
        else None
    )
    qvel_authority_loss_validation = (
        qvel_authority_loss_train
        if qvel_authority_cfg.get("scope") == "train_and_validation"
        else None
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
        condition_adherence_loss=condition_adherence_loss_validation,
        target_release_loss=target_release_loss_validation,
        cycle_phase_loss=cycle_phase_loss_validation,
        excursion_observed_loss=excursion_observed_loss_validation,
        return_commit_loss=return_commit_loss_validation,
        action_primitive_islands=action_primitive_islands,
        work_return_context=work_return_context,
        task_state_v2=task_state_v2,
        qvel_zero_state_hold_loss=qvel_zero_state_hold_loss_train,
        qvel_authority_loss=qvel_authority_loss_validation,
        factual_semantic_sampling=None,
        goal_effect=goal_effect,
    )

    split_info["dataset_max_episode_len"] = int(max_episode_len)
    split_info["loader_episode_len"] = int(target_episode_len)
    split_info["low_dim_keys"] = list(selected_low_dim_keys)
    split_info["low_dim_dim"] = int(norm_stats["proprio_dim"])
    split_info["action_chunk_size"] = None if action_chunk_size is None else int(action_chunk_size)
    split_info["image_transform"] = str(image_transform or "none")
    split_info["deadzone_intent_enabled"] = bool(
        deadzone_intent_cfg["enabled"]
    )
    split_info["state_hold_transition"] = {
        "enabled": bool(state_hold_transition_cfg["enabled"]),
        "probability": float(state_hold_transition_cfg["probability"]),
        "hold_horizon_steps": int(state_hold_transition_cfg["hold_horizon_steps"]),
        "append_samples_per_episode": int(
            state_hold_transition_cfg["append_samples_per_episode"]
        ),
        "start_count_by_episode": {
            int(ep_id): int(state_hold_transition_start_count.get(ep_id, 0))
            for ep_id in available
        },
    }
    split_info["condition_adherence_loss_train"] = {
        "enabled": bool(condition_adherence_cfg["enabled"]),
        "scope": condition_adherence_cfg.get("scope", "train_only"),
        "action_label_scope": condition_adherence_cfg.get(
            "action_label_scope", "anchor_only"
        ),
        "anchor_rule": str(condition_adherence_cfg["anchor_rule"]),
        "anchor_count_by_episode": {
            int(ep_id): int(condition_adherence_anchor_count.get(ep_id, 0))
            for ep_id in train_ids
        },
        "anchor_index_by_episode": {
            int(ep_id): condition_adherence_anchor_index.get(ep_id)
            for ep_id in train_ids
        },
        "anchor_count": int(
            sum(condition_adherence_anchor_count.get(ep_id, 0) for ep_id in train_ids)
        ),
        "validation_anchor_count": int(
            sum(condition_adherence_anchor_count.get(ep_id, 0) for ep_id in val_ids)
        ),
    }
    split_info["target_release_loss_train"] = {
        "enabled": bool(target_release_cfg["enabled"]),
        "scope": target_release_cfg.get("scope", "train_only"),
        "contract_path": target_release_cfg.get("contract_path"),
        "append_samples_per_episode": int(
            target_release_cfg["append_samples_per_episode"]
        ),
        "decision_qpos_range": list(target_release_cfg["decision_qpos_range"]),
        "action_window_steps": int(target_release_cfg["action_window_steps"]),
        "start_count_by_episode": {
            int(ep_id): int(target_release_start_count.get(ep_id, 0))
            for ep_id in available
        },
        "train_start_count": int(
            sum(target_release_start_count.get(ep_id, 0) for ep_id in train_ids)
        ),
        "validation_start_count": int(
            sum(target_release_start_count.get(ep_id, 0) for ep_id in val_ids)
        ),
    }
    split_info["cycle_phase_loss_train"] = {
        "enabled": bool(cycle_phase_cfg["enabled"]),
        "scope": cycle_phase_cfg.get("scope", "train_only"),
        "threshold_path": cycle_phase_cfg.get("threshold_path"),
        "append_samples_per_episode": int(
            cycle_phase_cfg["append_samples_per_episode"]
        ),
        "action_window_steps": int(cycle_phase_cfg["action_window_steps"]),
        "pre_positive_start_count_by_episode": {
            int(ep_id): int(cycle_phase_pre_start_count.get(ep_id, 0))
            for ep_id in available
        },
        "return_negative_start_count_by_episode": {
            int(ep_id): int(cycle_phase_return_start_count.get(ep_id, 0))
            for ep_id in available
        },
        "train_pre_positive_start_count": int(
            sum(cycle_phase_pre_start_count.get(ep_id, 0) for ep_id in train_ids)
        ),
        "train_return_negative_start_count": int(
            sum(cycle_phase_return_start_count.get(ep_id, 0) for ep_id in train_ids)
        ),
        "validation_pre_positive_start_count": int(
            sum(cycle_phase_pre_start_count.get(ep_id, 0) for ep_id in val_ids)
        ),
        "validation_return_negative_start_count": int(
            sum(cycle_phase_return_start_count.get(ep_id, 0) for ep_id in val_ids)
        ),
    }
    split_info["excursion_observed_loss_train"] = {
        "enabled": bool(excursion_observed_cfg["enabled"]),
        "scope": excursion_observed_cfg.get("scope", "train_only"),
        "threshold_path": excursion_observed_cfg.get("threshold_path"),
        "append_samples_per_episode": int(
            excursion_observed_cfg["append_samples_per_episode"]
        ),
        "action_window_steps": int(
            excursion_observed_cfg["action_window_steps"]
        ),
        "pre_positive_start_count_by_episode": {
            int(ep_id): int(excursion_pre_start_count.get(ep_id, 0))
            for ep_id in available
        },
        "apex_negative_start_count_by_episode": {
            int(ep_id): int(excursion_apex_start_count.get(ep_id, 0))
            for ep_id in available
        },
        "moving_negative_start_count_by_episode": {
            int(ep_id): int(excursion_moving_start_count.get(ep_id, 0))
            for ep_id in available
        },
        "train_pre_positive_start_count": int(
            sum(excursion_pre_start_count.get(ep_id, 0) for ep_id in train_ids)
        ),
        "train_apex_negative_start_count": int(
            sum(excursion_apex_start_count.get(ep_id, 0) for ep_id in train_ids)
        ),
        "train_moving_negative_start_count": int(
            sum(excursion_moving_start_count.get(ep_id, 0) for ep_id in train_ids)
        ),
        "validation_pre_positive_start_count": int(
            sum(excursion_pre_start_count.get(ep_id, 0) for ep_id in val_ids)
        ),
        "validation_apex_negative_start_count": int(
            sum(excursion_apex_start_count.get(ep_id, 0) for ep_id in val_ids)
        ),
        "validation_moving_negative_start_count": int(
            sum(excursion_moving_start_count.get(ep_id, 0) for ep_id in val_ids)
        ),
    }
    split_info["return_commit_loss_train"] = {
        "enabled": bool(return_commit_cfg["enabled"]),
        "scope": return_commit_cfg.get("scope", "train_only"),
        "threshold_path": return_commit_cfg.get("threshold_path"),
        "append_samples_per_episode": int(
            return_commit_cfg["append_samples_per_episode"]
        ),
        "action_window_steps": int(return_commit_cfg["action_window_steps"]),
        "dig_positive_start_count_by_episode": {
            int(ep_id): int(return_commit_dig_start_count.get(ep_id, 0))
            for ep_id in available
        },
        "dig_non_swing_transition_start_count_by_episode": {
            int(ep_id): int(
                return_commit_dig_non_swing_start_count.get(ep_id, 0)
            )
            for ep_id in available
        },
        "dig_candidate_mode": return_commit_cfg.get(
            "dig_candidate_mode", "swing_positive"
        ),
        "dig_candidate_key": return_commit_cfg.get(
            "dig_candidate_key", "dig_positive"
        ),
        "return_onset_start_count_by_episode": {
            int(ep_id): int(return_commit_onset_start_count.get(ep_id, 0))
            for ep_id in available
        },
        "return_effective_start_count_by_episode": {
            int(ep_id): int(return_commit_effective_start_count.get(ep_id, 0))
            for ep_id in available
        },
        "train_dig_positive_start_count": int(
            sum(return_commit_dig_start_count.get(ep_id, 0) for ep_id in train_ids)
        ),
        "train_dig_non_swing_transition_start_count": int(
            sum(
                return_commit_dig_non_swing_start_count.get(ep_id, 0)
                for ep_id in train_ids
            )
        ),
        "train_return_onset_start_count": int(
            sum(
                return_commit_onset_start_count.get(ep_id, 0)
                for ep_id in train_ids
            )
        ),
        "train_return_effective_start_count": int(
            sum(
                return_commit_effective_start_count.get(ep_id, 0)
                for ep_id in train_ids
            )
        ),
        "validation_dig_positive_start_count": int(
            sum(return_commit_dig_start_count.get(ep_id, 0) for ep_id in val_ids)
        ),
        "validation_dig_non_swing_transition_start_count": int(
            sum(
                return_commit_dig_non_swing_start_count.get(ep_id, 0)
                for ep_id in val_ids
            )
        ),
        "validation_return_onset_start_count": int(
            sum(
                return_commit_onset_start_count.get(ep_id, 0)
                for ep_id in val_ids
            )
        ),
        "validation_return_effective_start_count": int(
            sum(
                return_commit_effective_start_count.get(ep_id, 0)
                for ep_id in val_ids
            )
        ),
    }
    split_info["action_primitive_islands"] = {
        "enabled": bool(action_primitive_cfg["enabled"]),
        "condition_key": action_primitive_cfg["condition_key"],
        "primitive_names": list(action_primitive_cfg["primitive_names"]),
        "manifest_path": action_primitive_cfg.get("manifest_path"),
        "threshold_path": action_primitive_cfg.get("threshold_path"),
        "action_window_steps": int(
            action_primitive_cfg["action_window_steps"]
        ),
        "append_samples_per_episode": int(
            action_primitive_cfg["append_samples_per_episode"]
        ),
        "start_count_by_episode": {
            int(ep_id): {
                name: int(
                    action_primitive_start_count.get(ep_id, {}).get(name, 0)
                )
                for name in PRIMITIVE_NAMES
            }
            for ep_id in available
        },
        "train_start_count": {
            name: int(
                sum(
                    action_primitive_start_count.get(ep_id, {}).get(name, 0)
                    for ep_id in train_ids
                )
            )
            for name in PRIMITIVE_NAMES
        },
        "validation_start_count": {
            name: int(
                sum(
                    action_primitive_start_count.get(ep_id, {}).get(name, 0)
                    for ep_id in val_ids
                )
            )
            for name in PRIMITIVE_NAMES
        },
    }
    split_info["work_return_context"] = {
        "enabled": bool(work_return_cfg["enabled"]),
        "condition_key": work_return_cfg["condition_key"],
        "manifest_path": work_return_cfg.get("manifest_path"),
        "threshold_path": work_return_cfg.get("threshold_path"),
        "action_window_steps": int(work_return_cfg["action_window_steps"]),
        "append_samples_per_episode": int(
            work_return_cfg["append_samples_per_episode"]
        ),
        "start_count_by_episode": {
            int(ep_id): {
                route: int(
                    work_return_start_count.get(ep_id, {}).get(route, 0)
                )
                for route in ROUTE_NAMES
            }
            for ep_id in available
        },
        "train_start_count": {
            route: int(
                sum(
                    work_return_start_count.get(ep_id, {}).get(route, 0)
                    for ep_id in train_ids
                )
            )
            for route in ROUTE_NAMES
        },
        "validation_start_count": {
            route: int(
                sum(
                    work_return_start_count.get(ep_id, {}).get(route, 0)
                    for ep_id in val_ids
                )
            )
            for route in ROUTE_NAMES
        },
    }
    split_info["task_state_v2"] = {
        "enabled": bool(task_state_v2_cfg["enabled"]),
        "condition_key": task_state_v2_cfg["condition_key"],
        "manifest_path": task_state_v2_cfg.get("manifest_path"),
        "action_window_steps": int(task_state_v2_cfg["action_window_steps"]),
        "append_samples_per_episode": int(
            task_state_v2_cfg["append_samples_per_episode"]
        ),
        "tier_names": list(task_state_v2_cfg["tier_names"]),
        "start_count_by_episode": {
            int(ep_id): {
                tier: int(
                    task_state_v2_start_count.get(ep_id, {}).get(tier, 0)
                )
                for tier in TASK_STATE_V2_TIERS
            }
            for ep_id in available
        },
        "train_start_count": {
            tier: int(
                sum(
                    task_state_v2_start_count.get(ep_id, {}).get(tier, 0)
                    for ep_id in train_ids
                )
            )
            for tier in TASK_STATE_V2_TIERS
        },
        "validation_start_count": {
            tier: int(
                sum(
                    task_state_v2_start_count.get(ep_id, {}).get(tier, 0)
                    for ep_id in val_ids
                )
            )
            for tier in TASK_STATE_V2_TIERS
        },
    }
    split_info["qvel_zero_state_hold_loss_train"] = {
        "enabled": bool(qvel_zero_state_hold_cfg["enabled"]),
        "threshold_path": qvel_zero_state_hold_cfg.get("threshold_path"),
        "window_steps": int(qvel_zero_state_hold_cfg["window_steps"]),
        "weight": float(qvel_zero_state_hold_cfg["weight"]),
    }
    split_info["qvel_authority_loss_train"] = {
        "enabled": bool(qvel_authority_cfg["enabled"]),
        "manifest_path": qvel_authority_cfg.get("manifest_path"),
        "append_samples_per_episode": int(
            qvel_authority_cfg["append_samples_per_episode"]
        ),
        "action_window_steps": int(qvel_authority_cfg["action_window_steps"]),
        "weight": float(qvel_authority_cfg["weight"]),
    }
    split_info["factual_semantic_sampling_train"] = {
        "enabled": bool(factual_semantic_cfg["enabled"]),
        "scope": factual_semantic_cfg["scope"],
        "manifest_path": factual_semantic_cfg.get("manifest_path"),
        "append_samples_per_episode": int(
            factual_semantic_cfg["append_samples_per_episode"]
        ),
        "tier_names": list(factual_semantic_cfg["tier_names"]),
    }
    split_info["goal_effect"] = {
        "enabled": bool(goal_effect_cfg.enabled),
        "horizons": list(goal_effect_cfg.horizons),
        "unsupported_axes": list(goal_effect_cfg.unsupported_axes),
        "delta_scale": (
            np.asarray(norm_stats["goal_effect_delta_scale"], dtype=np.float32).tolist()
            if "goal_effect_delta_scale" in norm_stats
            else None
        ),
    }
    split_info["normalization_episode_ids"] = [
        int(ep_id) for ep_id in normalization_episode_ids
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

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size_val,   shuffle=True,  **loader_kw)

    return train_loader, val_loader, norm_stats, train_ds.is_real, split_info


def _resolve_manifest_episode_split(
    *,
    manifest_path: Path,
    available_episode_ids: list[int],
) -> tuple[list[int], list[int], dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"split_manifest_path does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "real_transition_cycle_split_manifest_v1":
        raise ValueError(
            "split_manifest_path must use schema "
            "'real_transition_cycle_split_manifest_v1'"
        )
    if payload.get("split_owner") != "source_block_before_cycle_materialization":
        raise ValueError(
            "split manifest owner must be source_block_before_cycle_materialization"
        )
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("split manifest must contain an episodes list")

    available = {int(ep_id) for ep_id in available_episode_ids}
    rows_by_id: dict[int, dict[str, Any]] = {}
    for raw in episodes:
        if not isinstance(raw, dict) or "episode_id" not in raw:
            raise ValueError("split manifest episode rows must include episode_id")
        episode_id = int(raw["episode_id"])
        if episode_id in rows_by_id:
            raise ValueError(f"split manifest repeats episode_id {episode_id}")
        rows_by_id[episode_id] = dict(raw)

    missing = sorted(available - set(rows_by_id))
    if missing:
        raise ValueError(
            "split manifest is missing train-ready episode ids: "
            + ", ".join(str(ep_id) for ep_id in missing)
        )
    train_ids = sorted(
        ep_id for ep_id in available if str(rows_by_id[ep_id].get("split")) == "train"
    )
    val_ids = sorted(
        ep_id for ep_id in available
        if str(rows_by_id[ep_id].get("split")) == "validation"
    )
    locked_test_ids = sorted(
        ep_id for ep_id in available
        if str(rows_by_id[ep_id].get("split")) == "locked_test"
    )
    known = set(train_ids) | set(val_ids) | set(locked_test_ids)
    unknown = sorted(available - known)
    if unknown:
        raise ValueError(
            "split manifest has unsupported split labels for episode ids: "
            + ", ".join(str(ep_id) for ep_id in unknown)
        )
    if not train_ids or not val_ids:
        raise ValueError(
            "split manifest must provide at least one train and one validation episode"
        )

    blocks_by_split = {
        split: {
            str(rows_by_id[ep_id].get("source_block_id", "")) for ep_id in ids
        }
        for split, ids in {
            "train": train_ids,
            "validation": val_ids,
            "locked_test": locked_test_ids,
        }.items()
    }
    if "" in set().union(*blocks_by_split.values()):
        raise ValueError("split manifest episode rows must include source_block_id")
    overlap = (
        (blocks_by_split["train"] & blocks_by_split["validation"])
        | (blocks_by_split["train"] & blocks_by_split["locked_test"])
        | (blocks_by_split["validation"] & blocks_by_split["locked_test"])
    )
    if overlap:
        raise ValueError(
            "source_block_id appears in multiple splits: " + ", ".join(sorted(overlap))
        )

    return train_ids, val_ids, {
        "schema_version": 1,
        "split_owner": str(payload.get("split_owner", "")),
        "split_manifest_path": str(manifest_path.resolve()),
        "available_episode_ids": sorted(available),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "locked_test_ids": locked_test_ids,
        "train_source_block_ids": sorted(blocks_by_split["train"]),
        "val_source_block_ids": sorted(blocks_by_split["validation"]),
        "locked_test_source_block_ids": sorted(blocks_by_split["locked_test"]),
        "reused_existing_split": True,
    }


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
        raise ValueError("Saved split file must contain non-empty train_ids and val_ids.")

    split_ids = set(train_ids) | set(val_ids)
    missing = sorted(split_ids - available)
    if missing:
        raise ValueError(
            "Saved split file references episode ids not available in the current dataset: "
            + ", ".join(str(ep_id) for ep_id in missing)
        )
