"""
Low-level HDF5 read/write for demo episodes.

Writes the real-excavator episode schema used by record/QC/train.
Optional fields let the same writer handle mock, noop, and future hardware
adapter runs without requiring ROS or CAN at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.data.schema import (
    ATTR_IS_REAL,
    ATTR_SCHEMA_VERSION,
    DS_ACTION,
    DS_QPOS,
    DS_QVEL,
    DS_REWARDS,
    DS_ENV_STATE,
    DS_STEP_ID,
    DS_STEP_NS,
    DS_ACTION_SRC_TYPE,
    DS_ACTION_SRC_ID,
    GRP_DIAGNOSTICS,
    GRP_METADATA,
    GRP_TIMESTAMPS,
    GRP_ACTION_SOURCE,
    SCHEMA_VERSION,
)


# ─── Write ────────────────────────────────────────────────────────────────────

def write_episode(
    path: str | Path,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    actions: np.ndarray,
    images: dict[str, np.ndarray] | None = None,
    rewards: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    compress: bool = True,
    # ── Optional per-step additions ─────────────────────────────────────────
    env_state: np.ndarray | None = None,          # (T, M) float32
    step_ids: np.ndarray | None = None,           # (T,) int64
    step_ns: np.ndarray | None = None,            # (T,) int64
    action_src_types: list[str] | None = None,    # (T,) str
    action_src_ids: list[str] | None = None,      # (T,) str
    diagnostics: dict[str, Any] | None = None,    # optional per-step diagnostics
) -> None:
    """
    Write one demonstration episode to an HDF5 file.

    env_state, timestamps, action source, and diagnostics are optional so
    mock/noop runs can stay lightweight.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {"compression": "lzf"} if compress else {}
    str_dtype = h5py.special_dtype(vlen=str)

    with h5py.File(path, "w") as f:
        # ── metadata ─────────────────────────────────────────────────────────
        is_real = bool(metadata.get(ATTR_IS_REAL, True)) if metadata else True
        meta = f.create_group(GRP_METADATA)
        meta.attrs[ATTR_SCHEMA_VERSION] = SCHEMA_VERSION
        meta.attrs[ATTR_IS_REAL] = is_real
        if metadata:
            for k, v in metadata.items():
                if v is None:
                    continue
                meta.attrs[k] = v

        # ── root-level attrs for quick discovery ────────────────────────────
        f.attrs[ATTR_IS_REAL] = is_real

        # ── observations ─────────────────────────────────────────────────────
        obs_grp = f.create_group("observations")
        obs_grp.create_dataset("qpos", data=qpos.astype(np.float32))
        obs_grp.create_dataset("qvel", data=qvel.astype(np.float32))

        if env_state is not None:
            obs_grp.create_dataset(
                "env_state", data=env_state.astype(np.float32)
            )

        if images:
            img_grp = obs_grp.create_group("images")
            for cam, arr in images.items():
                img_grp.create_dataset(
                    cam, data=arr.astype(np.uint8), **kwargs
                )

        # ── action ───────────────────────────────────────────────────────────
        f.create_dataset("action", data=actions.astype(np.float32))

        # ── rewards ──────────────────────────────────────────────────────────
        if rewards is not None:
            f.create_dataset("rewards", data=rewards.astype(np.float32))

        # ── timestamps ──────────────────────────────────────────────────────
        if step_ids is not None or step_ns is not None:
            ts_grp = f.create_group(GRP_TIMESTAMPS)
            if step_ids is not None:
                ts_grp.create_dataset("step_id", data=np.asarray(step_ids, dtype=np.int64))
            if step_ns is not None:
                ts_grp.create_dataset("step_ns", data=np.asarray(step_ns, dtype=np.int64))

        # ── action_source ───────────────────────────────────────────────────
        if action_src_types is not None or action_src_ids is not None:
            src_grp = f.create_group(GRP_ACTION_SOURCE)
            if action_src_types is not None:
                ds = src_grp.create_dataset("type", (len(action_src_types),), dtype=str_dtype)
                for i, s in enumerate(action_src_types):
                    ds[i] = s
            if action_src_ids is not None:
                ds = src_grp.create_dataset("id", (len(action_src_ids),), dtype=str_dtype)
                for i, s in enumerate(action_src_ids):
                    ds[i] = s

        # ── diagnostics (optional; real-world collection path) ──────────────
        if diagnostics:
            diag_grp = f.create_group(GRP_DIAGNOSTICS)
            for name, value in diagnostics.items():
                if value is None:
                    continue
                _write_optional_dataset(diag_grp, str(name), value, str_dtype=str_dtype)


def read_episode(path: str | Path) -> dict[str, Any]:
    """
    Read a full real-excavator episode from HDF5.

    Returns
    -------
    {
      "qpos":             (T, Nq) float32,
      "qvel":             (T, Nq) float32,
      "actions":          (T, Na) float32,
      "images":           {cam: (T, H, W, 3) uint8},
      "rewards":          (T,) float32 | None,
      "env_state":        (T, M) float32 | None,
      "step_ids":         (T,) int64 | None,
      "step_ns":          (T,) int64 | None,
      "action_src_types": list[str] | None,
      "action_src_ids":   list[str] | None,
      "diagnostics":      dict[str, np.ndarray | list[str]],
      "metadata":         dict,
      "is_real":          bool,
    }
    """
    path = Path(path)
    result: dict[str, Any] = {}

    with h5py.File(path, "r") as f:
        result["qpos"]    = f[DS_QPOS][()].astype(np.float32)
        result["qvel"]    = f[DS_QVEL][()].astype(np.float32)
        result["actions"] = f[DS_ACTION][()].astype(np.float32)

        # images
        images = {}
        if "observations/images" in f:
            for cam in f["observations/images"]:
                images[cam] = f[f"observations/images/{cam}"][()]
        result["images"] = images

        # rewards (optional)
        result["rewards"] = f[DS_REWARDS][()] if DS_REWARDS in f else None

        # env_state
        result["env_state"] = (
            f[DS_ENV_STATE][()].astype(np.float32)
            if DS_ENV_STATE in f else None
        )

        # timestamps
        result["step_ids"] = f[DS_STEP_ID][()] if DS_STEP_ID in f else None
        result["step_ns"]  = f[DS_STEP_NS][()] if DS_STEP_NS in f else None

        # action_source
        if DS_ACTION_SRC_TYPE in f:
            result["action_src_types"] = [
                s.decode() if isinstance(s, bytes) else s
                for s in f[DS_ACTION_SRC_TYPE][()]
            ]
        else:
            result["action_src_types"] = None

        if DS_ACTION_SRC_ID in f:
            result["action_src_ids"] = [
                s.decode() if isinstance(s, bytes) else s
                for s in f[DS_ACTION_SRC_ID][()]
            ]
        else:
            result["action_src_ids"] = None

        # diagnostics
        diagnostics: dict[str, Any] = {}
        if GRP_DIAGNOSTICS in f:
            for name in f[GRP_DIAGNOSTICS]:
                diagnostics[name] = _read_optional_dataset(f[GRP_DIAGNOSTICS][name])
        result["diagnostics"] = diagnostics

        # metadata
        meta: dict[str, Any] = {}
        if GRP_METADATA in f:
            meta.update(dict(f[GRP_METADATA].attrs))
        meta.update(dict(f.attrs))
        result["metadata"] = meta
        result["is_real"] = bool(meta.get(ATTR_IS_REAL, True))

    return result


def _write_optional_dataset(
    group: h5py.Group,
    name: str,
    value: Any,
    *,
    str_dtype: Any,
) -> None:
    if isinstance(value, str):
        ds = group.create_dataset(name, (1,), dtype=str_dtype)
        ds[0] = value
        return

    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S", "O"}:
        ds = group.create_dataset(name, arr.shape, dtype=str_dtype)
        for index in np.ndindex(arr.shape):
            ds[index] = _decode_text(arr[index])
        return

    group.create_dataset(name, data=arr)


def _read_optional_dataset(dataset: h5py.Dataset) -> np.ndarray | list[str]:
    data = dataset[()]
    arr = np.asarray(data)
    if arr.dtype.kind in {"S", "O", "U"}:
        return [_decode_text(item) for item in arr.reshape(-1)]
    return data


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.bytes_):
        return value.decode()
    return str(value)


# ─── Discovery ────────────────────────────────────────────────────────────────

def list_episodes(dataset_dir: str | Path) -> list[Path]:
    """
    Return sorted list of episode_N.hdf5 paths in a dataset directory.
    Only files matching the pattern episode_<int>.hdf5 are returned.
    """
    dataset_dir = Path(dataset_dir)
    eps = []
    for p in dataset_dir.glob("episode_*.hdf5"):
        try:
            int(p.stem.split("_", 1)[1])
            eps.append(p)
        except (IndexError, ValueError):
            continue
    return sorted(eps, key=lambda p: int(p.stem.split("_", 1)[1]))


def episode_id_from_path(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])
