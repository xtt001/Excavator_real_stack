"""Build 20Hz ACT training datasets from repaired 50Hz real episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.data.bucket_repair import parse_episode_spec


DEFAULT_INPUT_DIR = Path("/media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_bucket_v1")
DEFAULT_OUTPUT_DIR = Path("/media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1")


def build_20hz_dataset(
    *,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    episode_ids: list[int] | None = None,
    target_hz: float = 20.0,
    action_label_offset_s: float = -0.02,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = episode_ids if episode_ids is not None else _discover_episode_ids(input_dir)
    rows: list[dict[str, Any]] = []
    for episode_id in ids:
        src = input_dir / f"episode_{episode_id}.hdf5"
        if not src.exists():
            continue
        dst = output_dir / src.name
        rows.append(
            build_20hz_episode(
                input_path=src,
                output_path=dst,
                target_hz=target_hz,
                action_label_offset_s=action_label_offset_s,
            )
        )
    summary = {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_hz": float(target_hz),
        "action_label_offset_s": float(action_label_offset_s),
        "episodes": rows,
    }
    with (output_dir / "resample_20hz_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_20hz_episode(
    *,
    input_path: str | Path,
    output_path: str | Path,
    target_hz: float = 20.0,
    action_label_offset_s: float = -0.02,
) -> dict[str, Any]:
    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as in_f:
        obs_idx, action_idx, manual_end = select_20hz_indices(
            in_f,
            target_hz=target_hz,
            action_label_offset_s=action_label_offset_s,
        )
        if obs_idx.size == 0:
            raise ValueError(f"No 20Hz samples selected for {src}")
        if dst.exists():
            dst.unlink()
        with h5py.File(dst, "w") as out_f:
            _copy_attrs(in_f, out_f)
            _copy_selected_episode(
                in_f,
                out_f,
                obs_idx=obs_idx,
                action_idx=action_idx,
                target_hz=target_hz,
                action_label_offset_s=action_label_offset_s,
                manual_end=manual_end,
                source_path=src,
            )
    duration_s = (obs_idx.size - 1) / float(target_hz) if obs_idx.size > 1 else 0.0
    return {
        "episode_id": _episode_id_num(src),
        "input_path": str(src),
        "output_path": str(dst),
        "source_steps": int(manual_end),
        "output_steps": int(obs_idx.size),
        "target_hz": float(target_hz),
        "duration_s": float(duration_s),
        "action_label_offset_s": float(action_label_offset_s),
        "first_source_index": int(obs_idx[0]),
        "last_source_index": int(obs_idx[-1]),
    }


def select_20hz_indices(
    f: h5py.File,
    *,
    target_hz: float,
    action_label_offset_s: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_steps = int(f["action"].shape[0])
    manual_end = _manual_end_index(f, n_steps=n_steps)
    if manual_end <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), manual_end
    obs_ts = _observation_timestamps(f)[:manual_end]
    action_ts = _action_timestamps(f)[:manual_end]
    valid = obs_ts > 0
    if np.count_nonzero(valid) < 2:
        obs_ts = np.arange(manual_end, dtype=np.int64) * int(1_000_000_000 / 50)
        valid = np.ones(manual_end, dtype=bool)

    # Use each FPV timestamp at most once. If image timestamps repeat because
    # recorder runs faster than camera, keep the first observation carrying
    # that frame.
    unique_ts: list[int] = []
    unique_indices: list[int] = []
    seen: set[int] = set()
    for idx, ts in enumerate(obs_ts):
        its = int(ts)
        if not valid[idx] or its in seen:
            continue
        seen.add(its)
        unique_ts.append(its)
        unique_indices.append(idx)
    if len(unique_ts) < 2:
        return np.asarray(unique_indices, dtype=np.int64), np.asarray(unique_indices, dtype=np.int64), manual_end

    unique_ts_arr = np.asarray(unique_ts, dtype=np.int64)
    unique_idx_arr = np.asarray(unique_indices, dtype=np.int64)
    step_ns = int(round(1_000_000_000.0 / float(target_hz)))
    targets = np.arange(unique_ts_arr[0], unique_ts_arr[-1] + 1, step_ns, dtype=np.int64)
    selected: list[int] = []
    cursor = 0
    for target in targets:
        while cursor < unique_ts_arr.size and unique_ts_arr[cursor] < target:
            cursor += 1
        if cursor >= unique_ts_arr.size:
            break
        selected.append(int(unique_idx_arr[cursor]))
        cursor += 1
    obs_idx = np.asarray(sorted(set(selected)), dtype=np.int64)
    if obs_idx.size == 0:
        return obs_idx, obs_idx.copy(), manual_end

    action_offset_ns = int(round(float(action_label_offset_s) * 1_000_000_000.0))
    action_idx = np.empty_like(obs_idx)
    action_ts_valid = np.asarray(action_ts, dtype=np.int64)
    for out_i, src_i in enumerate(obs_idx):
        target = int(obs_ts[src_i]) + action_offset_ns
        idx = int(np.searchsorted(action_ts_valid, target, side="right") - 1)
        action_idx[out_i] = min(max(idx, 0), manual_end - 1)
    return obs_idx, action_idx, manual_end


def _copy_selected_episode(
    in_f: h5py.File,
    out_f: h5py.File,
    *,
    obs_idx: np.ndarray,
    action_idx: np.ndarray,
    target_hz: float,
    action_label_offset_s: float,
    manual_end: int,
    source_path: Path,
) -> None:
    # Metadata.
    meta = out_f.require_group("metadata")
    if "metadata" in in_f:
        for key, value in in_f["metadata"].attrs.items():
            meta.attrs[key] = value
    meta.attrs["record_hz"] = float(target_hz)
    meta.attrs["control_hz"] = float(target_hz)
    meta.attrs["dt"] = float(1.0 / target_hz)
    meta.attrs["n_steps"] = int(obs_idx.size)
    meta.attrs["source_dataset_path"] = str(source_path)
    meta.attrs["source_manual_end_index"] = int(manual_end)
    meta.attrs["source_observation_indices"] = "diagnostics/source_observation_index"
    meta.attrs["source_action_indices"] = "diagnostics/source_action_index"
    meta.attrs["action_label_offset_s"] = float(action_label_offset_s)
    meta.attrs["action_prealigned"] = True
    meta.attrs["sampling_hz"] = float(target_hz)
    meta.attrs["excluded_go_home"] = True
    out_f.attrs["is_real"] = bool(in_f.attrs.get("is_real", True))

    _copy_dataset(in_f, out_f, "observations/qpos", obs_idx)
    _copy_dataset(in_f, out_f, "observations/qvel", obs_idx)
    if "observations/env_state" in in_f:
        _copy_dataset(in_f, out_f, "observations/env_state", obs_idx)
    _copy_images(in_f, out_f, obs_idx)
    _copy_dataset(in_f, out_f, "action", action_idx)
    if "rewards" in in_f:
        _copy_dataset(in_f, out_f, "rewards", obs_idx)
    if "timestamps/step_id" in in_f:
        _copy_dataset(in_f, out_f, "timestamps/step_id", obs_idx)
    if "timestamps/step_ns" in in_f:
        _copy_dataset(in_f, out_f, "timestamps/step_ns", obs_idx)
    if "action_source/type" in in_f:
        _copy_dataset(in_f, out_f, "action_source/type", action_idx)
    if "action_source/id" in in_f:
        _copy_dataset(in_f, out_f, "action_source/id", action_idx)
    if "diagnostics" in in_f:
        for name in in_f["diagnostics"]:
            _copy_dataset(in_f, out_f, f"diagnostics/{name}", obs_idx)
    diag = out_f.require_group("diagnostics")
    diag.create_dataset("source_observation_index", data=obs_idx.astype(np.int64))
    diag.create_dataset("source_action_index", data=action_idx.astype(np.int64))
    source_obs_ts = _observation_timestamps(in_f)
    if source_obs_ts.size:
        diag.create_dataset("source_observation_timestamp_ns", data=source_obs_ts[obs_idx].astype(np.int64))


def _copy_images(in_f: h5py.File, out_f: h5py.File, obs_idx: np.ndarray) -> None:
    for group_path in ("observations/images", "observations/encoded_images"):
        if group_path not in in_f:
            continue
        out_group = out_f.require_group(group_path)
        for name in in_f[group_path]:
            ds = in_f[f"{group_path}/{name}"]
            data = np.asarray(ds[()])[obs_idx]
            out = out_group.create_dataset(name, data=data, dtype=ds.dtype)
            for key, value in ds.attrs.items():
                out.attrs[key] = value


def _copy_dataset(in_f: h5py.File, out_f: h5py.File, path: str, indices: np.ndarray) -> None:
    if path not in in_f:
        return
    ds = in_f[path]
    parent_path, name = path.rsplit("/", 1) if "/" in path else ("", path)
    parent = out_f.require_group(parent_path) if parent_path else out_f
    if ds.shape and ds.shape[0] >= int(np.max(indices)) + 1:
        data = np.asarray(ds[()])[indices]
    else:
        data = ds[()]
    out = parent.create_dataset(name, data=data, dtype=ds.dtype)
    for key, value in ds.attrs.items():
        out.attrs[key] = value


def _copy_attrs(in_f: h5py.File, out_f: h5py.File) -> None:
    for key, value in in_f.attrs.items():
        out_f.attrs[key] = value


def _manual_end_index(f: h5py.File, *, n_steps: int) -> int:
    candidates: list[int] = []
    for path in ("diagnostics/go_home_requested", "diagnostics/go_home_running"):
        if path in f:
            idx = np.flatnonzero(np.asarray(f[path][()]).reshape(-1) > 0)
            if idx.size:
                candidates.append(int(idx[0]))
    return min(candidates) if candidates else int(n_steps)


def _observation_timestamps(f: h5py.File) -> np.ndarray:
    for path in (
        "diagnostics/image_timestamp_ns_fpv",
        "diagnostics/image_timestamp_ns",
        "diagnostics/joint_timestamp_ns",
        "timestamps/step_ns",
    ):
        if path in f:
            return np.asarray(f[path][()], dtype=np.int64).reshape(-1)
    n_steps = int(f["action"].shape[0])
    return np.arange(n_steps, dtype=np.int64) * int(1_000_000_000 / 50)


def _action_timestamps(f: h5py.File) -> np.ndarray:
    for path in (
        "diagnostics/action_sample_timestamp_ns",
        "diagnostics/action_send_timestamp_ns",
        "timestamps/step_ns",
    ):
        if path in f:
            return np.asarray(f[path][()], dtype=np.int64).reshape(-1)
    n_steps = int(f["action"].shape[0])
    return np.arange(n_steps, dtype=np.int64) * int(1_000_000_000 / 50)


def _discover_episode_ids(dataset_dir: Path) -> list[int]:
    ids: list[int] = []
    for path in dataset_dir.glob("episode_*.hdf5"):
        episode_id = _episode_id_num(path)
        if episode_id >= 0:
            ids.append(episode_id)
    return sorted(ids)


def _episode_id_num(path: Path) -> int:
    try:
        return int(path.stem.split("_", 1)[1])
    except Exception:
        return -1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build 20Hz real-excavator training HDF5 data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", default="26-65")
    parser.add_argument("--target-hz", type=float, default=20.0)
    parser.add_argument("--action-label-offset-s", type=float, default=-0.02)
    args = parser.parse_args(argv)
    summary = build_20hz_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        episode_ids=parse_episode_spec(args.episodes),
        target_hz=args.target_hz,
        action_label_offset_s=args.action_label_offset_s,
    )
    print(f"20Hz dataset summary written to {args.output_dir / 'resample_20hz_summary.json'}")
    print(f"Episodes: {len(summary['episodes'])}")


if __name__ == "__main__":
    main()
