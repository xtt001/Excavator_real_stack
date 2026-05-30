#!/usr/bin/env python3
"""Analyze go-home direction signs from one recorded HDF5 episode."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np


AXES = ("swing", "boom", "stick", "bucket")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Episode HDF5 path. Defaults to the newest episode under --dataset-dir.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/media/mundane/EXTERNAL_USB/real_teleop_v1"),
    )
    parser.add_argument("--min-command", type=float, default=1e-4)
    parser.add_argument("--bad-slope-eps", type=float, default=2e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episode_path = args.file or _latest_episode(args.dataset_dir)
    print(f"[go-home-direction] file={episode_path}")
    with h5py.File(episode_path, "r") as h5:
        _analyze(
            h5,
            min_command=float(args.min_command),
            bad_slope_eps=float(args.bad_slope_eps),
        )
    return 0


def _latest_episode(dataset_dir: Path) -> Path:
    candidates = sorted(
        dataset_dir.glob("**/*.hdf5"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no HDF5 episodes under {dataset_dir}")
    return candidates[-1]


def _analyze(h5: h5py.File, *, min_command: float, bad_slope_eps: float) -> None:
    required = [
        "diagnostics/go_home_running",
        "diagnostics/go_home_error",
        "diagnostics/go_home_commanded_action",
        "observations/qvel",
    ]
    for key in required:
        if key not in h5:
            raise KeyError(f"missing dataset {key!r}")

    running = h5["diagnostics/go_home_running"][:].astype(bool)
    idx = np.flatnonzero(running)
    if len(idx) < 2:
        raise RuntimeError("episode has fewer than 2 go-home samples")

    err = h5["diagnostics/go_home_error"][:][idx].astype(np.float64)
    cmd = h5["diagnostics/go_home_commanded_action"][:][idx].astype(np.float64)
    qvel = h5["observations/qvel"][:][idx].astype(np.float64)
    active = _dataset_or_default(
        h5,
        "diagnostics/go_home_axis_active",
        np.ones_like(cmd, dtype=np.int32),
    )[idx].astype(bool)
    control_signs = _dataset_or_default(
        h5,
        "diagnostics/go_home_control_signs",
        np.ones_like(cmd, dtype=np.float32),
    )[idx].astype(np.float64)

    sent = _dataset_or_default(h5, "diagnostics/commanded_action", cmd)[idx].astype(
        np.float64
    )
    mismatch = np.max(np.abs(sent - cmd), axis=0)

    print(f"[go-home-direction] go_home_samples={len(idx)}")
    print("[go-home-direction] max_abs(sent - go_home_cmd)=" + _fmt_vec(mismatch))
    print(
        "axis,current_sign,active_samples,cmd_mean,qvel_mean,err_first,err_last,"
        "abs_err_delta,mean_d_abs,recommendation"
    )

    abs_err = np.abs(err)
    d_abs = np.diff(abs_err, axis=0)
    for axis_idx, axis_name in enumerate(AXES):
        axis_mask = active[:-1, axis_idx] & (
            np.abs(cmd[:-1, axis_idx]) > min_command
        )
        active_samples = int(np.sum(axis_mask))
        if active_samples == 0:
            print(
                f"{axis_name},{_last_sign(control_signs, axis_idx):+.0f},0,"
                "nan,nan,nan,nan,nan,nan,no-command"
            )
            continue

        sample_idx = np.flatnonzero(axis_mask)
        first = int(sample_idx[0])
        last = int(sample_idx[-1] + 1)
        cmd_mean = float(np.mean(cmd[:-1, axis_idx][axis_mask]))
        qvel_mean = float(np.mean(qvel[:-1, axis_idx][axis_mask]))
        err_first = float(err[first, axis_idx])
        err_last = float(err[last, axis_idx])
        abs_err_delta = float(abs(err_last) - abs(err_first))
        mean_d_abs = float(np.mean(d_abs[:, axis_idx][axis_mask]))
        recommendation = "flip" if mean_d_abs > bad_slope_eps else "keep"
        print(
            f"{axis_name},{_last_sign(control_signs, axis_idx):+.0f},"
            f"{active_samples},{cmd_mean:+.4f},{qvel_mean:+.5f},"
            f"{err_first:+.5f},{err_last:+.5f},{abs_err_delta:+.5f},"
            f"{mean_d_abs:+.6f},{recommendation}"
        )


def _dataset_or_default(h5: h5py.File, key: str, default: np.ndarray) -> np.ndarray:
    if key not in h5:
        return default
    return h5[key][:]


def _last_sign(values: np.ndarray, axis_idx: int) -> float:
    return float(values[-1, axis_idx])


def _fmt_vec(values: Any) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ",".join(f"{float(v):.4f}" for v in arr) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
