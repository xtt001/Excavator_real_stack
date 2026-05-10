"""Dataset quality-control utilities for recorded HDF5 episodes."""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from testbed.data.hdf5_io import list_episodes, read_episode
from testbed.data.schema import (
    ATTR_IS_REAL,
    ATTR_ENV_STATE_ORDER,
    ATTR_EPISODE_ID,
    ATTR_HYDRAULIC_CYLINDER_AVAILABLE,
    ATTR_PLATFORM,
    ATTR_QPOS_UNITS,
    ATTR_QVEL_UNITS,
    DEFAULT_PLATFORM,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def run_dataset_qc(
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    short_episode_threshold: int = 50,
    profile: str = "real",
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir is not None else dataset_dir / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = str(profile).lower()
    if profile != "real":
        raise ValueError(f"Unsupported QC profile {profile!r}. Expected 'real'.")

    episode_paths = list_episodes(dataset_dir)
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found under {dataset_dir}")

    rows: list[dict[str, Any]] = []
    actions_all: list[np.ndarray] = []
    qpos_all: list[np.ndarray] = []
    qvel_all: list[np.ndarray] = []
    env_state_all: list[np.ndarray] = []
    env_state_orders: set[str] = set()
    unreadable_episode_ids: list[str] = []
    unreadable_episode_errors: dict[str, str] = {}

    short_episode_ids: list[str] = []
    missing_image_ids: list[str] = []
    missing_env_state_ids: list[str] = []
    missing_step_ns_ids: list[str] = []
    non_monotonic_step_ids: list[str] = []
    nan_or_inf_ids: list[str] = []
    shape_mismatch_ids: list[str] = []
    real_metadata_mismatch_ids: list[str] = []
    real_units_mismatch_ids: list[str] = []
    real_diagnostic_missing_ids: list[str] = []

    for path in episode_paths:
        episode_id = path.stem
        try:
            episode = read_episode(path)
        except Exception as exc:
            unreadable_episode_ids.append(episode_id)
            unreadable_episode_errors[episode_id] = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "episode_id": episode_id,
                    "path": str(path),
                    "n_steps": -1,
                    "success": 0,
                    "has_images": 0,
                    "has_env_state": 0,
                    "env_state_dim": 0,
                    "step_ids_monotonic": 0,
                    "has_step_ns": 0,
                    "action_dim": -1,
                    "qpos_dim": -1,
                    "qvel_dim": -1,
                    "timestamp": "",
                    "operator_id": "",
                    "session_id": "",
                    "profile": profile,
                    "is_real": "",
                    "platform": "",
                    "guard_trigger_rate": "",
                    "controller_ack_rate": "",
                    "warnings": "unreadable_episode",
                    "error": unreadable_episode_errors[episode_id],
                }
            )
            continue

        metadata = dict(episode.get("metadata", {}))
        episode_id = str(metadata.get(ATTR_EPISODE_ID, path.stem))
        actions = np.asarray(episode["actions"], dtype=np.float32)
        qpos = np.asarray(episode["qpos"], dtype=np.float32)
        qvel = np.asarray(episode["qvel"], dtype=np.float32)
        env_state_raw = episode.get("env_state")
        env_state = None if env_state_raw is None else np.asarray(env_state_raw, dtype=np.float32)
        step_ids = episode.get("step_ids")
        step_ns = episode.get("step_ns")
        images = episode.get("images", {})
        diagnostics = episode.get("diagnostics", {})
        success = bool(int(metadata.get("success", 0)))
        n_steps = int(len(actions))
        platform = str(metadata.get(ATTR_PLATFORM, ""))
        is_real = bool(metadata.get(ATTR_IS_REAL, episode.get("is_real", True)))

        warnings: list[str] = []
        if n_steps < short_episode_threshold:
            warnings.append("short_episode")
            short_episode_ids.append(episode_id)
        if not images:
            warnings.append("missing_images")
            missing_image_ids.append(episode_id)
        if step_ns is None:
            warnings.append("missing_step_ns")
            missing_step_ns_ids.append(episode_id)
        if step_ids is not None and len(step_ids) > 1 and np.any(np.diff(step_ids) <= 0):
            warnings.append("non_monotonic_step_id")
            non_monotonic_step_ids.append(episode_id)

        expected_len = n_steps
        if qpos.shape[0] != expected_len or qvel.shape[0] != expected_len:
            warnings.append("shape_mismatch")
            shape_mismatch_ids.append(episode_id)
        if env_state is not None and env_state.shape[0] != expected_len:
            warnings.append("shape_mismatch")
            if episode_id not in shape_mismatch_ids:
                shape_mismatch_ids.append(episode_id)
        real_contract_ok = (
            actions.ndim == 2
            and qpos.ndim == 2
            and qvel.ndim == 2
            and actions.shape[1] == 4
            and qpos.shape[1] == 4
            and qvel.shape[1] == 4
        )
        if not real_contract_ok:
            warnings.append("real_contract_shape_mismatch")
            if episode_id not in shape_mismatch_ids:
                shape_mismatch_ids.append(episode_id)
        if not is_real or platform != DEFAULT_PLATFORM:
            warnings.append("real_metadata_mismatch")
            if episode_id not in real_metadata_mismatch_ids:
                real_metadata_mismatch_ids.append(episode_id)
        hydraulic_available = metadata.get(ATTR_HYDRAULIC_CYLINDER_AVAILABLE, False)
        if str(hydraulic_available).lower() not in {"false", "0"}:
            warnings.append("hydraulic_cylinder_marked_available")
            if episode_id not in real_metadata_mismatch_ids:
                real_metadata_mismatch_ids.append(episode_id)
        if metadata.get(ATTR_QPOS_UNITS) != "rad" or metadata.get(ATTR_QVEL_UNITS) != "rad/s":
            warnings.append("real_units_mismatch")
            real_units_mismatch_ids.append(episode_id)
        required_diag = {
            "raw_action",
            "guard_triggered",
            "guard_reason",
            "controller_ack",
            "controller_fault_code",
            "controller_timestamp_ns",
            "commanded_action",
        }
        if not required_diag.issubset(set(diagnostics)):
            warnings.append("real_diagnostics_missing")
            real_diagnostic_missing_ids.append(episode_id)

        arrays_to_check = [actions, qpos, qvel]
        if env_state is not None:
            arrays_to_check.append(env_state)
        if any(np.isnan(arr).any() or np.isinf(arr).any() for arr in arrays_to_check):
            warnings.append("nan_or_inf")
            nan_or_inf_ids.append(episode_id)

        env_state_order = metadata.get(ATTR_ENV_STATE_ORDER, "")
        if env_state_order:
            env_state_orders.add(str(env_state_order))

        actions_all.append(actions)
        qpos_all.append(qpos)
        qvel_all.append(qvel)
        if env_state is not None:
            env_state_all.append(env_state)

        rows.append(
            {
                "episode_id": episode_id,
                "path": str(path),
                "n_steps": n_steps,
                "success": int(success),
                "has_images": int(bool(images)),
                "has_env_state": int(env_state is not None),
                "env_state_dim": 0 if env_state is None else int(env_state.shape[1]),
                "step_ids_monotonic": int("non_monotonic_step_id" not in warnings),
                "has_step_ns": int(step_ns is not None),
                "action_dim": int(actions.shape[1]) if actions.ndim == 2 else -1,
                "qpos_dim": int(qpos.shape[1]) if qpos.ndim == 2 else -1,
                "qvel_dim": int(qvel.shape[1]) if qvel.ndim == 2 else -1,
                "timestamp": str(metadata.get("timestamp", "")),
                "operator_id": str(metadata.get("operator_id", "")),
                "session_id": str(metadata.get("session_id", "")),
                "profile": profile,
                "is_real": int(is_real),
                "platform": platform,
                "guard_trigger_rate": _optional_rate(diagnostics.get("guard_triggered")),
                "controller_ack_rate": _optional_rate(diagnostics.get("controller_ack")),
                "warnings": ";".join(warnings),
                "error": "",
            }
        )

    if not actions_all:
        raise RuntimeError(
            "No readable episodes were found for dataset QC. "
            f"Unreadable episodes: {', '.join(unreadable_episode_ids) if unreadable_episode_ids else 'none'}"
        )

    actions_cat = np.concatenate(actions_all, axis=0)
    qpos_cat = np.concatenate(qpos_all, axis=0)
    qvel_cat = np.concatenate(qvel_all, axis=0)
    env_state_cat = np.concatenate(env_state_all, axis=0) if env_state_all else None
    lengths = np.array([int(row["n_steps"]) for row in rows], dtype=np.int32)
    success_values = np.array([int(row["success"]) for row in rows], dtype=np.int32)

    summary = {
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "profile": profile,
        "n_episodes": len(rows),
        "n_success": int(success_values.sum()),
        "success_rate": float(success_values.mean()) if len(success_values) > 0 else 0.0,
        "episode_length": _series_stats(lengths.reshape(-1, 1)),
        "stats": {
            "action": _series_stats(actions_cat),
            "qpos": _series_stats(qpos_cat),
            "qvel": _series_stats(qvel_cat),
            "env_state": None if env_state_cat is None else _series_stats(env_state_cat),
        },
        "warnings": {
            "unreadable_episode_ids": unreadable_episode_ids,
            "unreadable_episode_errors": unreadable_episode_errors,
            "short_episode_ids": short_episode_ids,
            "missing_image_ids": missing_image_ids,
            "missing_env_state_ids": missing_env_state_ids,
            "missing_step_ns_ids": missing_step_ns_ids,
            "non_monotonic_step_ids": non_monotonic_step_ids,
            "nan_or_inf_ids": nan_or_inf_ids,
            "shape_mismatch_ids": shape_mismatch_ids,
            "real_metadata_mismatch_ids": real_metadata_mismatch_ids,
            "real_units_mismatch_ids": real_units_mismatch_ids,
            "real_diagnostic_missing_ids": real_diagnostic_missing_ids,
        },
        "env_state_order_values": sorted(env_state_orders),
        "env_state_order_consistent": len(env_state_orders) <= 1,
    }

    summary_path = output_dir / "summary.json"
    episodes_csv_path = output_dir / "episodes.csv"
    _write_json(summary_path, summary)
    _write_csv(episodes_csv_path, rows)
    _plot_episode_length_hist(lengths, output_dir / "episode_length_hist.png")
    _plot_action_distribution(actions_cat, output_dir / "action_distribution.png")
    _plot_state_ranges(
        qpos=qpos_cat,
        qvel=qvel_cat,
        env_state=env_state_cat,
        path=output_dir / "state_ranges.png",
    )
    return {
        "summary_path": str(summary_path),
        "episodes_csv_path": str(episodes_csv_path),
        "output_dir": str(output_dir),
        "summary": summary,
    }


def _series_stats(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array, dtype=np.float64)
    return {
        "shape": list(array.shape),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
    }


def _optional_rate(value: Any) -> float | str:
    if value is None:
        return ""
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return ""
    return float(arr.mean())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_episode_length_hist(lengths: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = min(20, max(5, len(lengths)))
    ax.hist(lengths, bins=bins, color="#4c78a8", edgecolor="white")
    ax.set_title("Episode Length Distribution")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_action_distribution(actions: np.ndarray, path: Path) -> None:
    num_dims = actions.shape[1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(7, max(3, 2 * num_dims)), squeeze=False)
    for dim in range(num_dims):
        ax = axes[dim, 0]
        ax.hist(actions[:, dim], bins=40, color="#f58518", edgecolor="white")
        ax.set_title(f"Action dim {dim}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_state_ranges(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    env_state: np.ndarray | None,
    path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    _plot_range_panel(axes[0], qpos, "qpos")
    _plot_range_panel(axes[1], qvel, "qvel")
    if env_state is None:
        axes[2].set_title("env_state")
        axes[2].text(0.5, 0.5, "missing in dataset", ha="center", va="center")
        axes[2].set_axis_off()
    else:
        _plot_range_panel(axes[2], env_state, "env_state")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_range_panel(ax, array: np.ndarray, title: str) -> None:
    mins = np.min(array, axis=0)
    maxs = np.max(array, axis=0)
    means = np.mean(array, axis=0)
    xs = np.arange(array.shape[1])
    ax.fill_between(xs, mins, maxs, color="#72b7b2", alpha=0.35, label="min/max")
    ax.plot(xs, means, color="#54a24b", linewidth=2, label="mean")
    ax.set_title(title)
    ax.set_xlabel("Dimension")
    ax.legend(loc="best")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
