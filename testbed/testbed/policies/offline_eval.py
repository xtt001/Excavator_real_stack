"""Offline policy replay and action-distribution diagnostics."""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from testbed.data.image_transforms import IMAGE_TRANSFORM_CHOICES, build_image_transform


AXIS_NAMES = ("swing", "boom", "stick", "bucket")


def load_train_ready_episode_ids(manifest_path: str | Path) -> list[str]:
    """Return train-ready episode ids from a QC manifest."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ids = manifest.get("train_ready_episode_ids", [])
    if not ids:
        raise ValueError(f"manifest has no train_ready_episode_ids: {path}")
    return [normalize_episode_id(item) for item in ids]


def normalize_episode_id(value: str | int) -> str:
    text = str(value).strip()
    if text.startswith("episode_"):
        return text
    return f"episode_{int(text)}"


def episode_path(dataset_dir: str | Path, episode_id: str | int) -> Path:
    return Path(dataset_dir) / f"{normalize_episode_id(episode_id)}.hdf5"


def compute_episode_features(path: str | Path) -> np.ndarray:
    """Compute robust selection features for one HDF5 episode."""

    import h5py

    with h5py.File(path, "r") as f:
        action = np.asarray(f["action"][()], dtype=np.float32)
        qpos = np.asarray(f["observations/qpos"][()], dtype=np.float32)

    if action.ndim != 2 or action.shape[1] != 4:
        raise ValueError(f"action must have shape (T, 4), got {action.shape}")
    if qpos.ndim != 2 or qpos.shape[1] != 4:
        raise ValueError(f"qpos must have shape (T, 4), got {qpos.shape}")

    abs_action = np.abs(action.astype(np.float64))
    qpos64 = qpos.astype(np.float64)
    return np.concatenate(
        [
            np.asarray([float(action.shape[0])], dtype=np.float64),
            np.mean(abs_action, axis=0),
            np.percentile(abs_action, 95, axis=0),
            np.std(action.astype(np.float64), axis=0),
            np.mean(qpos64, axis=0),
            np.std(qpos64, axis=0),
            np.ptp(qpos64, axis=0),
        ]
    ).astype(np.float64)


def load_episode_features(
    dataset_dir: str | Path,
    episode_ids: Iterable[str | int],
) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for episode_id in episode_ids:
        normalized = normalize_episode_id(episode_id)
        features[normalized] = compute_episode_features(
            episode_path(dataset_dir, normalized)
        )
    return features


def select_representative_episode(
    episode_ids: Iterable[str | int],
    features: dict[str, np.ndarray],
) -> tuple[str, list[dict[str, Any]]]:
    """Select the train-ready episode closest to the robust feature median."""

    ordered_ids = [normalize_episode_id(item) for item in episode_ids]
    if not ordered_ids:
        raise ValueError("episode_ids must not be empty")

    missing = [episode_id for episode_id in ordered_ids if episode_id not in features]
    if missing:
        raise KeyError(f"missing features for episode ids: {missing}")

    matrix = np.stack([np.asarray(features[episode_id], dtype=np.float64) for episode_id in ordered_ids])
    center = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)
    iqr = q75 - q25
    std = np.nanstd(matrix, axis=0)
    scale = np.where(iqr > 1e-9, iqr, np.where(std > 1e-9, std, 1.0))
    z = (matrix - center.reshape(1, -1)) / scale.reshape(1, -1)
    scores = np.sqrt(np.nanmean(z * z, axis=1))

    rows = [
        {
            "episode_id": episode_id,
            "representative_score": float(score),
        }
        for episode_id, score in zip(ordered_ids, scores)
    ]
    rows.sort(key=lambda row: (row["representative_score"], _episode_id_number(row["episode_id"])))
    return str(rows[0]["episode_id"]), rows


def compute_action_metrics(expert_action: np.ndarray, policy_action: np.ndarray) -> dict[str, Any]:
    expert = np.asarray(expert_action, dtype=np.float64)
    policy = np.asarray(policy_action, dtype=np.float64)
    if expert.shape != policy.shape:
        raise ValueError(f"expert and policy actions must share shape, got {expert.shape} vs {policy.shape}")
    if expert.ndim != 2 or expert.shape[1] != 4:
        raise ValueError(f"actions must have shape (T, 4), got {expert.shape}")
    if expert.shape[0] == 0:
        raise ValueError("actions must contain at least one step")

    err = policy - expert
    axes: dict[str, Any] = {}
    for idx, name in enumerate(AXIS_NAMES):
        expert_axis = expert[:, idx]
        policy_axis = policy[:, idx]
        err_axis = err[:, idx]
        axes[name] = {
            "expert_mean": float(np.mean(expert_axis)),
            "policy_mean": float(np.mean(policy_axis)),
            "bias": float(np.mean(err_axis)),
            "mae": float(np.mean(np.abs(err_axis))),
            "rmse": float(np.sqrt(np.mean(err_axis * err_axis))),
            "correlation": _correlation(expert_axis, policy_axis),
            "expert_p50_abs": _percentile_abs(expert_axis, 50),
            "expert_p95_abs": _percentile_abs(expert_axis, 95),
            "expert_p99_abs": _percentile_abs(expert_axis, 99),
            "expert_max_abs": float(np.max(np.abs(expert_axis))),
            "policy_p50_abs": _percentile_abs(policy_axis, 50),
            "policy_p95_abs": _percentile_abs(policy_axis, 95),
            "policy_p99_abs": _percentile_abs(policy_axis, 99),
            "policy_max_abs": float(np.max(np.abs(policy_axis))),
        }

    return {
        "n_steps": int(expert.shape[0]),
        "overall": {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "max_abs_error": float(np.max(np.abs(err))),
            "expert_p95_abs": _percentile_abs(expert.reshape(-1), 95),
            "policy_p95_abs": _percentile_abs(policy.reshape(-1), 95),
            "expert_max_abs": float(np.max(np.abs(expert))),
            "policy_max_abs": float(np.max(np.abs(policy))),
        },
        "axes": axes,
    }


def load_policy_for_episode(
    *,
    bundle_dir: str | Path,
    ckpt_path: str | Path,
    resolved_config_path: str | Path | None,
    stats_path: str | Path | None,
    max_episode_len: int,
    temporal_agg: bool,
    device: str | None,
    inference_precision: str = "fp32",
) -> Any:
    """Load ACT policy while patching auto episode_len for long offline replay."""

    from testbed.actions.policy import _act_policy_config_from_resolved
    from testbed.policies.act.adapter import ACTAdapter

    bundle = Path(bundle_dir)
    resolved_path = (
        Path(resolved_config_path) if resolved_config_path else bundle / "resolved_config.yaml"
    )
    norm_stats_path = Path(stats_path) if stats_path else bundle / "dataset_stats.pkl"
    with resolved_path.open("r", encoding="utf-8") as f:
        resolved = yaml.safe_load(f) or {}

    patched = deepcopy(resolved)
    task = patched.setdefault("task", {})
    if task.get("episode_len") is None:
        task["episode_len"] = int(max_episode_len)
    policy_config = _act_policy_config_from_resolved(patched)
    return ACTAdapter.from_checkpoint(
        ckpt_path=ckpt_path,
        policy_config=policy_config,
        norm_stats_path=norm_stats_path,
        temporal_agg=bool(temporal_agg),
        device=str(device or patched.get("policy", {}).get("device", "cuda")),
        inference_precision=inference_precision,
    )


def evaluate_episode(
    *,
    policy: Any,
    episode_file: str | Path,
    camera_names: list[str],
    image_episode_file: str | Path | None = None,
    image_step_mode: str = "progress",
    image_transform: str = "none",
    max_steps: int | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Replay one HDF5 episode through a loaded policy."""

    import h5py

    from testbed.data.dataset import _read_camera_image

    episode_file = Path(episode_file)
    image_episode_path = Path(image_episode_file) if image_episode_file is not None else None
    image_transform_fn = build_image_transform(image_transform)
    with h5py.File(episode_file, "r") as f:
        image_f_ctx = h5py.File(image_episode_path, "r") if image_episode_path else None
        image_f = image_f_ctx if image_f_ctx is not None else f
        qpos = np.asarray(f["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(f["observations/qvel"][()], dtype=np.float32)
        expert_action = np.asarray(f["action"][()], dtype=np.float32)
        condition = None
        if "conditions/real_transition_condition_v1" in f:
            condition = np.asarray(
                f["conditions/real_transition_condition_v1"][()],
                dtype=np.float32,
            )
            if condition.ndim != 2 or condition.shape[1] != 2:
                raise ValueError(
                    "conditions/real_transition_condition_v1 must have shape (T, 2)"
                )
        step_count = int(min(qpos.shape[0], qvel.shape[0], expert_action.shape[0]))
        if max_steps is not None:
            step_count = min(step_count, int(max_steps))
        if step_count <= 0:
            raise ValueError(f"episode has no steps: {episode_file}")
        dt = _episode_dt(f)
        image_step_count = _episode_step_count(image_f)
        image_step_map, image_match_metrics = _build_image_step_map(
            target_qpos=qpos[:step_count],
            image_h5=image_f,
            mode=image_step_mode,
        )
        policy_action = np.zeros((step_count, 4), dtype=np.float32)
        if hasattr(policy, "reset"):
            policy.reset()
        try:
            for step in range(step_count):
                image_step = _map_image_step(
                    step=step,
                    target_steps=step_count,
                    image_steps=image_step_count,
                    mode=image_step_mode,
                    nearest_steps=image_step_map,
                )
                obs: dict[str, Any] = {
                    "qpos": qpos[step],
                    "qvel": qvel[step],
                }
                if condition is not None:
                    obs["real_transition_condition_v1"] = condition[step]
                for camera_name in camera_names:
                    image = _read_camera_image(
                        image_f,
                        camera_name,
                        image_step,
                    )
                    if image_transform_fn is not None:
                        image = image_transform_fn(image)
                    obs[f"image_{camera_name}"] = image
                policy_action[step] = np.asarray(policy.predict(obs), dtype=np.float32).reshape(4)
                if progress_every > 0 and (step + 1) % progress_every == 0:
                    print(f"offline eval replayed {step + 1}/{step_count} steps")
        finally:
            if image_f_ctx is not None:
                image_f_ctx.close()

    expert = expert_action[:step_count].astype(np.float32, copy=False)
    metrics = compute_action_metrics(expert, policy_action)
    return {
        "episode_path": str(episode_file),
        "image_episode_path": str(image_episode_path) if image_episode_path else str(episode_file),
        "image_step_mode": str(image_step_mode),
        "image_transform": str(image_transform),
        "image_match_metrics": image_match_metrics,
        "n_steps": int(step_count),
        "dt": float(dt),
        "expert_action": expert,
        "policy_action": policy_action,
        "metrics": metrics,
    }


def write_eval_report(
    *,
    result: dict[str, Any],
    output_dir: str | Path,
    metadata: dict[str, Any],
    representative_scores: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write JSON/CSV/PNG artifacts for one offline eval result."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expert = np.asarray(result["expert_action"], dtype=np.float32)
    policy = np.asarray(result["policy_action"], dtype=np.float32)
    times = np.arange(expert.shape[0], dtype=np.float64) * float(result["dt"])

    summary = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        **metadata,
        "episode_path": result["episode_path"],
        "image_episode_path": result.get("image_episode_path", result["episode_path"]),
        "image_step_mode": result.get("image_step_mode"),
        "image_transform": result.get("image_transform", metadata.get("image_transform", "none")),
        "image_match_metrics": result.get("image_match_metrics"),
        "n_steps": int(result["n_steps"]),
        "dt": float(result["dt"]),
        "metrics": result["metrics"],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    actions_csv = output / "actions.csv"
    _write_actions_csv(actions_csv, times=times, expert=expert, policy=policy)

    npz_path = output / "actions.npz"
    np.savez_compressed(npz_path, time_s=times, expert_action=expert, policy_action=policy)

    timeseries_path = output / "action_timeseries.png"
    distribution_path = output / "action_distribution.png"
    _plot_action_timeseries(times, expert, policy, timeseries_path)
    _plot_action_distribution(expert, policy, distribution_path)

    paths = {
        "summary": summary_path,
        "actions_csv": actions_csv,
        "actions_npz": npz_path,
        "timeseries_plot": timeseries_path,
        "distribution_plot": distribution_path,
    }
    if representative_scores is not None:
        scores_csv = output / "representative_scores.csv"
        _write_representative_scores_csv(scores_csv, representative_scores)
        paths["representative_scores_csv"] = scores_csv
    return paths


def aggregate_episode_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode eval results into global metrics and CSV rows."""

    if not results:
        raise ValueError("results must not be empty")

    expert_parts = []
    policy_parts = []
    episode_rows = []
    for index, result in enumerate(results):
        expert = np.asarray(result["expert_action"], dtype=np.float32)
        policy = np.asarray(result["policy_action"], dtype=np.float32)
        expert_parts.append(expert)
        policy_parts.append(policy)
        episode_id = str(result.get("episode_id") or Path(str(result["episode_path"])).stem)
        metrics = result["metrics"]
        row: dict[str, Any] = {
            "episode_id": episode_id,
            "episode_index": index,
            "n_steps": int(result["n_steps"]),
            "overall_mae": float(metrics["overall"]["mae"]),
            "overall_rmse": float(metrics["overall"]["rmse"]),
            "expert_p95_abs": float(metrics["overall"]["expert_p95_abs"]),
            "policy_p95_abs": float(metrics["overall"]["policy_p95_abs"]),
            "expert_max_abs": float(metrics["overall"]["expert_max_abs"]),
            "policy_max_abs": float(metrics["overall"]["policy_max_abs"]),
        }
        for axis_name in AXIS_NAMES:
            axis_metrics = metrics["axes"][axis_name]
            row[f"mae_{axis_name}"] = float(axis_metrics["mae"])
            row[f"rmse_{axis_name}"] = float(axis_metrics["rmse"])
            row[f"bias_{axis_name}"] = float(axis_metrics["bias"])
            row[f"expert_p95_abs_{axis_name}"] = float(axis_metrics["expert_p95_abs"])
            row[f"policy_p95_abs_{axis_name}"] = float(axis_metrics["policy_p95_abs"])
            row[f"expert_max_abs_{axis_name}"] = float(axis_metrics["expert_max_abs"])
            row[f"policy_max_abs_{axis_name}"] = float(axis_metrics["policy_max_abs"])
            row[f"correlation_{axis_name}"] = axis_metrics["correlation"]
        episode_rows.append(row)

    expert_all = np.concatenate(expert_parts, axis=0)
    policy_all = np.concatenate(policy_parts, axis=0)
    return {
        "n_episodes": int(len(results)),
        "n_steps": int(expert_all.shape[0]),
        "global_metrics": compute_action_metrics(expert_all, policy_all),
        "episode_rows": episode_rows,
        "expert_action": expert_all,
        "policy_action": policy_all,
    }


def write_collection_report(
    *,
    aggregate: dict[str, Any],
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Write aggregate JSON/CSV/PNG artifacts for a multi-episode eval."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        **metadata,
        "n_episodes": int(aggregate["n_episodes"]),
        "n_steps": int(aggregate["n_steps"]),
        "global_metrics": aggregate["global_metrics"],
    }
    summary_path = output / "collection_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    episode_metrics_path = output / "episode_metrics.csv"
    _write_episode_metrics_csv(episode_metrics_path, aggregate["episode_rows"])

    expert = np.asarray(aggregate["expert_action"], dtype=np.float32)
    policy = np.asarray(aggregate["policy_action"], dtype=np.float32)
    distribution_path = output / "collection_action_distribution.png"
    _plot_action_distribution(expert, policy, distribution_path)

    p95_path = output / "episode_policy_vs_expert_p95.png"
    _plot_episode_p95_scatter(aggregate["episode_rows"], p95_path)

    mae_path = output / "episode_mae_by_axis.png"
    _plot_episode_mae_by_axis(aggregate["episode_rows"], mae_path)

    return {
        "collection_summary": summary_path,
        "episode_metrics_csv": episode_metrics_path,
        "distribution_plot": distribution_path,
        "p95_plot": p95_path,
        "mae_plot": mae_path,
    }


def default_output_dir(root: str | Path, *, episode_id: str, ckpt_path: str | Path) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path(root) / f"{stamp}_{Path(ckpt_path).stem}_{normalize_episode_id(episode_id)}"


def default_collection_output_dir(root: str | Path, *, ckpt_path: str | Path) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path(root) / f"{stamp}_{Path(ckpt_path).stem}_all_train_ready"


def choose_default_ckpt(bundle_dir: str | Path) -> Path:
    bundle = Path(bundle_dir)
    for name in ("policy_best.ckpt", "policy_latest.ckpt"):
        path = bundle / name
        if path.exists():
            return path
    candidates = sorted(bundle.glob("policy_epoch_*_seed_*.ckpt"), key=_ckpt_sort_key)
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"no checkpoint found under {bundle}")


def _episode_id_number(episode_id: str) -> int:
    return int(str(episode_id).split("_", 1)[1])


def _percentile_abs(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.abs(np.asarray(values, dtype=np.float64)), percentile))


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr if math.isfinite(corr) else None


def _episode_dt(h5_file: Any) -> float:
    if "metadata" in h5_file:
        attrs = h5_file["metadata"].attrs
        if attrs.get("dt") is not None:
            return float(attrs["dt"])
        if attrs.get("control_hz") is not None:
            hz = float(attrs["control_hz"])
            if hz > 0:
                return 1.0 / hz
    return 0.05


def _episode_step_count(h5_file: Any) -> int:
    if "action" in h5_file:
        return int(h5_file["action"].shape[0])
    if "observations/qpos" in h5_file:
        return int(h5_file["observations/qpos"].shape[0])
    raise KeyError("episode file must contain action or observations/qpos")


def _map_image_step(
    *,
    step: int,
    target_steps: int,
    image_steps: int,
    mode: str,
    nearest_steps: np.ndarray | None = None,
) -> int:
    if image_steps <= 0:
        raise ValueError("image_steps must be positive")
    if mode == "nearest_qpos":
        if nearest_steps is None:
            raise ValueError("nearest_steps is required when image_step_mode='nearest_qpos'.")
        return max(0, min(int(nearest_steps[int(step)]), int(image_steps) - 1))
    if target_steps <= 1 or image_steps == 1:
        return 0
    if mode == "same_index":
        return max(0, min(int(step), int(image_steps) - 1))
    if mode == "progress":
        alpha = max(0.0, min(float(step) / float(target_steps - 1), 1.0))
        return int(round(alpha * float(image_steps - 1)))
    raise ValueError(f"Unsupported image_step_mode {mode!r}.")


def _build_image_step_map(
    *,
    target_qpos: np.ndarray,
    image_h5: Any,
    mode: str,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if mode != "nearest_qpos":
        return None, None
    if "observations/qpos" not in image_h5:
        raise KeyError("nearest_qpos image mapping requires observations/qpos in image episode")

    target = np.asarray(target_qpos, dtype=np.float32).reshape(-1, 4)
    image_qpos = np.asarray(image_h5["observations/qpos"][()], dtype=np.float32).reshape(-1, 4)
    if target.shape[0] == 0 or image_qpos.shape[0] == 0:
        raise ValueError("nearest_qpos mapping requires non-empty target and image qpos")

    indices = np.zeros(target.shape[0], dtype=np.int64)
    normalized = np.zeros(target.shape[0], dtype=np.float64)
    max_abs_rad = np.zeros(target.shape[0], dtype=np.float64)
    scale = np.asarray([0.08, 0.05, 0.03, 0.08], dtype=np.float64)
    for idx, qpos in enumerate(target.astype(np.float64, copy=False)):
        diff = _real_qpos_delta_matrix(qpos, image_qpos.astype(np.float64, copy=False))
        dist = np.linalg.norm(diff / scale.reshape(1, -1), axis=1)
        best = int(np.argmin(dist))
        indices[idx] = best
        normalized[idx] = float(dist[best])
        max_abs_rad[idx] = float(np.max(np.abs(diff[best])))

    metrics = {
        "qpos_match_mode": "nearest_qpos",
        "mean_normalized_distance": float(np.mean(normalized)),
        "p95_normalized_distance": float(np.percentile(normalized, 95)),
        "max_normalized_distance": float(np.max(normalized)),
        "mean_max_abs_delta_rad": float(np.mean(max_abs_rad)),
        "p95_max_abs_delta_rad": float(np.percentile(max_abs_rad, 95)),
        "max_abs_delta_rad": float(np.max(max_abs_rad)),
        "unique_image_steps": int(np.unique(indices).size),
    }
    return indices, metrics


def _real_qpos_delta_matrix(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    diff = np.asarray(target, dtype=np.float64).reshape(1, 4) - np.asarray(
        current,
        dtype=np.float64,
    ).reshape(-1, 4)
    diff[:, 0] = (diff[:, 0] + math.pi) % (2.0 * math.pi) - math.pi
    return diff


def _write_actions_csv(
    path: Path,
    *,
    times: np.ndarray,
    expert: np.ndarray,
    policy: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["step", "time_s"]
        for prefix in ("expert", "policy", "error"):
            header.extend(f"{prefix}_{axis}" for axis in AXIS_NAMES)
        writer.writerow(header)
        for idx in range(expert.shape[0]):
            error = policy[idx] - expert[idx]
            writer.writerow(
                [idx, float(times[idx])]
                + [float(v) for v in expert[idx]]
                + [float(v) for v in policy[idx]]
                + [float(v) for v in error]
            )


def _write_representative_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_id", "representative_score"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_episode_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_action_timeseries(
    times: np.ndarray,
    expert: np.ndarray,
    policy: np.ndarray,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for idx, axis_name in enumerate(AXIS_NAMES):
        ax = axes[idx]
        ax.plot(times, expert[:, idx], label="expert", linewidth=1.1)
        ax.plot(times, policy[:, idx], label="policy", linewidth=1.1, alpha=0.85)
        ax.axhline(0.0, color="0.6", linewidth=0.7)
        ax.set_ylabel(axis_name)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("time_s")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_episode_p95_scatter(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = np.arange(len(rows))
    labels = [str(row["episode_id"]).replace("episode_", "ep") for row in rows]
    for idx, axis_name in enumerate(AXIS_NAMES):
        ax = axes.reshape(-1)[idx]
        expert = [float(row[f"expert_p95_abs_{axis_name}"]) for row in rows]
        policy = [float(row[f"policy_p95_abs_{axis_name}"]) for row in rows]
        ax.plot(x, expert, marker="o", linewidth=1.0, label="expert")
        ax.plot(x, policy, marker="o", linewidth=1.0, label="policy")
        ax.set_title(axis_name)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right")
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_episode_mae_by_axis(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["episode_id"]).replace("episode_", "ep") for row in rows]
    x = np.arange(len(rows))
    width = 0.20
    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, axis_name in enumerate(AXIS_NAMES):
        values = [float(row[f"mae_{axis_name}"]) for row in rows]
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=axis_name)
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_action_distribution(expert: np.ndarray, policy: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, axis_name in enumerate(AXIS_NAMES):
        ax = axes.reshape(-1)[idx]
        ax.hist(expert[:, idx], bins=40, alpha=0.55, label="expert")
        ax.hist(policy[:, idx], bins=40, alpha=0.55, label="policy")
        ax.set_title(axis_name)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _ckpt_sort_key(path: Path) -> tuple[int, str]:
    import re

    match = re.search(r"policy_epoch_(\d+)", path.name)
    return (int(match.group(1)) if match else -1, path.name)
