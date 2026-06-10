"""Training-focused QC metrics and plots for real excavator episodes."""

from __future__ import annotations

import csv
import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


AXES = ("swing", "boom", "stick", "bucket")
BUCKET_AXIS = 3


@dataclass(frozen=True)
class TrainingQcThresholds:
    qpos_jump_fail_rad: float = 0.20
    raw_qpos_branch_jump_fail_rad: float = 3.0
    qvel_residual_p95_warn_rad_s: float = 1.50
    fpv_unique_fps_fail_hz: float = 19.5
    fpv_gap_warn_ms: float = 100.0
    fpv_gap_fail_ms: float = 250.0
    fpv_age_p95_fail_ms: float = 250.0
    sync_skew_p95_fail_ms: float = 80.0
    health_rate_fail: float = 0.995
    bucket_reference_margin_rad: float = 0.25
    length_mad_fail_multiplier: float = 3.0


def run_training_qc(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    mode: str = "quick",
    reference_episode_ids: list[int] | None = None,
    thresholds: TrainingQcThresholds | None = None,
) -> dict[str, Any]:
    """Run training-focused QC and write report artifacts."""

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = str(mode).lower()
    if mode not in {"quick", "full"}:
        raise ValueError(f"Unsupported training QC mode {mode!r}.")
    th = thresholds or TrainingQcThresholds()

    episode_paths = _list_episodes(dataset_dir)
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found under {dataset_dir}")

    ref_ids = reference_episode_ids if reference_episode_ids is not None else list(range(26, 47))
    reference = build_reference_stats(
        dataset_dir=dataset_dir,
        episode_ids=ref_ids,
        thresholds=th,
    )

    rows: list[dict[str, Any]] = []
    for path in episode_paths:
        metrics = episode_training_metrics(
            path,
            reference_stats=reference,
            thresholds=th,
            make_plot=mode == "full",
            plot_dir=output_dir / "episodes",
        )
        rows.append(metrics)

    summary = _summary_from_rows(
        rows,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        mode=mode,
        reference=reference,
    )
    _write_json(output_dir / "training_qc_summary.json", summary)
    _write_csv(output_dir / "training_qc_episodes.csv", rows)
    _write_json(output_dir / "train_ready_manifest.json", _train_ready_manifest(rows, summary))
    if mode == "full":
        _plot_dataset_heatmap(rows, output_dir / "training_qc_heatmap.png")
    return {
        "summary": summary,
        "rows": rows,
        "summary_path": str(output_dir / "training_qc_summary.json"),
        "episodes_csv_path": str(output_dir / "training_qc_episodes.csv"),
        "manifest_path": str(output_dir / "train_ready_manifest.json"),
    }


def build_reference_stats(
    *,
    dataset_dir: str | Path,
    episode_ids: list[int],
    thresholds: TrainingQcThresholds | None = None,
) -> dict[str, Any]:
    """Build robust reference stats from healthy episodes."""

    dataset_dir = Path(dataset_dir)
    th = thresholds or TrainingQcThresholds()
    candidates: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        path = dataset_dir / f"episode_{episode_id}.hdf5"
        if not path.exists():
            continue
        try:
            metrics = episode_training_metrics(
                path,
                reference_stats=None,
                thresholds=th,
                make_plot=False,
                plot_dir=None,
            )
        except Exception:
            continue
        if _hard_reference_candidate_ok(metrics):
            candidates.append(metrics)

    lengths = np.asarray([row["manual_end_index"] for row in candidates], dtype=np.float64)
    total_lengths = np.asarray([row["source_total_steps"] for row in candidates], dtype=np.float64)
    if lengths.size:
        keep, length_bounds = _length_keep_mask(lengths, thresholds=th)
        total_keep, total_length_bounds = _length_keep_mask(total_lengths, thresholds=th)
        keep = keep & total_keep
        selected_ids = [int(row["episode_id_num"]) for row, ok in zip(candidates, keep) if ok]
    else:
        selected_ids = []
        length_bounds = {"count": 0}
        total_length_bounds = {"count": 0}

    qpos_parts: list[np.ndarray] = []
    bucket_parts: list[np.ndarray] = []
    selected_lengths: list[int] = []
    for episode_id in selected_ids:
        path = dataset_dir / f"episode_{episode_id}.hdf5"
        with h5py.File(path, "r") as f:
            qpos = np.asarray(f["observations/qpos"][()], dtype=np.float32)
            diagnostics = f.get("diagnostics")
            manual_end = _manual_end_index(
                n_steps=int(qpos.shape[0]),
                go_home_requested=_read_optional(diagnostics, "go_home_requested"),
                go_home_running=_read_optional(diagnostics, "go_home_running"),
            )
        qpos_parts.append(qpos)
        bucket_parts.append(qpos[:, BUCKET_AXIS])
        selected_lengths.append(int(manual_end))
    qpos_cat = np.concatenate(qpos_parts, axis=0) if qpos_parts else np.zeros((0, 4), dtype=np.float32)
    bucket_cat = np.concatenate(bucket_parts, axis=0) if bucket_parts else np.zeros((0,), dtype=np.float32)
    return {
        "candidate_episode_ids": [int(row["episode_id_num"]) for row in candidates],
        "selected_episode_ids": selected_ids,
        "rejected_episode_ids": [
            int(row["episode_id_num"])
            for row in candidates
            if int(row["episode_id_num"]) not in selected_ids
        ],
        "manual_length": {
            **_robust_stats(np.asarray(selected_lengths, dtype=np.float64)),
            "candidate": _robust_stats(lengths),
            **length_bounds,
        },
        "length": {
            **_robust_stats(np.asarray(selected_lengths, dtype=np.float64)),
            "candidate": _robust_stats(lengths),
            **length_bounds,
        },
        "source_total_steps": {
            "candidate": _robust_stats(total_lengths),
            **total_length_bounds,
        },
        "qpos": _robust_axis_stats(qpos_cat),
        "bucket_qpos": _robust_stats(bucket_cat),
    }


def episode_training_metrics(
    episode_path: str | Path,
    *,
    reference_stats: dict[str, Any] | None,
    thresholds: TrainingQcThresholds | None = None,
    make_plot: bool,
    plot_dir: str | Path | None,
) -> dict[str, Any]:
    """Compute training-focused metrics for one episode."""

    path = Path(episode_path)
    th = thresholds or TrainingQcThresholds()
    with h5py.File(path, "r") as f:
        qpos = np.asarray(f["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(f["observations/qvel"][()], dtype=np.float32)
        action = np.asarray(f["action"][()], dtype=np.float32)
        diagnostics = f.get("diagnostics")
        metadata = _metadata(f)
        timestamps_ns = _read_optional(diagnostics, "joint_timestamp_ns")
        if timestamps_ns is None:
            timestamps_ns = _read_optional(f.get("timestamps"), "step_ns")
        image_ts = _read_optional(diagnostics, "image_timestamp_ns_fpv")
        fpv_age = _read_optional(diagnostics, "fpv_age_ms")
        sync_skew = _read_optional(diagnostics, "sync_max_skew_ns")
        controller_ack = _read_optional(diagnostics, "controller_ack")
        receiver_health = _read_optional(diagnostics, "receiver_health_ok")
        imu_online = _read_optional(diagnostics, "imu_online")
        imu_valid = _read_optional(diagnostics, "imu_valid_attitude")
        go_home_requested = _read_optional(diagnostics, "go_home_requested")
        go_home_running = _read_optional(diagnostics, "go_home_running")
        repair_mask = _read_repair_mask(f)
        fpv_decode = _sample_fpv_decode_metrics(f)

    n_steps = int(action.shape[0]) if action.ndim else 0
    source_total_steps = _source_total_steps(path=path, metadata=metadata, fallback=n_steps)
    dt = _dt_seconds(timestamps_ns, n_steps)
    qpos_jump = _max_abs_diff(qpos)
    raw_qpos_jump = _max_abs_raw_diff(qpos)
    qvel_residual = _qvel_residual(qpos=qpos, qvel=qvel, dt=dt)
    fpv = _fpv_time_metrics(image_ts=image_ts)
    bucket_ref_status = _bucket_reference_status(
        bucket=qpos[:, BUCKET_AXIS] if qpos.ndim == 2 and qpos.shape[1] > BUCKET_AXIS else np.zeros(0),
        reference_stats=reference_stats,
        margin_rad=th.bucket_reference_margin_rad,
    )
    manual_end = _manual_end_index(
        n_steps=n_steps,
        go_home_requested=go_home_requested,
        go_home_running=go_home_running,
    )
    length_ref_status = _length_reference_status(
        length=manual_end,
        reference_stats=reference_stats,
        key="manual_length",
    )
    total_steps_ref_status = _length_reference_status(
        length=source_total_steps,
        reference_stats=reference_stats,
        key="source_total_steps",
    )
    success_ok = _boolish(metadata.get("success", 0))
    go_home_result = str(metadata.get("go_home_result", ""))
    warnings, status = _episode_status(
        success_ok=success_ok,
        go_home_result=go_home_result,
        qpos_jump=qpos_jump,
        raw_qpos_jump=raw_qpos_jump,
        qvel_residual=qvel_residual,
        fpv=fpv,
        fpv_age=fpv_age,
        sync_skew=sync_skew,
        controller_ack=controller_ack,
        receiver_health=receiver_health,
        imu_online=imu_online,
        imu_valid=imu_valid,
        fpv_decode=fpv_decode,
        bucket_ref_status=bucket_ref_status,
        length_ref_status=length_ref_status,
        total_steps_ref_status=total_steps_ref_status,
        thresholds=th,
    )
    episode_id_num = _episode_id_num(path)
    plot_path = ""
    if make_plot and plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = str(plot_dir / f"{path.stem}_timeline.png")
        _plot_episode_timeline(
            qpos=qpos,
            qvel=qvel,
            action=action,
            dt=dt,
            fpv_age=fpv_age,
            sync_skew=sync_skew,
            image_ts=image_ts,
            repair_mask=repair_mask,
            go_home_requested=go_home_requested,
            go_home_running=go_home_running,
            output_path=Path(plot_path),
        )
    return {
        "episode_id": path.stem,
        "episode_id_num": episode_id_num,
        "path": str(path),
        "n_steps": n_steps,
        "source_total_steps": int(source_total_steps),
        "manual_end_index": int(manual_end),
        "success": int(success_ok),
        "go_home_result": go_home_result,
        "training_status": status,
        "training_ready": int(status in {"PASS", "WARN"}),
        "training_warnings": ";".join(warnings),
        "qpos_max_jump_swing": qpos_jump[0],
        "qpos_max_jump_boom": qpos_jump[1],
        "qpos_max_jump_stick": qpos_jump[2],
        "qpos_max_jump_bucket": qpos_jump[3],
        "raw_qpos_max_jump_swing": raw_qpos_jump[0],
        "raw_qpos_max_jump_boom": raw_qpos_jump[1],
        "raw_qpos_max_jump_stick": raw_qpos_jump[2],
        "raw_qpos_max_jump_bucket": raw_qpos_jump[3],
        "qvel_residual_p95_swing": qvel_residual["p95_abs"][0],
        "qvel_residual_p95_boom": qvel_residual["p95_abs"][1],
        "qvel_residual_p95_stick": qvel_residual["p95_abs"][2],
        "qvel_residual_p95_bucket": qvel_residual["p95_abs"][3],
        "fpv_unique_fps": fpv["unique_fps"],
        "fpv_duplicate_ratio": fpv["duplicate_ratio"],
        "fpv_gap_gt100_count": fpv["gap_gt100_count"],
        "fpv_gap_gt150_count": fpv["gap_gt150_count"],
        "fpv_gap_gt250_count": fpv["gap_gt250_count"],
        "fpv_max_gap_ms": fpv["max_gap_ms"],
        "fpv_timestamp_backward_count": fpv["timestamp_backward_count"],
        "fpv_age_p95_ms": _pctl(fpv_age, 95),
        "fpv_age_max_ms": _max(fpv_age),
        "sync_skew_p95_ms": _pctl_ns_to_ms(sync_skew, 95),
        "controller_ack_rate": _rate(controller_ack),
        "receiver_health_ok_rate": _rate(receiver_health),
        "imu_online_all_rate": _all_bits_rate(imu_online),
        "imu_valid_all_rate": _all_bits_rate(imu_valid),
        "fpv_decode_bad_sample_count": fpv_decode["bad_sample_count"],
        "fpv_black_sample_ratio": fpv_decode["black_sample_ratio"],
        "fpv_near_duplicate_sample_ratio": fpv_decode["near_duplicate_sample_ratio"],
        "bucket_reference_status": bucket_ref_status["status"],
        "bucket_ref_low_margin": bucket_ref_status["low_margin"],
        "bucket_ref_high_margin": bucket_ref_status["high_margin"],
        "length_reference_status": length_ref_status["status"],
        "length_ref_low_margin": length_ref_status["low_margin"],
        "length_ref_high_margin": length_ref_status["high_margin"],
        "source_total_steps_reference_status": total_steps_ref_status["status"],
        "source_total_steps_low_margin": total_steps_ref_status["low_margin"],
        "source_total_steps_high_margin": total_steps_ref_status["high_margin"],
        "bucket_repair_fraction": float(np.mean(repair_mask)) if repair_mask.size else 0.0,
        "plot_path": plot_path,
    }


def _hard_reference_candidate_ok(row: dict[str, Any]) -> bool:
    return (
        int(row.get("success", 0)) == 1
        and float(row.get("controller_ack_rate", 0.0)) >= 0.999
        and float(row.get("receiver_health_ok_rate", 0.0)) >= 0.999
        and float(row.get("imu_online_all_rate", 0.0)) >= 0.999
        and float(row.get("imu_valid_all_rate", 0.0)) >= 0.999
        and float(row.get("fpv_unique_fps", 0.0)) >= 19.5
        and float(row.get("fpv_gap_gt250_count", 0.0)) == 0.0
        and max(
            float(row.get("qpos_max_jump_swing", 0.0)),
            float(row.get("qpos_max_jump_boom", 0.0)),
            float(row.get("qpos_max_jump_stick", 0.0)),
            float(row.get("qpos_max_jump_bucket", 0.0)),
        )
        <= 0.20
        and float(row.get("fpv_decode_bad_sample_count", 1.0)) == 0.0
    )


def _episode_status(
    *,
    success_ok: bool,
    go_home_result: str,
    qpos_jump: list[float],
    raw_qpos_jump: list[float],
    qvel_residual: dict[str, list[float]],
    fpv: dict[str, float | int],
    fpv_age: np.ndarray | None,
    sync_skew: np.ndarray | None,
    controller_ack: np.ndarray | None,
    receiver_health: np.ndarray | None,
    imu_online: np.ndarray | None,
    imu_valid: np.ndarray | None,
    fpv_decode: dict[str, float | int],
    bucket_ref_status: dict[str, Any],
    length_ref_status: dict[str, Any],
    total_steps_ref_status: dict[str, Any],
    thresholds: TrainingQcThresholds,
) -> tuple[list[str], str]:
    warnings: list[str] = []
    fail = False
    if not success_ok:
        warnings.append("success_false")
        fail = True
    normalized_go_home = str(go_home_result).strip().lower()
    if normalized_go_home not in {"", "succeeded", "done", "success", "completed"}:
        warnings.append("go_home_not_succeeded")
        fail = True
    if max(qpos_jump) > thresholds.qpos_jump_fail_rad:
        warnings.append("qpos_jump")
        fail = True
    if max(raw_qpos_jump) > thresholds.raw_qpos_branch_jump_fail_rad:
        warnings.append("raw_qpos_branch_jump")
        fail = True
    if max(qvel_residual["p95_abs"]) > thresholds.qvel_residual_p95_warn_rad_s:
        warnings.append("qvel_residual_high")
    if float(fpv["unique_fps"]) < thresholds.fpv_unique_fps_fail_hz:
        warnings.append("fpv_unique_fps_low")
        fail = True
    if int(fpv["gap_gt250_count"]) > 0 or float(fpv["max_gap_ms"]) > thresholds.fpv_gap_fail_ms:
        warnings.append("fpv_gap_fail")
        fail = True
    if _pctl(fpv_age, 95) > thresholds.fpv_age_p95_fail_ms:
        warnings.append("fpv_age_high")
        fail = True
    if _pctl_ns_to_ms(sync_skew, 95) > thresholds.sync_skew_p95_fail_ms:
        warnings.append("sync_skew_high")
    if _rate(controller_ack) < thresholds.health_rate_fail:
        warnings.append("controller_ack_low")
        fail = True
    if _rate(receiver_health) < thresholds.health_rate_fail:
        warnings.append("receiver_health_low")
        fail = True
    if _all_bits_rate(imu_online) < thresholds.health_rate_fail:
        warnings.append("imu_online_low")
        fail = True
    if _all_bits_rate(imu_valid) < thresholds.health_rate_fail:
        warnings.append("imu_valid_low")
        fail = True
    if int(fpv_decode["bad_sample_count"]) > 0 or float(fpv_decode["black_sample_ratio"]) > 0.0:
        warnings.append("fpv_decode_or_black")
        fail = True
    if bucket_ref_status["status"] == "FAIL":
        warnings.append("bucket_reference_outlier")
        fail = True
    elif bucket_ref_status["status"] == "WARN":
        warnings.append("bucket_reference_warn")
    if length_ref_status["status"] == "FAIL":
        warnings.append("episode_length_outlier")
        fail = True
    if total_steps_ref_status["status"] == "FAIL":
        warnings.append("episode_total_steps_outlier")
        fail = True
    if fail:
        return warnings, "FAIL"
    if warnings:
        return warnings, "WARN"
    return warnings, "PASS"


def _fpv_time_metrics(*, image_ts: np.ndarray | None) -> dict[str, float | int]:
    if image_ts is None:
        return {
            "unique_fps": 0.0,
            "duplicate_ratio": 1.0,
            "gap_gt100_count": 0,
            "gap_gt150_count": 0,
            "gap_gt250_count": 0,
            "max_gap_ms": 0.0,
            "timestamp_backward_count": 0,
        }
    ts = np.asarray(image_ts, dtype=np.int64).reshape(-1)
    positive = ts[ts > 0]
    if positive.size < 2:
        return {
            "unique_fps": 0.0,
            "duplicate_ratio": 1.0,
            "gap_gt100_count": 0,
            "gap_gt150_count": 0,
            "gap_gt250_count": 0,
            "max_gap_ms": 0.0,
            "timestamp_backward_count": int(np.sum(np.diff(ts) < 0)) if ts.size > 1 else 0,
        }
    unique = np.unique(positive)
    duplicate_ratio = 1.0 - float(unique.size) / float(positive.size)
    deltas_ms = np.diff(unique).astype(np.float64) * 1e-6
    positive_deltas = deltas_ms[deltas_ms > 0.0]
    if positive_deltas.size:
        duration_s = float((unique[-1] - unique[0]) * 1e-9)
        fps = float((unique.size - 1) / duration_s) if duration_s > 0.0 else 0.0
        max_gap = float(np.max(positive_deltas))
    else:
        fps = 0.0
        max_gap = 0.0
    return {
        "unique_fps": fps,
        "duplicate_ratio": duplicate_ratio,
        "gap_gt100_count": int(np.sum(positive_deltas > 100.0)),
        "gap_gt150_count": int(np.sum(positive_deltas > 150.0)),
        "gap_gt250_count": int(np.sum(positive_deltas > 250.0)),
        "max_gap_ms": max_gap,
        "timestamp_backward_count": int(np.sum(np.diff(ts) < 0)),
    }


def _sample_fpv_decode_metrics(f: h5py.File, *, limit: int = 24) -> dict[str, float | int]:
    raw_group = f.get("observations/images")
    encoded_group = f.get("observations/encoded_images")
    group = raw_group if raw_group is not None else encoded_group
    if group is None or "fpv" not in group:
        return {"bad_sample_count": 1, "black_sample_ratio": 1.0, "near_duplicate_sample_ratio": 1.0}
    dataset = group["fpv"]
    count = int(dataset.shape[0]) if dataset.shape else 0
    if count <= 0:
        return {"bad_sample_count": 1, "black_sample_ratio": 1.0, "near_duplicate_sample_ratio": 1.0}
    indices = _sample_indices(count, limit)
    means: list[float] = []
    fingerprints: list[int] = []
    bad = 0
    for idx in indices:
        try:
            if raw_group is not None:
                frame = np.asarray(dataset[idx], dtype=np.uint8)
            else:
                frame = _decode_jpeg(np.asarray(dataset[idx], dtype=np.uint8).reshape(-1))
            means.append(float(frame.mean()) if frame.size else 0.0)
            fingerprints.append(hash(frame[:: max(1, frame.shape[0] // 16), :: max(1, frame.shape[1] // 16)].tobytes()))
        except Exception:
            bad += 1
    if not means:
        return {"bad_sample_count": bad or 1, "black_sample_ratio": 1.0, "near_duplicate_sample_ratio": 1.0}
    duplicate = 1.0 - len(set(fingerprints)) / max(1, len(fingerprints))
    return {
        "bad_sample_count": int(bad),
        "black_sample_ratio": float(np.mean(np.asarray(means) <= 5.0)),
        "near_duplicate_sample_ratio": float(duplicate),
    }


def _decode_jpeg(data: np.ndarray) -> np.ndarray:
    import cv2

    bgr = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode returned None")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _bucket_reference_status(
    *,
    bucket: np.ndarray,
    reference_stats: dict[str, Any] | None,
    margin_rad: float,
) -> dict[str, Any]:
    if reference_stats is None or not reference_stats.get("bucket_qpos"):
        return {"status": "UNKNOWN", "low_margin": 0.0, "high_margin": 0.0}
    ref = reference_stats.get("bucket_qpos", {})
    p1 = ref.get("p1")
    p99 = ref.get("p99")
    if p1 is None or p99 is None or np.asarray(bucket).size == 0:
        return {"status": "UNKNOWN", "low_margin": 0.0, "high_margin": 0.0}
    low_margin = float(np.min(bucket) - float(p1))
    high_margin = float(float(p99) - np.max(bucket))
    if low_margin < -margin_rad or high_margin < -margin_rad:
        status = "FAIL"
    elif low_margin < 0.0 or high_margin < 0.0:
        status = "WARN"
    else:
        status = "PASS"
    return {"status": status, "low_margin": low_margin, "high_margin": high_margin}


def _length_keep_mask(
    lengths: np.ndarray,
    *,
    thresholds: TrainingQcThresholds,
) -> tuple[np.ndarray, dict[str, float | int]]:
    arr = np.asarray(lengths, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.zeros(0, dtype=bool), {"count": 0}
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = 1.4826 * mad
    if scale <= 1e-6:
        lower = float(np.min(arr))
        upper = float(np.max(arr))
    else:
        lower = median - thresholds.length_mad_fail_multiplier * scale
        upper = median + thresholds.length_mad_fail_multiplier * scale
    keep = (np.asarray(lengths, dtype=np.float64) >= lower) & (
        np.asarray(lengths, dtype=np.float64) <= upper
    )
    return keep, {
        "candidate_count": int(arr.size),
        "candidate_median": median,
        "candidate_mad": mad,
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "mad_fail_multiplier": float(thresholds.length_mad_fail_multiplier),
    }


def _length_reference_status(
    *,
    length: int,
    reference_stats: dict[str, Any] | None,
    key: str = "length",
) -> dict[str, Any]:
    if reference_stats is None:
        return {"status": "UNKNOWN", "low_margin": 0.0, "high_margin": 0.0}
    ref = reference_stats.get(key, {})
    lower = ref.get("lower_bound")
    upper = ref.get("upper_bound")
    if lower is None or upper is None:
        return {"status": "UNKNOWN", "low_margin": 0.0, "high_margin": 0.0}
    low_margin = float(length) - float(lower)
    high_margin = float(upper) - float(length)
    status = "PASS" if low_margin >= 0.0 and high_margin >= 0.0 else "FAIL"
    return {"status": status, "low_margin": low_margin, "high_margin": high_margin}


def _source_total_steps(*, path: Path, metadata: dict[str, Any], fallback: int) -> int:
    raw = metadata.get("source_total_steps")
    if raw is not None:
        try:
            return int(raw)
        except Exception:
            pass
    source_path_raw = metadata.get("source_dataset_path")
    if source_path_raw:
        source_path = Path(str(source_path_raw))
        if source_path.exists():
            try:
                with h5py.File(source_path, "r") as f:
                    if "action" in f:
                        return int(f["action"].shape[0])
            except Exception:
                pass
    return int(fallback)


def _qvel_residual(*, qpos: np.ndarray, qvel: np.ndarray, dt: np.ndarray) -> dict[str, list[float]]:
    if qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape != qvel.shape or qpos.shape[0] < 2:
        return {"p95_abs": [0.0, 0.0, 0.0, 0.0], "max_abs": [0.0, 0.0, 0.0, 0.0]}
    delta = np.diff(qpos.astype(np.float64), axis=0)
    if delta.shape[1] > 0:
        delta[:, 0] = (delta[:, 0] + np.pi) % (2.0 * np.pi) - np.pi
    diff_qvel = delta / dt.reshape(-1, 1)
    residual = diff_qvel - qvel[1:].astype(np.float64)
    return {
        "p95_abs": np.percentile(np.abs(residual), 95, axis=0).tolist(),
        "max_abs": np.max(np.abs(residual), axis=0).tolist(),
    }


def _max_abs_diff(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return [0.0, 0.0, 0.0, 0.0]
    delta = np.diff(arr, axis=0)
    if delta.shape[1] > 0:
        # swing is a circular yaw-like axis; use shortest-angle step for QC.
        delta[:, 0] = (delta[:, 0] + np.pi) % (2.0 * np.pi) - np.pi
    out = np.max(np.abs(delta), axis=0).tolist()
    return [float(x) for x in out]


def _max_abs_raw_diff(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return [0.0, 0.0, 0.0, 0.0]
    out = np.max(np.abs(np.diff(arr, axis=0)), axis=0).tolist()
    return [float(x) for x in out]


def _dt_seconds(timestamps_ns: np.ndarray | None, n_steps: int) -> np.ndarray:
    if n_steps < 2:
        return np.zeros(0, dtype=np.float64)
    if timestamps_ns is None:
        return np.full(n_steps - 1, 0.02, dtype=np.float64)
    ts = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    if ts.size != n_steps:
        return np.full(n_steps - 1, 0.02, dtype=np.float64)
    dt = np.diff(ts).astype(np.float64) * 1e-9
    valid = np.isfinite(dt) & (dt > 0.0) & (dt < 1.0)
    fill = float(np.median(dt[valid])) if np.any(valid) else 0.02
    dt[~valid] = fill
    return dt


def _manual_end_index(
    *,
    n_steps: int,
    go_home_requested: np.ndarray | None,
    go_home_running: np.ndarray | None,
) -> int:
    candidates: list[int] = []
    for arr in (go_home_requested, go_home_running):
        if arr is None:
            continue
        idx = np.flatnonzero(np.asarray(arr).reshape(-1) > 0)
        if idx.size:
            candidates.append(int(idx[0]))
    return min(candidates) if candidates else int(n_steps)


def _read_repair_mask(f: h5py.File) -> np.ndarray:
    path = "repairs/bucket_qpos_v1/repair_mask"
    if path in f:
        return np.asarray(f[path][()], dtype=bool)
    if "action" in f:
        return np.zeros(int(f["action"].shape[0]), dtype=bool)
    return np.zeros(0, dtype=bool)


def _read_optional(group: h5py.Group | h5py.File | None, name: str) -> np.ndarray | None:
    if group is None or name not in group:
        return None
    return np.asarray(group[name][()])


def _metadata(f: h5py.File) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "metadata" in f:
        out.update(dict(f["metadata"].attrs))
    out.update(dict(f.attrs))
    return out


def _rate(value: np.ndarray | None) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(np.mean(arr > 0)) if arr.size else 0.0


def _all_bits_rate(value: np.ndarray | None) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    if arr.ndim == 1:
        return float(np.mean(arr > 0))
    return float(np.mean(np.all(arr > 0, axis=1)))


def _pctl(value: np.ndarray | None, percentile: float) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, percentile)) if arr.size else 0.0


def _pctl_ns_to_ms(value: np.ndarray | None, percentile: float) -> float:
    return _pctl(value, percentile) * 1e-6


def _max(value: np.ndarray | None) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else 0.0


def _boolish(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return bool(int(value))
    except Exception:
        return str(value).lower() in {"true", "yes", "succeeded"}


def _robust_stats(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    q25, q75 = np.percentile(arr, [25, 75])
    median = float(np.median(arr))
    return {
        "count": int(arr.size),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(q25),
        "median": median,
        "p75": float(q75),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "iqr": float(q75 - q25),
        "mad": float(np.median(np.abs(arr - median))),
    }


def _robust_axis_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return {"count": 0}
    return {
        "count": int(arr.shape[0]),
        "axes": list(AXES[: arr.shape[1]]),
        "p1": np.percentile(arr, 1, axis=0).tolist(),
        "p5": np.percentile(arr, 5, axis=0).tolist(),
        "median": np.median(arr, axis=0).tolist(),
        "p95": np.percentile(arr, 95, axis=0).tolist(),
        "p99": np.percentile(arr, 99, axis=0).tolist(),
    }


def _summary_from_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_dir: Path,
    output_dir: Path,
    mode: str,
    reference: dict[str, Any],
) -> dict[str, Any]:
    counts = {status: sum(1 for row in rows if row["training_status"] == status) for status in ("PASS", "WARN", "FAIL")}
    return {
        "schema_version": 1,
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "mode": mode,
        "n_episodes": len(rows),
        "training_status_counts": counts,
        "strict_pass_episode_ids": [
            row["episode_id"] for row in rows if row["training_status"] == "PASS"
        ],
        "warn_episode_ids": [row["episode_id"] for row in rows if row["training_status"] == "WARN"],
        "train_ready_episode_ids": [row["episode_id"] for row in rows if row["training_ready"]],
        "failed_episode_ids": [row["episode_id"] for row in rows if row["training_status"] == "FAIL"],
        "reference": reference,
    }


def _train_ready_manifest(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": summary["generated_at"],
        "dataset_dir": summary["dataset_dir"],
        "strict_pass_episode_ids": summary.get("strict_pass_episode_ids", []),
        "warn_episode_ids": summary.get("warn_episode_ids", []),
        "train_ready_episode_ids": summary["train_ready_episode_ids"],
        "failed_episode_ids": summary.get("failed_episode_ids", []),
        "excluded_episode_ids": summary.get("failed_episode_ids", []),
        "excluded": [
            {
                "episode_id": row["episode_id"],
                "status": row["training_status"],
                "warnings": row["training_warnings"],
            }
            for row in rows
            if not row["training_ready"]
        ],
    }


def _plot_episode_timeline(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    dt: np.ndarray,
    fpv_age: np.ndarray | None,
    sync_skew: np.ndarray | None,
    image_ts: np.ndarray | None,
    repair_mask: np.ndarray,
    go_home_requested: np.ndarray | None,
    go_home_running: np.ndarray | None,
    output_path: Path,
) -> None:
    t = np.arange(action.shape[0], dtype=np.float64) * 0.02
    if dt.size == max(0, action.shape[0] - 1):
        t = np.concatenate([[0.0], np.cumsum(dt)])
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    for axis, name in enumerate(AXES):
        axes[0].plot(t, qpos[:, axis], label=name)
        axes[1].plot(t, qvel[:, axis], label=name)
        axes[2].plot(t, action[:, axis], label=name)
    axes[0].set_ylabel("qpos rad")
    axes[1].set_ylabel("qvel rad/s")
    axes[2].set_ylabel("action")
    for ax in axes[:3]:
        ax.legend(loc="upper right", ncol=4, fontsize=8)
    if qpos.shape[0] > 1:
        diff_qvel = np.diff(qpos, axis=0) / dt.reshape(-1, 1)
        axes[3].plot(t[1:], diff_qvel[:, BUCKET_AXIS] - qvel[1:, BUCKET_AXIS], label="bucket diff-qvel")
    if repair_mask.size:
        axes[3].fill_between(t, 0, 1, where=repair_mask[: t.size], transform=axes[3].get_xaxis_transform(), color="tab:red", alpha=0.15, label="bucket repair")
    axes[3].set_ylabel("bucket residual")
    axes[3].legend(loc="upper right", fontsize=8)
    if fpv_age is not None:
        axes[4].plot(t[: len(fpv_age)], np.asarray(fpv_age).reshape(-1), label="fpv_age_ms")
    if sync_skew is not None:
        axes[4].plot(t[: len(sync_skew)], np.asarray(sync_skew).reshape(-1) * 1e-6, label="sync_skew_ms")
    if image_ts is not None:
        fpv = _fpv_time_metrics(image_ts=image_ts)
        axes[4].set_title(
            f"FPV unique_fps={fpv['unique_fps']:.1f} max_gap_ms={fpv['max_gap_ms']:.1f}"
        )
    axes[4].set_ylabel("ms")
    axes[4].set_xlabel("time s")
    axes[4].legend(loc="upper right", fontsize=8)
    _shade_go_home(axes, t, go_home_requested=go_home_requested, go_home_running=go_home_running)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _shade_go_home(
    axes: np.ndarray,
    t: np.ndarray,
    *,
    go_home_requested: np.ndarray | None,
    go_home_running: np.ndarray | None,
) -> None:
    mask = np.zeros(t.size, dtype=bool)
    for arr in (go_home_requested, go_home_running):
        if arr is None:
            continue
        values = np.asarray(arr).reshape(-1)[: t.size] > 0
        mask[: values.size] |= values
    if not np.any(mask):
        return
    start = t[int(np.flatnonzero(mask)[0])]
    end = t[int(np.flatnonzero(mask)[-1])]
    for ax in axes:
        ax.axvspan(start, end, color="tab:purple", alpha=0.08)


def _plot_dataset_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "qpos_max_jump_bucket",
        "qvel_residual_p95_bucket",
        "fpv_unique_fps",
        "fpv_max_gap_ms",
        "fpv_age_p95_ms",
        "sync_skew_p95_ms",
        "imu_valid_all_rate",
        "controller_ack_rate",
        "receiver_health_ok_rate",
        "bucket_repair_fraction",
    ]
    data = np.asarray([[float(row.get(field, 0.0) or 0.0) for field in fields] for row in rows], dtype=np.float64)
    if data.size == 0:
        return
    med = np.nanmedian(data, axis=0)
    mad = np.nanmedian(np.abs(data - med), axis=0)
    scale = np.where(mad > 1e-9, 1.4826 * mad, 1.0)
    z = (data - med) / scale
    fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.22)))
    im = ax.imshow(z, aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["episode_id"] for row in rows], fontsize=7)
    ax.set_xticks(np.arange(len(fields)))
    ax.set_xticklabels(fields, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label="robust z")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _sample_indices(count: int, limit: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    return sorted(set(np.linspace(0, count - 1, limit, dtype=np.int64).tolist()))


def _list_episodes(dataset_dir: Path) -> list[Path]:
    out: list[Path] = []
    for path in dataset_dir.glob("episode_*.hdf5"):
        try:
            int(path.stem.split("_", 1)[1])
        except Exception:
            continue
        out.append(path)
    return sorted(out, key=_episode_id_num)


def _episode_id_num(path: Path) -> int:
    try:
        return int(path.stem.split("_", 1)[1])
    except Exception:
        return -1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
