"""Lightweight per-episode QC for field recording feedback."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_DIAGNOSTICS = {
    "raw_action",
    "guard_triggered",
    "guard_reason",
    "controller_ack",
    "controller_fault_code",
    "controller_timestamp_ns",
    "commanded_action",
}


def run_episode_qc(
    episode_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_sampled_frames: int = 24,
    black_mean_threshold: float = 5.0,
    duplicate_warn_ratio: float = 0.50,
) -> dict[str, Any]:
    """Run lightweight checks on one completed HDF5 episode."""

    path = Path(episode_path)
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    try:
        with h5py.File(path, "r") as f:
            metadata = dict(f["metadata"].attrs) if "metadata" in f else {}
            metadata.update(dict(f.attrs))
            qpos = _read_array(f, "observations/qpos", errors)
            qvel = _read_array(f, "observations/qvel", errors)
            action = _read_array(f, "action", errors)
            _check_shapes(qpos=qpos, qvel=qvel, action=action, errors=errors, metrics=metrics)
            _check_finite("qpos", qpos, errors)
            _check_finite("qvel", qvel, errors)
            _check_finite("action", action, errors)
            _check_timestamps(f, errors=errors, warnings=warnings, metrics=metrics)
            _check_diagnostics(f, errors=errors, warnings=warnings, metrics=metrics)
            _check_images(
                f,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                max_sampled_frames=max_sampled_frames,
                black_mean_threshold=black_mean_threshold,
                duplicate_warn_ratio=duplicate_warn_ratio,
            )
    except Exception as exc:
        errors.append(f"unreadable_hdf5:{type(exc).__name__}:{exc}")

    success = _bool_attr(metadata.get("success", 0))
    if not success:
        warnings.append("episode_success_attr_false")

    result = {
        "episode_path": str(path),
        "episode_id": str(metadata.get("episode_id", path.stem)),
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "ok": not errors,
        "success": bool(success),
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(result), f, indent=2, ensure_ascii=False)
        result["output_path"] = str(out_path)
    return result


def _read_array(f: h5py.File, path: str, errors: list[str]) -> np.ndarray | None:
    if path not in f:
        errors.append(f"missing_dataset:{path}")
        return None
    try:
        return np.asarray(f[path][()])
    except Exception as exc:
        errors.append(f"read_dataset_failed:{path}:{type(exc).__name__}:{exc}")
        return None


def _check_shapes(
    *,
    qpos: np.ndarray | None,
    qvel: np.ndarray | None,
    action: np.ndarray | None,
    errors: list[str],
    metrics: dict[str, Any],
) -> None:
    arrays = {"qpos": qpos, "qvel": qvel, "action": action}
    lengths: dict[str, int] = {}
    for name, arr in arrays.items():
        if arr is None:
            continue
        metrics[f"{name}_shape"] = list(arr.shape)
        if arr.ndim != 2 or arr.shape[1] != 4:
            errors.append(f"{name}_shape_invalid:{arr.shape}")
        lengths[name] = int(arr.shape[0]) if arr.ndim >= 1 else -1
    if lengths:
        metrics["n_steps"] = int(next(iter(lengths.values())))
    if len(set(lengths.values())) > 1:
        errors.append(f"length_mismatch:{lengths}")


def _check_finite(name: str, arr: np.ndarray | None, errors: list[str]) -> None:
    if arr is None:
        return
    if not np.all(np.isfinite(arr)):
        errors.append(f"{name}_nan_or_inf")


def _check_timestamps(
    f: h5py.File,
    *,
    errors: list[str],
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    if "timestamps/step_id" in f:
        step_id = np.asarray(f["timestamps/step_id"][()])
        if step_id.size > 1 and np.any(np.diff(step_id) <= 0):
            errors.append("step_id_not_strictly_increasing")
    else:
        warnings.append("missing_step_id")
    if "timestamps/step_ns" in f:
        step_ns = np.asarray(f["timestamps/step_ns"][()])
        metrics["has_step_ns"] = True
        if step_ns.size > 1 and np.any(np.diff(step_ns) <= 0):
            errors.append("step_ns_not_strictly_increasing")
    else:
        metrics["has_step_ns"] = False
        warnings.append("missing_step_ns")


def _check_diagnostics(
    f: h5py.File,
    *,
    errors: list[str],
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    if "diagnostics" not in f:
        errors.append("missing_diagnostics_group")
        return
    keys = set(f["diagnostics"].keys())
    missing = sorted(REQUIRED_DIAGNOSTICS - keys)
    if missing:
        errors.append("missing_required_diagnostics:" + ",".join(missing))
    if "controller_ack" in f["diagnostics"]:
        ack = np.asarray(f["diagnostics/controller_ack"][()], dtype=np.float32)
        metrics["controller_ack_rate"] = float(ack.mean()) if ack.size else 0.0
        if ack.size and np.any(ack <= 0):
            errors.append("controller_ack_false")
    if "receiver_health_ok" in f["diagnostics"]:
        health = np.asarray(f["diagnostics/receiver_health_ok"][()], dtype=np.float32)
        metrics["receiver_health_ok_rate"] = float(health.mean()) if health.size else 0.0
        if health.size and np.any(health <= 0):
            errors.append("receiver_health_not_ok")
    if "controller_fault_code" in f["diagnostics"]:
        faults = _read_string_dataset(f["diagnostics/controller_fault_code"])
        nonempty = [fault for fault in faults if fault]
        if nonempty:
            errors.append("controller_fault_code_present")
            metrics["controller_fault_samples"] = len(nonempty)
    else:
        warnings.append("missing_controller_fault_code")


def _check_images(
    f: h5py.File,
    *,
    errors: list[str],
    warnings: list[str],
    metrics: dict[str, Any],
    max_sampled_frames: int,
    black_mean_threshold: float,
    duplicate_warn_ratio: float,
) -> None:
    raw_group = f.get("observations/images")
    encoded_group = f.get("observations/encoded_images")
    if raw_group is None and encoded_group is None:
        errors.append("missing_fpv_images")
        return
    group = raw_group if raw_group is not None else encoded_group
    assert group is not None
    if "fpv" not in group:
        errors.append("missing_fpv_camera")
        return
    dataset = group["fpv"]
    frame_count = int(dataset.shape[0]) if dataset.shape else 0
    metrics["fpv_frame_count"] = frame_count
    if frame_count <= 0:
        errors.append("fpv_empty")
        return
    indices = _sample_indices(frame_count, max_sampled_frames)
    means: list[float] = []
    fingerprints: list[bytes] = []
    bad_frames = 0
    for idx in indices:
        try:
            frame = dataset[idx]
            if raw_group is not None:
                arr = np.asarray(frame, dtype=np.uint8)
            else:
                arr = _decode_jpeg(np.asarray(frame, dtype=np.uint8).reshape(-1))
            means.append(float(arr.mean()) if arr.size else 0.0)
            fingerprints.append(_fingerprint(arr))
        except Exception:
            bad_frames += 1
    metrics["fpv_sample_count"] = len(indices)
    metrics["fpv_bad_sample_count"] = bad_frames
    if bad_frames:
        errors.append("fpv_bad_frames")
    if means:
        black_ratio = float(np.mean(np.asarray(means) <= black_mean_threshold))
        duplicate_ratio = _duplicate_ratio(fingerprints)
        metrics["fpv_black_sample_ratio"] = black_ratio
        metrics["fpv_duplicate_sample_ratio"] = duplicate_ratio
        if black_ratio > 0.0:
            errors.append("fpv_black_frames")
        if duplicate_ratio >= duplicate_warn_ratio:
            warnings.append("fpv_duplicate_frames_high")


def _sample_indices(count: int, limit: int) -> list[int]:
    if count <= 0:
        return []
    limit = max(1, int(limit))
    if count <= limit:
        return list(range(count))
    return sorted(set(np.linspace(0, count - 1, limit, dtype=np.int64).tolist()))


def _decode_jpeg(data: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required to decode JPEG frames for QC") from exc
    bgr = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode returned None")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _fingerprint(arr: np.ndarray) -> bytes:
    flat = np.asarray(arr, dtype=np.uint8).reshape(-1)
    if flat.size <= 256:
        return flat.tobytes()
    return flat[:: max(1, flat.size // 256)][:256].tobytes()


def _duplicate_ratio(items: list[bytes]) -> float:
    if len(items) <= 1:
        return 0.0
    same = sum(1 for a, b in zip(items, items[1:]) if a == b)
    return float(same / (len(items) - 1))


def _read_string_dataset(dataset: h5py.Dataset) -> list[str]:
    data = np.asarray(dataset[()])
    out: list[str] = []
    for item in data.reshape(-1):
        if isinstance(item, bytes):
            out.append(item.decode())
        else:
            out.append(str(item))
    return out


def _bool_attr(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return str(value).strip().lower() in {"true", "yes", "ok"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value
