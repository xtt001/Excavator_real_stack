"""Shared bucket trajectory semantic checks for training-usability QC."""

from __future__ import annotations

from typing import Any

import numpy as np


BUCKET_AXIS = 3
BUCKET_SEMANTIC_FEATURES = (
    "start",
    "end",
    "min",
    "max",
    "range",
    "argmin",
    "argmax",
    "early_max",
    "late_max",
    "max_jump",
)


def bucket_semantic_features_from_qpos(
    qpos: np.ndarray,
    *,
    manual_end_index: int,
    bucket_axis: int = BUCKET_AXIS,
) -> dict[str, float]:
    qpos_arr = np.asarray(qpos, dtype=np.float64)
    if qpos_arr.ndim != 2 or qpos_arr.shape[0] == 0 or qpos_arr.shape[1] <= bucket_axis:
        return {key: 0.0 for key in BUCKET_SEMANTIC_FEATURES}
    end = int(manual_end_index) if manual_end_index > 0 else int(qpos_arr.shape[0])
    end = max(1, min(end, int(qpos_arr.shape[0])))
    bucket = qpos_arr[:end, bucket_axis]
    n = int(bucket.size)
    window = max(3, int(round(n * 0.05)))
    early_end = max(window, int(round(n * 0.35)))
    late_start = min(n - window, int(round(n * 0.60)))
    smoothed = moving_average(bucket, max(3, int(round(n * 0.02))))
    jump = float(np.max(np.abs(np.diff(bucket)))) if n > 1 else 0.0
    return {
        "start": float(np.median(bucket[:window])),
        "end": float(np.median(bucket[-window:])),
        "min": float(np.min(bucket)),
        "max": float(np.max(bucket)),
        "range": float(np.max(bucket) - np.min(bucket)),
        "argmin": float(np.argmin(smoothed) / max(1, n - 1)),
        "argmax": float(np.argmax(smoothed) / max(1, n - 1)),
        "early_max": float(np.max(bucket[:early_end])),
        "late_max": float(np.max(bucket[late_start:])),
        "max_jump": jump,
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    win = max(1, int(window))
    if win % 2 == 0:
        win += 1
    if arr.size < win:
        return arr
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(arr, kernel, mode="same")


def bucket_semantic_reference(
    features: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in BUCKET_SEMANTIC_FEATURES:
        values = np.asarray([item[key] for item in features], dtype=np.float64)
        out[key] = {
            "p1": float(np.percentile(values, 1)),
            "p5": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }
    return out


def bucket_semantic_decision(
    features: dict[str, float],
    reference: dict[str, Any],
) -> tuple[str, list[str]]:
    notes: list[str] = []
    end_drop_threshold = float(reference["end"]["p5"]) - 0.02
    max_drop_threshold = float(reference["max"]["p5"]) - 0.20
    late_max_drop_threshold = float(reference["late_max"]["p5"]) - 0.20
    if features["end"] < end_drop_threshold and (
        features["max"] < max_drop_threshold
        or features["late_max"] < late_max_drop_threshold
    ):
        notes.append("bucket_end_or_late_recovery_too_low")
        return "drop", notes

    shallow_min_threshold = float(reference["min"]["p99"]) + 0.10
    if features["min"] > shallow_min_threshold:
        notes.append("bucket_min_too_shallow")
    jump_threshold = max(0.18, float(reference["max_jump"]["p99"]) + 0.03)
    if features["max_jump"] > jump_threshold:
        notes.append("bucket_jump_needs_review")
    if notes:
        return "review", notes
    return "keep", notes
