"""Online training-usability QC for real-machine recording."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from testbed.data.bucket_semantic import (
    BUCKET_SEMANTIC_FEATURES,
    bucket_semantic_decision,
    bucket_semantic_features_from_qpos,
)


ONLINE_QC_PASS = "PASS"
ONLINE_QC_WARN_MASK = "WARN_MASK"
ONLINE_QC_FAIL_EPISODE = "FAIL_EPISODE"


@dataclass(frozen=True)
class OnlineQcConfig:
    enabled: bool = True
    reference: dict[str, Any] | None = None
    reference_path: str | None = None
    mask_backfill_window_steps: int = 5
    qpos_warn_consecutive_steps: int = 5
    qpos_fail_consecutive_steps: int = 25
    qpos_distribution_hard_fail: bool = False
    qpos_jump_fail_rad: float = 0.20
    imu_qpos_delta_warn_consecutive_steps: int = 5
    imu_qpos_delta_fail_consecutive_steps: int = 25
    max_policy_raw_qpos_delta_rad: Any = (0.08, 0.08, 0.08, 0.08)
    fpv_sample_interval_steps: int = 5
    fpv_drift_warn_consecutive_samples: int = 5
    fpv_drift_fail_consecutive_samples: int = 25
    fpv_drift_hard_fail: bool = False
    fpv_black_mean_threshold: float = 5.0
    bucket_reference_margin_rad: float = 0.25
    bucket_semantic_enabled: bool = True
    bucket_semantic_review_is_train_ready: bool = True
    bucket_semantic_min_reference_count: int = 5
    min_episode_steps: int = 300
    min_healthy_steps: int = 300
    min_healthy_fraction: float = 0.30

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None) -> OnlineQcConfig:
        cfg = dict(cfg or {})
        reference = cfg.get("reference")
        reference_path = cfg.get("reference_path")
        if reference is None and reference_path:
            path = Path(str(reference_path))
            if path.exists():
                reference = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            reference=reference if isinstance(reference, dict) else None,
            reference_path=str(reference_path) if reference_path else None,
            mask_backfill_window_steps=int(cfg.get("mask_backfill_window_steps", 5)),
            qpos_warn_consecutive_steps=int(cfg.get("qpos_warn_consecutive_steps", 5)),
            qpos_fail_consecutive_steps=int(cfg.get("qpos_fail_consecutive_steps", 25)),
            qpos_distribution_hard_fail=bool(
                cfg.get("qpos_distribution_hard_fail", False)
            ),
            qpos_jump_fail_rad=float(cfg.get("qpos_jump_fail_rad", 0.20)),
            imu_qpos_delta_warn_consecutive_steps=int(
                cfg.get("imu_qpos_delta_warn_consecutive_steps", 5)
            ),
            imu_qpos_delta_fail_consecutive_steps=int(
                cfg.get("imu_qpos_delta_fail_consecutive_steps", 25)
            ),
            max_policy_raw_qpos_delta_rad=cfg.get(
                "max_policy_raw_qpos_delta_rad", (0.08, 0.08, 0.08, 0.08)
            ),
            fpv_sample_interval_steps=int(cfg.get("fpv_sample_interval_steps", 5)),
            fpv_drift_warn_consecutive_samples=int(
                cfg.get("fpv_drift_warn_consecutive_samples", 5)
            ),
            fpv_drift_fail_consecutive_samples=int(
                cfg.get("fpv_drift_fail_consecutive_samples", 25)
            ),
            fpv_drift_hard_fail=bool(cfg.get("fpv_drift_hard_fail", False)),
            fpv_black_mean_threshold=float(cfg.get("fpv_black_mean_threshold", 5.0)),
            bucket_reference_margin_rad=float(
                cfg.get("bucket_reference_margin_rad", 0.25)
            ),
            bucket_semantic_enabled=bool(cfg.get("bucket_semantic_enabled", True)),
            bucket_semantic_review_is_train_ready=bool(
                cfg.get("bucket_semantic_review_is_train_ready", True)
            ),
            bucket_semantic_min_reference_count=int(
                cfg.get("bucket_semantic_min_reference_count", 5)
            ),
            min_episode_steps=int(cfg.get("min_episode_steps", 300)),
            min_healthy_steps=int(cfg.get("min_healthy_steps", 300)),
            min_healthy_fraction=float(cfg.get("min_healthy_fraction", 0.30)),
        )


@dataclass(frozen=True)
class OnlineQcSnapshot:
    status: str
    error_code: str
    warning_codes: tuple[str, ...] = ()
    train_exclude: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


class OnlineTrainingQcEvaluator:
    """Evaluate whether the current frame is usable for training."""

    def __init__(self, config: OnlineQcConfig | None = None) -> None:
        self.config = config or OnlineQcConfig()
        self.reference = dict(self.config.reference or {})
        self._last_qpos: np.ndarray | None = None
        self._qpos_warn_count = 0
        self._qpos_fail_count = 0
        self._imu_qpos_delta_count = 0
        self._step_count = 0
        self._fpv_drift_count = 0
        self._last_fpv_hash: int | None = None
        self._train_exclude_steps = 0
        self._current_healthy_run = 0
        self._max_healthy_run = 0
        self._semantic_qpos_rows: list[np.ndarray] = []

    def evaluate(
        self,
        *,
        obs: dict[str, Any],
        now_ns: int | None = None,
        semantic_sample: bool = True,
    ) -> OnlineQcSnapshot:
        del now_ns
        self._step_count += 1
        qpos = _vector4(obs.get("qpos"))
        if bool(semantic_sample):
            self._semantic_qpos_rows.append(qpos.copy())
        errors: list[str] = []
        warnings: list[str] = []
        train_exclude = False

        if self._last_qpos is not None:
            jump = np.abs(qpos - self._last_qpos)
            if bool(np.any(jump > float(self.config.qpos_jump_fail_rad))):
                errors.append("qpos_jump")
                train_exclude = True
        self._last_qpos = qpos.copy()

        qpos_ref = _qpos_reference(self.reference)
        outside_p5_p95 = False
        outside_p1_p99 = False
        if qpos_ref is None:
            warnings.append("reference_missing")
        else:
            outside_p5_p95 = bool(
                np.any(qpos < qpos_ref["p5"]) or np.any(qpos > qpos_ref["p95"])
            )
            outside_p1_p99 = bool(
                np.any(qpos < qpos_ref["p1"]) or np.any(qpos > qpos_ref["p99"])
            )

        self._qpos_warn_count = self._next_count(
            self._qpos_warn_count, outside_p5_p95
        )
        self._qpos_fail_count = self._next_count(
            self._qpos_fail_count, outside_p1_p99
        )

        if self._qpos_fail_count >= int(self.config.qpos_fail_consecutive_steps):
            train_exclude = True
            if self.config.qpos_distribution_hard_fail:
                errors.append("qpos_outside_p1_p99")
            else:
                warnings.append("qpos_outside_p1_p99")
        elif self._qpos_warn_count >= int(self.config.qpos_warn_consecutive_steps):
            warnings.append("qpos_outside_p5_p95")
            train_exclude = True

        imu_delta = np.zeros(4, dtype=np.float32)
        if "qpos_raw_imu" not in obs and "qpos_raw_imu_deg" not in obs:
            warnings.append("imu_qpos_reference_missing")
            self._imu_qpos_delta_count = 0
        else:
            raw_imu = _raw_imu_qpos(obs)
            imu_delta = np.abs(qpos - raw_imu)
            delta_threshold = _vector4(self.config.max_policy_raw_qpos_delta_rad)
            delta_high = bool(np.any(imu_delta > delta_threshold))
            self._imu_qpos_delta_count = self._next_count(
                self._imu_qpos_delta_count, delta_high
            )
            if self._imu_qpos_delta_count >= int(
                self.config.imu_qpos_delta_fail_consecutive_steps
            ):
                errors.append("imu_qpos_delta_high")
                train_exclude = True
            elif self._imu_qpos_delta_count >= int(
                self.config.imu_qpos_delta_warn_consecutive_steps
            ):
                warnings.append("imu_qpos_delta_high")
                train_exclude = True

        fpv_metrics = {
            "sampled": 0,
            "brightness": 0.0,
            "contrast": 0.0,
            "jpeg_size": 0.0,
            "drift_score": 0.0,
        }
        if self._should_sample_fpv():
            fpv_status = self._evaluate_fpv(obs)
            fpv_metrics.update(fpv_status["metrics"])
            if fpv_status["error"]:
                errors.append(str(fpv_status["error"]))
                train_exclude = True
            elif fpv_status["drift"]:
                self._fpv_drift_count += 1
                if self._fpv_drift_count >= int(
                    self.config.fpv_drift_fail_consecutive_samples
                ):
                    if self.config.fpv_drift_hard_fail:
                        errors.append("fpv_drift")
                    else:
                        warnings.append("fpv_drift")
                    train_exclude = True
                elif self._fpv_drift_count >= int(
                    self.config.fpv_drift_warn_consecutive_samples
                ):
                    warnings.append("fpv_drift")
                    train_exclude = True
            else:
                self._fpv_drift_count = 0

        status = ONLINE_QC_PASS
        error_code = ""
        if errors:
            status = ONLINE_QC_FAIL_EPISODE
            error_code = errors[0]
        elif train_exclude or warnings:
            status = ONLINE_QC_WARN_MASK if train_exclude else ONLINE_QC_PASS
        self._update_episode_counts(train_exclude=bool(train_exclude))

        diagnostics = {
            "online_qc_status": status,
            "online_qc_error_code": error_code,
            "online_qc_warning_codes": ",".join(warnings),
            "online_qc_train_exclude": int(train_exclude),
            "train_exclude_mask": int(train_exclude),
            "online_qc_qpos_warn_count": int(self._qpos_warn_count),
            "online_qc_qpos_fail_count": int(self._qpos_fail_count),
            "online_qc_imu_qpos_delta": imu_delta.astype(np.float32, copy=True),
            "online_qc_imu_qpos_delta_count": int(self._imu_qpos_delta_count),
            "online_qc_fpv_sampled": int(fpv_metrics["sampled"]),
            "online_qc_fpv_brightness": float(fpv_metrics["brightness"]),
            "online_qc_fpv_contrast": float(fpv_metrics["contrast"]),
            "online_qc_fpv_jpeg_size": float(fpv_metrics["jpeg_size"]),
            "online_qc_fpv_drift_score": float(fpv_metrics["drift_score"]),
            "online_qc_fpv_drift_count": int(self._fpv_drift_count),
            **self._episode_diagnostics(),
        }
        return OnlineQcSnapshot(
            status=status,
            error_code=error_code,
            warning_codes=tuple(warnings),
            train_exclude=bool(train_exclude),
            diagnostics=diagnostics,
        )

    def finalize_episode(self, *, recorded_steps: int | None = None) -> OnlineQcSnapshot:
        total_steps = int(self._step_count if recorded_steps is None else recorded_steps)
        healthy_steps = max(0, total_steps - int(self._train_exclude_steps))
        min_episode_steps = int(self.config.min_episode_steps)
        min_healthy_steps = int(self.config.min_healthy_steps)
        min_healthy_fraction = float(self.config.min_healthy_fraction)
        healthy_fraction = float(healthy_steps / total_steps) if total_steps > 0 else 0.0

        error_code = ""
        if total_steps < min_episode_steps:
            error_code = "episode_too_short"
        elif healthy_steps < min_healthy_steps or healthy_fraction < min_healthy_fraction:
            error_code = "insufficient_healthy_steps"
        warnings: list[str] = []
        final_train_exclude = bool(error_code)
        train_ready_candidate = int(not bool(error_code))

        diagnostics = self._episode_diagnostics(total_steps=total_steps)
        bucket_reference = self._bucket_reference_final_status()
        diagnostics.update(bucket_reference)
        bucket_semantic = self._bucket_semantic_final_status(
            bucket_reference_status=str(
                bucket_reference["online_qc_bucket_reference_status"]
            )
        )
        diagnostics.update(bucket_semantic)
        if not error_code:
            if bucket_reference["online_qc_bucket_reference_status"] == "FAIL":
                error_code = "bucket_reference_outlier"
                final_train_exclude = True
                train_ready_candidate = 0
            elif bucket_reference["online_qc_bucket_reference_status"] == "WARN":
                decision = str(
                    bucket_semantic.get("online_qc_bucket_semantic_decision", "")
                )
                if decision == "drop":
                    error_code = "bucket_semantic_outlier"
                    final_train_exclude = True
                    train_ready_candidate = 0
                elif decision == "review":
                    warnings.append("bucket_semantic_review")
                    final_train_exclude = True
                    train_ready_candidate = int(
                        self.config.bucket_semantic_review_is_train_ready
                    )
                elif decision == "keep":
                    train_ready_candidate = 1
                else:
                    warnings.append("bucket_semantic_reference_missing")
                    final_train_exclude = True
                    train_ready_candidate = 0
            elif bucket_reference["online_qc_bucket_reference_status"] == "UNKNOWN":
                warnings.append("bucket_reference_missing")
        if error_code:
            status = ONLINE_QC_FAIL_EPISODE
        elif final_train_exclude:
            status = ONLINE_QC_WARN_MASK
        else:
            status = ONLINE_QC_PASS
        diagnostics.update(
            {
                "online_qc_status": status,
                "online_qc_error_code": error_code,
                "online_qc_warning_codes": ",".join(warnings),
                "online_qc_train_exclude": int(final_train_exclude),
                "train_exclude_mask": int(final_train_exclude),
                "online_qc_train_ready_candidate": int(train_ready_candidate),
                "online_qc_reference_id": str(self.reference.get("reference_id", "")),
            }
        )
        return OnlineQcSnapshot(
            status=status,
            error_code=error_code,
            warning_codes=tuple(warnings),
            train_exclude=bool(final_train_exclude),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _next_count(current: int, active: bool) -> int:
        return int(current) + 1 if active else 0

    def _update_episode_counts(self, *, train_exclude: bool) -> None:
        if train_exclude:
            self._train_exclude_steps += 1
            self._current_healthy_run = 0
            return
        self._current_healthy_run += 1
        self._max_healthy_run = max(self._max_healthy_run, self._current_healthy_run)

    def _episode_diagnostics(self, *, total_steps: int | None = None) -> dict[str, Any]:
        total = int(self._step_count if total_steps is None else total_steps)
        healthy = max(0, total - int(self._train_exclude_steps))
        return {
            "online_qc_total_steps": total,
            "online_qc_train_exclude_steps": int(self._train_exclude_steps),
            "online_qc_healthy_steps": int(healthy),
            "online_qc_max_healthy_run": int(self._max_healthy_run),
            "online_qc_healthy_fraction": float(healthy / total) if total > 0 else 0.0,
        }

    def _bucket_reference_final_status(self) -> dict[str, Any]:
        bucket_ref = self.reference.get("bucket_qpos", {})
        p1 = bucket_ref.get("p1") if isinstance(bucket_ref, dict) else None
        p99 = bucket_ref.get("p99") if isinstance(bucket_ref, dict) else None
        if p1 is None or p99 is None or not self._semantic_qpos_rows:
            return {
                "online_qc_bucket_reference_status": "UNKNOWN",
                "online_qc_bucket_ref_low_margin": 0.0,
                "online_qc_bucket_ref_high_margin": 0.0,
            }
        qpos = np.stack(self._semantic_qpos_rows).astype(np.float64, copy=False)
        bucket = qpos[:, 3]
        low_margin = float(np.min(bucket) - float(p1))
        high_margin = float(float(p99) - np.max(bucket))
        hard_margin = float(self.config.bucket_reference_margin_rad)
        if low_margin < -hard_margin or high_margin < -hard_margin:
            status = "FAIL"
        elif low_margin < 0.0 or high_margin < 0.0:
            status = "WARN"
        else:
            status = "PASS"
        return {
            "online_qc_bucket_reference_status": status,
            "online_qc_bucket_ref_low_margin": low_margin,
            "online_qc_bucket_ref_high_margin": high_margin,
        }

    def _bucket_semantic_final_status(
        self,
        *,
        bucket_reference_status: str,
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "online_qc_bucket_semantic_decision": "",
            "online_qc_bucket_semantic_notes": "",
            "online_qc_bucket_semantic_info": "",
            "online_qc_bucket_semantic_reference_count": 0,
        }
        for key in BUCKET_SEMANTIC_FEATURES:
            diagnostics[f"online_qc_bucket_semantic_{key}"] = 0.0
        if (
            bucket_reference_status != "WARN"
            or not self.config.bucket_semantic_enabled
        ):
            return diagnostics
        semantic_ref = self.reference.get("bucket_semantic", {})
        if not isinstance(semantic_ref, dict):
            return diagnostics
        ref_count = int(semantic_ref.get("count", 0) or 0)
        diagnostics["online_qc_bucket_semantic_reference_count"] = ref_count
        if ref_count < int(self.config.bucket_semantic_min_reference_count):
            return diagnostics
        if not self._semantic_qpos_rows:
            return diagnostics
        qpos = np.stack(self._semantic_qpos_rows).astype(np.float64, copy=False)
        features = bucket_semantic_features_from_qpos(
            qpos,
            manual_end_index=int(qpos.shape[0]),
        )
        decision, notes = bucket_semantic_decision(features, semantic_ref)
        diagnostics["online_qc_bucket_semantic_decision"] = decision
        diagnostics["online_qc_bucket_semantic_notes"] = ";".join(notes)
        if decision == "keep":
            diagnostics["online_qc_bucket_semantic_info"] = (
                "bucket_reference_semantic_keep"
            )
        for key, value in features.items():
            diagnostics[f"online_qc_bucket_semantic_{key}"] = float(value)
        return diagnostics

    def _should_sample_fpv(self) -> bool:
        interval = max(1, int(self.config.fpv_sample_interval_steps))
        return (self._step_count - 1) % interval == 0

    def _evaluate_fpv(self, obs: dict[str, Any]) -> dict[str, Any]:
        try:
            frame, jpeg_size = _fpv_frame(obs)
        except Exception:
            return {
                "error": "fpv_decode_failed",
                "drift": False,
                "metrics": {
                    "sampled": 1,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "jpeg_size": 0.0,
                    "drift_score": 0.0,
                },
            }

        brightness = float(np.mean(frame)) if frame.size else 0.0
        contrast = float(np.std(frame)) if frame.size else 0.0
        fingerprint = _fingerprint(frame)
        frame_hash = hash(np.asarray(frame, dtype=np.uint8).tobytes())
        drift_score = _fpv_drift_score(
            reference=self.reference,
            brightness=brightness,
            contrast=contrast,
            jpeg_size=jpeg_size,
            fingerprint=fingerprint,
        )
        metrics = {
            "sampled": 1,
            "brightness": brightness,
            "contrast": contrast,
            "jpeg_size": float(jpeg_size),
            "drift_score": float(drift_score),
        }
        if brightness <= float(self.config.fpv_black_mean_threshold):
            return {"error": "fpv_black", "drift": False, "metrics": metrics}
        if self._last_fpv_hash is not None and frame_hash == self._last_fpv_hash:
            return {"error": "fpv_duplicate", "drift": False, "metrics": metrics}
        self._last_fpv_hash = frame_hash
        return {"error": "", "drift": drift_score >= 1.0, "metrics": metrics}


def _qpos_reference(reference: dict[str, Any]) -> dict[str, np.ndarray] | None:
    qpos = reference.get("qpos")
    if not isinstance(qpos, dict):
        return None
    out: dict[str, np.ndarray] = {}
    for key in ("p1", "p5", "p95", "p99"):
        if key not in qpos:
            return None
        out[key] = _vector4(qpos[key])
    return out


def _vector4(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        padded = np.zeros(4, dtype=np.float32)
        padded[: arr.size] = arr
        return padded
    return arr[:4].astype(np.float32, copy=False)


def _raw_imu_qpos(obs: dict[str, Any]) -> np.ndarray:
    if "qpos_raw_imu" in obs:
        return _vector4(obs.get("qpos_raw_imu"))
    return np.deg2rad(_vector4(obs.get("qpos_raw_imu_deg"))).astype(np.float32)


def _fpv_frame(obs: dict[str, Any]) -> tuple[np.ndarray, float]:
    images = obs.get("images")
    if isinstance(images, dict) and "fpv" in images:
        frame = np.asarray(images["fpv"], dtype=np.uint8)
        if frame.size == 0:
            raise ValueError("empty fpv image")
        return frame, float(frame.nbytes)

    encoded_images = obs.get("encoded_images")
    if isinstance(encoded_images, dict) and "fpv" in encoded_images:
        payload = encoded_images["fpv"]
        if isinstance(payload, dict):
            data = payload.get("data", payload.get("bytes", b""))
        else:
            data = payload
        encoded = np.asarray(data, dtype=np.uint8).reshape(-1)
        return _decode_jpeg(encoded), float(encoded.size)

    raise ValueError("missing fpv image")


def _decode_jpeg(data: np.ndarray) -> np.ndarray:
    import cv2

    bgr = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode fpv jpeg")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _fingerprint(frame: np.ndarray) -> np.ndarray:
    gray = np.asarray(frame, dtype=np.float32)
    if gray.ndim == 3:
        gray = np.mean(gray, axis=2)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(64, dtype=np.float32)
    y_edges = np.linspace(0, h, 9, dtype=np.int32)
    x_edges = np.linspace(0, w, 9, dtype=np.int32)
    values: list[float] = []
    for yi in range(8):
        for xi in range(8):
            patch = gray[y_edges[yi] : y_edges[yi + 1], x_edges[xi] : x_edges[xi + 1]]
            values.append(float(np.mean(patch)) if patch.size else 0.0)
    return np.asarray(values, dtype=np.float32)


def _fpv_drift_score(
    *,
    reference: dict[str, Any],
    brightness: float,
    contrast: float,
    jpeg_size: float,
    fingerprint: np.ndarray,
) -> float:
    fpv = reference.get("fpv")
    if not isinstance(fpv, dict):
        return 0.0
    scores = [
        _scalar_outlier_score(brightness, fpv.get("brightness")),
        _scalar_outlier_score(contrast, fpv.get("contrast")),
        _scalar_outlier_score(jpeg_size, fpv.get("jpeg_size")),
    ]
    ref_fingerprint = fpv.get("fingerprint")
    if ref_fingerprint is not None:
        ref = np.asarray(ref_fingerprint, dtype=np.float32).reshape(-1)
        if ref.size == fingerprint.size and ref.size:
            scores.append(float(np.mean(np.abs(fingerprint - ref)) / 25.0))
    return float(max(scores)) if scores else 0.0


def _scalar_outlier_score(value: float, stats: Any) -> float:
    if not isinstance(stats, dict):
        return 0.0
    p1 = stats.get("p1")
    p99 = stats.get("p99")
    if p1 is not None and float(value) < float(p1):
        scale = max(1.0, float(stats.get("mad", 1.0) or 1.0))
        return float((float(p1) - float(value)) / scale)
    if p99 is not None and float(value) > float(p99):
        scale = max(1.0, float(stats.get("mad", 1.0) or 1.0))
        return float((float(value) - float(p99)) / scale)
    return 0.0
