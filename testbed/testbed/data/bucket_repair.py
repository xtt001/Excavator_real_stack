"""Bucket qpos repair for real-excavator HDF5 episodes.

The repair keeps trusted samples from the recorded policy qpos and only
reconstructs discontinuous spans with qvel integration between trusted anchors.
Original files are never modified by these helpers.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


BUCKET_AXIS = 3
REPAIR_GROUP = "repairs/bucket_qpos_v1"
REPAIR_VERSION = "bucket_qpos_repair_v1"


@dataclass(frozen=True)
class BucketRepairConfig:
    jump_threshold_rad: float = 0.20
    expected_delta_abs_tol_rad: float = 0.12
    expected_delta_scale: float = 8.0
    expand_bad_steps: int = 2
    min_anchor_count: int = 2


@dataclass(frozen=True)
class EpisodeRepairResult:
    episode_id: int
    input_path: Path
    output_path: Path
    repaired: bool
    degraded: bool
    source_imu_log: str
    bad_sample_count: int
    repaired_sample_count: int
    max_abs_jump_before: float
    max_abs_jump_after: float
    max_abs_delta_from_original: float


def repair_dataset(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    imu_log_dir: str | Path | None = None,
    episode_ids: list[int] | None = None,
    config: BucketRepairConfig | None = None,
) -> dict[str, Any]:
    """Copy and repair selected episodes from ``input_dir`` into ``output_dir``."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or BucketRepairConfig()
    ids = episode_ids if episode_ids is not None else list(range(26, 66))
    results: list[EpisodeRepairResult] = []
    for episode_id in ids:
        src = input_dir / f"episode_{episode_id}.hdf5"
        if not src.exists():
            continue
        dst = output_dir / src.name
        results.append(
            repair_episode(
                src,
                dst,
                imu_log_dir=imu_log_dir,
                config=cfg,
            )
        )
    summary = {
        "schema_version": 1,
        "repair_version": REPAIR_VERSION,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "imu_log_dir": "" if imu_log_dir is None else str(imu_log_dir),
        "episodes": [_result_to_dict(result) for result in results],
    }
    with (output_dir / "bucket_repair_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def repair_episode(
    input_path: str | Path,
    output_path: str | Path,
    *,
    imu_log_dir: str | Path | None = None,
    config: BucketRepairConfig | None = None,
) -> EpisodeRepairResult:
    """Copy one HDF5 episode and repair bucket qpos in the copy."""

    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    cfg = config or BucketRepairConfig()

    with h5py.File(dst, "r+") as f:
        qpos = np.asarray(f["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(f["observations/qvel"][()], dtype=np.float32)
        timestamps_ns = _episode_timestamps_ns(f)
        original_bucket = qpos[:, BUCKET_AXIS].astype(np.float64, copy=True)
        repaired_bucket, repair_mask, bad_mask, degraded = repair_bucket_series(
            bucket_qpos=original_bucket,
            bucket_qvel=qvel[:, BUCKET_AXIS],
            timestamps_ns=timestamps_ns,
            config=cfg,
        )
        qpos[:, BUCKET_AXIS] = repaired_bucket.astype(np.float32)
        f["observations/qpos"][...] = qpos
        if "observations/env_state" in f:
            env_state = np.asarray(f["observations/env_state"][()], dtype=np.float32)
            if env_state.ndim == 2 and env_state.shape[1] > BUCKET_AXIS:
                env_state[:, BUCKET_AXIS] = qpos[:, BUCKET_AXIS]
                f["observations/env_state"][...] = env_state

        source_imu_log = _best_effort_imu_log_match(
            f,
            imu_log_dir=None if imu_log_dir is None else Path(imu_log_dir),
        )
        _write_repair_group(
            f,
            original_bucket=original_bucket,
            repaired_bucket=repaired_bucket,
            repair_mask=repair_mask,
            bad_mask=bad_mask,
            source_imu_log=source_imu_log,
            degraded=degraded or source_imu_log == "",
        )

    max_jump_before = _max_abs_step(original_bucket)
    max_jump_after = _max_abs_step(repaired_bucket)
    max_delta = float(np.max(np.abs(repaired_bucket - original_bucket))) if original_bucket.size else 0.0
    episode_id = _episode_id_from_path(src)
    return EpisodeRepairResult(
        episode_id=episode_id,
        input_path=src,
        output_path=dst,
        repaired=bool(np.any(repair_mask)),
        degraded=bool(degraded or source_imu_log == ""),
        source_imu_log=source_imu_log,
        bad_sample_count=int(np.sum(bad_mask)),
        repaired_sample_count=int(np.sum(repair_mask)),
        max_abs_jump_before=max_jump_before,
        max_abs_jump_after=max_jump_after,
        max_abs_delta_from_original=max_delta,
    )


def repair_bucket_series(
    *,
    bucket_qpos: np.ndarray,
    bucket_qvel: np.ndarray,
    timestamps_ns: np.ndarray | None,
    config: BucketRepairConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return repaired bucket qpos, repaired mask, bad mask, degraded flag."""

    cfg = config or BucketRepairConfig()
    qpos = np.asarray(bucket_qpos, dtype=np.float64).reshape(-1)
    qvel = np.asarray(bucket_qvel, dtype=np.float64).reshape(-1)
    if qpos.shape != qvel.shape:
        raise ValueError(f"bucket qpos/qvel shape mismatch: {qpos.shape} vs {qvel.shape}")
    n = qpos.size
    if n == 0:
        return qpos.copy(), np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), False
    if n == 1:
        return qpos.copy(), np.zeros(1, dtype=bool), np.zeros(1, dtype=bool), False

    dt = _dt_seconds(timestamps_ns, n)
    raw_delta = np.diff(qpos)
    expected_delta = 0.5 * (qvel[:-1] + qvel[1:]) * dt
    allowed = np.maximum(
        float(cfg.expected_delta_abs_tol_rad),
        np.abs(expected_delta) * float(cfg.expected_delta_scale) + 0.02,
    )
    bad_boundary = (
        np.abs(raw_delta) > float(cfg.jump_threshold_rad)
    ) | (np.abs(raw_delta - expected_delta) > allowed)

    bad_mask = np.zeros(n, dtype=bool)
    bad_indices = np.flatnonzero(bad_boundary)
    for first, second in zip(bad_indices[:-1], bad_indices[1:]):
        # A branch excursion commonly appears as one large jump onto the wrong
        # branch followed by a large jump back. The samples between those two
        # boundaries can be locally smooth, so boundary expansion alone would
        # leave the middle of the wrong branch marked trusted.
        if second > first and second - first < max(10, n // 2):
            bad_mask[int(first) + 1 : int(second) + 1] = True
    for boundary in bad_indices:
        start = max(0, int(boundary) - int(cfg.expand_bad_steps) + 1)
        end = min(n, int(boundary) + int(cfg.expand_bad_steps) + 2)
        bad_mask[start:end] = True

    if not np.any(bad_mask):
        return qpos.copy(), np.zeros(n, dtype=bool), bad_mask, False

    trusted = ~bad_mask
    degraded = int(np.sum(trusted)) < int(cfg.min_anchor_count)
    repaired = qpos.copy()
    if degraded:
        repaired = _integrate_from_anchor(qpos, qvel, dt, anchor_index=0)
        repair_mask = np.ones(n, dtype=bool)
        repair_mask[0] = False
        return repaired, repair_mask, bad_mask, True

    trusted_indices = np.flatnonzero(trusted)
    repair_mask = bad_mask.copy()
    for left, right in zip(trusted_indices[:-1], trusted_indices[1:]):
        if right <= left + 1:
            continue
        if not np.any(bad_mask[left + 1 : right]):
            continue
        segment = _integrate_between_anchors(
            left_value=float(repaired[left]),
            right_value=float(repaired[right]),
            qvel=qvel,
            dt=dt,
            left=int(left),
            right=int(right),
        )
        repaired[left : right + 1] = segment
        repair_mask[left + 1 : right] = True

    first_anchor = int(trusted_indices[0])
    if first_anchor > 0:
        prefix = _integrate_backward_from_anchor(qpos, qvel, dt, anchor_index=first_anchor)
        repaired[: first_anchor + 1] = prefix
        repair_mask[:first_anchor] = True

    last_anchor = int(trusted_indices[-1])
    if last_anchor < n - 1:
        suffix = _integrate_forward_from_anchor(qpos, qvel, dt, anchor_index=last_anchor)
        repaired[last_anchor:] = suffix
        repair_mask[last_anchor + 1 :] = True

    return repaired, repair_mask, bad_mask, False


def _integrate_between_anchors(
    *,
    left_value: float,
    right_value: float,
    qvel: np.ndarray,
    dt: np.ndarray,
    left: int,
    right: int,
) -> np.ndarray:
    values = np.empty(right - left + 1, dtype=np.float64)
    values[0] = left_value
    for out_idx, src_idx in enumerate(range(left + 1, right + 1), start=1):
        values[out_idx] = values[out_idx - 1] + 0.5 * (
            qvel[src_idx - 1] + qvel[src_idx]
        ) * dt[src_idx - 1]
    drift = right_value - values[-1]
    values += np.linspace(0.0, drift, values.size)
    return values


def _integrate_from_anchor(
    qpos: np.ndarray,
    qvel: np.ndarray,
    dt: np.ndarray,
    *,
    anchor_index: int,
) -> np.ndarray:
    values = _integrate_forward_from_anchor(qpos, qvel, dt, anchor_index=anchor_index)
    if anchor_index > 0:
        prefix = _integrate_backward_from_anchor(qpos, qvel, dt, anchor_index=anchor_index)
        values[: anchor_index + 1] = prefix
    return values


def _integrate_forward_from_anchor(
    qpos: np.ndarray,
    qvel: np.ndarray,
    dt: np.ndarray,
    *,
    anchor_index: int,
) -> np.ndarray:
    values = qpos.copy()
    for i in range(anchor_index + 1, qpos.size):
        values[i] = values[i - 1] + 0.5 * (qvel[i - 1] + qvel[i]) * dt[i - 1]
    return values[anchor_index:]


def _integrate_backward_from_anchor(
    qpos: np.ndarray,
    qvel: np.ndarray,
    dt: np.ndarray,
    *,
    anchor_index: int,
) -> np.ndarray:
    values = qpos.copy()
    for i in range(anchor_index - 1, -1, -1):
        values[i] = values[i + 1] - 0.5 * (qvel[i] + qvel[i + 1]) * dt[i]
    return values[: anchor_index + 1]


def _dt_seconds(timestamps_ns: np.ndarray | None, n: int) -> np.ndarray:
    if timestamps_ns is None:
        return np.full(n - 1, 0.02, dtype=np.float64)
    ts = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    if ts.size != n or np.count_nonzero(ts > 0) < 2:
        return np.full(n - 1, 0.02, dtype=np.float64)
    dt = np.diff(ts).astype(np.float64) * 1e-9
    valid = np.isfinite(dt) & (dt > 0.0) & (dt < 1.0)
    median = float(np.median(dt[valid])) if np.any(valid) else 0.02
    dt[~valid] = median
    return dt


def _episode_timestamps_ns(f: h5py.File) -> np.ndarray | None:
    for path in (
        "diagnostics/joint_timestamp_ns",
        "timestamps/step_ns",
        "diagnostics/observation_timestamp_ns",
    ):
        if path in f:
            return np.asarray(f[path][()], dtype=np.int64)
    return None


def _write_repair_group(
    f: h5py.File,
    *,
    original_bucket: np.ndarray,
    repaired_bucket: np.ndarray,
    repair_mask: np.ndarray,
    bad_mask: np.ndarray,
    source_imu_log: str,
    degraded: bool,
) -> None:
    if REPAIR_GROUP in f:
        del f[REPAIR_GROUP]
    group = f.create_group(REPAIR_GROUP)
    group.attrs["repair_version"] = REPAIR_VERSION
    group.attrs["axis"] = "bucket"
    group.attrs["axis_index"] = BUCKET_AXIS
    group.attrs["source_imu_log"] = source_imu_log
    group.attrs["degraded"] = bool(degraded)
    group.attrs["max_abs_jump_before"] = _max_abs_step(original_bucket)
    group.attrs["max_abs_jump_after"] = _max_abs_step(repaired_bucket)
    group.attrs["max_abs_delta_from_original"] = (
        float(np.max(np.abs(repaired_bucket - original_bucket)))
        if original_bucket.size
        else 0.0
    )
    group.create_dataset("original_bucket_qpos", data=original_bucket.astype(np.float32))
    group.create_dataset("repaired_bucket_qpos", data=repaired_bucket.astype(np.float32))
    group.create_dataset("repair_mask", data=repair_mask.astype(np.uint8))
    group.create_dataset("bad_mask", data=bad_mask.astype(np.uint8))


def _best_effort_imu_log_match(f: h5py.File, *, imu_log_dir: Path | None) -> str:
    if imu_log_dir is None or not imu_log_dir.exists():
        return ""
    timestamps = _episode_timestamps_ns(f)
    if timestamps is None:
        return ""
    ts = np.asarray(timestamps, dtype=np.int64)
    ts = ts[ts > 0]
    if ts.size == 0:
        return ""
    ep_start = int(ts[0])
    ep_end = int(ts[-1])
    best_path = ""
    best_overlap = 0
    for jsonl in sorted(imu_log_dir.glob("imu_qvel_*.jsonl")):
        if jsonl.stat().st_size <= 0:
            continue
        first, last = _jsonl_time_span_ns(jsonl)
        if first <= 0 or last <= 0:
            continue
        overlap = max(0, min(ep_end, last) - max(ep_start, first))
        if overlap > best_overlap:
            best_overlap = overlap
            best_path = str(jsonl)
    return best_path


def _jsonl_time_span_ns(path: Path) -> tuple[int, int]:
    first = _jsonl_edge_timestamp_ns(path, from_end=False)
    last = _jsonl_edge_timestamp_ns(path, from_end=True)
    return first, last


def _jsonl_edge_timestamp_ns(path: Path, *, from_end: bool) -> int:
    try:
        if not from_end:
            with path.open("rb") as f:
                for raw in f:
                    value = _timestamp_from_jsonl_line(raw)
                    if value > 0:
                        return value
            return 0
        with path.open("rb") as f:
            f.seek(0, 2)
            pos = f.tell()
            buf = b""
            while pos > 0:
                step = min(8192, pos)
                pos -= step
                f.seek(pos)
                buf = f.read(step) + buf
                lines = buf.splitlines()
                for raw in reversed(lines):
                    value = _timestamp_from_jsonl_line(raw)
                    if value > 0:
                        return value
                if len(buf) > 1_000_000:
                    buf = buf[:8192]
            return 0
    except OSError:
        return 0


def _timestamp_from_jsonl_line(raw: bytes) -> int:
    try:
        row = json.loads(raw.decode("utf-8"))
    except Exception:
        return 0
    for key in ("joint_timestamp_ns", "sensor_timestamp_ns", "timestamp_ns"):
        value = row.get(key)
        try:
            ivalue = int(value)
        except Exception:
            continue
        if ivalue > 0:
            return ivalue
    state = row.get("state")
    if isinstance(state, dict):
        for key in ("joint_timestamp_ns", "sensor_timestamp_ns", "timestamp_ns"):
            try:
                ivalue = int(state.get(key))
            except Exception:
                continue
            if ivalue > 0:
                return ivalue
    return 0


def _max_abs_step(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(arr))))


def _episode_id_from_path(path: Path) -> int:
    try:
        return int(path.stem.split("_", 1)[1])
    except Exception:
        return -1


def _result_to_dict(result: EpisodeRepairResult) -> dict[str, Any]:
    return {
        "episode_id": int(result.episode_id),
        "input_path": str(result.input_path),
        "output_path": str(result.output_path),
        "repaired": bool(result.repaired),
        "degraded": bool(result.degraded),
        "source_imu_log": result.source_imu_log,
        "bad_sample_count": int(result.bad_sample_count),
        "repaired_sample_count": int(result.repaired_sample_count),
        "max_abs_jump_before": float(result.max_abs_jump_before),
        "max_abs_jump_after": float(result.max_abs_jump_after),
        "max_abs_delta_from_original": float(result.max_abs_delta_from_original),
    }


def parse_episode_spec(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Repair real excavator bucket qpos HDF5 data.")
    parser.add_argument("--input-dir", type=Path, default=Path("/media/mundane/EXTERNAL_USB/real_teleop_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("/media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_bucket_v1"))
    parser.add_argument("--imu-log-dir", type=Path, default=Path("/media/mundane/EXTERNAL_USB/imu_qvel_tests"))
    parser.add_argument("--episodes", default="26-65")
    args = parser.parse_args(argv)
    summary = repair_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        imu_log_dir=args.imu_log_dir,
        episode_ids=parse_episode_spec(args.episodes),
    )
    print(f"Bucket repair summary written to {args.output_dir / 'bucket_repair_summary.json'}")
    print(
        "Episodes: "
        f"{len(summary['episodes'])}, repaired="
        f"{sum(1 for row in summary['episodes'] if row['repaired'])}"
    )


if __name__ == "__main__":
    main()
