"""Build a lightweight reference bundle for online training QC."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.data.camera_contract import select_primary_camera
from testbed.data.online_qc import _decode_jpeg, _fingerprint
from testbed.data.training_qc import (
    _bucket_semantic_features,
    _bucket_semantic_reference,
    _manual_end_index,
    _read_optional,
)


def build_online_qc_reference(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    training_qc_summary_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    manifest = _read_json(Path(manifest_path))
    training_qc_summary = _read_json(Path(training_qc_summary_path))
    episode_ids = _episode_ids(manifest)
    strict_pass_episode_ids = _summary_episode_ids(
        training_qc_summary, "strict_pass_episode_ids"
    )
    train_ready_episode_ids = _summary_episode_ids(
        training_qc_summary, "train_ready_episode_ids"
    )

    brightness: list[float] = []
    contrast: list[float] = []
    jpeg_size: list[float] = []
    fingerprints: list[np.ndarray] = []
    qpos_rows: list[np.ndarray] = []
    camera_names: set[str] = set()

    for episode_id in episode_ids:
        path = dataset_dir / f"episode_{episode_id}.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            qpos_rows.extend(_iter_qpos_rows(f))
            for frame, size, camera_name in _iter_fpv_frames(f):
                brightness.append(float(np.mean(frame)) if frame.size else 0.0)
                contrast.append(float(np.std(frame)) if frame.size else 0.0)
                jpeg_size.append(float(size))
                fingerprints.append(_fingerprint(frame))
                camera_names.add(camera_name)

    qpos_reference = _qpos_reference(qpos_rows)
    fingerprint = (
        np.mean(np.stack(fingerprints), axis=0).astype(float).tolist()
        if fingerprints
        else [0.0] * 64
    )
    reference = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "training_qc_summary_path": str(training_qc_summary_path),
        "episode_ids": episode_ids,
        "qpos": qpos_reference,
        "bucket_qpos": dict(
            (training_qc_summary.get("reference", {}) or {}).get("bucket_qpos", {})
        ),
        "bucket_semantic": _build_bucket_semantic_reference(
            dataset_dir=dataset_dir,
            episode_ids=strict_pass_episode_ids,
        ),
        "source_reference": {
            "manifest_path": str(manifest_path),
            "training_qc_summary_path": str(training_qc_summary_path),
            "strict_pass_episode_ids": strict_pass_episode_ids,
            "train_ready_episode_ids": train_ready_episode_ids,
        },
        "fpv": {
            "camera_names": sorted(camera_names),
            "brightness": _stats(brightness),
            "contrast": _stats(contrast),
            "jpeg_size": _stats(jpeg_size),
            "fingerprint": fingerprint,
        },
    }
    reference["reference_id"] = _reference_id(reference)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reference, indent=2, sort_keys=True), encoding="utf-8")
    return reference


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build online_qc_reference.json from train-ready real HDF5 data."
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--training-qc-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    reference = build_online_qc_reference(
        dataset_dir=args.dataset_dir,
        manifest_path=args.manifest,
        training_qc_summary_path=args.training_qc_summary,
        output_path=args.output,
    )
    print(f"wrote {args.output} reference_id={reference['reference_id']}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_ids(manifest: dict[str, Any]) -> list[int]:
    raw = manifest.get("train_ready_episode_ids", [])
    return _coerce_episode_ids(raw)


def _summary_episode_ids(summary: dict[str, Any], key: str) -> list[int]:
    return _coerce_episode_ids(summary.get(key, []))


def _coerce_episode_ids(raw: Any) -> list[int]:
    out: list[int] = []
    for item in raw:
        if isinstance(item, str) and item.startswith("episode_"):
            item = item.removeprefix("episode_")
        out.append(int(item))
    return out


def _iter_fpv_frames(f: h5py.File) -> list[tuple[np.ndarray, float, str]]:
    raw_group = f.get("observations/images")
    encoded_group = f.get("observations/encoded_images")
    camera_name = select_primary_camera(
        metadata=_metadata(f),
        raw_group=raw_group,
        encoded_group=encoded_group,
    )
    if raw_group is not None and camera_name in raw_group:
        data = raw_group[camera_name]
        return [
            (np.asarray(data[idx], dtype=np.uint8), float(data[idx].nbytes), camera_name)
            for idx in range(int(data.shape[0]))
        ]
    if encoded_group is not None and camera_name in encoded_group:
        data = encoded_group[camera_name]
        frames = []
        for idx in range(int(data.shape[0])):
            encoded = np.asarray(data[idx], dtype=np.uint8).reshape(-1)
            frames.append((_decode_jpeg(encoded), float(encoded.size), camera_name))
        return frames
    return []


def _metadata(f: h5py.File) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "metadata" in f:
        out.update(dict(f["metadata"].attrs))
    out.update(dict(f.attrs))
    return out


def _iter_qpos_rows(f: h5py.File) -> list[np.ndarray]:
    if "observations/qpos" not in f:
        return []
    qpos = np.asarray(f["observations/qpos"], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] < 4:
        return []
    qpos = qpos[:, :4]
    if "diagnostics/train_exclude_mask" in f:
        mask = np.asarray(f["diagnostics/train_exclude_mask"]).reshape(-1).astype(bool)
        if mask.shape[0] == qpos.shape[0] and np.any(~mask):
            qpos = qpos[~mask]
    return [row.astype(np.float32, copy=False) for row in qpos]


def _build_bucket_semantic_reference(
    *,
    dataset_dir: Path,
    episode_ids: list[int],
    min_count: int = 5,
) -> dict[str, Any]:
    features: list[dict[str, float]] = []
    for episode_id in episode_ids:
        path = dataset_dir / f"episode_{episode_id}.hdf5"
        if not path.exists():
            continue
        manual_end = _manual_end_index_for_episode(path)
        try:
            features.append(
                _bucket_semantic_features(path, manual_end_index=manual_end)
            )
        except Exception:
            continue
    if len(features) < int(min_count):
        return {"count": 0}
    reference = _bucket_semantic_reference(features)
    reference["count"] = int(len(features))
    return reference


def _manual_end_index_for_episode(path: Path) -> int:
    with h5py.File(path, "r") as f:
        n_steps = int(f["observations/qpos"].shape[0]) if "observations/qpos" in f else 0
        diagnostics = f.get("diagnostics")
        return _manual_end_index(
            n_steps=n_steps,
            go_home_requested=_read_optional(diagnostics, "go_home_requested"),
            go_home_running=_read_optional(diagnostics, "go_home_running"),
        )


def _qpos_reference(rows: list[np.ndarray]) -> dict[str, Any]:
    if not rows:
        return {}
    arr = np.stack(rows).astype(np.float64, copy=False)
    return {
        "count": int(arr.shape[0]),
        "axes": ["swing", "boom", "stick", "bucket"],
        "p1": np.percentile(arr, 1, axis=0).astype(float).tolist(),
        "p5": np.percentile(arr, 5, axis=0).astype(float).tolist(),
        "median": np.median(arr, axis=0).astype(float).tolist(),
        "p95": np.percentile(arr, 95, axis=0).astype(float).tolist(),
        "p99": np.percentile(arr, 99, axis=0).astype(float).tolist(),
    }


def _stats(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    median = float(np.median(arr))
    return {
        "count": int(arr.size),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "median": median,
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mad": float(np.median(np.abs(arr - median))),
    }


def _reference_id(reference: dict[str, Any]) -> str:
    payload = json.dumps(reference, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


if __name__ == "__main__":
    main()
