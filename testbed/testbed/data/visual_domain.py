"""Offline visual-domain clustering helpers for real teleop HDF5 batches."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw

from testbed.data.dataset import _read_camera_image


@dataclass(frozen=True)
class VisualDomainConfig:
    dataset_dir: Path
    manifest_path: Path
    output_dir: Path
    camera_names: tuple[str, ...] = ("video4", "video5", "video6", "video7")
    k: int = 6
    max_frames_per_episode: int = 24
    seed: int = 7
    feature_size: int = 96
    contact_sheet_per_cluster: int = 30


def load_train_ready_episode_ids(manifest_path: Path) -> list[str]:
    payload = json.loads(Path(manifest_path).read_text())
    episode_ids = payload.get("train_ready_episode_ids")
    if not isinstance(episode_ids, list) or not episode_ids:
        raise ValueError(f"manifest missing non-empty train_ready_episode_ids: {manifest_path}")
    return [str(episode_id) for episode_id in episode_ids]


def task_frame_indices(
    *,
    total_steps: int,
    max_frames: int,
    train_exclude_mask: np.ndarray | None,
    gohome_requested: np.ndarray | None,
    gohome_running: np.ndarray | None,
) -> np.ndarray:
    """Return evenly sampled valid task frames before gohome automation starts."""
    if total_steps <= 0:
        return np.asarray([], dtype=np.int64)
    valid = np.ones(int(total_steps), dtype=bool)
    if train_exclude_mask is not None:
        mask = np.asarray(train_exclude_mask, dtype=bool).reshape(-1)
        if mask.size == total_steps:
            valid &= ~mask

    stop_candidates: list[int] = []
    for signal in (gohome_requested, gohome_running):
        if signal is None:
            continue
        arr = np.asarray(signal).reshape(-1)
        if arr.size != total_steps:
            continue
        hits = np.flatnonzero(arr.astype(bool))
        if hits.size:
            stop_candidates.append(int(hits[0]))
    if stop_candidates:
        valid[min(stop_candidates) :] = False

    candidates = np.flatnonzero(valid)
    if candidates.size <= max_frames:
        return candidates.astype(np.int64)
    pick_positions = np.linspace(0, candidates.size - 1, int(max_frames)).round().astype(np.int64)
    return candidates[pick_positions].astype(np.int64)


def kmeans_numpy(
    features: np.ndarray,
    *,
    k: int,
    seed: int,
    max_iter: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic k-means implementation to avoid adding sklearn."""
    data = np.asarray(features, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {data.shape}")
    if data.shape[0] < int(k):
        raise ValueError(f"k={k} requires at least {k} samples, got {data.shape[0]}")
    rng = np.random.default_rng(int(seed))
    centers = data[rng.choice(data.shape[0], size=int(k), replace=False)].copy()
    labels = np.zeros(data.shape[0], dtype=np.int64)

    for _ in range(int(max_iter)):
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1).astype(np.int64)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster_id in range(int(k)):
            members = data[labels == cluster_id]
            if members.size:
                centers[cluster_id] = members.mean(axis=0)
            else:
                nearest = distances.min(axis=1)
                centers[cluster_id] = data[int(nearest.argmax())]
    order = np.argsort(-centers.sum(axis=1))
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    return remap[labels].astype(np.int64), centers[order]


def assign_episode_domains(rows: Iterable[dict[str, Any]], *, k: int) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row["episode_id"])].append(int(row["cluster"]))

    output: dict[str, dict[str, Any]] = {}
    for episode_id, clusters in sorted(grouped.items()):
        counts = np.bincount(np.asarray(clusters, dtype=np.int64), minlength=int(k))
        total = int(counts.sum())
        dominant = int(counts.argmax()) if total else 0
        proportions = {
            f"texture_domain_{idx}": (float(counts[idx]) / float(total) if total else 0.0)
            for idx in range(int(k))
        }
        output[episode_id] = {
            "episode_id": episode_id,
            "sample_count": total,
            "dominant_domain": f"texture_domain_{dominant}",
            "dominant_fraction": proportions[f"texture_domain_{dominant}"],
            "domain_proportions": proportions,
        }
    return output


def run_visual_domain_clustering(config: VisualDomainConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    episode_ids = load_train_ready_episode_ids(config.manifest_path)

    sample_rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    thumbnails: list[np.ndarray] = []
    for episode_id in episode_ids:
        h5_path = config.dataset_dir / f"{episode_id}.hdf5"
        with h5py.File(h5_path, "r") as f:
            total_steps = int(f["action"].shape[0])
            indices = task_frame_indices(
                total_steps=total_steps,
                max_frames=config.max_frames_per_episode,
                train_exclude_mask=_optional_dataset(f, "diagnostics/train_exclude_mask"),
                gohome_requested=_optional_dataset(f, "diagnostics/go_home_requested"),
                gohome_running=_optional_dataset(f, "diagnostics/go_home_running"),
            )
            for timestep in indices:
                images = [_read_camera_image(f, camera, int(timestep)) for camera in config.camera_names]
                feature_rows.append(_multi_camera_feature(images, size=config.feature_size))
                thumbnails.append(_camera_mosaic(images, tile_width=160))
                sample_rows.append(
                    {
                        "episode_id": episode_id,
                        "timestep": int(timestep),
                        "cluster": -1,
                    }
                )

    features = np.stack(feature_rows, axis=0).astype(np.float32)
    normalized, mean, std = _standardize(features)
    labels, centers = kmeans_numpy(normalized, k=config.k, seed=config.seed)
    for row, label in zip(sample_rows, labels, strict=True):
        row["cluster"] = int(label)

    episode_domains = assign_episode_domains(sample_rows, k=config.k)
    np.save(config.output_dir / "sample_features.npy", features)
    np.save(config.output_dir / "sample_features_mean.npy", mean)
    np.save(config.output_dir / "sample_features_std.npy", std)
    np.save(config.output_dir / "cluster_centers_standardized.npy", centers)
    _write_sample_rows(config.output_dir / "sample_domains.csv", sample_rows)
    _write_episode_domains(config.output_dir / "episode_domains.csv", episode_domains, k=config.k)
    _write_contact_sheets(
        config.output_dir / "contact_sheets",
        rows=sample_rows,
        thumbnails=thumbnails,
        k=config.k,
        per_cluster=config.contact_sheet_per_cluster,
    )

    summary = {
        "dataset_dir": str(config.dataset_dir),
        "manifest_path": str(config.manifest_path),
        "camera_names": list(config.camera_names),
        "k": int(config.k),
        "seed": int(config.seed),
        "episode_count": len(episode_ids),
        "sample_count": len(sample_rows),
        "max_frames_per_episode": int(config.max_frames_per_episode),
        "cluster_counts": {
            f"texture_domain_{idx}": int((labels == idx).sum()) for idx in range(int(config.k))
        },
        "episode_domains": episode_domains,
    }
    (config.output_dir / "cluster_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


def _optional_dataset(h5_file: h5py.File, path: str) -> np.ndarray | None:
    if path not in h5_file:
        return None
    return np.asarray(h5_file[path][()])


def _multi_camera_feature(images: list[np.ndarray], *, size: int) -> np.ndarray:
    return np.concatenate([_image_feature(image, size=size) for image in images]).astype(np.float32)


def _image_feature(image: np.ndarray, *, size: int) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    resized = cv2.resize(rgb, (int(size), int(size)), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    color_stats = np.concatenate([arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))])
    hue_hist = np.histogram(hsv[..., 0], bins=12, range=(0, 180), density=True)[0]
    sat_hist = np.histogram(hsv[..., 1], bins=8, range=(0, 256), density=True)[0]
    val_hist = np.histogram(hsv[..., 2], bins=8, range=(0, 256), density=True)[0]
    gray_hist = np.histogram(gray, bins=12, range=(0.0, 1.0), density=True)[0]
    grad_stats = np.asarray([grad.mean(), grad.std(), np.percentile(grad, 75), np.percentile(grad, 95)])
    return np.concatenate([color_stats, hue_hist, sat_hist, val_hist, gray_hist, grad_stats]).astype(
        np.float32
    )


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((features - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _camera_mosaic(images: list[np.ndarray], *, tile_width: int) -> np.ndarray:
    tiles = []
    for image in images:
        arr = np.asarray(image, dtype=np.uint8)
        h, w = arr.shape[:2]
        tile_height = max(1, int(round(tile_width * h / w)))
        tiles.append(cv2.resize(arr, (tile_width, tile_height), interpolation=cv2.INTER_AREA))
    return np.concatenate(tiles, axis=1)


def _write_sample_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_id", "timestep", "cluster", "domain"])
        writer.writeheader()
        for row in rows:
            cluster = int(row["cluster"])
            writer.writerow({**row, "domain": f"texture_domain_{cluster}"})


def _write_episode_domains(path: Path, domains: dict[str, dict[str, Any]], *, k: int) -> None:
    fieldnames = ["episode_id", "sample_count", "dominant_domain", "dominant_fraction"] + [
        f"texture_domain_{idx}" for idx in range(int(k))
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in domains.values():
            output = {
                "episode_id": row["episode_id"],
                "sample_count": row["sample_count"],
                "dominant_domain": row["dominant_domain"],
                "dominant_fraction": row["dominant_fraction"],
            }
            output.update(row["domain_proportions"])
            writer.writerow(output)


def _write_contact_sheets(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
    thumbnails: list[np.ndarray],
    k: int,
    per_cluster: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[int(row["cluster"])].append(idx)
    for cluster_id in range(int(k)):
        selected = grouped.get(cluster_id, [])[: int(per_cluster)]
        if not selected:
            continue
        sheet = _make_contact_sheet(
            [thumbnails[idx] for idx in selected],
            [f'{rows[idx]["episode_id"]}@{rows[idx]["timestep"]}' for idx in selected],
        )
        sheet.save(output_dir / f"texture_domain_{cluster_id}.jpg", quality=88)


def _make_contact_sheet(images: list[np.ndarray], labels: list[str]) -> Image.Image:
    cols = 3
    label_h = 18
    thumb_h, thumb_w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, image in enumerate(images):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        sheet.paste(Image.fromarray(image), (x, y))
        draw.text((x + 4, y + thumb_h + 2), labels[idx], fill=(0, 0, 0))
    return sheet
