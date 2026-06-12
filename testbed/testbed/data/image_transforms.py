"""Deterministic RGB image transforms shared by training and offline replay."""

from __future__ import annotations

from typing import Callable

import numpy as np


IMAGE_TRANSFORM_CHOICES = (
    "none",
    "center_zoom_085",
    "center_zoom_075",
    "center_zoom_085_blur",
    "center_zoom_075_blur",
    "downsample_080",
    "downsample_060",
)


def build_image_transform(name: str) -> Callable[[np.ndarray], np.ndarray] | None:
    """Return a deterministic RGB image transform preserving shape and dtype."""

    normalized = str(name or "none").strip().lower()
    if normalized == "none":
        return None
    if normalized == "center_zoom_085":
        return lambda image: _center_zoom_image(image, crop_fraction=0.85)
    if normalized == "center_zoom_075":
        return lambda image: _center_zoom_image(image, crop_fraction=0.75)
    if normalized == "center_zoom_085_blur":
        return lambda image: _blur_image(_center_zoom_image(image, crop_fraction=0.85))
    if normalized == "center_zoom_075_blur":
        return lambda image: _blur_image(_center_zoom_image(image, crop_fraction=0.75))
    if normalized == "downsample_080":
        return lambda image: _downsample_upsample_image(image, scale=0.80)
    if normalized == "downsample_060":
        return lambda image: _downsample_upsample_image(image, scale=0.60)
    raise ValueError(
        f"Unsupported image transform {name!r}. "
        f"Supported transforms: {', '.join(IMAGE_TRANSFORM_CHOICES)}."
    )


def _center_zoom_image(image: np.ndarray, *, crop_fraction: float) -> np.ndarray:
    import cv2

    arr = _require_rgb_image(image)
    height, width = arr.shape[:2]
    fraction = float(crop_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"crop_fraction must be in (0, 1], got {crop_fraction}")
    crop_h = max(1, int(round(height * fraction)))
    crop_w = max(1, int(round(width * fraction)))
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    cropped = arr[top : top + crop_h, left : left + crop_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def _downsample_upsample_image(image: np.ndarray, *, scale: float) -> np.ndarray:
    import cv2

    arr = _require_rgb_image(image)
    height, width = arr.shape[:2]
    ratio = float(scale)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    small_w = max(1, int(round(width * ratio)))
    small_h = max(1, int(round(height * ratio)))
    small = cv2.resize(arr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def _blur_image(image: np.ndarray) -> np.ndarray:
    import cv2

    arr = _require_rgb_image(image)
    return cv2.GaussianBlur(arr, (5, 5), 0.0)


def _require_rgb_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected RGB image with shape (H, W, 3), got {arr.shape}")
    return np.ascontiguousarray(arr)
