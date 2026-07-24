"""Frozen, observable-only image features for the SimVerify M0 labeler.

The production path is deliberately offline: callers must provide a local
ImageNet ResNet-18 checkpoint, and no torchvision weight enum or download
helper is used.  Source JPEGs are decoded as RGB and pass through the frozen
policy resize before the standard ResNet evaluation preprocessing.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from testbed.simverify.contracts import IMAGE_TRANSFORM_ID

FEATURE_SCHEMA = "frozen_imagenet_resnet18_feature_v1"
PAIR_FEATURE_SCHEMA = "frozen_imagenet_resnet18_pair_feature_v1"
FEATURE_DIM = 512
PAIR_FEATURE_DIM = FEATURE_DIM * 2

POLICY_IMAGE_WIDTH = 384
POLICY_IMAGE_HEIGHT = 216
RESNET_RESIZE_SHORTER = 256
RESNET_CENTER_CROP = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EYE_PAIR = ("eye_left", "eye_right")
STICK_PAIR = ("stick_down", "stick_up")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one local regular file."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(
            f"local checkpoint is required and was not found: {source}"
        )
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def policy_resize_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Apply the frozen no-crop policy transform to one RGB uint8 image."""

    import cv2

    rgb = _require_rgb_uint8(image)
    resized = cv2.resize(
        rgb,
        (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.ascontiguousarray(resized, dtype=np.uint8)


def preprocess_resnet18_rgb_batch(
    images: Sequence[np.ndarray] | np.ndarray,
) -> torch.Tensor:
    """Return the frozen ResNet input tensor with shape ``(N,3,224,224)``.

    Transform order is part of the contract:

    1. RGB uint8, no crop, OpenCV linear resize to 384x216.
    2. Convert to float in [0, 1].
    3. Resize the shorter side to 256 with bilinear antialiasing.
    4. Center-crop 224x224 and apply ImageNet mean/std normalization.
    """

    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform_functional

    batch = _coerce_image_batch(images)
    policy_images = np.stack(
        [policy_resize_rgb_uint8(image) for image in batch],
        axis=0,
    )
    tensor = (
        torch.from_numpy(np.ascontiguousarray(policy_images))
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div_(255.0)
    )
    tensor = transform_functional.resize(
        tensor,
        RESNET_RESIZE_SHORTER,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = transform_functional.center_crop(
        tensor,
        [RESNET_CENTER_CROP, RESNET_CENTER_CROP],
    )
    tensor = transform_functional.normalize(
        tensor,
        mean=list(IMAGENET_MEAN),
        std=list(IMAGENET_STD),
    )
    return tensor.contiguous()


def read_hdf5_source_camera_rgb(
    hdf5_path: str | Path,
    camera_name: str,
    indices: Iterable[int],
) -> list[np.ndarray]:
    """Decode selected source-camera JPEG rows as RGB uint8 images.

    Requested order and duplicate indices are preserved.  Only the encoded
    source-camera path is accepted; there is no fallback to privileged or
    unencoded image fields.
    """

    import cv2

    source = Path(hdf5_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = _coerce_indices(indices)
    if not rows:
        raise ValueError("indices must contain at least one source row")
    dataset_path = f"observations/encoded_images/{camera_name}"
    encoded_rows: list[np.ndarray] = []
    with h5py.File(source, "r") as handle:
        if dataset_path not in handle:
            raise KeyError(f"missing source camera dataset: {dataset_path}")
        dataset = handle[dataset_path]
        encoding = _decode_text(dataset.attrs.get("encoding", ""))
        if encoding.strip().lower() != "jpeg":
            raise ValueError(
                f"{dataset_path} must declare encoding='jpeg', got {encoding!r}"
            )
        for index in rows:
            if index < 0 or index >= len(dataset):
                raise IndexError(
                    f"{dataset_path} row {index} outside [0, {len(dataset)})"
                )
            encoded_rows.append(
                np.asarray(dataset[index], dtype=np.uint8).reshape(-1).copy()
            )

    decoded: list[np.ndarray] = []
    for index, encoded in zip(rows, encoded_rows):
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(
                f"failed to decode JPEG {dataset_path} at source row {index}"
            )
        decoded.append(
            np.ascontiguousarray(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                dtype=np.uint8,
            )
        )
    return decoded


def concatenate_normalized_pair_features(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Concatenate two normalized 512-D streams and L2-normalize to 1024-D."""

    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(second, dtype=np.float32)
    if left.ndim != 2 or left.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"first features must have shape (N,{FEATURE_DIM}), got {left.shape}"
        )
    if right.shape != left.shape:
        raise ValueError(
            f"second features must have shape {left.shape}, got {right.shape}"
        )
    pair = np.concatenate((left, right), axis=1).astype(np.float32, copy=False)
    return _l2_normalize_numpy(pair, expected_dim=PAIR_FEATURE_DIM)


class FrozenResNet18FeatureExtractor:
    """Batch extractor backed by a strictly local ImageNet ResNet-18."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_checkpoint_sha256: str | None = None,
        device: str | torch.device = "cpu",
        batch_size: int = 64,
        model_for_testing: torch.nn.Module | None = None,
    ) -> None:
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "frozen ResNet-18 requires an existing local checkpoint; "
                f"network fallback is disabled: {checkpoint}"
            )
        checkpoint = checkpoint.resolve()
        checkpoint_sha256 = sha256_file(checkpoint)
        if expected_checkpoint_sha256 is not None:
            expected = str(expected_checkpoint_sha256).strip().lower()
            if checkpoint_sha256 != expected:
                raise ValueError(
                    "checkpoint SHA-256 mismatch: "
                    f"expected {expected}, got {checkpoint_sha256}"
                )
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        resolved_device = torch.device(device)
        if model_for_testing is None:
            model = _load_local_resnet18(checkpoint)
            model_source = "local_checkpoint_strict"
            checkpoint_loaded = True
        else:
            model = model_for_testing
            model_source = "injected_test_model"
            checkpoint_loaded = False
        model.eval()
        model.requires_grad_(False)
        model.to(resolved_device)

        import torchvision

        self._model = model
        self._device = resolved_device
        self._batch_size = int(batch_size)
        self._provenance: dict[str, Any] = {
            "schema": FEATURE_SCHEMA,
            "architecture": "torchvision_resnet18_fc_identity",
            "feature_dim": FEATURE_DIM,
            "feature_normalization": "l2",
            "checkpoint": {
                "path": str(checkpoint),
                "size_bytes": int(checkpoint.stat().st_size),
                "sha256": checkpoint_sha256,
                "loaded_into_model": checkpoint_loaded,
            },
            "model_source": model_source,
            "network_download_allowed": False,
            "torch_version": str(torch.__version__),
            "torchvision_version": str(torchvision.__version__),
            "device": str(resolved_device),
            "preprocess": {
                "input": {
                    "color_space": "RGB",
                    "layout": "HWC",
                    "dtype": "uint8",
                },
                "policy_transform": {
                    "transform_id": IMAGE_TRANSFORM_ID,
                    "crop": "none",
                    "resize_width": POLICY_IMAGE_WIDTH,
                    "resize_height": POLICY_IMAGE_HEIGHT,
                    "interpolation": "opencv_inter_linear",
                },
                "resnet_transform": {
                    "scale": "uint8_div_255",
                    "resize_shorter": RESNET_RESIZE_SHORTER,
                    "interpolation": "torchvision_bilinear",
                    "antialias": True,
                    "center_crop": [
                        RESNET_CENTER_CROP,
                        RESNET_CENTER_CROP,
                    ],
                    "mean": list(IMAGENET_MEAN),
                    "std": list(IMAGENET_STD),
                },
            },
            "pair_features": {
                "schema": PAIR_FEATURE_SCHEMA,
                "eye_order": list(EYE_PAIR),
                "stick_order": list(STICK_PAIR),
                "operation": "concatenate_then_l2_normalize",
                "feature_dim": PAIR_FEATURE_DIM,
            },
        }

    @property
    def provenance(self) -> dict[str, Any]:
        """Return a caller-safe copy of the extractor provenance."""

        return copy.deepcopy(self._provenance)

    def extract_rgb_batch(
        self,
        images: Sequence[np.ndarray] | np.ndarray,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Extract deterministic L2-normalized 512-D features in input order."""

        batch = _coerce_image_batch(images)
        size = self._batch_size if batch_size is None else int(batch_size)
        if size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(batch), size):
                inputs = preprocess_resnet18_rgb_batch(
                    batch[start : start + size]
                ).to(self._device)
                raw = self._model(inputs)
                if not isinstance(raw, torch.Tensor):
                    raise TypeError(
                        "feature model must return one torch.Tensor, "
                        f"got {type(raw).__name__}"
                    )
                flattened = raw.reshape(raw.shape[0], -1)
                if flattened.shape[1] != FEATURE_DIM:
                    raise ValueError(
                        "feature model output must have shape "
                        f"(N,{FEATURE_DIM}), got {tuple(flattened.shape)}"
                    )
                if not torch.isfinite(flattened).all():
                    raise ValueError("feature model output contains NaN or infinity")
                norms = torch.linalg.vector_norm(
                    flattened,
                    ord=2,
                    dim=1,
                    keepdim=True,
                )
                if torch.any(norms <= torch.finfo(flattened.dtype).eps):
                    raise ValueError("feature model produced a zero-norm row")
                normalized = flattened / norms
                outputs.append(
                    normalized.to(device="cpu", dtype=torch.float32).numpy()
                )
        result = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
        if result.shape != (len(batch), FEATURE_DIM):
            raise AssertionError(f"unexpected feature shape {result.shape}")
        return result

    def extract_hdf5_camera(
        self,
        hdf5_path: str | Path,
        camera_name: str,
        indices: Iterable[int],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Extract one source-camera feature stream at selected row indices."""

        rows = _coerce_indices(indices)
        images = read_hdf5_source_camera_rgb(
            hdf5_path,
            camera_name,
            rows,
        )
        return self.extract_rgb_batch(images, batch_size=batch_size)

    def extract_hdf5_camera_pair(
        self,
        hdf5_path: str | Path,
        camera_names: tuple[str, str],
        indices: Iterable[int],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Extract, concatenate, and normalize one ordered camera pair."""

        if len(camera_names) != 2:
            raise ValueError(
                f"camera_names must contain exactly two names, got {camera_names}"
            )
        rows = _coerce_indices(indices)
        first = self.extract_hdf5_camera(
            hdf5_path,
            camera_names[0],
            rows,
            batch_size=batch_size,
        )
        second = self.extract_hdf5_camera(
            hdf5_path,
            camera_names[1],
            rows,
            batch_size=batch_size,
        )
        return concatenate_normalized_pair_features(first, second)

    def extract_hdf5_eye_pair(
        self,
        hdf5_path: str | Path,
        indices: Iterable[int],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Return ordered ``eye_left + eye_right`` normalized features."""

        return self.extract_hdf5_camera_pair(
            hdf5_path,
            EYE_PAIR,
            indices,
            batch_size=batch_size,
        )

    def extract_hdf5_stick_pair(
        self,
        hdf5_path: str | Path,
        indices: Iterable[int],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Return ordered ``stick_down + stick_up`` normalized features."""

        return self.extract_hdf5_camera_pair(
            hdf5_path,
            STICK_PAIR,
            indices,
            batch_size=batch_size,
        )


def _load_local_resnet18(checkpoint_path: Path) -> torch.nn.Module:
    import torchvision

    model = torchvision.models.resnet18(weights=None)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(payload)
    model.load_state_dict(state_dict, strict=True)
    model.fc = torch.nn.Identity()
    return model


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if payload and all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in payload.items()
        ):
            return payload
        for key in ("state_dict", "model_state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and candidate and all(
                isinstance(name, str) and isinstance(value, torch.Tensor)
                for name, value in candidate.items()
            ):
                return candidate
    raise ValueError(
        "local ResNet-18 checkpoint must contain a tensor state_dict"
    )


def _coerce_image_batch(
    images: Sequence[np.ndarray] | np.ndarray,
) -> list[np.ndarray]:
    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            batch = [images]
        elif images.ndim == 4:
            batch = [images[index] for index in range(images.shape[0])]
        else:
            raise ValueError(
                "images must have shape (H,W,3) or (N,H,W,3), "
                f"got {images.shape}"
            )
    else:
        batch = list(images)
    if not batch:
        raise ValueError("images must contain at least one RGB frame")
    return [_require_rgb_uint8(image) for image in batch]


def _require_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"expected RGB uint8 image, got dtype {array.dtype}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"expected RGB uint8 image with shape (H,W,3), got {array.shape}"
        )
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"image dimensions must be positive, got {array.shape}")
    return np.ascontiguousarray(array)


def _coerce_indices(indices: Iterable[int]) -> list[int]:
    rows: list[int] = []
    for value in indices:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("source row indices must be integers, not booleans")
        if not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"source row index must be an integer, got {type(value).__name__}"
            )
        rows.append(int(value))
    return rows


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _l2_normalize_numpy(
    values: np.ndarray,
    *,
    expected_dim: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ValueError(
            f"features must have shape (N,{expected_dim}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("features contain NaN or infinity")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("features contain a zero-norm row")
    return np.ascontiguousarray(array / norms, dtype=np.float32)
