from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import torch

from testbed.simverify.features import (
    FEATURE_DIM,
    PAIR_FEATURE_DIM,
    FrozenResNet18FeatureExtractor,
    concatenate_normalized_pair_features,
    policy_resize_rgb_uint8,
    preprocess_resnet18_rgb_batch,
    read_hdf5_source_camera_rgb,
)


class _Tiny512FeatureModel(torch.nn.Module):
    """Cheap deterministic stand-in; production still loads ResNet-18."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        channel_means = images.mean(dim=(2, 3))
        repeats = (FEATURE_DIM + channel_means.shape[1] - 1) // (
            channel_means.shape[1]
        )
        repeated = channel_means.repeat(1, repeats)[:, :FEATURE_DIM]
        ramp = torch.linspace(
            0.001,
            0.512,
            FEATURE_DIM,
            dtype=images.dtype,
            device=images.device,
        )
        return repeated + ramp.unsqueeze(0)


def test_policy_and_resnet_preprocess_preserve_rgb_and_shape() -> None:
    red = np.zeros((19, 31, 3), dtype=np.uint8)
    red[..., 0] = 255

    policy = policy_resize_rgb_uint8(red)
    preprocessed = preprocess_resnet18_rgb_batch([red])

    assert policy.shape == (216, 384, 3)
    assert policy.dtype == np.uint8
    assert np.all(policy[..., 0] == 255)
    assert np.all(policy[..., 1:] == 0)
    assert preprocessed.shape == (1, 3, 224, 224)
    expected = torch.tensor(
        [
            (1.0 - 0.485) / 0.229,
            (0.0 - 0.456) / 0.224,
            (0.0 - 0.406) / 0.225,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(
        preprocessed.mean(dim=(0, 2, 3)),
        expected,
        atol=1e-5,
        rtol=0.0,
    )
    with pytest.raises(ValueError, match="uint8"):
        policy_resize_rgb_uint8(red.astype(np.float32))


def test_injected_model_is_deterministic_512d_and_records_provenance(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "synthetic-local-checkpoint.pth"
    checkpoint.write_bytes(b"synthetic checkpoint identity")
    extractor = FrozenResNet18FeatureExtractor(
        checkpoint,
        device="cpu",
        batch_size=1,
        model_for_testing=_Tiny512FeatureModel(),
    )
    images = np.stack(
        [
            _solid_rgb((255, 0, 0)),
            _solid_rgb((0, 255, 0)),
            _solid_rgb((0, 0, 255)),
        ],
        axis=0,
    )

    first = extractor.extract_rgb_batch(images, batch_size=1)
    second = extractor.extract_rgb_batch(images, batch_size=2)

    assert first.shape == (3, FEATURE_DIM)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    assert not extractor._model.training
    provenance = extractor.provenance
    assert provenance["checkpoint"]["sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert provenance["checkpoint"]["loaded_into_model"] is False
    assert provenance["model_source"] == "injected_test_model"
    assert provenance["network_download_allowed"] is False
    assert provenance["configured_default_batch_size"] == 1
    assert provenance["torch_version"]
    assert provenance["torchvision_version"]
    assert provenance["preprocess"]["policy_transform"]["crop"] == "none"
    assert provenance["preprocess"]["policy_transform"]["resize_width"] == 384
    assert provenance["preprocess"]["policy_transform"]["resize_height"] == 216
    assert (
        provenance["preprocess"]["resnet_transform"]["resize_shorter"] == 256
    )
    assert provenance["preprocess"]["resnet_transform"]["center_crop"] == [
        224,
        224,
    ]


def test_missing_checkpoint_fails_closed_even_with_injected_model(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="network fallback is disabled"):
        FrozenResNet18FeatureExtractor(
            tmp_path / "missing-resnet18.pth",
            model_for_testing=_Tiny512FeatureModel(),
        )


def test_hdf5_indexed_jpeg_and_ordered_pair_features(tmp_path: Path) -> None:
    checkpoint = tmp_path / "synthetic-local-checkpoint.pth"
    checkpoint.write_bytes(b"synthetic checkpoint identity")
    episode = tmp_path / "episode_7.hdf5"
    camera_frames = {
        "eye_left": [
            _solid_rgb((240, 10, 10)),
            _solid_rgb((10, 240, 10)),
        ],
        "eye_right": [
            _solid_rgb((10, 10, 240)),
            _solid_rgb((220, 220, 10)),
        ],
        "stick_down": [
            _solid_rgb((200, 20, 180)),
            _solid_rgb((20, 180, 200)),
        ],
        "stick_up": [
            _solid_rgb((180, 100, 20)),
            _solid_rgb((30, 200, 80)),
        ],
    }
    _write_source_jpegs(episode, camera_frames)
    extractor = FrozenResNet18FeatureExtractor(
        checkpoint,
        model_for_testing=_Tiny512FeatureModel(),
    )
    requested = [1, 0]

    decoded = read_hdf5_source_camera_rgb(
        episode,
        "eye_left",
        requested,
    )
    eye = extractor.extract_hdf5_eye_pair(episode, requested)
    stick = extractor.extract_hdf5_stick_pair(episode, requested)

    assert len(decoded) == 2
    assert decoded[0].dtype == np.uint8
    assert decoded[0].shape == camera_frames["eye_left"][1].shape
    assert int(np.argmax(decoded[0].mean(axis=(0, 1)))) == 1
    assert int(np.argmax(decoded[1].mean(axis=(0, 1)))) == 0
    assert eye.shape == (2, PAIR_FEATURE_DIM)
    assert stick.shape == (2, PAIR_FEATURE_DIM)
    assert np.allclose(np.linalg.norm(eye, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(stick, axis=1), 1.0, atol=1e-6)

    left = extractor.extract_hdf5_camera(episode, "eye_left", requested)
    right = extractor.extract_hdf5_camera(episode, "eye_right", requested)
    assert np.allclose(
        eye,
        concatenate_normalized_pair_features(left, right),
        atol=1e-7,
        rtol=0.0,
    )


def _solid_rgb(color: tuple[int, int, int]) -> np.ndarray:
    image = np.empty((24, 40, 3), dtype=np.uint8)
    image[...] = np.asarray(color, dtype=np.uint8)
    return image


def _write_source_jpegs(
    path: Path,
    frames_by_camera: dict[str, list[np.ndarray]],
) -> None:
    dtype = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(path, "w") as handle:
        images = handle.create_group("observations/encoded_images")
        for camera_name, frames in frames_by_camera.items():
            dataset = images.create_dataset(
                camera_name,
                shape=(len(frames),),
                dtype=dtype,
            )
            dataset.attrs["encoding"] = "jpeg"
            for index, rgb in enumerate(frames):
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 100],
                )
                assert ok
                dataset[index] = np.asarray(encoded, dtype=np.uint8)
