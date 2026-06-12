from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from testbed.data.dataset import EpisodicDataset


class DatasetImageTransformTests(unittest.TestCase):
    def test_episodic_dataset_applies_configured_image_transform(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            image = self._checkerboard_image(height=12, width=16)
            self._write_episode(dataset_dir / "episode_0.hdf5", image=image)
            norm_stats = {
                "action_mean": np.zeros(4, dtype=np.float32),
                "action_std": np.ones(4, dtype=np.float32),
                "proprio_mean": np.zeros(4, dtype=np.float32),
                "proprio_std": np.ones(4, dtype=np.float32),
                "proprio_dim": 4,
                "qpos_only_dim": 4,
            }

            raw_ds = EpisodicDataset(
                [0],
                dataset_dir,
                ["fpv"],
                norm_stats,
                image_transform="none",
            )
            transformed_ds = EpisodicDataset(
                [0],
                dataset_dir,
                ["fpv"],
                norm_stats,
                image_transform="downsample_060",
            )

            raw_image = raw_ds[0][0].numpy()
            transformed_image = transformed_ds[0][0].numpy()

            self.assertEqual(transformed_image.shape, raw_image.shape)
            self.assertFalse(np.allclose(transformed_image, raw_image))

    @staticmethod
    def _checkerboard_image(*, height: int, width: int) -> np.ndarray:
        yy, xx = np.indices((height, width))
        pattern = ((xx + yy) % 2 * 255).astype(np.uint8)
        return np.stack(
            [
                pattern,
                np.flipud(pattern),
                np.full_like(pattern, 127),
            ],
            axis=-1,
        )

    @staticmethod
    def _write_episode(path: Path, *, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.zeros((2, 4), dtype=np.float32))
            obs.create_dataset("qvel", data=np.zeros((2, 4), dtype=np.float32))
            images = obs.create_group("images")
            images.create_dataset(
                "fpv",
                data=np.stack([image, image], axis=0).astype(np.uint8),
            )
            f.create_dataset("action", data=np.zeros((2, 4), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
