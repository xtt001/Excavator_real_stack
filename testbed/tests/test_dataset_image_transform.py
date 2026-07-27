from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from testbed.data.dataset import EpisodicDataset, _valid_start_indices, load_data
from testbed.data.image_transforms import build_image_transform


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

    def test_episodic_dataset_can_return_deadzone_intent_masks_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            image = self._checkerboard_image(height=12, width=16)
            self._write_episode_with_handoff_masks(dataset_dir / "episode_0.hdf5", image=image)
            norm_stats = {
                "action_mean": np.zeros(4, dtype=np.float32),
                "action_std": np.ones(4, dtype=np.float32),
                "proprio_mean": np.zeros(4, dtype=np.float32),
                "proprio_std": np.ones(4, dtype=np.float32),
                "proprio_dim": 4,
                "qpos_only_dim": 4,
            }
            deadzone_intent = {
                "enabled": True,
                "thresholds": {
                    "swing": {"pos": 0.5, "neg": 0.5},
                    "boom": {"pos": 0.4, "neg": 0.4},
                    "stick": {"pos": 0.3, "neg": 0.3},
                    "bucket": {"pos": 0.2, "neg": 0.2},
                },
                "use_handoff_masks": True,
            }

            legacy_ds = EpisodicDataset(
                [0],
                dataset_dir,
                ["fpv"],
                norm_stats,
                episode_len=5,
                action_chunk_size=3,
                image_transform="none",
            )
            masked_ds = EpisodicDataset(
                [0],
                dataset_dir,
                ["fpv"],
                norm_stats,
                episode_len=5,
                action_chunk_size=3,
                image_transform="none",
                deadzone_intent=deadzone_intent,
            )

            legacy_sample = legacy_ds[0]
            masked_sample = masked_ds[0]

            self.assertIsInstance(legacy_sample, tuple)
            self.assertIsInstance(masked_sample, dict)
            self.assertEqual(masked_sample["action"].shape, (5, 4))
            self.assertEqual(masked_sample["is_pad"].shape, (5,))
            self.assertEqual(masked_sample["deadzone_move_mask"].shape, (5, 4, 2))
            self.assertEqual(masked_sample["deadzone_stop_mask"].shape, (5,))
            self.assertEqual(masked_sample["deadzone_wrong_mask"].shape, (5, 4, 2))
            self.assertEqual(masked_sample["action_loss_mask"].shape, (5,))

            self.assertTrue(bool(masked_sample["deadzone_move_mask"][0, 0, 0]))
            self.assertFalse(bool(masked_sample["deadzone_move_mask"][1].any()))
            self.assertTrue(bool(masked_sample["deadzone_stop_mask"][1]))
            self.assertFalse(bool(masked_sample["action_loss_mask"][1]))
            self.assertTrue(bool(masked_sample["is_pad"][4]))

    def test_load_data_can_compute_action_stats_from_action_loss_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            image = self._checkerboard_image(height=12, width=16)
            self._write_episode_with_stat_tail(dataset_dir / "episode_0.hdf5", image=image)
            deadzone_intent = {
                "enabled": True,
                "thresholds": {
                    "swing": {"pos": 0.5, "neg": 0.5},
                    "boom": {"pos": 0.4, "neg": 0.4},
                    "stick": {"pos": 0.3, "neg": 0.3},
                    "bucket": {"pos": 0.2, "neg": 0.2},
                },
                "use_handoff_masks": True,
                "use_action_loss_mask_for_stats": True,
            }

            _train_loader, _val_loader, norm_stats, _is_real, _split_info = load_data(
                dataset_dir=dataset_dir,
                num_episodes=1,
                camera_names=["fpv"],
                episode_len=4,
                batch_size_train=1,
                batch_size_val=1,
                num_workers=0,
                prefetch_factor=1,
                persistent_workers=False,
                pin_memory=False,
                split_seed=0,
                train_split_ratio=0.5,
                reuse_split=False,
                low_dim_keys=["qpos"],
                deadzone_intent=deadzone_intent,
            )

            self.assertEqual(float(norm_stats["action_mean"][0]), 1.0)
            self.assertAlmostEqual(float(norm_stats["action_std"][0]), 0.01, places=6)

    def test_load_data_applies_camera_loss_to_train_manifest_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            image = self._checkerboard_image(height=12, width=16)
            for episode_id in (0, 1):
                self._write_four_camera_episode(
                    dataset_dir / f"episode_{episode_id}.hdf5",
                    image=image,
                )

            _train, _val, _stats, _is_real, split = load_data(
                dataset_dir=dataset_dir,
                num_episodes=2,
                camera_names=["video4", "video5", "video6", "video7"],
                episode_len=2,
                batch_size_train=1,
                batch_size_val=1,
                num_workers=0,
                prefetch_factor=1,
                persistent_workers=False,
                pin_memory=False,
                split_seed=0,
                train_split_ratio=0.5,
                reuse_split=False,
                low_dim_keys=["qpos"],
                action_chunk_size=1,
                camera_loss_augmentation_train={
                    "enabled": True,
                    "scope": "train_only",
                    "target_camera": "video7",
                    "probability": 0.999999999,
                    "seed": 20260727,
                    "mask_rgb": [0, 0, 0],
                    "decision_key": [
                        "seed",
                        "source_episode_id",
                        "source_tick",
                    ],
                },
            )

            train_manifest = split["camera_loss_augmentation_train"]
            val_manifest = split["camera_loss_augmentation_validation"]
            self.assertTrue(train_manifest["enabled"])
            self.assertEqual(train_manifest["eligible_row_count"], 2)
            self.assertEqual(train_manifest["selected_row_count"], 2)
            self.assertEqual(
                train_manifest["source_episode_ids"],
                split["train_ids"],
            )
            self.assertFalse(val_manifest["enabled"])
            self.assertEqual(val_manifest["eligible_row_count"], 0)
            self.assertEqual(val_manifest["selected_row_count"], 0)
            self.assertEqual(
                val_manifest["source_episode_ids"],
                split["val_ids"],
            )

    def test_valid_start_indices_can_require_action_loss_inside_chunk(self) -> None:
        valid = _valid_start_indices(
            total_steps=5,
            train_exclude_mask=None,
            action_chunk_size=2,
            action_loss_mask=np.asarray([1, 0, 0, 1, 0], dtype=bool),
            require_action_loss_in_chunk=True,
        )

        np.testing.assert_array_equal(valid, np.asarray([0, 2, 3], dtype=np.int64))

    def test_random_downsample_transform_preserves_shape_dtype_and_is_seeded(self) -> None:
        image = self._checkerboard_image(height=24, width=32)
        transform_a = build_image_transform("random_downsample_060_100_seed7")
        transform_b = build_image_transform("random_downsample_060_100_seed7")

        outputs_a = [transform_a(image) for _ in range(6)]
        outputs_b = [transform_b(image) for _ in range(6)]

        for out_a, out_b in zip(outputs_a, outputs_b):
            self.assertEqual(out_a.shape, image.shape)
            self.assertEqual(out_a.dtype, image.dtype)
            np.testing.assert_array_equal(out_a, out_b)
        unique_outputs = {out.tobytes() for out in outputs_a}
        self.assertGreater(len(unique_outputs), 1)

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

    @staticmethod
    def _write_four_camera_episode(path: Path, *, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.zeros((2, 4), dtype=np.float32))
            obs.create_dataset("qvel", data=np.zeros((2, 4), dtype=np.float32))
            images = obs.create_group("images")
            for camera in ("video4", "video5", "video6", "video7"):
                images.create_dataset(
                    camera,
                    data=np.stack([image, image], axis=0).astype(np.uint8),
                )
            f.create_dataset("action", data=np.zeros((2, 4), dtype=np.float32))

    @staticmethod
    def _write_episode_with_handoff_masks(path: Path, *, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.zeros((4, 4), dtype=np.float32))
            obs.create_dataset("qvel", data=np.zeros((4, 4), dtype=np.float32))
            images = obs.create_group("images")
            images.create_dataset(
                "fpv",
                data=np.stack([image, image, image, image], axis=0).astype(np.uint8),
            )
            f.create_dataset(
                "action",
                data=np.asarray(
                    [
                        [0.60, 0.00, 0.00, 0.00],
                        [0.60, 0.00, 0.00, 0.00],
                        [0.00, -0.50, 0.00, 0.00],
                        [0.00, 0.00, 0.00, 0.00],
                    ],
                    dtype=np.float32,
                ),
            )
            handoff = f.create_group("handoff")
            handoff.create_dataset("action_loss_mask", data=np.asarray([1, 0, 0, 0], dtype=np.uint8))
            handoff.create_dataset("tail_idle_mask", data=np.asarray([0, 1, 0, 0], dtype=np.uint8))
            handoff.create_dataset("owner_automation", data=np.asarray([0, 0, 1, 1], dtype=np.uint8))
            diagnostics = f.create_group("diagnostics")
            diagnostics.create_dataset("train_exclude_mask", data=np.asarray([0, 0, 0, 1], dtype=np.uint8))

    @staticmethod
    def _write_episode_with_stat_tail(path: Path, *, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.zeros((3, 4), dtype=np.float32))
            obs.create_dataset("qvel", data=np.zeros((3, 4), dtype=np.float32))
            images = obs.create_group("images")
            images.create_dataset(
                "fpv",
                data=np.stack([image, image, image], axis=0).astype(np.uint8),
            )
            f.create_dataset(
                "action",
                data=np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [-9.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
            handoff = f.create_group("handoff")
            handoff.create_dataset("action_loss_mask", data=np.asarray([1, 1, 0], dtype=np.uint8))
            handoff.create_dataset("tail_idle_mask", data=np.asarray([0, 0, 1], dtype=np.uint8))
            handoff.create_dataset("owner_automation", data=np.asarray([0, 0, 1], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
