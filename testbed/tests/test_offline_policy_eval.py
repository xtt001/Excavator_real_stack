from __future__ import annotations

import unittest

import numpy as np

from testbed.policies.offline_eval import (
    aggregate_episode_results,
    build_image_transform,
    compute_action_metrics,
    _build_image_step_map,
    _map_image_step,
    select_representative_episode,
)


class OfflinePolicyEvalTests(unittest.TestCase):
    def test_select_representative_episode_uses_robust_middle(self) -> None:
        features = {
            "episode_10": np.array([0.0, 0.0, 10.0], dtype=np.float32),
            "episode_11": np.array([1.0, 1.0, 11.0], dtype=np.float32),
            "episode_12": np.array([20.0, 20.0, 80.0], dtype=np.float32),
        }

        selected, scores = select_representative_episode(
            ["episode_10", "episode_11", "episode_12"],
            features,
        )

        self.assertEqual(selected, "episode_11")
        self.assertEqual(scores[0]["episode_id"], "episode_11")
        self.assertLess(scores[0]["representative_score"], scores[-1]["representative_score"])

    def test_compute_action_metrics_reports_per_axis_error_and_distribution(self) -> None:
        expert = np.array(
            [
                [0.0, 0.5, -0.5, 1.0],
                [0.2, 0.0, -0.2, 0.5],
                [-0.2, -0.5, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        policy = np.array(
            [
                [0.1, 0.4, -0.4, 0.8],
                [0.0, 0.1, -0.1, 0.4],
                [-0.1, -0.4, 0.0, 0.2],
            ],
            dtype=np.float32,
        )

        metrics = compute_action_metrics(expert, policy)

        self.assertEqual(metrics["n_steps"], 3)
        self.assertAlmostEqual(metrics["overall"]["mae"], 1.6 / 12.0, places=6)
        self.assertAlmostEqual(metrics["axes"]["swing"]["expert_p95_abs"], 0.2, places=6)
        self.assertAlmostEqual(metrics["axes"]["bucket"]["policy_max_abs"], 0.8, places=6)
        self.assertGreater(metrics["axes"]["boom"]["correlation"], 0.9)

    def test_aggregate_episode_results_combines_global_metrics_and_episode_rows(self) -> None:
        results = [
            {
                "episode_id": "episode_10",
                "n_steps": 2,
                "expert_action": np.array(
                    [[0.0, 0.5, 0.0, 1.0], [0.2, 0.0, 0.0, 0.5]],
                    dtype=np.float32,
                ),
                "policy_action": np.array(
                    [[0.1, 0.4, 0.0, 0.8], [0.0, 0.1, 0.0, 0.4]],
                    dtype=np.float32,
                ),
                "metrics": compute_action_metrics(
                    np.array([[0.0, 0.5, 0.0, 1.0], [0.2, 0.0, 0.0, 0.5]], dtype=np.float32),
                    np.array([[0.1, 0.4, 0.0, 0.8], [0.0, 0.1, 0.0, 0.4]], dtype=np.float32),
                ),
            },
            {
                "episode_id": "episode_11",
                "n_steps": 1,
                "expert_action": np.array([[-0.2, -0.5, 0.0, 0.0]], dtype=np.float32),
                "policy_action": np.array([[-0.1, -0.4, 0.0, 0.2]], dtype=np.float32),
                "metrics": compute_action_metrics(
                    np.array([[-0.2, -0.5, 0.0, 0.0]], dtype=np.float32),
                    np.array([[-0.1, -0.4, 0.0, 0.2]], dtype=np.float32),
                ),
            },
        ]

        aggregate = aggregate_episode_results(results)

        self.assertEqual(aggregate["n_episodes"], 2)
        self.assertEqual(aggregate["n_steps"], 3)
        self.assertEqual([row["episode_id"] for row in aggregate["episode_rows"]], ["episode_10", "episode_11"])
        self.assertAlmostEqual(aggregate["global_metrics"]["overall"]["mae"], 1.2 / 12.0, places=6)
        self.assertAlmostEqual(
            aggregate["episode_rows"][0]["policy_p95_abs_bucket"],
            results[0]["metrics"]["axes"]["bucket"]["policy_p95_abs"],
            places=6,
        )

    def test_map_image_step_supports_progress_and_same_index(self) -> None:
        self.assertEqual(
            _map_image_step(step=0, target_steps=5, image_steps=9, mode="progress"),
            0,
        )
        self.assertEqual(
            _map_image_step(step=2, target_steps=5, image_steps=9, mode="progress"),
            4,
        )
        self.assertEqual(
            _map_image_step(step=4, target_steps=5, image_steps=9, mode="progress"),
            8,
        )
        self.assertEqual(
            _map_image_step(step=12, target_steps=5, image_steps=9, mode="same_index"),
            8,
        )

    def test_nearest_qpos_image_step_map_reports_match_quality(self) -> None:
        class FakeH5(dict):
            pass

        target = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.10, 0.0, 0.0, 0.0],
                [0.20, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        image = FakeH5(
            {
                "observations/qpos": np.array(
                    [
                        [0.21, 0.0, 0.0, 0.0],
                        [0.01, 0.0, 0.0, 0.0],
                        [0.11, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                )
            }
        )

        steps, metrics = _build_image_step_map(
            target_qpos=target,
            image_h5=image,
            mode="nearest_qpos",
        )

        np.testing.assert_array_equal(steps, np.array([1, 2, 0]))
        self.assertLess(metrics["max_abs_delta_rad"], 0.011)
        self.assertEqual(
            _map_image_step(
                step=1,
                target_steps=3,
                image_steps=3,
                mode="nearest_qpos",
                nearest_steps=steps,
            ),
            2,
        )

    def test_image_transform_preserves_rgb_shape_and_dtype(self) -> None:
        rows = np.arange(12, dtype=np.uint8).reshape(12, 1, 1)
        cols = np.arange(16, dtype=np.uint8).reshape(1, 16, 1)
        image = np.concatenate(
            [
                np.broadcast_to(rows * 10, (12, 16, 1)),
                np.broadcast_to(cols * 12, (12, 16, 1)),
                np.full((12, 16, 1), 127, dtype=np.uint8),
            ],
            axis=2,
        )

        for name in (
            "center_zoom_085",
            "center_zoom_075_blur",
            "downsample_060",
        ):
            transformed = build_image_transform(name)(image)
            self.assertEqual(transformed.shape, image.shape)
            self.assertEqual(transformed.dtype, image.dtype)

    def test_image_transform_rejects_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            build_image_transform("not_a_transform")


if __name__ == "__main__":
    unittest.main()
