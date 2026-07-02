from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from testbed.actions.base import ActionInfo
from testbed.actions.policy import (
    PolicyActionSource,
    _policy_obs_from_real_obs,
    load_act_policy_from_bundle,
)
from testbed.actions.policy_remote import RemoteArmedPolicyActionSource
from testbed.cli.record_real import (
    ReceiverHealthSnapshot,
    ReceiverTestLogger,
    _add_policy_action_diagnostics,
)


class DummyPolicy:
    def __init__(self, action: np.ndarray | list[float]):
        self.action = np.asarray(action, dtype=np.float32)
        self.reset_count = 0
        self.seen_obs: list[dict] = []

    def reset(self) -> None:
        self.reset_count += 1

    def predict(self, obs: dict) -> np.ndarray:
        self.seen_obs.append(obs)
        return self.action.copy()


class DummyActionSource:
    def __init__(self, samples: list[tuple[np.ndarray, ActionInfo]]):
        self.samples = list(samples)
        self.reset_count = 0
        self.closed = False
        self.published: list[dict] = []

    def reset(self) -> None:
        self.reset_count += 1

    def next_action(self, obs: dict) -> tuple[np.ndarray, ActionInfo]:
        if self.samples:
            return self.samples.pop(0)
        return np.zeros(4, dtype=np.float32), ActionInfo(
            source_type="teleop",
            source_id="remote:idle",
            extras={"remote_action_connected": 1},
        )

    def close(self) -> None:
        self.closed = True

    def publish_status(self, payload: dict) -> None:
        self.published.append(dict(payload))


def _obs() -> dict:
    return {
        "qpos": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "qvel": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "images": {
            "fpv": np.zeros((8, 10, 3), dtype=np.uint8),
        },
    }


class PolicyActionSourceTests(unittest.TestCase):
    def test_remote_armed_policy_toggles_on_policy_start(self) -> None:
        remote = DummyActionSource(
            [
                (
                    np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
                    ActionInfo(
                        source_type="teleop",
                        source_id="remote:unit",
                        latency_ms=3.0,
                        extras={"remote_action_connected": 1},
                    ),
                ),
                (
                    np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
                    ActionInfo(
                        source_type="teleop",
                        source_id="remote:unit",
                        latency_ms=4.0,
                        extras={
                            "remote_action_connected": 1,
                            "policy_start_requested": True,
                            "toggle_mask": 1,
                        },
                    ),
                ),
                (
                    np.array([-0.3, 0.0, 0.0, 0.0], dtype=np.float32),
                    ActionInfo(
                        source_type="teleop",
                        source_id="remote:unit",
                        latency_ms=5.0,
                        extras={
                            "remote_action_connected": 1,
                            "policy_start_requested": True,
                        },
                    ),
                ),
            ]
        )
        policy = PolicyActionSource(
            policy=DummyPolicy([0.5, -0.25, 0.1, -0.9]),
            source_id="policy:unit",
            output_mode="control",
            action_scale=0.2,
        )
        source = RemoteArmedPolicyActionSource(
            remote=remote,
            policy=policy,
            source_id="policy_remote",
        )

        manual_action, manual_info = source.next_action(_obs())
        policy_action, policy_info = source.next_action(_obs())
        manual_again_action, manual_again_info = source.next_action(_obs())

        np.testing.assert_allclose(manual_action, [0.1, 0.0, 0.0, 0.0])
        self.assertEqual(manual_info.extras["policy_remote_mode"], "manual")
        np.testing.assert_allclose(policy_action, [0.1, -0.05, 0.02, -0.18])
        self.assertEqual(policy_info.source_type, "policy")
        self.assertEqual(policy_info.extras["policy_remote_mode"], "policy")
        self.assertEqual(policy_info.extras["policy_remote_activated"], 1)
        self.assertEqual(policy_info.extras["model_control"], 1)
        self.assertTrue(policy_info.extras["policy_start_requested"])
        self.assertFalse(policy_info.extras["record_start_requested"])
        self.assertEqual(policy_info.extras["toggle_mask"], 1)
        np.testing.assert_allclose(
            policy_info.extras["policy_remote_remote_action"],
            [0.2, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(manual_again_action, [-0.3, 0.0, 0.0, 0.0])
        self.assertEqual(manual_again_info.source_type, "teleop")
        self.assertEqual(manual_again_info.extras["policy_remote_mode"], "manual")
        self.assertEqual(manual_again_info.extras["policy_remote_deactivated"], 1)
        self.assertEqual(manual_again_info.extras["model_control"], 0)

    def test_shadow_zero_records_policy_action_but_returns_zero(self) -> None:
        policy = DummyPolicy([0.5, -0.25, 0.1, -0.9])
        source = PolicyActionSource(
            policy=policy,
            source_id="unit",
            action_scale=[0.2, 0.3, 0.4, 0.5],
            output_mode="shadow_zero",
            record_start_on_reset=True,
        )

        source.reset()
        action, info = source.next_action(_obs())

        np.testing.assert_allclose(action, np.zeros(4, dtype=np.float32))
        self.assertEqual(info.source_type, "policy")
        self.assertEqual(info.source_id, "unit")
        self.assertTrue(info.extras["record_start_requested"])
        np.testing.assert_allclose(
            info.extras["policy_action"],
            [0.5, -0.25, 0.1, -0.9],
        )
        np.testing.assert_allclose(
            info.extras["policy_scaled_action"],
            [0.1, -0.075, 0.04, -0.45],
        )
        np.testing.assert_allclose(info.extras["policy_returned_action"], np.zeros(4))

        _, info_again = source.next_action(_obs())
        self.assertFalse(info_again.extras["record_start_requested"])

    def test_control_mode_returns_scaled_and_clipped_action(self) -> None:
        policy = DummyPolicy([2.0, -0.5, 0.5, -2.0])
        source = PolicyActionSource(
            policy=policy,
            source_id="unit",
            action_scale=0.5,
            output_mode="control",
            clip=0.6,
        )

        action, info = source.next_action(_obs())

        np.testing.assert_allclose(action, [0.3, -0.25, 0.25, -0.3])
        np.testing.assert_allclose(info.extras["policy_action"], [0.6, -0.5, 0.5, -0.6])
        np.testing.assert_allclose(
            info.extras["policy_returned_action"],
            [0.3, -0.25, 0.25, -0.3],
        )

    def test_deadzone_assist_lifts_stable_intent_above_directional_deadzone(self) -> None:
        policy = DummyPolicy([0.33, -0.26, 0.1, -0.2])
        source = PolicyActionSource(
            policy=policy,
            source_id="unit",
            output_mode="control",
            deadzone_assist={
                "enabled": True,
                "trigger_fraction": 0.5,
                "margin": 0.02,
                "min_consecutive_steps": 2,
                "deadzone_positive": [0.6, 0.4, 0.5, 0.5],
                "deadzone_negative": [0.7, 0.5, 0.5, 0.5],
            },
        )

        first_action, first_info = source.next_action(_obs())
        second_action, second_info = source.next_action(_obs())

        np.testing.assert_allclose(first_action, [0.33, -0.26, 0.1, -0.2])
        self.assertEqual(first_info.extras["policy_deadzone_assist_active"], 0)
        np.testing.assert_allclose(second_action, [0.62, -0.52, 0.1, -0.2])
        np.testing.assert_allclose(
            second_info.extras["policy_deadzone_assist_mask"],
            [1, 1, 0, 0],
        )
        self.assertEqual(second_info.extras["policy_deadzone_assist_active"], 1)
        self.assertEqual(second_info.extras["policy_deadzone_assist_axes"], "swing+,boom-")
        np.testing.assert_allclose(
            second_info.extras["policy_scaled_action"],
            [0.33, -0.26, 0.1, -0.2],
        )
        np.testing.assert_allclose(
            second_info.extras["policy_assisted_action"],
            [0.62, -0.52, 0.1, -0.2],
        )
        np.testing.assert_allclose(second_info.extras["policy_returned_action"], second_action)

    def test_deadzone_assist_resets_when_direction_changes(self) -> None:
        class SequencePolicy:
            def __init__(self) -> None:
                self.actions = [
                    np.array([0.33, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.array([-0.36, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.array([-0.36, 0.0, 0.0, 0.0], dtype=np.float32),
                ]

            def predict(self, obs: dict) -> np.ndarray:
                return self.actions.pop(0).copy()

        source = PolicyActionSource(
            policy=SequencePolicy(),
            source_id="unit",
            output_mode="control",
            deadzone_assist={
                "enabled": True,
                "trigger_fraction": 0.5,
                "margin": 0.02,
                "min_consecutive_steps": 2,
                "deadzone_positive": [0.6, 0.4, 0.5, 0.5],
                "deadzone_negative": [0.7, 0.5, 0.5, 0.5],
            },
        )

        first_action, first_info = source.next_action(_obs())
        second_action, second_info = source.next_action(_obs())
        third_action, third_info = source.next_action(_obs())

        np.testing.assert_allclose(first_action, [0.33, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(second_action, [-0.36, 0.0, 0.0, 0.0])
        self.assertEqual(second_info.extras["policy_deadzone_assist_active"], 0)
        np.testing.assert_allclose(third_action, [-0.72, 0.0, 0.0, 0.0])
        self.assertEqual(third_info.extras["policy_deadzone_assist_axes"], "swing-")

    def test_qvel_zero_mode_feeds_zero_to_policy(self) -> None:
        policy = DummyPolicy([0.1, 0.2, 0.3, 0.4])
        source = PolicyActionSource(
            policy=policy,
            source_id="unit",
            output_mode="shadow_zero",
            qvel_mode="zero",
        )

        _action, info = source.next_action(_obs())

        np.testing.assert_allclose(policy.seen_obs[-1]["qvel"], np.zeros(4))
        np.testing.assert_allclose(info.extras["policy_qvel_input"], np.zeros(4))
        self.assertEqual(info.extras["policy_qvel_mode"], "zero")

    def test_qpos_diff_mode_feeds_filtered_qpos_derivative(self) -> None:
        policy = DummyPolicy([0.1, 0.2, 0.3, 0.4])
        source = PolicyActionSource(
            policy=policy,
            source_id="unit",
            output_mode="shadow_zero",
            qvel_mode="qpos_diff",
            qvel_diff_tau_s=0.0,
            qvel_diff_clip_rad_s=10.0,
        )
        obs0 = _obs()
        obs0["joint_timestamp_ns"] = 1_000_000_000
        obs1 = _obs()
        obs1["qpos"] = np.array([1.1, 1.8, 3.0, 4.4], dtype=np.float32)
        obs1["joint_timestamp_ns"] = 1_100_000_000

        source.next_action(obs0)
        _action, info = source.next_action(obs1)

        np.testing.assert_allclose(
            policy.seen_obs[-1]["qvel"],
            [1.0, -2.0, 0.0, 4.0],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            info.extras["policy_qvel_input"],
            [1.0, -2.0, 0.0, 4.0],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_fail_safe_zero_on_missing_camera(self) -> None:
        policy = DummyPolicy([0.1, 0.2, 0.3, 0.4])
        source = PolicyActionSource(
            policy=policy,
            source_id="unit",
            output_mode="control",
            fail_safe_zero=True,
        )
        bad_obs = {
            "qpos": np.zeros(4, dtype=np.float32),
            "qvel": np.zeros(4, dtype=np.float32),
            "images": {},
        }

        action, info = source.next_action(bad_obs)

        np.testing.assert_allclose(action, np.zeros(4, dtype=np.float32))
        self.assertIn("fail_safe_zero", info.source_id)
        self.assertIn("missing camera", info.extras["policy_error"])

    def test_policy_obs_converts_real_obs_to_act_keys(self) -> None:
        converted = _policy_obs_from_real_obs(_obs(), camera_name="fpv")

        np.testing.assert_allclose(converted["qpos"], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(converted["qvel"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(converted["image_fpv"].shape, (8, 10, 3))

    def test_policy_obs_converts_multiple_cameras(self) -> None:
        obs = _obs()
        obs["images"] = {
            "video4": np.zeros((8, 10, 3), dtype=np.uint8),
            "video5": np.ones((8, 10, 3), dtype=np.uint8),
            "video6": np.full((8, 10, 3), 2, dtype=np.uint8),
            "video7": np.full((8, 10, 3), 3, dtype=np.uint8),
        }

        converted = _policy_obs_from_real_obs(
            obs,
            camera_names=["video4", "video5", "video6", "video7"],
        )

        self.assertEqual(converted["image_video4"].shape, (8, 10, 3))
        self.assertEqual(converted["image_video7"].shape, (8, 10, 3))

    def test_policy_diagnostics_are_added_to_record_step(self) -> None:
        diagnostics: dict = {}
        _add_policy_action_diagnostics(
            diagnostics,
            {
                "policy_action": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                "policy_scaled_action": np.array([0.01, 0.02, 0.03, 0.04]),
                "policy_returned_action": np.zeros(4),
                "policy_action_scale": np.full(4, 0.1),
                "policy_output_mode": "shadow_zero",
                "policy_qvel_mode": "zero",
                "policy_qvel_input": np.zeros(4),
                "policy_error": "",
                "policy_step": 7,
                "policy_inference_latency_ms": 1.5,
                "policy_assisted_action": np.array([0.1, 0.55, 0.3, 0.4]),
                "policy_deadzone_assist_enabled": 1,
                "policy_deadzone_assist_active": 1,
                "policy_deadzone_assist_mask": np.array([0, 1, 0, 0]),
                "policy_deadzone_assist_axes": "boom+",
                "policy_bundle_dir": "policy_bundles/real_one_dig_v1",
            },
        )

        np.testing.assert_allclose(diagnostics["policy_action"], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(
            diagnostics["policy_assisted_action"],
            [0.1, 0.55, 0.3, 0.4],
        )
        self.assertEqual(diagnostics["policy_deadzone_assist_enabled"], 1)
        self.assertEqual(diagnostics["policy_deadzone_assist_active"], 1)
        np.testing.assert_allclose(
            diagnostics["policy_deadzone_assist_mask"],
            [0, 1, 0, 0],
        )
        self.assertEqual(diagnostics["policy_deadzone_assist_axes"], "boom+")
        self.assertEqual(diagnostics["policy_output_mode"], "shadow_zero")
        self.assertEqual(diagnostics["policy_qvel_mode"], "zero")
        np.testing.assert_allclose(diagnostics["policy_qvel_input"], np.zeros(4))
        self.assertEqual(diagnostics["policy_step"], 7)
        self.assertEqual(
            diagnostics["policy_bundle_dir"],
            "policy_bundles/real_one_dig_v1",
        )

    def test_receiver_test_logger_writes_lightweight_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = ReceiverTestLogger.from_config(
                {"output_dir": tmp, "run_name": "unit"},
                metadata={"task_name": "policy_test"},
                record_config_yaml="teleop:\n  input: policy\n",
            )
            logger.record_step(
                local_step=3,
                receiver_mode="armed",
                obs=_obs() | {"step_id": 3, "timestamp_ns": 10, "joint_timestamp_ns": 9},
                raw_action=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                safe_action=np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32),
                action_info=ActionInfo(
                    source_type="policy",
                    source_id="unit",
                    latency_ms=2.0,
                    extras={
                        "policy_action": np.array([0.1, 0.2, 0.3, 0.4]),
                        "policy_scaled_action": np.array([0.01, 0.02, 0.03, 0.04]),
                        "policy_assisted_action": np.array([0.01, 0.55, 0.03, 0.04]),
                        "policy_returned_action": np.zeros(4),
                        "policy_output_mode": "shadow_zero",
                        "policy_deadzone_assist_enabled": 1,
                        "policy_deadzone_assist_active": 1,
                        "policy_deadzone_assist_mask": np.array([0, 1, 0, 0]),
                        "policy_deadzone_assist_axes": "boom+",
                        "policy_error": "",
                    },
                ),
                action_sample_timestamp_ns=11,
                action_send_timestamp_ns=12,
                guard=SimpleNamespace(triggered=False, reasons=()),
                control_result={
                    "ack": True,
                    "fault_code": "",
                    "controller_timestamp_ns": 13,
                    "commanded_action": np.zeros(4, dtype=np.float32),
                },
                receiver_health=ReceiverHealthSnapshot(
                    ok=True,
                    error_code="",
                    errors=(),
                    imu_summary="1111",
                    diagnostics={},
                ),
                record_start_requested=False,
                go_home_requested=False,
            )
            logger.close()

            run_dir = Path(tmp) / "unit"
            step_lines = (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
            summary = yaml.safe_load((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(len(step_lines), 1)
        self.assertIn('"policy_output_mode": "shadow_zero"', step_lines[0])
        self.assertIn('"policy_deadzone_assist_active": 1', step_lines[0])
        self.assertIn('"policy_deadzone_assist_axes": "boom+"', step_lines[0])
        self.assertEqual(summary["steps"], 1)

    def test_load_act_policy_from_bundle_resolves_repo_a_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            (bundle / "policy_best.ckpt").write_bytes(b"ckpt")
            (bundle / "dataset_stats.pkl").write_bytes(b"stats")
            (bundle / "resolved_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "task": {
                            "camera_names": ["fpv"],
                            "episode_len": 123,
                            "equipment_model": "agxunity",
                        },
                        "policy": {
                            "device": "cpu",
                            "low_dim_keys": ["qpos", "qvel"],
                            "act_params": {
                                "chunk_size": 25,
                                "kl_weight": 0.0,
                                "hidden_dim": 512,
                                "dim_feedforward": 3200,
                                "vision_feature_scale": 0.25,
                                "proprio_feature_scale": 1.5,
                                "train_with_zero_latent": True,
                            },
                        },
                        "train": {"lr": 1e-5},
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "testbed.policies.act.adapter.ACTAdapter.from_checkpoint",
                return_value="loaded",
            ) as from_checkpoint:
                loaded = load_act_policy_from_bundle(bundle_dir=bundle, temporal_agg=True)

        self.assertEqual(loaded, "loaded")
        kwargs = from_checkpoint.call_args.kwargs
        self.assertEqual(kwargs["policy_config"]["num_queries"], 25)
        self.assertEqual(kwargs["policy_config"]["state_dim"], 8)
        self.assertEqual(kwargs["policy_config"]["camera_names"], ["fpv"])
        self.assertEqual(kwargs["policy_config"]["vision_feature_scale"], 0.25)
        self.assertEqual(kwargs["policy_config"]["proprio_feature_scale"], 1.5)
        self.assertTrue(kwargs["policy_config"]["train_with_zero_latent"])
        self.assertTrue(kwargs["temporal_agg"])
        self.assertEqual(kwargs["device"], "cpu")

    def test_load_act_policy_from_bundle_allows_null_episode_len(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            (bundle / "policy_best.ckpt").write_bytes(b"ckpt")
            (bundle / "dataset_stats.pkl").write_bytes(b"stats")
            (bundle / "resolved_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "task": {
                            "camera_names": ["fpv"],
                            "episode_len": None,
                            "equipment_model": "real_excavator",
                        },
                        "policy": {
                            "device": "cpu",
                            "low_dim_keys": ["qpos"],
                            "act_params": {"chunk_size": 20},
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "testbed.policies.act.adapter.ACTAdapter.from_checkpoint",
                return_value="loaded",
            ) as from_checkpoint:
                loaded = load_act_policy_from_bundle(bundle_dir=bundle)

        self.assertEqual(loaded, "loaded")
        kwargs = from_checkpoint.call_args.kwargs
        self.assertEqual(kwargs["policy_config"]["num_queries"], 20)
        self.assertEqual(kwargs["policy_config"]["state_dim"], 4)
        self.assertEqual(kwargs["policy_config"]["low_dim_keys"], ["qpos"])
        self.assertEqual(kwargs["policy_config"]["max_episode_len"], 400)

    def test_act_temporal_aggregation_grows_past_initial_horizon(self) -> None:
        import torch

        from testbed.policies.act.adapter import ACTAdapter

        adapter = ACTAdapter.__new__(ACTAdapter)
        adapter.device = torch.device("cpu")
        adapter._num_queries = 3
        adapter._t = 0
        adapter._all_time_actions = None
        adapter._temporal_weight_cache = {}
        adapter._max_episode_len = 4

        for step in range(7):
            a_hat = torch.ones((1, adapter._num_queries, 4), dtype=torch.float32) * (step + 1)
            action = adapter._aggregate(a_hat)
            self.assertEqual(action.shape, (4,))
            adapter._t += 1

        self.assertGreaterEqual(adapter._all_time_actions.shape[0], 7)
        self.assertGreaterEqual(
            adapter._all_time_actions.shape[1],
            adapter._all_time_actions.shape[0] + adapter._num_queries,
        )


if __name__ == "__main__":
    unittest.main()
