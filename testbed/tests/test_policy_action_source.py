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
    def test_remote_armed_policy_switches_on_record_start(self) -> None:
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

        np.testing.assert_allclose(manual_action, [0.1, 0.0, 0.0, 0.0])
        self.assertEqual(manual_info.extras["policy_remote_mode"], "manual")
        np.testing.assert_allclose(policy_action, [0.1, -0.05, 0.02, -0.18])
        self.assertEqual(policy_info.source_type, "policy")
        self.assertEqual(policy_info.extras["policy_remote_mode"], "policy")
        self.assertEqual(policy_info.extras["policy_remote_activated"], 1)
        self.assertTrue(policy_info.extras["policy_start_requested"])
        self.assertFalse(policy_info.extras["record_start_requested"])
        self.assertEqual(policy_info.extras["toggle_mask"], 1)
        np.testing.assert_allclose(
            policy_info.extras["policy_remote_remote_action"],
            [0.2, 0.0, 0.0, 0.0],
        )

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
                "policy_error": "",
                "policy_step": 7,
                "policy_inference_latency_ms": 1.5,
                "policy_bundle_dir": "policy_bundles/real_one_dig_v1",
            },
        )

        np.testing.assert_allclose(diagnostics["policy_action"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(diagnostics["policy_output_mode"], "shadow_zero")
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
                        "policy_returned_action": np.zeros(4),
                        "policy_output_mode": "shadow_zero",
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
        self.assertTrue(kwargs["policy_config"]["train_with_zero_latent"])
        self.assertTrue(kwargs["temporal_agg"])
        self.assertEqual(kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
