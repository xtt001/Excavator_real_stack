from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import yaml
from scripts.summarize_policy_test_log import _compute_metrics

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
    _add_scripted_cycle_diagnostics,
    _pace_before_observation,
    _policy_frame_alignment_active,
    _policy_remote_status_payload,
)
from testbed.cli.teleop_remote import _format_receiver_policy_status
from testbed.data.recorder import EpisodeRecorder
from testbed.policies.runtime_gate_stack import RuntimeGateResult
from testbed.tasks.act_cycle_planner import ABCyclePlanner


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


class DummyIntentPolicy(DummyPolicy):
    def predict_action_and_intent(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        self.seen_obs.append(obs)
        return self.action.copy(), np.full(8, 0.75, dtype=np.float32)


class DummyRuntimeGateStack:
    stack_id = "E52-test"

    def __init__(self) -> None:
        self.reset_count = 0
        self.calls = 0

    def reset(self) -> None:
        self.reset_count += 1

    def step(self, **kwargs) -> RuntimeGateResult:
        self.calls += 1
        return RuntimeGateResult(
            action=np.array([0.4, -0.3, 0.2, -0.1], dtype=np.float32),
            gohome_requested=True,
            diagnostics={
                "policy_gate_stack_id": self.stack_id,
                "policy_phase_gated_action": np.array([0.45, -0.3, 0.2, -0.1]),
                "policy_snap_action": np.array([0.5, -0.3, 0.2, -0.1]),
                "policy_temporal_direction_action": np.array([0.4, -0.3, 0.2, -0.1]),
                "gohome_raw_active": 1,
                "gohome_request_active": 1,
            },
        )


def _obs() -> dict:
    return {
        "qpos": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "qvel": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "images": {
            "fpv": np.zeros((8, 10, 3), dtype=np.uint8),
        },
    }


def _runtime_gate_extras() -> dict:
    return {
        "policy_gate_stack_id": "E52",
        "policy_intent_probabilities": np.arange(8, dtype=np.float32) / 10.0,
        "phase_gate_prob": 0.91,
        "phase_gate_threshold": 0.15,
        "phase_gate_inactive_scale": 0.5,
        "phase_gate_active": 1,
        "policy_phase_gated_action": np.array([0.0, 0.2, 0.0, 0.4]),
        "policy_snap_active_mask": np.array([0, 1, 0, 0]),
        "policy_snap_action": np.array([0.0, 0.25, 0.0, 0.4]),
        "policy_snap_margin": 0.02,
        "policy_snap_intent_threshold": 0.7,
        "temporal_direction_gate_probabilities": np.linspace(0.1, 0.8, 8),
        "temporal_direction_gate_threshold": 0.5,
        "temporal_direction_gate_inactive_scale": 0.75,
        "temporal_direction_gate_active_mask": np.array([0, 0, 0, 1, 0, 0, 0, 1]),
        "policy_temporal_direction_action": np.array([0.0, 0.18, 0.0, 0.3]),
        "gohome_candidate_probability": 0.99,
        "gohome_candidate_threshold": 0.97,
        "gohome_candidate_required_steps": 10,
        "gohome_candidate_consecutive_steps": 10,
        "gohome_eligibility_probability": 0.82,
        "gohome_eligibility_threshold": 0.8,
        "gohome_eligibility_required_steps": 3,
        "gohome_eligibility_consecutive_steps": 3,
        "gohome_request_probability": 0.82,
        "gohome_raw_active": 1,
        "gohome_request_active": 1,
        "gohome_request_suppressed": 1,
        "gohome_request_suppression_reason": "policy_output_mode_shadow_zero",
    }


class PolicyActionSourceTests(unittest.TestCase):
    def test_policy_summary_reports_aligned_pump_chain_timing(self) -> None:
        steps = []
        for index in range(3):
            base = 1_000_000_000 + index * 50_000_000
            steps.append(
                {
                    "local_step": index,
                    "wall_time_ns": base,
                    "policy_remote_mode": "policy",
                    "policy_inference_latency_ms": 25.0,
                    "action_sample_timestamp_ns": base,
                    "action_update_timestamp_ns": base + 25_000_000,
                    "action_send_timestamp_ns": base + 27_000_000,
                    "image_timestamp_ns": {"video4": base - 10_000_000},
                    "policy_frame_alignment_enabled": 1,
                    "policy_frame_reused": int(index > 0),
                    "action_pump_command_current": 1,
                    "receiver_health_ok": 1,
                    "controller_ack": 1,
                }
            )

        metrics = _compute_metrics(steps, warmup_steps=0)

        self.assertEqual(metrics["policy_loop_p50_ms"], 50.0)
        self.assertEqual(metrics["sample_to_update_p50_ms"], 25.0)
        self.assertEqual(metrics["update_to_send_p50_ms"], 2.0)
        self.assertEqual(metrics["sample_to_send_p50_ms"], 27.0)
        self.assertEqual(metrics["image_to_sample_p50_ms"], 10.0)
        self.assertEqual(metrics["image_to_send_p50_ms"], 37.0)
        self.assertEqual(metrics["frame_alignment_enabled_count"], 3)
        self.assertEqual(metrics["frame_reused_count"], 2)
        self.assertEqual(metrics["pump_current_count"], 3)
        self.assertEqual(metrics["pump_stale_count"], 0)

    def test_policy_prepare_warms_once_and_resets_temporal_state(self) -> None:
        policy = DummyPolicy([0.5, -0.25, 0.1, -0.9])
        source = PolicyActionSource(
            policy=policy,
            source_id="unit",
            output_mode="control",
            inference_warmup_steps=3,
        )

        result = source.prepare(_obs())
        repeated = source.prepare(_obs())

        self.assertEqual(result["warmup_steps"], 3)
        self.assertEqual(result["prepared"], 1)
        self.assertGreaterEqual(result["elapsed_s"], 0.0)
        self.assertEqual(len(policy.seen_obs), 3)
        self.assertEqual(policy.reset_count, 1)
        self.assertEqual(repeated["warmup_steps"], 0)
        self.assertEqual(len(policy.seen_obs), 3)

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

    def test_policy_remote_advances_policy_once_per_new_image_frame(self) -> None:
        remote = DummyActionSource([])
        model = DummyPolicy([0.5, -0.25, 0.1, -0.9])
        policy = PolicyActionSource(
            policy=model,
            source_id="policy:unit",
            output_mode="control",
        )
        source = RemoteArmedPolicyActionSource(
            remote=remote,
            policy=policy,
            start_in_policy=True,
            infer_on_new_frame=True,
        )
        obs = _obs()
        obs["image_timestamp_ns"] = {"fpv": 1_000_000_000}

        first_action, first_info = source.next_action(obs)
        reused_action, reused_info = source.next_action(obs)
        newer_obs = dict(obs)
        newer_obs["image_timestamp_ns"] = {"fpv": 1_050_000_000}
        newer_action, newer_info = source.next_action(newer_obs)

        self.assertEqual(len(model.seen_obs), 2)
        np.testing.assert_allclose(reused_action, first_action)
        np.testing.assert_allclose(newer_action, first_action)
        self.assertEqual(first_info.extras["policy_frame_reused"], 0)
        self.assertEqual(reused_info.extras["policy_frame_reused"], 1)
        self.assertEqual(reused_info.extras["policy_frame_reuse_count"], 1)
        self.assertEqual(newer_info.extras["policy_frame_reused"], 0)
        self.assertEqual(newer_info.extras["policy_frame_reuse_count"], 0)

    def test_frame_alignment_paces_only_when_enabled(self) -> None:
        with patch("testbed.cli.record_real._sleep_to_rate") as sleep:
            _pace_before_observation(enabled=False, rate_hz=20.0)
            sleep.assert_not_called()

            stop = lambda: False
            _pace_before_observation(
                enabled=True,
                rate_hz=20.0,
                should_stop=stop,
            )
            sleep.assert_called_once_with(20.0, should_stop=stop)

    def test_policy_remote_frame_alignment_never_paces_manual_mode(self) -> None:
        class Source:
            def __init__(self, model_control: bool) -> None:
                self.model_control = model_control

            def policy_status(self) -> dict[str, int]:
                return {"model_control": int(self.model_control)}

        source = Source(model_control=False)
        self.assertFalse(
            _policy_frame_alignment_active(
                enabled=True,
                input_device="policy_remote",
                action_source=source,
            )
        )
        source.model_control = True
        self.assertTrue(
            _policy_frame_alignment_active(
                enabled=True,
                input_device="policy_remote",
                action_source=source,
            )
        )
        self.assertTrue(
            _policy_frame_alignment_active(
                enabled=True,
                input_device="policy",
                action_source=object(),
            )
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

    def test_act_only_mode_reports_intent_without_runtime_gate(self) -> None:
        source = PolicyActionSource(
            policy=DummyIntentPolicy([0.5, -0.25, 0.1, -0.9]),
            source_id="unit",
            output_mode="control",
            report_intent=True,
        )

        action, info = source.next_action(_obs())

        np.testing.assert_allclose(action, [0.5, -0.25, 0.1, -0.9])
        np.testing.assert_allclose(
            info.extras["policy_intent_probabilities"],
            np.full(8, 0.75),
        )
        self.assertNotIn("policy_gate_stack_id", info.extras)

    def test_policy_remote_status_reports_raw_command_and_intent(self) -> None:
        action_info = ActionInfo(
            source_type="policy",
            source_id="policy:unit",
            extras={
                "policy_remote_mode": "policy",
                "model_control": 1,
                "policy_action": np.array([0.1, 0.2, 0.3, 0.4]),
                "policy_assisted_action": np.array([0.1, 0.2, 0.3, 0.42]),
                "policy_returned_action": np.array([0.1, 0.2, 0.3, 0.4]),
                "policy_intent_probabilities": np.arange(8) / 10.0,
                "scripted_cycle_enabled": 1,
                "scripted_cycle_active": 1,
                "scripted_cycle_ready_side": "B",
                "scripted_cycle_ready_blockers": "swing_not_stable",
                "scripted_cycle_excursion_observed": 1,
                "scripted_cycle_review_due": 0,
                "scripted_cycle_event": "goal_committed",
                "scripted_cycle_fault": "",
                "scripted_cycle_activation_rejected_reason": "",
                "planner_cycle_index": 0,
                "planner_target_side": "A",
            },
        )
        payload = _policy_remote_status_payload(object(), action_info)
        payload["commanded_action"] = [0.11, 0.19, 0.31, 0.39]

        np.testing.assert_allclose(
            payload["policy_action"], [0.1, 0.2, 0.3, 0.4]
        )
        self.assertEqual(len(payload["policy_intent_probabilities"]), 8)
        text = _format_receiver_policy_status(
            SimpleNamespace(payload=payload, receive_time_ns=1)
        )
        self.assertIn("model_raw=", text)
        self.assertIn("assisted=", text)
        self.assertIn("commanded=", text)
        self.assertIn("intent8(sw+,sw-,bo+,bo-,st+,st-,bk+,bk-)=", text)
        self.assertIn("bucket_intent=+0.60/-0.70", text)
        self.assertIn("scripted_cycle=active", text)
        self.assertIn("cycle=0", text)
        self.assertIn("goal=A(left)", text)

    def test_runtime_gates_keep_shadow_zero_and_expose_raw_gohome_decision(self) -> None:
        policy = DummyIntentPolicy([0.5, -0.25, 0.1, -0.9])
        gate_stack = DummyRuntimeGateStack()
        source = PolicyActionSource(
            policy=policy,
            source_id="unit",
            action_scale=0.5,
            output_mode="shadow_zero",
            runtime_gate_stack=gate_stack,
        )

        source.reset()
        action, info = source.next_action(_obs())

        np.testing.assert_allclose(action, np.zeros(4))
        np.testing.assert_allclose(info.extras["policy_action"], [0.5, -0.25, 0.1, -0.9])
        np.testing.assert_allclose(info.extras["policy_scaled_action"], [0.2, -0.15, 0.1, -0.05])
        np.testing.assert_allclose(info.extras["policy_returned_action"], np.zeros(4))
        self.assertEqual(info.extras["policy_gate_stack_id"], "E52-test")
        self.assertEqual(info.extras["gohome_raw_active"], 1)
        self.assertEqual(info.extras["gohome_request_active"], 1)
        self.assertEqual(info.extras["gohome_request_suppressed"], 1)
        self.assertEqual(
            info.extras["gohome_request_suppression_reason"],
            "policy_output_mode_shadow_zero",
        )
        self.assertFalse(info.extras["go_home_requested"])
        self.assertEqual(gate_stack.reset_count, 1)
        self.assertEqual(gate_stack.calls, 1)

    def test_runtime_gates_emit_gohome_request_in_control_mode(self) -> None:
        source = PolicyActionSource(
            policy=DummyIntentPolicy([0.5, -0.25, 0.1, -0.9]),
            source_id="unit",
            output_mode="control",
            runtime_gate_stack=DummyRuntimeGateStack(),
        )

        action, info = source.next_action(_obs())

        np.testing.assert_allclose(action, [0.4, -0.3, 0.2, -0.1])
        self.assertTrue(info.extras["go_home_requested"])
        self.assertEqual(info.extras["gohome_request_suppressed"], 0)
        self.assertEqual(info.extras["gohome_request_suppression_reason"], "")

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
        np.testing.assert_allclose(
            second_info.extras["policy_deadzone_assist_trigger_fraction"],
            [0.5, 0.5, 0.5, 0.5],
        )
        np.testing.assert_allclose(
            second_info.extras["policy_returned_action"], second_action
        )

    def test_deadzone_assist_can_be_limited_to_bucket_axis(self) -> None:
        source = PolicyActionSource(
            policy=DummyPolicy([0.4, -0.3, 0.3, 0.3]),
            source_id="unit",
            output_mode="control",
            deadzone_assist={
                "enabled": True,
                "axis_enabled": [False, False, False, True],
                "trigger_fraction": 0.5,
                "margin": 0.02,
                "min_consecutive_steps": 2,
                "deadzone_positive": [0.6, 0.4, 0.5, 0.408],
                "deadzone_negative": [0.7, 0.5, 0.5, 0.508],
            },
        )

        first, _ = source.next_action(_obs())
        second, info = source.next_action(_obs())

        np.testing.assert_allclose(first, [0.4, -0.3, 0.3, 0.3])
        np.testing.assert_allclose(second, [0.4, -0.3, 0.3, 0.428])
        np.testing.assert_array_equal(
            info.extras["policy_deadzone_assist_axis_enabled"], [0, 0, 0, 1]
        )
        np.testing.assert_array_equal(
            info.extras["policy_deadzone_assist_mask"], [0, 0, 0, 1]
        )
        self.assertEqual(info.extras["policy_deadzone_assist_axes"], "bucket+")

    def test_deadzone_assist_supports_per_axis_trigger_fraction(self) -> None:
        source = PolicyActionSource(
            policy=DummyPolicy([0.24, 0.1, 0.2, 0.16]),
            source_id="unit",
            output_mode="control",
            deadzone_assist={
                "enabled": True,
                "axis_enabled": [True, True, True, True],
                "trigger_fraction": [0.36, 0.5, 0.5, 0.375],
                "margin": [0.02, 0.02, 0.02, 0.02],
                "min_consecutive_steps": 2,
                "deadzone_positive": [0.661, 0.259, 0.5, 0.408],
                "deadzone_negative": [0.721, 0.357, 0.5, 0.508],
            },
        )

        first, _ = source.next_action(_obs())
        second, info = source.next_action(_obs())

        np.testing.assert_allclose(first, [0.24, 0.1, 0.2, 0.16])
        np.testing.assert_allclose(second, [0.681, 0.1, 0.2, 0.428])
        np.testing.assert_array_equal(
            info.extras["policy_deadzone_assist_mask"], [1, 0, 0, 1]
        )
        np.testing.assert_allclose(
            info.extras["policy_deadzone_assist_trigger_fraction"],
            [0.36, 0.5, 0.5, 0.375],
        )

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

    def test_policy_obs_preserves_real_transition_condition(self) -> None:
        obs = _obs()
        obs["real_transition_condition_v1"] = np.asarray([1.0, 1.0])

        converted = _policy_obs_from_real_obs(obs, camera_name="fpv")

        np.testing.assert_allclose(
            converted["real_transition_condition_v1"], [1.0, 1.0]
        )

    def test_policy_source_attaches_committed_planner_condition(self) -> None:
        planner = ABCyclePlanner("ABBABABA", loop=False)
        policy = DummyPolicy([0.1, 0.2, 0.3, 0.4])
        source = PolicyActionSource(
            policy=policy,
            source_id="planner-test",
            camera_name="fpv",
            cycle_planner=planner,
            output_mode="shadow_zero",
        )

        goal = source.commit_cycle_goal()
        self.assertEqual(policy.reset_count, 1)
        action, info = source.next_action(_obs())

        np.testing.assert_allclose(action, np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(
            policy.seen_obs[-1]["real_transition_condition_v1"], [1.0, 1.0]
        )
        assert info.extras["planner_goal_epoch"] == goal.goal_epoch
        assert info.extras["planner_target_side"] == "B"
        assert source.mark_cycle_target_ready("B").transition == "B->B"
        source.commit_cycle_goal()
        self.assertEqual(policy.reset_count, 2)

    def test_policy_source_can_preserve_policy_state_across_goal_commit(self) -> None:
        planner = ABCyclePlanner("ABA", loop=False)
        policy = DummyPolicy([0.1, 0.2, 0.3, 0.4])
        source = PolicyActionSource(
            policy=policy,
            source_id="planner-continuous-test",
            camera_name="fpv",
            cycle_planner=planner,
            reset_policy_on_goal=False,
            output_mode="control",
        )

        source.commit_cycle_goal()
        source.mark_cycle_target_ready("B")
        source.commit_cycle_goal()

        self.assertEqual(policy.reset_count, 0)

    @patch("testbed.actions.policy.load_act_policy_from_bundle")
    def test_policy_source_from_config_wires_cycle_planner(self, load_policy) -> None:
        load_policy.return_value = DummyPolicy([0.1, 0.2, 0.3, 0.4])

        source = PolicyActionSource.from_config(
            {
                "bundle_dir": "/tmp/conditioned-act-bundle",
                "cycle_planner": {
                    "enabled": True,
                    "pattern": "ABBABABA",
                    "loop": False,
                },
            }
        )

        assert source.cycle_planner is not None
        assert source.cycle_planner.initial_side == "A"
        assert source.cycle_planner.done is False

    def test_policy_source_prepare_waits_for_planner_commit(self) -> None:
        source = PolicyActionSource(
            policy=DummyPolicy([0.1, 0.2, 0.3, 0.4]),
            source_id="planner-prepare-test",
            camera_name="fpv",
            cycle_planner=ABCyclePlanner("AB", loop=False),
            inference_warmup_steps=3,
        )

        result = source.prepare(_obs())

        assert result["planner_waiting_for_commit"] == 1
        assert result["warmup_steps"] == 0

    @patch("testbed.actions.policy.load_act_policy_from_bundle")
    def test_policy_source_from_config_wires_inline_variable_script(
        self, load_policy
    ) -> None:
        load_policy.return_value = DummyPolicy([0.1, 0.2, 0.3, 0.4])

        source = PolicyActionSource.from_config(
            {
                "bundle_dir": "/tmp/conditioned-act-bundle",
                "cycle_planner": {
                    "enabled": True,
                    "script": {
                        "initial_side": "A",
                        "steps": [
                            {"target_side": "B"},
                            {"target_side": "A"},
                        ],
                    },
                },
            }
        )

        goal = source.commit_cycle_goal()
        assert goal.transition == "A->B"
        assert goal.script_step_index == 0

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

    def test_scripted_cycle_diagnostics_are_added_to_record_step(self) -> None:
        diagnostics: dict = {}

        _add_scripted_cycle_diagnostics(
            diagnostics,
            {
                "scripted_cycle_enabled": 1,
                "scripted_cycle_active": 1,
                "scripted_cycle_completed": 0,
                "scripted_cycle_goal_changed": 1,
                "scripted_cycle_excursion_observed": 1,
                "scripted_cycle_review_due": 0,
                "scripted_cycle_ready_window_complete": 1,
                "scripted_cycle_ready_swing_stable": 1,
                "scripted_cycle_cycle_elapsed_s": 12.5,
                "scripted_cycle_run_elapsed_s": 24.5,
                "scripted_cycle_ready_swing_qpos_rad": 0.2,
                "scripted_cycle_ready_swing_qvel_abs_max_rad_s": 0.01,
                "scripted_cycle_ready_side": "B",
                "scripted_cycle_ready_blockers": "",
                "scripted_cycle_event": "cycle_advanced",
                "scripted_cycle_fault": "",
                "scripted_cycle_stop_reason": "",
                "scripted_cycle_activation_rejected_reason": "",
                "planner_cycle_index": 1,
                "planner_goal_epoch": 2,
                "planner_target_side": "A",
                "planner_condition": np.asarray([-1.0, 1.0]),
            },
        )

        self.assertEqual(diagnostics["planner_cycle_index"], 1)
        self.assertEqual(diagnostics["planner_target_side"], "A")
        self.assertEqual(diagnostics["scripted_cycle_ready_side"], "B")
        np.testing.assert_allclose(diagnostics["planner_condition"], [-1.0, 1.0])

    def test_runtime_gate_diagnostics_are_added_to_hdf5_step(self) -> None:
        diagnostics: dict = {}
        _add_policy_action_diagnostics(
            diagnostics,
            {
                "policy_action": np.array([0.1, 0.2, 0.3, 0.4]),
                **_runtime_gate_extras(),
            },
        )

        for key in _runtime_gate_extras():
            self.assertIn(key, diagnostics)
        np.testing.assert_allclose(
            diagnostics["policy_intent_probabilities"],
            np.arange(8) / 10.0,
        )
        np.testing.assert_array_equal(
            diagnostics["temporal_direction_gate_active_mask"],
            [0, 0, 0, 1, 0, 0, 0, 1],
        )
        self.assertEqual(diagnostics["gohome_candidate_required_steps"], 10)
        self.assertEqual(diagnostics["gohome_eligibility_consecutive_steps"], 3)
        self.assertEqual(diagnostics["gohome_request_suppressed"], 1)
        self.assertEqual(
            diagnostics["gohome_request_suppression_reason"],
            "policy_output_mode_shadow_zero",
        )
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(tmp, 0, camera_names=[])
            recorder.record(
                {"qpos": np.zeros(4), "qvel": np.zeros(4)},
                np.zeros(4),
                diagnostics=diagnostics,
            )
            path = recorder.save()
            with h5py.File(path, "r") as handle:
                np.testing.assert_allclose(
                    handle["diagnostics/policy_intent_probabilities"][0],
                    np.arange(8) / 10.0,
                )
                np.testing.assert_array_equal(
                    handle["diagnostics/temporal_direction_gate_active_mask"][0],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                )
                self.assertEqual(
                    handle["diagnostics/gohome_candidate_required_steps"][0],
                    10,
                )
                self.assertEqual(
                    handle[
                        "diagnostics/gohome_request_suppression_reason"
                    ].asstr()[0],
                    "policy_output_mode_shadow_zero",
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
                        "remote_action_seq": 7,
                        "remote_action_host_sample_ns": 100,
                        "remote_action_receive_ns": 110,
                        "remote_action_age_ms": 1.5,
                        "remote_action_stale": 0,
                        "remote_action_drop_count": 2,
                        "remote_action_connected": 1,
                        "scripted_cycle_enabled": 1,
                        "scripted_cycle_active": 1,
                        "scripted_cycle_completed": 0,
                        "scripted_cycle_goal_changed": 1,
                        "scripted_cycle_excursion_observed": 1,
                        "scripted_cycle_review_due": 0,
                        "scripted_cycle_ready_window_complete": 1,
                        "scripted_cycle_ready_swing_stable": 1,
                        "scripted_cycle_ready_target_supported": 1,
                        "scripted_cycle_stop_latched": 0,
                        "scripted_cycle_ready_side": "B",
                        "scripted_cycle_ready_blockers": "",
                        "scripted_cycle_fault": "",
                        "scripted_cycle_stop_reason": "",
                        "scripted_cycle_event": "goal_committed",
                        "scripted_cycle_activation_rejected_reason": "",
                        "planner_cycle_index": 0,
                        "planner_goal_epoch": 1,
                        "planner_target_side": "A",
                        "planner_condition": np.array([-1.0, 1.0]),
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
                    "raw_low_level_command": np.arange(8, dtype=np.float32),
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
            logger.record_termination(
                stop_reason="aborted",
                zero_result={
                    "ack": True,
                    "fault_code": "",
                    "controller_timestamp_ns": 14,
                    "commanded_action": np.zeros(4, dtype=np.float32),
                    "raw_low_level_command": np.zeros(4, dtype=np.float32),
                },
            )
            logger.close()

            run_dir = Path(tmp) / "unit"
            step_lines = (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
            summary = yaml.safe_load((run_dir / "summary.json").read_text(encoding="utf-8"))
            termination = yaml.safe_load(
                (run_dir / "termination.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(step_lines), 1)
        self.assertIn('"policy_output_mode": "shadow_zero"', step_lines[0])
        self.assertIn('"policy_deadzone_assist_active": 1', step_lines[0])
        self.assertIn('"policy_deadzone_assist_axes": "boom+"', step_lines[0])
        self.assertIn('"remote_action_seq": 7', step_lines[0])
        self.assertIn('"remote_action_host_sample_ns": 100', step_lines[0])
        self.assertIn('"remote_action_receive_ns": 110', step_lines[0])
        self.assertIn('"scripted_cycle_enabled": 1', step_lines[0])
        self.assertIn('"planner_target_side": "A"', step_lines[0])
        self.assertIn('"planner_condition": [-1.0, 1.0]', step_lines[0])
        self.assertIn('"raw_low_level_command": [0.0, 1.0, 2.0', step_lines[0])
        self.assertEqual(summary["steps"], 1)
        self.assertTrue(summary["zero_command_confirmed"])
        self.assertEqual(termination["stop_reason"], "aborted")
        self.assertTrue(termination["zero_command_confirmed"])
        self.assertEqual(termination["last_step"]["local_step"], 3)

    def test_receiver_test_logger_passes_gate_stack_policy_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = ReceiverTestLogger.from_config(
                {"output_dir": tmp, "run_name": "unit"},
                metadata={"task_name": "policy_test"},
                record_config_yaml="teleop:\n  input: policy\n",
            )
            logger.record_step(
                local_step=4,
                receiver_mode="armed",
                obs=_obs() | {"step_id": 4, "timestamp_ns": 10, "joint_timestamp_ns": 9},
                raw_action=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                safe_action=np.zeros(4, dtype=np.float32),
                action_info=ActionInfo(
                    source_type="policy",
                    source_id="unit",
                    latency_ms=2.0,
                    extras={
                        "policy_action": np.array([0.1, 0.2, 0.3, 0.4]),
                        "policy_output_mode": "shadow_zero",
                        **_runtime_gate_extras(),
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

            step = yaml.safe_load(
                ((Path(tmp) / "unit" / "steps.jsonl").read_text(encoding="utf-8").splitlines())[0]
            )

        self.assertEqual(step["policy_gate_stack_id"], "E52")
        self.assertEqual(step["go_home_requested"], 0)
        self.assertEqual(step["gohome_request_active"], 1)
        self.assertEqual(step["gohome_request_suppressed"], 1)
        self.assertEqual(
            step["gohome_request_suppression_reason"],
            "policy_output_mode_shadow_zero",
        )
        self.assertEqual(len(step["policy_intent_probabilities"]), 8)
        self.assertEqual(len(step["temporal_direction_gate_probabilities"]), 8)
        self.assertAlmostEqual(step["phase_gate_threshold"], 0.15)
        self.assertEqual(step["gohome_candidate_required_steps"], 10)
        self.assertEqual(step["gohome_eligibility_consecutive_steps"], 3)
        np.testing.assert_allclose(
            step["policy_temporal_direction_action"],
            [0.0, 0.18, 0.0, 0.3],
        )
        self.assertAlmostEqual(step["gohome_request_probability"], 0.82)

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
                loaded = load_act_policy_from_bundle(
                    bundle_dir=bundle,
                    temporal_agg=True,
                    inference_precision="fp16",
                    inference_compile=True,
                    inference_compile_mode="reduce-overhead",
                    inference_compile_dynamic=False,
                    device_uint8_preprocess=True,
                )

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
        self.assertEqual(kwargs["inference_precision"], "fp16")
        self.assertTrue(kwargs["inference_compile"])
        self.assertEqual(kwargs["inference_compile_mode"], "reduce-overhead")
        self.assertFalse(kwargs["inference_compile_dynamic"])
        self.assertTrue(kwargs["device_uint8_preprocess"])

    def test_load_act_policy_resolves_relative_checkpoint_inside_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            accepted = bundle / "policy_accepted.ckpt"
            accepted.write_bytes(b"accepted")
            (bundle / "dataset_stats.pkl").write_bytes(b"stats")
            (bundle / "resolved_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "task": {
                            "camera_names": ["fpv"],
                            "episode_len": 20,
                        },
                        "policy": {
                            "device": "cpu",
                            "low_dim_keys": ["qpos"],
                            "act_params": {"chunk_size": 20},
                        },
                        "train": {},
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "testbed.policies.act.adapter.ACTAdapter.from_checkpoint",
                return_value="accepted-loaded",
            ) as from_checkpoint:
                loaded = load_act_policy_from_bundle(
                    bundle_dir=bundle,
                    ckpt_path="policy_accepted.ckpt",
                )

        self.assertEqual(loaded, "accepted-loaded")
        self.assertEqual(from_checkpoint.call_args.kwargs["ckpt_path"], accepted)

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

    def test_load_act_policy_from_bundle_relocates_deadzone_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            (bundle / "policy_best.ckpt").write_bytes(b"ckpt")
            (bundle / "dataset_stats.pkl").write_bytes(b"stats")
            bundled_thresholds = bundle / "deadzone_policy_raw_for_runtime_scale.json"
            bundled_thresholds.write_text("{}", encoding="utf-8")
            (bundle / "resolved_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "task": {"camera_names": ["video4", "video5"]},
                        "policy": {"device": "cpu", "low_dim_keys": ["qpos"]},
                        "train": {
                            "intent_loss": {
                                "enabled": True,
                                "threshold_json": (
                                    "/data/training/deadzone_policy_raw_for_runtime_scale.json"
                                ),
                            }
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
        intent_loss = from_checkpoint.call_args.kwargs["policy_config"]["intent_loss"]
        self.assertEqual(intent_loss["threshold_json"], str(bundled_thresholds))

    def test_load_act_policy_from_bundle_relocates_training_deadzone_from_contracts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            (bundle / "policy_best.ckpt").write_bytes(b"ckpt")
            (bundle / "dataset_stats.pkl").write_bytes(b"stats")
            contracts = bundle / "contracts"
            contracts.mkdir()
            bundled_thresholds = contracts / "direct_deadzone.json"
            bundled_thresholds.write_text("{}", encoding="utf-8")
            (bundle / "resolved_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "task": {"camera_names": ["video4"]},
                        "policy": {"device": "cpu", "low_dim_keys": ["qpos"]},
                        "train": {
                            "deadzone_loss": {
                                "enabled": True,
                                "threshold_json": "/missing/direct_deadzone.json",
                            }
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
        deadzone_loss = from_checkpoint.call_args.kwargs["policy_config"][
            "deadzone_loss"
        ]
        self.assertEqual(deadzone_loss["threshold_json"], str(bundled_thresholds))

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
