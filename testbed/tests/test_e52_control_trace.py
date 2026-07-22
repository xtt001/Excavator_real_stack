from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from scripts.summarize_e52_control_trace import summarize_trace

ROOT = Path(__file__).resolve().parents[2]
CONTROL_SCRIPT = ROOT / "scripts/run_e52_policy_control_trace.sh"
ACT_BASELINE_CONTROL_SCRIPT = (
    ROOT / "scripts/run_e52_act_baseline_control_trace.sh"
)
ACT_BASELINE_POLICY_REMOTE_SCRIPT = (
    ROOT / "scripts/run_e52_act_baseline_policy_remote_stack.sh"
)


def _step(index: int) -> dict:
    action = [0.1 + index * 0.01, -0.2, 0.0, 0.3]
    return {
        "wall_time_ns": 1_000_000_000 + index * 20_000_000,
        "local_step": index,
        "qpos": [0.01 * index, -0.1 - 0.005 * index, -0.5, 0.2 + 0.02 * index],
        "qvel": [0.02, -0.03, 0.0, 0.04],
        "policy_action": action,
        "policy_scaled_action": action,
        "policy_intent_probabilities": [0.8, 0.1, 0.2, 0.7, 0.0, 0.0, 0.1, 0.9],
        "phase_gate_prob": 0.8,
        "policy_phase_gated_action": action,
        "policy_snap_action": action,
        "temporal_direction_gate_probabilities": [0.8] * 8,
        "policy_temporal_direction_action": action,
        "policy_returned_action": action,
        "safe_action": action,
        "commanded_action": action,
        "raw_low_level_command": [10.0, 20.0, 30.0, 40.0],
        "controller_ack": 1,
        "controller_fault_code": "",
        "receiver_health_ok": 1,
        "receiver_health_error_code": "",
        "guard_triggered": 0,
        "guard_reasons": [],
        "go_home_requested": 0,
        "gohome_request_probability": 0.1,
        "policy_output_mode": "control",
        "policy_error": "",
        "fpv_image_path": "camera_frames/video4_000000.jpg" if index == 0 else None,
    }


def test_summarize_trace_writes_manifest_timeline_and_terminal_context(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    steps = [_step(index) for index in range(3)]
    (run / "steps.jsonl").write_text(
        "".join(json.dumps(step) + "\n" for step in steps), encoding="utf-8"
    )
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "metadata": {"task_name": "fixture"},
                "record_config_yaml": (
                    "teleop:\n  input: policy\n  policy:\n    output_mode: control\n"
                ),
                "image_capture": {"interval_steps": 5},
            }
        ),
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps({"steps": 3, "stop_reason": "aborted"}), encoding="utf-8"
    )
    (run / "termination.json").write_text(
        json.dumps(
            {
                "stop_reason": "aborted",
                "zero_command_requested": True,
                "zero_command_confirmed": True,
                "zero_control_result": {"ack": True, "commanded_action": [0, 0, 0, 0]},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"teleop": {"policy": {"output_mode": "shadow_zero"}}}),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "candidate_package_manifest.json").write_text(
        json.dumps(
            {
                "candidate_id": "E52-fixture",
                "artifacts": [
                    {"name": "policy", "path": "/source/policy", "sha256": "abc"}
                ],
            }
        ),
        encoding="utf-8",
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"ok": True}), encoding="utf-8")

    paths = summarize_trace(
        run_dir=run,
        config_path=config,
        bundle_dir=bundle,
        preflight_report=preflight,
    )

    assert set(paths) == {
        "run_manifest",
        "trace_summary",
        "trace_context",
        "trace_timeline",
    }
    assert all(path.is_file() for path in paths.values())
    summary = json.loads(paths["trace_summary"].read_text(encoding="utf-8"))
    assert summary["trace_complete"] is True
    assert summary["steps"] == 3
    assert summary["termination"]["zero_command_confirmed"] is True
    assert summary["stick"]["commanded_max_abs"] == 0.0
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    assert manifest["bundle"]["candidate_id"] == "E52-fixture"
    assert manifest["preflight_report"]["ok"] is True
    assert manifest["config"]["record_policy_output_mode"] == "control"
    assert manifest["config"]["observed_policy_output_modes"] == ["control"]
    context = json.loads(paths["trace_context"].read_text(encoding="utf-8"))
    assert context["anchor_kind"] == "last_step"


def test_control_trace_entrypoint_requires_confirmation_and_runs_analysis() -> None:
    source = CONTROL_SCRIPT.read_text(encoding="utf-8")

    assert os.access(CONTROL_SCRIPT, os.X_OK)
    assert "CONFIRM_HARDWARE_MOTION" in source
    assert "verify_e52_runtime_bundle.py" in source
    assert "--policy-output-mode control" in source
    assert "--test-log-image-interval-steps" in source
    assert "summarize_e52_control_trace.py" in source
    assert "No steps.jsonl found under ${SESSION_ROOT}" in source
    assert "exit 1" in source
    assert "ssh" not in source.lower()
    assert "rsync" not in source.lower()


def test_act_only_baseline_entrypoint_requires_gohome_and_explicit_bypass() -> None:
    source = ACT_BASELINE_CONTROL_SCRIPT.read_text(encoding="utf-8")

    assert os.access(ACT_BASELINE_CONTROL_SCRIPT, os.X_OK)
    assert "CONFIRM_GO_HOME_DONE" in source
    assert "CONFIRM_HARDWARE_MOTION" in source
    assert "CONFIRM_ACT_ONLY_BASELINE" in source
    assert 'policy["runtime_gates"] = {"enabled": False}' in source
    assert 'policy["report_intent"] = True' in source
    assert 'assist["axis_enabled"] = [True, True, True, True]' in source
    assert "--act-only-baseline" in source
    assert "--policy-output-mode control" in source
    assert "summarize_policy_test_log.py" in source
    assert "automatic gohome" in source
    assert "receiver port ${RECEIVER_PORT} is already listening" in source
    assert "--no-receiver" in source
    assert "ssh" not in source.lower()
    assert "rsync" not in source.lower()


def test_act_only_policy_remote_entrypoint_uses_host_policy_button() -> None:
    source = ACT_BASELINE_POLICY_REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert os.access(ACT_BASELINE_POLICY_REMOTE_SCRIPT, os.X_OK)
    assert "CONFIRM_GO_HOME_BEFORE_POLICY" in source
    assert "CONFIRM_HARDWARE_MOTION" in source
    assert "CONFIRM_ACT_ONLY_BASELINE" in source
    assert 'policy["runtime_gates"] = {"enabled": False}' in source
    assert 'policy["report_intent"] = True' in source
    assert 'assist["axis_enabled"] = [True, True, True, True]' in source
    assert 'teleop.setdefault("policy_remote", {})["start_in_policy"] = False' in source
    assert "--act-only-baseline" in source
    assert 'EXCAVATOR_RECEIVER_INPUT="policy_remote"' in source
    assert 'EXCAVATOR_POLICY_OUTPUT_MODE="control"' in source
    assert "deadzone assist=all-axis" in source
    assert 'POLICY_REMOTE_MAX_STEPS="${E52_POLICY_REMOTE_MAX_STEPS:-50000}"' in source
    assert 'EXCAVATOR_MAX_STEPS="${POLICY_REMOTE_MAX_STEPS}"' in source
    assert "Ignoring generic MAX_STEPS=" in source
    assert "go_home_done" in source
    assert "policy button 4" in source
    assert "slave_real_stack.sh run --force --policy-remote" in source
    assert "ssh" not in source.lower()
    assert "rsync" not in source.lower()
