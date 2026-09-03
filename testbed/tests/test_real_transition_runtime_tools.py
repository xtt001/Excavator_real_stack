from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from scripts.build_real_transition_target_release_bundle import (
    FILES,
    build_bundle,
)
from scripts.summarize_policy_test_log import (
    _compute_metrics,
    _resolve_run_dir,
    _verdict,
)
from scripts.verify_real_transition_target_release_runtime import verify_runtime

from testbed.tasks.home_side_contract import build_rule_ready_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_RESOLVER = REPO_ROOT / "scripts/real_transition_target_release_paths.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bundle(root: Path) -> Path:
    source = root / "source"
    for relative in FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "accepted_model.json":
            path.write_text(
                json.dumps(
                    {
                        "status": "OFFLINE_ACCEPTED_FIELD_CANDIDATE",
                        "checkpoint": "policy_accepted.ckpt",
                        "evidence_boundary": "offline only",
                    }
                ),
                encoding="utf-8",
            )
        elif relative == "contracts/ready_contract.json":
            path.write_text(json.dumps(build_rule_ready_contract()), encoding="utf-8")
        elif relative == "contracts/target_release_contract_v2.json":
            path.write_text(
                json.dumps(
                    {
                        "schema": "real_transition_target_release_contract_v1",
                        "decision_region": {
                            "train_A_endpoint_range_rad": [-0.38, -0.09],
                            "swing_qpos_range_rad": [0.11, 0.39],
                        },
                    }
                ),
                encoding="utf-8",
            )
        elif relative == "resolved_config.yaml":
            path.write_text(
                yaml.safe_dump(
                    {
                        "task": {
                            "camera_names": ["video4", "video5", "video6", "video7"]
                        },
                        "policy": {
                            "low_dim_keys": ["qpos", "real_transition_condition_v1"]
                        },
                    }
                ),
                encoding="utf-8",
            )
        else:
            path.write_bytes(relative.encode("utf-8"))
    (source / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(source / relative)}  {relative}\n" for relative in sorted(FILES)
        ),
        encoding="utf-8",
    )
    return source


def _runtime_config(root: Path, bundle: Path) -> Path:
    script = root / "cycle_script.json"
    script.write_text(
        json.dumps(
            {
                "schema": "act_cycle_script_v1",
                "initial_side": "B",
                "steps": [{"target_side": "A"}],
                "loop": False,
            }
        ),
        encoding="utf-8",
    )
    config = root / "runtime.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "teleop": {
                    "input": "policy_remote",
                    "policy": {
                        "bundle_dir": str(bundle),
                        "ckpt_path": "policy_accepted.ckpt",
                        "camera_names": ["video4", "video5", "video6", "video7"],
                        "temporal_agg": True,
                        "inference_precision": "fp32",
                        "output_mode": "shadow_zero",
                        "action_scale": [1.0, 1.0, 1.0, 1.0],
                        "deadzone_assist": {
                            "enabled": False,
                            "deadzone_positive": [0.661, 0.259, 0.500, 0.408],
                            "deadzone_negative": [0.721, 0.357, 0.500, 0.508],
                        },
                        "reset_policy_on_goal": True,
                        "cycle_planner": {
                            "enabled": True,
                            "script_path": str(script),
                            "loop": False,
                        },
                    },
                    "policy_remote": {
                        "start_in_policy": False,
                        "scripted_cycle": {
                            "enabled": True,
                            "auto_start_after_arm": True,
                            "ready_contract": "contracts/ready_contract.json",
                            "target_region_contract": (
                                "contracts/target_release_contract_v2.json"
                            ),
                            "stop_on_wrong_ready": True,
                            "swing_landing": {
                                "enabled": True,
                                "coast_stop_time_s": 0.50,
                                "edge_margin_rad": 0.03,
                                "p_gain": 0.60,
                                "d_gain": 0.12,
                                "return_confirm_drop_rad": 0.05,
                                "return_min_qvel_rad_s": 0.05,
                                "pd_blend_width_rad": 0.03,
                                "pd_blend_time_s": 0.25,
                                "policy_gain_time_s": 0.25,
                                "min_action_positive": 0.661,
                                "min_action_negative": 0.721,
                                "max_action_positive": 0.72,
                                "max_action_negative": 0.78,
                                "qvel_stable_rad_s": 0.015,
                            },
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return config


def _drive_bundle(
    drive: Path,
    version: str,
    *,
    bundle_name: str = "real_transition_target_release_v2",
) -> Path:
    bundle = (
        drive
        / "Excavator_real_stack_runtime"
        / version
        / "policy_bundles"
        / bundle_name
    )
    bundle.mkdir(parents=True)
    (bundle / "policy_accepted.ckpt").write_bytes(b"accepted")
    return bundle


def _run_path_resolver(
    *,
    repo_root: Path,
    search_roots: list[Path],
    bundle_dir: Path | None = None,
    log_root: Path | None = None,
    runtime_bundle_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("BUNDLE_DIR", None)
    env.pop("LOG_ROOT", None)
    env.pop("REAL_TRANSITION_RUNTIME_BUNDLE_NAME", None)
    env["EXCAVATOR_RUNTIME_SEARCH_ROOTS"] = os.pathsep.join(
        str(path) for path in search_roots
    )
    if bundle_dir is not None:
        env["BUNDLE_DIR"] = str(bundle_dir)
    if log_root is not None:
        env["LOG_ROOT"] = str(log_root)
    if runtime_bundle_name is not None:
        env["REAL_TRANSITION_RUNTIME_BUNDLE_NAME"] = runtime_bundle_name
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                "set -euo pipefail\n"
                'source "$1"\n'
                'real_transition_resolve_runtime_paths "$2"\n'
                'printf "%s\\n" "$REAL_TRANSITION_BUNDLE_DIR" '
                '"$REAL_TRANSITION_LOG_ROOT" '
                '"$REAL_TRANSITION_DRIVE_ROOT" '
                '"$REAL_TRANSITION_BUNDLE_SOURCE"\n'
            ),
            "resolver-test",
            str(PATH_RESOLVER),
            str(repo_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_portable_bundle_and_runtime_preflight(tmp_path: Path) -> None:
    source = _source_bundle(tmp_path)
    output = tmp_path / "real_transition_target_release_v2"

    built = build_bundle(source=source, output=output, overwrite=False)
    config = _runtime_config(tmp_path, output)
    result = verify_runtime(
        config_path=config,
        bundle=output,
        expected_output_mode="shadow_zero",
    )

    assert built["status"] == "PASS"
    assert result["status"] == "PASS"
    assert (output / "policy_accepted.ckpt").is_file()


def test_runtime_preflight_rejects_training_best_checkpoint(tmp_path: Path) -> None:
    source = _source_bundle(tmp_path)
    output = tmp_path / "real_transition_target_release_v2"
    build_bundle(source=source, output=output, overwrite=False)
    config = _runtime_config(tmp_path, output)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["teleop"]["policy"]["ckpt_path"] = "policy_best.ckpt"
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="policy_accepted.ckpt"):
        verify_runtime(
            config_path=config,
            bundle=output,
            expected_output_mode="shadow_zero",
        )


def test_runtime_path_resolver_prefers_unique_external_bundle(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "checkout"
    local_bundle = repo_root / "policy_bundles/real_transition_target_release_v2"
    local_bundle.mkdir(parents=True)
    (local_bundle / "policy_accepted.ckpt").write_bytes(b"local")
    drive = tmp_path / "EXTERNAL_USB"
    external_bundle = _drive_bundle(
        drive, "real_transition_target_release_v2_20260828_test"
    )

    result = _run_path_resolver(repo_root=repo_root, search_roots=[drive])

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(external_bundle.resolve()),
        str((drive / "policy_control_tests").resolve()),
        str(drive.resolve()),
        "external",
    ]


def test_runtime_path_resolver_supports_task_state_v2_bundle_name(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    drive = tmp_path / "EXTERNAL_USB"
    name = "real_transition_task_state_v2_allow2"
    external_bundle = _drive_bundle(
        drive,
        f"{name}_20260902_test",
        bundle_name=name,
    )

    result = _run_path_resolver(
        repo_root=repo_root,
        search_roots=[drive],
        runtime_bundle_name=name,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == str(external_bundle.resolve())


def test_runtime_path_resolver_rejects_multiple_external_bundles(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    drive = tmp_path / "EXTERNAL_USB"
    _drive_bundle(drive, "real_transition_target_release_v2_20260828_a")
    _drive_bundle(drive, "real_transition_target_release_v2_20260828_b")

    result = _run_path_resolver(repo_root=repo_root, search_roots=[drive])

    assert result.returncode == 2
    assert "Multiple target-release runtime bundles" in result.stderr


def test_runtime_path_resolver_allows_explicit_override(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    drive = tmp_path / "EXTERNAL_USB"
    _drive_bundle(drive, "real_transition_target_release_v2_20260828_a")
    _drive_bundle(drive, "real_transition_target_release_v2_20260828_b")
    override = tmp_path / "reviewed_bundle"
    override.mkdir()
    (override / "policy_accepted.ckpt").write_bytes(b"override")
    log_root = tmp_path / "reviewed_logs"

    result = _run_path_resolver(
        repo_root=repo_root,
        search_roots=[drive],
        bundle_dir=override,
        log_root=log_root,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(override.resolve()),
        str(log_root.resolve()),
        "",
        "explicit",
    ]


def test_log_verdict_requires_clean_scripted_cycle_status() -> None:
    steps = [
        {
            "local_step": index,
            "wall_time_ns": 1_000_000_000 + index * 50_000_000,
            "policy_remote_mode": "policy",
            "policy_remote_activated": int(index == 0),
            "policy_output_mode": "shadow_zero",
            "policy_inference_latency_ms": 10.0,
            "policy_action": [0.1, 0.0, 0.0, 0.0],
            "policy_returned_action": [0.0, 0.0, 0.0, 0.0],
            "raw_action": [0.0, 0.0, 0.0, 0.0],
            "safe_action": [0.0, 0.0, 0.0, 0.0],
            "commanded_action": [0.0, 0.0, 0.0, 0.0],
            "receiver_health_ok": 1,
            "controller_ack": 1,
            "scripted_cycle_enabled": 1,
            "scripted_cycle_active": 1,
            "scripted_cycle_fault": "",
            "scripted_cycle_activation_rejected_reason": "",
            "planner_target_side": "A",
        }
        for index in range(3)
    ]
    metrics = _compute_metrics(steps, warmup_steps=0)

    ok, reasons = _verdict(
        summary={"stop_reason": "complete"},
        metrics=metrics,
        expect_output_mode="shadow_zero",
        allow_stop_reasons={"complete"},
        require_shadow_zero=True,
        expect_policy_remote=True,
        expect_scripted_cycle=True,
        min_steps=3,
        max_shadow_command_abs=1e-6,
    )

    assert ok is True
    assert reasons == []


def test_log_verdict_checks_task_state_sequence_and_planner_match() -> None:
    stages = (
        ("work", [1.0, 1.0, 0.0, 0.0, 0.0], 0),
        ("work_complete", [1.0, 1.0, 1.0, 0.0, 0.0], 1),
        ("return_committed", [1.0, 1.0, 1.0, 1.0, -1.0], 1),
    )
    steps = []
    for index, (stage, token, changed) in enumerate(stages):
        steps.append(
            {
                "wall_time_ns": 1_000_000_000 + index * 50_000_000,
                "policy_remote_mode": "policy",
                "policy_remote_activated": int(index == 0),
                "policy_output_mode": "shadow_zero",
                "policy_inference_latency_ms": 1.0,
                "policy_action": [0.0] * 4,
                "policy_task_state_v2": token,
                "planner_task_state_v2": token,
                "scripted_cycle_enabled": 1,
                "scripted_cycle_active": 1,
                "scripted_cycle_fault": "",
                "scripted_cycle_activation_rejected_reason": "",
                "scripted_cycle_task_state_v2_enabled": 1,
                "scripted_cycle_task_state_stage": stage,
                "scripted_cycle_task_state_changed": changed,
                "scripted_cycle_task_state_advance_requested": 0,
                "scripted_cycle_task_state_advance_rejected_reason": "",
                "scripted_cycle_task_auto_progress_enabled": 1,
                "scripted_cycle_task_auto_work_liveness": 1,
                "scripted_cycle_task_auto_bucket_effective_observed": 1,
                "scripted_cycle_task_state_applied_event": (
                    ""
                    if index == 0
                    else ("work_complete" if index == 1 else "return_commit")
                ),
                "planner_target_side": "A",
                "policy_returned_action": [0.0] * 4,
                "raw_action": [0.0] * 4,
                "safe_action": [0.0] * 4,
                "commanded_action": [0.0] * 4,
                "receiver_health_ok": 1,
                "controller_ack": 1,
            }
        )
    metrics = _compute_metrics(steps, warmup_steps=0)

    ok, reasons = _verdict(
        summary={"stop_reason": "complete"},
        metrics=metrics,
        expect_output_mode="shadow_zero",
        allow_stop_reasons={"complete"},
        require_shadow_zero=True,
        expect_policy_remote=True,
        expect_scripted_cycle=True,
        expect_task_state_v2=True,
        min_steps=3,
        max_shadow_command_abs=1e-6,
    )

    assert ok is True
    assert reasons == []

    steps[-1]["policy_task_state_v2"] = [1.0, 1.0, 1.0, 0.0, -1.0]
    invalid_metrics = _compute_metrics(steps, warmup_steps=0)
    invalid_ok, invalid_reasons = _verdict(
        summary={"stop_reason": "complete"},
        metrics=invalid_metrics,
        expect_output_mode="shadow_zero",
        allow_stop_reasons={"complete"},
        require_shadow_zero=True,
        expect_policy_remote=True,
        expect_scripted_cycle=True,
        expect_task_state_v2=True,
        min_steps=3,
        max_shadow_command_abs=1e-6,
    )
    assert invalid_ok is False
    assert any("invalid policy vectors" in reason for reason in invalid_reasons)


def test_stationary_shadow_requires_work_state_and_no_automatic_progress() -> None:
    token = [1.0, 1.0, 0.0, 0.0, 0.0]
    steps = [
        {
            "local_step": index,
            "wall_time_ns": 1_000_000_000 + index * 50_000_000,
            "policy_remote_mode": "policy",
            "policy_remote_activated": int(index == 0),
            "policy_output_mode": "shadow_zero",
            "policy_inference_latency_ms": 1.0,
            "policy_action": [0.2, -0.1, 0.05, 0.1],
            "policy_task_state_v2": token,
            "planner_task_state_v2": token,
            "policy_returned_action": [0.0] * 4,
            "raw_action": [0.0] * 4,
            "safe_action": [0.0] * 4,
            "commanded_action": [0.0] * 4,
            "receiver_health_ok": 1,
            "controller_ack": 1,
            "scripted_cycle_enabled": 1,
            "scripted_cycle_active": 1,
            "scripted_cycle_fault": "",
            "scripted_cycle_activation_rejected_reason": "",
            "scripted_cycle_task_state_v2_enabled": 1,
            "scripted_cycle_task_state_stage": "work",
            "scripted_cycle_task_state_changed": 0,
            "scripted_cycle_task_state_advance_requested": 0,
            "scripted_cycle_task_state_advance_rejected_reason": "",
            "scripted_cycle_task_auto_progress_enabled": 1,
            "scripted_cycle_task_auto_work_liveness": 0,
            "scripted_cycle_task_auto_bucket_effective_observed": 0,
            "scripted_cycle_task_auto_pending_event": "",
            "scripted_cycle_task_state_applied_event": "",
            "planner_target_side": "A",
        }
        for index in range(25)
    ]
    metrics = _compute_metrics(steps, warmup_steps=0)

    ok, reasons = _verdict(
        summary={"stop_reason": "aborted"},
        metrics=metrics,
        expect_output_mode="shadow_zero",
        allow_stop_reasons={"aborted"},
        require_shadow_zero=True,
        expect_policy_remote=True,
        expect_scripted_cycle=True,
        expect_task_state_v2_stationary_shadow=True,
        min_steps=20,
        max_shadow_command_abs=1e-6,
    )

    assert ok is True
    assert reasons == []

    steps[-1]["scripted_cycle_task_auto_work_liveness"] = 1
    bad_metrics = _compute_metrics(steps, warmup_steps=0)
    bad_ok, bad_reasons = _verdict(
        summary={"stop_reason": "aborted"},
        metrics=bad_metrics,
        expect_output_mode="shadow_zero",
        allow_stop_reasons={"aborted"},
        require_shadow_zero=True,
        expect_policy_remote=True,
        expect_scripted_cycle=True,
        expect_task_state_v2_stationary_shadow=True,
        min_steps=20,
        max_shadow_command_abs=1e-6,
    )
    assert bad_ok is False
    assert "stationary shadow falsely confirmed work liveness" in bad_reasons


def test_latest_log_resolution_finds_nested_receiver_run(tmp_path: Path) -> None:
    run = tmp_path / "session" / "receiver_run"
    run.mkdir(parents=True)
    (run / "steps.jsonl").write_text("{}\n", encoding="utf-8")

    assert _resolve_run_dir(None, tmp_path) == run
