from __future__ import annotations

import hashlib
import json
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
                            "ready_contract": "contracts/ready_contract.json",
                            "target_region_contract": (
                                "contracts/target_release_contract_v2.json"
                            ),
                            "stop_on_wrong_ready": True,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return config


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


def test_latest_log_resolution_finds_nested_receiver_run(tmp_path: Path) -> None:
    run = tmp_path / "session" / "receiver_run"
    run.mkdir(parents=True)
    (run / "steps.jsonl").write_text("{}\n", encoding="utf-8")

    assert _resolve_run_dir(None, tmp_path) == run
