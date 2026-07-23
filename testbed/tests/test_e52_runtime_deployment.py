from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from scripts.verify_e52_runtime_bundle import verify_e52_runtime_bundle

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "testbed/testbed/configs/policy_real_gmsl_four_camera_v1.yaml"
E52_CONFIG = ROOT / "testbed/testbed/configs/policy_real_gmsl_eye2_e52_v1.yaml"
SHADOW_SCRIPT = ROOT / "scripts/run_e52_policy_shadow_check.sh"


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_e52_config_is_mechanical_shadow_only_eye2_derivative() -> None:
    base = _read_yaml(BASE_CONFIG)
    actual = _read_yaml(E52_CONFIG)
    expected = copy.deepcopy(base)

    expected["real"]["data_side_defaults"]["host"]["dataset_dir"] = (
        "data/policy_shadow_real_gmsl_eye2_e52_v1"
    )
    expected["real"]["data_side_defaults"]["slave"]["dataset_dir"] = (
        "/data/policy_shadow_real_gmsl_eye2_e52_v1"
    )
    expected["teleop"]["learning_target"] = (
        "e52_policy_shadow_from_real_gmsl_observation"
    )
    expected["teleop"]["metadata"]["notes"] = (
        "E52 eye-only GMSL gated policy shadow runtime"
    )
    policy = expected["teleop"]["policy"]
    policy["bundle_dir"] = "policy_bundles/real_gmsl_eye2_e52_v1"
    policy["source_id"] = "policy:act:real_gmsl_eye2_e52_v1"
    policy["camera_names"] = ["video4", "video5"]
    policy["qvel_mode"] = "raw"
    policy["action_scale"] = [1.0, 1.0, 1.0, 1.0]
    policy["deadzone_assist"]["deadzone_positive"] = [
        0.661,
        0.259,
        0.500,
        0.408,
    ]
    policy["deadzone_assist"]["deadzone_negative"] = [
        0.721,
        0.357,
        0.500,
        0.508,
    ]
    policy["runtime_gates"] = {
        "enabled": True,
        "bundle_dir": "policy_bundles/real_gmsl_eye2_e52_v1",
        "manifest_path": (
            "policy_bundles/real_gmsl_eye2_e52_v1/candidate_package_manifest.json"
        ),
        "deadzone_json": "deadzone_policy_raw_for_runtime_scale.json",
        "snap_epsilon": 0.001,
    }
    expected["teleop"]["recording"]["go_home"]["success_tolerance_rad"] = [
        0.05,
        0.05,
        0.035,
        0.04,
    ]
    expected["teleop"]["test_log"]["output_dir"] = (
        "runs/policy_control_tests/e52_runtime_shadow"
    )
    expected["task"]["task_name"] = "real_gmsl_eye2_e52_policy_shadow_test"
    expected["task"]["dataset_dir"] = "data/policy_shadow_real_gmsl_eye2_e52_v1"
    expected["task"]["camera_names"] = ["video4", "video5"]
    expected["task"]["param_version"] = "real_gmsl_eye2_e52_v1"
    expected["sync"]["required_cameras"] = ["video4", "video5"]

    assert actual == expected
    assert actual["teleop"]["policy"]["output_mode"] == "shadow_zero"
    assert actual["teleop"]["policy"]["deadzone_assist"]["enabled"] is False
    assert actual["teleop"]["policy"]["action_scale"] == [1.0, 1.0, 1.0, 1.0]
    assert actual["teleop"]["recording"]["go_home"][
        "success_tolerance_rad"
    ] == [0.05, 0.05, 0.035, 0.04]


def test_verify_e52_runtime_bundle_checks_base_contract_and_loads_gate_owner(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "policy_best.ckpt").touch()
    (bundle / "dataset_stats.pkl").touch()
    (bundle / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "task": {"camera_names": ["video4", "video5"]},
                "policy": {"low_dim_keys": ["qpos"]},
            }
        ),
        encoding="utf-8",
    )
    (bundle / "run_metadata.json").write_text("{}\n", encoding="utf-8")
    (bundle / "deadzone_policy_raw_for_runtime_scale.json").write_text(
        "{}\n", encoding="utf-8"
    )
    artifact_names = (
        "phase_gate_model",
        "tail_candidate_model",
        "gohome_eligibility_model",
        "temporal_direction_model",
        "temporal_direction_metadata",
    )
    for name in artifact_names:
        (bundle / f"{name}.pt").touch()
    manifest = bundle / "candidate_package_manifest.json"
    base_artifacts = {
        "action_policy_best": bundle / "policy_best.ckpt",
        "action_dataset_stats": bundle / "dataset_stats.pkl",
        "action_resolved_config": bundle / "resolved_config.yaml",
        "action_run_metadata": bundle / "run_metadata.json",
    }
    all_artifacts = {
        **base_artifacts,
        **{name: bundle / f"{name}.pt" for name in artifact_names},
    }
    manifest.write_text(
        json.dumps(
            {
                "candidate_id": "E52-test",
                "artifacts": [
                    {
                        "name": name,
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in all_artifacts.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    runtime_config = {
        "enabled": True,
        "bundle_dir": str(bundle),
        "manifest_path": str(manifest),
        "deadzone_json": "deadzone_policy_raw_for_runtime_scale.json",
        "snap_epsilon": 0.001,
    }
    config_path.write_text(
        yaml.safe_dump(
            {
                "teleop": {
                    "policy": {
                        "bundle_dir": str(bundle),
                        "camera_names": ["video4", "video5"],
                        "output_mode": "shadow_zero",
                        "qvel_mode": "raw",
                        "action_scale": [1.0, 1.0, 1.0, 1.0],
                        "deadzone_assist": {"enabled": False},
                        "runtime_gates": runtime_config,
                    },
                    "recording": {
                        "go_home": {
                            "success_tolerance_rad": [0.05, 0.05, 0.035, 0.04]
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "scripts.verify_e52_runtime_bundle.RuntimeGateStack.from_config",
        return_value=SimpleNamespace(stack_id="E52-test"),
    ) as load_gate_stack:
        report = verify_e52_runtime_bundle(
            config_path=config_path,
            bundle_dir=bundle,
        )

    assert report["ok"] is True
    assert report["candidate_id"] == "E52-test"
    assert report["camera_names"] == ["video4", "video5"]
    assert report["low_dim_keys"] == ["qpos"]
    assert report["action_scale"] == [1.0, 1.0, 1.0, 1.0]
    assert report["go_home_success_tolerance_rad"] == [
        0.05,
        0.05,
        0.035,
        0.04,
    ]
    assert report["runtime_artifact_hashes_verified"] == list(all_artifacts)
    expected_owner_config = dict(runtime_config)
    expected_owner_config["artifacts"] = {
        name: str((bundle / f"{name}.pt").resolve()) for name in artifact_names
    }
    load_gate_stack.assert_called_once_with(
        expected_owner_config,
        default_bundle_dir=bundle,
    )

    baseline_config = _read_yaml(config_path)
    baseline_config["teleop"]["policy"]["runtime_gates"] = {"enabled": False}
    baseline_config["teleop"]["policy"]["report_intent"] = True
    baseline_config["teleop"]["policy"]["deadzone_assist"] = {
        "enabled": True,
        "axis_enabled": [True, True, True, True],
        "trigger_fraction": [0.36, 0.50, 0.50, 0.375],
        "min_consecutive_steps": 2,
        "margin": [0.02, 0.02, 0.02, 0.02],
        "deadzone_positive": [0.661, 0.259, 0.500, 0.408],
        "deadzone_negative": [0.721, 0.357, 0.500, 0.508],
    }
    config_path.write_text(
        yaml.safe_dump(baseline_config),
        encoding="utf-8",
    )
    with patch(
        "scripts.verify_e52_runtime_bundle.RuntimeGateStack.from_config"
    ) as load_gate_stack:
        baseline_report = verify_e52_runtime_bundle(
            config_path=config_path,
            bundle_dir=bundle,
            require_runtime_gates=False,
        )

    assert baseline_report["ok"] is True
    assert baseline_report["profile"] == "act_only_baseline"
    assert baseline_report["runtime_gate_stack_loaded"] is False
    assert baseline_report["intent_reporting_enabled"] is True
    assert baseline_report["deadzone_assist_enabled"] is True
    assert baseline_report["deadzone_assist_axis_enabled"] == [
        True,
        True,
        True,
        True,
    ]
    assert baseline_report["runtime_artifact_hashes_verified"] == list(
        base_artifacts
    )
    load_gate_stack.assert_not_called()


def test_e52_shadow_entrypoint_is_executable_and_enforces_no_motion_verifier() -> None:
    source = SHADOW_SCRIPT.read_text(encoding="utf-8")

    assert os.access(SHADOW_SCRIPT, os.X_OK)
    assert "policy_real_gmsl_eye2_e52_v1.yaml" in source
    assert "verify_e52_runtime_bundle.py" in source
    assert "-m testbed.cli.record_real" in source
    assert "--policy-output-mode shadow_zero" in source
    assert "--no-record" in source
    assert "e53_verify_no_motion_policy_log.py" in source
    assert "--require-runtime-gate-diagnostics" in source
    assert "CONFIRM_GO_HOME_DONE" in source
    assert "ssh" not in source.lower()
    assert "rsync" not in source.lower()
