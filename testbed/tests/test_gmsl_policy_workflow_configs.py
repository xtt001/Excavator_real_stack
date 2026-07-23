from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "testbed" / "testbed" / "configs"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_policy_shadow_check_defaults_to_gmsl_bundle_preflight() -> None:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["TEST_LOG_DIR"] = "/tmp/excavator-policy-shadow-test-unused"
    result = subprocess.run(
        ["bash", "scripts/run_policy_shadow_check.sh"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 2
    assert "Bundle: policy_bundles/real_gmsl_four_camera_v1" in result.stdout
    assert "MISSING policy_best.ckpt" in result.stdout
    assert "MISSING dataset_stats.pkl" in result.stdout
    assert "MISSING resolved_config.yaml" in result.stdout
    assert "policy_bundles/real_one_dig_v1" not in result.stdout


def test_gmsl_training_configs_define_qpos_and_qpos_qvel_variants() -> None:
    qpos_cfg = _load_yaml(CONFIG_DIR / "act_real_gmsl_four_camera_qpos_v1.yaml")
    qpos_qvel_cfg = _load_yaml(CONFIG_DIR / "act_real_gmsl_four_camera_qpos_qvel_v1.yaml")

    expected_cameras = ["video4", "video5", "video6", "video7"]
    for cfg in (qpos_cfg, qpos_qvel_cfg):
        assert cfg["task"]["camera_names"] == expected_cameras
        assert cfg["task"]["dataset_dir"] == "/media/mundane/EXTERNAL_USB/real_gmsl_four_camera_20hz_v1"
        assert (
            cfg["task"]["train_ready_manifest_path"]
            == "/media/mundane/EXTERNAL_USB/real_gmsl_four_camera_20hz_v1/qc_full/train_ready_manifest.json"
        )
        assert cfg["train"]["ckpt_dir"].startswith("runs/ckpts/")
        assert cfg["train"]["device"] == "cuda"

    assert qpos_cfg["policy"]["low_dim_keys"] == ["qpos"]
    assert qpos_cfg["policy"]["act_params"]["state_dim"] == 4
    assert qpos_qvel_cfg["policy"]["low_dim_keys"] == ["qpos", "qvel"]
    assert qpos_qvel_cfg["policy"]["act_params"]["state_dim"] == 8


def test_policy_bundle_preflight_rejects_wrong_camera_contract(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("policy_best.ckpt", "dataset_stats.pkl"):
        (bundle / name).write_bytes(b"placeholder")
    (bundle / "resolved_config.yaml").write_text(
        yaml.safe_dump({"task": {"camera_names": ["fpv"]}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_policy_test_log.py",
            "--bundle-dir",
            str(bundle),
            "--expect-camera-names",
            "video4,video5,video6,video7",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 2
    assert "MISMATCH camera_names" in result.stdout
    assert "expected=['video4', 'video5', 'video6', 'video7']" in result.stdout
    assert "actual=['fpv']" in result.stdout


def test_policy_runtime_configs_keep_gmsl_and_legacy_fpv_contracts_separate() -> None:
    gmsl_cfg = _load_yaml(CONFIG_DIR / "policy_real_gmsl_four_camera_v1.yaml")
    one_dig_cfg = _load_yaml(CONFIG_DIR / "policy_real_one_dig_v1.yaml")

    expected_gmsl = ["video4", "video5", "video6", "video7"]
    assert gmsl_cfg["teleop"]["policy"]["bundle_dir"] == "policy_bundles/real_gmsl_four_camera_v1"
    assert gmsl_cfg["teleop"]["policy"]["camera_names"] == expected_gmsl
    assert gmsl_cfg["task"]["camera_names"] == expected_gmsl
    assert gmsl_cfg["sync"]["required_cameras"] == expected_gmsl
    assert gmsl_cfg["receiver"]["online_qc"]["primary_camera"] == "video4"
    assert gmsl_cfg["safety"]["action_clip"] == 1.0
    assert gmsl_cfg["safety"]["max_delta_per_step"] == 1.0

    assert one_dig_cfg["teleop"]["policy"]["bundle_dir"] == "policy_bundles/real_one_dig_v1"
    assert one_dig_cfg["teleop"]["policy"]["camera_names"] == ["fpv"]
    assert one_dig_cfg["task"]["camera_names"] == ["fpv"]
    assert one_dig_cfg["sync"]["required_cameras"] == ["fpv"]
    assert one_dig_cfg["receiver"]["online_qc"]["primary_camera"] == "fpv"


def test_g49_n5_runtime_configs_enable_strict_fp32_compiled_inference() -> None:
    for name in (
        "policy_real_gmsl_fourcam_g49_n5_control_v1.yaml",
        "policy_real_gmsl_fourcam_g49_n5_shadow_v1.yaml",
    ):
        cfg = _load_yaml(CONFIG_DIR / name)
        policy = cfg["teleop"]["policy"]
        assert policy["inference_precision"] == "fp32"
        assert policy["inference_compile"] is True
        assert policy["inference_compile_mode"] == "reduce-overhead"
        assert policy["inference_compile_dynamic"] is False
        assert policy["device_uint8_preprocess"] is True
        assert policy["inference_warmup_steps"] == 3
        assert policy["frame_alignment"]["enabled"] is True
        assert policy["frame_alignment"]["wait_timeout_ms"] == 200
        # A second immediate producer can outrun the 50 Hz low-level consumer.
        assert cfg["real"]["control_pump"]["send_immediately_on_update"] is False


def test_teleop_recording_config_declares_gmsl_online_qc_primary_camera() -> None:
    cfg = _load_yaml(CONFIG_DIR / "teleop_real_v1.yaml")

    expected_gmsl = ["video4", "video5", "video6", "video7"]
    assert cfg["task"]["camera_names"] == expected_gmsl
    assert cfg["receiver"]["health"]["primary_camera"] == "video4"
    assert cfg["receiver"]["online_qc"]["primary_camera"] == "video4"


def test_pro_real_teleop_preserves_axis_signs_and_shapes_hydraulic_response() -> None:
    cfg = _load_yaml(CONFIG_DIR / "teleop_real_v1.yaml")
    joystick = cfg["teleop"]["joystick"]

    assert joystick["joystick_ids"] == [0, 1, 0, 1]
    assert joystick["axis_map"] == [1, 1, 0, 0]
    assert joystick["invert"] == [True, False, True, False]
    assert joystick["deadzone"] == 0.05
    assert joystick["scale"] == [0.8, 0.8, 0.8, 0.7]
    assert joystick["response_profile"] == {
        "enabled": True,
        "use_measured_dt": False,
        "deadzone": [0.0, 0.0, 0.0, 0.0],
        "attack_rate": [2.0, 1.5, 1.5, 1.5],
        "release_rate": 6.0,
        "recenter_rate": 7.0,
        "exponent": 1.0,
        "positive_exponent": [0.22, 0.74, 0.52, 0.42],
        "negative_exponent": [0.22, 0.35, 0.52, 0.25],
    }
    assert cfg["teleop"]["learning_target"] == "operator_command_from_observation"
    assert cfg["task"]["dataset_dir"] == "data/real_teleop_v1"

    # The paired scale/exponent change must preserve the field-measured
    # hydraulic onset at about 30% physical stick travel. This avoids trading
    # lower top speed for a larger no-motion region.
    remapped_30_percent = (0.30 - joystick["deadzone"]) / (
        1.0 - joystick["deadzone"]
    )
    positive_onset = [
        scale * remapped_30_percent**exponent
        for scale, exponent in zip(
            joystick["scale"],
            joystick["response_profile"]["positive_exponent"],
        )
    ]
    negative_onset = [
        scale * remapped_30_percent**exponent
        for scale, exponent in zip(
            joystick["scale"],
            joystick["response_profile"]["negative_exponent"],
        )
    ]
    assert positive_onset == pytest.approx([0.6, 0.3, 0.4, 0.4], abs=0.005)
    assert negative_onset == pytest.approx([0.6, 0.5, 0.4, 0.5], abs=0.005)
