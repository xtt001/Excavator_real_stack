from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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

    assert one_dig_cfg["teleop"]["policy"]["bundle_dir"] == "policy_bundles/real_one_dig_v1"
    assert one_dig_cfg["teleop"]["policy"]["camera_names"] == ["fpv"]
    assert one_dig_cfg["task"]["camera_names"] == ["fpv"]
    assert one_dig_cfg["sync"]["required_cameras"] == ["fpv"]
