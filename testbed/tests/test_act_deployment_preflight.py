from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from testbed.cli.package_act_runtime_bundle import package_act_runtime_bundle
from testbed.cli.preflight_short_horizon_contract import (
    preflight_short_horizon_contract,
)
from testbed.policies.act.deployment_preflight import (
    BUNDLE_MANIFEST_FILENAME,
    verify_shadow_deployment,
)

ROOT = Path(__file__).resolve().parents[2]
N5_SHADOW_CONFIG = (
    ROOT
    / "testbed/testbed/configs/policy_real_gmsl_fourcam_g49_n5_shadow_v1.yaml"
)
FIELD_TEMPLATE = (
    ROOT / "testbed/testbed/configs/short_horizon_g49_n5_field_adapter_v1.yaml"
)


def _resolved() -> dict:
    return {
        "experiment_contract": {
            "policy_action_scale": [1.0, 1.0, 1.0, 1.0],
        },
        "task": {
            "task_name": "n5-test",
            "camera_names": ["video4", "video5", "video6", "video7"],
            "backend": {"dt": 0.05},
        },
        "policy": {
            "low_dim_keys": ["qpos"],
            "act_params": {
                "camera_role_encoding": {
                    "enabled": True,
                    "roles": {
                        "video4": "eye",
                        "video5": "eye",
                        "video6": "stick",
                        "video7": "stick",
                    },
                }
            },
        },
    }


def _source_bundle(path: Path) -> Path:
    source = path / "source"
    source.mkdir()
    (source / "policy_best.ckpt").write_bytes(b"checkpoint")
    (source / "dataset_stats.pkl").write_bytes(b"stats")
    (source / "resolved_config.yaml").write_text(
        yaml.safe_dump(_resolved(), sort_keys=False),
        encoding="utf-8",
    )
    (source / "run_metadata.json").write_text(
        json.dumps({"status": "completed"}) + "\n",
        encoding="utf-8",
    )
    return source


def _runtime_config(path: Path, bundle: Path) -> Path:
    payload = yaml.safe_load(N5_SHADOW_CONFIG.read_text(encoding="utf-8"))
    payload["teleop"]["policy"]["bundle_dir"] = str(bundle)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_package_and_verify_n5_shadow_bundle(tmp_path: Path) -> None:
    source = _source_bundle(tmp_path)
    bundle = tmp_path / "bundle"
    package = package_act_runtime_bundle(
        source_dir=source,
        output_dir=bundle,
        candidate_id="g49-n5-test",
    )
    manifest = json.loads(
        (bundle / BUNDLE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    config = _runtime_config(tmp_path / "runtime.yaml", bundle)

    report = verify_shadow_deployment(
        config_path=config,
        bundle_dir=bundle,
        manifest=manifest,
    )

    assert package["ok"] is True
    assert report["ok"] is True
    assert report["candidate_id"] == "g49-n5-test"
    assert report["policy_sampling_hz"] == 20.0
    assert report["control_pump_hz"] == 50.0
    assert report["action_scale"] == [1.0, 1.0, 1.0, 1.0]
    assert set(report["verified_bundle_hashes"]) == {
        "policy_best.ckpt",
        "dataset_stats.pkl",
        "resolved_config.yaml",
        "run_metadata.json",
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda cfg: cfg["task"].update({"record_hz": 50, "dt": 0.02}),
            "training time base",
        ),
        (
            lambda cfg: cfg["teleop"]["policy"].update(
                {"action_scale": [1.0, 1.0, 1.0, 0.75]}
            ),
            "evaluated action domain",
        ),
        (
            lambda cfg: cfg["teleop"]["policy"].update({"output_mode": "control"}),
            "shadow_zero",
        ),
    ],
)
def test_shadow_preflight_rejects_runtime_semantic_drift(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    source = _source_bundle(tmp_path)
    bundle = tmp_path / "bundle"
    package_act_runtime_bundle(
        source_dir=source,
        output_dir=bundle,
        candidate_id="g49-n5-test",
    )
    config_payload = yaml.safe_load(N5_SHADOW_CONFIG.read_text(encoding="utf-8"))
    config_payload["teleop"]["policy"]["bundle_dir"] = str(bundle)
    mutate(config_payload)
    config = tmp_path / "runtime.yaml"
    config.write_text(
        yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8"
    )
    manifest = json.loads(
        (bundle / BUNDLE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )

    with pytest.raises(ValueError, match=message):
        verify_shadow_deployment(
            config_path=config,
            bundle_dir=bundle,
            manifest=manifest,
        )


def test_field_adapter_template_fails_closed_until_values_are_supplied(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path)
    bundle = tmp_path / "bundle"
    package_act_runtime_bundle(
        source_dir=source,
        output_dir=bundle,
        candidate_id="g49-n5-test",
    )

    with pytest.raises((TypeError, ValueError)):
        preflight_short_horizon_contract(
            config_path=FIELD_TEMPLATE,
            bundle_dir=bundle,
        )


def test_field_adapter_contract_resolves_bundle_hashes_after_review(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path)
    bundle = tmp_path / "bundle"
    package_act_runtime_bundle(
        source_dir=source,
        output_dir=bundle,
        candidate_id="g49-n5-test",
    )
    payload = copy.deepcopy(
        yaml.safe_load(FIELD_TEMPLATE.read_text(encoding="utf-8"))
    )
    payload.update(
        {
            "test_id": "field-reviewed-test",
            "max_observation_gap_ms": 100.0,
            "max_camera_age_ms": 100.0,
            "command_abs_limit": [0.1, 0.1, 0.1, 0.1],
            "command_delta_limit": [0.02, 0.02, 0.02, 0.02],
            "qvel_abort_limit": [0.1, 0.1, 0.1, 0.1],
            "qpos_lower_limit": [-1.0, -1.0, -1.0, -1.0],
            "qpos_upper_limit": [1.0, 1.0, 1.0, 1.0],
            "allowed_direction_mask": [
                [True, True],
                [True, True],
                [True, True],
                [True, True],
            ],
        }
    )
    config = tmp_path / "field.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    resolved = preflight_short_horizon_contract(
        config_path=config,
        bundle_dir=bundle,
    )

    assert resolved["test_id"] == "field-reviewed-test"
    assert len(resolved["checkpoint_sha256"]) == 64
    assert len(resolved["resolved_config_sha256"]) == 64
