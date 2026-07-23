#!/usr/bin/env python3
"""Preflight the portable E52 ACT and runtime-gate bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from testbed.policies.runtime_gate_stack import RuntimeGateStack


EXPECTED_CAMERA_NAMES = ["video4", "video5"]
EXPECTED_LOW_DIM_KEYS = ["qpos"]
EXPECTED_ACTION_SCALE = [1.0, 1.0, 1.0, 1.0]
EXPECTED_GO_HOME_SUCCESS_TOLERANCE = [0.05, 0.05, 0.035, 0.04]
EXPECTED_DEADZONE_POSITIVE = [0.661, 0.259, 0.500, 0.408]
EXPECTED_DEADZONE_NEGATIVE = [0.721, 0.357, 0.500, 0.508]
EXPECTED_ASSIST_AXIS_ENABLED = [True, True, True, True]
EXPECTED_ASSIST_TRIGGER_FRACTION = [0.36, 0.50, 0.50, 0.375]
EXPECTED_ASSIST_MARGIN = [0.02, 0.02, 0.02, 0.02]
REQUIRED_ACT_FILES = (
    "policy_best.ckpt",
    "dataset_stats.pkl",
    "resolved_config.yaml",
    "run_metadata.json",
)
REQUIRED_GATE_ARTIFACTS = (
    "phase_gate_model",
    "tail_candidate_model",
    "gohome_eligibility_model",
    "temporal_direction_model",
    "temporal_direction_metadata",
)
REQUIRED_RUNTIME_ARTIFACT_FILES = {
    "action_policy_best": "policy_best.ckpt",
    "action_dataset_stats": "dataset_stats.pkl",
    "action_resolved_config": "resolved_config.yaml",
    "action_run_metadata": "run_metadata.json",
    **{name: None for name in REQUIRED_GATE_ARTIFACTS},
}


def verify_e52_runtime_bundle(
    *,
    config_path: Path,
    bundle_dir: Path,
    require_runtime_gates: bool = True,
) -> dict[str, Any]:
    config = _read_yaml(config_path)
    policy_config = dict(config.get("teleop", {}).get("policy", {}) or {})
    configured_bundle = Path(str(policy_config.get("bundle_dir", ""))).expanduser()
    if configured_bundle.resolve() != bundle_dir.expanduser().resolve():
        raise ValueError(
            "config policy bundle_dir does not match preflight bundle: "
            f"{configured_bundle} != {bundle_dir}"
        )
    if str(policy_config.get("output_mode", "")) != "shadow_zero":
        raise ValueError("E52 runtime config output_mode must be shadow_zero")
    if list(policy_config.get("camera_names", [])) != EXPECTED_CAMERA_NAMES:
        raise ValueError(
            f"E52 runtime config camera_names must be {EXPECTED_CAMERA_NAMES!r}"
        )
    if str(policy_config.get("qvel_mode", "")) != "raw":
        raise ValueError("E52 runtime config qvel_mode must be raw")
    if list(policy_config.get("action_scale", [])) != EXPECTED_ACTION_SCALE:
        raise ValueError(
            "E52 policy action_scale must preserve the ACT normalized-command "
            f"domain: {EXPECTED_ACTION_SCALE!r}"
        )
    assist_config = dict(policy_config.get("deadzone_assist", {}) or {})
    assist_enabled = bool(assist_config.get("enabled", False))
    if require_runtime_gates and assist_enabled:
        raise ValueError("gated E52 runtime config deadzone_assist must remain disabled")
    if not require_runtime_gates:
        expected_assist = {
            "axis_enabled": EXPECTED_ASSIST_AXIS_ENABLED,
            "trigger_fraction": EXPECTED_ASSIST_TRIGGER_FRACTION,
            "min_consecutive_steps": 2,
            "margin": EXPECTED_ASSIST_MARGIN,
            "deadzone_positive": EXPECTED_DEADZONE_POSITIVE,
            "deadzone_negative": EXPECTED_DEADZONE_NEGATIVE,
        }
        if not assist_enabled:
            raise ValueError("E52 ACT-only baseline requires all-axis deadzone assist")
        for key, expected in expected_assist.items():
            if assist_config.get(key) != expected:
                raise ValueError(
                    f"E52 ACT-only deadzone_assist.{key} must be {expected!r}"
                )
    go_home_config = dict(
        config.get("teleop", {}).get("recording", {}).get("go_home", {}) or {}
    )
    if list(go_home_config.get("success_tolerance_rad", [])) != (
        EXPECTED_GO_HOME_SUCCESS_TOLERANCE
    ):
        raise ValueError(
            "E52 go-home success_tolerance_rad must keep the stick inside "
            f"training-home support: {EXPECTED_GO_HOME_SUCCESS_TOLERANCE!r}"
        )

    missing = [
        bundle_dir / name
        for name in REQUIRED_ACT_FILES
        if not (bundle_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing E52 ACT bundle file(s): "
            + ", ".join(str(path) for path in missing)
        )

    resolved = _read_yaml(bundle_dir / "resolved_config.yaml")
    resolved_cameras = list(resolved.get("task", {}).get("camera_names", []))
    low_dim_keys = list(resolved.get("policy", {}).get("low_dim_keys", []))
    if resolved_cameras != EXPECTED_CAMERA_NAMES:
        raise ValueError(
            "resolved_config camera_names mismatch: "
            f"{resolved_cameras!r} != {EXPECTED_CAMERA_NAMES!r}"
        )
    if low_dim_keys != EXPECTED_LOW_DIM_KEYS:
        raise ValueError(
            "resolved_config low_dim_keys mismatch: "
            f"{low_dim_keys!r} != {EXPECTED_LOW_DIM_KEYS!r}"
        )

    runtime_config = dict(policy_config.get("runtime_gates", {}) or {})
    runtime_gates_enabled = bool(runtime_config.get("enabled", False))
    if not require_runtime_gates:
        if runtime_gates_enabled:
            raise ValueError("E52 ACT-only baseline requires runtime_gates.enabled=false")
        if not bool(policy_config.get("report_intent", False)):
            raise ValueError("E52 ACT-only baseline requires report_intent=true")
        manifest_path = bundle_dir / "candidate_package_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_artifacts = {
            str(item.get("name", "")): item
            for item in manifest.get("artifacts", [])
            if isinstance(item, dict)
        }
        act_artifacts = {
            name: filename
            for name, filename in REQUIRED_RUNTIME_ARTIFACT_FILES.items()
            if filename is not None
        }
        missing_hashes = [
            name
            for name in act_artifacts
            if not str(manifest_artifacts.get(name, {}).get("sha256", ""))
        ]
        if missing_hashes:
            raise ValueError(
                "candidate manifest missing ACT artifact sha256: "
                + ", ".join(missing_hashes)
            )
        hash_mismatches = [
            name
            for name, filename in act_artifacts.items()
            if _sha256(bundle_dir / filename)
            != str(manifest_artifacts[name]["sha256"])
        ]
        if hash_mismatches:
            raise ValueError(
                "E52 ACT artifact sha256 mismatch: " + ", ".join(hash_mismatches)
            )
        return {
            "ok": True,
            "bundle_dir": str(bundle_dir),
            "config_path": str(config_path),
            "candidate_id": str(manifest.get("candidate_id", "")),
            "profile": "act_only_baseline",
            "camera_names": resolved_cameras,
            "low_dim_keys": low_dim_keys,
            "output_mode": "shadow_zero",
            "qvel_mode": "raw",
            "action_scale": EXPECTED_ACTION_SCALE,
            "deadzone_assist_enabled": True,
            "deadzone_assist_axis_enabled": EXPECTED_ASSIST_AXIS_ENABLED,
            "go_home_success_tolerance_rad": EXPECTED_GO_HOME_SUCCESS_TOLERANCE,
            "required_act_files": list(REQUIRED_ACT_FILES),
            "runtime_gate_stack_loaded": False,
            "intent_reporting_enabled": True,
            "runtime_artifact_hashes_verified": list(act_artifacts),
        }
    if not runtime_gates_enabled:
        raise ValueError("E52 runtime_gates must be enabled")
    runtime_bundle = Path(
        str(runtime_config.get("bundle_dir", bundle_dir))
    ).expanduser()
    if runtime_bundle.resolve() != bundle_dir.expanduser().resolve():
        raise ValueError("runtime_gates.bundle_dir must match policy.bundle_dir")
    manifest_path = Path(
        str(
            runtime_config.get(
                "manifest_path",
                bundle_dir / "candidate_package_manifest.json",
            )
        )
    ).expanduser()
    expected_manifest = bundle_dir / "candidate_package_manifest.json"
    if manifest_path.resolve() != expected_manifest.resolve():
        raise ValueError(
            "runtime_gates.manifest_path must select candidate_package_manifest.json "
            "inside the E52 bundle"
        )
    if (
        runtime_config.get("deadzone_json")
        != "deadzone_policy_raw_for_runtime_scale.json"
    ):
        raise ValueError(
            "runtime_gates.deadzone_json must be the relative evaluated deadzone filename"
        )
    if float(runtime_config.get("snap_epsilon", -1.0)) != 0.001:
        raise ValueError("runtime_gates.snap_epsilon must be 0.001")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_artifacts = {
        str(item.get("name", "")): item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict)
    }
    missing_hashes = [
        name
        for name in REQUIRED_RUNTIME_ARTIFACT_FILES
        if not str(manifest_artifacts.get(name, {}).get("sha256", ""))
    ]
    if missing_hashes:
        raise ValueError(
            "candidate manifest missing gate artifact sha256: "
            + ", ".join(missing_hashes)
        )
    local_gate_artifacts = {
        name: bundle_dir / Path(str(manifest_artifacts[name]["path"])).name
        for name in REQUIRED_GATE_ARTIFACTS
    }
    missing_local_artifacts = [
        path for path in local_gate_artifacts.values() if not path.is_file()
    ]
    if missing_local_artifacts:
        raise FileNotFoundError(
            "missing portable E52 gate artifact(s): "
            + ", ".join(str(path) for path in missing_local_artifacts)
        )
    local_runtime_artifacts = {
        name: (
            bundle_dir / filename
            if filename is not None
            else local_gate_artifacts[name]
        )
        for name, filename in REQUIRED_RUNTIME_ARTIFACT_FILES.items()
    }
    hash_mismatches = [
        name
        for name, path in local_runtime_artifacts.items()
        if _sha256(path) != str(manifest_artifacts[name]["sha256"])
    ]
    if hash_mismatches:
        raise ValueError(
            "E52 runtime artifact sha256 mismatch: " + ", ".join(hash_mismatches)
        )
    owner_config = dict(runtime_config)
    owner_config["artifacts"] = {
        **dict(runtime_config.get("artifacts", {}) or {}),
        **{name: str(path.resolve()) for name, path in local_gate_artifacts.items()},
    }
    gate_stack = RuntimeGateStack.from_config(
        owner_config,
        default_bundle_dir=bundle_dir,
    )

    return {
        "ok": True,
        "bundle_dir": str(bundle_dir),
        "config_path": str(config_path),
        "candidate_id": gate_stack.stack_id,
        "camera_names": resolved_cameras,
        "low_dim_keys": low_dim_keys,
        "output_mode": "shadow_zero",
        "qvel_mode": "raw",
        "action_scale": EXPECTED_ACTION_SCALE,
        "deadzone_assist_enabled": False,
        "go_home_success_tolerance_rad": EXPECTED_GO_HOME_SUCCESS_TOLERANCE,
        "required_act_files": list(REQUIRED_ACT_FILES),
        "runtime_gate_stack_loaded": True,
        "runtime_artifact_hashes_verified": list(REQUIRED_RUNTIME_ARTIFACT_FILES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--act-only-baseline",
        action="store_true",
        help="Verify the E52 ACT bundle with runtime gates explicitly disabled.",
    )
    args = parser.parse_args()
    try:
        report = verify_e52_runtime_bundle(
            config_path=args.config,
            bundle_dir=args.bundle_dir,
            require_runtime_gates=not args.act_only_baseline,
        )
    except Exception as exc:
        report = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(1) from exc
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
