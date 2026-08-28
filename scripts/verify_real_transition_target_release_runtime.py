#!/usr/bin/env python3
"""Fail-closed preflight for the planner-conditioned ACT field runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from testbed.tasks.act_cycle_planner import ScriptCyclePlanner
from testbed.tasks.home_side_contract import validate_rule_ready_contract


CAMERAS = ["video4", "video5", "video6", "video7"]
EXPECTED_DEADZONE_POSITIVE = [0.661, 0.259, 0.500, 0.408]
EXPECTED_DEADZONE_NEGATIVE = [0.721, 0.357, 0.500, 0.508]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--expect-output-mode", choices=("shadow_zero", "control"), required=True
    )
    args = parser.parse_args()
    report = verify_runtime(
        config_path=args.config.resolve(),
        bundle=args.bundle_dir.resolve(),
        expected_output_mode=str(args.expect_output_mode),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def verify_runtime(
    *, config_path: Path, bundle: Path, expected_output_mode: str
) -> dict[str, Any]:
    errors: list[str] = []
    if not config_path.is_file():
        raise FileNotFoundError(f"runtime config does not exist: {config_path}")
    if not bundle.is_dir():
        raise FileNotFoundError(f"runtime bundle does not exist: {bundle}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    teleop = dict(config.get("teleop", {}) or {})
    policy = dict(teleop.get("policy", {}) or {})
    remote_mode = dict(teleop.get("policy_remote", {}) or {})
    scripted = dict(remote_mode.get("scripted_cycle", {}) or {})
    planner = dict(policy.get("cycle_planner", {}) or {})

    _check(
        errors,
        teleop.get("input") == "policy_remote",
        "teleop.input must be policy_remote",
    )
    _check(
        errors,
        Path(str(policy.get("bundle_dir", ""))).name == bundle.name,
        "policy.bundle_dir must select the verified bundle",
    )
    _check(
        errors,
        str(policy.get("ckpt_path", "")) == "policy_accepted.ckpt",
        "policy.ckpt_path must be policy_accepted.ckpt",
    )
    _check(
        errors,
        list(policy.get("camera_names", ())) == CAMERAS,
        "policy camera order must be video4,video5,video6,video7",
    )
    _check(
        errors,
        bool(policy.get("temporal_agg", False)),
        "temporal aggregation must be enabled",
    )
    _check(
        errors,
        str(policy.get("inference_precision", "")) == "fp32",
        "inference precision must remain fp32",
    )
    _check(
        errors,
        str(policy.get("output_mode", "")) == expected_output_mode,
        "policy output mode mismatch",
    )
    _check(
        errors,
        _float_list(policy.get("action_scale"), 4) == [1.0] * 4,
        "policy action scale must be identity",
    )
    _check(
        errors,
        not bool(dict(policy.get("deadzone_assist", {}) or {}).get("enabled", False)),
        "deadzone assist must remain disabled",
    )
    _check(
        errors,
        bool(policy.get("reset_policy_on_goal", False)),
        "reset_policy_on_goal must be true",
    )
    _check(errors, bool(planner.get("enabled", False)), "cycle planner must be enabled")
    _check(
        errors,
        not bool(planner.get("loop", True)),
        "field cycle script must be non-looping",
    )
    _check(
        errors,
        not bool(remote_mode.get("start_in_policy", True)),
        "policy_remote must start in manual mode",
    )
    _check(
        errors,
        bool(scripted.get("enabled", False)),
        "scripted-cycle runtime must be enabled",
    )
    _check(
        errors,
        bool(scripted.get("stop_on_wrong_ready", False)),
        "wrong stable side must stop the script",
    )

    checkpoint = bundle / "policy_accepted.ckpt"
    accepted_path = bundle / "accepted_model.json"
    manifest_path = bundle / "runtime_bundle_manifest.json"
    for path in (
        checkpoint,
        bundle / "dataset_stats.pkl",
        bundle / "resolved_config.yaml",
        accepted_path,
        manifest_path,
        bundle / "SHA256SUMS",
    ):
        _check(errors, path.is_file(), f"missing bundle file: {path.name}")
    if not errors:
        _verify_sha256(bundle, errors)

    accepted = (
        json.loads(accepted_path.read_text(encoding="utf-8"))
        if accepted_path.is_file()
        else {}
    )
    _check(
        errors,
        accepted.get("status") == "OFFLINE_ACCEPTED_FIELD_CANDIDATE",
        "accepted model status is invalid",
    )
    _check(
        errors,
        accepted.get("checkpoint") == "policy_accepted.ckpt",
        "accepted model checkpoint name is invalid",
    )
    expected_checkpoint_sha = str(accepted.get("checkpoint_sha256", "") or "")
    if expected_checkpoint_sha and checkpoint.is_file():
        _check(
            errors,
            _sha256(checkpoint) == expected_checkpoint_sha,
            "accepted checkpoint SHA-256 mismatch",
        )

    resolved_path = bundle / "resolved_config.yaml"
    resolved = (
        yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        if resolved_path.is_file()
        else {}
    )
    _check(
        errors,
        list((resolved.get("task", {}) or {}).get("camera_names", ())) == CAMERAS,
        "bundle camera order mismatch",
    )
    _check(
        errors,
        list((resolved.get("policy", {}) or {}).get("low_dim_keys", ()))
        == ["qpos", "real_transition_condition_v1"],
        "bundle low-dimensional input contract mismatch",
    )

    script_path = Path(str(planner.get("script_path", "")))
    _check(errors, script_path.is_file(), f"cycle script does not exist: {script_path}")
    script_manifest: dict[str, Any] = {}
    if script_path.is_file():
        try:
            script_manifest = ScriptCyclePlanner.from_script(
                script_path, loop=False
            ).manifest()
        except Exception as exc:
            errors.append(f"cycle script invalid: {type(exc).__name__}: {exc}")

    ready_source = Path(str(scripted.get("ready_contract", "")))
    ready_candidates = (
        ready_source,
        bundle / ready_source,
        bundle / ready_source.name,
        bundle / "contracts" / ready_source.name,
    )
    ready_path = next((path for path in ready_candidates if path.is_file()), None)
    _check(errors, ready_path is not None, "scripted-cycle ready contract is missing")
    if ready_path is not None:
        try:
            validate_rule_ready_contract(
                json.loads(ready_path.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            errors.append(f"ready contract invalid: {type(exc).__name__}: {exc}")

    target_source = Path(str(scripted.get("target_region_contract", "")))
    target_candidates = (
        target_source,
        bundle / target_source,
        bundle / target_source.name,
        bundle / "contracts" / target_source.name,
    )
    target_path = next((path for path in target_candidates if path.is_file()), None)
    _check(
        errors,
        target_path is not None,
        "scripted-cycle target-region contract is missing",
    )
    if target_path is not None:
        try:
            target_payload = json.loads(target_path.read_text(encoding="utf-8"))
            if (
                target_payload.get("schema")
                != "real_transition_target_release_contract_v1"
            ):
                raise ValueError("schema mismatch")
            decision = dict(target_payload.get("decision_region", {}) or {})
            _finite_pair(decision.get("train_A_endpoint_range_rad"))
            _finite_pair(decision.get("swing_qpos_range_rad"))
        except Exception as exc:
            errors.append(
                f"target-region contract invalid: {type(exc).__name__}: {exc}"
            )

    assist = dict(policy.get("deadzone_assist", {}) or {})
    _check(
        errors,
        _float_list(assist.get("deadzone_positive"), 4) == EXPECTED_DEADZONE_POSITIVE,
        "positive deadzone table mismatch",
    )
    _check(
        errors,
        _float_list(assist.get("deadzone_negative"), 4) == EXPECTED_DEADZONE_NEGATIVE,
        "negative deadzone table mismatch",
    )

    result = {
        "schema": "real_transition_target_release_runtime_preflight_v1",
        "status": "PASS" if not errors else "FAIL",
        "config": str(config_path),
        "bundle": str(bundle),
        "output_mode": expected_output_mode,
        "checkpoint": str(checkpoint),
        "script": script_manifest.get("script", {}),
        "errors": errors,
        "evidence_boundary": (
            "This verifies files and runtime contracts only; it does not prove "
            "hydraulic response or physical cycle completion."
        ),
    }
    if errors:
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _verify_sha256(bundle: Path, errors: list[str]) -> None:
    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=bundle,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        errors.append("runtime bundle SHA256SUMS verification failed")


def _check(errors: list[str], condition: bool, message: str) -> None:
    if not bool(condition):
        errors.append(str(message))


def _float_list(value: Any, width: int) -> list[float]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return []
    if array.shape != (int(width),):
        return []
    return [float(item) for item in array]


def _finite_pair(value: Any) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError("range must contain two finite values")
    low, high = float(array[0]), float(array[1])
    if low >= high:
        raise ValueError("range must be increasing")
    return low, high


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
