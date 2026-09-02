#!/usr/bin/env python3
"""Fail-closed static preflight for task-state-v2 shadow/control runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from testbed.actions.policy import _act_policy_config_from_resolved
from testbed.tasks.act_cycle_planner import ScriptCyclePlanner
from testbed.tasks.home_side_contract import validate_rule_ready_contract
from testbed.tasks.scripted_cycle_runtime import SwingLandingConfig


CAMERAS = ["video4", "video5", "video6", "video7"]
LOW_DIM_KEYS = ["qpos", "qvel", "real_transition_task_state_v2"]
CHECKPOINT_SHA256 = "e57bd59f07650f674f58eb9dfdaae2c06ead22b903922039cb2e6400daacaa4b"
EXPECTED_DEADZONE_POSITIVE = [0.661, 0.259, 0.500, 0.408]
EXPECTED_DEADZONE_NEGATIVE = [0.721, 0.357, 0.500, 0.508]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--expect-output-mode", choices=("shadow_zero", "control"), required=True
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Also construct the checkpoint and run three synthetic no-backend ticks.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    report = verify_runtime(
        config_path=args.config.resolve(),
        bundle=args.bundle_dir.resolve(),
        expected_output_mode=str(args.expect_output_mode),
    )
    if args.load_model:
        try:
            model_stdout = StringIO()
            with redirect_stdout(model_stdout):
                report["model_load_smoke"] = run_model_load_smoke(
                    bundle=args.bundle_dir.resolve(),
                    output_mode=str(args.expect_output_mode),
                    device=args.device,
                )
            report["model_load_smoke"]["captured_model_stdout"] = (
                model_stdout.getvalue().strip()
            )
        except Exception as exc:
            report["status"] = "FAIL"
            report["errors"].append(
                f"model load smoke failed: {type(exc).__name__}: {exc}"
            )
            raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2)) from exc
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
    joystick = dict(teleop.get("joystick", {}) or {})
    remote_mode = dict(teleop.get("policy_remote", {}) or {})
    scripted = dict(remote_mode.get("scripted_cycle", {}) or {})
    task_state = dict(scripted.get("task_state_v2", {}) or {})
    landing = dict(scripted.get("swing_landing", {}) or {})
    planner = dict(policy.get("cycle_planner", {}) or {})
    real = dict(config.get("real", {}) or {})
    control_pump = dict(real.get("control_pump", {}) or {})
    safety = dict(config.get("safety", {}) or {})

    _check(
        errors, teleop.get("input") == "policy_remote", "input must be policy_remote"
    )
    try:
        configured_bundle = Path(str(policy.get("bundle_dir", ""))).resolve()
    except OSError:
        configured_bundle = Path("/__invalid_bundle__")
    _check(
        errors,
        configured_bundle == bundle,
        "policy.bundle_dir must equal verified bundle",
    )
    _check(
        errors,
        policy.get("ckpt_path") == "policy_accepted.ckpt",
        "checkpoint name mismatch",
    )
    _check(
        errors, list(policy.get("camera_names", ())) == CAMERAS, "camera order mismatch"
    )
    _check(
        errors,
        bool(policy.get("temporal_agg", False)),
        "temporal aggregation must be enabled",
    )
    _check(
        errors, policy.get("qvel_mode") == "raw", "runtime must pass measured raw qvel"
    )
    _check(
        errors,
        policy.get("output_mode") == expected_output_mode,
        "output mode mismatch",
    )
    _check(
        errors,
        _float_list(policy.get("action_scale"), 4) == [1.0] * 4,
        "action scale must be identity",
    )
    _check(
        errors,
        not bool(dict(policy.get("deadzone_assist", {}) or {}).get("enabled", False)),
        "deadzone assist must stay disabled",
    )
    _check(
        errors,
        bool(policy.get("reset_policy_on_goal", False)),
        "goal commit must reset ACT",
    )
    _check(
        errors,
        bool(policy.get("reset_policy_on_phase_change", False)),
        "task-state changes must reset ACT",
    )
    _check(
        errors,
        not bool(remote_mode.get("start_in_policy", True)),
        "receiver must start in manual mode",
    )

    _check(
        errors,
        bool(task_state.get("enabled", False)),
        "task-state-v2 runtime owner must be enabled",
    )
    _check(
        errors,
        task_state.get("advance_source") == "operator_mark",
        "task-state owner must be operator_mark",
    )
    _check(
        errors,
        bool(task_state.get("require_excursion_before_work_complete", False)),
        "work-complete mark must require confirmed excursion",
    )
    _check(
        errors,
        joystick.get("mark_button") == 1,
        "physical button 2 must own task-state marks",
    )
    _check(
        errors,
        joystick.get("policy_start_button") == 6,
        "physical button 7 must own policy arm/stop",
    )
    _check(
        errors,
        joystick.get("record_start_button") is None,
        "record-start must be disabled so physical button 2 has one meaning",
    )
    _check(
        errors,
        list(joystick.get("joystick_ids", ())) == [0, 1, 0, 1]
        and list(joystick.get("axis_map", ())) == [1, 1, 0, 0]
        and list(joystick.get("invert", ())) == [True, False, True, False],
        "host joystick action mapping must match the field-proven four-axis mapping",
    )
    _check(
        errors,
        list(joystick.get("button_joystick_ids", ())) == [0],
        "task and policy buttons must come from the left joystick",
    )
    _check(
        errors, bool(scripted.get("enabled", False)), "scripted cycle must be enabled"
    )
    _check(
        errors,
        bool(scripted.get("auto_start_after_arm", False)),
        "script must wait for ready after arm",
    )
    _check(
        errors,
        bool(scripted.get("stop_on_wrong_ready", False)),
        "wrong ready side must stop",
    )
    _check(errors, bool(planner.get("enabled", False)), "cycle planner must be enabled")
    _check(errors, not bool(planner.get("loop", True)), "cycle planner must be finite")

    try:
        landing_cfg = SwingLandingConfig.from_mapping(landing)
        _check(
            errors, landing_cfg.enabled, "measured-state swing landing must be enabled"
        )
        _check(
            errors,
            landing_cfg.min_action_positive == EXPECTED_DEADZONE_POSITIVE[0]
            and landing_cfg.min_action_negative == EXPECTED_DEADZONE_NEGATIVE[0],
            "landing mechanical threshold mismatch",
        )
    except Exception as exc:
        errors.append(f"swing landing invalid: {type(exc).__name__}: {exc}")

    _check(
        errors,
        bool(control_pump.get("enabled", False)),
        "50 Hz real control pump must be enabled",
    )
    _check(
        errors,
        float(control_pump.get("hz", 0.0)) == 50.0,
        "control pump must run at 50 Hz",
    )
    _check(
        errors,
        bool(control_pump.get("zero_on_stop", False)),
        "control pump must zero on stop",
    )
    _check(
        errors, bool(safety.get("deadman_enabled", False)), "deadman must be enabled"
    )
    _check(errors, bool(safety.get("estop_enabled", False)), "estop must be enabled")
    _check(
        errors,
        bool(safety.get("manual_override_enabled", False)),
        "manual override must be enabled",
    )

    required = (
        "policy_accepted.ckpt",
        "dataset_stats.pkl",
        "resolved_config.yaml",
        "run_metadata.json",
        "accepted_model.json",
        "runtime_bundle_manifest.json",
        "SOURCE_COMMIT.txt",
        "SHA256SUMS",
        "contracts/ready_contract.json",
        "contracts/target_release_contract_v2.json",
        "contracts/direct_policy_output_mechanical_deadzone.json",
        "contracts/task_state_runtime_contract.json",
        "manifest/task_state_manifest.json",
        "evaluation/probe_result.json",
    )
    for name in required:
        _check(errors, (bundle / name).is_file(), f"missing bundle file: {name}")
    if (bundle / "SHA256SUMS").is_file():
        _verify_sha256(bundle, errors)

    checkpoint = bundle / "policy_accepted.ckpt"
    if checkpoint.is_file():
        _check(
            errors,
            _sha256(checkpoint) == CHECKPOINT_SHA256,
            "checkpoint SHA-256 mismatch",
        )
    accepted = _optional_json(bundle / "accepted_model.json", errors)
    _check(
        errors,
        accepted.get("status") == "OFFLINE_CANDIDATE_ONLY",
        "bundle must retain offline-candidate evidence status",
    )
    _check(
        errors,
        accepted.get("checkpoint") == "policy_accepted.ckpt",
        "accepted checkpoint filename mismatch",
    )
    accepted_runtime = dict(accepted.get("runtime", {}) or {})
    _check(
        errors,
        bool(accepted_runtime.get("task_state_owner_implemented", False)),
        "accepted manifest must record task-state owner implementation",
    )
    _check(
        errors,
        bool(accepted_runtime.get("control_path_implemented", False)),
        "accepted manifest must record control path implementation",
    )
    _check(
        errors,
        not bool(accepted_runtime.get("controlled_motion_authorized_by_bundle", True)),
        "bundle must not self-authorize physical motion",
    )

    resolved = (
        yaml.safe_load((bundle / "resolved_config.yaml").read_text(encoding="utf-8"))
        or {}
        if (bundle / "resolved_config.yaml").is_file()
        else {}
    )
    resolved_policy = dict(resolved.get("policy", {}) or {})
    resolved_train = dict(resolved.get("train", {}) or {})
    inference_policy = _act_policy_config_from_resolved(resolved)
    _check(
        errors,
        list((resolved.get("task", {}) or {}).get("camera_names", ())) == CAMERAS,
        "bundle camera order mismatch",
    )
    _check(
        errors,
        list(resolved_policy.get("low_dim_keys", ())) == LOW_DIM_KEYS,
        "bundle low-dimensional input contract mismatch",
    )
    _check(
        errors,
        int(dict(resolved_policy.get("act_params", {}) or {}).get("state_dim", -1))
        == 13,
        "bundle state_dim must be 13",
    )
    _check(
        errors,
        int(dict(resolved_policy.get("act_params", {}) or {}).get("chunk_size", -1))
        == 20,
        "bundle ACT chunk size must be 20",
    )
    _check(
        errors,
        bool(
            dict(resolved_train.get("state_visual_residual", {}) or {}).get(
                "enabled", False
            )
        ),
        "state/visual residual architecture must be enabled",
    )
    _check(
        errors,
        bool(
            dict(resolved_train.get("task_state_v2_adherence_loss", {}) or {}).get(
                "enabled", False
            )
        ),
        "task-state adherence training contract missing",
    )
    _check(
        errors,
        bool(dict(resolved_train.get("deadzone_loss", {}) or {}).get("enabled", False)),
        "deadzone training contract missing",
    )
    _check(
        errors,
        inference_policy.get("backbone_pretrained") is False,
        "checkpoint inference must disable pretrained backbone downloads",
    )

    runtime_contract = _optional_json(
        bundle / "contracts/task_state_runtime_contract.json", errors
    )
    _check(
        errors,
        runtime_contract.get("schema")
        == "real_transition_task_state_v2_runtime_contract_v1",
        "task-state runtime contract schema mismatch",
    )
    owner = dict(runtime_contract.get("owner", {}) or {})
    _check(
        errors,
        owner.get("type") == "planner_plus_explicit_operator_mark",
        "bundle runtime owner mismatch",
    )
    _check(
        errors,
        owner.get("automatic_observation_inference") is False,
        "task events must not be inferred from observations",
    )

    _verify_ready_and_target_contracts(bundle, scripted, errors)
    script_manifest = _verify_scripts(planner, errors)
    source_commit = (
        (bundle / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
        if (bundle / "SOURCE_COMMIT.txt").is_file()
        else ""
    )
    current_commit = _git_head()
    _check(errors, len(source_commit) == 40, "bundle SOURCE_COMMIT is invalid")
    if len(source_commit) == 40:
        _check(
            errors,
            _is_ancestor(source_commit, current_commit),
            "runtime checkout does not contain bundle source commit",
        )

    report = {
        "schema": "real_transition_task_state_v2_runtime_preflight_v1",
        "status": "PASS" if not errors else "FAIL",
        "config": str(config_path),
        "bundle": str(bundle),
        "output_mode": expected_output_mode,
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else "",
        "bundle_source_commit": source_commit,
        "runtime_checkout_commit": current_commit,
        "task_state_owner": "planner_plus_explicit_operator_mark",
        "script": script_manifest,
        "software_control_path": [
            "PolicyActionSource",
            "ScriptedCycleRuntime",
            "RemoteArmedPolicyActionSource",
            "SwingLanding",
            "ActionGuard",
            "RealActionPump",
            "BridgeLowLevelController",
        ],
        "errors": errors,
        "evidence_boundary": (
            "Static files, task-state ownership, and software action path only. "
            "This does not prove hydraulic response, soil effect, or physical "
            "closed-loop completion."
        ),
    }
    if errors:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def run_model_load_smoke(
    *, bundle: Path, output_mode: str, device: str | None = None
) -> dict[str, Any]:
    """Exercise model construction, task tokens, aggregation reset, and output mode.

    This function never constructs a backend, action pump, bridge client, or
    network socket. Its synthetic observations are only a software-load test.
    """

    from testbed.actions.policy import PolicyActionSource

    source = PolicyActionSource.from_config(
        {
            "bundle_dir": str(bundle),
            "ckpt_path": "policy_accepted.ckpt",
            "device": device,
            "camera": "video4",
            "camera_names": CAMERAS,
            "temporal_agg": True,
            "device_uint8_preprocess": True,
            "temporal_aggregation_diagnostics": True,
            "output_mode": str(output_mode),
            "qvel_mode": "raw",
            "action_scale": [1.0] * 4,
            "deadzone_assist": {"enabled": False},
            "reset_policy_on_goal": True,
            "reset_policy_on_phase_change": True,
            "cycle_planner": {
                "enabled": True,
                "pattern": "BA",
                "loop": False,
            },
        }
    )
    try:
        source.commit_cycle_goal()
        observation = {
            "qpos": np.asarray([0.2, -0.5, 0.7, -0.8], dtype=np.float32),
            "qvel": np.asarray([0.0, 0.01, -0.02, 0.01], dtype=np.float32),
            "images": {
                name: np.zeros((216, 384, 3), dtype=np.uint8) for name in CAMERAS
            },
            "timestamp_ns": 1_000_000_000,
        }
        returned: list[np.ndarray] = []
        infos = []
        action, info = source.next_action(observation)
        returned.append(np.asarray(action, dtype=np.float32))
        infos.append(info)
        source.set_task_dig_complete(completed=True)
        action, info = source.next_action(observation)
        returned.append(np.asarray(action, dtype=np.float32))
        infos.append(info)
        source.set_task_return_commit(committed=True)
        action, info = source.next_action(observation)
        returned.append(np.asarray(action, dtype=np.float32))
        infos.append(info)
        tokens = [
            np.asarray(info.extras["policy_task_state_v2"], dtype=np.float32)
            for info in infos
        ]
        expected_tokens = [
            np.asarray([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float32),
        ]
        if not all(
            np.array_equal(actual, expected)
            for actual, expected in zip(tokens, expected_tokens, strict=True)
        ):
            raise RuntimeError("task token sequence mismatch")
        errors = [str(info.extras.get("policy_error", "")) for info in infos]
        if any(errors):
            raise RuntimeError(f"policy reported errors: {errors}")
        raw = [
            np.asarray(info.extras["policy_action"], dtype=np.float32) for info in infos
        ]
        if not all(np.isfinite(value).all() for value in (*raw, *returned)):
            raise RuntimeError("model smoke produced non-finite action")
        if output_mode == "shadow_zero" and any(
            float(np.max(np.abs(value))) > 1.0e-7 for value in returned
        ):
            raise RuntimeError("shadow_zero returned a nonzero action")
        if output_mode == "control":
            assisted = [
                np.asarray(info.extras["policy_assisted_action"], dtype=np.float32)
                for info in infos
            ]
            if not all(
                np.allclose(actual, expected)
                for actual, expected in zip(returned, assisted, strict=True)
            ):
                raise RuntimeError(
                    "control output does not match assisted policy action"
                )
        return {
            "status": "PASS",
            "device": str(getattr(source._policy, "device", device or "resolved")),
            "output_mode": str(output_mode),
            "task_tokens": [value.tolist() for value in tokens],
            "raw_action_max_abs": max(float(np.max(np.abs(value))) for value in raw),
            "returned_action_max_abs": max(
                float(np.max(np.abs(value))) for value in returned
            ),
            "backend_constructed": False,
            "evidence_boundary": "Synthetic observations; no backend or physical motion.",
        }
    finally:
        source.close()


def _verify_ready_and_target_contracts(
    bundle: Path, scripted: dict[str, Any], errors: list[str]
) -> None:
    ready_path = _bundle_contract(bundle, scripted.get("ready_contract"))
    if ready_path is None:
        errors.append("ready contract is missing")
    else:
        try:
            validate_rule_ready_contract(_json(ready_path))
        except Exception as exc:
            errors.append(f"ready contract invalid: {type(exc).__name__}: {exc}")
    target_path = _bundle_contract(bundle, scripted.get("target_region_contract"))
    if target_path is None:
        errors.append("target-region contract is missing")
    else:
        try:
            payload = _json(target_path)
            if payload.get("schema") != "real_transition_target_release_contract_v1":
                raise ValueError("schema mismatch")
            decision = dict(payload.get("decision_region", {}) or {})
            _finite_pair(decision.get("train_A_endpoint_range_rad"))
            _finite_pair(decision.get("swing_qpos_range_rad"))
        except Exception as exc:
            errors.append(
                f"target-region contract invalid: {type(exc).__name__}: {exc}"
            )


def _verify_scripts(planner: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    raw = planner.get("script_paths_by_initial_side")
    single_path = str(planner.get("script_path", "") or "").strip()
    if raw is not None and single_path:
        errors.append("planner cannot set both one script and side-matched scripts")
        return {}
    if single_path:
        path = Path(single_path).resolve()
        if not path.is_file():
            errors.append(f"cycle script does not exist: {path}")
            return {}
        try:
            manifest = ScriptCyclePlanner.from_script(path, loop=False).manifest()
            cycles = list(manifest.get("cycles", ()))
            if not cycles or any(
                row["current_side"] == row["target_side"] for row in cycles
            ):
                raise ValueError("script must contain alternating A/B cycles")
            return {"single": manifest["script"]}
        except Exception as exc:
            errors.append(f"cycle script invalid: {type(exc).__name__}: {exc}")
            return {}
    if not isinstance(raw, dict) or set(str(key).upper() for key in raw) != {"A", "B"}:
        errors.append("side-matched A/B scripts are required")
        return {}
    result: dict[str, Any] = {}
    for raw_side, raw_path in raw.items():
        side = str(raw_side).upper()
        path = Path(str(raw_path)).resolve()
        if not path.is_file():
            errors.append(f"cycle script does not exist: {path}")
            continue
        try:
            manifest = ScriptCyclePlanner.from_script(path, loop=False).manifest()
            cycles = list(manifest.get("cycles", ()))
            if manifest["script"]["initial_side"] != side:
                raise ValueError("declared initial side mismatch")
            if len(cycles) != 4:
                raise ValueError("script must contain four cycles")
            if not all(row["current_side"] != row["target_side"] for row in cycles):
                raise ValueError("script must alternate A/B")
            result[side] = manifest["script"]
        except Exception as exc:
            errors.append(f"cycle script {side} invalid: {type(exc).__name__}: {exc}")
    return result


def _bundle_contract(bundle: Path, raw: Any) -> Path | None:
    value = Path(str(raw or ""))
    for candidate in (
        value,
        bundle / value,
        bundle / value.name,
        bundle / "contracts" / value.name,
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


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


def _optional_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _json(path)
    except Exception as exc:
        errors.append(f"invalid JSON {path.name}: {type(exc).__name__}: {exc}")
        return {}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _check(errors: list[str], condition: bool, message: str) -> None:
    if not bool(condition):
        errors.append(message)


def _float_list(value: Any, width: int) -> list[float]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return []
    if array.shape != (width,) or not np.isfinite(array).all():
        return []
    return [float(item) for item in array]


def _finite_pair(value: Any) -> tuple[float, float]:
    result = _float_list(value, 2)
    if len(result) != 2 or result[0] >= result[1]:
        raise ValueError("range must be two increasing finite values")
    return result[0], result[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            check=False,
        ).returncode
        == 0
    )


if __name__ == "__main__":
    main()
