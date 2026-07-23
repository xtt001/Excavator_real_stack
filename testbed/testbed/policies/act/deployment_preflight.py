"""Portable ACT bundle and shadow-runtime deployment validation.

This module owns immutable bundle identity and configuration compatibility.  It
does not load a model, send a command, or promote a shadow configuration to
control mode.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

BUNDLE_SCHEMA_VERSION = "act_runtime_bundle_v1"
REQUIRED_BUNDLE_FILES = (
    "policy_best.ckpt",
    "dataset_stats.pkl",
    "resolved_config.yaml",
    "run_metadata.json",
)
BUNDLE_MANIFEST_FILENAME = "runtime_bundle_manifest.json"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Read one YAML mapping without accepting scalar/list roots."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML root must be a mapping: {source}")
    return dict(payload)


def verify_bundle_manifest(
    *,
    bundle_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Verify required portable files against an immutable bundle manifest."""

    bundle = Path(bundle_dir).expanduser().resolve()
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"bundle manifest schema_version must be {BUNDLE_SCHEMA_VERSION!r}"
        )
    raw_files = manifest.get("files")
    if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
        raise ValueError("bundle manifest files must be a sequence")
    records = {
        str(item.get("name", "")): item
        for item in raw_files
        if isinstance(item, Mapping)
    }
    missing_records = [name for name in REQUIRED_BUNDLE_FILES if name not in records]
    if missing_records:
        raise ValueError(
            "bundle manifest missing required file record(s): "
            + ", ".join(missing_records)
        )

    verified: dict[str, str] = {}
    for name in REQUIRED_BUNDLE_FILES:
        path = bundle / name
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_size = records[name].get("size_bytes")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool):
            raise ValueError(f"bundle manifest {name} size_bytes must be an integer")
        if path.stat().st_size != expected_size:
            raise ValueError(f"bundle file size mismatch: {name}")
        expected_sha = str(records[name].get("sha256", ""))
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"bundle file SHA-256 mismatch: {name}")
        verified[name] = actual_sha
    return verified


def verify_shadow_deployment(
    *,
    config_path: str | Path,
    bundle_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a no-motion ACT deployment configuration and bundle identity."""

    config_source = Path(config_path).expanduser().resolve()
    bundle = Path(bundle_dir).expanduser().resolve()
    config = read_yaml_mapping(config_source)
    verified_hashes = verify_bundle_manifest(bundle_dir=bundle, manifest=manifest)
    resolved = read_yaml_mapping(bundle / "resolved_config.yaml")

    teleop = _mapping(config.get("teleop"), "teleop")
    policy = _mapping(teleop.get("policy"), "teleop.policy")
    task = _mapping(config.get("task"), "task")
    real = _mapping(config.get("real"), "real")
    sync = _mapping(config.get("sync"), "sync")
    receiver = _mapping(config.get("receiver"), "receiver")
    health = _mapping(receiver.get("health"), "receiver.health")

    configured_bundle = Path(str(policy.get("bundle_dir", ""))).expanduser()
    if configured_bundle.resolve() != bundle:
        raise ValueError(
            "teleop.policy.bundle_dir does not match the verified bundle: "
            f"{configured_bundle} != {bundle}"
        )
    if str(policy.get("output_mode", "")) != "shadow_zero":
        raise ValueError("shadow deployment output_mode must be shadow_zero")
    if bool(_mapping(policy.get("deadzone_assist", {}), "deadzone_assist").get("enabled")):
        raise ValueError("shadow deployment deadzone_assist must be disabled")
    if bool(_mapping(policy.get("runtime_gates", {}), "runtime_gates").get("enabled")):
        raise ValueError("G49 N5 shadow deployment runtime_gates must be disabled")

    resolved_task = _mapping(resolved.get("task"), "resolved task")
    resolved_policy = _mapping(resolved.get("policy"), "resolved policy")
    resolved_params = _mapping(resolved_policy.get("act_params"), "resolved act_params")
    cameras = _string_list(resolved_task.get("camera_names"), "resolved camera_names")
    for name, raw in (
        ("teleop.policy.camera_names", policy.get("camera_names")),
        ("task.camera_names", task.get("camera_names")),
        ("sync.required_cameras", sync.get("required_cameras")),
    ):
        actual = _string_list(raw, name)
        if actual != cameras:
            raise ValueError(f"{name} differs from bundle camera order: {actual!r} != {cameras!r}")

    low_dim_keys = _string_list(resolved_policy.get("low_dim_keys"), "low_dim_keys")
    if low_dim_keys != ["qpos"]:
        raise ValueError(f"G49 N5 low_dim_keys must be ['qpos'], got {low_dim_keys!r}")

    role_config = _mapping(
        resolved_params.get("camera_role_encoding"),
        "resolved camera_role_encoding",
    )
    if not bool(role_config.get("enabled", False)):
        raise ValueError("G49 N5 camera_role_encoding must be enabled")
    roles = _mapping(role_config.get("roles"), "resolved camera roles")
    if list(roles) != cameras:
        raise ValueError("camera role keys must preserve the bundle camera order")

    experiment = _mapping(resolved.get("experiment_contract"), "experiment_contract")
    expected_scale = _float_vector(
        experiment.get("policy_action_scale"),
        "experiment_contract.policy_action_scale",
    )
    actual_scale = _float_vector(policy.get("action_scale"), "policy.action_scale")
    if actual_scale != expected_scale:
        raise ValueError(
            f"policy.action_scale changes the evaluated action domain: {actual_scale!r} != {expected_scale!r}"
        )

    backend = _mapping(resolved_task.get("backend"), "resolved task.backend")
    training_dt = _positive_float(backend.get("dt"), "resolved task.backend.dt")
    record_hz = _positive_float(task.get("record_hz"), "task.record_hz")
    config_dt = _positive_float(task.get("dt"), "task.dt")
    expected_hz = 1.0 / training_dt
    if abs(record_hz - expected_hz) > 1e-6 or abs(config_dt - training_dt) > 1e-9:
        raise ValueError(
            "policy sampling must match training time base: "
            f"record_hz={record_hz}, dt={config_dt}, expected_hz={expected_hz}, expected_dt={training_dt}"
        )
    control_hz = _positive_float(real.get("control_hz"), "real.control_hz")
    pump = _mapping(real.get("control_pump"), "real.control_pump")
    pump_hz = _positive_float(pump.get("hz"), "real.control_pump.hz")
    if control_hz < record_hz or pump_hz < record_hz:
        raise ValueError("control and pump rates must not be below policy sampling rate")
    if not bool(pump.get("zero_on_stop", False)):
        raise ValueError("control_pump.zero_on_stop must be true")
    if str(health.get("mode", "")) != "strict":
        raise ValueError("receiver.health.mode must be strict")

    return {
        "ok": True,
        "candidate_id": str(manifest.get("candidate_id", "")),
        "config_path": str(config_source),
        "config_sha256": sha256_file(config_source),
        "bundle_dir": str(bundle),
        "camera_names": cameras,
        "camera_roles": {str(key): str(value) for key, value in roles.items()},
        "low_dim_keys": low_dim_keys,
        "policy_sampling_hz": record_hz,
        "control_hz": control_hz,
        "control_pump_hz": pump_hz,
        "output_mode": "shadow_zero",
        "action_scale": actual_scale,
        "verified_bundle_hashes": verified_hashes,
    }


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    result = [str(item).strip() for item in value]
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique nonempty strings")
    return result


def _float_vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a four-value sequence")
    result = [float(item) for item in value]
    if len(result) != 4:
        raise ValueError(f"{name} must contain four values")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


__all__ = [
    "BUNDLE_MANIFEST_FILENAME",
    "BUNDLE_SCHEMA_VERSION",
    "REQUIRED_BUNDLE_FILES",
    "read_yaml_mapping",
    "sha256_file",
    "verify_bundle_manifest",
    "verify_shadow_deployment",
]
