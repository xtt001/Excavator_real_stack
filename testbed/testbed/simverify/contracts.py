"""Frozen M0 contracts and provenance helpers for SimVerify exports.

This module intentionally owns only recorded-observation contracts.  It does
not import PACT, inspect simulator runtime state, or provide a deployment path
for a sim-domain checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py

CAMERA_MAPPING_ID = "sim_yulong_to_real_g49_semantic_roles_v1"
IMAGE_TRANSFORM_ID = "sim_jpeg_rgb_resize_512x288_to_384x216_linear_v1"
STATE_ACTION_TIME_CONTRACT_ID = "sim_source_state_action_time_v1"
EXPORT_EPISODE_SCHEMA = "sim_observable_cycle_export_episode_v1"
DATASET_MANIFEST_SCHEMA = "sim_observable_cycle_export_v1"
CONDITION_SCHEMA_VERSION = "cycle_condition_v1"

SOURCE_CAMERA_ORDER = ("stick_up", "stick_down", "eye_left", "eye_right")
POLICY_CAMERA_ORDER = ("video4", "video5", "video6", "video7")
SOURCE_TO_POLICY_CAMERA = {
    "eye_left": "video4",
    "eye_right": "video5",
    "stick_down": "video6",
    "stick_up": "video7",
}
POLICY_CAMERA_ROLES = {
    "video4": "eye_left",
    "video5": "eye_right",
    "video6": "stick_down",
    "video7": "stick_up",
}

SOURCE_QPOS_ORDER = (
    "swing_position_norm",
    "boom_position_norm",
    "stick_position_norm",
    "bucket_position_norm",
)
SOURCE_QVEL_ORDER = (
    "swing_speed",
    "boom_speed",
    "stick_speed",
    "bucket_speed",
)
SOURCE_ACTION_ORDER = (
    "swing_speed_cmd",
    "boom_speed_cmd",
    "stick_speed_cmd",
    "bucket_speed_cmd",
)
POLICY_AXIS_ORDER = ("swing", "boom", "stick", "bucket")
SOURCE_ACTION_GENERATION = {
    "joystick_axis_map": [1, 1, 0, 0],
    "joystick_invert": [True, False, False, True],
    "scale": [1.0, 1.0, 1.0, 1.0],
    "symmetric_limit": [1.0, 1.0, 1.0, 1.0],
    "deadzone": [0.05, 0.05, 0.05, 0.05],
    "response_profile": {
        "enabled": True,
        "attack_rate": [4.0, 4.0, 4.0, 4.0],
        "release_rate": [6.0, 6.0, 6.0, 6.0],
        "recenter_rate": [7.0, 7.0, 7.0, 7.0],
        "exponent": [1.0, 1.0, 1.0, 1.0],
        "use_measured_dt": False,
    },
}

FROZEN_SOURCE_DT_S = 0.02
FROZEN_SOURCE_HZ = 50.0
FROZEN_TARGET_HZ = 20.0
FROZEN_ACTION_LABEL_OFFSET_S = 0.0

ALLOWED_EXPORT_GROUPS = frozenset(
    {
        "metadata",
        "observations",
        "observations/encoded_images",
        "conditions",
        "timestamps",
        "diagnostics",
    }
)
ALLOWED_EXPORT_DATASETS = frozenset(
    {
        "observations/qpos",
        "observations/qvel",
        "observations/encoded_images/video4",
        "observations/encoded_images/video5",
        "observations/encoded_images/video6",
        "observations/encoded_images/video7",
        "conditions/cycle_condition_v1",
        "conditions/cycle_id",
        "conditions/valid_mask",
        "action",
        "timestamps/step_id",
        "timestamps/sim_time_s",
        "diagnostics/source_observation_index",
        "diagnostics/source_action_index",
        "diagnostics/source_step_id",
        "diagnostics/source_sim_time_s",
        "diagnostics/target_tick",
        "diagnostics/target_sim_time_s",
        "diagnostics/selection_error_s",
    }
)
ALLOWED_ROOT_ATTRIBUTES = frozenset({"sim", "is_real", "simverify_export"})
ALLOWED_CONDITIONS_GROUP_ATTRIBUTES = frozenset(
    {
        "schema_id",
        "dim",
        "encoding",
        "normalization",
        "source",
        "scope",
        "materialized_from_sha256",
        "schema_sha256",
    }
)
ALLOWED_METADATA_ATTRIBUTES = frozenset(
    {
        "schema_version",
        "evidence_scope",
        "source_dataset_path",
        "source_dataset_sha256",
        "source_episode_id",
        "source_n_steps",
        "n_steps",
        "source_dt_s",
        "source_hz",
        "record_hz",
        "control_hz",
        "dt",
        "sampling_hz",
        "source_time_basis",
        "source_step_ns_used",
        "output_step_id_semantics",
        "action_label_offset_s",
        "action_prealigned",
        "camera_names",
        "source_camera_names",
        "camera_mapping_id",
        "camera_contract_sha256",
        "image_transform_id",
        "source_image_width",
        "source_image_height",
        "output_image_width",
        "output_image_height",
        "image_color_space",
        "image_resize_filter",
        "image_crop_policy",
        "geometric_equivalence",
        "qpos_order",
        "qvel_order",
        "action_order",
        "qpos_semantics",
        "qvel_semantics",
        "action_semantics",
        "state_domain",
        "action_domain",
        "checkpoint_restriction",
        "state_action_time_contract_id",
        "state_action_time_contract_sha256",
        "condition_schema_version",
        "condition_dim",
        "condition_status",
        "condition_source",
        "command_source",
        "export_git_commit",
        "export_git_branch",
        "export_git_dirty",
    }
)
ALLOWED_DATASET_ATTRIBUTES = frozenset(
    {
        "encoding",
        "color_space",
        "width",
        "height",
        "transform_id",
        "source_camera",
        "policy_camera",
        "schema_id",
        "dim",
        "normalization",
        "source",
        "scope",
    }
)

PRIVILEGED_NAME_TOKENS = (
    "env_state",
    "reward",
    "bucket_mass",
    "mass_in_bucket",
    "terrain_grid",
    "removed_depth",
    "bucket_tip",
    "tip_world",
    "planner",
    "goal_token",
    "contact",
    "step_ns",
)


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash one JSON mapping using a stable, whitespace-free encoding."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def camera_transform_contract() -> dict[str, Any]:
    """Return the frozen physical-role and pixel-transform contract."""

    payload: dict[str, Any] = {
        "schema_version": "sim_camera_transform_contract_v1",
        "mapping_id": CAMERA_MAPPING_ID,
        "source_camera_order": list(SOURCE_CAMERA_ORDER),
        "source_to_policy": dict(SOURCE_TO_POLICY_CAMERA),
        "policy_order": list(POLICY_CAMERA_ORDER),
        "policy_roles": dict(POLICY_CAMERA_ROLES),
        "physical_role_mapping_only": True,
        "geometric_equivalence": False,
        "source": {
            "storage": "jpeg",
            "decoded_color_space": "RGB",
            "layout": "HWC",
            "width": 512,
            "height": 288,
        },
        "transform": {
            "transform_id": IMAGE_TRANSFORM_ID,
            "decode": "opencv_imdecode_color_then_bgr_to_rgb",
            "crop": "none",
            "resize": {
                "width": 384,
                "height": 216,
                "filter": "linear",
            },
            "output_storage": "jpeg",
            "output_color_space": "RGB",
            "output_layout": "HWC",
        },
        "capability_boundary": [
            "The mapping freezes physical camera roles only.",
            "It does not claim equal intrinsics, extrinsics, field of view, or geometry.",
        ],
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    return payload


def state_action_time_contract() -> dict[str, Any]:
    """Return the frozen source-domain state, action, condition, and time contract."""

    payload: dict[str, Any] = {
        "schema_version": STATE_ACTION_TIME_CONTRACT_ID,
        "qpos": {
            "order": list(SOURCE_QPOS_ORDER),
            "representation": "sim_source_representation",
            "real_unit_mapping": None,
        },
        "qvel": {
            "order": list(SOURCE_QVEL_ORDER),
            "representation": "sim_source_representation",
            "real_unit_mapping": None,
        },
        "action": {
            "source_order": list(SOURCE_ACTION_ORDER),
            "policy_axis_order": list(POLICY_AXIS_ORDER),
            "semantics": "actuator_speed_cmd",
            "domain": "sim_source_domain",
            "positive_direction": (
                "source_recorded_axis_positive_no_real_equivalence_claim"
            ),
            "real_amplitude_mapping": None,
            "source_generation": SOURCE_ACTION_GENERATION,
            "source_generation_provenance": {
                "array_fields": {
                    "joystick_axis_map": "metadata.attrs.axis_map",
                    "joystick_invert": "metadata.attrs.invert",
                    "scale": "metadata.attrs.scale",
                    "symmetric_limit": "metadata.attrs.limit",
                    "deadzone": "metadata.attrs.deadzone",
                    "response_profile.enabled": (
                        "metadata.attrs.response_profile_enabled"
                    ),
                    "response_profile.attack_rate": (
                        "metadata.attrs.response_profile_attack_rate"
                    ),
                    "response_profile.release_rate": (
                        "metadata.attrs.response_profile_release_rate"
                    ),
                    "response_profile.recenter_rate": (
                        "metadata.attrs.response_profile_recenter_rate"
                    ),
                    "response_profile.exponent": (
                        "metadata.attrs.response_profile_exponent"
                    ),
                },
                "use_measured_dt": (
                    "metadata.attrs.record_config_yaml.teleop.joystick."
                    "response_profile.use_measured_dt"
                ),
            },
        },
        "condition": {
            "schema_version": CONDITION_SCHEMA_VERSION,
            "dim": 6,
            "field_order": [
                "current_left",
                "current_center",
                "current_right",
                "next_left",
                "next_center",
                "next_right",
            ],
            "command_source": "unknown_not_recorded",
            "policy_condition_source": "hindsight_outcome",
        },
        "time": {
            "source_hz": FROZEN_SOURCE_HZ,
            "source_dt_s": FROZEN_SOURCE_DT_S,
            "target_hz": FROZEN_TARGET_HZ,
            "target_dt_s": 1.0 / FROZEN_TARGET_HZ,
            "source_time_basis": "timestamps/step_id * metadata.dt",
            "wall_clock_step_ns_used": False,
            "selection": "first_complete_source_row_not_earlier_than_target",
            "same_source_row_for_all_fields": True,
            "action_label_offset_s": FROZEN_ACTION_LABEL_OFFSET_S,
        },
        "checkpoint_restriction": "sim_state_domain_only_not_real_deployable",
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    return payload


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one regular file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: str | Path) -> dict[str, Any]:
    """Return an immutable identity record for one file."""

    source = Path(path).resolve(strict=True)
    stat = source.stat()
    return {
        "path": str(source),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(source),
    }


def git_provenance(repo_root: str | Path) -> dict[str, Any]:
    """Return commit, branch, and dirty state without mutating the repository."""

    root = Path(repo_root).resolve()

    def run(*args: str) -> tuple[int, str, str]:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    code, inside, error = run("rev-parse", "--is-inside-work-tree")
    if code != 0 or inside != "true":
        return {
            "path": str(root),
            "git_available": False,
            "error": error,
        }
    _, commit, _ = run("rev-parse", "HEAD")
    _, branch, _ = run("branch", "--show-current")
    _, status, _ = run("status", "--short")
    return {
        "path": str(root),
        "git_available": True,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def git_ref_provenance(
    repo_root: str | Path,
    ref: str,
) -> dict[str, Any]:
    """Resolve a frozen Git ref to both its object and peeled commit."""

    root = Path(repo_root).resolve(strict=True)

    def resolve(expression: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", expression],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(
                f"cannot resolve frozen Git ref {expression!r}: "
                f"{result.stderr.strip()}"
            )
        return result.stdout.strip()

    object_sha = resolve(ref)
    commit_sha = resolve(f"{ref}^{{commit}}")
    return {
        "ref": str(ref),
        "object_sha": object_sha,
        "commit_sha": commit_sha,
    }


def collect_hdf5_source_provenance(path: str | Path) -> list[dict[str, Any]]:
    """Hash one HDF5 file and every backing file in its VDS chain."""

    records: list[dict[str, Any]] = []
    visited: set[Path] = set()

    def visit_file(candidate: Path) -> None:
        resolved = candidate.resolve(strict=True)
        if resolved in visited:
            return
        visited.add(resolved)
        records.append(file_provenance(resolved))

        backing_files: set[Path] = set()
        with h5py.File(resolved, "r") as handle:

            def collect(name: str, obj: h5py.Group | h5py.Dataset) -> None:
                if not isinstance(obj, h5py.Dataset) or not obj.is_virtual:
                    return
                for source in obj.virtual_sources():
                    raw_name = _decode_hdf5_text(source.file_name)
                    if raw_name in {"", "."}:
                        continue
                    source_path = Path(raw_name)
                    if not source_path.is_absolute():
                        source_path = resolved.parent / source_path
                    try:
                        backing = source_path.resolve(strict=True)
                    except FileNotFoundError as exc:
                        raise FileNotFoundError(
                            f"{resolved}: virtual dataset {name!r} references "
                            f"missing backing file {raw_name!r}"
                        ) from exc
                    if not backing.is_file():
                        raise FileNotFoundError(
                            f"{resolved}: virtual dataset {name!r} backing "
                            f"path is not a file: {backing}"
                        )
                    backing_files.add(backing)

            handle.visititems(collect)
        for backing in sorted(backing_files, key=str):
            visit_file(backing)

    visit_file(Path(path))
    return records


def assert_source_provenance_unchanged(records: list[Mapping[str, Any]]) -> None:
    """Fail if a previously hashed source file changed while an export ran."""

    for record in records:
        path = Path(str(record["path"]))
        if not path.is_file():
            raise RuntimeError(f"source file disappeared during export: {path}")
        stat = path.stat()
        expected = (int(record["size_bytes"]), int(record["mtime_ns"]))
        actual = (int(stat.st_size), int(stat.st_mtime_ns))
        if actual != expected:
            raise RuntimeError(
                f"source file changed during export: {path}; "
                f"expected size/mtime={expected}, got {actual}"
            )


def scan_export_for_privilege(path: str | Path) -> dict[str, Any]:
    """Fail-closed audit of one materialized policy-input HDF5 file."""

    source = Path(path)
    errors: list[str] = []
    dataset_paths: list[str] = []
    group_paths: list[str] = []
    group_attribute_names: dict[str, list[str]] = {}
    dataset_attribute_names: dict[str, list[str]] = {}
    virtual_dataset_paths: list[str] = []
    external_link_paths: list[str] = []
    external_storage_dataset_paths: list[str] = []

    with h5py.File(source, "r") as handle:

        def inspect_link(
            name: str,
            link: h5py.HardLink | h5py.SoftLink | h5py.ExternalLink,
        ) -> None:
            if isinstance(link, h5py.ExternalLink):
                external_link_paths.append(name)
                errors.append(f"external_link:{name}")

        handle.visititems_links(inspect_link)

        def inspect(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Group):
                group_paths.append(name)
                if name not in ALLOWED_EXPORT_GROUPS:
                    errors.append(f"unexpected_group:{name}")
                attr_names = sorted(str(key) for key in obj.attrs)
                group_attribute_names[name] = attr_names
                if name != "metadata":
                    allowed_attrs = (
                        ALLOWED_CONDITIONS_GROUP_ATTRIBUTES
                        if name == "conditions"
                        else frozenset()
                    )
                    unexpected_attrs = sorted(
                        set(attr_names) - set(allowed_attrs)
                    )
                    for attr_name in unexpected_attrs:
                        errors.append(
                            f"unexpected_group_attr:{name}:{attr_name}"
                        )
                    for attr_name in attr_names:
                        if (
                            attr_name not in allowed_attrs
                            and _contains_privileged_token(attr_name)
                        ):
                            errors.append(
                                f"privileged_group_attr:{name}:{attr_name}"
                            )
            elif isinstance(obj, h5py.Dataset):
                dataset_paths.append(name)
                if obj.is_virtual:
                    virtual_dataset_paths.append(name)
                    errors.append(f"virtual_dataset:{name}")
                if obj.external:
                    external_storage_dataset_paths.append(name)
                    errors.append(f"external_storage:{name}")
                if name not in ALLOWED_EXPORT_DATASETS:
                    errors.append(f"unexpected_dataset:{name}")
                attr_names = sorted(str(key) for key in obj.attrs)
                dataset_attribute_names[name] = attr_names
                unexpected_attrs = sorted(
                    set(attr_names) - set(ALLOWED_DATASET_ATTRIBUTES)
                )
                for attr_name in unexpected_attrs:
                    errors.append(f"unexpected_dataset_attr:{name}:{attr_name}")
                for attr_name in attr_names:
                    if (
                        attr_name not in ALLOWED_DATASET_ATTRIBUTES
                        and _contains_privileged_token(attr_name)
                    ):
                        errors.append(f"privileged_dataset_attr:{name}:{attr_name}")

            if _contains_privileged_token(name):
                errors.append(f"privileged_name:{name}")

        handle.visititems(inspect)

        root_attrs = sorted(str(key) for key in handle.attrs)
        for key in sorted(set(root_attrs) - set(ALLOWED_ROOT_ATTRIBUTES)):
            errors.append(f"unexpected_root_attr:{key}")
        for key in root_attrs:
            if (
                key not in ALLOWED_ROOT_ATTRIBUTES
                and _contains_privileged_token(key)
            ):
                errors.append(f"privileged_root_attr:{key}")

        metadata_attrs: list[str] = []
        if "metadata" not in handle:
            errors.append("missing_group:metadata")
        else:
            metadata_attrs = sorted(str(key) for key in handle["metadata"].attrs)
            for key in sorted(
                set(metadata_attrs) - set(ALLOWED_METADATA_ATTRIBUTES)
            ):
                errors.append(f"unexpected_metadata_attr:{key}")
            for key in metadata_attrs:
                if (
                    key not in ALLOWED_METADATA_ATTRIBUTES
                    and _contains_privileged_token(key)
                ):
                    errors.append(f"privileged_metadata_attr:{key}")

        if not bool(handle.attrs.get("simverify_export", False)):
            errors.append("root_attr_simverify_export_not_true")
        if bool(handle.attrs.get("is_real", True)):
            errors.append("root_attr_is_real_not_false")

    errors = sorted(set(errors))
    return {
        "schema_version": "simverify_privilege_scan_v1",
        "path": str(source),
        "ok": not errors,
        "errors": errors,
        "group_paths": sorted(group_paths),
        "dataset_paths": sorted(dataset_paths),
        "root_attribute_names": root_attrs,
        "metadata_attribute_names": metadata_attrs,
        "group_attribute_names": group_attribute_names,
        "dataset_attribute_names": dataset_attribute_names,
        "virtual_dataset_paths": sorted(virtual_dataset_paths),
        "external_link_paths": sorted(external_link_paths),
        "external_storage_dataset_paths": sorted(
            external_storage_dataset_paths
        ),
    }


def _contains_privileged_token(name: str) -> bool:
    normalized = str(name).strip().lower()
    return any(token in normalized for token in PRIVILEGED_NAME_TOKENS)


def _decode_hdf5_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)
