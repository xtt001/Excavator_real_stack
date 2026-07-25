"""Bounded M1 import smoke for immutable SimVerify M0 packages."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from testbed.simverify.artifacts import write_json
from testbed.simverify.contracts import (
    ALLOWED_EXPORT_DATASETS,
    CAMERA_MAPPING_ID,
    CONDITION_SCHEMA_VERSION,
    IMAGE_TRANSFORM_ID,
    POLICY_CAMERA_ORDER,
    STATE_ACTION_TIME_CONTRACT_ID,
    git_provenance,
    scan_export_for_privilege,
    sha256_file,
)
from testbed.simverify.gates import validate_condition_materialization

REQUIRED_PACKAGE_FILES = (
    "dataset_manifest.json",
    "split_groups.json",
    "camera_mapping.json",
    "state_action_contract.json",
    "cycle_condition_v1.schema.json",
    "cycle_annotations.jsonl",
    "privilege_scan_report.json",
    "m0_authorization_report_v2.json",
)


def run_m1_import_smoke(
    package_root: str | Path,
    *,
    output_path: str | Path,
    repo_root: str | Path,
    train_episode_id: int | None = None,
    validation_episode_id: int | None = None,
) -> dict[str, Any]:
    """Validate one train and one validation export without invoking ACT."""

    root = Path(package_root).resolve(strict=True)
    if root.name.startswith("."):
        raise ValueError("M1 refuses hidden or temporary M0 package roots")
    output = Path(output_path).resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"immutable M1 report already exists: {output}")
    repository = Path(repo_root).resolve(strict=True)
    code_git = git_provenance(repository)
    if (
        not bool(code_git.get("git_available"))
        or code_git.get("branch") != "v2.0.0-simVerify"
        or bool(code_git.get("dirty"))
    ):
        raise ValueError(
            "M1 requires a clean v2.0.0-simVerify Real Stack worktree"
        )

    payloads = {
        name: _read_json(root / name)
        for name in REQUIRED_PACKAGE_FILES
        if name.endswith(".json")
    }
    manifest = payloads["dataset_manifest.json"]
    split = payloads["split_groups.json"]
    _require_m0_authorization(manifest, payloads)
    _require_camera_and_time_contracts(payloads)

    train_id = _select_episode(
        split,
        "train",
        requested=train_episode_id,
    )
    validation_id = _select_episode(
        split,
        "validation",
        requested=validation_episode_id,
    )
    if train_id == validation_id:
        raise ValueError("M1 train and validation episode selections overlap")

    checksum_inventory = _checksum_inventory(root / "checksums.sha256")
    selected = (("train", train_id), ("validation", validation_id))
    required_relatives = set(REQUIRED_PACKAGE_FILES)
    required_relatives.update(
        f"episodes/episode_{episode_id}.hdf5"
        for _split_name, episode_id in selected
    )
    verified_inputs = _verify_bounded_checksums(
        root,
        checksum_inventory,
        required_relatives,
    )
    annotations = _read_selected_annotations(
        root / "cycle_annotations.jsonl",
        {train_id, validation_id},
    )

    episode_results = []
    for split_name, episode_id in selected:
        path = root / f"episodes/episode_{episode_id}.hdf5"
        rows = annotations.get(episode_id, [])
        if not rows:
            raise ValueError(f"no annotation sidecar rows for episode {episode_id}")
        if any(str(row["split"]) != split_name for row in rows):
            raise ValueError(f"annotation split leakage for episode {episode_id}")
        episode_results.append(
            _validate_episode(
                path,
                episode_id=episode_id,
                split_name=split_name,
                annotations=rows,
            )
        )

    report = {
        "schema": "simverify_m1_import_smoke_v1",
        "stage": "M1",
        "status": "passed",
        "evidence_scope": "recorded-observation/offline",
        "passed": True,
        "package_root": str(root),
        "package_dataset_manifest_sha256": sha256_file(
            root / "dataset_manifest.json"
        ),
        "checksum_inventory_sha256": sha256_file(root / "checksums.sha256"),
        "code_git": code_git,
        "bounded_verified_inputs": verified_inputs,
        "episodes": episode_results,
        "imports": {
            "pact_package_imported": False,
            "simulator_backend_imported": False,
            "act_invoked": False,
        },
        "source_paths_opened": False,
        "oracle_audit_read": False,
        "closed_loop_execution": False,
        "training_started": False,
        "training_authorized": False,
        "held_out_test_read": False,
        "m2_authorized": True,
        "capability_boundary": (
            "M1 proves only that two bounded recorded-observation exports can "
            "be consumed by the Real Stack import boundary."
        ),
    }
    identity = write_json(output, report)
    return {
        **report,
        "report_path": str(output),
        "report_sha256": identity["sha256"],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_m0_authorization(
    manifest: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    if manifest.get("status") != "m0_artifacts_frozen_m1_import_smoke_pending":
        raise ValueError("M0 package status does not authorize M1 import smoke")
    if bool(manifest.get("training_authorized")):
        raise ValueError("M0 package unexpectedly authorizes training")
    if bool(manifest.get("held_out_test_authorized")):
        raise ValueError("M0 package unexpectedly authorizes held-out access")
    if bool(manifest.get("oracle_dependency")):
        raise ValueError("M0 package declares an oracle dependency")
    privilege = payloads["privilege_scan_report.json"]
    if not bool(privilege.get("ok")) or bool(privilege.get("oracle_dependency")):
        raise ValueError("M0 privilege scan did not pass")
    authorization = payloads["m0_authorization_report_v2.json"]
    if not bool(authorization.get("gate_preconditions_passed")):
        raise ValueError("M0 Gate preconditions did not pass")
    if not bool(
        authorization.get("m1_import_smoke_authorized_after_immutable_finalize")
    ):
        raise ValueError("M0 authorization report does not permit M1")


def _require_camera_and_time_contracts(
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    camera = payloads["camera_mapping.json"]
    if camera.get("mapping_id") != CAMERA_MAPPING_ID:
        raise ValueError("camera mapping ID mismatch")
    if tuple(camera.get("policy_order", ())) != POLICY_CAMERA_ORDER:
        raise ValueError("policy camera order mismatch")
    if camera.get("transform", {}).get("transform_id") != IMAGE_TRANSFORM_ID:
        raise ValueError("image transform ID is pending or mismatched")
    if camera.get("transform", {}).get("output_color_space") != "RGB":
        raise ValueError("camera output color space is not RGB")
    state = payloads["state_action_contract.json"]
    if state.get("contract_id") != STATE_ACTION_TIME_CONTRACT_ID:
        raise ValueError("state/action/time contract ID mismatch")
    if state.get("time", {}).get("source_time_basis") != (
        "timestamps/step_id * metadata.dt"
    ):
        raise ValueError("source time basis is not simulator step_id times dt")
    if bool(state.get("time", {}).get("wall_clock_step_ns_used")):
        raise ValueError("wall-clock time is enabled")
    if float(state.get("action", {}).get("label_offset_s", float("nan"))) != 0.0:
        raise ValueError("action label offset is non-zero")
    condition = payloads["cycle_condition_v1.schema.json"]
    if condition.get("schema_id") != CONDITION_SCHEMA_VERSION:
        raise ValueError("condition schema mismatch")


def _select_episode(
    split: Mapping[str, Any],
    split_name: str,
    *,
    requested: int | None,
) -> int:
    values = list(map(int, split["splits"][split_name]))
    if not values:
        raise ValueError(f"split {split_name} is empty")
    selected = values[0] if requested is None else int(requested)
    if selected not in values:
        raise ValueError(f"episode {selected} does not belong to {split_name}")
    held_out = set(map(int, split["splits"]["held_out_test"]))
    if selected in held_out:
        raise ValueError("M1 selection leaks into held-out test")
    return selected


def _checksum_inventory(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        expected, relative = raw.split("  ", 1)
        if relative in result:
            raise ValueError(f"duplicate checksum path: {relative}")
        result[relative] = expected
    return result


def _verify_bounded_checksums(
    root: Path,
    inventory: Mapping[str, str],
    relatives: set[str],
) -> list[dict[str, Any]]:
    verified = []
    for relative in sorted(relatives):
        if relative not in inventory:
            raise ValueError(f"required M1 input absent from checksums: {relative}")
        target = (root / relative).resolve(strict=True)
        target.relative_to(root)
        actual = sha256_file(target)
        expected = str(inventory[relative])
        if actual != expected:
            raise ValueError(f"checksum mismatch for {relative}")
        verified.append(
            {
                "path": relative,
                "size_bytes": int(target.stat().st_size),
                "sha256": actual,
            }
        )
    return verified


def _read_selected_annotations(
    path: Path,
    episode_ids: set[int],
) -> dict[int, list[dict[str, Any]]]:
    result = {episode_id: [] for episode_id in episode_ids}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            episode_id = int(row["episode_id"])
            if episode_id in result:
                result[episode_id].append(row)
    return result


def _validate_episode(
    path: Path,
    *,
    episode_id: int,
    split_name: str,
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    privilege = scan_export_for_privilege(path)
    if not bool(privilege["ok"]):
        raise ValueError(f"privilege scan failed for episode {episode_id}")
    with h5py.File(path, "r") as handle:
        datasets: set[str] = set()
        external_or_virtual: list[str] = []

        def visit(name: str, value: h5py.Group | h5py.Dataset) -> None:
            if not isinstance(value, h5py.Dataset):
                return
            datasets.add(name)
            if value.external is not None or value.is_virtual:
                external_or_virtual.append(name)

        handle.visititems(visit)
        if datasets != set(ALLOWED_EXPORT_DATASETS):
            raise ValueError(f"export dataset allowlist mismatch for episode {episode_id}")
        if external_or_virtual:
            raise ValueError(
                f"external or virtual datasets in episode {episode_id}: "
                f"{external_or_virtual}"
            )
        metadata = handle["metadata"].attrs
        _validate_episode_metadata(metadata, episode_id)
        count = int(metadata["n_steps"])
        qpos = np.asarray(handle["observations/qpos"], dtype=np.float32)
        qvel = np.asarray(handle["observations/qvel"], dtype=np.float32)
        action = np.asarray(handle["action"], dtype=np.float32)
        for name, values in (("qpos", qpos), ("qvel", qvel), ("action", action)):
            if values.shape != (count, 4) or values.dtype != np.float32:
                raise ValueError(f"{name} shape/dtype mismatch in episode {episode_id}")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")
        source_observation = np.asarray(
            handle["diagnostics/source_observation_index"],
            dtype=np.int64,
        )
        source_action = np.asarray(
            handle["diagnostics/source_action_index"],
            dtype=np.int64,
        )
        if not np.array_equal(source_observation, source_action):
            raise ValueError("observation/action source-index mismatch")
        if np.any(np.diff(source_observation) < 0):
            raise ValueError("source indices are not monotonic")
        selection_error = np.asarray(
            handle["diagnostics/selection_error_s"],
            dtype=np.float64,
        )
        if not np.isfinite(selection_error).all():
            raise ValueError("selection error contains non-finite values")
        condition = np.asarray(handle["conditions/cycle_condition_v1"])
        cycle_id = np.asarray(handle["conditions/cycle_id"])
        valid = np.asarray(handle["conditions/valid_mask"], dtype=bool)
        validate_condition_materialization(condition, cycle_id, valid)
        _validate_condition_sidecar(
            condition,
            cycle_id,
            valid,
            annotations=annotations,
        )
        decoded_count = _validate_all_images(handle, count)
    return {
        "episode_id": episode_id,
        "split": split_name,
        "path": f"episodes/{path.name}",
        "steps": count,
        "valid_condition_rows": int(valid.sum()),
        "annotation_cycle_count": len(annotations),
        "decoded_image_count": decoded_count,
        "source_index_alignment": "exact",
        "external_or_virtual_dataset_count": 0,
        "privilege_scan_ok": True,
    }


def _validate_episode_metadata(
    metadata: h5py.AttributeManager,
    episode_id: int,
) -> None:
    required = {
        "camera_names": ",".join(POLICY_CAMERA_ORDER),
        "camera_mapping_id": CAMERA_MAPPING_ID,
        "image_transform_id": IMAGE_TRANSFORM_ID,
        "image_color_space": "RGB",
        "state_action_time_contract_id": STATE_ACTION_TIME_CONTRACT_ID,
        "condition_schema_version": CONDITION_SCHEMA_VERSION,
        "source_time_basis": "timestamps/step_id * metadata.dt",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"metadata {key} mismatch in episode {episode_id}")
    if bool(metadata.get("source_step_ns_used")):
        raise ValueError("episode metadata enables wall-clock source time")
    if float(metadata.get("action_label_offset_s", float("nan"))) != 0.0:
        raise ValueError("episode action offset is non-zero")


def _validate_condition_sidecar(
    condition: np.ndarray,
    cycle_id: np.ndarray,
    valid: np.ndarray,
    *,
    annotations: Sequence[Mapping[str, Any]],
) -> None:
    expected_condition = np.zeros_like(condition, dtype=np.float32)
    expected_cycle = np.full(cycle_id.shape, -1, dtype=np.int64)
    expected_valid = np.zeros(valid.shape, dtype=bool)
    for row in annotations:
        vector = row["policy_condition"]["vector"]
        if row["quality"]["status"] != "accepted":
            if vector is not None:
                raise ValueError("non-accepted sidecar cycle carries a condition")
            continue
        if vector is None:
            raise ValueError("accepted sidecar cycle has no condition")
        start, end = map(int, row["target_steps_20hz"])
        if not 0 <= start <= end <= condition.shape[0]:
            raise ValueError("annotation target interval lies outside episode")
        if np.any(expected_valid[start:end]):
            raise ValueError("accepted annotation intervals overlap")
        expected_condition[start:end] = np.asarray(vector, dtype=np.float32)
        expected_cycle[start:end] = int(row["cycle_id"])
        expected_valid[start:end] = True
    if not np.array_equal(valid, expected_valid):
        raise ValueError("condition valid mask does not match annotation sidecar")
    if not np.array_equal(cycle_id, expected_cycle):
        raise ValueError("condition cycle IDs do not match annotation sidecar")
    if not np.array_equal(condition, expected_condition):
        raise ValueError("materialized condition does not match annotation sidecar")
    for value in np.unique(cycle_id[valid]):
        rows = condition[cycle_id == value]
        if rows.size and not np.all(rows == rows[0]):
            raise ValueError(f"condition is not constant within cycle {value}")


def _validate_all_images(handle: h5py.File, count: int) -> int:
    decoded_count = 0
    for camera in POLICY_CAMERA_ORDER:
        dataset = handle[f"observations/encoded_images/{camera}"]
        attrs = dataset.attrs
        if (
            attrs.get("policy_camera") != camera
            or attrs.get("transform_id") != IMAGE_TRANSFORM_ID
            or attrs.get("color_space") != "RGB"
            or int(attrs.get("height", -1)) != 216
            or int(attrs.get("width", -1)) != 384
        ):
            raise ValueError(f"camera dataset contract mismatch for {camera}")
        if dataset.shape != (count,):
            raise ValueError(f"camera row count mismatch for {camera}")
        for payload in dataset:
            encoded = np.asarray(payload, dtype=np.uint8)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape != (216, 384, 3):
                raise ValueError(f"invalid JPEG shape for {camera}")
            _rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            decoded_count += 1
    return decoded_count
