from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from testbed.simverify.contracts import (
    CAMERA_MAPPING_ID,
    POLICY_CAMERA_ORDER,
    SOURCE_TO_POLICY_CAMERA,
    camera_transform_contract,
    collect_hdf5_source_provenance,
    file_provenance,
    git_provenance,
    scan_export_for_privilege,
    sha256_file,
    state_action_time_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_camera_and_source_domain_contracts_are_explicit() -> None:
    camera = camera_transform_contract()
    state = state_action_time_contract()

    assert camera["mapping_id"] == CAMERA_MAPPING_ID
    assert camera["source_to_policy"] == SOURCE_TO_POLICY_CAMERA
    assert camera["policy_order"] == list(POLICY_CAMERA_ORDER)
    assert camera["physical_role_mapping_only"] is True
    assert camera["geometric_equivalence"] is False
    assert camera["transform"]["crop"] == "none"
    assert camera["transform"]["resize"] == {
        "width": 384,
        "height": 216,
        "filter": "linear",
    }
    assert len(camera["contract_sha256"]) == 64

    assert state["qpos"]["representation"] == "sim_source_representation"
    assert state["qpos"]["real_unit_mapping"] is None
    assert state["action"]["semantics"] == "actuator_speed_cmd"
    assert state["time"]["source_time_basis"] == (
        "timestamps/step_id * metadata.dt"
    )
    assert state["time"]["wall_clock_step_ns_used"] is False
    assert state["time"]["same_source_row_for_all_fields"] is True
    assert state["time"]["action_label_offset_s"] == 0.0
    assert (
        state["checkpoint_restriction"]
        == "sim_state_domain_only_not_real_deployable"
    )


def test_file_and_git_provenance_are_recomputable(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"simverify-provenance")

    record = file_provenance(artifact)

    assert record["path"] == str(artifact.resolve())
    assert record["size_bytes"] == len(b"simverify-provenance")
    assert record["sha256"] == sha256_file(artifact)
    assert len(record["sha256"]) == 64

    git = git_provenance(REPO_ROOT)
    assert git["git_available"] is True
    assert len(git["commit"]) == 40
    assert git["branch"] == "v2.0.0-simVerify"
    assert isinstance(git["dirty"], bool)


def test_privilege_scanner_accepts_only_allowlisted_observations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clean.hdf5"
    _write_minimal_export(path)

    report = scan_export_for_privilege(path)

    assert report["ok"] is True
    assert report["errors"] == []
    assert "observations/env_state" not in report["dataset_paths"]
    assert "timestamps/step_ns" not in report["dataset_paths"]


def test_privilege_scanner_rejects_privilege_and_unknown_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "contaminated.hdf5"
    _write_minimal_export(path)
    with h5py.File(path, "a") as handle:
        handle["observations"].create_dataset(
            "env_state", data=np.zeros((1, 2), dtype=np.float32)
        )
        handle["timestamps"].create_dataset(
            "step_ns", data=np.zeros(1, dtype=np.int64)
        )
        handle["metadata"].attrs["planner_goal"] = "forbidden"

    report = scan_export_for_privilege(path)

    assert report["ok"] is False
    assert "unexpected_dataset:observations/env_state" in report["errors"]
    assert "privileged_name:observations/env_state" in report["errors"]
    assert "unexpected_dataset:timestamps/step_ns" in report["errors"]
    assert "unexpected_metadata_attr:planner_goal" in report["errors"]
    assert "privileged_metadata_attr:planner_goal" in report["errors"]


def test_source_provenance_rejects_missing_relative_vds_backing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing_backing_vds.hdf5"
    layout = h5py.VirtualLayout(shape=(2,), dtype=np.float32)
    layout[:] = h5py.VirtualSource(
        "missing_relative_backing.hdf5",
        "data",
        shape=(2,),
    )
    with h5py.File(path, "w") as handle:
        handle.create_virtual_dataset("data", layout)

    with pytest.raises(
        FileNotFoundError,
        match="missing_relative_backing.hdf5",
    ):
        collect_hdf5_source_provenance(path)


def test_source_provenance_hashes_valid_recursive_vds_backings(
    tmp_path: Path,
) -> None:
    leaf = tmp_path / "leaf.hdf5"
    with h5py.File(leaf, "w") as handle:
        handle.create_dataset(
            "data",
            data=np.asarray([1.0, 2.0], dtype=np.float32),
        )

    middle = tmp_path / "middle.hdf5"
    middle_layout = h5py.VirtualLayout(shape=(2,), dtype=np.float32)
    middle_layout[:] = h5py.VirtualSource(
        leaf.name,
        "data",
        shape=(2,),
    )
    with h5py.File(middle, "w") as handle:
        handle.create_virtual_dataset("data", middle_layout)

    root = tmp_path / "root.hdf5"
    root_layout = h5py.VirtualLayout(shape=(2,), dtype=np.float32)
    root_layout[:] = h5py.VirtualSource(
        middle.name,
        "data",
        shape=(2,),
    )
    with h5py.File(root, "w") as handle:
        handle.create_virtual_dataset("data", root_layout)

    records = collect_hdf5_source_provenance(root)

    by_path = {Path(record["path"]): record for record in records}
    assert set(by_path) == {
        root.resolve(),
        middle.resolve(),
        leaf.resolve(),
    }
    for source_path, record in by_path.items():
        assert record["sha256"] == sha256_file(source_path)


def test_privilege_scanner_rejects_vds_output(tmp_path: Path) -> None:
    backing = tmp_path / "qpos_backing.hdf5"
    with h5py.File(backing, "w") as handle:
        handle.create_dataset(
            "qpos",
            data=np.zeros((1, 4), dtype=np.float32),
        )

    path = tmp_path / "vds_export.hdf5"
    _write_minimal_export(path)
    layout = h5py.VirtualLayout(shape=(1, 4), dtype=np.float32)
    layout[:] = h5py.VirtualSource(
        backing.name,
        "qpos",
        shape=(1, 4),
    )
    with h5py.File(path, "a") as handle:
        del handle["observations/qpos"]
        handle["observations"].create_virtual_dataset("qpos", layout)

    report = scan_export_for_privilege(path)

    assert report["ok"] is False
    assert "virtual_dataset:observations/qpos" in report["errors"]
    assert report["virtual_dataset_paths"] == ["observations/qpos"]


def test_privilege_scanner_rejects_external_link(tmp_path: Path) -> None:
    backing = tmp_path / "external_qpos.hdf5"
    with h5py.File(backing, "w") as handle:
        handle.create_dataset(
            "qpos",
            data=np.zeros((1, 4), dtype=np.float32),
        )

    path = tmp_path / "external_link_export.hdf5"
    _write_minimal_export(path)
    with h5py.File(path, "a") as handle:
        del handle["observations/qpos"]
        handle["observations/qpos"] = h5py.ExternalLink(
            str(backing),
            "/qpos",
        )

    report = scan_export_for_privilege(path)

    assert report["ok"] is False
    assert "external_link:observations/qpos" in report["errors"]
    assert report["external_link_paths"] == ["observations/qpos"]


def test_privilege_scanner_rejects_external_dataset_storage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "external_storage_export.hdf5"
    _write_minimal_export(path)
    with h5py.File(path, "a") as handle:
        del handle["action"]
        handle.create_dataset(
            "action",
            shape=(1, 4),
            dtype=np.float32,
            external=[
                (
                    "external_action.raw",
                    0,
                    h5py.h5f.UNLIMITED,
                )
            ],
        )

    report = scan_export_for_privilege(path)

    assert report["ok"] is False
    assert "external_storage:action" in report["errors"]
    assert report["external_storage_dataset_paths"] == ["action"]


def _write_minimal_export(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["sim"] = True
        handle.attrs["is_real"] = False
        handle.attrs["simverify_export"] = True
        metadata = handle.create_group("metadata")
        metadata.attrs["schema_version"] = (
            "sim_observable_cycle_export_episode_v1"
        )
        observations = handle.create_group("observations")
        observations.create_dataset(
            "qpos", data=np.zeros((1, 4), dtype=np.float32)
        )
        observations.create_dataset(
            "qvel", data=np.zeros((1, 4), dtype=np.float32)
        )
        images = observations.create_group("encoded_images")
        dtype = h5py.vlen_dtype(np.dtype("uint8"))
        for camera in POLICY_CAMERA_ORDER:
            images.create_dataset(camera, shape=(1,), dtype=dtype)[0] = (
                np.asarray([1, 2, 3], dtype=np.uint8)
            )
        conditions = handle.create_group("conditions")
        conditions.attrs["schema_id"] = "cycle_condition_v1"
        conditions.attrs["dim"] = 6
        conditions.attrs["encoding"] = (
            "current_sector_one_hot_3_plus_next_sector_one_hot_3"
        )
        conditions.attrs["normalization"] = "none_binary_one_hot"
        conditions.attrs["source"] = "hindsight_outcome_pending"
        conditions.attrs["scope"] = "constant_within_observable_cycle"
        condition = conditions.create_dataset(
            "cycle_condition_v1", data=np.zeros((1, 6), dtype=np.float32)
        )
        condition.attrs["schema_id"] = "cycle_condition_v1"
        condition.attrs["dim"] = 6
        condition.attrs["normalization"] = "none_binary_one_hot"
        condition.attrs["source"] = "hindsight_outcome_pending"
        condition.attrs["scope"] = "constant_within_observable_cycle"
        conditions.create_dataset(
            "cycle_id", data=np.full(1, -1, dtype=np.int64)
        )
        conditions.create_dataset(
            "valid_mask", data=np.zeros(1, dtype=np.uint8)
        )
        handle.create_dataset(
            "action", data=np.zeros((1, 4), dtype=np.float32)
        )
        timestamps = handle.create_group("timestamps")
        timestamps.create_dataset(
            "step_id", data=np.zeros(1, dtype=np.int64)
        )
        timestamps.create_dataset(
            "sim_time_s", data=np.zeros(1, dtype=np.float64)
        )
        diagnostics = handle.create_group("diagnostics")
        diagnostics.create_dataset(
            "source_observation_index", data=np.zeros(1, dtype=np.int64)
        )
        diagnostics.create_dataset(
            "source_action_index", data=np.zeros(1, dtype=np.int64)
        )
        diagnostics.create_dataset(
            "source_step_id", data=np.zeros(1, dtype=np.int64)
        )
        diagnostics.create_dataset(
            "source_sim_time_s", data=np.zeros(1, dtype=np.float64)
        )
        diagnostics.create_dataset(
            "target_tick", data=np.zeros(1, dtype=np.int64)
        )
        diagnostics.create_dataset(
            "target_sim_time_s", data=np.zeros(1, dtype=np.float64)
        )
        diagnostics.create_dataset(
            "selection_error_s", data=np.zeros(1, dtype=np.float64)
        )
