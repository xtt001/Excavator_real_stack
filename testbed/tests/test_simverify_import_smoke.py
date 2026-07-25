from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from testbed.simverify.contracts import (
    CAMERA_MAPPING_ID,
    CONDITION_SCHEMA_VERSION,
    IMAGE_TRANSFORM_ID,
    POLICY_CAMERA_ORDER,
    STATE_ACTION_TIME_CONTRACT_ID,
)
from testbed.simverify.import_smoke import (
    _validate_condition_sidecar,
    _validate_episode,
)


def _annotation() -> dict[str, object]:
    return {
        "episode_id": 3,
        "cycle_id": 7,
        "split": "train",
        "quality": {"status": "accepted"},
        "policy_condition": {
            "current_sector": "left",
            "next_ready_sector": "right",
            "vector": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        "target_steps_20hz": [0, 1],
    }


def _write_episode(path: Path) -> None:
    image = np.zeros((216, 384, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    with h5py.File(path, "w") as handle:
        handle.attrs["sim"] = True
        handle.attrs["is_real"] = False
        handle.attrs["simverify_export"] = True
        metadata = handle.create_group("metadata")
        metadata.attrs["n_steps"] = 1
        metadata.attrs["camera_names"] = ",".join(POLICY_CAMERA_ORDER)
        metadata.attrs["camera_mapping_id"] = CAMERA_MAPPING_ID
        metadata.attrs["image_transform_id"] = IMAGE_TRANSFORM_ID
        metadata.attrs["image_color_space"] = "RGB"
        metadata.attrs["state_action_time_contract_id"] = (
            STATE_ACTION_TIME_CONTRACT_ID
        )
        metadata.attrs["condition_schema_version"] = CONDITION_SCHEMA_VERSION
        metadata.attrs["source_time_basis"] = (
            "timestamps/step_id * metadata.dt"
        )
        metadata.attrs["source_step_ns_used"] = False
        metadata.attrs["action_label_offset_s"] = 0.0

        observations = handle.create_group("observations")
        observations.create_dataset(
            "qpos",
            data=np.zeros((1, 4), dtype=np.float32),
        )
        observations.create_dataset(
            "qvel",
            data=np.zeros((1, 4), dtype=np.float32),
        )
        images = observations.create_group("encoded_images")
        dtype = h5py.vlen_dtype(np.dtype("uint8"))
        for camera in POLICY_CAMERA_ORDER:
            dataset = images.create_dataset(camera, shape=(1,), dtype=dtype)
            dataset[0] = encoded.reshape(-1)
            dataset.attrs["policy_camera"] = camera
            dataset.attrs["transform_id"] = IMAGE_TRANSFORM_ID
            dataset.attrs["color_space"] = "RGB"
            dataset.attrs["height"] = 216
            dataset.attrs["width"] = 384
        handle.create_dataset(
            "action",
            data=np.zeros((1, 4), dtype=np.float32),
        )
        conditions = handle.create_group("conditions")
        conditions.create_dataset(
            "cycle_condition_v1",
            data=np.asarray([[1, 0, 0, 0, 0, 1]], dtype=np.float32),
        )
        conditions.create_dataset(
            "cycle_id",
            data=np.asarray([7], dtype=np.int64),
        )
        conditions.create_dataset(
            "valid_mask",
            data=np.asarray([1], dtype=np.uint8),
        )
        timestamps = handle.create_group("timestamps")
        timestamps.create_dataset(
            "step_id",
            data=np.asarray([0], dtype=np.int64),
        )
        timestamps.create_dataset(
            "sim_time_s",
            data=np.asarray([0.0], dtype=np.float64),
        )
        diagnostics = handle.create_group("diagnostics")
        diagnostics.create_dataset(
            "source_observation_index",
            data=np.asarray([0], dtype=np.int64),
        )
        diagnostics.create_dataset(
            "source_action_index",
            data=np.asarray([0], dtype=np.int64),
        )
        diagnostics.create_dataset(
            "source_step_id",
            data=np.asarray([0], dtype=np.int64),
        )
        diagnostics.create_dataset(
            "source_sim_time_s",
            data=np.asarray([0.0], dtype=np.float64),
        )
        diagnostics.create_dataset(
            "target_tick",
            data=np.asarray([0], dtype=np.int64),
        )
        diagnostics.create_dataset(
            "target_sim_time_s",
            data=np.asarray([0.0], dtype=np.float64),
        )
        diagnostics.create_dataset(
            "selection_error_s",
            data=np.asarray([0.0], dtype=np.float64),
        )


def test_m1_episode_import_validates_alignment_and_all_camera_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episode_3.hdf5"
    _write_episode(path)

    result = _validate_episode(
        path,
        episode_id=3,
        split_name="train",
        annotations=[_annotation()],
    )

    assert result["steps"] == 1
    assert result["valid_condition_rows"] == 1
    assert result["decoded_image_count"] == 4
    assert result["source_index_alignment"] == "exact"


def test_m1_condition_sidecar_mismatch_fails_closed() -> None:
    condition = np.asarray([[1, 0, 0, 0, 0, 1]], dtype=np.float32)
    cycle_id = np.asarray([7], dtype=np.int64)
    valid = np.asarray([True])
    annotation = _annotation()
    annotation["policy_condition"] = {
        **annotation["policy_condition"],
        "vector": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }

    with pytest.raises(ValueError, match="condition does not match"):
        _validate_condition_sidecar(
            condition,
            cycle_id,
            valid,
            annotations=[annotation],
        )
