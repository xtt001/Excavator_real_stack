from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from testbed.simverify.contracts import (
    POLICY_CAMERA_ORDER,
    SOURCE_CAMERA_ORDER,
    scan_export_for_privilege,
    sha256_file,
)
from testbed.simverify.export import (
    materialize_sim_episode,
    select_sim_time_indices,
    transition_preservation_qc,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sim_time_selector_uses_first_non_earlier_unique_source_row() -> None:
    selection = select_sim_time_indices(
        np.arange(11, dtype=np.int64),
        source_dt_s=0.02,
    )

    np.testing.assert_array_equal(
        selection.source_indices,
        np.asarray([0, 3, 5, 8, 10], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        selection.source_step_ids,
        np.asarray([0, 3, 5, 8, 10], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        selection.target_ticks,
        np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
    )
    np.testing.assert_allclose(
        selection.selection_error_s,
        np.asarray([0.0, 0.01, 0.0, 0.01, 0.0]),
        atol=1e-9,
    )
    assert np.all(selection.source_sim_time_s >= selection.target_sim_time_s)


@pytest.mark.parametrize(
    ("step_ids", "source_dt_s", "match"),
    [
        (np.asarray([0, 1, 1], dtype=np.int64), 0.02, "strictly increasing"),
        (np.asarray([0, 1, 2], dtype=np.float32), 0.02, "integer dtype"),
        (np.asarray([0, 1, 2], dtype=np.int64), 0.01, "frozen 50 Hz"),
        (np.asarray([0, 10, 11], dtype=np.int64), 0.02, "cadence gap"),
    ],
)
def test_sim_time_selector_fails_closed_on_invalid_time_contract(
    step_ids: np.ndarray,
    source_dt_s: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        select_sim_time_indices(step_ids, source_dt_s=source_dt_s)


def test_transition_qc_reports_short_miss_and_durable_preservation() -> None:
    actions = np.zeros((11, 4), dtype=np.float32)
    actions[1, 0] = 0.8
    actions[3:6, 1] = -0.7
    selected = np.asarray([0, 3, 5, 8, 10], dtype=np.int64)

    report = transition_preservation_qc(
        actions,
        step_ids=np.arange(11, dtype=np.int64),
        source_dt_s=0.02,
        selected_indices=selected,
        deadzone=0.05,
    )

    assert report["valid_segment_count"] == 2
    assert report["preserved_segment_count"] == 1
    assert report["missing_segment_count"] == 1
    assert report["durable_segment_count"] == 1
    assert report["durable_missing_segment_count"] == 0
    assert report["all_missing_segments_shorter_than_durable_min"] is True
    assert report["missing_segments"][0]["axis"] == "swing"
    assert report["missing_segments"][0]["duration_s"] == pytest.approx(0.02)


def test_materializer_uses_same_row_and_strips_all_privilege(
    tmp_path: Path,
) -> None:
    source = tmp_path / "episode_28.hdf5"
    output = tmp_path / "derived" / "episode_28.hdf5"
    _write_source_episode(source, n_steps=11)
    source_sha_before = sha256_file(source)

    result = materialize_sim_episode(
        source,
        output,
        repo_root=REPO_ROOT,
        chunk_size=2,
    )

    assert sha256_file(source) == source_sha_before
    assert result["source_steps"] == 11
    assert result["output_steps"] == 5
    assert result["privilege_scan"]["ok"] is True
    with h5py.File(source, "r") as source_h5, h5py.File(output, "r") as out:
        source_index = out["diagnostics/source_observation_index"][()]
        action_index = out["diagnostics/source_action_index"][()]
        np.testing.assert_array_equal(source_index, [0, 3, 5, 8, 10])
        np.testing.assert_array_equal(action_index, source_index)
        np.testing.assert_allclose(
            out["observations/qpos"][()],
            source_h5["observations/qpos"][source_index],
        )
        np.testing.assert_allclose(
            out["observations/qvel"][()],
            source_h5["observations/qvel"][source_index],
        )
        np.testing.assert_allclose(
            out["action"][()],
            source_h5["action"][source_index],
        )
        assert out["metadata"].attrs["action_label_offset_s"] == 0.0
        assert bool(out["metadata"].attrs["action_prealigned"]) is True
        assert out["metadata"].attrs["command_source"] == "unknown_not_recorded"
        assert (
            out["metadata"].attrs["condition_status"]
            == "placeholder_unlabeled"
        )
        np.testing.assert_array_equal(
            out["conditions/cycle_condition_v1"][()],
            np.zeros((5, 6), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            out["conditions/cycle_id"][()],
            np.full(5, -1, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            out["conditions/valid_mask"][()],
            np.zeros(5, dtype=np.uint8),
        )
        condition_attrs = out["conditions/cycle_condition_v1"].attrs
        assert condition_attrs["schema_id"] == "cycle_condition_v1"
        assert condition_attrs["dim"] == 6
        assert condition_attrs["normalization"] == "none_binary_one_hot"
        assert condition_attrs["source"] == "hindsight_outcome_pending"
        assert condition_attrs["scope"] == "constant_within_observable_cycle"
        assert "observations/env_state" not in out
        assert "v2" not in out
        assert "rewards" not in out
        assert "timestamps/step_ns" not in out
        assert sorted(out["observations/encoded_images"].keys()) == sorted(
            POLICY_CAMERA_ORDER
        )
        for camera in POLICY_CAMERA_ORDER:
            encoded = np.asarray(
                out[f"observations/encoded_images/{camera}"][0],
                dtype=np.uint8,
            )
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            assert decoded is not None
            assert decoded.shape == (216, 384, 3)

    assert scan_export_for_privilege(output)["ok"] is True


def test_materializer_selects_source_aligned_cycle_condition_and_cycle_id(
    tmp_path: Path,
) -> None:
    source = tmp_path / "episode_conditioned.hdf5"
    output = tmp_path / "conditioned_20hz.hdf5"
    _write_source_episode(source, n_steps=11)
    condition = np.zeros((11, 6), dtype=np.float32)
    condition[:, 0] = 1.0
    condition[:, 4] = 1.0
    cycle_id = np.arange(11, dtype=np.int64) + 100
    valid = np.ones(11, dtype=np.uint8)

    result = materialize_sim_episode(
        source,
        output,
        repo_root=REPO_ROOT,
        condition_rows=condition,
        condition_cycle_id=cycle_id,
        condition_valid=valid,
        condition_materialized_from_sha256="a" * 64,
        condition_schema_sha256="B" * 64,
        chunk_size=2,
    )

    selected = np.asarray([0, 3, 5, 8, 10], dtype=np.int64)
    assert result["condition_status"] == "labeled"
    with h5py.File(output, "r") as out:
        np.testing.assert_array_equal(
            out["conditions/cycle_condition_v1"][()],
            condition[selected],
        )
        np.testing.assert_array_equal(
            out["conditions/cycle_id"][()],
            cycle_id[selected],
        )
        np.testing.assert_array_equal(
            out["conditions/valid_mask"][()],
            valid[selected],
        )
        assert out["metadata"].attrs["condition_source"] == "hindsight_outcome"
        assert (
            out["conditions/cycle_condition_v1"].attrs["source"]
            == "hindsight_outcome"
        )
        condition_group_attrs = out["conditions"].attrs
        assert condition_group_attrs["schema_id"] == "cycle_condition_v1"
        assert condition_group_attrs["dim"] == 6
        assert condition_group_attrs["encoding"] == (
            "current_sector_one_hot_3_plus_next_sector_one_hot_3"
        )
        assert condition_group_attrs["normalization"] == (
            "none_binary_one_hot"
        )
        assert condition_group_attrs["source"] == "hindsight_outcome"
        assert condition_group_attrs["scope"] == (
            "constant_within_observable_cycle"
        )
        assert condition_group_attrs["materialized_from_sha256"] == "a" * 64
        assert condition_group_attrs["schema_sha256"] == "b" * 64


def test_materializer_rejects_non_placeholder_invalid_condition_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "episode_bad_condition.hdf5"
    output = tmp_path / "must_not_exist.hdf5"
    _write_source_episode(source, n_steps=11)
    condition = np.zeros((11, 6), dtype=np.float32)
    condition[:, 0] = 1.0
    condition[:, 4] = 1.0
    cycle_id = np.arange(11, dtype=np.int64)
    valid = np.ones(11, dtype=np.uint8)
    valid[1] = 0

    with pytest.raises(ValueError, match="all-zero placeholder"):
        materialize_sim_episode(
            source,
            output,
            repo_root=REPO_ROOT,
            condition_rows=condition,
            condition_cycle_id=cycle_id,
            condition_valid=valid,
        )

    assert not output.exists()


def test_materializer_never_overwrites_an_existing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "episode_1.hdf5"
    output = tmp_path / "output.hdf5"
    _write_source_episode(source, n_steps=11)
    output.write_bytes(b"do-not-replace")
    output_sha = sha256_file(output)
    source_sha = sha256_file(source)

    with pytest.raises(FileExistsError, match="already exists"):
        materialize_sim_episode(source, output, repo_root=REPO_ROOT)

    assert sha256_file(output) == output_sha
    assert sha256_file(source) == source_sha


def _write_source_episode(path: Path, *, n_steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _jpeg_frame()
    with h5py.File(path, "w") as handle:
        handle.attrs["sim"] = True
        metadata = handle.create_group("metadata")
        metadata.attrs["episode_id"] = path.stem
        metadata.attrs["dt"] = np.float32(0.02)
        metadata.attrs["control_hz"] = 50
        metadata.attrs["camera_names"] = ",".join(SOURCE_CAMERA_ORDER)
        metadata.attrs["qpos_order"] = (
            "swing_position_norm,boom_position_norm,"
            "stick_position_norm,bucket_position_norm"
        )
        metadata.attrs["qvel_order"] = (
            "swing_speed,boom_speed,stick_speed,bucket_speed"
        )
        metadata.attrs["action_order"] = (
            "swing_speed_cmd,boom_speed_cmd,stick_speed_cmd,bucket_speed_cmd"
        )
        metadata.attrs["action_semantics"] = "actuator_speed_cmd"
        metadata.attrs["deadzone"] = np.full(4, 0.05, dtype=np.float32)
        metadata.attrs["env_state_contract_version"] = "forbidden_source_only"

        observations = handle.create_group("observations")
        values = np.arange(n_steps * 4, dtype=np.float32).reshape(n_steps, 4)
        observations.create_dataset("qpos", data=values)
        observations.create_dataset("qvel", data=values + 100.0)
        observations.create_dataset(
            "env_state", data=np.ones((n_steps, 8), dtype=np.float32)
        )
        encoded_images = observations.create_group("encoded_images")
        dtype = h5py.vlen_dtype(np.dtype("uint8"))
        for camera in SOURCE_CAMERA_ORDER:
            dataset = encoded_images.create_dataset(
                camera, shape=(n_steps,), dtype=dtype
            )
            for index in range(n_steps):
                dataset[index] = frame

        action = np.zeros((n_steps, 4), dtype=np.float32)
        action[1, 0] = 0.8
        action[3:6, 1] = -0.7
        handle.create_dataset("action", data=action)
        handle.create_dataset(
            "rewards", data=np.ones(n_steps, dtype=np.float32)
        )

        timestamps = handle.create_group("timestamps")
        timestamps.create_dataset(
            "step_id", data=np.arange(n_steps, dtype=np.int64)
        )
        timestamps.create_dataset(
            "step_ns",
            data=np.arange(n_steps, dtype=np.int64) * 35_000_000,
        )
        v2 = handle.create_group("v2")
        v2.create_dataset(
            "goal_tokens", data=np.ones((n_steps, 10), dtype=np.float32)
        )


def _jpeg_frame() -> np.ndarray:
    yy, xx = np.indices((288, 512))
    rgb = np.stack(
        [
            (xx % 256).astype(np.uint8),
            (yy % 256).astype(np.uint8),
            ((xx + yy) % 256).astype(np.uint8),
        ],
        axis=-1,
    )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    assert ok
    return np.asarray(encoded, dtype=np.uint8).reshape(-1)
