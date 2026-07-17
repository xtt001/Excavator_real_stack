from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import yaml

from testbed.cli.build_execution_feedback_sidecars import (
    build_execution_feedback_sidecars,
    main,
)
from testbed.data.execution_feedback import (
    MANIFEST_FILENAME,
    validate_execution_feedback_manifest,
)


def test_cli_builds_only_explicit_split_ids_and_validates_manifest(
    tmp_path,
    capsys,
) -> None:
    dataset_dir = tmp_path / "resampled"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "sidecars"
    dataset_dir.mkdir()
    raw_dir.mkdir()
    for episode_id in (1, 2, 999):
        _write_episode(dataset_dir, raw_dir, episode_id=episode_id)
    split_path = tmp_path / "train_val_split.yaml"
    split_path.write_text(
        yaml.safe_dump({"train_ids": [1], "val_ids": [2]}),
        encoding="utf-8",
    )

    main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--split-path",
            str(split_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert "episodes=2" in capsys.readouterr().out
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = validate_execution_feedback_manifest(
        manifest_path,
        verify_hashes=True,
        expected_dataset_dir=dataset_dir,
        expected_split_path=split_path,
    )
    assert manifest["episode_ids"] == [1, 2]
    assert [record["episode_id"] for record in manifest["episodes"]] == [1, 2]
    assert not (output_dir / "episode_999.execution_feedback.npz").exists()
    for record in manifest["episodes"]:
        assert set(record) >= {
            "episode_id",
            "sidecar_path",
            "sidecar_sha256",
            "resampled_path",
            "resampled_sha256",
            "raw_source_path",
            "raw_source_sha256",
            "length",
            "reset_counts",
            "causality_age_summary_ns",
        }


def test_manifest_validation_rejects_tampered_sidecar_hash(tmp_path) -> None:
    dataset_dir = tmp_path / "resampled"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "sidecars"
    dataset_dir.mkdir()
    raw_dir.mkdir()
    _write_episode(dataset_dir, raw_dir, episode_id=3)
    _write_episode(dataset_dir, raw_dir, episode_id=4)
    split_path = tmp_path / "split.yaml"
    split_path.write_text(
        yaml.safe_dump({"train_ids": [3], "val_ids": [4]}),
        encoding="utf-8",
    )
    manifest = build_execution_feedback_sidecars(
        dataset_dir=dataset_dir,
        split_path=split_path,
        output_dir=output_dir,
    )
    sidecar_path = manifest["episodes"][0]["sidecar_path"]
    with open(sidecar_path, "ab") as file:
        file.write(b"tampered")

    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        validate_execution_feedback_manifest(
            output_dir / MANIFEST_FILENAME,
            verify_hashes=True,
        )


def test_split_requires_nonempty_train_and_val_ids(tmp_path) -> None:
    dataset_dir = tmp_path / "resampled"
    dataset_dir.mkdir()
    split_path = tmp_path / "split.yaml"
    split_path.write_text(json.dumps({"train_ids": [1], "val_ids": []}))

    with pytest.raises(ValueError, match="non-empty train_ids and val_ids"):
        build_execution_feedback_sidecars(
            dataset_dir=dataset_dir,
            split_path=split_path,
            output_dir=tmp_path / "output",
        )


def _write_episode(dataset_dir, raw_dir, *, episode_id: int) -> None:
    raw_path = raw_dir / f"episode_{episode_id}.hdf5"
    with h5py.File(raw_path, "w") as raw:
        diagnostics = raw.create_group("diagnostics")
        diagnostics.create_dataset(
            "commanded_action",
            data=np.asarray(
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0, 0.0],
                    [0.3, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        diagnostics.create_dataset(
            "action_send_timestamp_ns",
            data=np.asarray([150, 250, 450], dtype=np.int64),
        )
    resampled_path = dataset_dir / f"episode_{episode_id}.hdf5"
    with h5py.File(resampled_path, "w") as resampled:
        metadata = resampled.create_group("metadata")
        metadata.attrs["source_dataset_path"] = str(raw_path)
        diagnostics = resampled.create_group("diagnostics")
        diagnostics.create_dataset(
            "source_observation_timestamp_ns",
            data=np.asarray([100, 200, 300, 400, 500], dtype=np.int64),
        )
        diagnostics.create_dataset(
            "train_exclude_mask",
            data=np.asarray([0, 0, 1, 0, 0], dtype=np.uint8),
        )
        diagnostics.create_dataset(
            "source_time_gap_exceeds_threshold",
            data=np.asarray([0, 0, 0, 1, 0], dtype=np.uint8),
        )
