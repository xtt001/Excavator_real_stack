from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
import pytest

from testbed.data.semantic_cycle_crop import (
    build_semantic_cycle_dataset,
    write_semantic_cycle_annotation_template,
)


def test_build_semantic_cycle_dataset_uses_reviewed_cut_annotations(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _write_episode(input_dir / "episode_5.hdf5", n_steps=6)
    annotations = tmp_path / "cuts.csv"
    _write_annotations(
        annotations,
        [
            {
                "episode_id": "5",
                "cut_step": "4",
                "cut_label": "before_return_swing",
                "review_status": "manual_reviewed",
                "notes": "dump complete, return not included",
            }
        ],
    )

    summary = build_semantic_cycle_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        annotation_csv=annotations,
        episode_ids=[5],
    )

    assert summary["episodes"][0]["episode_id"] == 5
    assert summary["episodes"][0]["output_steps"] == 4
    out_path = output_dir / "episode_5.hdf5"
    with h5py.File(out_path, "r") as f:
        assert f["action"].shape == (4, 4)
        assert f["observations/qpos"].shape == (4, 4)
        assert f["observations/encoded_images/video4"].shape == (4,)
        assert f["diagnostics/not_time_axis"].shape == (2,)
        assert f["metadata"].attrs["semantic_cycle_cut"] == 1
        assert f["metadata"].attrs["semantic_cycle_cut_step"] == 4
        assert f["metadata"].attrs["semantic_cycle_cut_label"] == "before_return_swing"
        assert f["metadata"].attrs["semantic_cycle_review_status"] == "manual_reviewed"
        np.testing.assert_array_equal(
            f["diagnostics/source_cycle_keep_mask"][()],
            np.array([1, 1, 1, 1], dtype=np.uint8),
        )


def test_build_semantic_cycle_dataset_rejects_unreviewed_cut(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _write_episode(input_dir / "episode_2.hdf5", n_steps=5)
    annotations = tmp_path / "cuts.csv"
    _write_annotations(
        annotations,
        [
            {
                "episode_id": "2",
                "cut_step": "3",
                "cut_label": "auto_guess",
                "review_status": "needs_manual_review",
                "notes": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="not reviewed"):
        build_semantic_cycle_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            annotation_csv=annotations,
            episode_ids=[2],
        )


def test_write_semantic_cycle_annotation_template_leaves_cut_blank(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_episode(input_dir / "episode_7.hdf5", n_steps=9)
    template = tmp_path / "template.csv"

    summary = write_semantic_cycle_annotation_template(
        input_dir=input_dir,
        output_csv=template,
        episode_ids=[7],
    )

    assert summary["episodes"][0]["episode_id"] == 7
    with template.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {
            "episode_id": "7",
            "source_path": str(input_dir / "episode_7.hdf5"),
            "source_steps": "9",
            "cut_step": "",
            "cut_label": "",
            "review_status": "needs_manual_review",
            "notes": "",
        }
    ]


def _write_annotations(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_id",
                "cut_step",
                "cut_label",
                "review_status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_episode(path: Path, *, n_steps: int) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["is_real"] = True
        meta = f.create_group("metadata")
        meta.attrs["episode_id"] = path.stem
        obs = f.create_group("observations")
        obs.create_dataset(
            "qpos",
            data=np.arange(n_steps * 4, dtype=np.float32).reshape(n_steps, 4),
        )
        obs.create_dataset(
            "qvel",
            data=np.arange(n_steps * 4, dtype=np.float32).reshape(n_steps, 4) + 100,
        )
        encoded = obs.create_group("encoded_images")
        dtype = h5py.vlen_dtype(np.dtype("uint8"))
        ds = encoded.create_dataset("video4", (n_steps,), dtype=dtype)
        for idx in range(n_steps):
            ds[idx] = np.asarray([idx, idx + 1], dtype=np.uint8)
        f.create_dataset(
            "action",
            data=np.arange(n_steps * 4, dtype=np.float32).reshape(n_steps, 4),
        )
        diag = f.create_group("diagnostics")
        diag.create_dataset("source_observation_index", data=np.arange(n_steps))
        diag.create_dataset("not_time_axis", data=np.asarray([10, 11], dtype=np.int64))
