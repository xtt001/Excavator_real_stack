from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from testbed.data.dataset import _build_condition_shuffle_mapping
from testbed.simverify.habit_cycle_dataset import (
    CAMERAS,
    NUMERIC_DATASETS,
    _write_cycle_slice,
    select_semantic_smoke_scenarios,
)


def _source_episode(path: Path) -> None:
    length = 80
    with h5py.File(path, "w") as handle:
        for dataset_path in NUMERIC_DATASETS:
            if dataset_path == "action":
                values = np.zeros((length, 4), dtype=np.float32)
            elif dataset_path in ("observations/qpos", "observations/qvel"):
                values = np.zeros((length, 4), dtype=np.float32)
            elif dataset_path.endswith("source_observation_index"):
                values = np.arange(length, dtype=np.int64) * 2
            elif dataset_path.endswith("step_id") or dataset_path.endswith(
                "source_action_index"
            ) or dataset_path.endswith("target_tick"):
                values = np.arange(length, dtype=np.int64)
            else:
                values = np.arange(length, dtype=np.float64) * 0.05
            handle.create_dataset(dataset_path, data=values)
        for camera in CAMERAS:
            dataset = handle.create_dataset(
                f"observations/encoded_images/{camera}",
                shape=(length,),
                dtype=h5py.vlen_dtype(np.dtype("uint8")),
            )
            for index in range(length):
                dataset[index] = np.asarray([index % 255, 1, 2], dtype=np.uint8)


def test_cycle_slice_is_half_open_and_condition_activates_after_dump(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "derived.hdf5"
    _source_episode(source)
    result = _write_cycle_slice(
        source_path=source,
        output_path=output,
        row={
            "episode_id": 3,
            "cycle_id": 5,
            "cycle_ready_start_step": 20,
            "cycle_ready_end_step": 100,
            "dump_end_step": 60,
            "relative_intent": "step_right",
            "current_sector": "center",
            "hindsight_expert_target_sector": "right",
        },
        derived_episode_id=0,
        split="train",
        action_chunk_size=20,
    )
    assert result["status"] == "written"
    assert result["source_20hz_range"] == [10, 50]
    with h5py.File(output, "r") as handle:
        assert handle["action"].shape == (40, 4)
        condition = np.asarray(handle["conditions/cycle_condition_v1"])
        active = np.asarray(handle["conditions/target_committed_mask"])
        # raw dump step 60 is source tick 30; side=right activates at tick 31,
        # hence relative derived row 21.
        assert np.all(condition[:21] == 0.0)
        np.testing.assert_array_equal(condition[21], [0, 1, 0, 0, 0, 1])
        assert active[:21].sum() == 0
        assert active[21:].all()


def test_smoke_selection_takes_first_rank_per_observed_signature() -> None:
    candidates = [
        {
            "family": "move_adjacent",
            "current_sector": "center",
            "relative_intents": ["step_left"],
            "scenario_id": "first",
            "source_episode_id": 3,
            "source_cycle_ids": [2],
            "source_row_range": [10, 20],
        },
        {
            "family": "move_adjacent",
            "current_sector": "center",
            "relative_intents": ["step_left"],
            "scenario_id": "second",
            "source_episode_id": 4,
            "source_cycle_ids": [3],
            "source_row_range": [20, 30],
        },
        {
            "family": "repeat_same",
            "current_sector": "right",
            "relative_intents": ["stay", "stay"],
            "scenario_id": "repeat",
            "source_episode_id": 6,
            "source_cycle_ids": [4, 5],
            "source_row_range": [30, 50],
        },
    ]
    selected = select_semantic_smoke_scenarios(candidates)
    assert [row["scenario_id"] for row in selected] == ["first", "repeat"]


def test_committed_shuffle_preserves_pre_dump_and_current_sector(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.hdf5"
    episodes = tmp_path / "episodes"
    _source_episode(source)
    for derived_id, target in enumerate(("left", "center", "right")):
        result = _write_cycle_slice(
            source_path=source,
            output_path=episodes / f"episode_{derived_id}.hdf5",
            row={
                "episode_id": 3,
                "cycle_id": derived_id + 1,
                "cycle_ready_start_step": 20,
                "cycle_ready_end_step": 100,
                "dump_end_step": 60,
                "relative_intent": "stay",
                "current_sector": "center",
                "hindsight_expert_target_sector": target,
            },
            derived_episode_id=derived_id,
            split="train",
            action_chunk_size=20,
        )
        assert result["status"] == "written"
    mapping, manifest = _build_condition_shuffle_mapping(
        dataset_dir=episodes,
        episode_ids=[0, 1, 2],
        action_chunk_size=20,
        deadzone_intent={"require_action_loss_in_chunk": False},
        sample_valid_mask_path="conditions/valid_mask",
        seed=20260727,
        mode="next_sector_within_current_committed_only",
        committed_mask_path="conditions/target_committed_mask",
    )
    assert mapping is not None
    assert manifest["pre_commit_rows_unchanged"] is True
    assert manifest["current_sector_unchanged"] is True
    assert manifest["changed_row_count"] > 0
    np.testing.assert_array_equal(mapping[(0, 0)], np.zeros(6, dtype=np.float32))
    for key, vector in mapping.items():
        if key[1] >= 21:
            np.testing.assert_array_equal(vector[:3], [0, 1, 0])
