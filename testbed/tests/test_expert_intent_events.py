from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import yaml

from testbed.cli.audit_expert_intent_events import main
from testbed.data.expert_intent_events import (
    EVENTS_CSV_FILENAME,
    EVENTS_FILENAME,
    MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    build_expert_intent_event_sidecar,
    derive_expert_intent_events,
    sha256_file,
)

THRESHOLDS = {
    "swing": {"pos": 0.6, "neg": 0.7},
    "boom": {"pos": 0.25, "neg": 0.35},
    "stick": {"pos": 0.5, "neg": 0.5},
    "bucket": {"pos": 0.4, "neg": 0.5},
}


def test_derives_immediate_near_and_ordered_supported_intent(tmp_path) -> None:
    action = np.zeros((18, 4), dtype=np.float32)
    action[3:9, 2] = 0.6
    action[4:12, 1] = -0.4
    action[7:15, 3] = 0.45
    qpos = np.arange(72, dtype=np.float32).reshape(18, 4) / 10.0
    qvel = qpos / 100.0
    source = tmp_path / "episode_1.hdf5"
    source.write_bytes(b"source identity")

    events = derive_expert_intent_events(
        episode_id=1,
        split="train",
        action=action,
        qpos=qpos,
        qvel=qvel,
        thresholds=THRESHOLDS,
        support_horizon_ticks=11,
        source_path=source,
        source_sha256=sha256_file(source),
    )

    first = events[0]
    assert first["onset_step"] == 3
    assert first["newly_effective_directions"] == ["stick+"]
    assert first["anchor_intent"] == ["stick+"]
    assert first["immediate_intent_0_1"] == ["boom-", "stick+"]
    assert first["near_intent_2_5"] == ["boom-", "stick+", "bucket+"]
    assert first["near_intent_6_10"] == ["boom-", "bucket+"]
    assert first["single_demo_event_support_directions"] == [
        "boom-",
        "stick+",
        "bucket+",
    ]
    assert "task_supported_directions" not in first
    assert first["motif"] == "0:stick+>1:boom->4:bucket+"
    assert first["axis_first_onset_delay_ticks"] == {
        "swing": None,
        "boom": 1,
        "stick": 0,
        "bucket": 4,
    }
    details = {row["direction"]: row for row in first["direction_details"]}
    assert details["stick+"]["persistence_ticks"] == 6
    assert details["stick+"]["pre_idle_dwell_ticks"] == 3
    assert details["stick+"]["release_step_exclusive"] == 9
    assert first["release_bounds"] == {
        "earliest_release_step_exclusive": 9,
        "latest_release_step_exclusive": 15,
        "right_censored_direction_count": 0,
    }
    assert first["qpos_at_onset"]["stick"] == pytest.approx(qpos[3, 2])
    assert first["qvel_at_onset"]["bucket"] == pytest.approx(qvel[3, 3])
    assert events[1]["motif"] == "-1:stick+>0:boom->3:bucket+"
    assert events[1]["direction_details"][1]["active_at_event_onset"] is True


def test_sidecar_is_deterministic_and_does_not_modify_sources(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for episode_id in (1, 2):
        _write_episode(dataset / f"episode_{episode_id}.hdf5", offset=episode_id)
    deadzone = tmp_path / "deadzone.json"
    deadzone.write_text(json.dumps({"deadzone_action": THRESHOLDS}), encoding="utf-8")
    split = tmp_path / "split.yaml"
    split.write_text(
        yaml.safe_dump({"dataset_dir": str(dataset), "train_ids": [1], "val_ids": [2]}),
        encoding="utf-8",
    )
    source_hashes = {
        episode_id: sha256_file(dataset / f"episode_{episode_id}.hdf5")
        for episode_id in (1, 2)
    }

    manifests = []
    for name in ("first", "second"):
        output = tmp_path / name
        manifests.append(
            build_expert_intent_event_sidecar(
                dataset_dir=dataset,
                output_dir=output,
                thresholds=THRESHOLDS,
                threshold_source_path=deadzone,
                train_episode_ids=[1],
                validation_episode_ids=[2],
                support_horizon_ticks=11,
                split_path=split,
            )
        )
    assert manifests[0] == manifests[1]
    for filename in (
        MANIFEST_FILENAME,
        EVENTS_FILENAME,
        EVENTS_CSV_FILENAME,
        SUMMARY_FILENAME,
    ):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()
    assert source_hashes == {
        episode_id: sha256_file(dataset / f"episode_{episode_id}.hdf5")
        for episode_id in (1, 2)
    }
    summary = json.loads((tmp_path / "first" / SUMMARY_FILENAME).read_text())
    assert summary["by_split"]["train"]["episode_count"] == 1
    assert summary["by_split"]["validation"]["event_count"] == 1
    assert summary["by_split"]["train"]["first_event"] == {
        "event_count": 1,
        "anchor_intent_set_counts": {"stick+": 1},
        "single_demo_event_support_direction_set_counts": {"stick+": 1},
        "ordered_direction_motif_counts": {"stick+": 1},
    }


def test_cli_rejects_test_role_before_opening_any_episode(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    deadzone = tmp_path / "deadzone.json"
    deadzone.write_text(json.dumps({"deadzone_action": THRESHOLDS}), encoding="utf-8")
    split = tmp_path / "split.yaml"
    split.write_text(
        yaml.safe_dump(
            {
                "dataset_dir": str(dataset),
                "train_ids": [1],
                "val_ids": [2],
                "test_ids": [156],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden test role test_ids"):
        main(
            [
                "--dataset-dir",
                str(dataset),
                "--deadzone-json",
                str(deadzone),
                "--split-path",
                str(split),
                "--support-horizon-ticks",
                "11",
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert not (tmp_path / "output").exists()


def test_explicit_ids_reject_documented_sealed_episode(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    deadzone = tmp_path / "deadzone.json"
    deadzone.write_text(json.dumps({"deadzone_action": THRESHOLDS}), encoding="utf-8")

    with pytest.raises(ValueError, match="sealed/test episode IDs are forbidden"):
        main(
            [
                "--dataset-dir",
                str(dataset),
                "--deadzone-json",
                str(deadzone),
                "--train-id",
                "156",
                "--val-id",
                "2",
                "--support-horizon-ticks",
                "11",
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )


@pytest.mark.parametrize(
    ("split_payload", "train_ids", "validation_ids", "error"),
    [
        (
            {"train_ids": [1], "val_ids": [2], "test_ids": [156]},
            [1],
            [2],
            "forbidden test role test_ids",
        ),
        (
            {"train_ids": [1], "val_ids": [2]},
            [1],
            [3],
            "do not exactly match split file",
        ),
    ],
)
def test_programmatic_split_validation_precedes_episode_access(
    tmp_path,
    split_payload,
    train_ids,
    validation_ids,
    error,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    deadzone = tmp_path / "deadzone.json"
    deadzone.write_text(json.dumps({"deadzone_action": THRESHOLDS}), encoding="utf-8")
    split = tmp_path / "split.yaml"
    split_payload["dataset_dir"] = str(dataset)
    split.write_text(yaml.safe_dump(split_payload), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        build_expert_intent_event_sidecar(
            dataset_dir=dataset,
            output_dir=tmp_path / "output",
            thresholds=THRESHOLDS,
            threshold_source_path=deadzone,
            train_episode_ids=train_ids,
            validation_episode_ids=validation_ids,
            support_horizon_ticks=11,
            split_path=split,
        )
    assert not (tmp_path / "output").exists()


def _write_episode(path, *, offset: int) -> None:
    action = np.zeros((14, 4), dtype=np.float32)
    action[3:9, 2] = 0.6
    qpos = np.full_like(action, float(offset))
    qvel = np.zeros_like(action)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=action)
        handle.create_dataset("observations/qpos", data=qpos)
        handle.create_dataset("observations/qvel", data=qvel)
