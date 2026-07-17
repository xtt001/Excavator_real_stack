from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from testbed.cli.audit_startup_ownership import run_startup_ownership_audit
from testbed.policies.action_start_distribution import sha256_file
from testbed.policies.startup_ownership import audit_startup_ownership


def _thresholds() -> dict[str, dict[str, float]]:
    return {
        axis: {"pos": 0.5, "neg": 0.5}
        for axis in ("swing", "boom", "stick", "bucket")
    }


def _zeros(steps: int) -> np.ndarray:
    return np.zeros((steps, 4), dtype=np.float32)


def test_autonomy_scoring_only_reclassifies_clean_first_start_direction() -> None:
    expert = _zeros(5)
    expert[3:, 3] = 0.8
    policy = _zeros(5)
    policy[0, 3] = 0.8
    policy[1, 3] = 0.8
    policy[1, 1] = 0.8
    policy[2, 3] = -0.8

    report = audit_startup_ownership(
        expert_actions={"episode_1": expert},
        policy_collections={"model": {"episode_1": policy}},
        thresholds=_thresholds(),
        sample_hz=20.0,
    )

    expert_row = report["expert_episode_rows"][0]
    assert expert_row["prestart_frames"] == 3
    assert expert_row["expert_first_onset_step"] == 3
    assert expert_row["expert_first_onset_seconds"] == 0.15
    assert expert_row["expert_first_start_axis"] == "bucket"
    assert expert_row["expert_first_start_direction"] == "pos"
    assert expert_row["expert_prestart_all_axis_neutral_frames"] == 3

    row = report["model_episode_rows"][0]
    assert row["imitation_aligned_early_extra_frames"] == 3
    assert row["autonomy_aligned_early_start_frames"] == 1
    assert row["autonomy_wrong_or_extra_frames"] == 2
    assert row["autonomy_reclassified_early_start_frames"] == 1
    assert row["autonomy_mixed_supported_and_unsupported_frames"] == 1
    assert row["autonomy_opposite_start_axis_frames"] == 1
    assert row["autonomy_other_axis_frames"] == 1
    assert row["autonomy_unsupported_direction_activations"] == 2
    assert row["first_autonomy_aligned_early_start_step"] == 0
    assert row["autonomy_aligned_lead_ticks"] == 3
    assert row["autonomy_aligned_lead_seconds"] == 0.15


def test_onset_at_zero_has_empty_prestart_and_neutral_elsewhere() -> None:
    expert = _zeros(3)
    expert[0, 3] = 0.8
    policy = _zeros(3)

    report = audit_startup_ownership(
        expert_actions={"episode_2": expert},
        policy_collections={"model": {"episode_2": policy}},
        thresholds=_thresholds(),
        sample_hz=20.0,
    )

    expert_row = report["expert_episode_rows"][0]
    assert expert_row["prestart_frames"] == 0
    assert expert_row["expert_prestart_all_axis_neutral_pct"] is None
    assert expert_row["expert_elsewhere_frames"] == 3
    assert expert_row["expert_elsewhere_all_axis_neutral_frames"] == 2
    model_row = report["model_episode_rows"][0]
    assert model_row["imitation_aligned_early_extra_frames"] == 0
    assert model_row["imitation_aligned_early_extra_pct"] is None
    assert model_row["autonomy_wrong_or_extra_pct"] is None


def test_episode_without_expert_start_keeps_all_policy_motion_wrong() -> None:
    expert = _zeros(3)
    policy = _zeros(3)
    policy[0, 3] = 0.8

    report = audit_startup_ownership(
        expert_actions={"episode_3": expert},
        policy_collections={"model": {"episode_3": policy}},
        thresholds=_thresholds(),
        sample_hz=20.0,
    )

    expert_aggregate = report["aggregate"]["expert"]
    assert expert_aggregate["episodes_without_expert_start"] == 1
    row = report["model_episode_rows"][0]
    assert row["imitation_aligned_early_extra_frames"] == 1
    assert row["autonomy_aligned_early_start_frames"] == 0
    assert row["autonomy_wrong_or_extra_frames"] == 1


def test_aggregate_separates_prestart_and_elsewhere_neutral_frames() -> None:
    expert_one = _zeros(5)
    expert_one[3:, 3] = 0.8
    expert_two = _zeros(3)
    expert_two[0, 3] = 0.8
    policy_one = _zeros(5)
    policy_one[0, 3] = 0.8
    policy_two = _zeros(3)

    report = audit_startup_ownership(
        expert_actions={"episode_1": expert_one, "episode_2": expert_two},
        policy_collections={
            "model": {"episode_1": policy_one, "episode_2": policy_two}
        },
        thresholds=_thresholds(),
        sample_hz=20.0,
    )

    expert = report["aggregate"]["expert"]
    assert expert["steps"] == 8
    assert expert["prestart_frames"] == 3
    assert expert["prestart_all_axis_neutral_frames"] == 3
    assert expert["prestart_all_axis_neutral_pct"] == 100.0
    assert expert["elsewhere_frames"] == 5
    assert expert["elsewhere_all_axis_neutral_frames"] == 2
    assert expert["elsewhere_all_axis_neutral_pct"] == 40.0
    model = report["aggregate"]["models"]["model"]
    assert model["imitation_aligned_early_extra_frames"] == 1
    assert model["autonomy_wrong_or_extra_frames"] == 0
    assert model["autonomy_reclassified_early_start_frames"] == 1


def test_heldout_episode_is_rejected() -> None:
    with pytest.raises(ValueError, match="held-out"):
        audit_startup_ownership(
            expert_actions={"episode_105": _zeros(2)},
            policy_collections={"model": {"episode_105": _zeros(2)}},
            thresholds=_thresholds(),
            sample_hz=20.0,
        )


def test_cli_writes_hashed_artifacts_without_modifying_hdf5(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    expert = _zeros(4)
    expert[2:, 3] = 0.8
    episode_path = dataset / "episode_1.hdf5"
    with h5py.File(episode_path, "w") as handle:
        handle.create_dataset("action", data=expert)
    hdf5_sha_before = sha256_file(episode_path)

    split = tmp_path / "split.yaml"
    split.write_text(
        yaml.safe_dump(
            {
                "dataset_dir": str(dataset.resolve()),
                "train_ids": [1],
                "val_ids": [],
            }
        ),
        encoding="utf-8",
    )
    deadzone = tmp_path / "deadzone.json"
    deadzone.write_text(
        json.dumps({"deadzone_action": _thresholds()}), encoding="utf-8"
    )
    collection = tmp_path / "collection"
    action_dir = collection / "episodes" / "episode_1"
    action_dir.mkdir(parents=True)
    policy = _zeros(4)
    policy[0, 3] = 0.8
    np.savez(
        action_dir / "actions.npz",
        time_s=np.arange(4, dtype=np.float64) / 20.0,
        expert_action=expert,
        policy_action=policy,
    )
    (collection / "collection_summary.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset.resolve()),
                "episode_ids": ["episode_1"],
            }
        ),
        encoding="utf-8",
    )

    result = run_startup_ownership_audit(
        dataset_dir=dataset,
        split_path=split,
        deadzone_json=deadzone,
        policy_collections={"A": collection},
        sample_hz=20.0,
        output_dir=tmp_path / "report",
    )

    assert sha256_file(episode_path) == hdf5_sha_before
    assert result["report_sha256"] == sha256_file(result["report"])
    assert result["source_manifest_sha256"] == sha256_file(
        result["source_manifest"]
    )
    report = json.loads(result["report"].read_text(encoding="utf-8"))
    assert report["inputs"]["source_hdf5_modified"] is False
    assert report["inputs"]["policy_inference_performed"] is False
    assert report["inputs"]["first_expert_onset_used_at_deployment"] is False
    assert len(report["inputs"]["audit_implementation"]) == 2
    assert all(
        len(item["sha256"]) == 64
        for item in report["inputs"]["audit_implementation"]
    )
    assert report["aggregate"]["models"]["A"][
        "autonomy_reclassified_early_start_frames"
    ] == 1
