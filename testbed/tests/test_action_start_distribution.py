from __future__ import annotations

import h5py
import numpy as np

from testbed.policies.action_start_distribution import analyze_action_start_distribution


def test_action_start_distribution_reports_transitions_and_ambiguity(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for episode_id, offset in ((1, 0.0), (2, 0.1)):
        action = np.zeros((12, 4), dtype=np.float32)
        action[3:8, 1] = 0.30
        action[8:, 1] = 0.0
        qpos = np.zeros_like(action)
        qpos[:, 1] = offset + np.linspace(0.0, 0.3, 12, dtype=np.float32)
        qvel = np.zeros_like(action)
        qvel[3:8, 1] = 0.1
        with h5py.File(dataset / f"episode_{episode_id}.hdf5", "w") as handle:
            handle.create_dataset("action", data=action)
            handle.create_dataset("observations/qpos", data=qpos)
            handle.create_dataset("observations/qvel", data=qvel)

    report = analyze_action_start_distribution(
        dataset_dir=dataset,
        episode_ids=[1, 2],
        train_episode_ids=[1],
        thresholds={
            "swing": {"pos": 0.6, "neg": 0.7},
            "boom": {"pos": 0.25, "neg": 0.35},
            "stick": {"pos": 0.5, "neg": 0.5},
            "bucket": {"pos": 0.4, "neg": 0.5},
        },
    )
    boom = report["transition_summary"]["boom_pos"]
    assert boom["transition_count"] == 2
    assert boom["persistent_horizon_count"] == 2
    assert report["state_counts"]["boom"]["pos"] == 10
    assert report["combo_counts_by_split"]["train"]["boomp"] == 5
    assert report["boundary_distribution"]["boom_pos"]["effective_steps"] == 10
    assert report["first_transition_summary"]["label_counts"] == {"boom_pos": 2}
    assert report["transition_rows"][0]["pre_idle_run_length"] == 3
    assert report["image_ambiguity_measured"] is False
