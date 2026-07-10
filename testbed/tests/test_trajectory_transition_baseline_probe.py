from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from scripts.trajectory_transition_baseline_probe import run_probe


def _write_episode(path: Path, *, scale: float) -> None:
    steps = 30
    action = np.zeros((steps, 4), dtype=np.float32)
    action[:, 0] = 0.6 * scale
    action[:, 1] = 0.6 * scale
    action[:, 3] = -0.7 * scale
    mapped_effect = np.zeros_like(action)
    mapped_effect[:, 0] = np.maximum(action[:, 0] - 0.2, 0.0) / 0.8
    mapped_effect[:, 1] = -np.maximum(action[:, 1] - 0.2, 0.0) / 0.8
    mapped_effect[:, 3] = np.minimum(action[:, 3] + 0.2, 0.0) / 0.8
    qpos = np.cumsum(mapped_effect * 0.02, axis=0).astype(np.float32)
    qvel = np.zeros_like(qpos)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=action)
        obs = handle.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        obs.create_dataset("qvel", data=qvel)
        metadata = handle.create_group("metadata")
        metadata.attrs["dt"] = 0.05


def test_run_probe_writes_heldout_transition_report(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    episode_ids = ["episode_1", "episode_2", "episode_3"]
    for index, episode_id in enumerate(episode_ids):
        _write_episode(dataset / f"{episode_id}.hdf5", scale=0.8 + 0.1 * index)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train_ready_episode_ids": episode_ids}), encoding="utf-8")
    deadzone = tmp_path / "deadzone.json"
    deadzone.write_text(
        json.dumps(
            {
                "deadzone_action": {
                    axis: {"pos": 0.2, "neg": 0.2}
                    for axis in ("swing", "boom", "stick", "bucket")
                }
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "report"
    paths = run_probe(
        dataset_dir=dataset,
        manifest_path=manifest,
        deadzone_path=deadzone,
        output_dir=output,
        horizons=(5, 10),
        stride=5,
        qvel_to_qpos_sign=np.array([1.0, -1.0, 1.0, 1.0]),
        action_to_qpos_sign=np.array([1.0, -1.0, 1.0, 1.0]),
        inactive_axes=("stick",),
        bootstrap_samples=100,
        bootstrap_seed=4,
        argv=["trajectory_transition_baseline_probe.py", "--fixture"],
    )

    assert {path.name for path in paths.values()} == {
        "run_manifest.json",
        "state_contract.json",
        "transition_baseline_by_episode.csv",
        "transition_baseline_aggregate.csv",
        "summary.json",
        "transition_mae_by_horizon.png",
    }
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["claim_boundary"] == "held_out_expert_transition_probe_only"
    assert summary["episodes"] == 3
    aggregate = summary["aggregate"]
    stick_models = {row["model"] for row in aggregate if row["axis"] == "stick"}
    assert stick_models == {"constant_state", "initial_qvel"}
    swing_h5 = {
        row["model"]: row
        for row in aggregate
        if row["axis"] == "swing" and row["horizon_steps"] == 5
    }
    assert swing_h5["action_linear"]["mae_mean"] < swing_h5["constant_state"]["mae_mean"]
    contract = json.loads((output / "state_contract.json").read_text(encoding="utf-8"))
    assert contract["qvel_to_qpos_sign"]["boom"] == -1.0
    assert contract["inactive_axes"] == ["stick"]
