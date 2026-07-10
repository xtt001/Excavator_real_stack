from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from scripts.trajectory_candidate_effect_audit import run_audit


def _write_fixture(root: Path, episode_id: str, scale: float) -> None:
    steps = 36
    time = np.arange(steps, dtype=np.float64) * 0.05
    expert = np.zeros((steps, 4), dtype=np.float32)
    expert[:, 0] = (0.45 + 0.1 * np.sin(np.arange(steps) / 5.0)) * scale
    expert[:, 1] = (0.40 + 0.08 * np.cos(np.arange(steps) / 6.0)) * scale
    expert[:, 3] = (-0.50 + 0.06 * np.sin(np.arange(steps) / 4.0)) * scale
    raw = expert * 0.7
    candidate = expert * 0.95
    qvel = np.zeros_like(expert)
    qvel[:, 0] = 0.03 * np.sin(np.arange(steps) / 7.0)
    qvel[:, 1] = -0.02 * np.cos(np.arange(steps) / 8.0)
    qvel[:, 3] = 0.04 * np.sin(np.arange(steps) / 9.0)
    mapped = expert.copy()
    mapped[:, 1] *= -1.0
    qpos = np.cumsum((0.03 * qvel + 0.02 * mapped) * 0.05, axis=0).astype(np.float32)

    dataset = root / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset / f"{episode_id}.hdf5", "w") as handle:
        handle.create_dataset("action", data=expert)
        obs = handle.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        obs.create_dataset("qvel", data=qvel)
        metadata = handle.create_group("metadata")
        metadata.attrs["dt"] = 0.05
    for name, action in (("raw", raw), ("candidate", candidate)):
        path = root / name / "episodes" / episode_id
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "actions.npz", time_s=time, expert_action=expert, policy_action=action)
    phase = root / "phase"
    phase.mkdir(parents=True, exist_ok=True)
    np.savez(phase / f"{episode_id}.npz", phase_prob=np.linspace(0.0, 1.0, steps))


def test_run_audit_writes_heldout_candidate_resolvability_report(tmp_path: Path) -> None:
    episode_ids = ["episode_1", "episode_2", "episode_3"]
    for index, episode_id in enumerate(episode_ids):
        _write_fixture(tmp_path, episode_id, 0.9 + 0.1 * index)
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
    paths = run_audit(
        dataset_dir=tmp_path / "dataset",
        manifest_path=manifest,
        deadzone_path=deadzone,
        raw_eval_dir=tmp_path / "raw",
        candidate_eval_dir=tmp_path / "candidate",
        phase_prob_dir=tmp_path / "phase",
        candidate_name="fixture_candidate",
        output_dir=output,
        horizons=(5,),
        stride=5,
        qvel_to_qpos_sign=np.array([1.0, -1.0, 1.0, 1.0]),
        action_to_qpos_sign=np.array([1.0, -1.0, 1.0, 1.0]),
        support_quantile=0.99,
        bootstrap_samples=100,
        bootstrap_seed=3,
        argv=["trajectory_candidate_effect_audit.py", "--fixture"],
    )

    assert len(paths) == 6
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["claim_boundary"] == "held_out_candidate_effect_resolvability_audit_only"
    assert summary["episodes"] == 3
    assert {row["axis"] for row in summary["aggregate"]} == {"swing", "boom", "bucket"}
    assert all(row["candidate_target_mae"] < row["raw_target_mae"] for row in summary["aggregate"])
    strata = (output / "candidate_effect_stratum_aggregate.csv").read_text(encoding="utf-8")
    assert "phase_probability" in strata
    assert "active_axis_concurrency" in strata
