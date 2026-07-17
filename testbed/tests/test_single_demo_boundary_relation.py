from __future__ import annotations

import h5py
import numpy as np

from testbed.policies.single_demo_boundary_relation import (
    evaluate_single_demo_boundary_relation,
    fit_observed_boundary_proxy,
)


def _thresholds() -> dict[str, dict[str, float]]:
    return {
        "swing": {"pos": 0.6, "neg": 0.7},
        "boom": {"pos": 0.25, "neg": 0.35},
        "stick": {"pos": 0.5, "neg": 0.5},
        "bucket": {"pos": 0.4, "neg": 0.5},
    }


def test_boundary_proxy_is_train_only_and_can_exempt_outward_extra(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    episode = dataset / "episode_1.hdf5"
    qpos = np.zeros((8, 4), dtype=np.float32)
    qpos[:, 3] = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    action = np.zeros((8, 4), dtype=np.float32)
    action[:, 3] = 0.6
    with h5py.File(episode, "w") as handle:
        handle.create_dataset("observations/qpos", data=qpos)
        handle.create_dataset("action", data=action)

    eval_dir = tmp_path / "eval"
    (eval_dir / "episodes" / "episode_1").mkdir(parents=True)
    expert = np.zeros((8, 4), dtype=np.float32)
    policy = np.zeros((8, 4), dtype=np.float32)
    policy[-1, 3] = 0.6
    np.savez(eval_dir / "episodes" / "episode_1" / "actions.npz", expert_action=expert, policy_action=policy)

    proxy = fit_observed_boundary_proxy(
        dataset_dir=dataset,
        train_episode_ids=[1],
        thresholds=_thresholds(),
        lower_quantile=0.01,
        upper_quantile=0.99,
        progress_horizon=1,
        boundary_margin_fraction=0.05,
    )
    report = evaluate_single_demo_boundary_relation(
        eval_episode_ids=[1],
        dataset_dir=dataset,
        eval_dir=eval_dir,
        thresholds=_thresholds(),
        boundary_proxy=proxy,
        model="test",
    )
    assert report["physical_limit_ground_truth"] is False
    assert report["outside_single_demo_effective"] == 1
    assert report["outside_demo_boundary_exempt"] == 1
    assert report["outside_demo_nonexempt"] == 0
