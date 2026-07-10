import numpy as np

from testbed.data.visual_domain import (
    assign_episode_domains,
    kmeans_numpy,
    task_frame_indices,
)


def test_task_frame_indices_stop_before_first_gohome_and_sample_evenly():
    train_exclude = np.zeros(12, dtype=bool)
    train_exclude[[1, 10]] = True
    gohome = np.zeros(12, dtype=np.int64)
    gohome[8:] = 1

    indices = task_frame_indices(
        total_steps=12,
        max_frames=4,
        train_exclude_mask=train_exclude,
        gohome_requested=gohome,
        gohome_running=None,
    )

    assert len(indices) == 4
    assert indices[0] == 0
    assert indices[-1] == 7
    assert 1 not in indices
    assert np.all(indices < 8)


def test_kmeans_numpy_is_deterministic_and_separates_simple_clusters():
    points = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ],
        dtype=np.float32,
    )

    labels, centers = kmeans_numpy(points, k=2, seed=7, max_iter=20)

    assert labels.tolist() == [1, 1, 0, 0]
    assert centers.shape == (2, 2)


def test_assign_episode_domains_reports_dominant_and_proportions():
    rows = [
        {"episode_id": "episode_1", "cluster": 0},
        {"episode_id": "episode_1", "cluster": 0},
        {"episode_id": "episode_1", "cluster": 1},
        {"episode_id": "episode_2", "cluster": 1},
        {"episode_id": "episode_2", "cluster": 1},
    ]

    domains = assign_episode_domains(rows, k=2)

    assert domains["episode_1"]["dominant_domain"] == "texture_domain_0"
    assert domains["episode_1"]["dominant_fraction"] == 2 / 3
    assert domains["episode_1"]["domain_proportions"] == {
        "texture_domain_0": 2 / 3,
        "texture_domain_1": 1 / 3,
    }
    assert domains["episode_2"]["dominant_domain"] == "texture_domain_1"
