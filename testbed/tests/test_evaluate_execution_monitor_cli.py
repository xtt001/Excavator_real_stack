from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from testbed.cli.evaluate_execution_monitor import (
    evaluate_execution_monitor_sidecars,
)


def _write_response_sidecar(path, *, label: int) -> None:
    steps = 4
    response_mask = np.full((5, steps, 4), -1, dtype=np.int8)
    response_mask[4, 0, 1] = label
    event_mask = np.zeros((steps, 4), dtype=bool)
    event_mask[0, 1] = True
    np.savez_compressed(
        path,
        qpos=np.zeros((steps, 4), dtype=np.float32),
        qvel=np.zeros((steps, 4), dtype=np.float32),
        previous_final_command=np.tile(
            np.asarray([0.0, 0.3, 0.0, 0.0], dtype=np.float32),
            (steps, 1),
        ),
        command_send_timestamp_ns=np.arange(steps, dtype=np.int64) * 10 + 1,
        observation_timestamp_ns=np.arange(steps, dtype=np.int64) * 10 + 5,
        event_mask=event_mask,
        valid_mask=np.ones(steps, dtype=bool),
        reset_mask=np.zeros(steps, dtype=bool),
        response_mask=response_mask,
    )


def _write_manifest(response_dir, episode_ids: list[int]) -> None:
    response_dir.mkdir()
    episodes = []
    for episode_id in episode_ids:
        sidecar = response_dir / f"episode_{episode_id}.execution_response.npz"
        _write_response_sidecar(sidecar, label=0)
        episodes.append({"episode_id": episode_id, "sidecar_path": str(sidecar)})
    (response_dir / "execution_response_manifest.json").write_text(
        json.dumps(
            {
                "label_contract": "direct_command_qvel_response_v1",
                "action_domain": "direct_policy_output",
                "policy_action_scale": [1.0, 1.0, 1.0, 1.0],
                "positive_threshold": [0.661, 0.259, 0.5, 0.408],
                "negative_threshold": [0.721, 0.357, 0.5, 0.508],
                "qvel_noise": [0.1, 0.1, 0.1, 0.1],
                "qvel_noise_provenance": "train-only synthetic test",
                "supported_axes": ["swing", "boom", "bucket"],
                "response_horizons": [1, 2, 4, 8, 20],
                "episode_ids": episode_ids,
                "episodes": episodes,
            }
        ),
        encoding="utf-8",
    )


def test_cli_uses_exact_split_and_reports_extras(tmp_path) -> None:
    response_dir = tmp_path / "response"
    _write_manifest(response_dir, [1, 2, 3])
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump({"train_ids": [1], "val_ids": [2]}))

    report = evaluate_execution_monitor_sidecars(
        response_dir=response_dir,
        split_path=split,
        output_dir=tmp_path / "report",
    )

    assert report["selected_episode_ids"] == [1, 2]
    assert report["excluded_response_manifest_episode_ids"] == [3]
    assert report["heldout_episode_ids"] == [105, 106, 107, 108, 109]
    assert report["retry_policy_selected"] is False
    assert report["all_selected"]["event_count"] == 2


def test_cli_rejects_heldout_ids_in_split(tmp_path) -> None:
    response_dir = tmp_path / "response"
    _write_manifest(response_dir, [1, 105])
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump({"train_ids": [105], "val_ids": [1]}))

    with pytest.raises(ValueError, match="held-out episode 105..109"):
        evaluate_execution_monitor_sidecars(
            response_dir=response_dir,
            split_path=split,
            output_dir=tmp_path / "report",
        )
