import json

import h5py
import numpy as np

from testbed.cli.evaluate_complete_offline import build_report


def test_complete_offline_report_preserves_non_estimable_gohome_boundary(
    tmp_path,
) -> None:
    open_loop = tmp_path / "open_loop"
    episode_dir = open_loop / "episodes" / "episode_1"
    episode_dir.mkdir(parents=True)
    expert = np.zeros((10, 4), dtype=np.float32)
    expert[0, 0] = 0.8
    policy = np.zeros_like(expert)
    policy[:3, 3] = 0.8
    np.savez(episode_dir / "actions.npz", expert_action=expert, policy_action=policy)
    (open_loop / "collection_summary.json").write_text("{}")

    state_summary = tmp_path / "state_summary.json"
    state_detail = tmp_path / "state_detail.json"
    state_detail.write_text(
        json.dumps(
            {
                "aggregate": [
                    {
                        "group": "overall",
                        "anchors_total": 2,
                        "state_hold_demo_target_reproduced_anchors": 1,
                        "state_hold_demo_target_not_reproduced_anchors": 1,
                        "demo_target_reproduction_hidden_by_teacher_forcing_anchors": 1,
                    }
                ]
            }
        )
    )
    state_summary.write_text(
        json.dumps(
            {
                "candidate_id": "fixture",
                "pipeline_mode": "raw",
                "episode_ids": ["episode_1"],
                "reports": [
                    {
                        "mode": "assist_enabled",
                        "paths": {"summary": str(state_detail)},
                    }
                ],
            }
        )
    )
    monitor = tmp_path / "monitor.json"
    monitor.write_text(json.dumps({"val": {"retry_precision_estimable": False}}))
    deadzone = tmp_path / "deadzone.json"
    deadzone.write_text(
        json.dumps(
            {
                "deadzone_action": {
                    "swing": {"pos": 0.661, "neg": 0.721},
                    "boom": {"pos": 0.259, "neg": 0.357},
                    "stick": {"pos": 0.5, "neg": 0.5},
                    "bucket": {"pos": 0.408, "neg": 0.508},
                }
            }
        )
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with h5py.File(dataset / "episode_1.hdf5", "w") as handle:
        handle.create_dataset("action", data=expert)
        handle.create_dataset("observations/qpos", data=np.zeros_like(expert))
        handle.create_dataset("observations/qvel", data=np.zeros_like(expert))
        metadata = handle.create_group("metadata")
        metadata.attrs["excluded_go_home"] = True

    report = build_report(
        open_loop_dir=open_loop,
        state_hold_run_summary=state_summary,
        execution_monitor_json=monitor,
        deadzone_json=deadzone,
        dataset_dir=dataset,
        output_dir=tmp_path / "report",
        model_label="fixture",
    )
    assert report["single_demo_reproduction_diagnostic"]["promotion_gate"] is False
    assert "validation_gate" not in report
    assert report["gohome"]["estimable"] is False
    assert report["heldout_evaluated"] is False
    assert (tmp_path / "report" / "complete_offline_report.json").exists()
