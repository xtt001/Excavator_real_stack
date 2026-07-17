from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import yaml

from testbed.cli.offline_startup_activation import run_offline_startup_activation
from testbed.data.expert_intent_events import sha256_file

THRESHOLDS = {
    axis: {"pos": 0.5, "neg": 0.5} for axis in ("swing", "boom", "stick", "bucket")
}


class AlwaysEffectiveSource:
    def reset(self) -> None:
        pass

    def step(self, observation: dict[str, Any]) -> np.ndarray:
        del observation
        return np.array([0.0, 0.0, 0.7, 0.0], dtype=np.float32)

    def snapshot_state(self) -> Any:
        return None

    def restore_state(self, state: Any) -> None:
        del state


def _write_episode(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=np.zeros((4, 4), dtype=np.float32))
        handle.create_dataset(
            "observations/qpos", data=np.zeros((4, 4), dtype=np.float32)
        )
        handle.create_dataset(
            "observations/qvel", data=np.ones((4, 4), dtype=np.float32)
        )
        handle.create_dataset(
            "observations/images/video4",
            data=np.zeros((4, 2, 2, 3), dtype=np.uint8),
        )


def _prepare_inputs(
    tmp_path: Path, *, ids: tuple[int, ...] = (1000, 1001)
) -> dict[str, Path]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for episode_id in ids:
        _write_episode(dataset / f"episode_{episode_id}.hdf5")

    config = tmp_path / "config.yaml"
    policy = {
        "camera_names": ["video4"],
        "low_dim_keys": ["qpos", "qvel", "previous_final_command"],
        "qvel_mode": "raw",
        "output_mode": "shadow_zero",
        "action_scale": [2.0, 2.0, 2.0, 2.0],
        "fail_safe_zero": True,
        "deadzone_assist": {"enabled": True},
        "runtime_gates": {"enabled": True},
    }
    config.write_text(yaml.safe_dump({"teleop": {"policy": policy}}), encoding="utf-8")

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "policy_best.ckpt").write_bytes(b"checkpoint")
    (bundle / "dataset_stats.pkl").write_bytes(b"stats")
    (bundle / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_contract": {"expected_camera_names": ["video4"]},
                "task": {"camera_names": ["video4"]},
                "policy": {
                    "low_dim_keys": ["qpos", "qvel", "previous_final_command"],
                },
            }
        ),
        encoding="utf-8",
    )

    deadzone = tmp_path / "deadzone.json"
    deadzone.write_text(json.dumps({"deadzone_action": THRESHOLDS}), encoding="utf-8")

    event_dir = tmp_path / "events"
    event_dir.mkdir()
    events = [
        {
            "schema_version": "single_demo_intent_events_v2",
            "event_id": f"episode_{episode_id}:event_0000:step_2",
            "episode_id": episode_id,
            "split": "validation",
            "event_index": 0,
            "onset_step": 2,
            "anchor_intent": ["stick+"],
            "single_demo_event_support_directions": ["stick+", "bucket+"],
        }
        for episode_id in ids
    ]
    events_path = event_dir / "expert_intent_events.jsonl"
    events_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "single_demo_intent_events_v2",
        "dataset_dir": str(dataset.resolve()),
        "validation_ids": list(ids),
        "thresholds": THRESHOLDS,
        "threshold_source_sha256": sha256_file(deadzone),
        "artifacts": {"expert_intent_events.jsonl": sha256_file(events_path)},
        "episodes": [
            {
                "episode_id": episode_id,
                "split": "validation",
                "path": str((dataset / f"episode_{episode_id}.hdf5").resolve()),
                "sha256": sha256_file(dataset / f"episode_{episode_id}.hdf5"),
            }
            for episode_id in ids
        ],
    }
    (event_dir / "expert_intent_events_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return {
        "dataset": dataset,
        "config": config,
        "bundle": bundle,
        "deadzone": deadzone,
        "event_dir": event_dir,
        "output": tmp_path / "output",
    }


def test_cli_requires_exact_validation_coverage_and_writes_provenance_atomically(
    tmp_path: Path,
) -> None:
    paths = _prepare_inputs(tmp_path)
    seen_config: dict[str, Any] = {}

    def factory(config: dict[str, Any]) -> AlwaysEffectiveSource:
        seen_config.update(config)
        return AlwaysEffectiveSource()

    result = run_offline_startup_activation(
        model="N-test",
        config_path=paths["config"],
        bundle_dir=paths["bundle"],
        dataset_dir=paths["dataset"],
        event_dir=paths["event_dir"],
        deadzone_json=paths["deadzone"],
        hold_horizon_steps=5,
        sampling_hz=20.0,
        output_dir=paths["output"],
        device="cpu",
        step_source_factory=factory,
    )

    report = json.loads(Path(result["report"]).read_text())
    source = json.loads(
        (paths["output"] / "startup_activation_source_manifest.json").read_text()
    )
    rows = [
        json.loads(line)
        for line in (paths["output"] / "startup_activation_rows.jsonl")
        .read_text()
        .splitlines()
    ]
    assert result["episode_rows"] == 2
    assert report["validation_ids"] == [1000, 1001]
    assert report["aggregate"]["natural_liveness_count"] == 2
    assert report["algorithm_semantics"]["startup_axis_requirement"] == "none"
    assert report["capability_boundaries"]["single_demo_similarity_only"] is True
    assert all(row["activation_delay_ticks"] == 0 for row in rows)
    assert source["policy_inference_performed"] is True
    assert source["model_command_sent"] is False
    assert source["sealed_test_data_read"] is False
    assert len(source["source_hdf5"]) == 2
    assert len(source["implementation"]) == 4
    assert seen_config["action_scale"] == [1.0, 1.0, 1.0, 1.0]
    assert seen_config["runtime_gates"] == {"enabled": False}
    assert seen_config["deadzone_assist"]["enabled"] is False
    assert seen_config["fail_safe_zero"] is False
    assert not list(paths["output"].glob(".*.tmp"))


def test_cli_rejects_missing_first_event_before_hdf5_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_inputs(tmp_path)
    events_path = paths["event_dir"] / "expert_intent_events.jsonl"
    first_line = events_path.read_text().splitlines()[0] + "\n"
    events_path.write_text(first_line, encoding="utf-8")
    manifest_path = paths["event_dir"] / "expert_intent_events_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["expert_intent_events.jsonl"] = sha256_file(events_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        "testbed.cli.offline_startup_activation.h5py.File",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("HDF5 opened")),
    )
    with pytest.raises(ValueError, match="do not exactly match"):
        run_offline_startup_activation(
            model="N-test",
            config_path=paths["config"],
            bundle_dir=paths["bundle"],
            dataset_dir=paths["dataset"],
            event_dir=paths["event_dir"],
            deadzone_json=paths["deadzone"],
            hold_horizon_steps=5,
            sampling_hz=20.0,
            output_dir=paths["output"],
            step_source_factory=lambda _config: AlwaysEffectiveSource(),
        )


def test_cli_rejects_sealed_composite_or_source_before_hdf5_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_inputs(tmp_path, ids=(105,))
    monkeypatch.setattr(
        "testbed.cli.offline_startup_activation.h5py.File",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("HDF5 opened")),
    )
    with pytest.raises(ValueError, match="sealed/test"):
        run_offline_startup_activation(
            model="N-test",
            config_path=paths["config"],
            bundle_dir=paths["bundle"],
            dataset_dir=paths["dataset"],
            event_dir=paths["event_dir"],
            deadzone_json=paths["deadzone"],
            hold_horizon_steps=5,
            sampling_hz=20.0,
            output_dir=paths["output"],
            step_source_factory=lambda _config: AlwaysEffectiveSource(),
        )


def test_cli_rejects_sealed_source_id_before_hdf5_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_inputs(tmp_path, ids=(1000,))
    manifest_path = paths["event_dir"] / "expert_intent_events_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"][0]["path"] = str(tmp_path / "episode_156.hdf5")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        "testbed.cli.offline_startup_activation.h5py.File",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("HDF5 opened")),
    )
    with pytest.raises(ValueError, match="source episode IDs contains sealed/test"):
        run_offline_startup_activation(
            model="N-test",
            config_path=paths["config"],
            bundle_dir=paths["bundle"],
            dataset_dir=paths["dataset"],
            event_dir=paths["event_dir"],
            deadzone_json=paths["deadzone"],
            hold_horizon_steps=5,
            sampling_hz=20.0,
            output_dir=paths["output"],
            step_source_factory=lambda _config: AlwaysEffectiveSource(),
        )
