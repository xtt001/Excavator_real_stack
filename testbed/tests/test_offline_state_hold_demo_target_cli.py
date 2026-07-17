from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import yaml

from testbed.cli.offline_state_hold_demo_target import (
    Hdf5EpisodeObservations,
    build_offline_policy_config,
    resolve_candidate_artifacts,
    resolve_mechanical_deadzone,
    run_offline_state_hold_demo_target,
)


def _policy_config() -> dict[str, Any]:
    return {
        "bundle_dir": "old_bundle",
        "camera_names": ["video4", "video5"],
        "output_mode": "shadow_zero",
        "qvel_mode": "raw",
        "action_scale": [1.0, 1.0, 1.0, 0.75],
        "fail_safe_zero": True,
        "deadzone_assist": {
            "enabled": False,
            "trigger_fraction": 0.5,
            "min_consecutive_steps": 2,
            "margin": [0.02, 0.02, 0.02, 0.02],
            "deadzone_positive": [0.661, 0.259, 0.500, 0.408],
            "deadzone_negative": [0.721, 0.357, 0.500, 0.508],
        },
        "runtime_gates": {
            "enabled": True,
            "bundle_dir": "old_bundle",
            "manifest_path": "old_bundle/candidate_package_manifest.json",
            "deadzone_json": "deadzone_policy_raw_for_runtime_scale.json",
            "snap_epsilon": 0.001,
        },
    }


def _write_episode(path: Path, *, camera_names: tuple[str, ...]) -> None:
    action = np.zeros((4, 4), dtype=np.float32)
    action[0, 0] = 0.8
    action[2, 1] = -0.8
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("action", data=action)
        h5_file.create_dataset(
            "observations/qpos",
            data=np.arange(16, dtype=np.float32).reshape(4, 4),
        )
        h5_file.create_dataset(
            "observations/qvel",
            data=np.arange(16, dtype=np.float32).reshape(4, 4) + 0.25,
        )
        for camera_index, camera_name in enumerate(camera_names):
            h5_file.create_dataset(
                f"observations/images/{camera_name}",
                data=np.full(
                    (4, 2, 3, 3),
                    10 + camera_index,
                    dtype=np.uint8,
                ),
            )


def _write_stub_bundle(path: Path, *, low_dim_keys: list[str] | None = None) -> None:
    path.mkdir()
    (path / "policy_best.ckpt").write_bytes(b"stub")
    (path / "dataset_stats.pkl").write_bytes(b"stub")
    (path / "resolved_config.yaml").write_text(
        yaml.safe_dump({"policy": {"low_dim_keys": low_dim_keys or ["qpos"]}}),
        encoding="utf-8",
    )


class AlwaysEffectiveSource:
    def reset(self) -> None:
        pass

    def step(self, observation: Any) -> np.ndarray:
        del observation
        return np.array([0.9, -0.9, 0.9, -0.9], dtype=np.float32)


def test_resolve_mechanical_deadzone_uses_direct_assist_arrays(tmp_path: Path) -> None:
    config_path = tmp_path / "e52.yaml"
    config_path.write_text("teleop: {}\n", encoding="utf-8")

    thresholds, payload = resolve_mechanical_deadzone(
        policy_config=_policy_config(),
        config_path=config_path,
    )

    assert thresholds["swing"] == {"pos": 0.661, "neg": 0.721}
    assert thresholds["bucket"] == {"pos": 0.408, "neg": 0.508}
    assert payload["metadata"]["action_domain"] == "direct_policy_output"
    assert payload["metadata"]["policy_action_scale"] == [1.0, 1.0, 1.0, 1.0]
    assert payload["metadata"]["legacy_raw_scaled_deadzone_reused"] is False

    invalid = _policy_config()
    invalid["deadzone_assist"]["deadzone_positive"][2] = float("nan")
    with pytest.raises(ValueError, match="finite positive"):
        resolve_mechanical_deadzone(
            policy_config=invalid,
            config_path=config_path,
        )


def test_build_offline_config_enforces_identity_scale_and_records_overrides(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    manifest = tmp_path / "candidate.json"
    deadzone = tmp_path / "direct_deadzone.json"
    gate_artifacts = {
        "phase_gate_model": tmp_path / "phase.pt",
        "tail_candidate_model": tmp_path / "tail.pt",
    }

    resolved, overrides = build_offline_policy_config(
        policy_config=_policy_config(),
        bundle_dir=bundle,
        candidate_manifest_path=manifest,
        deadzone_path=deadzone,
        gate_artifact_paths=gate_artifacts,
        assist_enabled=True,
        device="cpu",
        temporal_aggregation_diagnostics=True,
    )

    assert resolved["output_mode"] == "control"
    assert resolved["action_scale"] == [1.0, 1.0, 1.0, 1.0]
    assert resolved["fail_safe_zero"] is False
    assert resolved["deadzone_assist"]["enabled"] is True
    assert resolved["runtime_gates"]["deadzone_json"] == str(deadzone)
    assert resolved["runtime_gates"]["manifest_path"] == str(manifest)
    assert resolved["runtime_gates"]["artifacts"] == {
        name: str(path) for name, path in gate_artifacts.items()
    }
    assert resolved["device"] == "cpu"
    assert resolved["temporal_aggregation_diagnostics"] is True
    assert overrides["configured"]["action_scale"] == [1.0, 1.0, 1.0, 0.75]
    assert overrides["offline"]["action_scale"] == [1.0, 1.0, 1.0, 1.0]
    assert overrides["offline"]["temporal_aggregation_diagnostics"] is True


def test_build_raw_offline_config_removes_runtime_gates(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    deadzone = tmp_path / "direct_deadzone.json"

    resolved, overrides = build_offline_policy_config(
        policy_config=_policy_config(),
        bundle_dir=bundle,
        candidate_manifest_path=None,
        deadzone_path=deadzone,
        gate_artifact_paths={},
        assist_enabled=True,
        device="cpu",
        pipeline_mode="raw",
    )

    assert resolved["output_mode"] == "control"
    assert resolved["action_scale"] == [1.0, 1.0, 1.0, 1.0]
    assert resolved["fail_safe_zero"] is False
    assert resolved["deadzone_assist"]["enabled"] is True
    assert resolved["runtime_gates"] == {"enabled": False}
    assert overrides["offline"]["runtime_gates.enabled"] is False
    assert overrides["offline"]["runtime_gates.removed_for_raw_policy"] is True


def test_hdf5_observation_sequence_is_lazy_and_preserves_configured_cameras(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode_001.hdf5"
    _write_episode(episode, camera_names=("video4", "video5"))

    with h5py.File(episode, "r") as h5_file:
        observations = Hdf5EpisodeObservations(
            h5_file,
            camera_names=["video4", "video5"],
        )
        row = observations[2]

        assert len(observations) == 4
        np.testing.assert_array_equal(row["qpos"], np.arange(8, 12, dtype=np.float32))
        np.testing.assert_array_equal(
            row["qvel"], np.arange(8, 12, dtype=np.float32) + 0.25
        )
        assert row["image_video4"].shape == (2, 3, 3)
        assert int(row["image_video4"][0, 0, 0]) == 10
        assert int(row["image_video5"][0, 0, 0]) == 11


def test_hdf5_state_hold_preflight_reads_shapes_without_decoding_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = tmp_path / "episode_001.hdf5"
    _write_episode(episode, camera_names=("video4", "video5"))
    decode_calls = 0

    def fail_if_decoded(*_args, **_kwargs):
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError("camera loader must not run during structural preflight")

    monkeypatch.setattr(
        "testbed.cli.offline_state_hold_demo_target._read_camera_image",
        fail_if_decoded,
    )
    with h5py.File(episode, "r") as h5_file:
        observations = Hdf5EpisodeObservations(
            h5_file,
            camera_names=["video4", "video5"],
        )
        observations.validate_state_hold_structure(required_steps=4)

    assert decode_calls == 0
    assert observations.decode_count == 0


def test_hdf5_observation_sequence_adds_validated_previous_final_command(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode_001.hdf5"
    _write_episode(episode, camera_names=("video4", "video5"))
    previous_command = np.arange(16, dtype=np.float32).reshape(4, 4) / 10.0

    with h5py.File(episode, "r") as h5_file:
        observations = Hdf5EpisodeObservations(
            h5_file,
            camera_names=["video4", "video5"],
            previous_final_command=previous_command,
        )
        np.testing.assert_array_equal(
            observations[2]["previous_final_command"],
            previous_command[2],
        )
        with pytest.raises(ValueError, match="must have shape"):
            Hdf5EpisodeObservations(
                h5_file,
                camera_names=["video4", "video5"],
                previous_final_command=previous_command[:-1],
            )


@pytest.mark.parametrize(
    ("low_dim_keys", "manifest_path", "message"),
    [
        (
            ["qpos", "qvel", "previous_final_command"],
            None,
            "is required",
        ),
        (["qpos"], "unused.json", "is forbidden"),
    ],
)
def test_execution_feedback_manifest_presence_must_match_bundle_contract(
    tmp_path: Path,
    low_dim_keys: list[str],
    manifest_path: str | None,
    message: str,
) -> None:
    bundle = tmp_path / "bundle"
    _write_stub_bundle(bundle, low_dim_keys=low_dim_keys)
    config = tmp_path / "field.yaml"
    config.write_text("teleop: {}\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    with pytest.raises(ValueError, match=message):
        run_offline_state_hold_demo_target(
            config_path=config,
            bundle_dir=bundle,
            candidate_manifest_path=None,
            dataset_dir=dataset,
            episode_ids=[1],
            hold_horizon_steps=1,
            output_dir=tmp_path / "report",
            pipeline_mode="raw",
            execution_feedback_manifest_path=(
                tmp_path / manifest_path if manifest_path is not None else None
            ),
        )


def test_candidate_artifacts_are_bound_to_explicit_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    action_entries = {
        "action_policy_best": "policy_best.ckpt",
        "action_dataset_stats": "dataset_stats.pkl",
        "action_resolved_config": "resolved_config.yaml",
    }
    gate_entries = {
        "phase_gate_model": "phase.pt",
        "tail_candidate_model": "tail.pt",
        "gohome_eligibility_model": "eligibility.pt",
        "temporal_direction_model": "temporal.pt",
        "temporal_direction_metadata": "temporal.json",
    }
    for filename in (*action_entries.values(), *gate_entries.values()):
        (bundle / filename).write_bytes(filename.encode())
    manifest = {
        "artifacts": [
            {"name": name, "path": f"/stale/original/{filename}"}
            for name, filename in {**action_entries, **gate_entries}.items()
        ]
    }

    paths = resolve_candidate_artifacts(
        candidate_manifest=manifest,
        bundle_dir=bundle,
    )

    assert paths == {
        name: (bundle / filename).resolve() for name, filename in gate_entries.items()
    }


def test_orchestration_writes_assist_ab_reports_without_loading_checkpoint(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _write_stub_bundle(bundle)
    gate_entries = {
        "phase_gate_model": "phase.pt",
        "tail_candidate_model": "tail.pt",
        "gohome_eligibility_model": "eligibility.pt",
        "temporal_direction_model": "temporal.pt",
        "temporal_direction_metadata": "temporal.json",
    }
    for name in gate_entries.values():
        (bundle / name).write_bytes(b"stub")
    manifest = tmp_path / "candidate_package_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "candidate_id": "E52",
                "artifacts": [
                    {"name": "action_policy_best", "path": "policy_best.ckpt"},
                    {"name": "action_dataset_stats", "path": "dataset_stats.pkl"},
                    {
                        "name": "action_resolved_config",
                        "path": "resolved_config.yaml",
                    },
                    *[
                        {"name": name, "path": filename}
                        for name, filename in gate_entries.items()
                    ],
                ],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "e52.yaml"
    config.write_text(
        yaml.safe_dump({"teleop": {"policy": _policy_config()}}),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_episode(dataset / "episode_1.hdf5", camera_names=("video4", "video5"))
    captured_configs: list[dict[str, Any]] = []

    def source_factory(policy_config: dict[str, Any]) -> AlwaysEffectiveSource:
        captured_configs.append(policy_config)
        return AlwaysEffectiveSource()

    result = run_offline_state_hold_demo_target(
        config_path=config,
        bundle_dir=bundle,
        candidate_manifest_path=manifest,
        dataset_dir=dataset,
        episode_ids=["001"],
        hold_horizon_steps=3,
        output_dir=tmp_path / "report",
        device="cpu",
        assist_mode="both",
        step_source_factory=source_factory,
    )

    assert len(captured_configs) == 2
    assert [cfg["deadzone_assist"]["enabled"] for cfg in captured_configs] == [
        False,
        True,
    ]
    assert all(cfg["action_scale"] == [1.0, 1.0, 1.0, 1.0] for cfg in captured_configs)
    assert all(cfg["output_mode"] == "control" for cfg in captured_configs)
    assert all(cfg["fail_safe_zero"] is False for cfg in captured_configs)
    assert all(
        "deadzone_policy_raw_for_runtime_scale.json"
        not in cfg["runtime_gates"]["deadzone_json"]
        for cfg in captured_configs
    )
    summary = json.loads(result["run_summary"].read_text())
    assert summary["assist_mode"] == "both"
    assert [item["mode"] for item in summary["reports"]] == [
        "assist_disabled",
        "assist_enabled",
    ]
    assert all(item["anchor_rows"] == 2 for item in summary["reports"])
    disabled_summary = json.loads(
        (tmp_path / "report/assist_disabled/state_hold_summary.json").read_text()
    )
    assert disabled_summary["metadata"]["offline_overrides"]["offline"][
        "action_scale"
    ] == [1.0, 1.0, 1.0, 1.0]
    assert (
        disabled_summary["metadata"]["deadzone_provenance"][
            "legacy_raw_scaled_deadzone_reused"
        ]
        is False
    )


def test_raw_orchestration_omits_gate_manifest_and_writes_assist_ab(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "h1_bundle"
    _write_stub_bundle(bundle)
    config = tmp_path / "field.yaml"
    config.write_text(
        yaml.safe_dump({"teleop": {"policy": _policy_config()}}),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_episode(dataset / "episode_1.hdf5", camera_names=("video4", "video5"))
    captured_configs: list[dict[str, Any]] = []

    def source_factory(policy_config: dict[str, Any]) -> AlwaysEffectiveSource:
        captured_configs.append(policy_config)
        return AlwaysEffectiveSource()

    result = run_offline_state_hold_demo_target(
        config_path=config,
        bundle_dir=bundle,
        candidate_manifest_path=None,
        dataset_dir=dataset,
        episode_ids=["001"],
        hold_horizon_steps=3,
        output_dir=tmp_path / "raw_report",
        device="cpu",
        assist_mode="both",
        pipeline_mode="raw",
        candidate_id="H1_direct_relabel",
        decompose_temporal_aggregation=True,
        step_source_factory=source_factory,
    )

    assert len(captured_configs) == 2
    assert all(cfg["runtime_gates"] == {"enabled": False} for cfg in captured_configs)
    assert all(cfg["action_scale"] == [1.0, 1.0, 1.0, 1.0] for cfg in captured_configs)
    assert all(cfg["temporal_aggregation_diagnostics"] for cfg in captured_configs)
    summary = json.loads(result["run_summary"].read_text())
    assert summary["candidate_id"] == "H1_direct_relabel"
    assert summary["pipeline_mode"] == "raw"
    assert summary["candidate_manifest"] is None
    assert summary["verified_gate_artifacts"] == {}
    assert summary["temporal_aggregation_decomposition"] is True
    assert [item["mode"] for item in summary["reports"]] == [
        "assist_disabled",
        "assist_enabled",
    ]
