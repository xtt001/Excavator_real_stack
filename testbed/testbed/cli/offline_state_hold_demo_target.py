"""Run deadzone-coupled state-hold diagnostics on raw or gated policies."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from testbed.actions.policy import PolicyActionSource
from testbed.data.dataset import _read_camera_image
from testbed.data.execution_feedback import (
    load_execution_feedback_sidecar,
    sha256_file,
    validate_execution_feedback_manifest,
)
from testbed.data.schema import GRP_ENCODED_IMAGES, GRP_IMAGES
from testbed.policies.deadzone_eval import load_deadzone_thresholds
from testbed.policies.offline_eval import (
    AXIS_NAMES,
    episode_path,
    normalize_episode_id,
)
from testbed.policies.state_hold_demo_target import (
    RuntimeActionStepSource,
    StatefulStepSource,
    evaluate_state_hold_demo_target,
    write_state_hold_demo_target_report,
)

ASSIST_MODES = ("disabled", "enabled", "both")
PIPELINE_MODES = ("gated", "raw")
IDENTITY_ACTION_SCALE = [1.0, 1.0, 1.0, 1.0]
_POLICY_BUNDLE_FILES = (
    "policy_best.ckpt",
    "dataset_stats.pkl",
    "resolved_config.yaml",
)
_ACTION_MANIFEST_FILES = {
    "action_policy_best": "policy_best.ckpt",
    "action_dataset_stats": "dataset_stats.pkl",
    "action_resolved_config": "resolved_config.yaml",
}
_GATE_MANIFEST_NAMES = (
    "phase_gate_model",
    "tail_candidate_model",
    "gohome_eligibility_model",
    "temporal_direction_model",
    "temporal_direction_metadata",
)


class Hdf5EpisodeObservations(Sequence[Mapping[str, Any]]):
    """Lazy observation sequence backed by an open HDF5 episode."""

    def __init__(
        self,
        h5_file: h5py.File,
        *,
        camera_names: Sequence[str],
        previous_final_command: np.ndarray | None = None,
    ) -> None:
        self._h5_file = h5_file
        self._camera_names = [str(name) for name in camera_names]
        if not self._camera_names:
            raise ValueError("camera_names must not be empty")
        required = ("observations/qpos", "observations/qvel", "action")
        missing = [name for name in required if name not in h5_file]
        if missing:
            raise KeyError("episode is missing dataset(s): " + ", ".join(missing))
        lengths = {name: int(h5_file[name].shape[0]) for name in required}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"episode dataset lengths do not match: {lengths}")
        self._length = lengths["action"]
        self._decode_count = 0
        if self._length <= 0:
            raise ValueError("episode has no steps")
        for name in ("observations/qpos", "observations/qvel", "action"):
            if tuple(h5_file[name].shape[1:]) != (len(AXIS_NAMES),):
                raise ValueError(
                    f"{name} must have shape (T, {len(AXIS_NAMES)}), "
                    f"got {h5_file[name].shape}"
                )
        for camera_name in self._camera_names:
            raw_path = f"{GRP_IMAGES}/{camera_name}"
            encoded_path = f"{GRP_ENCODED_IMAGES}/{camera_name}"
            camera_path = raw_path if raw_path in h5_file else encoded_path
            if camera_path not in h5_file:
                raise KeyError(
                    f"Camera {camera_name!r} not found as raw or encoded image data."
                )
            if int(h5_file[camera_path].shape[0]) != self._length:
                raise ValueError(
                    f"{camera_path} length must be {self._length}, "
                    f"got {h5_file[camera_path].shape[0]}"
                )
        self._previous_final_command: np.ndarray | None = None
        if previous_final_command is not None:
            command = np.asarray(previous_final_command, dtype=np.float32)
            expected_shape = (self._length, len(AXIS_NAMES))
            if command.shape != expected_shape:
                raise ValueError(
                    "previous_final_command must have shape "
                    f"{expected_shape}, got {command.shape}"
                )
            if not np.all(np.isfinite(command)):
                raise ValueError("previous_final_command must contain finite values")
            self._previous_final_command = command

    def __len__(self) -> int:
        return self._length

    @property
    def decode_count(self) -> int:
        return int(self._decode_count)

    def validate_state_hold_structure(self, *, required_steps: int) -> None:
        """Validate qvel coverage using HDF5 metadata without loading images."""

        if int(required_steps) > self._length:
            raise ValueError(
                f"state-hold requires {required_steps} steps, episode has {self._length}"
            )
        qvel = self._h5_file["observations/qvel"]
        expected = (self._length, len(AXIS_NAMES))
        if tuple(qvel.shape) != expected:
            raise ValueError(
                f"observations/qvel must have shape {expected}, got {qvel.shape}"
            )

    def has_observation_key(self, key: str) -> bool:
        if str(key) == "previous_final_command":
            return self._previous_final_command is not None
        return str(key) in {"qpos", "qvel"}

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        if isinstance(index, slice):
            raise TypeError("Hdf5EpisodeObservations does not support slices")
        normalized = int(index)
        if normalized < 0:
            normalized += self._length
        if normalized < 0 or normalized >= self._length:
            raise IndexError(index)
        observation: dict[str, Any] = {
            "qpos": np.asarray(
                self._h5_file["observations/qpos"][normalized], dtype=np.float32
            ),
            "qvel": np.asarray(
                self._h5_file["observations/qvel"][normalized], dtype=np.float32
            ),
        }
        if self._previous_final_command is not None:
            observation["previous_final_command"] = self._previous_final_command[
                normalized
            ].copy()
        for camera_name in self._camera_names:
            observation[f"image_{camera_name}"] = _read_camera_image(
                self._h5_file,
                camera_name,
                normalized,
            )
            self._decode_count += 1
        return observation


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.offline_state_hold_demo_target",
        description=(
            "Run a raw or gated policy with direct policy-output deadzones while "
            "freezing observations after ineffective actions."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--pipeline-mode",
        choices=PIPELINE_MODES,
        default="gated",
        help="Evaluate the configured runtime gates (default) or raw policy output.",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=None,
        help="Required for gated mode and forbidden for raw mode.",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="Raw-mode artifact identity; defaults to the action bundle directory name.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--execution-feedback-manifest",
        type=Path,
        default=None,
        help=(
            "Required exactly when the action bundle consumes previous_final_command."
        ),
    )
    parser.add_argument("--episode-id", action="append", required=True)
    parser.add_argument("--hold-horizon-steps", type=int, required=True)
    parser.add_argument(
        "--trace-full-horizon-after-reproduction",
        action="store_true",
        help=(
            "Continue every branch to the full hold horizon after first recovery "
            "so later wrong motion remains observable."
        ),
    )
    parser.add_argument(
        "--decompose-temporal-aggregation",
        action="store_true",
        help=(
            "Record legacy, newest-chunk, and recency-favoring ACT actions "
            "without changing the selected legacy action."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--assist-mode",
        choices=ASSIST_MODES,
        default="disabled",
        help="Evaluate the selected pipeline alone, with mechanical assist, or both.",
    )
    args = parser.parse_args()
    result = run_offline_state_hold_demo_target(
        config_path=args.config,
        bundle_dir=args.bundle_dir,
        candidate_manifest_path=args.candidate_manifest,
        dataset_dir=args.dataset_dir,
        episode_ids=args.episode_id,
        hold_horizon_steps=args.hold_horizon_steps,
        output_dir=args.output_dir,
        device=args.device,
        assist_mode=args.assist_mode,
        pipeline_mode=args.pipeline_mode,
        candidate_id=args.candidate_id,
        execution_feedback_manifest_path=args.execution_feedback_manifest,
        trace_full_horizon_after_reproduction=(
            args.trace_full_horizon_after_reproduction
        ),
        decompose_temporal_aggregation=bool(args.decompose_temporal_aggregation),
    )
    print(f"State-hold run summary: {result['run_summary']}")


def run_offline_state_hold_demo_target(
    *,
    config_path: str | Path,
    bundle_dir: str | Path,
    candidate_manifest_path: str | Path | None,
    dataset_dir: str | Path,
    episode_ids: Sequence[str | int],
    hold_horizon_steps: int,
    output_dir: str | Path,
    device: str | None = None,
    assist_mode: str = "disabled",
    pipeline_mode: str = "gated",
    candidate_id: str | None = None,
    execution_feedback_manifest_path: str | Path | None = None,
    trace_full_horizon_after_reproduction: bool = False,
    decompose_temporal_aggregation: bool = False,
    step_source_factory: Callable[[dict[str, Any]], StatefulStepSource] | None = None,
) -> dict[str, Any]:
    """Run one or both assist variants and write auditable artifacts."""

    if assist_mode not in ASSIST_MODES:
        raise ValueError(
            f"assist_mode must be one of {ASSIST_MODES}, got {assist_mode!r}"
        )
    if pipeline_mode not in PIPELINE_MODES:
        raise ValueError(
            f"pipeline_mode must be one of {PIPELINE_MODES}, got {pipeline_mode!r}"
        )
    if int(hold_horizon_steps) <= 0:
        raise ValueError("hold_horizon_steps must be positive")
    if not episode_ids:
        raise ValueError("episode_ids must not be empty")

    config_path = _required_file(config_path, "config")
    bundle_dir = _required_directory(bundle_dir, "bundle_dir")
    dataset_dir = _required_directory(dataset_dir, "dataset_dir")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _verify_policy_bundle(bundle_dir)

    bundle_low_dim_keys = _bundle_low_dim_keys(bundle_dir)
    consumes_execution_feedback = "previous_final_command" in bundle_low_dim_keys
    has_execution_feedback_manifest = execution_feedback_manifest_path is not None
    if consumes_execution_feedback != has_execution_feedback_manifest:
        requirement = "required" if consumes_execution_feedback else "forbidden"
        raise ValueError(
            "execution_feedback_manifest_path is "
            f"{requirement} for bundle low_dim_keys={bundle_low_dim_keys}"
        )

    execution_feedback_manifest: dict[str, Any] | None = None
    execution_feedback_records: dict[str, Mapping[str, Any]] = {}
    resolved_execution_feedback_manifest: Path | None = None
    if execution_feedback_manifest_path is not None:
        resolved_execution_feedback_manifest = _required_file(
            execution_feedback_manifest_path,
            "execution_feedback_manifest",
        )
        execution_feedback_manifest = validate_execution_feedback_manifest(
            resolved_execution_feedback_manifest,
            verify_hashes=True,
            expected_dataset_dir=dataset_dir,
        )
        execution_feedback_records = {
            normalize_episode_id(record["episode_id"]): record
            for record in execution_feedback_manifest["episodes"]
        }
    execution_feedback_metadata = {
        "enabled": consumes_execution_feedback,
        "recursive_state_hold": consumes_execution_feedback,
        "bundle_low_dim_keys": bundle_low_dim_keys,
        "manifest_path": (
            str(resolved_execution_feedback_manifest)
            if resolved_execution_feedback_manifest is not None
            else None
        ),
        "manifest_sha256": (
            sha256_file(resolved_execution_feedback_manifest)
            if resolved_execution_feedback_manifest is not None
            else None
        ),
        "alignment_mode": (
            execution_feedback_manifest.get("alignment_mode")
            if execution_feedback_manifest is not None
            else None
        ),
    }

    config = _read_mapping_yaml(config_path)
    policy_config = _policy_config(config)
    if decompose_temporal_aggregation and not bool(
        policy_config.get("temporal_agg", True)
    ):
        raise ValueError(
            "temporal aggregation decomposition requires "
            "teleop.policy.temporal_agg=true"
        )
    configured_low_dim_keys = _low_dim_keys(
        policy_config,
        source="teleop.policy",
    )
    if configured_low_dim_keys != bundle_low_dim_keys:
        raise ValueError(
            "teleop.policy.low_dim_keys must match action bundle: "
            f"{configured_low_dim_keys} != {bundle_low_dim_keys}"
        )
    camera_names = _camera_names(policy_config)
    qvel_mode = str(policy_config.get("qvel_mode", "raw"))
    if qvel_mode != "raw":
        raise ValueError(
            "offline state-hold requires teleop.policy.qvel_mode='raw' so held qvel "
            f"is explicit, got {qvel_mode!r}"
        )
    runtime_gates = dict(policy_config.get("runtime_gates", {}) or {})
    resolved_candidate_manifest: Path | None
    gate_artifact_paths: dict[str, Path]
    if pipeline_mode == "gated":
        if not bool(runtime_gates.get("enabled", False)):
            raise ValueError("teleop.policy.runtime_gates.enabled must be true")
        if candidate_manifest_path is None:
            raise ValueError("candidate_manifest_path is required for gated mode")
        resolved_candidate_manifest = _required_file(
            candidate_manifest_path, "candidate_manifest"
        )
        candidate_manifest = _read_mapping_json(resolved_candidate_manifest)
        manifest_candidate_id = str(candidate_manifest.get("candidate_id", ""))
        if manifest_candidate_id != "E52":
            raise ValueError(
                f"candidate manifest must identify E52, got {manifest_candidate_id!r}"
            )
        if candidate_id is not None and str(candidate_id) != manifest_candidate_id:
            raise ValueError(
                "candidate_id does not match candidate manifest: "
                f"{candidate_id!r} != {manifest_candidate_id!r}"
            )
        resolved_candidate_id = manifest_candidate_id
        gate_artifact_paths = resolve_candidate_artifacts(
            candidate_manifest=candidate_manifest,
            bundle_dir=bundle_dir,
        )
    else:
        if candidate_manifest_path is not None:
            raise ValueError("candidate_manifest_path must be omitted for raw mode")
        resolved_candidate_manifest = None
        resolved_candidate_id = str(candidate_id or bundle_dir.name)
        if not resolved_candidate_id:
            raise ValueError("candidate_id must not be empty")
        gate_artifact_paths = {}

    thresholds, deadzone_payload = resolve_mechanical_deadzone(
        policy_config=policy_config,
        config_path=config_path,
    )
    deadzone_path = output_dir / "resolved_direct_output_deadzone.json"
    deadzone_path.write_text(
        json.dumps(deadzone_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    load_deadzone_thresholds(deadzone_path)

    normalized_episode_ids = list(
        dict.fromkeys(normalize_episode_id(value) for value in episode_ids)
    )
    episode_paths = [
        episode_path(dataset_dir, value) for value in normalized_episode_ids
    ]
    missing_episodes = [path for path in episode_paths if not path.is_file()]
    if missing_episodes:
        raise FileNotFoundError(missing_episodes[0])
    missing_feedback = [
        episode_id
        for episode_id in normalized_episode_ids
        if consumes_execution_feedback and episode_id not in execution_feedback_records
    ]
    if missing_feedback:
        raise ValueError(
            "execution-feedback manifest does not cover requested episode(s): "
            + ", ".join(missing_feedback)
        )

    source_factory = step_source_factory or _default_step_source_factory
    modes = [False, True] if assist_mode == "both" else [assist_mode == "enabled"]
    reports: list[dict[str, Any]] = []
    for assist_enabled in modes:
        resolved_policy, overrides = build_offline_policy_config(
            policy_config=policy_config,
            bundle_dir=bundle_dir,
            candidate_manifest_path=resolved_candidate_manifest,
            deadzone_path=deadzone_path,
            gate_artifact_paths=gate_artifact_paths,
            assist_enabled=assist_enabled,
            device=device,
            pipeline_mode=pipeline_mode,
            temporal_aggregation_diagnostics=bool(decompose_temporal_aggregation),
        )
        step_source = source_factory(resolved_policy)
        rows: list[dict[str, Any]] = []
        performance_counters: dict[str, Any] = {
            "source_step_calls": 0,
            "jpeg_decodes": 0,
            "wall_time_seconds": 0.0,
            "episodes": {},
        }
        try:
            for episode_id, path in zip(normalized_episode_ids, episode_paths):
                with h5py.File(path, "r") as h5_file:
                    previous_final_command: np.ndarray | None = None
                    if consumes_execution_feedback:
                        record = execution_feedback_records[episode_id]
                        sidecar = load_execution_feedback_sidecar(
                            record["sidecar_path"],
                            expected_episode_id=episode_id,
                            expected_length=int(h5_file["action"].shape[0]),
                        )
                        previous_final_command = sidecar.previous_final_command
                    observations = Hdf5EpisodeObservations(
                        h5_file,
                        camera_names=camera_names,
                        previous_final_command=previous_final_command,
                    )
                    expert_action = np.asarray(h5_file["action"][()], dtype=np.float32)
                    episode_counters: dict[str, Any] = {}
                    decode_before = observations.decode_count
                    started = time.perf_counter()
                    episode_rows = evaluate_state_hold_demo_target(
                        episode_id=episode_id,
                        observations=observations,
                        expert_action=expert_action,
                        thresholds=thresholds,
                        step_source=step_source,
                        hold_horizon_steps=int(hold_horizon_steps),
                        trace_full_horizon_after_reproduction=bool(
                            trace_full_horizon_after_reproduction
                        ),
                        instrumentation=episode_counters,
                    )
                    elapsed = time.perf_counter() - started
                    episode_counters["jpeg_decodes"] = (
                        observations.decode_count - decode_before
                    )
                    episode_counters["wall_time_seconds"] = float(elapsed)
                    rows.extend(episode_rows)
                    performance_counters["source_step_calls"] += int(
                        episode_counters.get("source_step_calls", 0)
                    )
                    performance_counters["jpeg_decodes"] += int(
                        episode_counters["jpeg_decodes"]
                    )
                    performance_counters["wall_time_seconds"] += float(elapsed)
                    performance_counters["episodes"][episode_id] = episode_counters
        finally:
            close = getattr(step_source, "close", None)
            if callable(close):
                close()

        mode_name = "assist_enabled" if assist_enabled else "assist_disabled"
        report_dir = output_dir / mode_name
        metadata = {
            "candidate_id": resolved_candidate_id,
            "pipeline_mode": pipeline_mode,
            "config_path": str(config_path),
            "action_bundle_dir": str(bundle_dir),
            "candidate_manifest": (
                str(resolved_candidate_manifest)
                if resolved_candidate_manifest is not None
                else None
            ),
            "verified_gate_artifacts": {
                key: str(value) for key, value in gate_artifact_paths.items()
            },
            "dataset_dir": str(dataset_dir),
            "episode_ids": normalized_episode_ids,
            "camera_names": camera_names,
            "qvel_mode": qvel_mode,
            "execution_feedback": execution_feedback_metadata,
            "hold_horizon_steps": int(hold_horizon_steps),
            "trace_full_horizon_after_reproduction": bool(
                trace_full_horizon_after_reproduction
            ),
            "temporal_aggregation_decomposition": bool(decompose_temporal_aggregation),
            "assist_enabled": assist_enabled,
            "performance_counters": performance_counters,
            "resolved_direct_output_deadzone": str(deadzone_path),
            "deadzone_provenance": deadzone_payload["metadata"],
            "offline_overrides": overrides,
            "limitations": (
                "State-hold measures reproduction of one demo target. It does not "
                "determine generic liveness, correctness, safety, or machine response."
            ),
        }
        paths = write_state_hold_demo_target_report(
            output_dir=report_dir,
            rows=rows,
            metadata=metadata,
        )
        reports.append(
            {
                "mode": mode_name,
                "pipeline_mode": pipeline_mode,
                "assist_enabled": assist_enabled,
                "anchor_rows": len(rows),
                "paths": {key: str(path) for key, path in paths.items()},
                "offline_overrides": overrides,
                "performance_counters": performance_counters,
            }
        )

    run_summary_path = output_dir / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(
            {
                "diagnostic": "single_demo_target_state_hold_counterfactual",
                "candidate_id": resolved_candidate_id,
                "pipeline_mode": pipeline_mode,
                "config_path": str(config_path),
                "action_bundle_dir": str(bundle_dir),
                "candidate_manifest": (
                    str(resolved_candidate_manifest)
                    if resolved_candidate_manifest is not None
                    else None
                ),
                "verified_gate_artifacts": {
                    key: str(value) for key, value in gate_artifact_paths.items()
                },
                "dataset_dir": str(dataset_dir),
                "episode_ids": normalized_episode_ids,
                "execution_feedback": execution_feedback_metadata,
                "hold_horizon_steps": int(hold_horizon_steps),
                "trace_full_horizon_after_reproduction": bool(
                    trace_full_horizon_after_reproduction
                ),
                "temporal_aggregation_decomposition": bool(
                    decompose_temporal_aggregation
                ),
                "assist_mode": assist_mode,
                "resolved_direct_output_deadzone": str(deadzone_path),
                "deadzone_provenance": deadzone_payload["metadata"],
                "reports": reports,
                "limitations": (
                    "State-hold measures reproduction of one demo target. It does not "
                    "determine generic liveness, correctness, safety, or machine response."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "run_summary": run_summary_path,
        "deadzone_json": deadzone_path,
        "reports": reports,
    }


def resolve_mechanical_deadzone(
    *,
    policy_config: Mapping[str, Any],
    config_path: str | Path,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Resolve direct-output thresholds only from mechanical assist arrays."""

    assist = dict(policy_config.get("deadzone_assist", {}) or {})
    positive = _positive_axis_vector(
        assist.get("deadzone_positive"),
        name="teleop.policy.deadzone_assist.deadzone_positive",
    )
    negative = _positive_axis_vector(
        assist.get("deadzone_negative"),
        name="teleop.policy.deadzone_assist.deadzone_negative",
    )
    thresholds = {
        axis: {"pos": float(positive[index]), "neg": float(negative[index])}
        for index, axis in enumerate(AXIS_NAMES)
    }
    payload = {
        "deadzone_action": thresholds,
        "metadata": {
            "source_config": str(Path(config_path).expanduser().resolve()),
            "positive_source": ("teleop.policy.deadzone_assist.deadzone_positive"),
            "negative_source": ("teleop.policy.deadzone_assist.deadzone_negative"),
            "action_domain": "direct_policy_output",
            "policy_action_scale": IDENTITY_ACTION_SCALE,
            "legacy_raw_scaled_deadzone_reused": False,
        },
    }
    return thresholds, payload


def build_offline_policy_config(
    *,
    policy_config: Mapping[str, Any],
    bundle_dir: Path,
    candidate_manifest_path: Path | None,
    deadzone_path: Path,
    gate_artifact_paths: Mapping[str, Path],
    assist_enabled: bool,
    device: str | None,
    pipeline_mode: str = "gated",
    temporal_aggregation_diagnostics: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an in-memory raw/gated config with explicit offline overrides."""

    if pipeline_mode not in PIPELINE_MODES:
        raise ValueError(
            f"pipeline_mode must be one of {PIPELINE_MODES}, got {pipeline_mode!r}"
        )

    resolved = copy.deepcopy(dict(policy_config))
    original = {
        "output_mode": resolved.get("output_mode"),
        "action_scale": copy.deepcopy(resolved.get("action_scale")),
        "fail_safe_zero": resolved.get("fail_safe_zero"),
        "deadzone_assist.enabled": bool(
            dict(resolved.get("deadzone_assist", {}) or {}).get("enabled", False)
        ),
        "runtime_gates.bundle_dir": dict(resolved.get("runtime_gates", {}) or {}).get(
            "bundle_dir"
        ),
        "runtime_gates.manifest_path": dict(
            resolved.get("runtime_gates", {}) or {}
        ).get("manifest_path"),
        "runtime_gates.deadzone_json": dict(
            resolved.get("runtime_gates", {}) or {}
        ).get("deadzone_json"),
        "temporal_aggregation_diagnostics": bool(
            resolved.get("temporal_aggregation_diagnostics", False)
        ),
    }
    resolved.update(
        {
            "bundle_dir": str(bundle_dir),
            "ckpt_path": str(bundle_dir / "policy_best.ckpt"),
            "resolved_config_path": str(bundle_dir / "resolved_config.yaml"),
            "stats_path": str(bundle_dir / "dataset_stats.pkl"),
            "output_mode": "control",
            "action_scale": list(IDENTITY_ACTION_SCALE),
            "fail_safe_zero": False,
            "temporal_aggregation_diagnostics": bool(temporal_aggregation_diagnostics),
        }
    )
    if device is not None:
        resolved["device"] = str(device)
    assist = copy.deepcopy(dict(resolved.get("deadzone_assist", {}) or {}))
    assist["enabled"] = bool(assist_enabled)
    resolved["deadzone_assist"] = assist
    if pipeline_mode == "gated":
        if candidate_manifest_path is None:
            raise ValueError("candidate_manifest_path is required for gated mode")
        runtime = copy.deepcopy(dict(resolved.get("runtime_gates", {}) or {}))
        if not bool(runtime.get("enabled", False)):
            raise ValueError("teleop.policy.runtime_gates.enabled must be true")
        runtime.update(
            {
                "bundle_dir": str(bundle_dir),
                "manifest_path": str(candidate_manifest_path),
                "deadzone_json": str(deadzone_path),
                "artifacts": {
                    **dict(runtime.get("artifacts", {}) or {}),
                    **{name: str(path) for name, path in gate_artifact_paths.items()},
                },
            }
        )
        resolved["runtime_gates"] = runtime
        offline_runtime = {
            "runtime_gates.enabled": True,
            "runtime_gates.bundle_dir": str(bundle_dir),
            "runtime_gates.manifest_path": str(candidate_manifest_path),
            "runtime_gates.deadzone_json": str(deadzone_path),
            "runtime_gates.artifacts": {
                name: str(path) for name, path in gate_artifact_paths.items()
            },
        }
    else:
        resolved["runtime_gates"] = {"enabled": False}
        offline_runtime = {
            "runtime_gates.enabled": False,
            "runtime_gates.removed_for_raw_policy": True,
        }
    overrides = {
        "configured": original,
        "offline": {
            "output_mode": "control",
            "action_scale": list(IDENTITY_ACTION_SCALE),
            "fail_safe_zero": False,
            "deadzone_assist.enabled": bool(assist_enabled),
            "temporal_aggregation_diagnostics": bool(temporal_aggregation_diagnostics),
            **offline_runtime,
        },
        "reason": (
            "Evaluate direct model output without joystick action scaling; use "
            "mechanical deadzones in the same direct-output domain."
        ),
    }
    return resolved, overrides


def resolve_candidate_artifacts(
    *,
    candidate_manifest: Mapping[str, Any],
    bundle_dir: Path,
) -> dict[str, Path]:
    """Bind manifest action/gate artifacts to the explicit portable bundle."""

    artifacts = {
        str(item.get("name", "")): dict(item)
        for item in candidate_manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("name")
    }
    required_names = tuple(_ACTION_MANIFEST_FILES) + _GATE_MANIFEST_NAMES
    missing_entries = [name for name in required_names if name not in artifacts]
    if missing_entries:
        raise ValueError(
            "candidate manifest is missing artifact(s): " + ", ".join(missing_entries)
        )

    resolved_paths: dict[str, Path] = {}
    for name in required_names:
        entry = artifacts[name]
        if name in _ACTION_MANIFEST_FILES:
            path = bundle_dir / _ACTION_MANIFEST_FILES[name]
        else:
            raw_path = entry.get("path")
            if not raw_path:
                raise ValueError(f"candidate artifact {name!r} is missing path")
            path = bundle_dir / Path(str(raw_path)).name
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_sha = str(entry.get("sha256", ""))
        if expected_sha and _sha256(path) != expected_sha:
            raise ValueError(f"candidate artifact sha256 mismatch for {name!r}: {path}")
        resolved_paths[name] = path.resolve()
    return {name: resolved_paths[name] for name in _GATE_MANIFEST_NAMES}


def _default_step_source_factory(policy_config: dict[str, Any]) -> StatefulStepSource:
    return RuntimeActionStepSource(PolicyActionSource.from_config(policy_config))


def _policy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    teleop = config.get("teleop")
    if not isinstance(teleop, Mapping):
        raise ValueError("config teleop must be a mapping")
    policy = teleop.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("config teleop.policy must be a mapping")
    return dict(policy)


def _camera_names(policy_config: Mapping[str, Any]) -> list[str]:
    raw = policy_config.get("camera_names")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("teleop.policy.camera_names must be a sequence")
    names = [str(value) for value in raw]
    if not names or any(not name for name in names):
        raise ValueError("teleop.policy.camera_names must not be empty")
    return names


def _bundle_low_dim_keys(bundle_dir: Path) -> list[str]:
    resolved = _read_mapping_yaml(bundle_dir / "resolved_config.yaml")
    policy = resolved.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("action bundle resolved_config.yaml policy must be a mapping")
    return _low_dim_keys(policy, source="action bundle policy")


def _low_dim_keys(config: Mapping[str, Any], *, source: str) -> list[str]:
    raw = config.get("low_dim_keys", ["qpos"])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{source}.low_dim_keys must be a sequence")
    keys = [str(value) for value in raw]
    if not keys or any(not key for key in keys):
        raise ValueError(f"{source}.low_dim_keys must not be empty")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{source}.low_dim_keys must not contain duplicates")
    return keys


def _positive_axis_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (len(AXIS_NAMES),):
        raise ValueError(
            f"{name} must have shape ({len(AXIS_NAMES)},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array


def _verify_policy_bundle(bundle_dir: Path) -> None:
    missing = [
        bundle_dir / name
        for name in _POLICY_BUNDLE_FILES
        if not (bundle_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "action bundle is missing file(s): "
            + ", ".join(str(path) for path in missing)
        )


def _read_mapping_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _read_mapping_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _required_directory(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


if __name__ == "__main__":
    main()
