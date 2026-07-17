"""Run observe-only warmup -> arm -> frozen-observation startup evaluation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.cli.offline_state_hold_demo_target import (
    Hdf5EpisodeObservations,
    _bundle_low_dim_keys,
    _camera_names,
    _default_step_source_factory,
    _low_dim_keys,
    _policy_config,
    _read_mapping_yaml,
    _required_directory,
    _required_file,
    _verify_policy_bundle,
    build_offline_policy_config,
)
from testbed.data.expert_intent_events import (
    EVENTS_FILENAME,
    MANIFEST_FILENAME,
    SEALED_TEST_EPISODE_IDS,
    sha256_file,
)
from testbed.data.expert_intent_events import (
    SCHEMA_VERSION as EVENT_SCHEMA_VERSION,
)
from testbed.policies.deadzone_eval import load_deadzone_thresholds
from testbed.policies.startup_activation import (
    SCHEMA_VERSION,
    aggregate_startup_activation_rows,
    capability_boundaries,
    evaluate_startup_activation,
)
from testbed.policies.state_hold_demo_target import StatefulStepSource

REPORT_FILENAME = "startup_activation_report.json"
ROWS_FILENAME = "startup_activation_rows.jsonl"
ROWS_CSV_FILENAME = "startup_activation_rows.csv"
SOURCE_MANIFEST_FILENAME = "startup_activation_source_manifest.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.offline_startup_activation",
        description=(
            "Warm raw policy state with suppressed commands, explicitly arm on the "
            "last expert-ineffective frame reference, and freeze that observation."
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--hold-horizon-steps", type=int, required=True)
    parser.add_argument("--sampling-hz", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    result = run_offline_startup_activation(
        model=args.model,
        config_path=args.config,
        bundle_dir=args.bundle_dir,
        dataset_dir=args.dataset_dir,
        event_dir=args.event_dir,
        deadzone_json=args.deadzone_json,
        hold_horizon_steps=args.hold_horizon_steps,
        sampling_hz=args.sampling_hz,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def run_offline_startup_activation(
    *,
    model: str,
    config_path: str | Path,
    bundle_dir: str | Path,
    dataset_dir: str | Path,
    event_dir: str | Path,
    deadzone_json: str | Path,
    hold_horizon_steps: int,
    sampling_hz: float,
    output_dir: str | Path,
    device: str | None = None,
    step_source_factory: Callable[[dict[str, Any]], StatefulStepSource] | None = None,
) -> dict[str, Any]:
    """Validate provenance, run exact validation coverage, and write atomically."""

    model_label = str(model).strip()
    if not model_label:
        raise ValueError("model must not be empty")
    horizon = int(hold_horizon_steps)
    if horizon <= 0:
        raise ValueError("hold_horizon_steps must be positive")
    hz = float(sampling_hz)
    if not np.isfinite(hz) or hz <= 0.0:
        raise ValueError("sampling_hz must be finite and positive")

    config = _required_file(config_path, "config")
    bundle = _required_directory(bundle_dir, "bundle_dir")
    dataset = _required_directory(dataset_dir, "dataset_dir")
    events_root = _required_directory(event_dir, "event_dir")
    deadzone = _required_file(deadzone_json, "deadzone_json")
    event_manifest_path = _required_file(
        events_root / MANIFEST_FILENAME, "event manifest"
    )
    events_path = _required_file(events_root / EVENTS_FILENAME, "event JSONL")
    _verify_policy_bundle(bundle)

    event_manifest = _read_json_mapping(event_manifest_path)
    if event_manifest.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported ExpertIntentEvent manifest schema")
    if (
        Path(str(event_manifest.get("dataset_dir", ""))).expanduser().resolve()
        != dataset
    ):
        raise ValueError("event manifest dataset_dir does not match requested dataset")
    expected_event_hash = event_manifest.get("artifacts", {}).get(EVENTS_FILENAME)
    actual_event_hash = sha256_file(events_path)
    if not expected_event_hash or expected_event_hash != actual_event_hash:
        raise ValueError("ExpertIntentEvent JSONL hash does not match manifest")
    validation_ids = [int(value) for value in event_manifest.get("validation_ids", [])]
    if not validation_ids or len(validation_ids) != len(set(validation_ids)):
        raise ValueError("event manifest validation_ids must be unique and nonempty")

    # This complete rejection happens before any HDF5 path is opened or hashed.
    _reject_sealed_ids(validation_ids, context="composite validation IDs")
    validation_sources = _validation_sources(event_manifest, validation_ids)
    _reject_sealed_source_paths(validation_sources)
    first_events = _read_first_validation_events(events_path)
    if set(first_events) != set(validation_ids):
        raise ValueError(
            "first-event episode IDs do not exactly match manifest validation IDs"
        )

    thresholds = load_deadzone_thresholds(deadzone)
    if _normalized_thresholds(event_manifest.get("thresholds")) != thresholds:
        raise ValueError("deadzone thresholds differ from ExpertIntentEvent manifest")
    if event_manifest.get("threshold_source_sha256") != sha256_file(deadzone):
        raise ValueError("deadzone hash differs from ExpertIntentEvent manifest")

    root_config = _read_mapping_yaml(config)
    policy_config = _policy_config(root_config)
    camera_names = _camera_names(policy_config)
    bundle_resolved_path = bundle / "resolved_config.yaml"
    bundle_resolved = _read_mapping_yaml(bundle_resolved_path)
    bundle_policy = bundle_resolved.get("policy")
    if not isinstance(bundle_policy, Mapping):
        raise ValueError("bundle resolved_config.yaml policy must be a mapping")
    bundle_cameras = _bundle_camera_names(bundle_resolved)
    if camera_names != bundle_cameras:
        raise ValueError(
            "teleop.policy.camera_names must match bundle camera order: "
            f"{camera_names} != {bundle_cameras}"
        )
    configured_low_dim = _low_dim_keys(policy_config, source="teleop.policy")
    bundle_low_dim = _bundle_low_dim_keys(bundle)
    if configured_low_dim != bundle_low_dim:
        raise ValueError(
            "teleop.policy.low_dim_keys must match action bundle: "
            f"{configured_low_dim} != {bundle_low_dim}"
        )
    if str(policy_config.get("qvel_mode", "raw")) != "raw":
        raise ValueError("startup activation requires teleop.policy.qvel_mode='raw'")

    resolved_policy, overrides = build_offline_policy_config(
        policy_config=policy_config,
        bundle_dir=bundle,
        candidate_manifest_path=None,
        deadzone_path=deadzone,
        gate_artifact_paths={},
        assist_enabled=False,
        device=device,
        pipeline_mode="raw",
    )
    _validate_raw_overrides(resolved_policy)

    episode_paths: dict[int, Path] = {}
    hdf5_sources: list[dict[str, Any]] = []
    for episode_id in validation_ids:
        path = _required_file(dataset / f"episode_{episode_id}.hdf5", "validation HDF5")
        source = validation_sources[episode_id]
        if path != Path(str(source["path"])).expanduser().resolve():
            raise ValueError(
                f"episode_{episode_id} source path differs from event manifest"
            )
        actual_hash = sha256_file(path)
        if actual_hash != str(source.get("sha256", "")):
            raise ValueError(
                f"episode_{episode_id} HDF5 hash differs from event manifest"
            )
        episode_paths[episode_id] = path
        hdf5_sources.append(
            {
                "episode_id": episode_id,
                "path": str(path),
                "sha256": actual_hash,
                "source_episode_id": _source_episode_id(path),
            }
        )

    source_factory = step_source_factory or _default_step_source_factory
    step_source = source_factory(resolved_policy)
    rows: list[dict[str, Any]] = []
    performance: dict[str, Any] = {
        "source_step_calls": 0,
        "jpeg_decodes": 0,
        "wall_time_seconds": 0.0,
        "episodes": {},
    }
    try:
        for episode_id in validation_ids:
            counters: dict[str, Any] = {}
            started = time.perf_counter()
            with h5py.File(episode_paths[episode_id], "r") as handle:
                action_length = int(handle["action"].shape[0])
                previous_command = (
                    np.zeros((action_length, len(thresholds)), dtype=np.float32)
                    if "previous_final_command" in bundle_low_dim
                    else None
                )
                observations = Hdf5EpisodeObservations(
                    handle,
                    camera_names=camera_names,
                    previous_final_command=previous_command,
                )
                decode_before = observations.decode_count
                row = evaluate_startup_activation(
                    episode_id=episode_id,
                    first_event=first_events[episode_id],
                    observations=observations,
                    thresholds=thresholds,
                    step_source=step_source,
                    hold_horizon_steps=horizon,
                    sampling_hz=hz,
                    instrumentation=counters,
                )
                counters["jpeg_decodes"] = observations.decode_count - decode_before
            elapsed = time.perf_counter() - started
            counters["wall_time_seconds"] = elapsed
            rows.append(row)
            performance["source_step_calls"] += int(
                counters.get("source_step_calls", 0)
            )
            performance["jpeg_decodes"] += int(counters["jpeg_decodes"])
            performance["wall_time_seconds"] += elapsed
            performance["episodes"][str(episode_id)] = counters
    finally:
        close = getattr(step_source, "close", None)
        if callable(close):
            close()

    report = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "observe_only_warmup_armed_frozen_startup_activation",
        "model": model_label,
        "split": "validation",
        "episode_count": len(rows),
        "validation_ids": validation_ids,
        "hold_horizon_steps": horizon,
        "sampling_hz": hz,
        "algorithm_semantics": {
            "arm_step": "max(0, single_demo_first_onset_step - 1)",
            "arm_reference": "last_single_demo_ineffective_frame_reference",
            "warmup": "observations [0, arm_step), outputs ignored and commands suppressed",
            "post_arm": "repeat only observation arm_step with zero qvel and previous command",
            "termination": "first deadzone-effective action on any axis, else horizon",
            "delay": "zero-based; within_N_ticks means activation_delay_ticks < N",
            "startup_axis_requirement": "none",
            "single_demo_similarity_only": True,
            "promotion_gate": False,
            "safety_gate": False,
        },
        "capability_boundaries": capability_boundaries(),
        "aggregate": aggregate_startup_activation_rows(rows),
        "source_manifest": SOURCE_MANIFEST_FILENAME,
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": model_label,
        "split": "validation",
        "validation_ids": validation_ids,
        "config": {"path": str(config), "sha256": sha256_file(config)},
        "bundle": {
            "path": str(bundle),
            "policy_best.ckpt_sha256": sha256_file(bundle / "policy_best.ckpt"),
            "dataset_stats.pkl_sha256": sha256_file(bundle / "dataset_stats.pkl"),
            "resolved_config.yaml_path": str(bundle_resolved_path),
            "resolved_config.yaml_sha256": sha256_file(bundle_resolved_path),
            "camera_names": camera_names,
            "low_dim_keys": bundle_low_dim,
        },
        "event_manifest": {
            "path": str(event_manifest_path),
            "sha256": sha256_file(event_manifest_path),
        },
        "event_jsonl": {"path": str(events_path), "sha256": actual_event_hash},
        "deadzone_json": {"path": str(deadzone), "sha256": sha256_file(deadzone)},
        "thresholds": thresholds,
        "source_hdf5": hdf5_sources,
        "resolved_offline_policy_config": _jsonable(resolved_policy),
        "offline_overrides": overrides,
        "performance_counters": performance,
        "implementation": _implementation_hashes(),
        "source_hdf5_open_mode": "read_only",
        "policy_inference_performed": True,
        "model_command_sent": False,
        "model_command_changed": False,
        "sealed_test_data_read": False,
        "startup_axis_requirement": "none",
        "single_demo_similarity_only": True,
        "promotion_gate": False,
        "safety_gate": False,
    }
    _write_json_atomic(output / SOURCE_MANIFEST_FILENAME, source_manifest)
    _write_jsonl_atomic(output / ROWS_FILENAME, rows)
    _write_rows_csv_atomic(output / ROWS_CSV_FILENAME, rows)
    _write_json_atomic(output / REPORT_FILENAME, report)
    return {
        "report": str(output / REPORT_FILENAME),
        "report_sha256": sha256_file(output / REPORT_FILENAME),
        "episode_rows": len(rows),
        "model": model_label,
    }


def _read_first_validation_events(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"event line {line_number} is not a mapping")
            if (
                payload.get("split") != "validation"
                or int(payload.get("event_index", -1)) != 0
            ):
                continue
            episode_id = int(payload["episode_id"])
            if episode_id in result:
                raise ValueError(f"episode_{episode_id} has duplicate first events")
            result[episode_id] = payload
    if not result:
        raise ValueError("ExpertIntentEvent JSONL has no validation first events")
    return result


def _validation_sources(
    manifest: Mapping[str, Any], validation_ids: Sequence[int]
) -> dict[int, Mapping[str, Any]]:
    requested = set(validation_ids)
    result: dict[int, Mapping[str, Any]] = {}
    for raw in manifest.get("episodes", []):
        if not isinstance(raw, Mapping):
            raise ValueError("event manifest episode source must be a mapping")
        episode_id = int(raw.get("episode_id", -1))
        if episode_id not in requested:
            continue
        if str(raw.get("split")) != "validation" or episode_id in result:
            raise ValueError("event manifest validation source roles are invalid")
        result[episode_id] = raw
    if set(result) != requested:
        raise ValueError("event manifest sources do not exactly cover validation IDs")
    return result


def _reject_sealed_source_paths(sources: Mapping[int, Mapping[str, Any]]) -> None:
    source_ids = [
        _source_episode_id(Path(str(source.get("path", ""))))
        for source in sources.values()
    ]
    _reject_sealed_ids(source_ids, context="source episode IDs")


def _source_episode_id(path: Path) -> int:
    match = re.fullmatch(r"episode_(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"source path has no explicit episode ID: {path}")
    return int(match.group(1))


def _reject_sealed_ids(values: Sequence[int], *, context: str) -> None:
    forbidden = sorted(set(int(value) for value in values) & SEALED_TEST_EPISODE_IDS)
    if forbidden:
        raise ValueError(f"{context} contains sealed/test episode IDs: {forbidden}")


def _validate_raw_overrides(config: Mapping[str, Any]) -> None:
    if list(config.get("action_scale", [])) != [1.0, 1.0, 1.0, 1.0]:
        raise ValueError("offline startup activation requires identity action scale")
    if bool(dict(config.get("deadzone_assist", {}) or {}).get("enabled", False)):
        raise ValueError("offline startup activation requires deadzone assist disabled")
    if bool(dict(config.get("runtime_gates", {}) or {}).get("enabled", False)):
        raise ValueError("offline startup activation requires runtime gates disabled")
    if bool(config.get("fail_safe_zero", True)):
        raise ValueError("offline startup activation requires fail_safe_zero=false")


def _normalized_thresholds(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping):
        raise ValueError("event manifest thresholds must be a mapping")
    return {
        str(axis): {str(sign): float(threshold) for sign, threshold in raw.items()}
        for axis, raw in value.items()
        if isinstance(raw, Mapping)
    }


def _bundle_camera_names(resolved_config: Mapping[str, Any]) -> list[str]:
    task = resolved_config.get("task")
    if not isinstance(task, Mapping):
        raise ValueError("bundle resolved_config.yaml task must be a mapping")
    camera_names = _camera_names(task)
    contract = resolved_config.get("experiment_contract")
    if (
        isinstance(contract, Mapping)
        and contract.get("expected_camera_names") is not None
    ):
        expected = _camera_names({"camera_names": contract["expected_camera_names"]})
        if camera_names != expected:
            raise ValueError(
                "bundle task.camera_names differs from experiment contract: "
                f"{camera_names} != {expected}"
            )
    return camera_names


def _implementation_hashes() -> list[dict[str, str]]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "policies" / "startup_activation.py",
        root / "policies" / "state_hold_demo_target.py",
        root / "cli" / "offline_state_hold_demo_target.py",
    )
    return [{"path": str(path), "sha256": sha256_file(path)} for path in paths]


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text_atomic(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
    )


def _write_rows_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "episode_id",
        "event_id",
        "single_demo_first_onset_step",
        "arm_step",
        "warmup_ticks",
        "warmup_any_effective_output",
        "status",
        "natural_liveness",
        "activation_delay_ticks",
        "activation_delay_seconds",
        "first_action_vector",
        "first_direction_set",
        "startup_axis_requirement",
        "single_demo_similarity_only",
        "promotion_gate",
        "safety_gate",
        "single_demo_similarity",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        output = {field: row[field] for field in fields}
        for field in (
            "first_action_vector",
            "first_direction_set",
            "single_demo_similarity",
        ):
            output[field] = json.dumps(
                output[field], ensure_ascii=False, sort_keys=True
            )
        writer.writerow(output)
    _write_text_atomic(path, stream.getvalue())


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
