#!/usr/bin/env python3
"""Collect qpos -> fixed bucket-tip pairs from the live YuLong Unity rig.

This is an automated kinematic calibration, not a trajectory-replay benchmark.
Historical simulator qpos values are used only to choose task-relevant target
poses.  The current rig is driven through those poses, and every label is
paired with the qpos that Unity actually observed after a simulation step.

The fixed point is measured by ``FixedBucketTipFkCaptureProbe`` in the Unity
project.  It is the frozen center-tooth leading-edge point, not the historical
"lowest point of a measurement box" proxy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


DEFAULT_DATASET = Path(
    "/data/pingfan/excavator_testbed_data/"
    "yulong_v2_2_pro_full_task_four_camera_jpeg_20260717"
)
DEFAULT_UNITY_PROJECT = Path("/home/pingfan/AGXUnityE85ExcavatorSim")
DEFAULT_PACT_ROOT = Path("/home/pingfan/PACT/excavator_testbed")
DEFAULT_OUTPUT = Path("docs/evidence/sim_fixed_tip_fk_v0_2")
EXPECTED_QPOS_ORDER = (
    "swing_position_norm",
    "boom_position_norm",
    "stick_position_norm",
    "bucket_position_norm",
)
SAMPLE_SCHEMA_VERSION = "fixed_tip_fk_sample/v1"
MANIFEST_SCHEMA_VERSION = "fixed_tip_fk_collection/v1"
PROBE_CANDIDATE_ID = "bucket_center_tooth_leading_edge_midpoint_v0_1"


@dataclass(frozen=True)
class TargetCandidate:
    episode_id: int
    step: int
    qpos: np.ndarray


def parse_episode_ids(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("episode list must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"episode list contains duplicates: {text}")
    return values


def episode_id_from_path(path: Path) -> int:
    match = re.search(r"episode_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse episode id from {path}")
    return int(match.group(1))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _metadata_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _env_state_index(handle: h5py.File, field: str) -> int:
    metadata = handle.get("metadata")
    if metadata is None or "env_state_order" not in metadata.attrs:
        raise ValueError(f"{handle.filename}: metadata.env_state_order is required")
    order = tuple(
        part.strip()
        for part in _metadata_text(metadata.attrs["env_state_order"]).split(",")
        if part.strip()
    )
    if field not in order:
        raise ValueError(f"{handle.filename}: env_state field {field!r} is missing")
    return order.index(field)


def load_target_candidates(
    dataset: Path,
    episode_ids: Iterable[int],
    *,
    pool_stride: int,
) -> list[TargetCandidate]:
    if pool_stride < 1:
        raise ValueError("pool_stride must be >= 1")

    candidates: list[TargetCandidate] = []
    for episode_id in episode_ids:
        path = dataset / f"episode_{episode_id}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            qpos = np.asarray(handle["observations/qpos"], dtype=np.float32)
            mask_index = _env_state_index(
                handle, "bucket_dig_area_cell_in_bounds_mask"
            )
            in_bounds = (
                np.asarray(
                    handle["observations/env_state"][:, mask_index],
                    dtype=np.float32,
                )
                > 0.5
            )
            metadata = handle.get("metadata")
            if metadata is not None and "qpos_order" in metadata.attrs:
                order = tuple(
                    part.strip()
                    for part in _metadata_text(metadata.attrs["qpos_order"]).split(",")
                    if part.strip()
                )
                if order != EXPECTED_QPOS_ORDER:
                    raise ValueError(
                        f"{path}: qpos order {order} != {EXPECTED_QPOS_ORDER}"
                    )

        if qpos.ndim != 2 or qpos.shape[1] != 4:
            raise ValueError(f"{path}: expected qpos shape (T, 4), got {qpos.shape}")
        valid_steps = np.flatnonzero(in_bounds & np.isfinite(qpos).all(axis=1))
        for step in valid_steps[::pool_stride]:
            candidates.append(
                TargetCandidate(
                    episode_id=episode_id,
                    step=int(step),
                    qpos=qpos[step].copy(),
                )
            )

    if not candidates:
        raise ValueError(
            f"No finite in-dig-area qpos targets found in {dataset} "
            f"for episodes {tuple(episode_ids)}"
        )
    return candidates


def select_diverse_targets(
    candidates: list[TargetCandidate],
    count: int,
    *,
    seed: int,
) -> list[TargetCandidate]:
    """Greedy farthest-point selection in normalized qpos space."""

    if count < 1:
        raise ValueError("target count must be >= 1")
    if count >= len(candidates):
        return list(candidates)

    values = np.stack([candidate.qpos for candidate in candidates]).astype(np.float64)
    low = np.quantile(values, 0.005, axis=0)
    high = np.quantile(values, 0.995, axis=0)
    scale = np.maximum(high - low, 1.0e-6)
    normalized = np.clip((values - low) / scale, 0.0, 1.0)

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(candidates)))
    selected = [first]
    min_distance_sq = np.sum((normalized - normalized[first]) ** 2, axis=1)
    min_distance_sq[first] = -1.0

    while len(selected) < count:
        next_index = int(np.argmax(min_distance_sq))
        selected.append(next_index)
        distance_sq = np.sum(
            (normalized - normalized[next_index]) ** 2, axis=1
        )
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
        min_distance_sq[selected] = -1.0

    return [candidates[index] for index in selected]


def target_tracking_action(
    current_qpos: np.ndarray,
    target_qpos: np.ndarray,
    *,
    gain: float,
    max_command: float,
    deadband: float,
) -> np.ndarray:
    """Map normalized qpos error to the current YuLong command convention."""

    current = np.asarray(current_qpos, dtype=np.float32).reshape(4)
    target = np.asarray(target_qpos, dtype=np.float32).reshape(4)
    # Positive boom command decreases normalized boom position in this rig.
    command_to_qpos_direction = np.asarray([1.0, -1.0, 1.0, 1.0])
    error = target - current
    action = command_to_qpos_direction * gain * error
    action[np.abs(error) <= deadband] = 0.0
    return np.clip(action, -max_command, max_command).astype(np.float32)


def wait_for_probe_result(
    result_path: Path,
    *,
    sample_id: int,
    timeout_sec: float,
    poll_sec: float = 0.01,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result = None
            if isinstance(result, dict) and result.get("sample_id") == sample_id:
                return result
        time.sleep(poll_sec)
    raise TimeoutError(
        f"Timed out after {timeout_sec:.1f}s waiting for probe sample {sample_id}"
    )


def capture_probe(
    probe_directory: Path,
    *,
    sample_id: int,
    split: str,
    target: TargetCandidate,
    observed_qpos: np.ndarray,
    timeout_sec: float,
) -> dict[str, Any]:
    result_path = probe_directory / "result.json"
    command = {
        "sample_id": sample_id,
        "source_episode": target.episode_id,
        "source_step": target.step,
        "split": split,
        "requested_qpos": target.qpos,
        "observed_qpos": observed_qpos,
        "result_path": str(result_path.resolve()),
    }
    write_json_atomic(probe_directory / "command.json", command)
    result = wait_for_probe_result(
        result_path, sample_id=sample_id, timeout_sec=timeout_sec
    )
    if not result.get("success", False):
        raise RuntimeError(
            f"Unity fixed-tip probe failed for sample {sample_id}: "
            f"{result.get('error', 'unknown_error')}"
        )
    if result.get("candidate_id") != PROBE_CANDIDATE_ID:
        raise RuntimeError(
            "Unity fixed-tip candidate changed: "
            f"{result.get('candidate_id')!r} != {PROBE_CANDIDATE_ID!r}"
        )
    return result


def _client_info_manifest(info: Any) -> dict[str, Any]:
    return {
        "protocol_version": info.protocol_version,
        "runtime_build_id": info.runtime_build_id,
        "env_state_contract_version": info.env_state_contract_version,
        "dt": info.dt,
        "control_hz": info.control_hz,
        "action_order": list(info.action_order),
        "qpos_order": list(info.qpos_order),
        "qvel_order": list(info.qvel_order),
        "camera_names": list(info.camera_names),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--unity-project", type=Path, default=DEFAULT_UNITY_PROJECT)
    parser.add_argument("--pact-root", type=Path, default=DEFAULT_PACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--train-episodes",
        default="0,18,19,20,21,22,23,27,28,29",
        help="Episodes supplying task-relevant target qpos for training walks.",
    )
    parser.add_argument(
        "--validation-episodes",
        default="30,32,33,34",
        help="Disjoint episodes supplying target qpos for held-out walks.",
    )
    parser.add_argument("--train-trajectories", type=int, default=4)
    parser.add_argument("--validation-trajectories", type=int, default=2)
    parser.add_argument("--targets-per-trajectory", type=int, default=16)
    parser.add_argument("--max-steps-per-target", type=int, default=55)
    parser.add_argument("--capture-stride", type=int, default=3)
    parser.add_argument("--pool-stride", type=int, default=20)
    parser.add_argument("--tracking-gain", type=float, default=4.0)
    parser.add_argument("--max-command", type=float, default=0.85)
    parser.add_argument("--tracking-deadband", type=float, default=0.008)
    parser.add_argument("--target-tolerance", type=float, default=0.025)
    parser.add_argument("--settled-ticks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5057)
    parser.add_argument("--socket-timeout-sec", type=float, default=20.0)
    parser.add_argument("--probe-timeout-sec", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "train_trajectories",
        "validation_trajectories",
        "targets_per_trajectory",
        "max_steps_per_target",
        "capture_stride",
        "pool_stride",
        "settled_ticks",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be >= 1")
    if args.max_command <= 0.0 or args.max_command > 1.0:
        raise ValueError("max_command must be in (0, 1]")
    if args.target_tolerance <= 0.0:
        raise ValueError("target_tolerance must be > 0")


def main() -> int:
    args = parse_args()
    _validate_args(args)
    train_episode_ids = parse_episode_ids(args.train_episodes)
    validation_episode_ids = parse_episode_ids(args.validation_episodes)
    overlap = sorted(set(train_episode_ids) & set(validation_episode_ids))
    if overlap:
        raise ValueError(f"train/validation episode overlap: {overlap}")

    output_dir = args.output_dir.resolve()
    samples_path = output_dir / "fixed_tip_samples.jsonl"
    manifest_path = output_dir / "collection_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; pass --overwrite to replace this run"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in (samples_path, manifest_path):
            path.unlink(missing_ok=True)

    probe_directory = (
        args.unity_project.resolve() / "Temp/FixedBucketTipFkCapture"
    )
    enable_path = probe_directory / "enable"
    if not enable_path.is_file():
        raise RuntimeError(
            f"Unity fixed-tip probe is not enabled: missing {enable_path}"
        )

    train_candidates = load_target_candidates(
        args.dataset.resolve(), train_episode_ids, pool_stride=args.pool_stride
    )
    validation_candidates = load_target_candidates(
        args.dataset.resolve(), validation_episode_ids, pool_stride=args.pool_stride
    )

    pact_root = args.pact_root.resolve()
    if str(pact_root) not in sys.path:
        sys.path.insert(0, str(pact_root))
    from testbed.backends.agx.protocol import AgxSimClient

    started_at = time.time()
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "started_unix_sec": started_at,
        "dataset": str(args.dataset.resolve()),
        "unity_project": str(args.unity_project.resolve()),
        "probe_candidate_id": PROBE_CANDIDATE_ID,
        "sample_schema_version": SAMPLE_SCHEMA_VERSION,
        "train_episode_ids": train_episode_ids,
        "validation_episode_ids": validation_episode_ids,
        "train_candidate_count": len(train_candidates),
        "validation_candidate_count": len(validation_candidates),
        "parameters": vars(args),
        "sample_count": 0,
        "split_counts": {"train": 0, "validation": 0},
    }
    write_json_atomic(manifest_path, manifest)

    sample_counter = 0
    step_id = 10_000_000
    sample_id_base = time.time_ns() // 1_000

    try:
        with samples_path.open("a", encoding="utf-8") as sample_file:
            with AgxSimClient(
                host=args.host,
                port=args.port,
                timeout_s=args.socket_timeout_sec,
            ) as client:
                info = client.get_info()
                if tuple(info.qpos_order) != EXPECTED_QPOS_ORDER:
                    raise RuntimeError(
                        f"Runtime qpos order {info.qpos_order} "
                        f"!= {EXPECTED_QPOS_ORDER}"
                    )
                manifest["runtime"] = _client_info_manifest(info)
                write_json_atomic(manifest_path, manifest)

                split_specs = (
                    ("train", args.train_trajectories, train_candidates),
                    (
                        "validation",
                        args.validation_trajectories,
                        validation_candidates,
                    ),
                )
                for split, trajectory_count, candidates in split_specs:
                    for trajectory_index in range(trajectory_count):
                        trajectory_seed = (
                            args.seed
                            + (0 if split == "train" else 1_000_000)
                            + trajectory_index
                        )
                        targets = select_diverse_targets(
                            candidates,
                            args.targets_per_trajectory,
                            seed=trajectory_seed,
                        )
                        client.reset(
                            seed=trajectory_seed,
                            reset_terrain=(
                                split == "train" and trajectory_index == 0
                            ),
                            reset_pose=True,
                            diagnostic_terrain_mode="terrain_disabled",
                            control_compatibility_profile="production",
                        )
                        neutral = client.step(
                            step_id, np.zeros(4, dtype=np.float32)
                        )
                        step_id += 1
                        current_qpos = neutral.qpos

                        for target_index, target in enumerate(targets):
                            settled = 0
                            last_response = neutral
                            last_action = np.zeros(4, dtype=np.float32)
                            for local_step in range(args.max_steps_per_target):
                                last_action = target_tracking_action(
                                    current_qpos,
                                    target.qpos,
                                    gain=args.tracking_gain,
                                    max_command=args.max_command,
                                    deadband=args.tracking_deadband,
                                )
                                last_response = client.step(step_id, last_action)
                                step_id += 1
                                current_qpos = last_response.qpos
                                max_error = float(
                                    np.max(np.abs(target.qpos - current_qpos))
                                )
                                settled = (
                                    settled + 1
                                    if max_error <= args.target_tolerance
                                    else 0
                                )

                                should_capture = (
                                    local_step % args.capture_stride == 0
                                    or settled >= args.settled_ticks
                                    or local_step
                                    == args.max_steps_per_target - 1
                                )
                                if should_capture:
                                    sample_id = sample_id_base + sample_counter
                                    probe = capture_probe(
                                        probe_directory,
                                        sample_id=sample_id,
                                        split=split,
                                        target=target,
                                        observed_qpos=last_response.qpos,
                                        timeout_sec=args.probe_timeout_sec,
                                    )
                                    record = {
                                        "schema_version": SAMPLE_SCHEMA_VERSION,
                                        "sample_id": sample_id,
                                        "split": split,
                                        "trajectory_index": trajectory_index,
                                        "trajectory_seed": trajectory_seed,
                                        "target_index": target_index,
                                        "target_source_episode": target.episode_id,
                                        "target_source_step": target.step,
                                        "target_qpos": target.qpos,
                                        "local_step": local_step,
                                        "commanded_action": last_action,
                                        "observed_qpos": last_response.qpos,
                                        "observed_qvel": last_response.qvel,
                                        "target_max_abs_error": max_error,
                                        "runtime_warnings": list(
                                            last_response.warnings
                                        ),
                                        "probe": probe,
                                    }
                                    sample_file.write(
                                        json.dumps(
                                            jsonable(record),
                                            separators=(",", ":"),
                                        )
                                        + "\n"
                                    )
                                    sample_file.flush()
                                    sample_counter += 1
                                    manifest["sample_count"] = sample_counter
                                    manifest["split_counts"][split] += 1
                                    if sample_counter % 100 == 0:
                                        write_json_atomic(manifest_path, manifest)
                                        print(
                                            f"samples={sample_counter} "
                                            f"split={split} "
                                            f"trajectory={trajectory_index + 1}/"
                                            f"{trajectory_count} "
                                            f"target={target_index + 1}/"
                                            f"{len(targets)}",
                                            flush=True,
                                        )

                                if settled >= args.settled_ticks:
                                    break

        manifest["status"] = "completed"
        manifest["completed_unix_sec"] = time.time()
        manifest["elapsed_sec"] = time.time() - started_at
        write_json_atomic(manifest_path, manifest)
        print(
            f"completed samples={sample_counter} "
            f"elapsed_sec={manifest['elapsed_sec']:.1f} "
            f"output={output_dir}",
            flush=True,
        )
        return 0
    except Exception as exception:
        manifest["status"] = "failed"
        manifest["completed_unix_sec"] = time.time()
        manifest["elapsed_sec"] = time.time() - started_at
        manifest["error"] = repr(exception)
        write_json_atomic(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
