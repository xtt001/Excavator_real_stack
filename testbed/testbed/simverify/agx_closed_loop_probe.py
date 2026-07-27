"""Bounded, diagnostic-only AGX closed-loop probe for SimVerify checkpoints.

This module keeps the policy owner in Real Stack and treats the external
PACT/Unity checkout as a read-only environment provider.  It is not a
production backend and must not be imported by real-machine runtime code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import struct
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import numpy as np

from testbed.simverify.artifacts import (
    artifact_identity,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import (
    FROZEN_SOURCE_DT_S,
    FROZEN_TARGET_HZ,
    POLICY_CAMERA_ORDER,
    SOURCE_ACTION_ORDER,
    SOURCE_QPOS_ORDER,
    SOURCE_QVEL_ORDER,
    SOURCE_TO_POLICY_CAMERA,
    sha256_file,
)

PROBE_SCHEMA = "simverify_agx_closed_loop_diagnostic_v1"
TIMING_SCHEMA = "simverify_agx_ack_step_timing_v1"
ALLOWED_BASELINES = frozenset({"B0", "B1.4", "B2.4"})
SECTORS = ("left", "center", "right")
EXPECTED_SOURCE_CAMERAS = (
    "stick_up",
    "stick_down",
    "eye_left",
    "eye_right",
)
FRAME_HEADER = struct.Struct("!Q")
MAX_FRAME_BYTES = 16 * 1024 * 1024


class ProbeEnvironment(Protocol):
    """Small environment surface required by the diagnostic probe."""

    def get_info(self) -> Mapping[str, Any]: ...

    def reset(self, *, seed: int) -> Mapping[str, Any]: ...

    def step(self, action: np.ndarray) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class ExternalAgxWorker:
    """Run the existing PACT AGX backend in an isolated subprocess."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        pact_root: str | Path,
        worker_script: str | Path,
        python_executable: str | Path = sys.executable,
        host: str = "127.0.0.1",
        port: int = 5057,
        timeout_s: float = 10.0,
    ) -> None:
        self._repo_root = Path(repo_root).resolve(strict=True)
        self._pact_root = Path(pact_root).resolve(strict=True)
        self._worker_script = Path(worker_script).resolve(strict=True)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(self._pact_root)
        command = [
            str(Path(python_executable).resolve(strict=True)),
            str(self._worker_script),
            "--host",
            str(host),
            "--port",
            str(int(port)),
            "--timeout",
            str(float(timeout_s)),
        ]
        self._process = subprocess.Popen(
            command,
            cwd=self._pact_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._process.kill()
            raise RuntimeError("failed to create AGX worker pipes")

    def get_info(self) -> Mapping[str, Any]:
        return self._request({"op": "get_info"})

    def reset(self, *, seed: int) -> Mapping[str, Any]:
        return self._request({"op": "reset", "seed": int(seed)})

    def step(self, action: np.ndarray) -> Mapping[str, Any]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != (4,) or not np.all(np.isfinite(action_array)):
            raise ValueError("AGX probe action must be finite shape (4,)")
        return self._request(
            {"op": "step", "action": action_array.astype(float).tolist()}
        )

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                self._request({"op": "close"})
            except Exception:
                process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        self._process = None

    def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("AGX worker is closed")
        if process.poll() is not None:
            raise RuntimeError(self._worker_failure("AGX worker exited"))
        _write_frame(process.stdin, dict(payload))
        response = _read_frame(process.stdout)
        if not isinstance(response, Mapping):
            raise RuntimeError("AGX worker returned a non-mapping response")
        if not bool(response.get("ok")):
            raise RuntimeError(
                self._worker_failure(
                    str(response.get("error", "AGX worker request failed"))
                )
            )
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise RuntimeError("AGX worker result is not a mapping")
        return result

    def _worker_failure(self, message: str) -> str:
        process = self._process
        stderr_text = ""
        if process is not None and process.poll() is not None and process.stderr:
            stderr_text = process.stderr.read().decode("utf-8", errors="replace")
        return message if not stderr_text else f"{message}: {stderr_text.strip()}"


def build_cycle_condition(current_sector: str, next_sector: str) -> np.ndarray:
    """Return current+next one-hot condition in the frozen six-value order."""

    if current_sector not in SECTORS or next_sector not in SECTORS:
        raise ValueError(f"sectors must be one of {SECTORS}")
    condition = np.zeros(6, dtype=np.float32)
    condition[SECTORS.index(current_sector)] = 1.0
    condition[3 + SECTORS.index(next_sector)] = 1.0
    return condition


def policy_source_step(policy_tick: int, *, sim_dt: float) -> int:
    """Map a 20 Hz policy tick to the first non-early 50 Hz source step."""

    if policy_tick < 0:
        raise ValueError("policy_tick must be non-negative")
    if not math.isfinite(sim_dt) or sim_dt <= 0.0:
        raise ValueError("sim_dt must be positive and finite")
    policy_dt = 1.0 / FROZEN_TARGET_HZ
    return int(math.ceil((policy_tick * policy_dt / sim_dt) - 1.0e-12))


def validate_agx_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the observable AGX contract without accepting privilege input."""

    protocol = str(info.get("protocol_version", ""))
    dt = float(info.get("dt", float("nan")))
    if protocol != "agx-sim/v2":
        raise ValueError(f"AGX probe requires agx-sim/v2, got {protocol!r}")
    if not math.isclose(dt, FROZEN_SOURCE_DT_S, rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError(
            f"AGX dt {dt!r} does not match frozen source dt {FROZEN_SOURCE_DT_S}"
        )
    checks = (
        ("action_order", SOURCE_ACTION_ORDER),
        ("qpos_order", SOURCE_QPOS_ORDER),
        ("qvel_order", SOURCE_QVEL_ORDER),
        ("camera_names", EXPECTED_SOURCE_CAMERAS),
    )
    for key, expected in checks:
        actual = tuple(map(str, info.get(key, ())))
        if actual != tuple(expected):
            raise ValueError(f"AGX {key} mismatch: {actual} != {tuple(expected)}")
    if not bool(info.get("supports_images")):
        raise ValueError("AGX probe requires four observable cameras")
    return {
        "schema": "simverify_agx_observable_contract_v1",
        "protocol_version": protocol,
        "runtime_build_id": str(info.get("runtime_build_id", "")),
        "reported_dt": dt,
        "reported_control_hz": float(info.get("control_hz", 1.0 / dt)),
        "dt": FROZEN_SOURCE_DT_S,
        "control_hz": 1.0 / FROZEN_SOURCE_DT_S,
        "action_order": list(SOURCE_ACTION_ORDER),
        "qpos_order": list(SOURCE_QPOS_ORDER),
        "qvel_order": list(SOURCE_QVEL_ORDER),
        "camera_names": list(EXPECTED_SOURCE_CAMERAS),
        "env_state_policy_input": False,
        "time_basis": "applied_step_index_times_get_info_dt",
        "sim_time_ns_semantics": "unity_frame_clock_diagnostic_only",
    }


def validate_probe_bundle(bundle_root: str | Path) -> dict[str, Any]:
    """Reject unfinished or real-control-capable checkpoints."""

    bundle = Path(bundle_root).resolve(strict=True)
    metadata_path = bundle / "run_metadata.json"
    resolved_path = bundle / "resolved_config.yaml"
    stats_path = bundle / "dataset_stats.pkl"
    checkpoint_path = bundle / "policy_best.ckpt"
    for path in (metadata_path, resolved_path, stats_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"probe bundle artifact missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError("probe requires completed training metadata")
    experiment = metadata.get("experiment_contract", {})
    semantics = metadata.get("checkpoint_semantics", {})
    baseline_id = str(experiment.get("baseline_id", ""))
    expected_condition = {
        "B0": "absent",
        "B1.4": "cycle_condition_v1_next_sector_only",
        "B2.4": "cycle_condition_v1_next_sector_only",
    }
    if baseline_id not in ALLOWED_BASELINES:
        raise ValueError(
            f"probe baseline must be one of {sorted(ALLOWED_BASELINES)}"
        )
    if experiment.get("condition_input") != expected_condition[baseline_id]:
        raise ValueError("probe bundle condition contract mismatch")
    if (
        semantics.get("domain") != "sim"
        or semantics.get("source_action_domain") != "actuator_speed_cmd"
        or semantics.get("real_control_allowed") is not False
        or semantics.get("jetson_allowed") is not False
    ):
        raise ValueError("probe bundle lacks sim-only deployment prohibition")
    return {
        "bundle_root": str(bundle),
        "baseline_id": baseline_id,
        "condition_input": str(experiment["condition_input"]),
        "checkpoint_semantics": dict(semantics),
        "artifacts": {
            path.name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in (metadata_path, resolved_path, stats_path, checkpoint_path)
        },
    }


def external_git_provenance(path: str | Path) -> dict[str, Any]:
    """Capture external repository state without modifying it."""

    root = Path(path).resolve(strict=True)
    head = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--short")
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "path": str(root),
        "git_sha": head,
        "branch": branch,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def run_bounded_closed_loop_probe(
    *,
    policy: Any,
    environment: ProbeEnvironment,
    output_root: str | Path,
    bundle_contract: Mapping[str, Any],
    current_git: Mapping[str, Any],
    external_provenance: Mapping[str, Any],
    current_sector: str,
    next_sector: str,
    seed: int,
    policy_ticks: int,
    save_images: bool = True,
) -> dict[str, Any]:
    """Run bounded feedback execution and write immutable diagnostic evidence."""

    if policy_ticks <= 0:
        raise ValueError("policy_ticks must be positive")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable probe output exists: {destination}")
    condition = build_cycle_condition(current_sector, next_sector)
    baseline_id = str(bundle_contract["baseline_id"])
    pass_condition = baseline_id != "B0"
    info = validate_agx_info(environment.get_info())
    sim_dt = float(info["dt"])
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"probe temporary path exists: {temporary}")
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    try:
        observation = _validate_environment_observation(
            environment.reset(seed=int(seed))
        )
        if hasattr(policy, "reset"):
            policy.reset()
        held_action = np.zeros(4, dtype=np.float32)
        _append_step_row(
            step_rows,
            observation,
            sim_dt=sim_dt,
            sent_action=None,
            policy_tick=None,
            transition="reset_zero_step",
        )
        initial_qpos = np.asarray(observation["qpos"], dtype=np.float32)
        for policy_tick in range(policy_ticks):
            target_step = policy_source_step(policy_tick, sim_dt=sim_dt)
            while int(observation["step_id"]) < target_step:
                observation = _validate_environment_observation(
                    environment.step(held_action)
                )
                _append_step_row(
                    step_rows,
                    observation,
                    sim_dt=sim_dt,
                    sent_action=held_action,
                    policy_tick=policy_tick - 1 if policy_tick > 0 else None,
                    transition="action_hold",
                )
            if int(observation["step_id"]) != target_step:
                raise RuntimeError("AGX step sequence skipped a policy target step")
            policy_observation, encoded_frames = _policy_observation(
                observation,
                condition=condition if pass_condition else None,
                authoritative_time_ns=int(round(target_step * sim_dt * 1.0e9)),
            )
            aggregated = np.asarray(
                policy.predict(policy_observation),
                dtype=np.float32,
            ).reshape(-1)
            if aggregated.shape != (4,) or not np.all(np.isfinite(aggregated)):
                raise ValueError("policy produced a non-finite or non-4D action")
            runtime_safe = np.clip(aggregated, -1.0, 1.0).astype(np.float32)
            raw_normalized = np.asarray(
                policy.last_raw_action_chunk(),
                dtype=np.float32,
            )
            raw_direct = np.asarray(
                policy.last_raw_action_chunk_direct(),
                dtype=np.float32,
            )
            route = getattr(policy, "condition_route_diagnostics", None)
            policy_rows.append(
                {
                    "schema": "simverify_agx_policy_tick_v1",
                    "policy_tick": int(policy_tick),
                    "source_step_id": int(target_step),
                    "source_time_s": float(target_step * sim_dt),
                    "unity_sim_time_ns_diagnostic": int(
                        observation["sim_time_ns"]
                    ),
                    "condition": condition.astype(float).tolist(),
                    "condition_delivered": bool(pass_condition),
                    "raw_policy_chunk_normalized": raw_normalized.astype(
                        float
                    ).tolist(),
                    "raw_policy_chunk_direct": raw_direct.astype(float).tolist(),
                    "temporal_aggregation_action": aggregated.astype(
                        float
                    ).tolist(),
                    "future_runtime_safe_action": runtime_safe.astype(
                        float
                    ).tolist(),
                    "actual_sent_action": runtime_safe.astype(float).tolist(),
                    "condition_route": None if route is None else dict(route),
                    "qpos": np.asarray(
                        observation["qpos"], dtype=np.float32
                    ).astype(float).tolist(),
                    "qvel": np.asarray(
                        observation["qvel"], dtype=np.float32
                    ).astype(float).tolist(),
                }
            )
            if save_images:
                for source_name, frame in encoded_frames.items():
                    target = (
                        temporary
                        / "policy_tick_images"
                        / f"tick_{policy_tick:06d}_{source_name}.jpg"
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(bytes(frame["data"]))
                    identities.append(artifact_identity(target))
            held_action = runtime_safe
            next_target = policy_source_step(policy_tick + 1, sim_dt=sim_dt)
            while int(observation["step_id"]) < next_target:
                observation = _validate_environment_observation(
                    environment.step(held_action)
                )
                _append_step_row(
                    step_rows,
                    observation,
                    sim_dt=sim_dt,
                    sent_action=held_action,
                    policy_tick=policy_tick,
                    transition="action_hold",
                )
        final_qpos = np.asarray(observation["qpos"], dtype=np.float32)
        identities.append(write_jsonl(temporary / "policy_ticks.jsonl", policy_rows))
        identities.append(write_jsonl(temporary / "steps.jsonl", step_rows))
        summary = {
            "schema": PROBE_SCHEMA,
            "status": "completed_bounded_diagnostic",
            "evidence_scope": "sim_closed_loop_diagnostic_non_promotable",
            "closed_loop_execution": True,
            "task_success_claimed": False,
            "real_control_candidate": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "test_intent": {
                "question": (
                    "Can the frozen Real Stack policy path consume live observable "
                    "AGX feedback and causally change subsequent simulated state?"
                ),
                "observable_policy_inputs": [
                    "four_rgb_cameras",
                    "qpos",
                    "qvel",
                    *(
                        ["cycle_condition_v1"]
                        if pass_condition
                        else []
                    ),
                ],
                "intervention": {
                    "seed": int(seed),
                    "current_sector": current_sector,
                    "next_sector": next_sector,
                },
                "can_prove": [
                    "policy_environment_feedback_path_executes",
                    "actions_change_later_sim_observations",
                    "runtime_equivalent_policy_path_runs_at_20hz",
                ],
                "cannot_prove": [
                    "one_cycle_task_success",
                    "two_cycle_task_success",
                    "real_machine_transfer",
                    "real_control_readiness",
                ],
                "termination": f"bounded_after_{policy_ticks}_policy_ticks",
            },
            "timing_contract": {
                "schema": TIMING_SCHEMA,
                "source_dt_s": sim_dt,
                "policy_hz": FROZEN_TARGET_HZ,
                "authoritative_time": "applied_step_index_times_get_info_dt",
                "unity_sim_time_ns_used_for_scheduling": False,
                "unity_sim_time_ns_status": "diagnostic_only_unity_frame_clock",
                "policy_source_steps": [
                    policy_source_step(index, sim_dt=sim_dt)
                    for index in range(policy_ticks + 1)
                ],
            },
            "condition_contract": {
                "current_sector": current_sector,
                "next_sector": next_sector,
                "vector": condition.astype(float).tolist(),
                "delivered_to_policy": bool(pass_condition),
            },
            "agx_contract": info,
            "bundle_contract": dict(bundle_contract),
            "provenance": {
                "current_repo": dict(current_git),
                "external_read_only": dict(external_provenance),
            },
            "step_count": int(len(step_rows)),
            "policy_tick_count": int(len(policy_rows)),
            "initial_qpos": initial_qpos.astype(float).tolist(),
            "final_qpos": final_qpos.astype(float).tolist(),
            "qpos_delta": (final_qpos - initial_qpos).astype(float).tolist(),
            "max_abs_sent_action": float(
                np.max(
                    np.abs(
                        np.asarray(
                            [
                                row["actual_sent_action"]
                                for row in policy_rows
                            ],
                            dtype=np.float32,
                        )
                    )
                )
            ),
            "privilege_policy_input_scan": {
                "env_state": False,
                "bucket_mass": False,
                "terrain_grid": False,
                "exact_bucket_tip": False,
                "planner_private_state": False,
            },
        }
        identities.append(write_json(temporary / "run_manifest.json", summary))
        checksum_identity = write_checksums(
            temporary,
            identities,
            path=temporary / "checksums.sha256",
        )
        summary["artifacts"] = {
            "run_manifest": next(
                identity
                for identity in identities
                if Path(str(identity["path"])).name == "run_manifest.json"
            ),
            "checksums": checksum_identity,
        }
        temporary.rename(destination)
        summary["artifacts"] = {
            "run_manifest": {
                **summary["artifacts"]["run_manifest"],
                "path": str(destination / "run_manifest.json"),
            },
            "checksums": {
                **summary["artifacts"]["checksums"],
                "path": str(destination / "checksums.sha256"),
            },
        }
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _policy_observation(
    observation: Mapping[str, Any],
    *,
    condition: np.ndarray | None,
    authoritative_time_ns: int,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    encoded = observation["encoded_images"]
    policy_observation: dict[str, Any] = {
        "qpos": np.asarray(observation["qpos"], dtype=np.float32),
        "qvel": np.asarray(observation["qvel"], dtype=np.float32),
        "image_timestamp_ns": {
            camera: int(authoritative_time_ns) for camera in POLICY_CAMERA_ORDER
        },
    }
    for source_name, policy_name in SOURCE_TO_POLICY_CAMERA.items():
        policy_observation[f"image_{policy_name}"] = _decode_and_transform_jpeg(
            encoded[source_name]
        )
    if condition is not None:
        policy_observation["cycle_condition_v1"] = np.asarray(
            condition, dtype=np.float32
        ).copy()
    return policy_observation, encoded


def _decode_and_transform_jpeg(frame: Mapping[str, Any]) -> np.ndarray:
    if str(frame.get("encoding")) != "jpeg":
        raise ValueError("AGX probe requires JPEG camera payloads")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for AGX probe") from exc
    encoded = np.frombuffer(bytes(frame["data"]), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("AGX probe received undecodable JPEG")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    transformed = cv2.resize(rgb, (384, 216), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(transformed, dtype=np.uint8)


def _validate_environment_observation(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    qpos = np.asarray(observation.get("qpos"), dtype=np.float32).reshape(-1)
    qvel = np.asarray(observation.get("qvel"), dtype=np.float32).reshape(-1)
    if qpos.shape != (4,) or qvel.shape != (4,):
        raise ValueError("AGX observation requires qpos/qvel shape (4,)")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
        raise ValueError("AGX observation qpos/qvel must be finite")
    encoded = observation.get("encoded_images")
    if not isinstance(encoded, Mapping) or tuple(sorted(encoded)) != tuple(
        sorted(EXPECTED_SOURCE_CAMERAS)
    ):
        raise ValueError("AGX observation camera set mismatch")
    return {
        "qpos": qpos,
        "qvel": qvel,
        "encoded_images": dict(encoded),
        "step_id": int(observation["step_id"]),
        "sim_time_ns": int(observation["sim_time_ns"]),
        "warnings": list(observation.get("warnings", [])),
    }


def _append_step_row(
    rows: list[dict[str, Any]],
    observation: Mapping[str, Any],
    *,
    sim_dt: float,
    sent_action: np.ndarray | None,
    policy_tick: int | None,
    transition: str,
) -> None:
    step_id = int(observation["step_id"])
    rows.append(
        {
            "schema": "simverify_agx_environment_step_v1",
            "step_id": step_id,
            "source_time_s": float(step_id * sim_dt),
            "unity_sim_time_ns_diagnostic": int(observation["sim_time_ns"]),
            "policy_tick_owner": None if policy_tick is None else int(policy_tick),
            "transition": transition,
            "sent_action": (
                None
                if sent_action is None
                else np.asarray(sent_action, dtype=np.float32).astype(float).tolist()
            ),
            "qpos": np.asarray(
                observation["qpos"], dtype=np.float32
            ).astype(float).tolist(),
            "qvel": np.asarray(
                observation["qvel"], dtype=np.float32
            ).astype(float).tolist(),
            "warnings": list(observation.get("warnings", [])),
        }
    )


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _write_frame(stream: BinaryIO, payload: Any) -> None:
    encoded = pickle.dumps(payload, protocol=5)
    if len(encoded) > MAX_FRAME_BYTES:
        raise ValueError("worker frame exceeds bounded size")
    stream.write(FRAME_HEADER.pack(len(encoded)))
    stream.write(encoded)
    stream.flush()


def _read_frame(stream: BinaryIO) -> Any:
    header = _read_exact(stream, FRAME_HEADER.size)
    (length,) = FRAME_HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ValueError("worker frame exceeds bounded size")
    return pickle.loads(_read_exact(stream, int(length)))


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("worker pipe closed during framed message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
