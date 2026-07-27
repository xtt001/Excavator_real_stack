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
ALLOWED_BASELINES = frozenset({"B0", "B1", "B2", "B1.4", "B2.4"})
ACTION_SELECTION_MODES = frozenset(
    {
        "legacy_temporal_aggregation",
        "recency_temporal_aggregation_diagnostic",
        "newest_chunk_head_diagnostic",
    }
)
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


class ReadyBoundaryDetector(Protocol):
    """Observable-only boundary detector surface used by the probe."""

    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def reset(self) -> None: ...

    def observe(
        self,
        *,
        policy_tick: int,
        observation: Mapping[str, Any],
        policy_observation: Mapping[str, Any],
        held_action: np.ndarray,
        condition_route: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]: ...


class ConditionCommitDetector(Protocol):
    """Observable-only dump-end detector used by gated-condition policies."""

    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def reset(self) -> None: ...

    def observe(
        self,
        *,
        policy_tick: int,
        observation: Mapping[str, Any],
        held_action: np.ndarray,
    ) -> Mapping[str, Any]: ...


class ObservableDumpEndCommitDetector:
    """Commit a target at the first causal 20 Hz row after dump release.

    The detector is a runtime-safe adaptation of the frozen numeric annotation:
    a release is active only in the learned dump swing cluster while the
    previously sent bucket action is negative.  A two-policy-tick dwell rejects
    one-tick action spikes, and the first following inactive tick is the
    condition commit row.  It never reads future observations or privilege.
    """

    SCHEMA = "simverify_agx_observable_dump_end_commit_detector_v1"
    TICK_SCHEMA = "simverify_agx_observable_dump_end_commit_tick_v1"

    def __init__(
        self,
        *,
        dump_swing_threshold: float,
        action_deadzone: float,
        artifact_provenance: Mapping[str, Any],
        minimum_release_policy_ticks: int = 2,
    ) -> None:
        if not math.isfinite(float(dump_swing_threshold)):
            raise ValueError("dump_swing_threshold must be finite")
        if not math.isfinite(float(action_deadzone)) or float(action_deadzone) <= 0:
            raise ValueError("action_deadzone must be finite and positive")
        if int(minimum_release_policy_ticks) < 2:
            raise ValueError("minimum_release_policy_ticks must be at least two")
        self._dump_swing_threshold = float(dump_swing_threshold)
        self._action_deadzone = float(action_deadzone)
        self._minimum_release_policy_ticks = int(minimum_release_policy_ticks)
        self._provenance = {
            "schema": self.SCHEMA,
            "mode": "causal_live_adaptation_of_frozen_dump_release_end",
            "observable_inputs": [
                "swing_qpos",
                "previously_sent_bucket_action",
            ],
            "privilege_used": False,
            "future_observations_used": False,
            "commit_timing": "first_policy_row_after_sustained_release_becomes_inactive",
            "minimum_release_policy_ticks": self._minimum_release_policy_ticks,
            "thresholds": {
                "dump_swing_threshold": self._dump_swing_threshold,
                "action_deadzone": self._action_deadzone,
            },
            "artifacts": dict(artifact_provenance),
        }
        self.reset()

    @classmethod
    def from_definition_artifacts(
        cls,
        *,
        definition_root: str | Path,
    ) -> ObservableDumpEndCommitDetector:
        root = Path(definition_root).resolve(strict=True)
        audit_path = root / "dig_ready_boundary_audit_v1.json"
        if not audit_path.is_file():
            raise FileNotFoundError(
                "gated-condition probe requires dig_ready_boundary_audit_v1.json"
            )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        numeric = audit["numeric_thresholds"]
        return cls(
            dump_swing_threshold=float(
                numeric["dump_release"]["swing_threshold"]
            ),
            action_deadzone=float(numeric["action_deadzone"]),
            artifact_provenance={
                "definition_root": str(root),
                "dig_ready_boundary_audit_v1": {
                    "path": str(audit_path),
                    "sha256": sha256_file(audit_path),
                },
                "source_schema": str(numeric["schema"]),
                "source_fit_scope": str(numeric["fit_scope"]),
            },
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._provenance))

    def reset(self) -> None:
        self._release_ticks = 0
        self._armed = False
        self._committed = False

    def observe(
        self,
        *,
        policy_tick: int,
        observation: Mapping[str, Any],
        held_action: np.ndarray,
    ) -> dict[str, Any]:
        swing_qpos = float(np.asarray(observation["qpos"])[0])
        bucket_action = float(np.asarray(held_action, dtype=np.float32)[3])
        release_active = bool(
            swing_qpos > self._dump_swing_threshold
            and bucket_action < -self._action_deadzone
        )
        confirmed = False
        if not self._committed:
            if release_active:
                self._release_ticks += 1
                self._armed = (
                    self._release_ticks >= self._minimum_release_policy_ticks
                )
            elif self._armed:
                self._committed = True
                confirmed = True
        return {
            "schema": self.TICK_SCHEMA,
            "policy_tick": int(policy_tick),
            "state": (
                "committed"
                if self._committed
                else "armed"
                if self._armed
                else "searching_dump_release"
            ),
            "confirmed": confirmed,
            "committed": self._committed,
            "swing_qpos": swing_qpos,
            "previously_sent_bucket_action": bucket_action,
            "release_active": release_active,
            "release_ticks": int(self._release_ticks),
        }


class ObservableReadyBoundaryDetector:
    """Causal live adaptation of the frozen M0 observable ready contract.

    The offline M0 labeler may inspect a bounded interval around a numeric
    candidate.  A live probe cannot inspect future rows, so it first arms on a
    sustained observable dump-release command and then accepts the first
    return-to-work-area frame that passes the frozen ready prototypes and
    trailing stick-pair stability threshold.
    """

    SCHEMA = "simverify_agx_observable_ready_boundary_detector_v1"
    TICK_SCHEMA = "simverify_agx_observable_ready_boundary_tick_v1"

    def __init__(
        self,
        *,
        feature_extractor: Any,
        eye_ready_prototype: np.ndarray,
        stick_ready_prototype: np.ndarray,
        dump_swing_threshold: float,
        action_deadzone: float,
        eye_support_threshold: float,
        stick_support_threshold: float,
        stick_change_threshold: float,
        artifact_provenance: Mapping[str, Any],
        minimum_release_policy_ticks: int = 2,
    ) -> None:
        self._feature_extractor = feature_extractor
        self._eye_ready_prototype = _normalized_vector(
            eye_ready_prototype,
            name="eye_ready_prototype",
        )
        self._stick_ready_prototype = _normalized_vector(
            stick_ready_prototype,
            name="stick_ready_prototype",
        )
        if self._eye_ready_prototype.shape != (1024,):
            raise ValueError("eye ready prototype must have shape (1024,)")
        if self._stick_ready_prototype.shape != (1024,):
            raise ValueError("stick ready prototype must have shape (1024,)")
        numeric_values = (
            dump_swing_threshold,
            action_deadzone,
            eye_support_threshold,
            stick_support_threshold,
            stick_change_threshold,
        )
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("ready-boundary thresholds must be finite")
        if int(minimum_release_policy_ticks) < 2:
            raise ValueError("minimum_release_policy_ticks must be at least two")
        self._dump_swing_threshold = float(dump_swing_threshold)
        self._action_deadzone = float(action_deadzone)
        self._eye_support_threshold = float(eye_support_threshold)
        self._stick_support_threshold = float(stick_support_threshold)
        self._stick_change_threshold = float(stick_change_threshold)
        self._minimum_release_policy_ticks = int(minimum_release_policy_ticks)
        self._provenance = {
            "schema": self.SCHEMA,
            "mode": "causal_live_adaptation_of_frozen_m0_ready_contract",
            "observable_inputs": [
                "four_rgb_cameras",
                "swing_qpos",
                "previously_sent_bucket_action",
                "condition_route",
            ],
            "privilege_used": False,
            "future_observations_used": False,
            "confirmation_dwell_policy_ticks": 1,
            "minimum_release_policy_ticks": self._minimum_release_policy_ticks,
            "thresholds": {
                "dump_swing_threshold": self._dump_swing_threshold,
                "action_deadzone": self._action_deadzone,
                "eye_ready_support": self._eye_support_threshold,
                "stick_ready_support": self._stick_support_threshold,
                "trailing_stick_cosine_change": self._stick_change_threshold,
            },
            "artifacts": dict(artifact_provenance),
            "feature_extractor": dict(feature_extractor.provenance),
        }
        self.reset()

    @classmethod
    def from_m0_artifacts(
        cls,
        *,
        m0_root: str | Path,
        resnet18_checkpoint: str | Path,
        device: str = "cpu",
        expected_resnet18_sha256: str = (
            "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
        ),
    ) -> ObservableReadyBoundaryDetector:
        """Build the detector strictly from frozen local M0 artifacts."""

        from testbed.simverify.features import FrozenResNet18FeatureExtractor

        root = Path(m0_root).resolve(strict=True)
        thresholds_path = root / "annotation_thresholds_v2.json"
        prototypes_path = root / "annotation_feature_prototypes_v2.npz"
        if not thresholds_path.is_file() or not prototypes_path.is_file():
            raise FileNotFoundError(
                "M0 ready detector requires annotation_thresholds_v2.json "
                "and annotation_feature_prototypes_v2.npz"
            )
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
        numeric = thresholds["numeric"]
        visual = thresholds["visual_events"]
        with np.load(prototypes_path, allow_pickle=False) as prototypes:
            eye = np.asarray(prototypes["event_eye_ready"], dtype=np.float32)
            stick = np.asarray(prototypes["event_stick_ready"], dtype=np.float32)
        extractor = FrozenResNet18FeatureExtractor(
            resnet18_checkpoint,
            expected_checkpoint_sha256=expected_resnet18_sha256,
            device=device,
            batch_size=4,
        )
        return cls(
            feature_extractor=extractor,
            eye_ready_prototype=eye,
            stick_ready_prototype=stick,
            dump_swing_threshold=float(numeric["dump_release"]["swing_threshold"]),
            action_deadzone=float(numeric["action_deadzone"]),
            eye_support_threshold=float(visual["support_thresholds"]["ready"]["eye"]),
            stick_support_threshold=float(
                visual["support_thresholds"]["ready"]["stick"]
            ),
            stick_change_threshold=float(visual["change_thresholds"]["ready"]["stick"]),
            artifact_provenance={
                "m0_root": str(root),
                "annotation_thresholds_v2": {
                    "path": str(thresholds_path),
                    "sha256": sha256_file(thresholds_path),
                },
                "annotation_feature_prototypes_v2": {
                    "path": str(prototypes_path),
                    "sha256": sha256_file(prototypes_path),
                },
            },
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._provenance))

    def reset(self) -> None:
        self._armed = False
        self._confirmed = False
        self._release_ticks = 0
        self._previous_stick_feature: np.ndarray | None = None

    def observe(
        self,
        *,
        policy_tick: int,
        observation: Mapping[str, Any],
        policy_observation: Mapping[str, Any],
        held_action: np.ndarray,
        condition_route: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        route = (
            None if condition_route is None else str(condition_route.get("route", ""))
        )
        swing_qpos = float(np.asarray(observation["qpos"])[0])
        bucket_action = float(np.asarray(held_action, dtype=np.float32)[3])
        release_active = bool(
            route == "next"
            and swing_qpos > self._dump_swing_threshold
            and bucket_action < -self._action_deadzone
        )
        if not self._armed and not self._confirmed:
            self._release_ticks = self._release_ticks + 1 if release_active else 0
            self._armed = self._release_ticks >= self._minimum_release_policy_ticks

        diagnostic: dict[str, Any] = {
            "schema": self.TICK_SCHEMA,
            "policy_tick": int(policy_tick),
            "state": (
                "confirmed"
                if self._confirmed
                else "armed"
                if self._armed
                else "searching_dump_release"
            ),
            "confirmed": False,
            "condition_route_before_predict": route,
            "swing_qpos": swing_qpos,
            "previously_sent_bucket_action": bucket_action,
            "release_active": release_active,
            "release_ticks": int(self._release_ticks),
            "candidate": False,
            "eye_ready_support": None,
            "stick_ready_support": None,
            "trailing_stick_cosine_change": None,
        }
        if not self._armed or self._confirmed:
            return diagnostic

        images = [
            np.asarray(policy_observation[f"image_{camera}"], dtype=np.uint8)
            for camera in ("video4", "video5", "video6", "video7")
        ]
        features = np.asarray(
            self._feature_extractor.extract_rgb_batch(images),
            dtype=np.float32,
        )
        if features.shape != (4, 512):
            raise ValueError(
                "ready-boundary feature extractor must return shape (4,512)"
            )
        eye_feature = _normalized_vector(
            np.concatenate((features[0], features[1])),
            name="live_eye_pair_feature",
        )
        stick_feature = _normalized_vector(
            np.concatenate((features[2], features[3])),
            name="live_stick_pair_feature",
        )
        eye_support = float(np.dot(eye_feature, self._eye_ready_prototype))
        stick_support = float(np.dot(stick_feature, self._stick_ready_prototype))
        stick_change = (
            None
            if self._previous_stick_feature is None
            else float(1.0 - np.dot(stick_feature, self._previous_stick_feature))
        )
        self._previous_stick_feature = stick_feature
        candidate = bool(
            route == "next"
            and swing_qpos < self._dump_swing_threshold
            and bucket_action <= self._action_deadzone
        )
        confirmed = bool(
            candidate
            and eye_support >= self._eye_support_threshold
            and stick_support >= self._stick_support_threshold
            and stick_change is not None
            and stick_change <= self._stick_change_threshold
        )
        if confirmed:
            self._confirmed = True
        diagnostic.update(
            {
                "state": "confirmed" if confirmed else "armed",
                "confirmed": confirmed,
                "candidate": candidate,
                "eye_ready_support": eye_support,
                "stick_ready_support": stick_support,
                "trailing_stick_cosine_change": stick_change,
            }
        )
        return diagnostic


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
        "B1": "cycle_condition_v1_dump_end_gated_low_dim",
        "B2": "cycle_condition_v1_dump_end_gated_low_dim",
        "B1.4": "cycle_condition_v1_next_sector_only",
        "B2.4": "cycle_condition_v1_next_sector_only",
    }
    if baseline_id not in ALLOWED_BASELINES:
        raise ValueError(f"probe baseline must be one of {sorted(ALLOWED_BASELINES)}")
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


def load_action_prefix(
    policy_ticks_path: str | Path,
    *,
    count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an immutable actual-sent-action prefix from a prior probe."""

    path = Path(policy_ticks_path).resolve(strict=True)
    if int(count) <= 0:
        raise ValueError("action prefix count must be positive")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < int(count):
        raise ValueError(
            f"action prefix requests {count} rows but source has {len(rows)}"
        )
    selected = rows[: int(count)]
    if [int(row["policy_tick"]) for row in selected] != list(range(int(count))):
        raise ValueError("action prefix policy ticks must be contiguous from zero")
    actions = np.asarray(
        [row["actual_sent_action"] for row in selected],
        dtype=np.float32,
    )
    if actions.shape != (int(count), 4) or not np.all(np.isfinite(actions)):
        raise ValueError("action prefix must contain finite shape (count,4)")
    if np.max(np.abs(actions)) > 1.0 + 1.0e-6:
        raise ValueError("action prefix exceeds runtime-safe bounds")
    return actions, {
        "schema": "simverify_agx_shared_action_prefix_v1",
        "source_policy_ticks": {
            "path": str(path),
            "sha256": sha256_file(path),
        },
        "count": int(count),
        "action_order": list(SOURCE_ACTION_ORDER),
        "source_field": "actual_sent_action",
        "purpose": "same_state_paired_condition_branch_at_dump_end",
        "promotable": False,
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
    second_next_sector: str | None = None,
    ready_boundary_detector: ReadyBoundaryDetector | None = None,
    condition_commit_detector: ConditionCommitDetector | None = None,
    action_selection: str = "legacy_temporal_aggregation",
    seed: int,
    policy_ticks: int,
    policy_seed: int = 0,
    deterministic_inference: bool = False,
    action_prefix: np.ndarray | None = None,
    action_prefix_provenance: Mapping[str, Any] | None = None,
    save_images: bool = True,
) -> dict[str, Any]:
    """Run bounded feedback execution and write immutable diagnostic evidence."""

    if policy_ticks <= 0:
        raise ValueError("policy_ticks must be positive")
    if action_selection not in ACTION_SELECTION_MODES:
        raise ValueError(
            f"action_selection must be one of {sorted(ACTION_SELECTION_MODES)}"
        )
    prefix = (
        np.empty((0, 4), dtype=np.float32)
        if action_prefix is None
        else np.asarray(action_prefix, dtype=np.float32)
    )
    if prefix.ndim != 2 or prefix.shape[1:] != (4,):
        raise ValueError("action_prefix must have shape (T,4)")
    if prefix.shape[0] > policy_ticks:
        raise ValueError("action_prefix cannot exceed bounded policy ticks")
    if not np.all(np.isfinite(prefix)) or (
        prefix.size and np.max(np.abs(prefix)) > 1.0 + 1.0e-6
    ):
        raise ValueError("action_prefix must contain finite runtime-safe actions")
    if (prefix.shape[0] > 0) != (action_prefix_provenance is not None):
        raise ValueError(
            "non-empty action prefix and action_prefix_provenance are required together"
        )
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable probe output exists: {destination}")
    condition = build_cycle_condition(current_sector, next_sector)
    baseline_id = str(bundle_contract["baseline_id"])
    pass_condition = baseline_id != "B0"
    gated_condition = str(bundle_contract["condition_input"]) == (
        "cycle_condition_v1_dump_end_gated_low_dim"
    )
    if gated_condition != (condition_commit_detector is not None):
        raise ValueError(
            "dump-end-gated condition bundle and condition commit detector "
            "must be provided together"
        )
    ready_detection_enabled = ready_boundary_detector is not None
    lifecycle_enabled = second_next_sector is not None
    if ready_detection_enabled and not pass_condition:
        raise ValueError("condition lifecycle cannot be enabled for B0")
    if lifecycle_enabled and not ready_detection_enabled:
        raise ValueError(
            "ready boundary detector and second_next_sector must be provided together"
        )
    second_condition = (
        None
        if second_next_sector is None
        else build_cycle_condition(next_sector, second_next_sector)
    )
    active_condition = (
        np.zeros_like(condition) if gated_condition else condition.copy()
    )
    condition_committed = bool(pass_condition and not gated_condition)
    condition_commit_policy_tick: int | None = None
    condition_commit_policy_ticks: list[int] = []
    cycle_index = 0
    condition_reset_count = 0
    condition_reset_policy_tick: int | None = None
    requested_cycle_count = 2 if lifecycle_enabled else 1
    observable_cycle_completions: list[dict[str, Any]] = []
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
        if ready_boundary_detector is not None:
            ready_boundary_detector.reset()
        if condition_commit_detector is not None:
            condition_commit_detector.reset()
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
            condition_commit_diagnostic = None
            if condition_commit_detector is not None:
                condition_commit_diagnostic = dict(
                    condition_commit_detector.observe(
                        policy_tick=policy_tick,
                        observation=observation,
                        held_action=held_action,
                    )
                )
                if bool(condition_commit_diagnostic.get("confirmed")):
                    if condition_committed:
                        raise RuntimeError(
                            "condition commit detector confirmed more than once"
                        )
                    committed_condition = (
                        condition if cycle_index == 0 else second_condition
                    )
                    if committed_condition is None:
                        raise AssertionError("active cycle condition is missing")
                    active_condition = committed_condition.copy()
                    condition_committed = True
                    condition_commit_policy_ticks.append(int(policy_tick))
                    if condition_commit_policy_tick is None:
                        condition_commit_policy_tick = int(policy_tick)
            policy_observation, encoded_frames = _policy_observation(
                observation,
                condition=active_condition if pass_condition else None,
                authoritative_time_ns=int(round(target_step * sim_dt * 1.0e9)),
            )
            route_before_predict = getattr(policy, "condition_route_diagnostics", None)
            boundary_diagnostic = None
            reset_before_predict = False
            if ready_boundary_detector is not None:
                boundary_diagnostic = dict(
                    ready_boundary_detector.observe(
                        policy_tick=policy_tick,
                        observation=observation,
                        policy_observation=policy_observation,
                        held_action=held_action,
                        condition_route=route_before_predict,
                    )
                )
                if bool(boundary_diagnostic.get("confirmed")):
                    scripted_target_sector = (
                        next_sector
                        if cycle_index == 0
                        else second_next_sector
                    )
                    if scripted_target_sector is None:
                        raise AssertionError("active scripted target is missing")
                    realized_target_sector = str(
                        boundary_diagnostic.get(
                            "visual_sector_prediction",
                            boundary_diagnostic.get(
                                "qpos_sector",
                                scripted_target_sector,
                            ),
                        )
                    )
                    observable_cycle_completions.append(
                        {
                            "cycle_index": int(cycle_index),
                            "completion_policy_tick": int(policy_tick),
                            "scripted_target_sector": scripted_target_sector,
                            "realized_target_sector": realized_target_sector,
                        }
                    )
                    if lifecycle_enabled and cycle_index == 0:
                        if not hasattr(policy, "reset_condition_cycle"):
                            raise TypeError(
                                "conditioned policy lacks reset_condition_cycle()"
                            )
                        policy.reset_condition_cycle()
                        if second_condition is None:
                            raise AssertionError("second condition was not built")
                        active_condition = (
                            np.zeros_like(second_condition)
                            if gated_condition
                            else second_condition.copy()
                        )
                        condition_committed = not gated_condition
                        if gated_condition:
                            if condition_commit_detector is None:
                                raise AssertionError(
                                    "gated lifecycle lacks commit detector"
                                )
                            condition_commit_detector.reset()
                        ready_boundary_detector.reset()
                        policy_observation["cycle_condition_v1"] = (
                            active_condition.copy()
                        )
                        cycle_index = 1
                        condition_reset_count = 1
                        condition_reset_policy_tick = int(policy_tick)
                        reset_before_predict = True
            aggregated = np.asarray(
                policy.predict(policy_observation),
                dtype=np.float32,
            ).reshape(-1)
            if aggregated.shape != (4,) or not np.all(np.isfinite(aggregated)):
                raise ValueError("policy produced a non-finite or non-4D action")
            aggregation_diagnostics = getattr(
                policy,
                "temporal_aggregation_diagnostics",
                None,
            )
            selected_action = aggregated
            if action_selection in {
                "recency_temporal_aggregation_diagnostic",
                "newest_chunk_head_diagnostic",
            }:
                if not isinstance(aggregation_diagnostics, Mapping):
                    raise RuntimeError(
                        "diagnostic action selection requires temporal "
                        "aggregation diagnostics"
                    )
                diagnostic_key = (
                    "policy_temporal_aggregation_recency_action"
                    if action_selection == "recency_temporal_aggregation_diagnostic"
                    else "policy_temporal_aggregation_newest_action"
                )
                selected_action = np.asarray(
                    aggregation_diagnostics[diagnostic_key],
                    dtype=np.float32,
                ).reshape(-1)
                if selected_action.shape != (4,) or not np.all(
                    np.isfinite(selected_action)
                ):
                    raise ValueError(
                        "diagnostic temporal aggregation produced invalid action"
                    )
            runtime_safe = np.clip(selected_action, -1.0, 1.0).astype(np.float32)
            prefix_override = policy_tick < int(prefix.shape[0])
            actual_sent = (
                prefix[policy_tick].copy()
                if prefix_override
                else runtime_safe.copy()
            )
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
                    "unity_sim_time_ns_diagnostic": int(observation["sim_time_ns"]),
                    "cycle_index": int(cycle_index),
                    "condition": active_condition.astype(float).tolist(),
                    "condition_delivered": bool(pass_condition),
                    "condition_committed": bool(condition_committed),
                    "condition_commit": condition_commit_diagnostic,
                    "condition_router_reset_before_predict": bool(reset_before_predict),
                    "ready_boundary": boundary_diagnostic,
                    "raw_policy_chunk_normalized": raw_normalized.astype(
                        float
                    ).tolist(),
                    "raw_policy_chunk_direct": raw_direct.astype(float).tolist(),
                    "temporal_aggregation_action": aggregated.astype(float).tolist(),
                    "action_selection": action_selection,
                    "selected_action_before_safety_clip": selected_action.astype(
                        float
                    ).tolist(),
                    "temporal_aggregation_diagnostics": (
                        None
                        if aggregation_diagnostics is None
                        else dict(aggregation_diagnostics)
                    ),
                    "future_runtime_safe_action": runtime_safe.astype(float).tolist(),
                    "action_prefix_override": bool(prefix_override),
                    "actual_sent_action": actual_sent.astype(float).tolist(),
                    "condition_route": None if route is None else dict(route),
                    "qpos": np.asarray(observation["qpos"], dtype=np.float32)
                    .astype(float)
                    .tolist(),
                    "qvel": np.asarray(observation["qvel"], dtype=np.float32)
                    .astype(float)
                    .tolist(),
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
            held_action = actual_sent
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
        observable_cycle_completed = (
            len(observable_cycle_completions) == requested_cycle_count
        )
        observable_cycle_completion_tick = (
            int(observable_cycle_completions[-1]["completion_policy_tick"])
            if observable_cycle_completed
            else None
        )
        final_scripted_target_sector = (
            second_next_sector if lifecycle_enabled else next_sector
        )
        final_realized_target_sector = (
            str(observable_cycle_completions[-1]["realized_target_sector"])
            if observable_cycle_completions
            else None
        )
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
                    *(["cycle_condition_v1"] if pass_condition else []),
                ],
                "intervention": {
                    "seed": int(seed),
                    "current_sector": current_sector,
                    "next_sector": next_sector,
                    "second_next_sector": second_next_sector,
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
            "inference_reproducibility": {
                "policy_seed": int(policy_seed),
                "deterministic_algorithms": bool(deterministic_inference),
                "paired_diagnostic_requirement": (
                    "same_environment_seed_same_policy_seed_same_real_stack_commit"
                ),
            },
            "action_selection_contract": {
                "mode": action_selection,
                "single_factor_diagnostic": (
                    action_selection != "legacy_temporal_aggregation"
                ),
                "checkpoint_changed": False,
                "condition_changed": False,
                "ready_boundary_detector_changed": False,
                "legacy_temporal_aggregation_preserved_in_evidence": True,
                "selected_action_saved_before_safety_clip": True,
                "promotable": False,
            },
            "shared_action_prefix_contract": {
                "enabled": bool(prefix.shape[0]),
                "policy_tick_count": int(prefix.shape[0]),
                "closed_loop_policy_takeover_tick": int(prefix.shape[0]),
                "policy_still_evaluated_during_prefix": True,
                "actual_sent_action_overridden_during_prefix": bool(prefix.shape[0]),
                "provenance": (
                    None
                    if action_prefix_provenance is None
                    else dict(action_prefix_provenance)
                ),
                "evidence_scope": (
                    "paired_branch_closed_loop_after_shared_prefix_non_promotable"
                    if prefix.shape[0]
                    else "full_rollout_closed_loop_non_promotable"
                ),
            },
            "condition_contract": {
                "current_sector": current_sector,
                "next_sector": next_sector,
                "committed_vector": condition.astype(float).tolist(),
                "delivered_to_policy": bool(pass_condition),
                "activation": (
                    "first_causal_20hz_row_after_observable_dump_release_end"
                    if gated_condition
                    else "from_probe_start"
                    if pass_condition
                    else "absent"
                ),
                "commit_detector": (
                    None
                    if condition_commit_detector is None
                    else dict(condition_commit_detector.provenance)
                ),
                "committed": bool(condition_committed),
                "commit_policy_tick": condition_commit_policy_tick,
                "commit_policy_ticks": condition_commit_policy_ticks,
            },
            "observable_cycle_contract": {
                "ready_detection_enabled": ready_detection_enabled,
                "observable_cycle_completed": observable_cycle_completed,
                "completion_policy_tick": observable_cycle_completion_tick,
                "scripted_target_sector": final_scripted_target_sector,
                "realized_target_sector": final_realized_target_sector,
                "requested_cycle_count": requested_cycle_count,
                "completed_cycle_count": len(observable_cycle_completions),
                "cycle_completions": observable_cycle_completions,
                "physical_effect_validated": False,
                "detector": (
                    None
                    if ready_boundary_detector is None
                    else dict(ready_boundary_detector.provenance)
                ),
            },
            "condition_lifecycle_contract": {
                "enabled": lifecycle_enabled,
                "schema": "simverify_agx_condition_cycle_lifecycle_v1",
                "transition": (
                    None
                    if second_next_sector is None
                    else {
                        "before": {
                            "current_sector": current_sector,
                            "next_sector": next_sector,
                            "vector": condition.astype(float).tolist(),
                        },
                        "after": {
                            "current_sector": next_sector,
                            "next_sector": second_next_sector,
                            "vector": second_condition.astype(float).tolist(),
                        },
                    }
                ),
                "trigger": (
                    None
                    if ready_boundary_detector is None
                    else dict(ready_boundary_detector.provenance)
                ),
                "reset_count": int(condition_reset_count),
                "reset_policy_tick": condition_reset_policy_tick,
                "full_policy_reset_count": 1,
                "condition_router_reset_only": bool(lifecycle_enabled),
                "temporal_aggregation_reset_at_boundary": False,
                "visual_history_reset_at_boundary": False,
                "privilege_used": False,
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
                            [row["actual_sent_action"] for row in policy_rows],
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


def _normalized_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError(f"{name} must have non-zero norm")
    return np.asarray(vector / norm, dtype=np.float32)


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
            "qpos": np.asarray(observation["qpos"], dtype=np.float32)
            .astype(float)
            .tolist(),
            "qvel": np.asarray(observation["qvel"], dtype=np.float32)
            .astype(float)
            .tolist(),
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
