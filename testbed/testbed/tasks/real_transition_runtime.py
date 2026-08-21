"""Receiver-side task runtime for expert real-transition recording.

The task server never sends actuator commands.  It selects a frozen run,
records experimenter markers against the latest HDF5 row, and requests the
existing recorder state machine to start or stop the continuous run.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from testbed.tasks.home_side_contract import (
    classify_ready_swing_qpos,
    validate_rule_ready_contract,
)
from testbed.tasks.real_transition import (
    CONDITION_SCHEMA,
    DATA_CONTRACT_VERSION,
    REQUIRED_GOAL_ACK_SOURCES,
    SEQUENCE_MANIFEST_SCHEMA,
    TransitionContractError,
    TransitionRunPackage,
    TransitionRunSpec,
    find_run_spec,
    iter_run_specs,
    load_sequence_manifest,
    load_split_manifest,
    sha256_file,
    write_immutable_text,
)

log = logging.getLogger(__name__)

TRANSITION_CONTROL_PROTOCOL_VERSION = 1
TRANSITION_CONTROL_COMMAND = "real_transition.command"
TRANSITION_CONTROL_RESPONSE = "real_transition.response"
DEFAULT_TRANSITION_CONTROL_PORT = 8771
_MAX_FRAME_BYTES = 64 * 1024


@dataclass(frozen=True)
class TransitionStopRequest:
    success: bool
    stop_reason: str


class TransitionTaskRuntime:
    """Thread-safe bridge between task commands and the recorder loop."""

    def __init__(
        self,
        *,
        session_dir: Path | str,
        sequence_manifest_path: Path | str,
        split_manifest_path: Path | str,
        ready_contract_path: Path | str,
        resolved_record_config_yaml: str,
        git_commit: str,
        session_metadata: Mapping[str, Any] | None = None,
        field_context_defaults: Mapping[str, Any] | None = None,
        camera_sync: Mapping[str, Any] | None = None,
        cycle_review_s: float = 45.0,
        cycle_stop_s: float = 60.0,
        run_stop_s: float | None = None,
        run_stop_s_per_cycle: float = 60.0,
    ) -> None:
        self.session_dir = Path(session_dir).resolve()
        self.sequence_manifest_path = Path(sequence_manifest_path).resolve()
        self.split_manifest_path = Path(split_manifest_path).resolve()
        self.ready_contract_path = Path(ready_contract_path).resolve()
        self.git_commit = str(git_commit)
        self.cycle_review_s = float(cycle_review_s)
        self.cycle_stop_s = float(cycle_stop_s)
        self.run_stop_s = None if run_stop_s is None else float(run_stop_s)
        self.run_stop_s_per_cycle = float(run_stop_s_per_cycle)
        self._field_context_defaults = dict(field_context_defaults or {})
        self._camera_sync_config = dict(camera_sync or {})
        self._camera_sync_enabled = bool(
            self._camera_sync_config.get("enabled", False)
        )
        self._camera_sync_expected_cameras = tuple(
            str(value)
            for value in self._camera_sync_config.get(
                "expected_cameras", ("video4", "video5", "video6", "video7")
            )
        )
        self._camera_sync_window_s = float(
            self._camera_sync_config.get("stable_window_s", 2.0)
        )
        self._camera_sync_max_skew_ms = float(
            self._camera_sync_config.get("max_group_skew_ms", 5.0)
        )
        self._camera_sync_min_valid_fraction = float(
            self._camera_sync_config.get("min_valid_fraction", 0.98)
        )
        self._camera_sync_min_distinct_groups = int(
            self._camera_sync_config.get("min_distinct_groups", 30)
        )
        if self._camera_sync_enabled:
            if not self._camera_sync_expected_cameras:
                raise TransitionContractError(
                    "camera sync gate requires expected_cameras"
                )
            if self._camera_sync_window_s <= 0 or self._camera_sync_max_skew_ms < 0:
                raise TransitionContractError("camera sync window/skew is invalid")
            if not 0.0 <= self._camera_sync_min_valid_fraction <= 1.0:
                raise TransitionContractError(
                    "camera sync min_valid_fraction must be in [0,1]"
                )
            if self._camera_sync_min_distinct_groups <= 0:
                raise TransitionContractError(
                    "camera sync min_distinct_groups must be positive"
                )
        if not (0.0 < self.cycle_review_s < self.cycle_stop_s):
            raise TransitionContractError(
                "transition time limits must satisfy 0 < cycle_review < cycle_stop"
            )
        if self.run_stop_s is not None and self.run_stop_s < self.cycle_stop_s:
            raise TransitionContractError(
                "explicit run_stop_s must be at least cycle_stop_s"
            )
        if self.run_stop_s_per_cycle < self.cycle_stop_s:
            raise TransitionContractError(
                "run_stop_s_per_cycle must be at least cycle_stop_s"
            )
        self._lock = threading.RLock()
        self._manifest = load_sequence_manifest(self.sequence_manifest_path)
        if self._manifest.get("schema") != SEQUENCE_MANIFEST_SCHEMA:
            raise TransitionContractError(
                "legacy P0/P1 sequence manifests are read-only; prepare a v2 session"
            )
        self._split_manifest = load_split_manifest(
            self.split_manifest_path,
            sequence_manifest=self._manifest,
        )
        self._run_specs = tuple(iter_run_specs(self._manifest))
        if not self._run_specs:
            raise TransitionContractError("sequence manifest contains no runs")
        if self.sequence_manifest_path.parent != self.session_dir:
            raise TransitionContractError(
                "sequence_manifest must live directly under transition session_dir"
            )
        for name, path in (
            ("split_manifest", self.split_manifest_path),
            ("ready_contract", self.ready_contract_path),
        ):
            if not path.is_file():
                raise TransitionContractError(f"{name} does not exist: {path}")
        try:
            ready_contract = json.loads(
                self.ready_contract_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise TransitionContractError(
                f"cannot read ready contract {self.ready_contract_path}: {exc}"
            ) from exc
        if not isinstance(ready_contract, dict):
            raise TransitionContractError("ready contract root must be an object")
        validate_rule_ready_contract(ready_contract)
        self._ready_contract = ready_contract
        self.resolved_config_path = write_immutable_text(
            self.session_dir / "resolved_record_config.yaml",
            resolved_record_config_yaml,
        )
        self.resolved_config_sha256 = sha256_file(self.resolved_config_path)
        self.session_manifest_path = self._write_session_manifest(
            dict(session_metadata or {})
        )
        self._receiver_mode = "initializing"
        self._receiver_health_ok = False
        self._recording_attached = False
        self._record_start_requested = False
        self._pending_initial_ready_notes: str | None = None
        self._pending_initial_ready_evidence: dict[str, Any] | None = None
        self._pending_initial_ready_source = "experimenter"
        self._stop_request: TransitionStopRequest | None = None
        self._active_spec: TransitionRunSpec | None = None
        self._active_package: TransitionRunPackage | None = None
        self._field_context: dict[str, Any] = {}
        self._episode_idx: int | None = None
        self._last_step_id: int | None = None
        self._last_step_ns: int | None = None
        self._run_start_step_ns: int | None = None
        self._goal_commit_step_ns: int | None = None
        self._timing_warning = ""
        self._ready_samples: deque[tuple[int, np.ndarray, np.ndarray]] = deque()
        self._camera_sync_samples: deque[
            tuple[int, int, bool, float, tuple[str, ...]]
        ] = deque()
        self._last_mark_error = ""
        self._last_mark_result = ""
        self._sealed_run_count = 0
        self._next_run_spec_hint: TransitionRunSpec | None = None
        self._next_run_ordinal: int | None = None
        self._session_progress_error = ""
        self._refresh_session_progress()

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        *,
        config_dir: Path,
        resolved_record_config_yaml: str,
        repo_root: Path,
        session_metadata: Mapping[str, Any] | None = None,
    ) -> TransitionTaskRuntime:
        def resolve_required(name: str) -> Path:
            raw = str(config.get(name, "")).strip()
            if not raw:
                raise TransitionContractError(f"real_transition.{name} is required")
            path = Path(raw).expanduser()
            return (
                (config_dir / path).resolve()
                if not path.is_absolute()
                else path.resolve()
            )

        session_dir = resolve_required("session_dir")
        sequence_path = _resolve_optional_path(
            config.get("sequence_manifest"),
            default=session_dir / "sequence_manifest.json",
            config_dir=config_dir,
        )
        split_path = _resolve_optional_path(
            config.get("split_manifest"),
            default=session_dir / "split_manifest.json",
            config_dir=config_dir,
        )
        ready_path = _resolve_optional_path(
            config.get("ready_contract"),
            default=session_dir / "ready_contract.json",
            config_dir=config_dir,
        )
        time_limits = dict(config.get("time_limits", {}) or {})
        return cls(
            session_dir=session_dir,
            sequence_manifest_path=sequence_path,
            split_manifest_path=split_path,
            ready_contract_path=ready_path,
            resolved_record_config_yaml=resolved_record_config_yaml,
            git_commit=_git_commit(repo_root),
            session_metadata=session_metadata,
            field_context_defaults=dict(
                config.get("field_context_defaults", {}) or {}
            ),
            camera_sync=dict(config.get("camera_sync", {}) or {}),
            cycle_review_s=float(time_limits.get("cycle_review_s", 45.0)),
            cycle_stop_s=float(time_limits.get("cycle_stop_s", 60.0)),
            run_stop_s=(
                None
                if time_limits.get("run_stop_s") is None
                else float(time_limits["run_stop_s"])
            ),
            run_stop_s_per_cycle=float(
                time_limits.get("run_stop_s_per_cycle", 60.0)
            ),
        )

    @property
    def active_raw_path(self) -> Path:
        with self._lock:
            if self._active_package is None:
                raise TransitionContractError("no transition run is active")
            return self._active_package.raw_path

    @property
    def has_active_run(self) -> bool:
        with self._lock:
            return self._active_package is not None

    def update_receiver_state(self, *, mode: str, health_ok: bool) -> None:
        with self._lock:
            self._receiver_mode = str(mode)
            self._receiver_health_ok = bool(health_ok)

    def update_ready_observation(
        self,
        *,
        step_ns: int,
        qpos: Any,
        qvel: Any,
    ) -> None:
        """Update the live rolling window used by initial/target-ready gates."""

        stamp = int(step_ns)
        qpos_array = np.asarray(qpos, dtype=np.float64).reshape(-1)
        qvel_array = np.asarray(qvel, dtype=np.float64).reshape(-1)
        if stamp <= 0:
            raise TransitionContractError("ready observation timestamp must be positive")
        if qpos_array.shape != (4,) or qvel_array.shape != (4,):
            raise TransitionContractError("ready observation qpos/qvel must have shape (4,)")
        if not np.all(np.isfinite(qpos_array)) or not np.all(np.isfinite(qvel_array)):
            raise TransitionContractError("ready observation qpos/qvel must be finite")
        with self._lock:
            if self._ready_samples and stamp <= self._ready_samples[-1][0]:
                if stamp == self._ready_samples[-1][0]:
                    self._ready_samples[-1] = (
                        stamp,
                        qpos_array.copy(),
                        qvel_array.copy(),
                    )
                    return
                self._ready_samples.clear()
            self._ready_samples.append(
                (stamp, qpos_array.copy(), qvel_array.copy())
            )
            retention_ns = int(
                4.0
                * float(self._ready_contract["swing_axis"]["stable_window_s"])
                * 1_000_000_000
            )
            cutoff = stamp - retention_ns
            while self._ready_samples and self._ready_samples[0][0] < cutoff:
                self._ready_samples.popleft()

    def update_camera_sync_observation(
        self,
        *,
        step_ns: int,
        image_metadata: Any,
    ) -> None:
        """Update the rolling four-camera gate without decoding image payloads."""

        if not self._camera_sync_enabled:
            return
        stamp = int(step_ns)
        metadata_map = image_metadata if isinstance(image_metadata, Mapping) else {}
        missing: list[str] = []
        entries: list[Mapping[str, Any]] = []
        for camera in self._camera_sync_expected_cameras:
            value = metadata_map.get(camera)
            if not isinstance(value, Mapping):
                missing.append(camera)
            else:
                entries.append(value)
        group_ids = {
            int(value.get("group_id", 0) or 0) for value in entries
        }
        camera_counts = {
            int(value.get("group_camera_count", 0) or 0) for value in entries
        }
        skews = [
            float(value.get("group_skew_ms", float("inf"))) for value in entries
        ]
        valid = (
            not missing
            and len(entries) == len(self._camera_sync_expected_cameras)
            and len(group_ids) == 1
            and next(iter(group_ids), 0) > 0
            and camera_counts == {len(self._camera_sync_expected_cameras)}
            and all(int(value.get("group_valid", 0) or 0) == 1 for value in entries)
            and not any(int(value.get("v4l2_error", 0) or 0) for value in entries)
            and bool(skews)
            and max(skews) <= self._camera_sync_max_skew_ms
        )
        group_id = next(iter(group_ids), 0) if len(group_ids) == 1 else 0
        skew_ms = max(skews) if skews else float("inf")
        with self._lock:
            if self._camera_sync_samples and stamp <= self._camera_sync_samples[-1][0]:
                if stamp < self._camera_sync_samples[-1][0]:
                    self._camera_sync_samples.clear()
                else:
                    self._camera_sync_samples.pop()
            self._camera_sync_samples.append(
                (stamp, int(group_id), bool(valid), float(skew_ms), tuple(missing))
            )
            cutoff = stamp - int(4.0 * self._camera_sync_window_s * 1_000_000_000)
            while self._camera_sync_samples and self._camera_sync_samples[0][0] < cutoff:
                self._camera_sync_samples.popleft()

    def consume_record_start_request(self) -> bool:
        with self._lock:
            requested = self._record_start_requested
            self._record_start_requested = False
            return requested

    def attach_recording(self, *, episode_idx: int) -> None:
        with self._lock:
            if self._active_package is None:
                raise TransitionContractError(
                    "transition recording cannot attach without an active frozen run"
                )
            if self._recording_attached:
                raise TransitionContractError(
                    "transition recording is already attached"
                )
            self._recording_attached = True
            self._episode_idx = int(episode_idx)
            self._receiver_mode = "recording"

    def update_recorded_step(self, *, step_id: int, step_ns: int) -> None:
        """Publish the latest buffered HDF5 row for exact event alignment."""

        with self._lock:
            if not self._recording_attached or self._active_package is None:
                return
            step_id = int(step_id)
            step_ns = int(step_ns)
            if self._last_step_id is not None:
                if step_id <= self._last_step_id or step_ns <= int(
                    self._last_step_ns or 0
                ):
                    raise TransitionContractError(
                        "recorded transition step_id and step_ns must increase strictly"
                    )
            self._last_step_id = step_id
            self._last_step_ns = step_ns
            if self._active_package.phase == "new":
                self._active_package.start_run(step_id=step_id, step_ns=step_ns)
                self._run_start_step_ns = step_ns
                if self._pending_initial_ready_notes is not None:
                    self._active_package.mark_initial_ready(
                        step_id=step_id,
                        step_ns=step_ns,
                        notes=self._pending_initial_ready_notes,
                        ready_evidence=self._pending_initial_ready_evidence,
                        event_source=self._pending_initial_ready_source,
                    )
                    self._pending_initial_ready_notes = None
                    self._pending_initial_ready_evidence = None
                    self._pending_initial_ready_source = "experimenter"
                    self._commit_goal_automatically(
                        self._active_package,
                        step_id=step_id,
                        step_ns=step_ns,
                    )
            self._evaluate_time_limits(step_id=step_id, step_ns=step_ns)

    def consume_stop_request(self) -> TransitionStopRequest | None:
        with self._lock:
            request = self._stop_request
            self._stop_request = None
            return request

    def abort_on_latest_step(self, *, reason: str, safety_stop: bool) -> None:
        with self._lock:
            if self._active_package is None or self._active_package.phase in {
                "new",
                "complete",
                "aborted",
                "sealed",
            }:
                return
            step_id, step_ns = self._require_latest_step()
            self._active_package.abort_run(
                step_id=step_id,
                step_ns=step_ns,
                reason=reason,
                safety_stop=safety_stop,
            )
            self._stop_request = TransitionStopRequest(
                success=False,
                stop_reason=str(reason),
            )

    def seal_saved_run(
        self, *, raw_path: Path | str, stop_reason: str
    ) -> dict[str, Any]:
        with self._lock:
            package = self._require_active_package()
            if package.phase not in {"complete", "aborted"}:
                step_id, step_ns = self._require_latest_step()
                package.abort_run(
                    step_id=step_id,
                    step_ns=step_ns,
                    reason=stop_reason or "unexpected_recorder_stop",
                    safety_stop=False,
                )
            manifest = package.seal(
                raw_hdf5_path=raw_path,
                git_commit=self.git_commit,
                resolved_config_sha256=self.resolved_config_sha256,
                owner_artifacts={
                    "sequence_manifest": self.sequence_manifest_path,
                    "split_manifest": self.split_manifest_path,
                    "ready_contract": self.ready_contract_path,
                    "resolved_record_config": self.resolved_config_path,
                    "session_manifest": self.session_manifest_path,
                },
                field_context=self._field_context,
                stop_reason=stop_reason,
            )
            self._reset_active_state()
            self._refresh_session_progress()
            return manifest

    def handle_command(
        self, command: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = dict(payload or {})
        with self._lock:
            command = str(command).strip().replace("_", "-")
            if command == "status":
                return self.status()
            if command == "start-run":
                return self._start_run(payload)
            package = self._require_active_package()
            auto_goal_event: dict[str, Any] | None = None
            if command == "initial-ready":
                ready_evidence = self._require_ready_evidence(
                    expected_side=package.run_spec.initial_side,
                    payload=payload,
                )
                if package.phase == "new" and not self._recording_attached:
                    self._pending_initial_ready_notes = str(payload.get("notes", ""))
                    self._pending_initial_ready_evidence = ready_evidence
                    self._pending_initial_ready_source = str(
                        payload.get("event_source", "experimenter")
                    )
                    self._record_start_requested = True
                    result = self.status()
                    result["message"] = (
                        "initial ready accepted; recorder start requested and marker "
                        "will bind to the first HDF5 row"
                    )
                    return result
                step_id, step_ns = self._require_latest_step()
                event = package.mark_initial_ready(
                    step_id=step_id,
                    step_ns=step_ns,
                    notes=str(payload.get("notes", "")),
                    ready_evidence=ready_evidence,
                    event_source=str(payload.get("event_source", "experimenter")),
                )
                auto_goal_event = self._commit_goal_automatically(
                    package,
                    step_id=step_id,
                    step_ns=step_ns,
                )
            elif command == "commit-goal":
                if package.phase == "goal_committed":
                    result = self.status()
                    result["message"] = "goal was already committed automatically"
                    return result
                if not bool(payload.get("display_ack", False)):
                    raise TransitionContractError(
                        "commit-goal requires display_ack=true after the target "
                        "is visible"
                    )
                step_id, step_ns = self._require_latest_step()
                sign_value = payload.get("expected_return_swing_sign")
                sign = None if sign_value is None else int(sign_value)
                event = package.commit_next_goal(
                    step_id=step_id,
                    step_ns=step_ns,
                    commit_ack_sources=REQUIRED_GOAL_ACK_SOURCES,
                    expected_return_swing_sign=sign,
                    notes=str(payload.get("notes", "")),
                )
                self._goal_commit_step_ns = step_ns
                self._timing_warning = ""
            elif command == "dump-end":
                step_id, step_ns = self._require_latest_step()
                event = package.mark_dump_end(
                    step_id=step_id,
                    step_ns=step_ns,
                    notes=str(payload.get("notes", "")),
                    event_source=str(payload.get("event_source", "experimenter")),
                )
            elif command == "target-ready":
                step_id, step_ns = self._require_latest_step()
                ready_evidence = self._require_ready_evidence(
                    expected_side=None,
                    payload=payload,
                )
                realized_target_side = str(ready_evidence["actual_side"])
                ready_evidence["expected_side"] = package.next_target_side
                if realized_target_side != package.next_target_side:
                    event = package.abort_run(
                        step_id=step_id,
                        step_ns=step_ns,
                        reason="realized_target_mismatch",
                        safety_stop=False,
                        realized_target_side=realized_target_side,
                    )
                    self._stop_request = TransitionStopRequest(
                        success=False,
                        stop_reason="realized_target_mismatch",
                    )
                else:
                    event = package.mark_target_ready(
                        step_id=step_id,
                        step_ns=step_ns,
                        realized_target_side=realized_target_side,
                        notes=str(payload.get("notes", "")),
                        ready_evidence=ready_evidence,
                        event_source=str(
                            payload.get("event_source", "experimenter")
                        ),
                    )
                self._goal_commit_step_ns = None
                self._timing_warning = ""
                if package.phase == "cycles_complete":
                    package.complete_run(step_id=step_id, step_ns=step_ns)
                    self._stop_request = TransitionStopRequest(
                        success=True,
                        stop_reason="planned_cycles_complete",
                    )
                elif package.phase == "ready":
                    auto_goal_event = self._commit_goal_automatically(
                        package,
                        step_id=step_id,
                        step_ns=step_ns,
                    )
            elif command == "intervention":
                step_id, step_ns = self._require_latest_step()
                event = package.record_manual_intervention(
                    step_id=step_id,
                    step_ns=step_ns,
                    notes=_required_text(payload, "reason"),
                )
            elif command in {"abort", "safety-stop"}:
                step_id, step_ns = self._require_latest_step()
                reason = _required_text(payload, "reason")
                event = package.abort_run(
                    step_id=step_id,
                    step_ns=step_ns,
                    reason=reason,
                    safety_stop=command == "safety-stop",
                )
                self._stop_request = TransitionStopRequest(
                    success=False,
                    stop_reason=reason,
                )
            else:
                raise TransitionContractError(
                    f"unsupported transition command {command!r}"
                )
            result = self.status()
            result["accepted_event"] = event
            if auto_goal_event is not None:
                result["automatic_goal_event"] = auto_goal_event
            return result

    def handle_mark(self) -> dict[str, Any]:
        """Apply the one-button operator MARK according to the current phase."""

        with self._lock:
            try:
                phase = (
                    self._active_package.phase
                    if self._active_package is not None
                    else "idle"
                )
                if phase == "idle":
                    result = self._start_run(
                        {"field_context": self._automatic_field_context()}
                    )
                    action = "start-run"
                elif phase == "new":
                    result = self.handle_command(
                        "initial-ready",
                        {
                            "bucket_clear_confirmed": True,
                            "operator_confirmed": True,
                            "notes": "operator joystick MARK",
                            "event_source": "operator",
                        },
                    )
                    action = "initial-ready"
                elif phase == "goal_committed":
                    result = self.handle_command(
                        "dump-end",
                        {
                            "notes": "operator joystick MARK",
                            "event_source": "operator",
                        },
                    )
                    action = "dump-end"
                elif phase == "dump_marked":
                    result = self.handle_command(
                        "target-ready",
                        {
                            "bucket_clear_confirmed": True,
                            "operator_confirmed": True,
                            "notes": "operator joystick MARK",
                            "event_source": "operator",
                        },
                    )
                    action = "target-ready"
                else:
                    raise TransitionContractError(
                        f"MARK is not accepted while phase={phase}"
                    )
            except Exception as exc:
                self._last_mark_error = str(exc)
                self._last_mark_result = "rejected"
                raise
            self._last_mark_error = ""
            self._last_mark_result = action
            result["mark_action"] = action
            return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            package = self._active_package
            spec = self._active_spec
            return {
                "receiver_mode": self._receiver_mode,
                "receiver_health_ok": self._receiver_health_ok,
                "session_id": str(self._manifest["session_id"]),
                "sealed_run_count": self._sealed_run_count,
                "next_run_id": (
                    self._next_run_spec_hint.run_id
                    if self._next_run_spec_hint is not None
                    else None
                ),
                "next_run_ordinal": self._next_run_ordinal,
                "next_field_context": (
                    self._automatic_field_context()
                    if package is None and self._next_run_spec_hint is not None
                    else {}
                ),
                "session_progress_error": self._session_progress_error,
                "active": package is not None,
                "recording_attached": self._recording_attached,
                "run_id": spec.run_id if spec is not None else None,
                "block_id": spec.block_id if spec is not None else None,
                "split": spec.split if spec is not None else None,
                "sequence_id": spec.sequence_id if spec is not None else None,
                "legacy_template_id": (
                    spec.legacy_template_id if spec is not None else None
                ),
                "matched_start_pair_id": (
                    spec.matched_start_pair_id if spec is not None else None
                ),
                "paired_run_id": spec.paired_run_id if spec is not None else None,
                "matched_start_pair_member_rank": (
                    spec.matched_start_pair_member_rank
                    if spec is not None
                    else None
                ),
                "initial_side": spec.initial_side if spec is not None else None,
                "planned_cycle_count": spec.cycle_count if spec is not None else None,
                "planned_sequence": (
                    list(spec.sequence) if spec is not None else None
                ),
                "phase": package.phase if package is not None else "idle",
                "field_context": dict(self._field_context),
                "completed_cycles": package.cycle_index if package is not None else 0,
                "next_target_side": (
                    package.next_target_side if package is not None else None
                ),
                "next_target_side_code": (
                    None
                    if package is None or package.next_target_side is None
                    else (-1 if package.next_target_side == "A" else 1)
                ),
                "last_step_id": self._last_step_id,
                "last_step_ns": self._last_step_ns,
                "raw_path": str(package.raw_path) if package is not None else "",
                "stop_requested": self._stop_request is not None,
                "cycle_elapsed_s": self._elapsed_s(self._goal_commit_step_ns),
                "run_elapsed_s": self._elapsed_s(self._run_start_step_ns),
                "timing_warning": self._timing_warning,
                "mark_next_action": self._mark_next_action(package),
                "last_mark_result": self._last_mark_result,
                "last_mark_error": self._last_mark_error,
                "ready_state": self._ready_state_snapshot(),
                "camera_sync_state": self._camera_sync_state_snapshot(),
                "time_limits_s": {
                    "cycle_review": self.cycle_review_s,
                    "cycle_stop": self.cycle_stop_s,
                    "run_stop": (
                        self._effective_run_stop_s(spec) if spec is not None else None
                    ),
                    "run_stop_per_cycle": self.run_stop_s_per_cycle,
                    "max_planned_run_stop": max(
                        self._effective_run_stop_s(run_spec)
                        for run_spec in self._run_specs
                    ),
                },
                "planned_run_count": len(self._run_specs),
                "max_planned_cycle_count": max(
                    run_spec.cycle_count for run_spec in self._run_specs
                ),
            }

    def _ready_state_snapshot(self) -> dict[str, Any]:
        swing = self._ready_contract["swing_axis"]
        end_limit = (
            self._last_step_ns
            if self._recording_attached and self._last_step_ns is not None
            else None
        )
        samples = [
            sample
            for sample in self._ready_samples
            if end_limit is None or sample[0] <= end_limit
        ]
        base = {
            "contract_schema": self._ready_contract["schema"],
            "window_required_s": float(swing["stable_window_s"]),
            "swing_qvel_limit_rad_s": float(
                swing["swing_qvel_abs_max_rad_s"]
            ),
            "sample_count": 0,
            "window_duration_s": 0.0,
            "window_complete": False,
            "sample_gap_ok": False,
            "swing_stable": False,
            "clean_side_window": False,
            "actual_side": "unknown",
            "blockers": [],
        }
        if not samples:
            base["blockers"] = ["no_ready_observations"]
            return base
        latest_ns = int(samples[-1][0])
        window_ns = int(float(swing["stable_window_s"]) * 1_000_000_000)
        cutoff_ns = latest_ns - window_ns
        start_index = next(
            index
            for index, sample in enumerate(samples)
            if sample[0] >= cutoff_ns
        )
        # Keep the sample immediately before the cutoff when no sample lands
        # exactly on it.  Real 50 Hz observations are not phase-locked to a
        # 0.5 s boundary; dropping this predecessor makes the measured span
        # permanently shorter than 0.5 s (typically about 0.49 s).  Including
        # it is conservative because its qpos/qvel also participates in every
        # ready gate, while max_sample_gap_s still rejects missing coverage.
        if samples[start_index][0] > cutoff_ns and start_index > 0:
            start_index -= 1
        window = samples[start_index:]
        timestamps = np.asarray([sample[0] for sample in window], dtype=np.int64)
        qpos = np.stack([sample[1] for sample in window])
        qvel = np.stack([sample[2] for sample in window])
        duration_s = float((timestamps[-1] - timestamps[0]) / 1_000_000_000.0)
        max_gap_s = (
            float(np.max(np.diff(timestamps)) / 1_000_000_000.0)
            if len(timestamps) > 1
            else None
        )
        classifications = [
            classify_ready_swing_qpos(self._ready_contract, value)
            for value in qpos[:, 0]
        ]
        actual_side = classifications[-1]
        window_complete = duration_s + 1e-9 >= float(swing["stable_window_s"])
        sample_gap_ok = max_gap_s is not None and max_gap_s <= float(
            swing["max_sample_gap_s"]
        )
        swing_qvel_abs_max = float(np.max(np.abs(qvel[:, 0])))
        swing_stable = swing_qvel_abs_max <= float(
            swing["swing_qvel_abs_max_rad_s"]
        )
        clean_side_window = actual_side in {"A", "B"} and all(
            value == actual_side for value in classifications
        )
        blockers = []
        if not window_complete:
            blockers.append("swing_window_too_short")
        if not sample_gap_ok:
            blockers.append("swing_window_sample_gap")
        if not swing_stable:
            blockers.append("swing_not_stable")
        if not clean_side_window:
            blockers.append(f"swing_side_{actual_side}")
        return {
            **base,
            "sample_count": int(len(window)),
            "window_start_ns": int(timestamps[0]),
            "window_end_ns": int(timestamps[-1]),
            "window_duration_s": duration_s,
            "window_complete": window_complete,
            "max_sample_gap_s": max_gap_s,
            "sample_gap_ok": sample_gap_ok,
            "swing_qpos_current_rad": float(qpos[-1, 0]),
            "swing_qpos_window_min_rad": float(np.min(qpos[:, 0])),
            "swing_qpos_window_max_rad": float(np.max(qpos[:, 0])),
            "swing_qvel_abs_max_rad_s": swing_qvel_abs_max,
            "swing_stable": swing_stable,
            "clean_side_window": clean_side_window,
            "actual_side": actual_side,
            "non_swing_qpos_current_rad": qpos[-1, 1:].tolist(),
            "non_swing_qvel_abs_max_rad_s": np.max(
                np.abs(qvel[:, 1:]), axis=0
            ).tolist(),
            "non_swing_axes_gate_ready": False,
            "blockers": blockers,
        }

    def _require_ready_evidence(
        self,
        *,
        expected_side: str | None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if payload.get("bucket_clear_confirmed") is not True:
            raise TransitionContractError(
                "ready requires bucket_clear_confirmed=true"
            )
        if payload.get("operator_confirmed") is not True:
            raise TransitionContractError(
                "ready requires operator_confirmed=true"
            )
        evidence = self._ready_state_snapshot()
        blockers = list(evidence.get("blockers", ()))
        if blockers:
            raise TransitionContractError(
                "ready contract blocked: " + ", ".join(blockers)
            )
        camera_sync = self._camera_sync_state_snapshot()
        camera_blockers = list(camera_sync.get("blockers", ()))
        if camera_blockers:
            raise TransitionContractError(
                "camera sync gate blocked: " + ", ".join(camera_blockers)
            )
        actual_side = str(evidence["actual_side"])
        if expected_side is not None and actual_side != expected_side:
            raise TransitionContractError(
                f"actual ready side {actual_side} does not match expected {expected_side}"
            )
        evidence["expected_side"] = expected_side
        evidence["bucket_clear_confirmed"] = True
        evidence["operator_confirmed"] = True
        evidence["ready_contract_sha256"] = self._ready_contract[
            "contract_sha256"
        ]
        evidence["camera_sync"] = camera_sync
        return evidence

    def _start_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._active_package is not None:
            active_run_id = self._active_spec.run_id if self._active_spec else "?"
            raise TransitionContractError(
                f"run {active_run_id} is already active"
            )
        if self._receiver_mode != "armed":
            raise TransitionContractError(
                f"start-run requires receiver_mode=armed, got {self._receiver_mode}"
            )
        if not self._receiver_health_ok:
            raise TransitionContractError("start-run is blocked by receiver health")
        camera_blockers = list(
            self._camera_sync_state_snapshot().get("blockers", ())
        )
        if camera_blockers:
            raise TransitionContractError(
                "start-run is blocked by camera sync: "
                + ", ".join(camera_blockers)
            )
        requested_run_id = str(payload.get("run_id", "")).strip()
        spec = (
            find_run_spec(self._manifest, requested_run_id)
            if requested_run_id
            else self._next_available_run()
        )
        field_context = _validated_field_context(payload.get("field_context"))
        run_dir = self.session_dir / f"block_{spec.block_id}" / f"run_{spec.run_id}"
        package = TransitionRunPackage(run_dir=run_dir, run_spec=spec)
        self._active_spec = spec
        self._active_package = package
        self._field_context = field_context
        self._field_context["planned_matched_start_pair_id"] = (
            spec.matched_start_pair_id
        )
        self._field_context["planned_matched_start_pair_member_rank"] = (
            spec.matched_start_pair_member_rank
        )
        self._field_context["planned_sequence_id"] = spec.sequence_id
        self._field_context["planned_cycle_count"] = spec.cycle_count
        self._record_start_requested = False
        self._pending_initial_ready_notes = None
        self._pending_initial_ready_evidence = None
        self._pending_initial_ready_source = "experimenter"
        self._stop_request = None
        self._recording_attached = False
        self._last_step_id = None
        self._last_step_ns = None
        self._run_start_step_ns = None
        self._goal_commit_step_ns = None
        self._timing_warning = ""
        result = self.status()
        result["message"] = (
            "run selected; place the machine at initial_side, then send initial-ready; "
            "targets are already frozen and must not be changed from field observations"
        )
        return result

    def _automatic_field_context(self) -> dict[str, str]:
        ordinal = int(self._next_run_ordinal or (self._sealed_run_count + 1))
        prefix = str(
            self._field_context_defaults.get("workface_reset_id_prefix", "wf_")
            or "wf_"
        )
        action = str(
            self._field_context_defaults.get("workface_action", "fresh_strip")
            or "fresh_strip"
        )
        return {
            "workface_reset_id": f"{prefix}{ordinal:03d}",
            "workface_action": action,
            "context_source": "automatic_joystick_mark",
        }

    def _commit_goal_automatically(
        self,
        package: TransitionRunPackage,
        *,
        step_id: int,
        step_ns: int,
    ) -> dict[str, Any] | None:
        if package.phase != "ready" or package.next_target_side is None:
            return None
        event = package.commit_next_goal(
            step_id=step_id,
            step_ns=step_ns,
            commit_ack_sources=REQUIRED_GOAL_ACK_SOURCES,
            notes="automatic frozen-sequence goal commit",
        )
        self._goal_commit_step_ns = int(step_ns)
        self._timing_warning = ""
        return event

    @staticmethod
    def _mark_next_action(package: TransitionRunPackage | None) -> str:
        phase = package.phase if package is not None else "idle"
        return {
            "idle": "start-run",
            "new": "initial-ready",
            "goal_committed": "dump-end",
            "dump_marked": "target-ready",
            "cycles_complete": "wait-save",
            "complete": "wait-save",
            "aborted": "wait-save",
        }.get(phase, "wait")

    def _camera_sync_state_snapshot(self) -> dict[str, Any]:
        base = {
            "enabled": self._camera_sync_enabled,
            "expected_cameras": list(self._camera_sync_expected_cameras),
            "window_required_s": self._camera_sync_window_s,
            "max_group_skew_ms": self._camera_sync_max_skew_ms,
            "min_valid_fraction": self._camera_sync_min_valid_fraction,
            "min_distinct_groups": self._camera_sync_min_distinct_groups,
            "sample_count": 0,
            "distinct_group_count": 0,
            "valid_fraction": 0.0,
            "observed_max_skew_ms": None,
            "window_duration_s": 0.0,
            "ready": not self._camera_sync_enabled,
            "blockers": [],
        }
        if not self._camera_sync_enabled:
            return base
        samples = list(self._camera_sync_samples)
        if not samples:
            base["blockers"] = ["camera_sync_no_samples"]
            return base
        latest_ns = int(samples[-1][0])
        cutoff_ns = latest_ns - int(self._camera_sync_window_s * 1_000_000_000)
        start_index = next(
            (index for index, sample in enumerate(samples) if sample[0] >= cutoff_ns),
            len(samples) - 1,
        )
        if samples[start_index][0] > cutoff_ns and start_index > 0:
            start_index -= 1
        window = samples[start_index:]
        duration_s = float((window[-1][0] - window[0][0]) / 1_000_000_000.0)
        valid_fraction = sum(bool(sample[2]) for sample in window) / len(window)
        distinct_groups = {sample[1] for sample in window if sample[1] > 0}
        finite_skews = [sample[3] for sample in window if np.isfinite(sample[3])]
        max_skew = max(finite_skews) if finite_skews else None
        missing = sorted({camera for sample in window for camera in sample[4]})
        blockers: list[str] = []
        if duration_s + 1e-9 < self._camera_sync_window_s:
            blockers.append("camera_sync_window_too_short")
        if len(distinct_groups) < self._camera_sync_min_distinct_groups:
            blockers.append("camera_sync_groups_too_few")
        if valid_fraction < self._camera_sync_min_valid_fraction:
            blockers.append("camera_sync_valid_fraction")
        if max_skew is None or max_skew > self._camera_sync_max_skew_ms:
            blockers.append("camera_sync_skew")
        if missing:
            blockers.append("camera_sync_missing_" + ",".join(missing))
        return {
            **base,
            "sample_count": len(window),
            "distinct_group_count": len(distinct_groups),
            "valid_fraction": float(valid_fraction),
            "observed_max_skew_ms": None if max_skew is None else float(max_skew),
            "window_duration_s": duration_s,
            "missing_cameras": missing,
            "ready": not blockers,
            "blockers": blockers,
        }

    def _next_available_run(self) -> TransitionRunSpec:
        for spec in self._run_specs:
            run_dir = self.session_dir / f"block_{spec.block_id}" / f"run_{spec.run_id}"
            if not run_dir.exists():
                return spec
            if (run_dir / "run_manifest.json").is_file():
                continue
            raise TransitionContractError(
                f"next run directory is unsealed and requires review: {run_dir}"
            )
        raise TransitionContractError("all frozen runs already have sealed packages")

    def _refresh_session_progress(self) -> None:
        sealed = 0
        next_spec: TransitionRunSpec | None = None
        next_ordinal: int | None = None
        progress_error = ""
        for ordinal, spec in enumerate(self._run_specs, start=1):
            run_dir = self.session_dir / f"block_{spec.block_id}" / f"run_{spec.run_id}"
            if not run_dir.exists():
                if next_spec is None:
                    next_spec = spec
                    next_ordinal = ordinal
                continue
            if (run_dir / "run_manifest.json").is_file():
                sealed += 1
                continue
            progress_error = f"unsealed run requires review: {run_dir}"
            if next_spec is None:
                next_spec = spec
                next_ordinal = ordinal
            break
        self._sealed_run_count = sealed
        self._next_run_spec_hint = next_spec
        self._next_run_ordinal = next_ordinal
        self._session_progress_error = progress_error

    def _require_active_package(self) -> TransitionRunPackage:
        if self._active_package is None:
            raise TransitionContractError("no transition run is active")
        return self._active_package

    def _require_latest_step(self) -> tuple[int, int]:
        if not self._recording_attached:
            raise TransitionContractError("transition recorder is not attached")
        if self._last_step_id is None or self._last_step_ns is None:
            raise TransitionContractError(
                "no HDF5 row is available for event alignment"
            )
        return self._last_step_id, self._last_step_ns

    def _reset_active_state(self) -> None:
        self._active_spec = None
        self._active_package = None
        self._field_context = {}
        self._episode_idx = None
        self._recording_attached = False
        self._record_start_requested = False
        self._pending_initial_ready_notes = None
        self._pending_initial_ready_evidence = None
        self._pending_initial_ready_source = "experimenter"
        self._stop_request = None
        self._last_step_id = None
        self._last_step_ns = None
        self._run_start_step_ns = None
        self._goal_commit_step_ns = None
        self._timing_warning = ""

    def _write_session_manifest(self, metadata: Mapping[str, Any]) -> Path:
        path = self.session_dir / "session_manifest.json"
        payload = {
            "schema": "real_transition_session_manifest_v2",
            "data_contract_version": DATA_CONTRACT_VERSION,
            "condition_schema": CONDITION_SCHEMA,
            "session_id": str(self._manifest["session_id"]),
            "prepared_at_utc": str(self._manifest.get("created_at_utc", "")),
            "recording_mode": "expert_teleop_only",
            "policy_loaded": False,
            "n5_bundle": {
                "status": "not_loaded_expert_recording",
                "required_before_b1_policy_test": True,
            },
            "git_commit": self.git_commit,
            "machine_id": str(metadata.get("machine_id", "")),
            "operator_id": str(metadata.get("operator_id", "")),
            "artifacts": {
                "sequence_manifest.json": sha256_file(self.sequence_manifest_path),
                "split_manifest.json": sha256_file(self.split_manifest_path),
                "ready_contract.json": sha256_file(self.ready_contract_path),
                "resolved_record_config.yaml": self.resolved_config_sha256,
            },
            "raw_package_mutation_policy": "append_until_seal_then_immutable",
            "sequence_mode": "seeded_balanced_frozen_multisequence",
        }
        write_immutable_text(
            path,
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
        )
        return path

    def _evaluate_time_limits(self, *, step_id: int, step_ns: int) -> None:
        package = self._active_package
        spec = self._active_spec
        if (
            package is None
            or spec is None
            or package.phase in {"complete", "aborted", "sealed"}
        ):
            return
        run_elapsed = self._elapsed_s(self._run_start_step_ns, now_ns=step_ns)
        run_stop_s = self._effective_run_stop_s(spec)
        if run_elapsed is not None and run_elapsed >= run_stop_s:
            package.abort_run(
                step_id=step_id,
                step_ns=step_ns,
                reason="run_timeout",
            )
            self._stop_request = TransitionStopRequest(
                success=False,
                stop_reason="run_timeout",
            )
            self._timing_warning = "run_stop"
            return
        cycle_elapsed = self._elapsed_s(self._goal_commit_step_ns, now_ns=step_ns)
        if cycle_elapsed is None:
            return
        if cycle_elapsed >= self.cycle_stop_s:
            package.abort_run(
                step_id=step_id,
                step_ns=step_ns,
                reason="cycle_timeout",
            )
            self._stop_request = TransitionStopRequest(
                success=False,
                stop_reason="cycle_timeout",
            )
            self._timing_warning = "cycle_stop"
        elif cycle_elapsed >= self.cycle_review_s:
            self._timing_warning = "cycle_review"

    def _effective_run_stop_s(self, spec: TransitionRunSpec) -> float:
        if self.run_stop_s is not None:
            return self.run_stop_s
        return self.run_stop_s_per_cycle * spec.cycle_count

    def _elapsed_s(
        self,
        start_ns: int | None,
        *,
        now_ns: int | None = None,
    ) -> float | None:
        if start_ns is None:
            return None
        current_ns = self._last_step_ns if now_ns is None else int(now_ns)
        if current_ns is None:
            return None
        return max(0.0, (int(current_ns) - int(start_ns)) / 1_000_000_000.0)


class TransitionTaskServer:
    """Small one-command-per-connection TCP server for experimenter markers."""

    def __init__(
        self,
        *,
        runtime: TransitionTaskRuntime,
        bind_host: str = "0.0.0.0",
        port: int = DEFAULT_TRANSITION_CONTROL_PORT,
    ) -> None:
        self.runtime = runtime
        self.bind_host = str(bind_host)
        self.port = int(port)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind_host, self.port))
        self.port = int(sock.getsockname()[1])
        sock.listen(8)
        sock.settimeout(0.5)
        self._socket = sock
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._serve,
            name="real-transition-task-server",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "Real-transition task server listening on %s:%d.",
            self.bind_host,
            self.port,
        )

    def close(self) -> None:
        self._stop.set()
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)

    def _serve(self) -> None:
        while not self._stop.is_set():
            sock = self._socket
            if sock is None:
                break
            try:
                connection, _address = sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(2.0)
                response = self._handle_connection(connection)
                try:
                    connection.sendall(_encode_response(response))
                except OSError:
                    pass

    def _handle_connection(self, connection: socket.socket) -> dict[str, Any]:
        try:
            frame = _receive_frame(connection)
            command, payload = decode_transition_command(frame)
            result = self.runtime.handle_command(command, payload)
            return {"ok": True, "result": result, "error": ""}
        except Exception as exc:
            if not isinstance(exc, TransitionContractError):
                log.exception("Unexpected real-transition command failure.")
            return {"ok": False, "result": {}, "error": str(exc)}


def encode_transition_command(
    command: str, payload: Mapping[str, Any] | None = None
) -> bytes:
    message = {
        "version": TRANSITION_CONTROL_PROTOCOL_VERSION,
        "type": TRANSITION_CONTROL_COMMAND,
        "command": str(command),
        "payload": dict(payload or {}),
        "client_time_ns": time.time_ns(),
    }
    return (
        json.dumps(message, separators=(",", ":"), sort_keys=True).encode("utf-8")
        + b"\n"
    )


def decode_transition_command(frame: bytes | str) -> tuple[str, dict[str, Any]]:
    value = _decode_protocol_frame(frame, expected_type=TRANSITION_CONTROL_COMMAND)
    command = str(value.get("command", "")).strip()
    if not command:
        raise TransitionContractError("transition command is empty")
    payload = value.get("payload", {})
    if not isinstance(payload, Mapping):
        raise TransitionContractError("transition command payload must be an object")
    return command, dict(payload)


def decode_transition_response(frame: bytes | str) -> dict[str, Any]:
    value = _decode_protocol_frame(frame, expected_type=TRANSITION_CONTROL_RESPONSE)
    if not bool(value.get("ok", False)):
        raise TransitionContractError(
            str(value.get("error", "transition command failed"))
        )
    result = value.get("result", {})
    if not isinstance(result, Mapping):
        raise TransitionContractError("transition response result must be an object")
    return dict(result)


def send_transition_command(
    *,
    host: str,
    port: int,
    command: str,
    payload: Mapping[str, Any] | None = None,
    timeout_s: float = 3.0,
) -> dict[str, Any]:
    with socket.create_connection(
        (str(host), int(port)), timeout=float(timeout_s)
    ) as sock:
        sock.settimeout(float(timeout_s))
        sock.sendall(encode_transition_command(command, payload))
        return decode_transition_response(_receive_frame(sock))


def _encode_response(payload: Mapping[str, Any]) -> bytes:
    message = {
        "version": TRANSITION_CONTROL_PROTOCOL_VERSION,
        "type": TRANSITION_CONTROL_RESPONSE,
        **dict(payload),
    }
    return (
        json.dumps(message, separators=(",", ":"), sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _decode_protocol_frame(frame: bytes | str, *, expected_type: str) -> dict[str, Any]:
    text = frame.decode("utf-8") if isinstance(frame, bytes) else str(frame)
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise TransitionContractError(
            f"invalid transition protocol JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TransitionContractError("transition protocol frame must be an object")
    if int(value.get("version", -1)) != TRANSITION_CONTROL_PROTOCOL_VERSION:
        raise TransitionContractError("unsupported transition protocol version")
    if value.get("type") != expected_type:
        raise TransitionContractError(
            f"unexpected transition protocol type {value.get('type')!r}"
        )
    return value


def _receive_frame(sock: socket.socket) -> bytes:
    payload = bytearray()
    while len(payload) < _MAX_FRAME_BYTES:
        chunk = sock.recv(min(4096, _MAX_FRAME_BYTES - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
        newline = payload.find(b"\n")
        if newline >= 0:
            return bytes(payload[:newline])
    if len(payload) >= _MAX_FRAME_BYTES:
        raise TransitionContractError("transition protocol frame exceeds 64 KiB")
    if not payload:
        raise TransitionContractError("empty transition protocol frame")
    return bytes(payload)


def _resolve_optional_path(value: Any, *, default: Path, config_dir: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return default.resolve()
    path = Path(raw).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _validated_field_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TransitionContractError(
            "start-run requires a field_context object"
        )
    context = dict(value)
    for field in ("workface_reset_id", "workface_action"):
        text = str(context.get(field, "")).strip()
        if not text:
            raise TransitionContractError(
                f"start-run field_context requires {field}"
            )
        context[field] = text
    return context


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = str(payload.get(field, "")).strip()
    if not value:
        raise TransitionContractError(f"transition command requires {field}")
    return value


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise TransitionContractError(
            "cannot resolve git commit for transition provenance: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()
