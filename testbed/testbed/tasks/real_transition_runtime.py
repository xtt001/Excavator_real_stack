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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from testbed.tasks.home_side_contract import validate_home_side_contract
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
        home_side_contract_path: Path | str,
        resolved_record_config_yaml: str,
        git_commit: str,
        session_metadata: Mapping[str, Any] | None = None,
        cycle_review_s: float = 45.0,
        cycle_stop_s: float = 60.0,
        run_stop_s: float | None = None,
        run_stop_s_per_cycle: float = 60.0,
    ) -> None:
        self.session_dir = Path(session_dir).resolve()
        self.sequence_manifest_path = Path(sequence_manifest_path).resolve()
        self.split_manifest_path = Path(split_manifest_path).resolve()
        self.home_side_contract_path = Path(home_side_contract_path).resolve()
        self.git_commit = str(git_commit)
        self.cycle_review_s = float(cycle_review_s)
        self.cycle_stop_s = float(cycle_stop_s)
        self.run_stop_s = None if run_stop_s is None else float(run_stop_s)
        self.run_stop_s_per_cycle = float(run_stop_s_per_cycle)
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
            ("home_side_contract", self.home_side_contract_path),
        ):
            if not path.is_file():
                raise TransitionContractError(f"{name} does not exist: {path}")
        try:
            home_contract = json.loads(
                self.home_side_contract_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise TransitionContractError(
                f"cannot read home-side contract {self.home_side_contract_path}: {exc}"
            ) from exc
        if not isinstance(home_contract, dict):
            raise TransitionContractError("home-side contract root must be an object")
        validate_home_side_contract(home_contract)
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
        home_path = _resolve_optional_path(
            config.get("home_side_contract"),
            default=session_dir / "home_side_contract.json",
            config_dir=config_dir,
        )
        time_limits = dict(config.get("time_limits", {}) or {})
        return cls(
            session_dir=session_dir,
            sequence_manifest_path=sequence_path,
            split_manifest_path=split_path,
            home_side_contract_path=home_path,
            resolved_record_config_yaml=resolved_record_config_yaml,
            git_commit=_git_commit(repo_root),
            session_metadata=session_metadata,
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
                    )
                    self._pending_initial_ready_notes = None
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
                    "home_side_contract": self.home_side_contract_path,
                    "resolved_record_config": self.resolved_config_path,
                    "session_manifest": self.session_manifest_path,
                },
                field_context=self._field_context,
                stop_reason=stop_reason,
            )
            self._reset_active_state()
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
            if command == "initial-ready":
                if package.phase == "new" and not self._recording_attached:
                    self._pending_initial_ready_notes = str(payload.get("notes", ""))
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
                )
            elif command == "commit-goal":
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
                )
            elif command == "target-ready":
                step_id, step_ns = self._require_latest_step()
                realized_target_side = str(
                    payload.get("realized_target_side", "")
                ).strip()
                if realized_target_side not in {"A", "B"}:
                    raise TransitionContractError(
                        "target-ready requires realized_target_side A or B"
                    )
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
                    )
                self._goal_commit_step_ns = None
                self._timing_warning = ""
                if package.phase == "cycles_complete":
                    package.complete_run(step_id=step_id, step_ns=step_ns)
                    self._stop_request = TransitionStopRequest(
                        success=True,
                        stop_reason="planned_cycles_complete",
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
            return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            package = self._active_package
            spec = self._active_spec
            return {
                "receiver_mode": self._receiver_mode,
                "receiver_health_ok": self._receiver_health_ok,
                "session_id": str(self._manifest["session_id"]),
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
                "home_side_contract.json": sha256_file(self.home_side_contract_path),
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
