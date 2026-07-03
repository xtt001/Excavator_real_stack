"""
tb-receiver-real / tb-record-real - Run real excavator receiver and HDF5 record.

The receiver owns the live bridge/gateway/remote-action/control loop. A record
session is opened only after the record-start gate, and failed health gates are
quarantined under the dataset failed/ directory instead of the training set.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from testbed.actions.base import ActionInfo
from testbed.data.schema import (
    ATTR_ACTION_ORDER,
    ATTR_ACTION_SEMANTICS,
    ATTR_CAMERA_FPS,
    ATTR_CAMERA_HEIGHT,
    ATTR_CAMERA_NAMES,
    ATTR_CAMERA_ROW_ORDER,
    ATTR_CAMERA_WIDTH,
    ATTR_CONTROL_HZ,
    ATTR_DT,
    ATTR_EPISODE_ID,
    ATTR_HYDRAULIC_CYLINDER_AVAILABLE,
    ATTR_IMAGE_FORMAT,
    ATTR_IS_REAL,
    ATTR_NOTES,
    ATTR_OPERATOR_ID,
    ATTR_PARAM_VERSION,
    ATTR_PLATFORM,
    ATTR_QPOS_ORDER,
    ATTR_QPOS_SOURCE,
    ATTR_QPOS_UNITS,
    ATTR_QVEL_ORDER,
    ATTR_QVEL_SOURCE,
    ATTR_QVEL_UNITS,
    ATTR_RECORD_CONFIG_PATH,
    ATTR_RECORD_CONFIG_YAML,
    ATTR_SEED,
    ATTR_SESSION_ID,
    ATTR_TASK_NAME,
    ATTR_TELEOP_INPUT,
    DEFAULT_PLATFORM,
)

log = logging.getLogger(__name__)

_REAL_AXIS_NAMES = ("swing", "boom", "stick", "bucket")


def _go_home_start_context(receiver_mode: str, has_record_session: bool) -> str | None:
    """Return the active context if go-home may start from this receiver state."""
    if receiver_mode == "armed" and not has_record_session:
        return "armed"
    if receiver_mode == "recording" and has_record_session:
        return "recording"
    return None


def _publish_remote_receiver_status(
    action_source: Any,
    *,
    receiver_mode: str,
    episode_idx: int,
    saved: int,
    record_session: Any | None,
    go_home_update: Any | None = None,
    receiver_health: Any | None = None,
    record_steps: int | None = None,
    saved_path: Any | None = None,
    action_info: Any | None = None,
    message: str = "",
) -> None:
    publish = getattr(action_source, "publish_status", None)
    if not callable(publish):
        return
    steps = (
        int(record_steps)
        if record_steps is not None
        else int(len(record_session)) if record_session is not None else 0
    )
    go_home_result = ""
    if go_home_update is not None:
        if bool(getattr(go_home_update, "failed", False)):
            go_home_result = (
                "failed:" + str(getattr(go_home_update, "reason", "") or "unknown")
            )
        elif bool(getattr(go_home_update, "done", False)):
            go_home_result = "done"
        elif receiver_mode == "go_home":
            go_home_result = "running"
    elif receiver_mode == "go_home":
        go_home_result = "running"
    health_ok = True if receiver_health is None else bool(receiver_health.ok)
    health_error = (
        ""
        if receiver_health is None
        else str(getattr(receiver_health, "error_code", "") or "")
    )
    payload = {
        "timestamp_ns": int(time.time_ns()),
        "receiver_mode": str(receiver_mode),
        "episode_idx": int(episode_idx),
        "saved": int(saved),
        "recording": int(record_session is not None),
        "record_steps": steps,
        "go_home_running": int(receiver_mode == "go_home"),
        "go_home_result": go_home_result,
        "receiver_health_ok": int(health_ok),
        "receiver_health_error": health_error,
        "message": str(message),
    }
    payload.update(_policy_remote_status_payload(action_source, action_info))
    if saved_path is not None:
        payload["saved_path"] = str(saved_path)
    publish(payload)


def _policy_remote_status_payload(
    action_source: Any,
    action_info: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    extras = dict(getattr(action_info, "extras", {}) or {}) if action_info is not None else {}
    for key in (
        "policy_remote_mode",
        "policy_remote_activated",
        "policy_remote_deactivated",
        "policy_remote_activation_step",
        "policy_remote_toggle_count",
        "model_control",
    ):
        if key in extras:
            payload[key] = extras[key]
    status = getattr(action_source, "policy_status", None)
    if callable(status):
        try:
            payload.update(dict(status()))
        except Exception:
            log.debug("Failed to query policy_remote status.", exc_info=True)
    mode = str(payload.get("policy_remote_mode", "") or "")
    if mode:
        model_control = int(mode == "policy")
        payload["model_control"] = int(payload.get("model_control", model_control) or 0)
        payload["control_mode"] = "model" if int(payload["model_control"]) else "manual"
    return payload


def _hold_remote_control_zero(remote_control_loop: Any | None) -> None:
    if remote_control_loop is None:
        return
    remote_control_loop.set_scripted_action(np.zeros(4, dtype=np.float32))
    remote_control_loop.set_fault_hold(True)


def _release_remote_control(remote_control_loop: Any | None) -> None:
    if remote_control_loop is None:
        return
    remote_control_loop.set_scripted_action(None)
    remote_control_loop.set_fault_hold(False)


def _receiver_health_blocks_control(
    receiver_health: Any,
    *,
    receiver_mode: str,
    has_record_session: bool,
) -> bool:
    if bool(getattr(receiver_health, "ok", False)):
        return False
    errors = set(str(error) for error in getattr(receiver_health, "errors", ()) or ())
    if (
        errors
        and errors <= {"fpv_stale", "camera_stale"}
        and not has_record_session
        and receiver_mode in {"armed", "go_home"}
    ):
        return False
    return True


def _online_qc_blocks_record_start(online_qc: Any) -> bool:
    return str(getattr(online_qc, "status", "")) == "FAIL_EPISODE"


def _save_record_session_with_online_qc_final(
    session: Any,
    *,
    online_qc_evaluator: Any | None,
    error_time_ns: int | None = None,
    metadata_updates: dict[str, Any] | None = None,
) -> tuple[Path | None, bool, Any | None]:
    final_snapshot = None
    finalize = getattr(online_qc_evaluator, "finalize_episode", None)
    if callable(finalize):
        final_snapshot = finalize(recorded_steps=len(session))

    updates = dict(metadata_updates or {})
    if final_snapshot is not None:
        updates.update(_online_qc_final_metadata(final_snapshot))

    if _online_qc_blocks_record_start(final_snapshot):
        path = session.save_failed(
            error_code=str(getattr(final_snapshot, "error_code", "") or "online_qc_failed"),
            error_time_ns=int(error_time_ns if error_time_ns is not None else time.time_ns()),
            stop_reason="online_qc_failed",
            metadata_updates=updates,
        )
        return path, False, final_snapshot
    path = session.save_success(metadata_updates=updates or None)
    return path, True, final_snapshot


def _online_qc_final_metadata(final_snapshot: Any) -> dict[str, Any]:
    diagnostics = dict(getattr(final_snapshot, "diagnostics", {}) or {})
    metadata: dict[str, Any] = {
        "online_qc_final_status": str(getattr(final_snapshot, "status", "")),
        "online_qc_final_error_code": str(getattr(final_snapshot, "error_code", "") or ""),
        "online_qc_final_warning_codes": ",".join(
            str(code) for code in getattr(final_snapshot, "warning_codes", ()) or ()
        ),
    }
    for key in (
        "online_qc_reference_id",
        "online_qc_warning_codes",
        "online_qc_train_ready_candidate",
        "online_qc_total_steps",
        "online_qc_train_exclude_steps",
        "online_qc_healthy_steps",
        "online_qc_max_healthy_run",
        "online_qc_healthy_fraction",
        "online_qc_bucket_reference_status",
        "online_qc_bucket_semantic_decision",
        "online_qc_bucket_semantic_notes",
    ):
        if key in diagnostics:
            metadata[key] = diagnostics[key]
    if not metadata["online_qc_final_warning_codes"] and "online_qc_warning_codes" in diagnostics:
        metadata["online_qc_final_warning_codes"] = diagnostics["online_qc_warning_codes"]
    return metadata


def main(prog: str = "tb-record-real") -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run the real excavator receiver and optionally record v1 teleop "
            "data after the record-start gate."
        ),
    )
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=["mock", "noop", "bridge_mock", "bridge_tcp"],
        default=None,
    )
    parser.add_argument(
        "--state-reader",
        choices=["mock", "bridge_mock", "bridge_tcp"],
        default=None,
    )
    parser.add_argument("--num-episodes", "-n", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", "-o", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--input",
        choices=[
            "joystick",
            "keyboard",
            "oem_remote",
            "remote",
            "policy",
            "policy_remote",
            "zero",
        ],
        default=None,
    )
    parser.add_argument(
        "--policy-output-mode",
        choices=["shadow_zero", "control"],
        default=None,
        help=(
            "Override teleop.policy.output_mode for policy tests. shadow_zero logs "
            "predictions but sends zero; control sends the scaled policy action."
        ),
    )
    parser.add_argument(
        "--policy-action-scale",
        type=float,
        default=None,
        help="Override scalar teleop.policy.action_scale for policy control tests.",
    )
    parser.add_argument("--operator-id", type=str, default=None)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument(
        "--remote-port",
        type=int,
        default=None,
        help="Override teleop.remote.port for the receiver action server.",
    )
    parser.add_argument(
        "--bridge-host",
        type=str,
        default=None,
        help="Override real.bridge.host for bridge_tcp mode.",
    )
    parser.add_argument(
        "--bridge-port",
        type=int,
        default=None,
        help="Override real.bridge.port for bridge_tcp mode.",
    )
    parser.add_argument(
        "--bridge-timeout",
        type=float,
        default=None,
        help="Override real.bridge.timeout_s for bridge_tcp mode.",
    )
    parser.add_argument(
        "--data-side",
        choices=["host", "slave"],
        default=None,
        help=(
            "Where HDF5 is written. If unset, resolve from real.data_side, "
            "EXCAVATOR_DATA_SIDE, then default to slave=vehicle PC. "
            "slave: run on vehicle, data under /data/real_teleop_v1; "
            "host: run on operator PC, set EXCAVATOR_BRIDGE_HOST to vehicle IP."
        ),
    )
    parser.add_argument(
        "--live-action-line",
        action="store_true",
        help=(
            "Refresh one terminal line with the latest action sent to the bridge. "
            "Normal INFO logs and the pygame support prompt are suppressed."
        ),
    )
    parser.add_argument(
        "--qc-dashboard",
        action="store_true",
        help=(
            "Refresh a top-like terminal dashboard with receiver health and "
            "online training-usability QC fields."
        ),
    )
    parser.add_argument(
        "--qc-event-log",
        type=Path,
        default=None,
        help="Optional JSONL path for de-duplicated online QC WARN/FAIL events.",
    )
    parser.add_argument(
        "--wait-for-record-start",
        action="store_true",
        help=(
            "Receive and send controls immediately, but do not write episode "
            "steps until the action source reports record_start_requested."
        ),
    )
    record_mode = parser.add_mutually_exclusive_group()
    record_mode.add_argument(
        "--record",
        action="store_true",
        help="Enable HDF5 record sessions even if the config sets recording.enabled=false.",
    )
    record_mode.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Run the receiver/control loop without opening HDF5 record sessions. "
            "Use this for policy control tests where data should not be saved."
        ),
    )
    parser.add_argument(
        "--test-log-dir",
        type=Path,
        default=None,
        help=(
            "Directory for lightweight JSONL receiver test logs. This is separate "
            "from HDF5 dataset output and is useful with --no-record."
        ),
    )
    args = parser.parse_args()
    if args.live_action_line or args.qc_dashboard:
        logging.getLogger().setLevel(logging.ERROR)
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    cfg = _load_yaml_config(args.config)

    real_cfg = cfg.setdefault("real", {})
    teleop_cfg = cfg.setdefault("teleop", {})
    recording_cfg = teleop_cfg.setdefault("recording", {})
    task_cfg = cfg.setdefault("task", {})
    safety_cfg = cfg.setdefault("safety", {})
    sync_cfg = cfg.setdefault("sync", {})
    video_cfg = cfg.setdefault("video", {})
    receiver_cfg = cfg.setdefault("receiver", {})
    receiver_health_cfg = receiver_cfg.setdefault("health", {})
    receiver_online_qc_cfg = receiver_cfg.setdefault("online_qc", {})
    teleop_meta_cfg = teleop_cfg.setdefault("metadata", {})

    from testbed.cli.data_side import (
        apply_data_side_config,
        validate_data_side_for_bridge_tcp,
    )

    resolved_data_side = apply_data_side_config(
        cfg,
        data_side=args.data_side,
        cli_output_dir=str(args.output_dir) if args.output_dir is not None else None,
        cli_bridge_host=args.bridge_host,
        cli_bridge_port=args.bridge_port,
    )

    if args.backend is not None:
        real_cfg["backend"] = args.backend
    if args.state_reader is not None:
        real_cfg["state_reader"] = args.state_reader
    if args.num_episodes is not None:
        teleop_cfg["num_episodes"] = int(args.num_episodes)
    if args.max_steps is not None:
        task_cfg["max_steps"] = int(args.max_steps)
    if args.output_dir is not None:
        task_cfg["dataset_dir"] = str(args.output_dir)
    if args.seed is not None:
        task_cfg["seed"] = int(args.seed)
    if args.input is not None:
        teleop_cfg["input"] = args.input
    if args.policy_output_mode is not None:
        policy_cfg = teleop_cfg.setdefault("policy", {})
        policy_cfg["output_mode"] = args.policy_output_mode
    if args.policy_action_scale is not None:
        policy_cfg = teleop_cfg.setdefault("policy", {})
        policy_cfg["action_scale"] = float(args.policy_action_scale)
    if args.remote_port is not None:
        remote_cfg = teleop_cfg.setdefault("remote", {})
        remote_cfg["port"] = int(args.remote_port)
    if args.wait_for_record_start:
        recording_cfg["wait_for_start"] = True
    if args.record:
        recording_cfg["enabled"] = True
    elif args.no_record:
        recording_cfg["enabled"] = False
    if args.test_log_dir is not None:
        test_log_cfg = teleop_cfg.setdefault("test_log", {})
        test_log_cfg["enabled"] = True
        test_log_cfg["output_dir"] = str(args.test_log_dir)
    if args.operator_id is not None:
        teleop_meta_cfg["operator_id"] = args.operator_id
    if args.session_id is not None:
        teleop_meta_cfg["session_id"] = args.session_id
    if args.notes is not None:
        teleop_meta_cfg["notes"] = args.notes
    if (
        args.bridge_host is not None
        or args.bridge_port is not None
        or args.bridge_timeout is not None
    ):
        bridge_cfg = real_cfg.setdefault("bridge", {})
        if args.bridge_host is not None:
            bridge_cfg["host"] = args.bridge_host
        if args.bridge_port is not None:
            bridge_cfg["port"] = int(args.bridge_port)
        if args.bridge_timeout is not None:
            bridge_cfg["timeout_s"] = float(args.bridge_timeout)

    num_episodes = int(teleop_cfg.get("num_episodes", 1))
    dataset_dir = Path(task_cfg.get("dataset_dir", "data/real_teleop_v1"))
    seed = int(task_cfg.get("seed", -1))
    max_steps = int(task_cfg.get("max_steps", 1000))
    control_hz = float(task_cfg.get("control_hz", real_cfg.get("control_hz", 50)))
    record_hz = float(task_cfg.get("record_hz", control_hz))
    dt = float(task_cfg.get("dt", 1.0 / record_hz))
    input_device = str(teleop_cfg.get("input", "joystick"))
    recording_enabled = bool(recording_cfg.get("enabled", True))
    wait_for_record_start = bool(recording_cfg.get("wait_for_start", False))
    test_log_cfg = dict(teleop_cfg.get("test_log", {}) or {})
    test_log_enabled = bool(test_log_cfg.get("enabled", not recording_enabled))
    backend_mode = str(real_cfg.get("backend", "mock"))
    state_reader_mode = str(real_cfg.get("state_reader", "mock"))
    health_evaluator = ReceiverHealthEvaluator.from_config(
        receiver_health_cfg,
        input_device=input_device,
        state_reader_mode=state_reader_mode,
    )
    bridge_cfg = dict(real_cfg.get("bridge", {}) or {})
    validate_data_side_for_bridge_tcp(
        resolved_data_side,
        backend_mode=backend_mode,
        state_reader_mode=state_reader_mode,
        bridge_host=str(bridge_cfg.get("host", "127.0.0.1")),
    )
    sync_max_slop_ns = int(float(sync_cfg.get("max_observation_skew_ms", 40.0)) * 1_000_000)
    camera_names: list[str] = list(task_cfg.get("camera_names", ["fpv"]))
    _default_online_qc_primary_camera(
        receiver_online_qc_cfg,
        receiver_health_cfg=receiver_health_cfg,
        camera_names=camera_names,
    )
    record_config_yaml = _dump_yaml_config(cfg)

    from testbed.backends.real.backend import RealExcavatorBackend
    from testbed.backends.real.go_home import GoHomeConfig, GoHomeController
    from testbed.data.online_qc import OnlineQcConfig, OnlineTrainingQcEvaluator
    from testbed.data.recorder import EpisodeRecorder
    from testbed.runtime.guard import ActionGuard

    online_qc_config = OnlineQcConfig.from_mapping(receiver_online_qc_cfg)

    pump_cfg = dict(real_cfg.get("control_pump", {}) or {})
    control_pump_enabled = bool(pump_cfg.get("enabled", False)) and backend_mode == "bridge_tcp"
    backend_controller_mode = "noop" if control_pump_enabled else backend_mode
    bridge_client = _build_bridge_client(real_cfg, backend_controller_mode, state_reader_mode)
    control_pump = None
    if control_pump_enabled:
        from testbed.backends.real.action_pump import RealActionPump
        from testbed.backends.real.bridge import BridgeLowLevelController

        control_pump_client = _build_control_pump_client(real_cfg, pump_cfg)
        if control_pump_client is None:
            raise RuntimeError("control_pump requires a bridge_tcp control client")
        control_pump = RealActionPump(
            BridgeLowLevelController(control_pump_client),
            hz=float(pump_cfg.get("hz", control_hz)),
            send_immediately_on_update=bool(
                pump_cfg.get("send_immediately_on_update", True)
            ),
            zero_on_stop=bool(pump_cfg.get("zero_on_stop", True)),
        )
        log.info(
            "Control action pump enabled at %.1f Hz; recorder loop %.1f Hz.",
            control_pump.hz,
            record_hz,
        )

    backend = RealExcavatorBackend(
        controller_mode=backend_controller_mode,
        state_reader_mode=state_reader_mode,
        bridge_client=bridge_client,
        sync_max_slop_ns=sync_max_slop_ns,
        control_hz=control_hz,
        image_width=int(real_cfg.get("image_width", 160)),
        image_height=int(real_cfg.get("image_height", 120)),
        camera_names=camera_names,
        mock_velocity_scale_rad_s=float(real_cfg.get("mock_velocity_scale_rad_s", 0.5)),
    )
    action_source = _build_action_source(input_device, teleop_cfg, dt=dt)
    guard = ActionGuard(
        action_clip=safety_cfg.get("action_clip", 0.20),
        max_delta=safety_cfg.get("max_delta_per_step", 0.02),
        sensor_timeout_s=safety_cfg.get("sensor_timeout_s", 0.20),
    )
    go_home_config = GoHomeConfig.from_mapping(recording_cfg.get("go_home", {}))
    base_meta = _build_episode_metadata(
        task_cfg=task_cfg,
        teleop_cfg=teleop_cfg,
        real_cfg=real_cfg,
        safety_cfg=safety_cfg,
        input_device=input_device,
        camera_names=camera_names,
        config_path=args.config.resolve(),
        record_config_yaml=record_config_yaml,
        control_hz=control_hz,
        record_hz=record_hz,
        dt=dt,
        control_pump_enabled=control_pump_enabled,
        sync_cfg=sync_cfg,
        video_cfg=video_cfg,
        data_side=resolved_data_side or real_cfg.get("data_side"),
    )

    log.info(
        "Real v1 config: data_side=%s backend=%s state_reader=%s input=%s "
        "episodes=%d max_steps=%d output=%s",
        resolved_data_side or real_cfg.get("data_side", "-"),
        backend_mode,
        state_reader_mode,
        input_device,
        num_episodes,
        max_steps,
        dataset_dir,
    )
    test_logger = None
    if test_log_enabled:
        test_logger = ReceiverTestLogger.from_config(
            test_log_cfg,
            metadata=base_meta,
            record_config_yaml=record_config_yaml,
        )
        log.info("Receiver test log enabled: %s", test_logger.run_dir)

    abort = False
    sigint_count = 0

    def _sigint(_sig, _frame) -> None:
        nonlocal abort, sigint_count
        abort = True
        sigint_count += 1
        if sigint_count >= 2:
            log.warning("再次 Ctrl+C：强制退出。")
            raise SystemExit(130)
        log.warning(
            "Ctrl+C：本步结束后退出并保存当前片段（再按一次立即退出，不保存）。"
        )
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _sigint)

    if recording_enabled:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        episode_idx = _next_episode_idx(dataset_dir)
    else:
        episode_idx = 0
    saved = 0
    control_output_stopped = False

    def _stop_control_output() -> None:
        nonlocal control_output_stopped
        if control_output_stopped:
            return
        control_output_stopped = True
        if control_pump is not None:
            try:
                log.info("Stopping control pump and sending zero command before save.")
                control_pump.stop()
            except Exception:
                log.exception("Failed to stop control pump cleanly.")
            return
        try:
            backend.step(np.zeros(4, dtype=np.float32))
        except Exception:
            log.exception("Failed to send zero command during shutdown.")

    def _force_zero_control(obs: dict[str, Any] | None = None) -> None:
        zero = np.zeros(4, dtype=np.float32)
        if control_pump is not None:
            try:
                control_pump.update_action(zero, state=obs)
            except Exception:
                log.exception("Failed to force zero command through control pump.")
            return
        try:
            backend.step(zero)
        except Exception:
            log.exception("Failed to send zero command.")

    def _save_interrupt_partial(
        session: RecordSession | None,
        *,
        discard: bool,
        error_time_ns: int,
    ) -> Path | None:
        if session is None or discard or len(session) == 0:
            return None
        try:
            path = session.save_failed(
                error_code="interrupted",
                error_time_ns=error_time_ns,
                stop_reason="interrupted",
            )
        except KeyboardInterrupt:
            log.warning("保存 HDF5 被中断，已跳过写入。")
            return None
        log.info("Saved interrupted record: %d steps -> %s", len(session), path)
        return path

    def _shutdown() -> None:
        try:
            action_source.close()
        except Exception:
            pass
        _stop_control_output()
        if bridge_client is not None and hasattr(bridge_client, "force_close"):
            bridge_client.force_close()
            return
        try:
            backend.close()
        except Exception:
            pass

    if control_pump is not None:
        control_pump.start()

    qc_event_logger = (
        _QcEventLogger(args.qc_event_log) if args.qc_event_log is not None else None
    )
    qc_dashboard = _QcDashboard(
        enabled=bool(args.qc_dashboard),
        event_logger=qc_event_logger,
    )
    live_line = _LiveActionLine(
        enabled=bool(args.live_action_line) and not bool(args.qc_dashboard)
    )
    remote_control_loop_enabled = control_pump is not None and input_device == "remote"
    if remote_control_loop_enabled:
        log.info(
            "Remote action control loop enabled at %.1f Hz, independent of recorder sampling.",
            float(control_hz),
        )

    failed_dir = dataset_dir / "failed"

    def _episode_metadata(idx: int) -> tuple[int, dict[str, Any]]:
        ep_seed = seed if seed >= 0 else int(time.time()) % (2**31)
        meta = dict(base_meta)
        meta[ATTR_SEED] = ep_seed
        meta[ATTR_EPISODE_ID] = f"episode_{idx}"
        return ep_seed, meta

    try:
        while saved < num_episodes and not abort:
            ep_seed, _ = _episode_metadata(episode_idx)
            log.info(
                "Receiver starts episode window %d; no hardware reset is issued.",
                episode_idx,
            )
            if abort:
                break
            try:
                ts = backend.start_episode(seed=ep_seed)
            except KeyboardInterrupt:
                break
            action_source.reset()
            guard.reset()
            online_qc_evaluator = (
                OnlineTrainingQcEvaluator(online_qc_config)
                if online_qc_config.enabled
                else None
            )
            discard = False
            receiver_mode = "armed"
            record_start_pending = recording_enabled and not wait_for_record_start
            record_session: RecordSession | None = None
            go_home_controller: GoHomeController | None = None
            if not recording_enabled:
                log.info(
                    "Receiver is in no-record test mode for episode window %d.",
                    episode_idx,
                )
            elif wait_for_record_start:
                log.info(
                    "Receiver armed for episode %d; waiting for record_start_requested.",
                    episode_idx,
                )
            else:
                log.info("Receiver starts recording episode %d immediately.", episode_idx)
            remote_control_loop = None
            if remote_control_loop_enabled:
                remote_control_loop = _RemoteControlLoop(
                    action_source=action_source,
                    guard=guard,
                    control_pump=control_pump,
                    rate_hz=control_hz,
                    initial_obs=ts.observation,
                )
                remote_control_loop.start()

            try:
                local_step = 0
                while saved < num_episodes:
                    if abort:
                        break

                    obs = ts.observation
                    if (
                        record_start_pending
                        and receiver_mode == "armed"
                        and record_session is None
                    ):
                        _, meta = _episode_metadata(episode_idx)
                        record_session = RecordSession(
                            recorder_cls=EpisodeRecorder,
                            dataset_dir=dataset_dir,
                            failed_dir=failed_dir,
                            episode_idx=episode_idx,
                            metadata=meta,
                            camera_names=camera_names,
                        )
                        receiver_mode = "recording"
                        record_start_pending = False
                        log.info("Record session episode %d starts on this frame.", episode_idx)
                        live_line.message(f"mode=recording episode={episode_idx}")

                    discard, quit_now = _check_pygame_events(
                        enabled=input_device not in {"zero", "remote"}
                    )
                    if quit_now:
                        abort = True
                        break
                    if discard:
                        log.info("Episode discarded by user.")
                        break

                    record_start_requested = False
                    go_home_requested = False
                    go_home_start_accepted = False
                    go_home_start_rejected = False
                    go_home_start_reject_reason = ""
                    go_home_update = None
                    if receiver_mode == "go_home" and go_home_controller is not None:
                        go_home_update = go_home_controller.update(obs)
                        if remote_control_loop is not None:
                            remote_control_loop.set_scripted_action(go_home_update.action)
                    if remote_control_loop is not None:
                        remote_control_loop.update_observation(obs)
                        (
                            record_start_now,
                            go_home_now,
                            reset_now,
                            discard_now,
                            quit_now,
                        ) = remote_control_loop.consume_requests()
                        if quit_now:
                            log.info(
                                "Remote sender quit requested; receiver remains alive and waits for reconnect."
                            )
                            live_line.message("remote_sender_quit wait_reconnect=1")
                        if reset_now or discard_now:
                            discard = True
                            log.info("Episode discarded by action-source request.")
                            break
                        record_start_requested = bool(record_start_now)
                        go_home_requested = bool(go_home_now)

                        sample = remote_control_loop.latest_sample()
                        raw_action = sample.raw_action
                        safe_action = sample.safe_action
                        action_info = sample.action_info
                        action_sample_ns = sample.action_sample_timestamp_ns
                        action_send_ns = sample.action_send_timestamp_ns
                        guard_info = sample.guard_info
                        sensor_age_s = sample.sensor_age_s
                        pump_result = (
                            control_pump.latest_result
                            if control_pump is not None
                            else sample.control_result
                        )
                        action_send_ns = int(
                            pump_result.controller_timestamp_ns or action_send_ns
                        )
                        ts_next = backend.observe(
                            action_timestamp_ns=action_send_ns,
                            result=pump_result,
                        )
                    else:
                        raw_action, action_info = action_source.next_action(obs)
                        reset_now, discard_now, quit_now = _action_control_flags(action_info)
                        if quit_now:
                            abort = True
                            break
                        if reset_now or discard_now:
                            discard = True
                            log.info("Episode discarded by action-source request.")
                            break

                        action_sample_ns = _action_sample_timestamp_ns(action_info)
                        extras = getattr(action_info, "extras", {}) or {}
                        record_start_requested = bool(
                            extras.get("record_start_requested", False)
                        )
                        go_home_requested = bool(
                            extras.get("go_home_requested", False)
                        )
                        safety_state = dict(obs.get("safety_state", {}))
                        host_now_ns = time.time_ns()
                        sensor_age_s = _sensor_age_s(obs, now_ns=host_now_ns)
                        safe_action, _triggered = guard.check(
                            raw_action,
                            obs.get("qpos"),
                            deadman_pressed=bool(safety_state.get("deadman_pressed", True)),
                            estop_active=bool(safety_state.get("estop_active", False)),
                            manual_override_active=bool(
                                safety_state.get("manual_override_active", False)
                            ),
                            sensor_stale=bool(safety_state.get("sensor_stale", False)),
                            sensor_age_s=sensor_age_s,
                        )
                        guard_info = _GuardInfoSnapshot(
                            triggered=bool(guard.last_info.triggered),
                            reasons=tuple(str(reason) for reason in guard.last_info.reasons),
                        )
                        if go_home_update is not None:
                            safe_action, _triggered = guard.check(
                                go_home_update.action,
                                obs.get("qpos"),
                                deadman_pressed=bool(
                                    safety_state.get("deadman_pressed", True)
                                ),
                                estop_active=bool(
                                    safety_state.get("estop_active", False)
                                ),
                                manual_override_active=bool(
                                    safety_state.get("manual_override_active", False)
                                ),
                                sensor_stale=bool(
                                    safety_state.get("sensor_stale", False)
                                ),
                                sensor_age_s=sensor_age_s,
                            )
                            guard_info = _GuardInfoSnapshot(
                                triggered=bool(guard.last_info.triggered),
                                reasons=tuple(
                                    str(reason) for reason in guard.last_info.reasons
                                ),
                            )
                        if receiver_mode == "fault":
                            safe_action = np.zeros(4, dtype=np.float32)

                        toggle_mask = int(extras.get("toggle_mask", 0) or 0)
                        if toggle_mask and receiver_mode != "go_home":
                            if control_pump is not None:
                                control_pump.apply_status_toggle_mask(toggle_mask)
                            else:
                                backend.apply_status_toggle_mask(toggle_mask)

                        action_send_ns = time.time_ns()
                        if control_pump is not None:
                            control_pump.update_action(safe_action, state=obs)
                            pump_result = control_pump.latest_result
                            action_send_ns = int(
                                pump_result.controller_timestamp_ns or action_send_ns
                            )
                            ts_next = backend.observe(
                                action_timestamp_ns=action_send_ns,
                                result=pump_result,
                            )
                        else:
                            ts_next = backend.step(safe_action)
                    control_result = dict(ts_next.info.get("control_result", {}))
                    receiver_health = health_evaluator.evaluate(
                        obs=obs,
                        action_info=action_info,
                        control_result=control_result,
                    )
                    online_qc_snapshot = (
                        online_qc_evaluator.evaluate(
                            obs=obs,
                            now_ns=time.time_ns(),
                            semantic_sample=receiver_mode != "go_home",
                        )
                        if online_qc_evaluator is not None
                        else None
                    )
                    health_blocks_control = _receiver_health_blocks_control(
                        receiver_health,
                        receiver_mode=receiver_mode,
                        has_record_session=record_session is not None,
                    )
                    if receiver_mode == "fault" and receiver_health.ok:
                        receiver_mode = "armed"
                        _release_remote_control(remote_control_loop)
                        log.info("Receiver health recovered; mode=armed.")
                        live_line.message("mode=armed health=OK")

                    if recording_enabled and wait_for_record_start and record_start_requested:
                        if receiver_mode == "armed" and record_session is None:
                            if receiver_health.ok and not _online_qc_blocks_record_start(
                                online_qc_snapshot
                            ):
                                record_start_pending = True
                                log.info(
                                    "Record start accepted for episode %d; first HDF5 step is the next frame.",
                                    episode_idx,
                                )
                                live_line.message(
                                    f"record_start_accepted episode={episode_idx} next_frame=1"
                                )
                            elif _online_qc_blocks_record_start(online_qc_snapshot):
                                error_code = str(
                                    getattr(online_qc_snapshot, "error_code", "")
                                    or "online_qc_failed"
                                )
                                log.warning(
                                    "Record start blocked by online QC: %s",
                                    error_code,
                                )
                                live_line.message(
                                    "record_start_blocked "
                                    f"err={error_code}"
                                )
                            else:
                                log.warning(
                                    "Record start blocked by receiver health: %s",
                                    receiver_health.error_code,
                                )
                                live_line.message(
                                    "record_start_blocked "
                                    f"err={receiver_health.error_code}"
                                )

                    if not receiver_health.ok and health_blocks_control:
                        error_time_ns = time.time_ns()
                        if remote_control_loop is not None:
                            remote_control_loop.set_fault_hold(True)
                        if receiver_mode in {"recording", "go_home"} and record_session is not None:
                            _force_zero_control(obs)
                            stop_reason = (
                                "go_home_sensor_error"
                                if receiver_mode == "go_home"
                                else "sensor_error"
                            )
                            failed_path = record_session.save_failed(
                                error_code=receiver_health.error_code,
                                error_time_ns=error_time_ns,
                                stop_reason=stop_reason,
                            )
                            if failed_path is not None:
                                log.error(
                                    "Record episode %d stopped by receiver health %s; saved failed record to %s",
                                    episode_idx,
                                    receiver_health.error_code,
                                    failed_path,
                                )
                                live_line.message(
                                    "mode=fault "
                                    f"err={receiver_health.error_code} "
                                    f"failed={failed_path}"
                                )
                                episode_idx += 1
                            else:
                                log.error(
                                    "Record episode %d stopped before any step by receiver health %s.",
                                    episode_idx,
                                    receiver_health.error_code,
                                )
                                live_line.message(
                                    f"mode=fault err={receiver_health.error_code}"
                            )
                            record_session = None
                            go_home_controller = None
                            if remote_control_loop is not None:
                                remote_control_loop.set_scripted_action(None)
                        elif receiver_mode != "fault":
                            _force_zero_control(obs)
                            log.warning(
                                "Receiver health fault while armed: %s",
                                receiver_health.error_code,
                            )
                            live_line.message(
                                f"mode=fault err={receiver_health.error_code}"
                            )
                        receiver_mode = "fault"
                        record_start_pending = False

                    if (
                        online_qc_snapshot is not None
                        and _online_qc_blocks_record_start(online_qc_snapshot)
                        and receiver_mode in {"recording", "go_home"}
                        and record_session is not None
                    ):
                        error_time_ns = time.time_ns()
                        _force_zero_control(obs)
                        failed_path = record_session.save_failed(
                            error_code=str(
                                online_qc_snapshot.error_code or "online_qc_failed"
                            ),
                            error_time_ns=error_time_ns,
                            stop_reason="online_qc_failed",
                        )
                        if failed_path is not None:
                            log.error(
                                "Record episode %d stopped by online QC %s; saved failed record to %s",
                                episode_idx,
                                online_qc_snapshot.error_code,
                                failed_path,
                            )
                            live_line.message(
                                "mode=armed "
                                f"err={online_qc_snapshot.error_code} "
                                f"failed={failed_path}"
                            )
                            episode_idx += 1
                        else:
                            log.error(
                                "Record episode %d stopped before any step by online QC %s.",
                                episode_idx,
                                online_qc_snapshot.error_code,
                            )
                            live_line.message(
                                f"mode=armed err={online_qc_snapshot.error_code}"
                            )
                        record_session = None
                        go_home_controller = None
                        receiver_mode = "armed"
                        record_start_pending = False
                        guard.reset()

                    if (receiver_health.ok or not health_blocks_control) and go_home_requested:
                        go_home_context = _go_home_start_context(
                            receiver_mode,
                            record_session is not None,
                        )
                        if go_home_context is not None:
                            if go_home_config is None:
                                go_home_start_rejected = True
                                go_home_start_reject_reason = "config_missing"
                                log.warning(
                                    "go-home requested but teleop.recording.go_home.enabled/home_pose_rad is not configured."
                                )
                                live_line.message("go_home_blocked config_missing")
                            else:
                                controller = GoHomeController(go_home_config)
                                try:
                                    controller.start(ts_next.observation)
                                except ValueError as exc:
                                    go_home_start_rejected = True
                                    go_home_start_reject_reason = str(exc)
                                    log.warning("go-home start rejected: %s", exc)
                                    live_line.message(f"go_home_blocked {exc}")
                                else:
                                    go_home_start_accepted = True
                                    go_home_controller = controller
                                    receiver_mode = "go_home"
                                    guard.reset()
                                    if remote_control_loop is not None:
                                        remote_control_loop.set_scripted_action(
                                            np.zeros(4, dtype=np.float32)
                                        )
                                    log.info(
                                        "go-home started in %s mode for episode %d.",
                                        go_home_context,
                                        episode_idx,
                                    )
                                    record_label = (
                                        f" episode={episode_idx}"
                                        if go_home_context == "recording"
                                        else ""
                                    )
                                    live_line.message(
                                        f"mode=go_home context={go_home_context}{record_label}"
                                    )
                        elif receiver_mode != "go_home":
                            go_home_start_rejected = True
                            go_home_start_reject_reason = f"ignored_mode:{receiver_mode}"
                            log.info(
                                "go-home request ignored in receiver mode %s.",
                                receiver_mode,
                            )
                    live_line.update(
                        step=local_step,
                        mode=receiver_mode,
                        raw_action=raw_action,
                        safe_action=safe_action,
                        action_info=action_info,
                        sensor_age_s=sensor_age_s,
                        control_result=control_result,
                        guard_reasons=guard_info.reasons,
                        receiver_health=receiver_health,
                        go_home_update=go_home_update,
                    )
                    qc_dashboard.update(
                        mode=receiver_mode,
                        episode_idx=episode_idx,
                        saved=saved,
                        record_steps=(
                            len(record_session) if record_session is not None else 0
                        ),
                        receiver_health=receiver_health,
                        online_qc=online_qc_snapshot,
                        action_info=action_info,
                    )
                    _publish_remote_receiver_status(
                        action_source,
                        receiver_mode=receiver_mode,
                        episode_idx=episode_idx,
                        saved=saved,
                        record_session=record_session,
                        go_home_update=go_home_update,
                        receiver_health=receiver_health,
                        action_info=action_info,
                    )
                    if test_logger is not None:
                        test_logger.record_step(
                            local_step=local_step,
                            receiver_mode=receiver_mode,
                            obs=obs,
                            raw_action=raw_action,
                            safe_action=safe_action,
                            action_info=action_info,
                            action_sample_timestamp_ns=action_sample_ns,
                            action_send_timestamp_ns=action_send_ns,
                            guard=guard_info,
                            control_result=control_result,
                            receiver_health=receiver_health,
                            record_start_requested=record_start_requested,
                            go_home_requested=go_home_requested,
                            go_home_update=go_home_update,
                        )
                    if (
                        receiver_mode == "go_home"
                        and record_session is None
                        and go_home_update is not None
                    ):
                        if go_home_update.failed:
                            _force_zero_control(obs)
                            _release_remote_control(remote_control_loop)
                            log.warning(
                                "go-home failed while armed: %s; no record is saved.",
                                go_home_update.reason,
                            )
                            live_line.message(
                                "mode=armed go_home_failed "
                                f"err={go_home_update.reason or 'unknown'} "
                                f"final_err={_format_go_home_final_error(go_home_controller)}"
                            )
                            go_home_controller = None
                            receiver_mode = "armed"
                            guard.reset()
                            _publish_remote_receiver_status(
                                action_source,
                                receiver_mode=receiver_mode,
                                episode_idx=episode_idx,
                                saved=saved,
                                record_session=record_session,
                                go_home_update=go_home_update,
                                receiver_health=receiver_health,
                                action_info=action_info,
                                message="go_home_failed_armed",
                            )
                        elif go_home_update.done:
                            _force_zero_control(obs)
                            _release_remote_control(remote_control_loop)
                            log.info("go-home completed while armed; no record is saved.")
                            live_line.message(
                                "mode=armed go_home_done "
                                f"final_err={_format_go_home_final_error(go_home_controller)}"
                            )
                            go_home_controller = None
                            receiver_mode = "armed"
                            guard.reset()
                            _publish_remote_receiver_status(
                                action_source,
                                receiver_mode=receiver_mode,
                                episode_idx=episode_idx,
                                saved=saved,
                                record_session=record_session,
                                go_home_update=go_home_update,
                                receiver_health=receiver_health,
                                action_info=action_info,
                                message="go_home_done_armed",
                            )
                    if receiver_mode in {"recording", "go_home"} and record_session is not None:
                        step_diagnostics = _build_step_diagnostics(
                            obs=obs,
                            raw_action=raw_action,
                            safe_action=safe_action,
                            action_info=action_info,
                            action_sample_timestamp_ns=action_sample_ns,
                            action_send_timestamp_ns=action_send_ns,
                            guard=guard_info,
                            control_result=control_result,
                            receiver_health=receiver_health,
                            online_qc=online_qc_snapshot,
                        )
                        _ensure_go_home_step_diagnostics(step_diagnostics)
                        if go_home_update is not None:
                            step_diagnostics.update(go_home_update.diagnostics)
                        step_diagnostics["go_home_start_accepted"] = int(
                            go_home_start_accepted
                        )
                        step_diagnostics["go_home_start_rejected"] = int(
                            go_home_start_rejected
                        )
                        step_diagnostics["go_home_start_reject_reason"] = (
                            go_home_start_reject_reason
                        )
                        record_session.record_step(
                            obs=obs,
                            action=safe_action,
                            reward=0.0,
                            step_id=int(obs.get("step_id", local_step)),
                            step_ns=_record_step_timestamp_ns(
                                obs, fallback=action_send_ns
                            ),
                            action_src_type=action_info.source_type,
                            action_src_id=action_info.source_id,
                            diagnostics=step_diagnostics,
                        )
                        if (
                            online_qc_snapshot is not None
                            and bool(getattr(online_qc_snapshot, "train_exclude", False))
                        ):
                            record_session.mark_recent_train_exclude(
                                window_steps=online_qc_config.mask_backfill_window_steps
                            )
                        if go_home_update is not None and go_home_update.failed:
                            _hold_remote_control_zero(remote_control_loop)
                            _force_zero_control(obs)
                            failed_steps = len(record_session)
                            failed_path = record_session.save_failed(
                                error_code=go_home_update.reason or "go_home_failed",
                                error_time_ns=time.time_ns(),
                                stop_reason="go_home_failed",
                                metadata_updates=(
                                    go_home_controller.metadata()
                                    if go_home_controller is not None
                                    else None
                                ),
                            )
                            if failed_path is not None:
                                log.error(
                                    "go-home failed for episode %d: %s; saved failed record to %s",
                                    episode_idx,
                                    go_home_update.reason,
                                    failed_path,
                                )
                                live_line.message(
                                    "mode=armed go_home_failed "
                                    f"err={go_home_update.reason or 'unknown'} "
                                    f"final_err={_format_go_home_final_error(go_home_controller)} "
                                    f"failed={failed_path} ready_next=1"
                                )
                            record_session = None
                            go_home_controller = None
                            episode_idx += 1
                            guard.reset()
                            _release_remote_control(remote_control_loop)
                            receiver_mode = "armed"
                            record_start_pending = False
                            _publish_remote_receiver_status(
                                action_source,
                                receiver_mode=receiver_mode,
                                episode_idx=episode_idx,
                                saved=saved,
                                record_session=record_session,
                                go_home_update=go_home_update,
                                receiver_health=receiver_health,
                                record_steps=failed_steps,
                                saved_path=failed_path,
                                action_info=action_info,
                                message="go_home_failed_record_saved",
                            )
                            ts = ts_next
                            local_step += 1
                            continue
                        if go_home_update is not None and go_home_update.done:
                            _hold_remote_control_zero(remote_control_loop)
                            _force_zero_control(obs)
                            receiver_mode = "saving"
                            saved_steps = len(record_session)
                            saved_path, saved_success, final_qc_snapshot = (
                                _save_record_session_with_online_qc_final(
                                    record_session,
                                    online_qc_evaluator=online_qc_evaluator,
                                    metadata_updates=(
                                        go_home_controller.metadata()
                                        if go_home_controller is not None
                                        else None
                                    ),
                                )
                            )
                            if saved_success:
                                log.info(
                                    "Saved go-home completed record: %d steps -> %s",
                                    saved_steps,
                                    saved_path,
                                )
                                live_line.message(
                                    "go_home_done "
                                    f"final_err={_format_go_home_final_error(go_home_controller)} "
                                    f"saved steps={saved_steps} path={saved_path}"
                                )
                                saved += 1
                            else:
                                error_code = str(
                                    getattr(final_qc_snapshot, "error_code", "")
                                    or "online_qc_failed"
                                )
                                log.error(
                                    "go-home completed episode %d failed final online QC %s; saved failed record to %s",
                                    episode_idx,
                                    error_code,
                                    saved_path,
                                )
                                live_line.message(
                                    "go_home_done online_qc_failed "
                                    f"err={error_code} failed={saved_path}"
                                )
                            record_session = None
                            go_home_controller = None
                            episode_idx += 1
                            guard.reset()
                            _release_remote_control(remote_control_loop)
                            receiver_mode = (
                                "complete" if saved >= num_episodes else "armed"
                            )
                            record_start_pending = False
                            if receiver_mode == "armed":
                                if saved_success:
                                    live_line.message(
                                        f"mode=armed saved_episode={episode_idx - 1} ready_next=1"
                                    )
                                else:
                                    live_line.message(
                                        f"mode=armed failed_episode={episode_idx - 1} ready_next=1"
                                    )
                            _publish_remote_receiver_status(
                                action_source,
                                receiver_mode=receiver_mode,
                                episode_idx=episode_idx,
                                saved=saved,
                                record_session=record_session,
                                go_home_update=go_home_update,
                                receiver_health=receiver_health,
                                record_steps=saved_steps,
                                saved_path=saved_path,
                                action_info=action_info,
                                message=(
                                    "go_home_done_record_saved"
                                    if saved_success
                                    else "go_home_done_online_qc_failed"
                                ),
                            )
                            if saved >= num_episodes:
                                break
                            ts = ts_next
                            local_step += 1
                            continue
                        if len(record_session) >= max_steps:
                            receiver_mode = "saving"
                            saved_steps = len(record_session)
                            saved_path, saved_success, final_qc_snapshot = (
                                _save_record_session_with_online_qc_final(
                                    record_session,
                                    online_qc_evaluator=online_qc_evaluator,
                                )
                            )
                            if saved_success:
                                log.info(
                                    "Saved real v1 record: %d steps -> %s",
                                    saved_steps,
                                    saved_path,
                                )
                                live_line.message(
                                    f"saved steps={saved_steps} path={saved_path}"
                                )
                                saved += 1
                            else:
                                error_code = str(
                                    getattr(final_qc_snapshot, "error_code", "")
                                    or "online_qc_failed"
                                )
                                log.error(
                                    "Record episode %d failed final online QC %s; saved failed record to %s",
                                    episode_idx,
                                    error_code,
                                    saved_path,
                                )
                                live_line.message(
                                    "online_qc_failed "
                                    f"err={error_code} failed={saved_path}"
                                )
                            record_session = None
                            episode_idx += 1
                            break
                    if not recording_enabled and max_steps > 0 and local_step + 1 >= max_steps:
                        log.info("No-record receiver test reached max_steps=%d.", max_steps)
                        saved = num_episodes
                        break
                    ts = ts_next
                    local_step += 1
                    _sleep_to_rate(record_hz, should_stop=lambda: abort)
            except KeyboardInterrupt:
                abort = True
            finally:
                if remote_control_loop is not None:
                    remote_control_loop.stop()

            if abort or saved >= num_episodes:
                _stop_control_output()
            interrupt_path = _save_interrupt_partial(
                record_session,
                discard=discard,
                error_time_ns=time.time_ns(),
            )
            if interrupt_path is not None:
                live_line.message(f"failed steps={len(record_session)} path={interrupt_path}")
                episode_idx += 1
            elif discard:
                log.info("Discarded current partial episode; episode index is unchanged.")
    except KeyboardInterrupt:
        abort = True
    finally:
        _shutdown()
        if test_logger is not None:
            test_logger.close(stop_reason="aborted" if abort else "complete")
        live_line.finish()
        qc_dashboard.finish()
        if qc_event_logger is not None:
            qc_event_logger.close()

    if recording_enabled:
        log.info("Real v1 recording complete: %d / %d episode(s) saved.", saved, num_episodes)
    else:
        log.info("Real v1 no-record receiver test complete.")
    if abort and sigint_count >= 2:
        sys.exit(130)


def receiver_main() -> None:
    main(prog="tb-receiver-real")


def _build_action_source(input_device: str, teleop_cfg: dict[str, Any], *, dt: float):
    if input_device == "joystick":
        from testbed.actions.gamepad import JoystickActionSource

        return JoystickActionSource.from_config(
            teleop_cfg.get("joystick", {}),
            default_dt=dt,
        )
    if input_device == "keyboard":
        from testbed.actions.keyboard import KeyboardActionSource

        return KeyboardActionSource.from_config(teleop_cfg.get("keyboard", {}))
    if input_device == "oem_remote":
        from testbed.actions.oem_remote import OemRemoteActionSource

        return OemRemoteActionSource.from_config(teleop_cfg.get("oem_remote", {}))
    if input_device == "remote":
        from testbed.actions.remote import RemoteActionSource

        return RemoteActionSource.from_config(teleop_cfg.get("remote", {}))
    if input_device == "policy":
        from testbed.actions.policy import PolicyActionSource

        return PolicyActionSource.from_config(teleop_cfg.get("policy", {}))
    if input_device == "policy_remote":
        from testbed.actions.policy_remote import RemoteArmedPolicyActionSource

        return RemoteArmedPolicyActionSource.from_config(teleop_cfg)
    if input_device == "zero":
        return ZeroActionSource()
    raise ValueError(f"Unsupported real teleop input {input_device!r}.")


def _build_bridge_client(
    real_cfg: dict[str, Any],
    backend_mode: str,
    state_reader_mode: str,
):
    if backend_mode != "bridge_tcp" and state_reader_mode != "bridge_tcp":
        return None
    from testbed.backends.real.bridge_socket import JsonTcpBridgeClient

    bridge_cfg = dict(real_cfg.get("bridge", {}) or {})
    host = str(bridge_cfg.get("host", "127.0.0.1"))
    port = int(bridge_cfg.get("port", 0))
    timeout_s = float(bridge_cfg.get("timeout_s", 1.0))
    if port <= 0:
        raise ValueError(
            "real.bridge.port must be set to a positive TCP port when using bridge_tcp."
        )
    return JsonTcpBridgeClient(host=host, port=port, timeout_s=timeout_s)


def _build_control_pump_client(real_cfg: dict[str, Any], pump_cfg: dict[str, Any]):
    pump_bridge_cfg = pump_cfg.get("bridge", {})
    if isinstance(pump_bridge_cfg, dict) and pump_bridge_cfg:
        merged_cfg = dict(real_cfg)
        base_bridge_cfg = dict(real_cfg.get("bridge", {}) or {})
        base_bridge_cfg.update(pump_bridge_cfg)
        merged_cfg["bridge"] = base_bridge_cfg
        return _build_bridge_client(merged_cfg, "bridge_tcp", "mock")
    return _build_bridge_client(real_cfg, "bridge_tcp", "mock")


def _load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to run tb-record-real. Install project dependencies "
            "or use the adapter/unit tests for no-dependency checks."
        ) from exc

    with open(path) as f:
        return dict(yaml.safe_load(f) or {})


def _dump_yaml_config(cfg: dict[str, Any]) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to snapshot the recording config."
        ) from exc
    return str(yaml.safe_dump(cfg, sort_keys=False))


class ZeroActionSource:
    def reset(self) -> None:
        pass

    def next_action(self, obs: dict[str, Any]):
        return (
            np.zeros(4, dtype=np.float32),
            ActionInfo(source_type="teleop", source_id="zero"),
        )

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class ReceiverHealthSnapshot:
    ok: bool
    error_code: str
    errors: tuple[str, ...]
    imu_summary: str
    diagnostics: dict[str, Any]


class ReceiverHealthEvaluator:
    """Evaluate whether the receiver path is healthy enough to write HDF5."""

    def __init__(
        self,
        *,
        mode: str = "strict",
        require_machine_health: bool = False,
        require_remote_action: bool = False,
        bridge_snapshot_timeout_ms: float = 200.0,
        fpv_max_stale_ms: float = 1000.0,
        primary_camera: str = "fpv",
        camera_max_stale_ms: float | None = None,
        imu_require_online: bool = True,
        imu_require_valid_attitude: bool = True,
    ) -> None:
        self.mode = str(mode or "strict")
        self.require_machine_health = bool(require_machine_health)
        self.require_remote_action = bool(require_remote_action)
        self.bridge_snapshot_timeout_ms = float(bridge_snapshot_timeout_ms)
        self.fpv_max_stale_ms = float(fpv_max_stale_ms)
        self.primary_camera = str(primary_camera or "fpv")
        self.camera_max_stale_ms = float(
            self.fpv_max_stale_ms if camera_max_stale_ms is None else camera_max_stale_ms
        )
        self.imu_require_online = bool(imu_require_online)
        self.imu_require_valid_attitude = bool(imu_require_valid_attitude)

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        *,
        input_device: str,
        state_reader_mode: str,
    ) -> ReceiverHealthEvaluator:
        cfg = dict(cfg or {})
        return cls(
            mode=str(cfg.get("mode", "strict")),
            require_machine_health=bool(
                cfg.get("require_machine_health", state_reader_mode == "bridge_tcp")
            ),
            require_remote_action=bool(
                cfg.get("require_remote_action", input_device == "remote")
            ),
            bridge_snapshot_timeout_ms=float(
                cfg.get("bridge_snapshot_timeout_ms", 200.0)
            ),
            fpv_max_stale_ms=float(cfg.get("fpv_max_stale_ms", 1000.0)),
            primary_camera=str(cfg.get("primary_camera", "fpv")),
            camera_max_stale_ms=float(
                cfg.get("camera_max_stale_ms", cfg.get("fpv_max_stale_ms", 1000.0))
            ),
            imu_require_online=bool(cfg.get("imu_require_online", True)),
            imu_require_valid_attitude=bool(
                cfg.get("imu_require_valid_attitude", True)
            ),
        )

    def evaluate(
        self,
        *,
        obs: dict[str, Any],
        action_info,
        control_result: dict[str, Any],
        now_ns: int | None = None,
    ) -> ReceiverHealthSnapshot:
        now_ns = time.time_ns() if now_ns is None else int(now_ns)
        errors: list[str] = []
        sensor_health = obs.get("sensor_health") or {}
        if not isinstance(sensor_health, dict):
            sensor_health = {}
        imu_health = sensor_health.get("imu") or {}
        if not isinstance(imu_health, dict):
            imu_health = {}

        online = _health_bits(imu_health.get("online"), size=4)
        valid_attitude = _health_bits(imu_health.get("valid_attitude"), size=4)
        imu_summary = _imu_summary(online)
        bridge_age_ms = _float_or_default(
            sensor_health.get("bridge_snapshot_age_ms"), -1.0
        )
        camera_age_ms = _camera_age_ms(
            obs,
            camera_name=self.primary_camera,
            now_ns=now_ns,
        )
        fpv_age_ms = _fpv_age_ms(obs, now_ns=now_ns)
        if self.primary_camera != "fpv" and fpv_age_ms < 0.0:
            fpv_age_ms = camera_age_ms

        extras = getattr(action_info, "extras", {}) or {}
        remote_connected = bool(extras.get("remote_action_connected", False))
        remote_stale = bool(extras.get("remote_action_stale", False))

        controller_ack = bool(control_result.get("ack", False))
        controller_fault = str(control_result.get("fault_code", "") or "")

        if self.mode != "disabled":
            if self.require_machine_health:
                if self.imu_require_online:
                    missing = _bad_health_indices(online)
                    if missing:
                        errors.append(f"imu_missing:{','.join(str(i) for i in missing)}")
                if self.imu_require_valid_attitude:
                    invalid = _bad_health_indices(valid_attitude)
                    if invalid:
                        errors.append(
                            "imu_attitude_invalid:"
                            + ",".join(str(i) for i in invalid)
                        )
                if bridge_age_ms < 0.0 or bridge_age_ms > self.bridge_snapshot_timeout_ms:
                    errors.append("bridge_stale")
                if camera_age_ms < 0.0 or camera_age_ms > self.camera_max_stale_ms:
                    errors.append(
                        "fpv_stale" if self.primary_camera == "fpv" else "camera_stale"
                    )

            if self.require_remote_action:
                if not remote_connected:
                    errors.append("remote_disconnected")
                elif remote_stale:
                    errors.append("remote_stale")

            if not controller_ack or controller_fault:
                errors.append("control_fault")

        error_code = errors[0] if errors else ""
        diagnostics = {
            "receiver_health_ok": int(not errors),
            "receiver_health_error_code": error_code,
            "imu_online": online.astype(np.int32, copy=True),
            "imu_valid_attitude": valid_attitude.astype(np.int32, copy=True),
            "fpv_age_ms": float(fpv_age_ms),
            "camera_primary": self.primary_camera,
            "camera_age_ms": float(camera_age_ms),
            "bridge_snapshot_age_ms": float(bridge_age_ms),
            "remote_action_connected": int(remote_connected),
            "controller_ack": int(controller_ack),
        }
        return ReceiverHealthSnapshot(
            ok=not errors,
            error_code=error_code,
            errors=tuple(errors),
            imu_summary=imu_summary,
            diagnostics=diagnostics,
        )


class RecordSession:
    """HDF5 record session owned by the receiver state machine."""

    def __init__(
        self,
        *,
        recorder_cls,
        dataset_dir: Path,
        failed_dir: Path,
        episode_idx: int,
        metadata: dict[str, Any],
        camera_names: list[str],
    ) -> None:
        self.episode_idx = int(episode_idx)
        self.dataset_dir = Path(dataset_dir)
        self.failed_dir = Path(failed_dir)
        self.recorder = recorder_cls(
            output_dir=self.dataset_dir,
            episode_idx=self.episode_idx,
            metadata=metadata,
            camera_names=camera_names,
        )

    def __len__(self) -> int:
        return len(self.recorder)

    def record_step(self, **kwargs: Any) -> None:
        self.recorder.record(**kwargs)

    def mark_recent_train_exclude(self, *, window_steps: int) -> None:
        total = len(self.recorder)
        if total <= 0:
            return
        values = self.recorder._diagnostics.setdefault(  # noqa: SLF001
            "train_exclude_mask",
            [0] * total,
        )
        while len(values) < total:
            values.append(0)
        start = max(0, total - max(1, int(window_steps)))
        for idx in range(start, total):
            values[idx] = 1

    def save_success(self, *, metadata_updates: dict[str, Any] | None = None) -> Path:
        return self.recorder.save(
            success=True,
            metadata_updates=metadata_updates,
        )

    def save_failed(
        self,
        *,
        error_code: str,
        error_time_ns: int,
        stop_reason: str = "sensor_error",
        metadata_updates: dict[str, Any] | None = None,
    ) -> Path | None:
        if len(self.recorder) == 0:
            return None
        stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
        path = self.failed_dir / f"episode_{self.episode_idx}_failed_{stamp}.hdf5"
        updates = {
            "record_stop_reason": stop_reason,
            "record_error_code": str(error_code),
            "record_error_time_ns": int(error_time_ns),
        }
        if metadata_updates:
            updates.update(metadata_updates)
        return self.recorder.save(
            success=False,
            path=path,
            metadata_updates=updates,
        )


class ReceiverTestLogger:
    """Lightweight JSONL logger for receiver/control tests without HDF5 data."""

    def __init__(
        self,
        *,
        run_dir: Path,
        metadata: dict[str, Any],
        record_config_yaml: str,
        flush_every_steps: int = 1,
        capture_images: bool = False,
        image_camera: str = "fpv",
        image_interval_steps: int = 20,
        image_policy_only: bool = True,
        image_dir_name: str = "fpv_frames",
        image_jpeg_quality: int = 95,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.run_dir / "steps.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.metadata_path = self.run_dir / "metadata.json"
        self.flush_every_steps = max(1, int(flush_every_steps))
        self.capture_images = bool(capture_images)
        self.image_camera = str(image_camera)
        self.image_interval_steps = max(1, int(image_interval_steps))
        self.image_policy_only = bool(image_policy_only)
        self.image_jpeg_quality = int(image_jpeg_quality)
        self.image_dir = self.run_dir / str(image_dir_name)
        self._last_image_capture_step: int | None = None
        self._captured_images = 0
        self._started_ns = time.time_ns()
        self._steps = 0
        self._closed = False
        self._fh = self.steps_path.open("w", encoding="utf-8")
        if self.capture_images:
            self.image_dir.mkdir(parents=True, exist_ok=True)
        metadata_payload = {
            "started_at_ns": self._started_ns,
            "metadata": metadata,
            "record_config_yaml": record_config_yaml,
            "image_capture": {
                "enabled": self.capture_images,
                "camera": self.image_camera,
                "interval_steps": self.image_interval_steps,
                "policy_only": self.image_policy_only,
                "dir": str(self.image_dir.relative_to(self.run_dir)),
                "jpeg_quality": self.image_jpeg_quality,
            },
        }
        self.metadata_path.write_text(
            json.dumps(_jsonable(metadata_payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        *,
        metadata: dict[str, Any],
        record_config_yaml: str,
    ) -> ReceiverTestLogger:
        cfg = dict(cfg or {})
        root = Path(cfg.get("output_dir", "runs/receiver_tests"))
        run_name = str(cfg.get("run_name", "") or "")
        if not run_name:
            stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
            task_name = str(metadata.get(ATTR_TASK_NAME, "receiver_test"))
            run_name = f"{_sanitize_key(task_name)}_{stamp}"
        return cls(
            run_dir=root / run_name,
            metadata=metadata,
            record_config_yaml=record_config_yaml,
            flush_every_steps=int(cfg.get("flush_every_steps", 1)),
            capture_images=bool(
                cfg.get("capture_images", cfg.get("save_fpv_images", False))
            ),
            image_camera=str(cfg.get("image_camera", "fpv")),
            image_interval_steps=int(cfg.get("image_interval_steps", 20)),
            image_policy_only=bool(cfg.get("image_policy_only", True)),
            image_dir_name=str(cfg.get("image_dir", "fpv_frames")),
            image_jpeg_quality=int(cfg.get("image_jpeg_quality", 95)),
        )

    def record_step(
        self,
        *,
        local_step: int,
        receiver_mode: str,
        obs: dict[str, Any],
        raw_action: np.ndarray,
        safe_action: np.ndarray,
        action_info: Any,
        action_sample_timestamp_ns: int,
        action_send_timestamp_ns: int,
        guard: Any,
        control_result: dict[str, Any],
        receiver_health: ReceiverHealthSnapshot,
        record_start_requested: bool,
        go_home_requested: bool,
        go_home_update: Any | None = None,
    ) -> None:
        if self._closed:
            return
        extras = getattr(action_info, "extras", {}) or {}
        payload = {
            "wall_time_ns": time.time_ns(),
            "local_step": int(local_step),
            "receiver_mode": str(receiver_mode),
            "observation_step_id": int(obs.get("step_id", local_step)),
            "qpos": np.asarray(obs.get("qpos", np.zeros(4)), dtype=np.float32),
            "qvel": np.asarray(obs.get("qvel", np.zeros(4)), dtype=np.float32),
            "raw_action": np.asarray(raw_action, dtype=np.float32),
            "safe_action": np.asarray(safe_action, dtype=np.float32),
            "commanded_action": np.asarray(
                control_result.get("commanded_action", safe_action),
                dtype=np.float32,
            ),
            "action_source_type": str(getattr(action_info, "source_type", "")),
            "action_source_id": str(getattr(action_info, "source_id", "")),
            "action_source_latency_ms": float(
                getattr(action_info, "latency_ms", 0.0) or 0.0
            ),
            "action_sample_timestamp_ns": int(action_sample_timestamp_ns),
            "action_send_timestamp_ns": int(action_send_timestamp_ns),
            "observation_timestamp_ns": _int_timestamp(obs.get("timestamp_ns")),
            "joint_timestamp_ns": _int_timestamp(obs.get("joint_timestamp_ns")),
            "image_timestamp_ns": _jsonable(obs.get("image_timestamp_ns", {})),
            "record_start_requested": int(bool(record_start_requested)),
            "policy_start_requested": int(
                bool(extras.get("policy_start_requested", False))
            ),
            "go_home_requested": int(bool(go_home_requested)),
            "guard_triggered": int(bool(getattr(guard, "triggered", False))),
            "guard_reasons": list(getattr(guard, "reasons", ()) or ()),
            "controller_ack": int(bool(control_result.get("ack", False))),
            "controller_fault_code": str(control_result.get("fault_code", "")),
            "controller_timestamp_ns": _int_timestamp(
                control_result.get("controller_timestamp_ns")
            ),
            "receiver_health_ok": int(bool(receiver_health.ok)),
            "receiver_health_error_code": str(receiver_health.error_code),
            "receiver_health_errors": list(receiver_health.errors),
            "policy_action": extras.get("policy_action"),
            "policy_scaled_action": extras.get("policy_scaled_action"),
            "policy_assisted_action": extras.get("policy_assisted_action"),
            "policy_returned_action": extras.get("policy_returned_action"),
            "policy_output_mode": str(extras.get("policy_output_mode", "")),
            "policy_deadzone_assist_enabled": int(
                extras.get("policy_deadzone_assist_enabled", 0) or 0
            ),
            "policy_deadzone_assist_active": int(
                extras.get("policy_deadzone_assist_active", 0) or 0
            ),
            "policy_deadzone_assist_mask": extras.get(
                "policy_deadzone_assist_mask"
            ),
            "policy_deadzone_assist_axes": str(
                extras.get("policy_deadzone_assist_axes", "")
            ),
            "policy_qvel_mode": str(extras.get("policy_qvel_mode", "")),
            "policy_qvel_input": extras.get("policy_qvel_input"),
            "policy_error": str(extras.get("policy_error", "")),
            "policy_inference_latency_ms": float(
                extras.get("policy_inference_latency_ms", 0.0) or 0.0
            ),
            "policy_remote_mode": str(extras.get("policy_remote_mode", "")),
            "policy_remote_activated": int(
                extras.get("policy_remote_activated", 0) or 0
            ),
            "policy_remote_activation_step": int(
                extras.get("policy_remote_activation_step", -1) or -1
            ),
        }
        if go_home_update is not None:
            payload["go_home_done"] = int(bool(getattr(go_home_update, "done", False)))
            payload["go_home_failed"] = int(
                bool(getattr(go_home_update, "failed", False))
            )
            payload["go_home_reason"] = str(getattr(go_home_update, "reason", "") or "")
        image_info = self._maybe_capture_image(
            local_step=int(local_step),
            obs=obs,
            extras=extras,
        )
        if image_info:
            payload.update(image_info)
        self._fh.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")
        self._steps += 1
        if self._steps % self.flush_every_steps == 0:
            self._fh.flush()

    def _maybe_capture_image(
        self,
        *,
        local_step: int,
        obs: dict[str, Any],
        extras: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.capture_images:
            return {}
        if self.image_policy_only and extras.get("policy_action") is None:
            return {}
        if (
            self._last_image_capture_step is not None
            and local_step - self._last_image_capture_step < self.image_interval_steps
        ):
            return {}

        rel_path = Path(self.image_dir.name) / f"{self.image_camera}_{local_step:06d}.jpg"
        path = self.run_dir / rel_path
        encoded = dict(obs.get("encoded_images", {}) or {}).get(self.image_camera)
        if isinstance(encoded, dict) and str(encoded.get("encoding", "")) == "jpeg":
            data = encoded.get("data", encoded.get("bytes", b""))
            if isinstance(data, np.ndarray):
                raw = np.asarray(data, dtype=np.uint8).reshape(-1).tobytes()
            else:
                raw = bytes(data)
            path.write_bytes(raw)
            shape = tuple(int(v) for v in encoded.get("shape", ()) or ())
            source = "encoded_jpeg"
        else:
            images = dict(obs.get("images", {}) or {})
            if self.image_camera not in images:
                return {}
            image = np.asarray(images[self.image_camera], dtype=np.uint8)
            if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
                return {}
            self._write_rgb_jpeg(path, image)
            shape = tuple(int(v) for v in image.shape)
            source = "rgb_array"

        self._last_image_capture_step = int(local_step)
        self._captured_images += 1
        return {
            "fpv_image_path": str(rel_path),
            "fpv_image_capture_source": source,
            "fpv_image_shape": list(shape),
            "fpv_image_capture_index": self._captured_images - 1,
        }

    def _write_rgb_jpeg(self, path: Path, image: np.ndarray) -> None:
        try:
            from PIL import Image

            Image.fromarray(image).save(
                path,
                format="JPEG",
                quality=self.image_jpeg_quality,
            )
            return
        except ImportError:
            pass

        import cv2

        if image.shape[-1] == 3:
            encoded_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[-1] == 4:
            encoded_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            encoded_image = image
        ok, buf = cv2.imencode(
            ".jpg",
            encoded_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.image_jpeg_quality],
        )
        if not ok:
            raise RuntimeError("failed to encode FPV frame as JPEG")
        path.write_bytes(buf.tobytes())

    def close(self, *, stop_reason: str = "complete") -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._fh.flush()
            self._fh.close()
        finally:
            finished_ns = time.time_ns()
            summary = {
                "started_at_ns": self._started_ns,
                "finished_at_ns": finished_ns,
                "duration_s": (finished_ns - self._started_ns) / 1_000_000_000.0,
                "steps": self._steps,
                "stop_reason": str(stop_reason),
                "steps_path": str(self.steps_path),
                "flush_every_steps": self.flush_every_steps,
                "captured_images": self._captured_images,
                "image_dir": (
                    str(self.image_dir.relative_to(self.run_dir))
                    if self.capture_images
                    else ""
                ),
            }
            self.summary_path.write_text(
                json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


@dataclass(frozen=True)
class _GuardInfoSnapshot:
    triggered: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _ControlLoopSample:
    raw_action: np.ndarray
    safe_action: np.ndarray
    action_info: ActionInfo
    action_sample_timestamp_ns: int
    action_send_timestamp_ns: int
    control_result: Any
    guard_info: _GuardInfoSnapshot
    sensor_age_s: float | None


class _RemoteControlLoop:
    """Poll remote action and update the real action pump at fixed control rate."""

    def __init__(
        self,
        *,
        action_source,
        guard,
        control_pump,
        rate_hz: float,
        initial_obs: dict[str, Any],
    ) -> None:
        if rate_hz <= 0:
            raise ValueError("remote control loop rate_hz must be positive")
        self._action_source = action_source
        self._guard = guard
        self._control_pump = control_pump
        self._period_s = 1.0 / float(rate_hz)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_obs = initial_obs
        zero = np.zeros(4, dtype=np.float32)
        init_result = control_pump.latest_result
        self._latest_sample = _ControlLoopSample(
            raw_action=zero,
            safe_action=zero,
            action_info=ActionInfo(
                source_type="teleop",
                source_id="remote:control_loop:init",
                extras={},
            ),
            action_sample_timestamp_ns=0,
            action_send_timestamp_ns=int(init_result.controller_timestamp_ns or 0),
            control_result=init_result,
            guard_info=_GuardInfoSnapshot(triggered=False, reasons=()),
            sensor_age_s=None,
        )
        self._record_start_requested = False
        self._go_home_requested = False
        self._reset_requested = False
        self._discard_requested = False
        self._quit_requested = False
        self._fault_hold = False
        self._scripted_action: np.ndarray | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="remote-control-loop",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self._period_s * 4.0))
        self._thread = None

    def update_observation(self, obs: dict[str, Any]) -> None:
        with self._lock:
            self._latest_obs = obs

    def set_fault_hold(self, enabled: bool) -> None:
        with self._lock:
            self._fault_hold = bool(enabled)

    def set_scripted_action(self, action: np.ndarray | None) -> None:
        with self._lock:
            self._scripted_action = (
                None
                if action is None
                else np.asarray(action, dtype=np.float32).reshape(4).copy()
            )

    def latest_sample(self) -> _ControlLoopSample:
        with self._lock:
            return self._latest_sample

    def consume_requests(self) -> tuple[bool, bool, bool, bool, bool]:
        with self._lock:
            out = (
                self._record_start_requested,
                self._go_home_requested,
                self._reset_requested,
                self._discard_requested,
                self._quit_requested,
            )
            self._record_start_requested = False
            self._go_home_requested = False
            self._reset_requested = False
            self._discard_requested = False
            self._quit_requested = False
            return out

    def _run(self) -> None:
        next_tick_s = time.perf_counter()
        while not self._stop.is_set():
            next_tick_s += self._period_s
            try:
                self._step_once()
            except Exception:
                log.exception("remote control loop failed")
                with self._lock:
                    self._quit_requested = True
                return
            remaining_s = next_tick_s - time.perf_counter()
            if remaining_s > 0:
                self._stop.wait(remaining_s)
            else:
                next_tick_s = time.perf_counter()

    def _step_once(self) -> None:
        with self._lock:
            obs = self._latest_obs
            fault_hold = self._fault_hold
            scripted_action = (
                None
                if self._scripted_action is None
                else self._scripted_action.copy()
            )
        raw_action, action_info = self._action_source.next_action(obs)
        action_sample_ns = _action_sample_timestamp_ns(action_info)
        extras = getattr(action_info, "extras", {}) or {}
        reset_now, discard_now, quit_now = _action_control_flags(action_info)
        sensor_age_s = _sensor_age_s(obs, now_ns=time.time_ns())
        safety_state = dict(obs.get("safety_state", {}))
        guarded_input = raw_action if scripted_action is None else scripted_action
        safe_action, _triggered = self._guard.check(
            guarded_input,
            obs.get("qpos"),
            deadman_pressed=bool(safety_state.get("deadman_pressed", True)),
            estop_active=bool(safety_state.get("estop_active", False)),
            manual_override_active=bool(
                safety_state.get("manual_override_active", False)
            ),
            sensor_stale=bool(safety_state.get("sensor_stale", False)),
            sensor_age_s=sensor_age_s,
        )

        toggle_mask = int(extras.get("toggle_mask", 0) or 0)
        if fault_hold:
            safe_action = np.zeros(4, dtype=np.float32)
        if toggle_mask and scripted_action is None:
            self._control_pump.apply_status_toggle_mask(toggle_mask)

        action_send_ns = time.time_ns()
        pump_result = self._control_pump.update_action(safe_action, state=obs)
        action_send_ns = int(pump_result.controller_timestamp_ns or action_send_ns)
        guard_info = _GuardInfoSnapshot(
            triggered=bool(self._guard.last_info.triggered),
            reasons=tuple(str(reason) for reason in self._guard.last_info.reasons),
        )
        sample = _ControlLoopSample(
            raw_action=np.asarray(raw_action, dtype=np.float32).copy(),
            safe_action=np.asarray(safe_action, dtype=np.float32).copy(),
            action_info=action_info,
            action_sample_timestamp_ns=int(action_sample_ns),
            action_send_timestamp_ns=action_send_ns,
            control_result=pump_result,
            guard_info=guard_info,
            sensor_age_s=sensor_age_s,
        )
        with self._lock:
            self._latest_sample = sample
            self._record_start_requested = (
                self._record_start_requested
                or bool(extras.get("record_start_requested", False))
            )
            self._go_home_requested = self._go_home_requested or bool(
                extras.get("go_home_requested", False)
            )
            self._reset_requested = self._reset_requested or bool(reset_now)
            self._discard_requested = self._discard_requested or bool(discard_now)
            self._quit_requested = self._quit_requested or bool(quit_now)


class _QcEventLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def record(
        self,
        *,
        level: str,
        code: str,
        message: str,
        mode: str,
        episode_idx: int,
        record_steps: int,
        online_qc: Any | None,
    ) -> None:
        diagnostics = _qc_diagnostics(online_qc)
        payload = {
            "time_ns": time.time_ns(),
            "episode_idx": int(episode_idx),
            "record_steps": int(record_steps),
            "mode": str(mode),
            "level": str(level),
            "code": str(code),
            "message": str(message),
            "online_qc_status": str(_qc_attr(online_qc, "status", "")),
            "online_qc_reference_id": str(
                diagnostics.get("online_qc_reference_id", "")
            ),
            "diagnostics": _qc_jsonable(diagnostics),
        }
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class _QcDashboard:
    _WIDTHS = (18, 10, 30, 34, 16, 12)

    def __init__(
        self,
        *,
        enabled: bool,
        max_events: int = 8,
        event_logger: _QcEventLogger | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_events = max(1, int(max_events))
        self.event_logger = event_logger
        self._events: list[tuple[str, str]] = []
        self._last_event_key: tuple[str, str] | None = None
        self._rendered = False

    def add_event(self, level: str, message: str) -> None:
        if not self.enabled:
            return
        clean_level = str(level or "INFO").upper()[:8]
        clean_message = " ".join(str(message or "").split()) or "-"
        self._events.append((clean_level, clean_message))
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]

    def update(
        self,
        *,
        mode: str,
        episode_idx: int,
        saved: int,
        record_steps: int,
        receiver_health: ReceiverHealthSnapshot | None,
        online_qc: Any | None,
        action_info: Any | None,
        message: str = "",
    ) -> None:
        if not self.enabled:
            return
        self._add_snapshot_event(
            mode=mode,
            episode_idx=episode_idx,
            record_steps=record_steps,
            receiver_health=receiver_health,
            online_qc=online_qc,
        )
        text = self.render(
            mode=mode,
            episode_idx=episode_idx,
            saved=saved,
            record_steps=record_steps,
            receiver_health=receiver_health,
            online_qc=online_qc,
            action_info=action_info,
            message=message,
        )
        sys.stdout.write("\033[2J\033[H" + text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        self._rendered = True

    def render(
        self,
        *,
        mode: str,
        episode_idx: int,
        saved: int,
        record_steps: int,
        receiver_health: ReceiverHealthSnapshot | None,
        online_qc: Any | None,
        action_info: Any | None,
        message: str = "",
    ) -> str:
        qc_status = _qc_attr(online_qc, "status", "DISABLED")
        diagnostics = _qc_diagnostics(online_qc)
        warnings = _qc_warnings(online_qc)
        extras = getattr(action_info, "extras", {}) or {}
        remote_seq = extras.get("remote_action_seq", "-")
        remote_stale = int(bool(extras.get("remote_action_stale", False)))
        lines = [
            (
                "Online QC Dashboard  "
                f"mode={mode} episode={int(episode_idx)} steps={int(record_steps)} "
                f"saved={int(saved)} qc={qc_status} remote_seq={remote_seq} "
                f"stale={remote_stale} msg={message or '-'}"
            ),
            _qc_dashboard_row(
                "CHECK", "STATE", "CODE/WARN", "VALUE", "COUNT", "ACTION"
            ),
            "-" * (sum(self._WIDTHS) + len(self._WIDTHS) * 3 - 3),
        ]
        for row in (
            self._receiver_row(receiver_health),
            self._qpos_row(online_qc, warnings, diagnostics),
            self._bucket_reference_row(online_qc, diagnostics),
            self._bucket_semantic_row(online_qc, diagnostics),
            self._imu_row(online_qc, warnings, diagnostics),
            self._fpv_frame_row(online_qc, diagnostics),
            self._fpv_drift_row(online_qc, warnings, diagnostics),
            self._episode_final_row(online_qc, diagnostics, record_steps),
        ):
            lines.append(_qc_dashboard_row(*row))
        lines.extend(["", "RECENT EVENTS"])
        if not self._events:
            lines.append("  -")
        else:
            for level, event in self._events[-self.max_events :]:
                lines.append(f"  {level:<8} {event}")
        return "\n".join(lines)

    def finish(self) -> None:
        if not self.enabled or not self._rendered:
            return
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._rendered = False

    def _add_snapshot_event(
        self,
        *,
        mode: str,
        episode_idx: int,
        record_steps: int,
        receiver_health: ReceiverHealthSnapshot | None,
        online_qc: Any | None,
    ) -> None:
        error_code = str(_qc_attr(online_qc, "error_code", "") or "")
        warnings = _qc_warnings(online_qc)
        if error_code:
            event_key = ("FAIL", error_code)
        elif receiver_health is not None and not receiver_health.ok:
            event_key = ("FAIL", receiver_health.error_code or "receiver_health")
        elif warnings:
            event_key = ("WARN", ",".join(warnings))
        else:
            event_key = None
        if event_key is not None and event_key != self._last_event_key:
            self.add_event(event_key[0], event_key[1])
            if self.event_logger is not None:
                self.event_logger.record(
                    level=event_key[0],
                    code=event_key[1],
                    message=event_key[1],
                    mode=mode,
                    episode_idx=episode_idx,
                    record_steps=record_steps,
                    online_qc=online_qc,
                )
            self._last_event_key = event_key

    def _receiver_row(
        self, receiver_health: ReceiverHealthSnapshot | None
    ) -> tuple[str, str, str, str, str, str]:
        if receiver_health is None:
            return ("receiver_health", "UNKNOWN", "-", "-", "-", "hold")
        state = "OK" if receiver_health.ok else "FAIL"
        diagnostics = receiver_health.diagnostics or {}
        fpv_age = _qc_format_ms(diagnostics.get("fpv_age_ms"))
        bridge_age = _qc_format_ms(diagnostics.get("bridge_snapshot_age_ms"))
        code = receiver_health.error_code or "-"
        value = (
            f"fpv_age={fpv_age} bridge={bridge_age} imu={receiver_health.imu_summary}"
        )
        return (
            "receiver_health",
            state,
            code,
            value,
            f"errors={len(receiver_health.errors)}",
            "allow" if receiver_health.ok else "hold",
        )

    def _qpos_row(
        self,
        online_qc: Any | None,
        warnings: tuple[str, ...],
        diagnostics: dict[str, Any],
    ) -> tuple[str, str, str, str, str, str]:
        error_code = str(_qc_attr(online_qc, "error_code", "") or "")
        qpos_warnings = tuple(code for code in warnings if code.startswith("qpos_"))
        warn_count = int(diagnostics.get("online_qc_qpos_warn_count", 0) or 0)
        fail_count = int(diagnostics.get("online_qc_qpos_fail_count", 0) or 0)
        train_exclude = bool(_qc_attr(online_qc, "train_exclude", False))
        state = "OK"
        code = "-"
        if error_code.startswith("qpos_"):
            state = "FAIL"
            code = error_code
        elif qpos_warnings or warn_count or fail_count:
            state = "WARN"
            code = ",".join(qpos_warnings) or "qpos_outside_reference"
        return (
            "qpos_distribution",
            state,
            code,
            f"warn={warn_count} fail={fail_count}",
            f"mask={int(diagnostics.get('online_qc_train_exclude_steps', 0) or 0)}",
            "mask" if train_exclude else "keep",
        )

    def _bucket_reference_row(
        self,
        online_qc: Any | None,
        diagnostics: dict[str, Any],
    ) -> tuple[str, str, str, str, str, str]:
        error_code = str(_qc_attr(online_qc, "error_code", "") or "")
        status = str(diagnostics.get("online_qc_bucket_reference_status", "") or "UNKNOWN")
        state = "MISS" if status == "UNKNOWN" else status
        if error_code == "bucket_reference_outlier":
            state = "FAIL"
        code = error_code if error_code == "bucket_reference_outlier" else status
        low = _qc_format_float(diagnostics.get("online_qc_bucket_ref_low_margin"))
        high = _qc_format_float(diagnostics.get("online_qc_bucket_ref_high_margin"))
        return (
            "bucket_reference",
            state,
            code,
            f"low={low} high={high}",
            f"candidate={diagnostics.get('online_qc_train_ready_candidate', '-')}",
            "fail" if state == "FAIL" else "review" if state in {"WARN", "MISS"} else "keep",
        )

    def _bucket_semantic_row(
        self,
        online_qc: Any | None,
        diagnostics: dict[str, Any],
    ) -> tuple[str, str, str, str, str, str]:
        error_code = str(_qc_attr(online_qc, "error_code", "") or "")
        decision = str(
            diagnostics.get("online_qc_bucket_semantic_decision", "") or "-"
        )
        notes = str(diagnostics.get("online_qc_bucket_semantic_notes", "") or "-")
        ref_count = int(
            diagnostics.get("online_qc_bucket_semantic_reference_count", 0) or 0
        )
        if error_code == "bucket_semantic_outlier" or decision == "drop":
            state = "FAIL"
        elif decision == "review":
            state = "WARN"
        elif decision == "keep":
            state = "OK"
        else:
            state = "MISS" if ref_count == 0 else "OK"
        return (
            "bucket_semantic",
            state,
            decision,
            notes,
            f"ref={ref_count}",
            "fail" if state == "FAIL" else "review" if state in {"WARN", "MISS"} else "keep",
        )

    def _imu_row(
        self,
        online_qc: Any | None,
        warnings: tuple[str, ...],
        diagnostics: dict[str, Any],
    ) -> tuple[str, str, str, str, str, str]:
        error_code = str(_qc_attr(online_qc, "error_code", "") or "")
        imu_warnings = tuple(code for code in warnings if code.startswith("imu_"))
        state = "OK"
        code = "-"
        if error_code.startswith("imu_"):
            state = "FAIL"
            code = error_code
        elif "imu_qpos_reference_missing" in imu_warnings:
            state = "MISS"
            code = "imu_qpos_reference_missing"
        elif imu_warnings:
            state = "WARN"
            code = ",".join(imu_warnings)
        max_delta = diagnostics.get("online_qc_imu_qpos_max_delta_rad")
        return (
            "imu_qpos",
            state,
            code,
            f"max_delta={_qc_format_float(max_delta)}",
            f"count={int(diagnostics.get('online_qc_imu_qpos_delta_count', 0) or 0)}",
            "fail" if state == "FAIL" else "review" if state != "OK" else "keep",
        )

    def _fpv_frame_row(
        self,
        online_qc: Any | None,
        diagnostics: dict[str, Any],
    ) -> tuple[str, str, str, str, str, str]:
        error_code = str(_qc_attr(online_qc, "error_code", "") or "")
        frame_error = error_code in {
            "fpv_decode_failed",
            "fpv_black",
            "fpv_black_frame",
            "fpv_duplicate",
            "fpv_duplicate_frame",
        }
        state = "FAIL" if frame_error else "OK"
        code = error_code if frame_error else "-"
        value = (
            f"brightness={_qc_format_float(diagnostics.get('online_qc_fpv_brightness'))} "
            f"contrast={_qc_format_float(diagnostics.get('online_qc_fpv_contrast'))}"
        )
        return ("fpv_frame", state, code, value, "-", "fail" if frame_error else "keep")

    def _fpv_drift_row(
        self,
        online_qc: Any | None,
        warnings: tuple[str, ...],
        diagnostics: dict[str, Any],
    ) -> tuple[str, str, str, str, str, str]:
        error_code = str(_qc_attr(online_qc, "error_code", "") or "")
        drift_count = int(diagnostics.get("online_qc_fpv_drift_count", 0) or 0)
        state = "OK"
        code = "-"
        if error_code == "fpv_drift":
            state = "FAIL"
            code = error_code
        elif "fpv_drift" in warnings or drift_count:
            state = "WARN"
            code = "fpv_drift"
        train_exclude = bool(_qc_attr(online_qc, "train_exclude", False))
        return (
            "fpv_drift",
            state,
            code,
            f"score={_qc_format_float(diagnostics.get('online_qc_fpv_drift_score'))}",
            f"samples={drift_count}",
            "mask" if train_exclude and state == "WARN" else "fail" if state == "FAIL" else "keep",
        )

    def _episode_final_row(
        self,
        online_qc: Any | None,
        diagnostics: dict[str, Any],
        record_steps: int,
    ) -> tuple[str, str, str, str, str, str]:
        error_code = str(_qc_attr(online_qc, "error_code", "") or "")
        train_exclude = bool(_qc_attr(online_qc, "train_exclude", False))
        state = "WARN" if train_exclude else "OK"
        code = "train_exclude_mask" if train_exclude else "-"
        if error_code.startswith("episode_"):
            state = "FAIL"
            code = error_code
        healthy_steps = int(diagnostics.get("online_qc_healthy_steps", 0) or 0)
        healthy_fraction = diagnostics.get("online_qc_healthy_fraction")
        candidate = diagnostics.get("online_qc_train_ready_candidate", "-")
        decision = diagnostics.get("online_qc_bucket_semantic_decision", "-") or "-"
        return (
            "episode_final",
            state,
            code,
            (
                f"healthy={healthy_steps} "
                f"frac={_qc_format_float(healthy_fraction)} "
                f"candidate={candidate}"
            ),
            f"decision={decision}",
            "fail" if state == "FAIL" else "review" if state == "WARN" else "keep",
        )


def _qc_dashboard_row(
    check: str,
    state: str,
    code: str,
    value: str,
    count: str,
    action: str,
) -> str:
    cells = (check, state, code, value, count, action)
    return " | ".join(
        _qc_dashboard_cell(cell, width)
        for cell, width in zip(cells, _QcDashboard._WIDTHS, strict=True)
    )


def _qc_dashboard_cell(value: Any, width: int) -> str:
    text = " ".join(str(value or "-").split())
    if len(text) > width:
        text = text[: max(1, width - 1)] + "~"
    return f"{text:<{width}}"


def _qc_attr(snapshot: Any | None, name: str, default: Any = None) -> Any:
    if snapshot is None:
        return default
    return getattr(snapshot, name, default)


def _qc_diagnostics(snapshot: Any | None) -> dict[str, Any]:
    diagnostics = _qc_attr(snapshot, "diagnostics", {})
    return dict(diagnostics) if isinstance(diagnostics, dict) else {}


def _qc_warnings(snapshot: Any | None) -> tuple[str, ...]:
    warnings = _qc_attr(snapshot, "warning_codes", ())
    if warnings is None:
        return ()
    return tuple(str(code) for code in warnings)


def _qc_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _qc_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_qc_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _qc_format_ms(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f}ms"
    except (TypeError, ValueError):
        return "-"


def _qc_format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


class _LiveActionLine:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._last_len = 0
        self._last_update_s: float | None = None
        self._hz_ema: float | None = None

    def update(
        self,
        *,
        step: int,
        mode: str = "armed",
        raw_action: np.ndarray,
        safe_action: np.ndarray,
        action_info,
        sensor_age_s: float | None,
        control_result: dict[str, Any],
        guard_reasons: tuple[str, ...],
        receiver_health: ReceiverHealthSnapshot | None = None,
        go_home_update: Any | None = None,
    ) -> None:
        if not self.enabled:
            return
        now_s = time.monotonic()
        if self._last_update_s is not None:
            dt_s = max(1e-6, now_s - self._last_update_s)
            inst_hz = 1.0 / dt_s
            self._hz_ema = (
                inst_hz
                if self._hz_ema is None
                else (0.85 * self._hz_ema + 0.15 * inst_hz)
            )
        self._last_update_s = now_s
        commanded_action = control_result.get("commanded_action")
        if commanded_action is None:
            commanded_action = safe_action
        ack = int(bool(control_result.get("ack", False)))
        fault = str(control_result.get("fault_code", "") or "-")
        guard = ",".join(str(reason) for reason in guard_reasons) or "-"
        age_ms = -1.0 if sensor_age_s is None else float(sensor_age_s) * 1000.0
        extras = getattr(action_info, "extras", {}) or {}
        remote_age_ms = extras.get("remote_action_age_ms")
        remote_age_text = "-" if remote_age_ms is None else f"{float(remote_age_ms):.1f}"
        stale = int(bool(extras.get("remote_action_stale", False)))
        drops = int(extras.get("remote_action_drop_count", 0) or 0)
        hz_text = "-" if self._hz_ema is None else f"{self._hz_ema:.1f}"
        health_ok = receiver_health.ok if receiver_health is not None else True
        health_text = "OK" if health_ok else "ERR"
        control_mode = str(extras.get("policy_remote_mode", "") or "")
        control_text = (
            "model"
            if control_mode == "policy"
            else "manual" if control_mode == "manual" else "-"
        )
        assist_enabled = int(extras.get("policy_deadzone_assist_enabled", 0) or 0)
        assist_active = int(extras.get("policy_deadzone_assist_active", 0) or 0)
        assist_axes = str(extras.get("policy_deadzone_assist_axes", "") or "")
        assist_text = (
            "off"
            if not assist_enabled
            else assist_axes
            if assist_active and assist_axes
            else "idle"
        )
        err_text = (
            receiver_health.error_code
            if receiver_health is not None and receiver_health.error_code
            else "-"
        )
        imu_text = receiver_health.imu_summary if receiver_health is not None else "----"
        controller_ts_ns = int(control_result.get("controller_timestamp_ns", 0) or 0)
        control_age_text = (
            "-"
            if controller_ts_ns <= 0
            else f"{max(0.0, (time.time_ns() - controller_ts_ns) / 1_000_000.0):.1f}"
        )
        text = (
            f"mode={mode} control={control_text} "
            f"health={health_text} err={err_text} imu={imu_text} "
            f"{_format_go_home_live_status(mode, go_home_update)}"
            f"hz={hz_text} ctl_ms={control_age_text} "
            f"assist={assist_text} "
            f"raw={_format_action_line_values(raw_action)} "
            f"send={_format_action_line_values(commanded_action)} "
            f"step={int(step)} "
            f"remote_ms={remote_age_text} "
            f"stale={stale} drop={drops} "
            f"age={age_ms:.1f} "
            f"ack={ack} fault={fault} guard={guard}"
        )
        self.message(text)

    def message(self, text: str) -> None:
        if not self.enabled:
            return
        width = max(20, shutil.get_terminal_size((240, 20)).columns)
        text = text[: max(1, width - 1)]
        sys.stdout.write("\r\033[2K" + text)
        sys.stdout.flush()
        self._last_len = len(text)

    def finish(self) -> None:
        if not self.enabled or self._last_len == 0:
            return
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._last_len = 0


def _format_action_line_values(action: Any) -> str:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{float(value):+.3f}" for value in values) + "]"


def _format_go_home_live_status(mode: str, go_home_update: Any | None) -> str:
    if mode != "go_home" or go_home_update is None:
        return ""
    diagnostics = getattr(go_home_update, "diagnostics", {}) or {}
    if bool(getattr(go_home_update, "failed", False)):
        state = f"failed:{getattr(go_home_update, 'reason', '') or 'unknown'}"
    elif bool(getattr(go_home_update, "done", False)):
        state = "reached"
    elif int(diagnostics.get("go_home_in_position", 0) or 0):
        state = "settling"
    else:
        state = "going_home"
    elapsed = _float_or_default(diagnostics.get("go_home_elapsed_s"), 0.0)
    error = diagnostics.get("go_home_error")
    raw_error = diagnostics.get("go_home_raw_error")
    qvel = diagnostics.get("go_home_qvel")
    command = diagnostics.get("go_home_commanded_action")
    scheduled = diagnostics.get("go_home_axis_scheduled")
    center_approach = diagnostics.get("go_home_center_approach_axis")
    stalled = diagnostics.get("go_home_axis_stalled")
    wrong_direction = diagnostics.get("go_home_axis_wrong_direction")
    boost = diagnostics.get("go_home_stall_action_boost")
    reversal_blocked = diagnostics.get("go_home_sign_reversal_blocked")
    max_error = _max_abs_or_default(error, 0.0)
    raw_max_error = _max_abs_or_default(raw_error, 0.0)
    max_axis = _max_abs_axis_name(error)
    stalled_axes = _axis_mask_names(stalled)
    scheduled_axes = _axis_mask_names(scheduled)
    center_approach_axes = _axis_mask_names(center_approach)
    in_position = int(diagnostics.get("go_home_in_position", 0) or 0)
    stable = int(diagnostics.get("go_home_stable_velocity", 0) or 0)
    parts = [
        f"home={state}",
        f"t={elapsed:.1f}s",
        f"maxerr={max_error:.3f}",
        f"rawmaxerr={raw_max_error:.3f}",
        f"maxaxis={max_axis}",
        f"inpos={in_position}",
        f"stable={stable}",
    ]
    if stalled_axes:
        parts.append(f"stalled={stalled_axes}")
        parts.append(
            f"boost={_format_action_line_values(boost if boost is not None else np.zeros(4))}"
        )
    if scheduled_axes:
        parts.append(f"target={scheduled_axes}")
    if center_approach_axes:
        parts.append(f"fine={center_approach_axes}")
    wrong_direction_axes = _axis_mask_names(wrong_direction)
    if wrong_direction_axes:
        parts.append(f"wrongdir={wrong_direction_axes}")
    reversal_blocked_axes = _axis_mask_names(reversal_blocked)
    if reversal_blocked_axes:
        parts.append(f"revblk={reversal_blocked_axes}")
    parts.extend(
        [
            f"herr={_format_action_line_values(error if error is not None else np.zeros(4))}",
            f"hrawerr={_format_action_line_values(raw_error if raw_error is not None else np.zeros(4))}",
            f"hcmd={_format_action_line_values(command if command is not None else np.zeros(4))}",
            f"hqvel={_format_action_line_values(qvel if qvel is not None else np.zeros(4))}",
        ]
    )
    return " ".join(parts) + " "


def _format_go_home_final_error(controller: Any | None) -> str:
    if controller is None:
        return "-"
    return _format_action_line_values(getattr(controller, "final_error", np.zeros(4)))


def _max_abs_axis_name(value: Any) -> str:
    if value is None:
        return "-"
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return "-"
    idx = int(np.argmax(np.abs(arr)))
    if idx >= len(_REAL_AXIS_NAMES):
        return str(idx)
    return _REAL_AXIS_NAMES[idx]


def _axis_mask_names(value: Any) -> str:
    if value is None:
        return ""
    arr = np.asarray(value, dtype=np.int32).reshape(-1)
    names = [
        _REAL_AXIS_NAMES[idx] if idx < len(_REAL_AXIS_NAMES) else str(idx)
        for idx, flag in enumerate(arr)
        if int(flag) != 0
    ]
    return ",".join(names)


def _max_abs_or_default(value: Any, default: float) -> float:
    try:
        values = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return float(default)
    if values.size == 0:
        return float(default)
    return float(np.max(np.abs(values)))


def _optional_float_metadata(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _health_bits(value: Any, *, size: int) -> np.ndarray:
    bits = np.zeros(size, dtype=np.int32)
    if value is None:
        return bits
    try:
        arr = np.asarray(value, dtype=np.int32).reshape(-1)
    except (TypeError, ValueError):
        return bits
    count = min(size, int(arr.size))
    if count > 0:
        bits[:count] = (arr[:count] != 0).astype(np.int32)
    return bits


def _bad_health_indices(bits: np.ndarray) -> list[int]:
    return [int(i) for i, value in enumerate(np.asarray(bits).reshape(-1)) if int(value) == 0]


def _imu_summary(bits: np.ndarray) -> str:
    values = np.asarray(bits, dtype=np.int32).reshape(-1)
    if values.size == 0:
        return "----"
    return "".join("1" if int(value) else "0" for value in values[:4])


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _fpv_age_ms(obs: dict[str, Any], *, now_ns: int) -> float:
    return _camera_age_ms(obs, camera_name="fpv", now_ns=now_ns)


def _camera_age_ms(obs: dict[str, Any], *, camera_name: str, now_ns: int) -> float:
    image_timestamps = obs.get("image_timestamp_ns") or {}
    if isinstance(image_timestamps, dict):
        if str(camera_name) in image_timestamps:
            timestamp_ns = _int_timestamp(image_timestamps[str(camera_name)])
        else:
            timestamp_ns = _primary_image_timestamp_ns(image_timestamps)
    else:
        timestamp_ns = _int_timestamp(image_timestamps)
    if timestamp_ns <= 0:
        return -1.0
    return max(0.0, (int(now_ns) - int(timestamp_ns)) / 1_000_000.0)


def _build_step_diagnostics(
    *,
    obs: dict[str, Any],
    raw_action: np.ndarray,
    safe_action: np.ndarray,
    action_info,
    action_sample_timestamp_ns: int,
    action_send_timestamp_ns: int,
    guard,
    control_result: dict[str, Any],
    receiver_health: ReceiverHealthSnapshot | None = None,
    online_qc: Any | None = None,
) -> dict[str, Any]:
    commanded_action = control_result.get("commanded_action")
    if commanded_action is None:
        commanded_action = safe_action
    image_timestamps = obs.get("image_timestamp_ns") or {}
    if not isinstance(image_timestamps, dict):
        image_timestamps = {}
    primary_image_ts = _primary_image_timestamp_ns(image_timestamps)
    extras = getattr(action_info, "extras", {}) or {}
    guard_info = getattr(guard, "last_info", guard)
    diagnostics: dict[str, Any] = {
        "raw_action": np.asarray(raw_action, dtype=np.float32),
        "toggle_mask": int(extras.get("toggle_mask", 0) or 0),
        "status11": np.asarray(extras.get("status11", []), dtype=np.int32),
        "record_start_requested": int(bool(extras.get("record_start_requested", False))),
        "policy_start_requested": int(bool(extras.get("policy_start_requested", False))),
        "go_home_requested": int(bool(extras.get("go_home_requested", False))),
        "guard_triggered": int(bool(guard_info.triggered)),
        "guard_reason": ",".join(str(reason) for reason in guard_info.reasons),
        "controller_ack": int(bool(control_result.get("ack", False))),
        "controller_fault_code": str(control_result.get("fault_code", "")),
        "controller_timestamp_ns": int(control_result.get("controller_timestamp_ns", 0)),
        "commanded_action": np.asarray(commanded_action, dtype=np.float32),
        "action_sample_timestamp_ns": int(action_sample_timestamp_ns),
        "action_send_timestamp_ns": int(action_send_timestamp_ns),
        "action_source_latency_ms": float(getattr(action_info, "latency_ms", 0.0) or 0.0),
        "observation_timestamp_ns": _int_timestamp(obs.get("timestamp_ns")),
        "sensor_timestamp_ns": _int_timestamp(obs.get("sensor_timestamp_ns")),
        "joint_timestamp_ns": _int_timestamp(obs.get("joint_timestamp_ns")),
        "image_timestamp_ns": int(primary_image_ts),
        "sync_timestamp_ns": _int_timestamp(obs.get("sync_timestamp_ns")),
        "sync_max_skew_ns": int(obs.get("sync_max_skew_ns", 0) or 0),
        "sync_warnings": ",".join(str(w) for w in (obs.get("sync_warnings") or [])),
    }
    raw_low_level = control_result.get("raw_low_level_command")
    if raw_low_level is not None:
        diagnostics["raw_low_level_command"] = np.asarray(raw_low_level, dtype=np.float32)
    if obs.get("status") is not None:
        diagnostics["machine_status"] = np.asarray(obs["status"], dtype=np.int32)
    if obs.get("motor_rpm") is not None:
        diagnostics["motor_rpm"] = np.asarray(obs["motor_rpm"], dtype=np.float32)
    if obs.get("plan_rpm") is not None:
        diagnostics["plan_rpm"] = np.asarray(obs["plan_rpm"], dtype=np.float32)
    if receiver_health is not None:
        diagnostics.update(receiver_health.diagnostics)
    if online_qc is not None:
        diagnostics.update(dict(getattr(online_qc, "diagnostics", {}) or {}))
    for camera_name, timestamp_ns in image_timestamps.items():
        diagnostics[f"image_timestamp_ns_{_sanitize_key(camera_name)}"] = _int_timestamp(
            timestamp_ns
        )
    image_metadata = obs.get("image_metadata") or {}
    if isinstance(image_metadata, dict):
        for camera_name, metadata in image_metadata.items():
            if not isinstance(metadata, dict):
                continue
            safe_camera = _sanitize_key(camera_name)
            for key in (
                "sequence",
                "flags",
                "v4l2_timestamp_ns",
                "v4l2_error",
                "group_id",
                "group_target_v4l2_timestamp_ns",
                "group_skew_ns",
                "group_valid",
                "group_camera_count",
            ):
                if key in metadata:
                    diagnostics[f"image_{key}_{safe_camera}"] = _int_timestamp(metadata.get(key))
            if "group_skew_ms" in metadata:
                diagnostics[f"image_group_skew_ms_{safe_camera}"] = float(
                    metadata.get("group_skew_ms") or 0.0
                )
            for key in ("timestamp_clock", "timestamp_source", "group_source"):
                if key in metadata:
                    diagnostics[f"image_{key}_{safe_camera}"] = str(metadata.get(key) or "")
    _add_remote_action_diagnostics(diagnostics, extras)
    _add_policy_action_diagnostics(diagnostics, extras)
    _add_policy_remote_diagnostics(diagnostics, extras)
    return diagnostics


def _ensure_go_home_step_diagnostics(diagnostics: dict[str, Any]) -> None:
    diagnostics.setdefault("go_home_running", 0)
    diagnostics.setdefault("go_home_result_code", "")
    diagnostics.setdefault("go_home_error", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_raw_error", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_policy_raw_delta", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_feedback_consistent", 1)
    diagnostics.setdefault("go_home_raw_imu_qpos", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_has_raw_imu_qpos", 0)
    diagnostics.setdefault("go_home_qvel", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_raw_qvel", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_filtered_qpos", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_filtered_qvel", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_commanded_action", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_unsmoothed_action", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_axis_active", np.zeros(4, dtype=np.int32))
    diagnostics.setdefault("go_home_axis_scheduled", np.zeros(4, dtype=np.int32))
    diagnostics.setdefault("go_home_center_approach_axis", np.zeros(4, dtype=np.int32))
    diagnostics.setdefault("go_home_axis_stalled", np.zeros(4, dtype=np.int32))
    diagnostics.setdefault("go_home_axis_wrong_direction", np.zeros(4, dtype=np.int32))
    diagnostics.setdefault("go_home_stall_action_boost", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_effective_min_action", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_action_limit", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_action_ramp_limit", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_control_signs", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_command_sign", np.zeros(4, dtype=np.float32))
    diagnostics.setdefault("go_home_sign_reversal_blocked", np.zeros(4, dtype=np.int32))
    diagnostics.setdefault("go_home_decision_updated", 0)
    diagnostics.setdefault("go_home_control_decision_period_s", 0.0)
    diagnostics.setdefault("go_home_in_position", 0)
    diagnostics.setdefault("go_home_acceptable_position", 0)
    diagnostics.setdefault("go_home_stable_velocity", 0)
    diagnostics.setdefault("go_home_elapsed_s", 0.0)


def _add_remote_action_diagnostics(
    diagnostics: dict[str, Any],
    extras: dict[str, Any],
) -> None:
    if "remote_action_seq" not in extras:
        return
    int_keys = (
        "remote_action_seq",
        "remote_action_host_sample_ns",
        "remote_action_receive_ns",
        "remote_action_stale",
        "remote_action_drop_count",
        "remote_action_connected",
    )
    for key in int_keys:
        diagnostics[key] = _int_timestamp(extras.get(key))
    diagnostics["remote_action_age_ms"] = float(
        extras.get("remote_action_age_ms", 0.0) or 0.0
    )


def _add_policy_action_diagnostics(
    diagnostics: dict[str, Any],
    extras: dict[str, Any],
) -> None:
    if "policy_action" not in extras:
        return
    diagnostics["policy_action"] = np.asarray(
        extras.get("policy_action", np.zeros(4)),
        dtype=np.float32,
    )
    diagnostics["policy_scaled_action"] = np.asarray(
        extras.get("policy_scaled_action", np.zeros(4)),
        dtype=np.float32,
    )
    diagnostics["policy_assisted_action"] = np.asarray(
        extras.get("policy_assisted_action", diagnostics["policy_scaled_action"]),
        dtype=np.float32,
    )
    diagnostics["policy_returned_action"] = np.asarray(
        extras.get("policy_returned_action", np.zeros(4)),
        dtype=np.float32,
    )
    diagnostics["policy_action_scale"] = np.asarray(
        extras.get("policy_action_scale", np.ones(4)),
        dtype=np.float32,
    )
    diagnostics["policy_deadzone_assist_enabled"] = int(
        extras.get("policy_deadzone_assist_enabled", 0) or 0
    )
    diagnostics["policy_deadzone_assist_active"] = int(
        extras.get("policy_deadzone_assist_active", 0) or 0
    )
    diagnostics["policy_deadzone_assist_mask"] = np.asarray(
        extras.get("policy_deadzone_assist_mask", np.zeros(4)),
        dtype=np.int32,
    )
    diagnostics["policy_deadzone_assist_axes"] = str(
        extras.get("policy_deadzone_assist_axes", "")
    )
    diagnostics["policy_deadzone_assist_trigger_fraction"] = float(
        extras.get("policy_deadzone_assist_trigger_fraction", 0.0) or 0.0
    )
    diagnostics["policy_deadzone_assist_min_consecutive_steps"] = _int_timestamp(
        extras.get("policy_deadzone_assist_min_consecutive_steps")
    )
    diagnostics["policy_deadzone_assist_positive"] = np.asarray(
        extras.get("policy_deadzone_assist_positive", np.zeros(4)),
        dtype=np.float32,
    )
    diagnostics["policy_deadzone_assist_negative"] = np.asarray(
        extras.get("policy_deadzone_assist_negative", np.zeros(4)),
        dtype=np.float32,
    )
    diagnostics["policy_output_mode"] = str(extras.get("policy_output_mode", ""))
    diagnostics["policy_qvel_mode"] = str(extras.get("policy_qvel_mode", ""))
    diagnostics["policy_qvel_input"] = np.asarray(
        extras.get("policy_qvel_input", np.zeros(4)),
        dtype=np.float32,
    )
    diagnostics["policy_error"] = str(extras.get("policy_error", ""))
    diagnostics["policy_step"] = _int_timestamp(extras.get("policy_step"))
    diagnostics["policy_inference_latency_ms"] = float(
        extras.get("policy_inference_latency_ms", 0.0) or 0.0
    )
    if "policy_bundle_dir" in extras:
        diagnostics["policy_bundle_dir"] = str(extras.get("policy_bundle_dir", ""))


def _add_policy_remote_diagnostics(
    diagnostics: dict[str, Any],
    extras: dict[str, Any],
) -> None:
    if "policy_remote_mode" not in extras:
        return
    diagnostics["policy_remote_mode"] = str(extras.get("policy_remote_mode", ""))
    diagnostics["policy_remote_activated"] = int(
        extras.get("policy_remote_activated", 0) or 0
    )
    diagnostics["policy_remote_activation_step"] = int(
        extras.get("policy_remote_activation_step", -1) or -1
    )
    if "policy_remote_remote_action" in extras:
        diagnostics["policy_remote_remote_action"] = np.asarray(
            extras.get("policy_remote_remote_action", np.zeros(4)),
            dtype=np.float32,
        )


def _action_sample_timestamp_ns(action_info) -> int:
    extras = getattr(action_info, "extras", {}) or {}
    return _int_timestamp(extras.get("action_timestamp_ns"), default=time.time_ns())


def _record_step_timestamp_ns(obs: dict[str, Any], *, fallback: int) -> int:
    for key in ("timestamp_ns", "sync_timestamp_ns", "joint_timestamp_ns"):
        timestamp_ns = _int_timestamp(obs.get(key))
        if timestamp_ns > 0:
            return timestamp_ns
    return int(fallback)


def _primary_image_timestamp_ns(image_timestamps: dict[str, Any]) -> int:
    if "fpv" in image_timestamps:
        return _int_timestamp(image_timestamps["fpv"])
    for _name, timestamp_ns in sorted(image_timestamps.items()):
        return _int_timestamp(timestamp_ns)
    return 0


def _default_online_qc_primary_camera(
    online_qc_cfg: dict[str, Any],
    *,
    receiver_health_cfg: dict[str, Any],
    camera_names: list[str],
) -> str:
    raw = online_qc_cfg.get("primary_camera")
    if raw:
        return str(raw)
    primary = receiver_health_cfg.get("primary_camera")
    if not primary and camera_names:
        primary = camera_names[0]
    primary_name = str(primary or "fpv")
    online_qc_cfg["primary_camera"] = primary_name
    return primary_name


def _int_timestamp(value: Any, *, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _sanitize_key(value: Any) -> str:
    return "".join(ch if str(ch).isalnum() or ch == "_" else "_" for ch in str(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _sensor_age_s(obs: dict[str, Any], *, now_ns: int | None = None) -> float | None:
    timestamp_ns = obs.get("sensor_timestamp_ns")
    if timestamp_ns is None:
        return None
    current_ns = time.time_ns() if now_ns is None else int(now_ns)
    return max(0.0, (current_ns - int(timestamp_ns)) * 1e-9)


def _action_control_flags(action_info) -> tuple[bool, bool, bool]:
    extras = getattr(action_info, "extras", {}) or {}
    return (
        bool(extras.get("reset_requested", False)),
        bool(extras.get("discard_requested", False)),
        bool(extras.get("quit_requested", False)),
    )


def _check_pygame_events(*, enabled: bool = True) -> tuple[bool, bool]:
    if not enabled:
        return False, False
    try:
        import pygame

        for _event in pygame.event.get(pygame.QUIT):
            return False, True
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            return False, True
        if keys[pygame.K_d]:
            return True, False
    except Exception:
        pass
    return False, False


_last_step_time: float = 0.0


def _sleep_to_rate(
    control_hz: float,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    global _last_step_time
    target_dt = 1.0 / float(control_hz)
    deadline = _last_step_time + target_dt
    while True:
        if should_stop is not None and should_stop():
            break
        now = time.perf_counter()
        remaining = deadline - now
        if remaining <= 0.0:
            break
        time.sleep(min(remaining, 0.05))
    _last_step_time = time.perf_counter()


def _next_episode_idx(dataset_dir: Path) -> int:
    existing = sorted(
        dataset_dir.glob("episode_*.hdf5"),
        key=lambda p: int(p.stem.split("_", 1)[1]),
    )
    if not existing:
        return 0
    return int(existing[-1].stem.split("_", 1)[1]) + 1


def _build_episode_metadata(
    *,
    task_cfg: dict[str, Any],
    teleop_cfg: dict[str, Any],
    real_cfg: dict[str, Any],
    safety_cfg: dict[str, Any],
    input_device: str,
    camera_names: list[str],
    config_path: Path,
    record_config_yaml: str,
    control_hz: float,
    record_hz: float,
    dt: float,
    control_pump_enabled: bool,
    sync_cfg: dict[str, Any],
    video_cfg: dict[str, Any],
    data_side: str | None = None,
) -> dict[str, Any]:
    camera_width = int(real_cfg.get("image_width", 160))
    camera_height = int(real_cfg.get("image_height", 120))
    image_format = str(
        video_cfg.get(
            "record_format",
            "jpeg" if bool(video_cfg.get("prefer_compressed_transport", False)) else "raw_rgb",
        )
    )
    metadata: dict[str, Any] = {
        ATTR_TASK_NAME: task_cfg.get("task_name", "real_excavation_teleop_v1"),
        ATTR_IS_REAL: True,
        ATTR_PLATFORM: DEFAULT_PLATFORM,
        ATTR_CONTROL_HZ: int(round(control_hz)),
        ATTR_DT: float(dt),
        ATTR_ACTION_SEMANTICS: "normalized_teleop_cmd_v1",
        ATTR_CAMERA_NAMES: ",".join(camera_names),
        ATTR_IMAGE_FORMAT: image_format,
        ATTR_PARAM_VERSION: task_cfg.get("param_version", "real_v1"),
        ATTR_ACTION_ORDER: "swing,boom,stick,bucket",
        ATTR_QPOS_ORDER: "swing,boom,stick,bucket",
        ATTR_QVEL_ORDER: "swing,boom,stick,bucket",
        ATTR_QPOS_UNITS: "rad",
        ATTR_QVEL_UNITS: "rad/s",
        ATTR_QPOS_SOURCE: "joint_sensor_calibrated",
        ATTR_QVEL_SOURCE: "joint_sensor",
        ATTR_HYDRAULIC_CYLINDER_AVAILABLE: False,
        ATTR_TELEOP_INPUT: input_device,
        ATTR_RECORD_CONFIG_PATH: str(config_path),
        ATTR_RECORD_CONFIG_YAML: record_config_yaml,
        ATTR_CAMERA_WIDTH: camera_width,
        ATTR_CAMERA_HEIGHT: camera_height,
        ATTR_CAMERA_FPS: float(record_hz),
        ATTR_CAMERA_ROW_ORDER: "top_to_bottom",
        "real_backend": str(real_cfg.get("backend", "mock")),
        "real_state_reader": str(real_cfg.get("state_reader", "mock")),
        "record_hz": float(record_hz),
        "control_pump_enabled": int(bool(control_pump_enabled)),
        "data_side": str(data_side or real_cfg.get("data_side", "")),
        "learning_target": str(
            teleop_cfg.get("learning_target", "operator_command_from_observation")
        ),
        "sync_time_source": str(sync_cfg.get("time_source", "sensor_or_ros_header_stamp")),
        "sync_max_observation_skew_ms": float(
            sync_cfg.get("max_observation_skew_ms", 40.0)
        ),
        "video_latency_target_ms": float(video_cfg.get("target_latency_ms", 120.0)),
        "video_transport_hint": str(video_cfg.get("transport_hint", "low_latency")),
        "jpeg_quality": int(video_cfg.get("jpeg_quality", 95)),
        "oem_remote_required": int(
            bool(teleop_cfg.get("oem_remote", {}).get("required", False))
        ),
        "safety_deadman_enabled": int(bool(safety_cfg.get("deadman_enabled", True))),
        "safety_estop_enabled": int(bool(safety_cfg.get("estop_enabled", True))),
        "safety_manual_override_enabled": int(
            bool(safety_cfg.get("manual_override_enabled", True))
        ),
        "safety_action_clip": float(safety_cfg.get("action_clip", 0.20)),
        "safety_max_delta_per_step": _optional_float_metadata(
            safety_cfg.get("max_delta_per_step", 0.02)
        ),
        "safety_sensor_timeout_s": float(safety_cfg.get("sensor_timeout_s", 0.20)),
    }

    metadata_cfg = teleop_cfg.get("metadata", {})
    if metadata_cfg.get("operator_id"):
        metadata[ATTR_OPERATOR_ID] = str(metadata_cfg["operator_id"])
    if metadata_cfg.get("session_id"):
        metadata[ATTR_SESSION_ID] = str(metadata_cfg["session_id"])
    if metadata_cfg.get("notes"):
        metadata[ATTR_NOTES] = str(metadata_cfg["notes"])
    if input_device in {"remote", "policy_remote"}:
        from testbed.actions.remote import (
            DEFAULT_REMOTE_ACTION_PORT,
            DEFAULT_REMOTE_ACTION_TIMEOUT_MS,
        )

        remote_cfg = dict(teleop_cfg.get("remote", {}) or {})
        metadata["remote_action_transport"] = "json_tcp"
        metadata["remote_action_bind_host"] = str(
            remote_cfg.get("bind_host", "0.0.0.0")
        )
        metadata["remote_action_port"] = int(
            remote_cfg.get("port", DEFAULT_REMOTE_ACTION_PORT)
        )
        metadata["remote_action_timeout_ms"] = float(
            remote_cfg.get("timeout_ms", DEFAULT_REMOTE_ACTION_TIMEOUT_MS)
        )
        metadata["remote_action_source_id"] = str(
            remote_cfg.get("source_id", "remote_teleop")
        )
    return metadata


if __name__ == "__main__":
    main()
