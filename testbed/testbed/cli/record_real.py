"""
tb-record-real - Record real excavator teleop episodes to HDF5.

The real-world v1 path is intentionally narrow: it records calibrated joint
qpos/qvel, fpv images, post-guard normalized actions, and diagnostics for the
low-level controller boundary. It does not reset hardware and does not call any
planner.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="tb-record-real",
        description="Record real excavator v1 teleop data through the low-level control boundary.",
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
        choices=["joystick", "keyboard", "oem_remote", "zero"],
        default=None,
    )
    parser.add_argument("--operator-id", type=str, default=None)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--notes", type=str, default=None)
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
        default="slave",
        help=(
            "Where HDF5 is written (default slave=vehicle PC). "
            "slave: run on vehicle, data under /data/real_teleop_v1; "
            "host: run on operator PC, set EXCAVATOR_BRIDGE_HOST to vehicle IP. "
            "Override via real.data_side or EXCAVATOR_DATA_SIDE."
        ),
    )
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config)

    real_cfg = cfg.setdefault("real", {})
    teleop_cfg = cfg.setdefault("teleop", {})
    task_cfg = cfg.setdefault("task", {})
    safety_cfg = cfg.setdefault("safety", {})
    sync_cfg = cfg.setdefault("sync", {})
    video_cfg = cfg.setdefault("video", {})
    teleop_meta_cfg = teleop_cfg.setdefault("metadata", {})

    from testbed.cli.data_side import apply_data_side_config, validate_data_side_for_bridge_tcp

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
    dt = float(task_cfg.get("dt", 1.0 / control_hz))
    input_device = str(teleop_cfg.get("input", "joystick"))
    backend_mode = str(real_cfg.get("backend", "mock"))
    state_reader_mode = str(real_cfg.get("state_reader", "mock"))
    bridge_cfg = dict(real_cfg.get("bridge", {}) or {})
    validate_data_side_for_bridge_tcp(
        resolved_data_side,
        backend_mode=backend_mode,
        state_reader_mode=state_reader_mode,
        bridge_host=str(bridge_cfg.get("host", "127.0.0.1")),
    )
    sync_max_slop_ns = int(float(sync_cfg.get("max_observation_skew_ms", 40.0)) * 1_000_000)
    camera_names: list[str] = list(task_cfg.get("camera_names", ["fpv"]))
    record_config_yaml = _dump_yaml_config(cfg)

    from testbed.backends.real.backend import RealExcavatorBackend
    from testbed.data.recorder import EpisodeRecorder
    from testbed.runtime.guard import ActionGuard

    bridge_client = _build_bridge_client(real_cfg, backend_mode, state_reader_mode)

    backend = RealExcavatorBackend(
        controller_mode=backend_mode,
        state_reader_mode=state_reader_mode,
        bridge_client=bridge_client,
        sync_max_slop_ns=sync_max_slop_ns,
        control_hz=control_hz,
        image_width=int(real_cfg.get("image_width", 160)),
        image_height=int(real_cfg.get("image_height", 120)),
        mock_velocity_scale_rad_s=float(real_cfg.get("mock_velocity_scale_rad_s", 0.5)),
    )
    action_source = _build_action_source(input_device, teleop_cfg, dt=dt)
    guard = ActionGuard(
        action_clip=safety_cfg.get("action_clip", 0.20),
        max_delta=safety_cfg.get("max_delta_per_step", 0.02),
        sensor_timeout_s=safety_cfg.get("sensor_timeout_s", 0.20),
    )
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
        dt=dt,
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

    abort = False

    def _sigint(_sig, _frame) -> None:
        nonlocal abort
        abort = True
        log.warning("Ctrl+C received - will save the current partial episode if it has data.")

    signal.signal(signal.SIGINT, _sigint)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    episode_idx = _next_episode_idx(dataset_dir)
    saved = 0

    try:
        while saved < num_episodes and not abort:
            ep_seed = seed if seed >= 0 else int(time.time()) % (2**31)
            meta = dict(base_meta)
            meta[ATTR_SEED] = ep_seed
            meta[ATTR_EPISODE_ID] = f"episode_{episode_idx}"
            recorder = EpisodeRecorder(
                output_dir=dataset_dir,
                episode_idx=episode_idx,
                metadata=meta,
                camera_names=camera_names,
            )

            log.info("Episode %d starts by operator/CLI control; no hardware reset is issued.", episode_idx)
            ts = backend.start_episode(seed=ep_seed)
            action_source.reset()
            guard.reset()
            discard = False

            for local_step in range(max_steps):
                if abort:
                    break

                discard, quit_now = _check_pygame_events(enabled=input_device != "zero")
                if quit_now:
                    abort = True
                    break
                if discard:
                    log.info("Episode discarded by user.")
                    break

                obs = ts.observation
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
                safety_state = dict(obs.get("safety_state", {}))
                sensor_age_s = _sensor_age_s(obs)
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

                extras = getattr(action_info, "extras", {}) or {}
                toggle_mask = int(extras.get("toggle_mask", 0) or 0)
                if toggle_mask:
                    backend.apply_status_toggle_mask(toggle_mask)

                action_send_ns = time.time_ns()
                ts_next = backend.step(safe_action)
                control_result = dict(ts_next.info.get("control_result", {}))
                recorder.record(
                    obs=obs,
                    action=safe_action,
                    reward=0.0,
                    step_id=int(obs.get("step_id", local_step)),
                    step_ns=action_send_ns,
                    action_src_type=action_info.source_type,
                    action_src_id=action_info.source_id,
                    diagnostics=_build_step_diagnostics(
                        obs=obs,
                        raw_action=raw_action,
                        safe_action=safe_action,
                        action_info=action_info,
                        action_sample_timestamp_ns=action_sample_ns,
                        action_send_timestamp_ns=action_send_ns,
                        guard=guard,
                        control_result=control_result,
                    ),
                )
                ts = ts_next
                _sleep_to_rate(control_hz)

            if not discard and len(recorder) > 0:
                path = recorder.save(success=False)
                log.info("Saved real v1 episode: %d steps -> %s", len(recorder), path)
                saved += 1
                episode_idx += 1
            elif discard:
                log.info("Discarded current partial episode; episode index is unchanged.")
    finally:
        backend.close()
        action_source.close()

    log.info("Real v1 recording complete: %d / %d episode(s) saved.", saved, num_episodes)


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
        from testbed.actions.base import ActionInfo

        return (
            np.zeros(4, dtype=np.float32),
            ActionInfo(source_type="teleop", source_id="zero"),
        )

    def close(self) -> None:
        pass


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
) -> dict[str, Any]:
    commanded_action = control_result.get("commanded_action")
    if commanded_action is None:
        commanded_action = safe_action
    image_timestamps = obs.get("image_timestamp_ns") or {}
    if not isinstance(image_timestamps, dict):
        image_timestamps = {}
    primary_image_ts = _primary_image_timestamp_ns(image_timestamps)
    extras = getattr(action_info, "extras", {}) or {}
    diagnostics: dict[str, Any] = {
        "raw_action": np.asarray(raw_action, dtype=np.float32),
        "toggle_mask": int(extras.get("toggle_mask", 0) or 0),
        "status11": np.asarray(extras.get("status11", []), dtype=np.int32),
        "guard_triggered": int(guard.last_info.triggered),
        "guard_reason": ",".join(guard.last_info.reasons),
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
    for camera_name, timestamp_ns in image_timestamps.items():
        diagnostics[f"image_timestamp_ns_{_sanitize_key(camera_name)}"] = _int_timestamp(
            timestamp_ns
        )
    return diagnostics


def _action_sample_timestamp_ns(action_info) -> int:
    extras = getattr(action_info, "extras", {}) or {}
    return _int_timestamp(extras.get("action_timestamp_ns"), default=time.time_ns())


def _primary_image_timestamp_ns(image_timestamps: dict[str, Any]) -> int:
    if "fpv" in image_timestamps:
        return _int_timestamp(image_timestamps["fpv"])
    for _name, timestamp_ns in sorted(image_timestamps.items()):
        return _int_timestamp(timestamp_ns)
    return 0


def _int_timestamp(value: Any, *, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _sanitize_key(value: Any) -> str:
    return "".join(ch if str(ch).isalnum() or ch == "_" else "_" for ch in str(value))


def _sensor_age_s(obs: dict[str, Any]) -> float | None:
    timestamp_ns = obs.get("sensor_timestamp_ns")
    if timestamp_ns is None:
        return None
    return max(0.0, (time.time_ns() - int(timestamp_ns)) * 1e-9)


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


def _sleep_to_rate(control_hz: float) -> None:
    global _last_step_time
    target_dt = 1.0 / float(control_hz)
    now = time.perf_counter()
    elapsed = now - _last_step_time
    if elapsed < target_dt:
        time.sleep(target_dt - elapsed)
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
    dt: float,
    sync_cfg: dict[str, Any],
    video_cfg: dict[str, Any],
    data_side: str | None = None,
) -> dict[str, Any]:
    camera_width = int(real_cfg.get("image_width", 160))
    camera_height = int(real_cfg.get("image_height", 120))
    metadata: dict[str, Any] = {
        ATTR_TASK_NAME: task_cfg.get("task_name", "real_excavation_teleop_v1"),
        ATTR_IS_REAL: True,
        ATTR_PLATFORM: DEFAULT_PLATFORM,
        ATTR_CONTROL_HZ: int(round(control_hz)),
        ATTR_DT: float(dt),
        ATTR_ACTION_SEMANTICS: "normalized_teleop_cmd_v1",
        ATTR_CAMERA_NAMES: ",".join(camera_names),
        ATTR_IMAGE_FORMAT: "raw_rgb",
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
        ATTR_CAMERA_FPS: float(control_hz),
        ATTR_CAMERA_ROW_ORDER: "top_to_bottom",
        "real_backend": str(real_cfg.get("backend", "mock")),
        "real_state_reader": str(real_cfg.get("state_reader", "mock")),
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
        "oem_remote_required": int(
            bool(teleop_cfg.get("oem_remote", {}).get("required", False))
        ),
        "safety_deadman_enabled": int(bool(safety_cfg.get("deadman_enabled", True))),
        "safety_estop_enabled": int(bool(safety_cfg.get("estop_enabled", True))),
        "safety_manual_override_enabled": int(
            bool(safety_cfg.get("manual_override_enabled", True))
        ),
        "safety_action_clip": float(safety_cfg.get("action_clip", 0.20)),
        "safety_max_delta_per_step": float(safety_cfg.get("max_delta_per_step", 0.02)),
        "safety_sensor_timeout_s": float(safety_cfg.get("sensor_timeout_s", 0.20)),
    }

    metadata_cfg = teleop_cfg.get("metadata", {})
    if metadata_cfg.get("operator_id"):
        metadata[ATTR_OPERATOR_ID] = str(metadata_cfg["operator_id"])
    if metadata_cfg.get("session_id"):
        metadata[ATTR_SESSION_ID] = str(metadata_cfg["session_id"])
    if metadata_cfg.get("notes"):
        metadata[ATTR_NOTES] = str(metadata_cfg["notes"])
    return metadata


if __name__ == "__main__":
    main()
