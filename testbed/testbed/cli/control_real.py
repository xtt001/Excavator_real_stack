"""
tb-control-real - Drive the real excavator without writing HDF5 episodes.

This command shares the same bridge, action source, action guard, and optional
fixed-rate action pump as tb-record-real. It is intended for live joystick
bring-up/control when recording should stay off.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from testbed.cli.record_real import (
    _action_control_flags,
    _build_action_source,
    _build_bridge_client,
    _check_pygame_events,
    _load_yaml_config,
    _sensor_age_s,
    _sleep_to_rate,
)

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="tb-control-real",
        description=(
            "Control the real excavator through the bridge without recording HDF5."
        ),
    )
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=["mock", "bridge_mock", "bridge_tcp"],
        default="bridge_tcp",
        help="Control backend. Real machine control uses bridge_tcp.",
    )
    parser.add_argument(
        "--state-reader",
        choices=["mock", "bridge_mock", "bridge_tcp"],
        default="bridge_tcp",
        help="State reader. Real machine telemetry uses bridge_tcp.",
    )
    parser.add_argument(
        "--data-side",
        choices=["host", "slave"],
        default="host",
        help="Use host/slave bridge defaults from the config. No HDF5 is written.",
    )
    parser.add_argument(
        "--input",
        choices=["joystick", "keyboard", "oem_remote", "remote", "zero"],
        default=None,
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Stop after this many seconds. 0 means run until Ctrl+C/q.",
    )
    parser.add_argument(
        "--loop-hz",
        type=float,
        default=None,
        help=(
            "Main observe/action sampling loop rate. "
            "Defaults to task.record_hz/control_hz."
        ),
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
        "--confirm-real-control",
        action="store_true",
        help=(
            "Required when --backend bridge_tcp, because commands are sent "
            "to the machine."
        ),
    )
    parser.add_argument(
        "--log-interval-s",
        type=float,
        default=1.0,
        help="Interval for live action/bridge status logs.",
    )
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config)
    real_cfg = cfg.setdefault("real", {})
    teleop_cfg = cfg.setdefault("teleop", {})
    task_cfg = cfg.setdefault("task", {})
    safety_cfg = cfg.setdefault("safety", {})
    sync_cfg = cfg.setdefault("sync", {})

    from testbed.cli.data_side import (
        apply_data_side_config,
        validate_data_side_for_bridge_tcp,
    )

    apply_data_side_config(
        cfg,
        data_side=args.data_side,
        cli_bridge_host=args.bridge_host,
        cli_bridge_port=args.bridge_port,
        log_dataset=False,
    )

    real_cfg["backend"] = args.backend
    real_cfg["state_reader"] = args.state_reader
    if args.input is not None:
        teleop_cfg["input"] = args.input
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

    backend_mode = str(real_cfg.get("backend", "bridge_tcp"))
    state_reader_mode = str(real_cfg.get("state_reader", "bridge_tcp"))
    bridge_cfg = dict(real_cfg.get("bridge", {}) or {})
    bridge_host = str(bridge_cfg.get("host", "127.0.0.1"))
    bridge_port = int(bridge_cfg.get("port", 0))

    if backend_mode == "bridge_tcp" and not args.confirm_real_control:
        parser.error("--confirm-real-control is required when --backend bridge_tcp")

    validate_data_side_for_bridge_tcp(
        str(real_cfg.get("data_side", args.data_side)),
        backend_mode=backend_mode,
        state_reader_mode=state_reader_mode,
        bridge_host=bridge_host,
    )

    control_hz = float(task_cfg.get("control_hz", real_cfg.get("control_hz", 50)))
    loop_hz = float(args.loop_hz or task_cfg.get("record_hz", control_hz))
    dt = float(task_cfg.get("dt", 1.0 / loop_hz))
    input_device = str(teleop_cfg.get("input", "joystick"))
    max_steps = args.max_steps
    sync_max_slop_ns = int(
        float(sync_cfg.get("max_observation_skew_ms", 40.0)) * 1_000_000
    )

    from testbed.backends.real.backend import RealExcavatorBackend
    from testbed.runtime.guard import ActionGuard

    pump_cfg = dict(real_cfg.get("control_pump", {}) or {})
    control_pump_enabled = (
        bool(pump_cfg.get("enabled", False)) and backend_mode == "bridge_tcp"
    )
    backend_controller_mode = "noop" if control_pump_enabled else backend_mode
    bridge_client = _build_bridge_client(
        real_cfg,
        backend_controller_mode,
        state_reader_mode,
    )
    control_pump = None
    if control_pump_enabled:
        from testbed.backends.real.action_pump import RealActionPump
        from testbed.backends.real.bridge import BridgeLowLevelController

        control_pump_client = _build_bridge_client(real_cfg, "bridge_tcp", "mock")
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

    backend = RealExcavatorBackend(
        controller_mode=backend_controller_mode,
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

    abort = False

    def _sigint(_sig, _frame) -> None:
        nonlocal abort
        abort = True
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _sigint)

    log.info(
        "Real control starts: backend=%s state_reader=%s input=%s bridge=%s:%d "
        "loop_hz=%.1f pump=%s duration_s=%.1f max_steps=%s",
        backend_mode,
        state_reader_mode,
        input_device,
        bridge_host,
        bridge_port,
        loop_hz,
        "on" if control_pump is not None else "off",
        float(args.duration_s),
        str(max_steps or "unlimited"),
    )

    if control_pump is not None:
        control_pump.start()

    step = 0
    start_s = time.monotonic()
    last_log_s = 0.0
    last_action = np.zeros(4, dtype=np.float32)

    try:
        seed_raw = int(task_cfg.get("seed", -1))
        ts = backend.start_episode(seed=seed_raw if seed_raw >= 0 else None)
        action_source.reset()
        guard.reset()
        while not abort:
            if max_steps is not None and step >= max_steps:
                break
            elapsed_s = time.monotonic() - start_s
            if args.duration_s > 0.0 and elapsed_s >= args.duration_s:
                break

            discard_now, quit_now = _check_pygame_events(
                enabled=input_device not in {"zero", "remote"}
            )
            if quit_now:
                log.info("Quit requested from pygame/q.")
                break
            if discard_now:
                log.info("Stop requested from pygame/d.")
                break

            obs = ts.observation
            raw_action, action_info = action_source.next_action(obs)
            reset_req, discard_req, quit_req = _action_control_flags(action_info)
            if reset_req or discard_req or quit_req:
                log.info("Stop requested from action source.")
                break

            safety_state = dict(obs.get("safety_state", {}))
            safe_action, _triggered = guard.check(
                raw_action,
                obs.get("qpos"),
                deadman_pressed=bool(safety_state.get("deadman_pressed", True)),
                estop_active=bool(safety_state.get("estop_active", False)),
                manual_override_active=bool(
                    safety_state.get("manual_override_active", False)
                ),
                sensor_stale=bool(safety_state.get("sensor_stale", False)),
                sensor_age_s=_sensor_age_s(obs),
            )

            extras = getattr(action_info, "extras", {}) or {}
            toggle_mask = int(extras.get("toggle_mask", 0) or 0)
            if toggle_mask:
                if control_pump is not None:
                    control_pump.apply_status_toggle_mask(toggle_mask)
                else:
                    backend.apply_status_toggle_mask(toggle_mask)

            action_send_ns = time.time_ns()
            if control_pump is not None:
                pump_result = control_pump.update_action(safe_action, state=obs)
                action_send_ns = int(
                    pump_result.controller_timestamp_ns or action_send_ns
                )
                ts = backend.observe(
                    action_timestamp_ns=action_send_ns,
                    result=pump_result,
                )
            else:
                ts = backend.step(safe_action)

            control_result = dict(ts.info.get("control_result", {}))
            last_action = np.asarray(safe_action, dtype=np.float32).copy()
            now_s = time.monotonic()
            if args.log_interval_s > 0 and now_s - last_log_s >= args.log_interval_s:
                sensor_age = _sensor_age_s(ts.observation)
                sensor_age_ms = -1.0 if sensor_age is None else sensor_age * 1000.0
                log.info(
                    "step=%d action=[%.3f %.3f %.3f %.3f] ack=%s fault=%s "
                    "guard=%s sensor_age_ms=%.1f",
                    step,
                    float(last_action[0]),
                    float(last_action[1]),
                    float(last_action[2]),
                    float(last_action[3]),
                    bool(control_result.get("ack", False)),
                    str(control_result.get("fault_code", "")),
                    (
                        ",".join(guard.last_info.reasons)
                        if guard.last_info.triggered
                        else ""
                    ),
                    sensor_age_ms,
                )
                last_log_s = now_s

            step += 1
            _sleep_to_rate(loop_hz, should_stop=lambda: abort)
    except KeyboardInterrupt:
        abort = True
        log.info("Ctrl+C received; stopping control and sending zero if configured.")
    finally:
        _shutdown(
            backend=backend,
            action_source=action_source,
            control_pump=control_pump,
            send_zero_without_pump=backend_controller_mode != "noop",
        )

    log.info(
        "Real control stopped after %d step(s). Last safe action="
        "[%.3f %.3f %.3f %.3f].",
        step,
        float(last_action[0]),
        float(last_action[1]),
        float(last_action[2]),
        float(last_action[3]),
    )
    if abort:
        sys.exit(130)


def _shutdown(
    *,
    backend: Any,
    action_source: Any,
    control_pump: Any,
    send_zero_without_pump: bool,
) -> None:
    try:
        action_source.close()
    except Exception:
        pass
    if control_pump is not None:
        try:
            control_pump.stop()
        except Exception:
            pass
    elif send_zero_without_pump:
        try:
            backend.step(np.zeros(4, dtype=np.float32))
        except Exception as exc:
            log.warning("zero command on shutdown failed: %s", exc)
    try:
        backend.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
