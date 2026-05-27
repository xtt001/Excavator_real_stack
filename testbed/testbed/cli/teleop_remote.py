"""Host-side remote teleop sender for slave-side real recording."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from testbed.actions.base import ActionInfo, ActionSource
from testbed.actions.gamepad import JoystickActionSource
from testbed.actions.remote import (
    DEFAULT_REMOTE_ACTION_PORT,
    RemoteActionClient,
)

log = logging.getLogger(__name__)


class _ZeroActionSource(ActionSource):
    def reset(self) -> None:
        pass

    def next_action(self, obs: dict) -> tuple[np.ndarray, ActionInfo]:
        return np.zeros(4, dtype=np.float32), ActionInfo(
            source_type="teleop",
            source_id="zero",
            extras={},
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="tb-teleop-remote",
        description="Send host joystick actions to a slave RemoteActionSource.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("testbed/testbed/configs/teleop_real_v1.yaml"),
    )
    parser.add_argument("--host", required=True, help="Slave host/IP.")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--input",
        choices=["joystick", "zero"],
        default=None,
        help="Host-side input device. Defaults to teleop.input unless it is remote.",
    )
    parser.add_argument("--rate-hz", type=float, default=None)
    parser.add_argument("--connect-timeout", type=float, default=2.0)
    parser.add_argument("--source-id", type=str, default=None)
    parser.add_argument(
        "--record-start-button",
        type=int,
        default=None,
        help=(
            "Physical one-based joystick button number that sends record_start_requested. "
            "For left physical button 2, pass 2; this maps to pygame button index 1."
        ),
    )
    parser.add_argument(
        "--record-start-button-index",
        type=int,
        default=None,
        help=(
            "Low-level pygame zero-based joystick button index. "
            "Use only when debugging pygame button numbers."
        ),
    )
    parser.add_argument(
        "--record-start-physical-button",
        type=int,
        default=None,
        help=(
            "Physical one-based button number that sends record_start_requested. "
            "For left physical button 2, pass 2; this maps to pygame button index 1."
        ),
    )
    parser.add_argument(
        "--record-start-joystick-id",
        type=int,
        default=None,
        help=(
            "Joystick device id that owns the record-start button. "
            "Sets joystick.button_joystick_ids to this single device."
        ),
    )
    parser.add_argument(
        "--go-home-button",
        type=int,
        default=None,
        help=(
            "Physical one-based joystick button number that sends go_home_requested. "
            "This asks the slave receiver to run the near-home go-home controller."
        ),
    )
    parser.add_argument(
        "--go-home-button-index",
        type=int,
        default=None,
        help="Low-level pygame zero-based joystick button index for go_home_requested.",
    )
    parser.add_argument(
        "--status-button-device",
        type=int,
        default=None,
        help="Joystick device id that owns machine status buttons.",
    )
    parser.add_argument(
        "--confirm-remote-control",
        action="store_true",
        help="Required because this stream can drive the real machine via the slave.",
    )
    parser.add_argument(
        "--log-interval-s",
        type=float,
        default=1.0,
        help="Interval for action sender status logs.",
    )
    args = parser.parse_args()

    if not args.confirm_remote_control:
        parser.error("--confirm-remote-control is required")

    cfg = _load_yaml_config(args.config)
    teleop_cfg = cfg.setdefault("teleop", {})
    task_cfg = cfg.setdefault("task", {})
    remote_cfg = dict(teleop_cfg.get("remote", {}) or {})
    joystick_cfg = teleop_cfg.setdefault("joystick", {})
    if args.status_button_device is not None:
        joystick_cfg["status_button_device"] = int(args.status_button_device)
    record_start_button_args = [
        args.record_start_button is not None,
        args.record_start_button_index is not None,
        args.record_start_physical_button is not None,
    ]
    if sum(record_start_button_args) > 1:
        parser.error(
            "--record-start-button, --record-start-button-index, and "
            "--record-start-physical-button are mutually exclusive"
        )
    if args.record_start_button is not None or args.record_start_physical_button is not None:
        physical_button = (
            args.record_start_button
            if args.record_start_button is not None
            else args.record_start_physical_button
        )
        if physical_button < 1:
            parser.error("--record-start-button must be >= 1")
        joystick_cfg["record_start_button"] = int(physical_button) - 1
    elif args.record_start_button_index is not None:
        joystick_cfg["record_start_button"] = int(args.record_start_button_index)
    if args.record_start_joystick_id is not None:
        joystick_cfg["button_joystick_ids"] = [int(args.record_start_joystick_id)]
    if args.go_home_button is not None and args.go_home_button_index is not None:
        parser.error("--go-home-button and --go-home-button-index are mutually exclusive")
    if args.go_home_button is not None:
        if args.go_home_button < 1:
            parser.error("--go-home-button must be >= 1")
        joystick_cfg["go_home_button"] = int(args.go_home_button) - 1
    elif args.go_home_button_index is not None:
        joystick_cfg["go_home_button"] = int(args.go_home_button_index)
    input_device = args.input or str(teleop_cfg.get("input", "joystick"))
    if input_device == "remote":
        input_device = "joystick"
    port = int(args.port or remote_cfg.get("port", DEFAULT_REMOTE_ACTION_PORT))
    rate_hz = float(
        args.rate_hz
        or remote_cfg.get("rate_hz")
        or task_cfg.get("control_hz", 50.0)
    )
    source_id = str(args.source_id or remote_cfg.get("source_id", "host_joystick"))

    action_source = _build_host_action_source(
        input_device,
        teleop_cfg,
        default_dt=1.0 / rate_hz,
    )
    client = RemoteActionClient(
        host=args.host,
        port=port,
        timeout_s=float(args.connect_timeout),
    )
    abort = False

    def _sigint(_sig, _frame) -> None:
        nonlocal abort
        abort = True
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _sigint)
    log.info(
        "Remote teleop sender starts: target=%s:%d input=%s rate_hz=%.1f source_id=%s",
        args.host,
        port,
        input_device,
        rate_hz,
        source_id,
    )
    if input_device == "joystick":
        record_start_index = joystick_cfg.get("record_start_button")
        record_start_physical = (
            None if record_start_index is None else int(record_start_index) + 1
        )
        log.info(
            "Joystick buttons: status_button_device=%s record_start_button_index=%s "
            "record_start_physical_button=%s button_joystick_ids=%s",
            joystick_cfg.get("status_button_device", 0),
            record_start_index,
            record_start_physical,
            joystick_cfg.get("button_joystick_ids"),
        )

    seq = 0
    last_log_s = 0.0
    try:
        client.connect()
        action_source.reset()
        while not abort:
            action, info = action_source.next_action({})
            extras = getattr(info, "extras", {}) or {}
            host_sample_time_ns = time.time_ns()
            sender_source_id = (
                source_id if source_id else str(getattr(info, "source_id", "remote"))
            )
            client.send_update(
                seq=seq,
                action=action,
                host_sample_time_ns=host_sample_time_ns,
                source_id=sender_source_id,
                toggle_mask=int(extras.get("toggle_mask", 0) or 0),
                reset_requested=bool(extras.get("reset_requested", False)),
                discard_requested=bool(extras.get("discard_requested", False)),
                quit_requested=bool(extras.get("quit_requested", False)),
                record_start_requested=bool(extras.get("record_start_requested", False)),
                go_home_requested=bool(extras.get("go_home_requested", False)),
            )
            event_flags = {
                "toggle_mask": int(extras.get("toggle_mask", 0) or 0),
                "record_start": bool(extras.get("record_start_requested", False)),
                "go_home": bool(extras.get("go_home_requested", False)),
                "reset": bool(extras.get("reset_requested", False)),
                "discard": bool(extras.get("discard_requested", False)),
                "quit": bool(extras.get("quit_requested", False)),
            }
            if any(event_flags.values()):
                log.info(
                    "remote_event seq=%d toggle_mask=%d record_start=%s go_home=%s reset=%s discard=%s quit=%s",
                    seq,
                    event_flags["toggle_mask"],
                    event_flags["record_start"],
                    event_flags["go_home"],
                    event_flags["reset"],
                    event_flags["discard"],
                    event_flags["quit"],
                )
            now_s = time.monotonic()
            if now_s - last_log_s >= float(args.log_interval_s):
                last_log_s = now_s
                log.info(
                    "remote_action seq=%d action=%s toggle_mask=%d",
                    seq,
                    _format_action(action),
                    int(extras.get("toggle_mask", 0) or 0),
                )
            seq += 1
            if bool(extras.get("quit_requested", False)):
                break
            _sleep_to_rate(rate_hz)
    except KeyboardInterrupt:
        abort = True
    finally:
        _send_stop(client, seq=seq, source_id=source_id)
        action_source.close()
        client.close()
    if abort:
        log.info("Remote teleop sender stopped.")


def _load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must decode to a mapping: {path}")
    return data


def _build_host_action_source(
    input_device: str,
    teleop_cfg: dict[str, Any],
    *,
    default_dt: float,
) -> ActionSource:
    if input_device == "joystick":
        return JoystickActionSource.from_config(
            teleop_cfg.get("joystick", {}),
            default_dt=default_dt,
        )
    if input_device == "zero":
        return _ZeroActionSource()
    raise ValueError(f"Unsupported remote teleop input {input_device!r}.")


def _send_stop(client: RemoteActionClient, *, seq: int, source_id: str) -> None:
    try:
        client.send_update(
            seq=seq,
            action=np.zeros(4, dtype=np.float32),
            host_sample_time_ns=time.time_ns(),
            source_id=source_id,
            quit_requested=True,
            go_home_requested=False,
        )
    except Exception:
        pass


def _sleep_to_rate(rate_hz: float) -> None:
    target_dt = 1.0 / float(rate_hz)
    time.sleep(max(0.0, target_dt))


def _format_action(action: Any) -> str:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{float(value):+.3f}" for value in values) + "]"


if __name__ == "__main__":
    main()
