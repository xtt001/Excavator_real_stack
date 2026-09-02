"""Host-side remote teleop sender for slave-side real recording."""

from __future__ import annotations

import argparse
import logging
import shutil
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from testbed.actions.base import ActionInfo, ActionSource
from testbed.actions.gamepad import JoystickActionSource
from testbed.actions.remote import (
    DEFAULT_REMOTE_ACTION_PORT,
    RemoteActionClient,
)
from testbed.config_loader import load_yaml_config
from testbed.host_status import (
    DEFAULT_HOST_STATUS_HOST,
    DEFAULT_HOST_STATUS_PORT,
    LocalHostStatusPublisher,
    build_host_status_snapshot,
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
        "--policy-start-button",
        type=int,
        default=None,
        help=(
            "Physical one-based joystick button number that sends policy_start_requested. "
            "For left physical button 4, pass 4; this maps to pygame button index 3."
        ),
    )
    parser.add_argument(
        "--mark-button",
        type=int,
        default=None,
        help=(
            "Physical one-based joystick button number that sends mark_requested. "
            "Task-state-v2 uses this as the explicit work/return advance event."
        ),
    )
    parser.add_argument(
        "--go-home-button-index",
        type=int,
        default=None,
        help="Low-level pygame zero-based joystick button index for go_home_requested.",
    )
    parser.add_argument(
        "--policy-start-button-index",
        type=int,
        default=None,
        help="Low-level pygame zero-based joystick button index for policy_start_requested.",
    )
    parser.add_argument(
        "--mark-button-index",
        type=int,
        default=None,
        help="Low-level pygame zero-based joystick button index for mark_requested.",
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
        help="Interval for action sender status logs when the fixed monitor is disabled.",
    )
    monitor_group = parser.add_mutually_exclusive_group()
    monitor_group.add_argument(
        "--monitor",
        dest="monitor",
        action="store_true",
        default=None,
        help="Show a fixed top-like sender status monitor. Defaults to on for TTY stdout.",
    )
    monitor_group.add_argument(
        "--no-monitor",
        dest="monitor",
        action="store_false",
        help="Disable the fixed monitor and use periodic log lines instead.",
    )
    parser.add_argument(
        "--monitor-interval-s",
        type=float,
        default=0.1,
        help="Minimum redraw interval for the fixed sender status monitor.",
    )
    parser.add_argument(
        "--local-status-host",
        default=DEFAULT_HOST_STATUS_HOST,
        help="Local read-only dashboard UDP destination. Default: 127.0.0.1.",
    )
    parser.add_argument(
        "--local-status-port",
        type=int,
        default=DEFAULT_HOST_STATUS_PORT,
        help="Local read-only dashboard UDP port. Default: 8781.",
    )
    parser.add_argument(
        "--local-status-hz",
        type=float,
        default=10.0,
        help="Maximum localhost dashboard telemetry rate. Default: 10 Hz.",
    )
    parser.add_argument(
        "--no-local-status",
        action="store_true",
        help="Disable the localhost read-only dashboard telemetry mirror.",
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
    if (
        args.record_start_button is not None
        or args.record_start_physical_button is not None
    ):
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
        parser.error(
            "--go-home-button and --go-home-button-index are mutually exclusive"
        )
    if args.go_home_button is not None:
        if args.go_home_button < 1:
            parser.error("--go-home-button must be >= 1")
        joystick_cfg["go_home_button"] = int(args.go_home_button) - 1
    elif args.go_home_button_index is not None:
        joystick_cfg["go_home_button"] = int(args.go_home_button_index)
    if (
        args.policy_start_button is not None
        and args.policy_start_button_index is not None
    ):
        parser.error(
            "--policy-start-button and --policy-start-button-index are mutually exclusive"
        )
    if args.policy_start_button is not None:
        if args.policy_start_button < 1:
            parser.error("--policy-start-button must be >= 1")
        joystick_cfg["policy_start_button"] = int(args.policy_start_button) - 1
    elif args.policy_start_button_index is not None:
        joystick_cfg["policy_start_button"] = int(args.policy_start_button_index)
    if args.mark_button is not None and args.mark_button_index is not None:
        parser.error("--mark-button and --mark-button-index are mutually exclusive")
    if args.mark_button is not None:
        if args.mark_button < 1:
            parser.error("--mark-button must be >= 1")
        joystick_cfg["mark_button"] = int(args.mark_button) - 1
    elif args.mark_button_index is not None:
        joystick_cfg["mark_button"] = int(args.mark_button_index)
    input_device = args.input or str(teleop_cfg.get("input", "joystick"))
    if input_device == "remote":
        input_device = "joystick"
    port = int(args.port or remote_cfg.get("port", DEFAULT_REMOTE_ACTION_PORT))
    rate_hz = float(
        args.rate_hz or remote_cfg.get("rate_hz") or task_cfg.get("control_hz", 50.0)
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
        policy_start_index = joystick_cfg.get("policy_start_button")
        policy_start_physical = (
            None if policy_start_index is None else int(policy_start_index) + 1
        )
        mark_index = joystick_cfg.get("mark_button")
        mark_physical = None if mark_index is None else int(mark_index) + 1
        log.info(
            "Joystick buttons: status_button_device=%s record_start_button_index=%s "
            "record_start_physical_button=%s policy_start_button_index=%s "
            "policy_start_physical_button=%s mark_button_index=%s "
            "mark_physical_button=%s button_joystick_ids=%s",
            joystick_cfg.get("status_button_device", 0),
            record_start_index,
            record_start_physical,
            policy_start_index,
            policy_start_physical,
            mark_index,
            mark_physical,
            joystick_cfg.get("button_joystick_ids"),
        )

    seq = 0
    last_log_s = 0.0
    monitor_enabled = (
        sys.stdout.isatty() if args.monitor is None else bool(args.monitor)
    )
    monitor = _RemoteTeleopMonitor(
        enabled=monitor_enabled,
        target=f"{args.host}:{port}",
        input_device=input_device,
        rate_hz=rate_hz,
        source_id=source_id,
        min_interval_s=float(args.monitor_interval_s),
    )
    local_status = LocalHostStatusPublisher(
        host=str(args.local_status_host),
        port=int(args.local_status_port),
        max_hz=float(args.local_status_hz),
        enabled=not bool(args.no_local_status),
    )
    try:
        client.connect()
        action_source.reset()
        monitor.start()
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
                record_start_requested=bool(
                    extras.get("record_start_requested", False)
                ),
                mark_requested=bool(extras.get("mark_requested", False)),
                policy_start_requested=bool(
                    extras.get("policy_start_requested", False)
                ),
                go_home_requested=bool(extras.get("go_home_requested", False)),
            )
            event_flags = {
                "toggle_mask": int(extras.get("toggle_mask", 0) or 0),
                "record_start": bool(extras.get("record_start_requested", False)),
                "mark": bool(extras.get("mark_requested", False)),
                "policy_start": bool(extras.get("policy_start_requested", False)),
                "go_home": bool(extras.get("go_home_requested", False)),
                "reset": bool(extras.get("reset_requested", False)),
                "discard": bool(extras.get("discard_requested", False)),
                "quit": bool(extras.get("quit_requested", False)),
            }
            if any(event_flags.values()):
                if not monitor.enabled:
                    log.info(
                        "remote_event seq=%d toggle_mask=%d record_start=%s mark=%s policy_start=%s go_home=%s reset=%s discard=%s quit=%s",
                        seq,
                        event_flags["toggle_mask"],
                        event_flags["record_start"],
                        event_flags["mark"],
                        event_flags["policy_start"],
                        event_flags["go_home"],
                        event_flags["reset"],
                        event_flags["discard"],
                        event_flags["quit"],
                    )
            now_s = time.monotonic()
            receiver_status = client.latest_status()
            local_status.publish(
                build_host_status_snapshot(
                    seq=seq,
                    target=f"{args.host}:{port}",
                    configured_hz=rate_hz,
                    input_device=input_device,
                    source_id=sender_source_id,
                    action=action,
                    action_latency_ms=getattr(info, "latency_ms", None),
                    extras=extras,
                    event_flags=event_flags,
                    receiver_status=receiver_status,
                ),
                force=any(event_flags.values()),
            )
            if monitor.enabled:
                monitor.update(
                    seq=seq,
                    action=action,
                    info=info,
                    extras=extras,
                    event_flags=event_flags,
                    receiver_status=receiver_status,
                )
            elif float(args.log_interval_s) > 0.0 and now_s - last_log_s >= float(
                args.log_interval_s
            ):
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
        monitor.finish()
        local_status.close()
        _send_stop(client, seq=seq, source_id=source_id)
        action_source.close()
        client.close()
    if abort:
        log.info("Remote teleop sender stopped.")


def _load_yaml_config(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


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


class _RemoteTeleopMonitor:
    def __init__(
        self,
        *,
        enabled: bool,
        target: str,
        input_device: str,
        rate_hz: float,
        source_id: str,
        min_interval_s: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.target = str(target)
        self.input_device = str(input_device)
        self.rate_hz = float(rate_hz)
        self.source_id = str(source_id)
        self.min_interval_s = max(0.02, float(min_interval_s))
        self._started = False
        self._start_s = 0.0
        self._last_update_s: float | None = None
        self._last_draw_s = 0.0
        self._hz_ema: float | None = None
        self._event_history: deque[str] = deque(maxlen=8)
        self._event_counts = {
            "toggle": 0,
            "record_start": 0,
            "mark": 0,
            "policy_start": 0,
            "go_home": 0,
            "reset": 0,
            "discard": 0,
            "quit": 0,
        }
        self._last_event_by_name: dict[str, tuple[int, float]] = {}
        self._last_receiver_key: tuple[str, str, int, int, int, str, str] | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._started = True
        self._start_s = time.monotonic()
        sys.stdout.write("\033[?25l\033[2J\033[H")
        sys.stdout.flush()

    def update(
        self,
        *,
        seq: int,
        action: Any,
        info: ActionInfo,
        extras: dict[str, Any],
        event_flags: dict[str, Any],
        receiver_status: Any | None,
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
        self._record_events(seq=seq, now_s=now_s, event_flags=event_flags)
        self._record_receiver_status(now_s=now_s, receiver_status=receiver_status)
        if now_s - self._last_draw_s < self.min_interval_s and not any(
            event_flags.values()
        ):
            return
        self._last_draw_s = now_s
        self._draw(
            seq=seq,
            action=action,
            info=info,
            extras=extras,
            now_s=now_s,
            receiver_status=receiver_status,
        )

    def finish(self) -> None:
        if not self.enabled or not self._started:
            return
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()
        self._started = False

    def _record_events(
        self,
        *,
        seq: int,
        now_s: float,
        event_flags: dict[str, Any],
    ) -> None:
        names: list[str] = []
        toggle_mask = int(event_flags.get("toggle_mask", 0) or 0)
        if toggle_mask:
            names.append(f"toggle_mask=0x{toggle_mask:x}")
            self._event_counts["toggle"] += 1
            self._last_event_by_name["toggle"] = (int(seq), now_s)
        for key, label in (
            ("record_start", "record_start_requested"),
            ("mark", "mark_requested"),
            ("policy_start", "policy_start_requested"),
            ("go_home", "go_home_requested"),
            ("reset", "reset_requested"),
            ("discard", "discard_requested"),
            ("quit", "quit_requested"),
        ):
            if bool(event_flags.get(key, False)):
                names.append(label)
                self._event_counts[key] += 1
                self._last_event_by_name[key] = (int(seq), now_s)
        if names:
            stamp = time.strftime("%H:%M:%S")
            self._event_history.appendleft(f"{stamp} seq={int(seq)} " + " ".join(names))

    def _record_receiver_status(
        self,
        *,
        now_s: float,
        receiver_status: Any | None,
    ) -> None:
        if receiver_status is None:
            return
        payload = dict(getattr(receiver_status, "payload", {}) or {})
        control_mode = str(payload.get("control_mode", "") or "")
        if not control_mode:
            control_mode = (
                "model" if int(payload.get("model_control", 0) or 0) else "manual"
            )
        key = (
            str(payload.get("receiver_mode", "")),
            control_mode,
            int(payload.get("recording", 0) or 0),
            int(payload.get("episode_idx", -1) or -1),
            int(payload.get("saved", -1) or -1),
            str(payload.get("go_home_result", "")),
            str(payload.get("message", "")),
        )
        if getattr(self, "_last_receiver_key", None) == key:
            return
        self._last_receiver_key = key
        stamp = time.strftime("%H:%M:%S")
        rec = "yes" if key[2] else "no"
        go_home = key[5] or "-"
        message = f" msg={key[6]}" if key[6] else ""
        self._event_history.appendleft(
            f"{stamp} receiver mode={key[0] or '-'} control={key[1] or '-'} "
            f"rec={rec} episode={key[3]} saved={key[4]} "
            f"go_home={go_home}{message}"
        )

    def _draw(
        self,
        *,
        seq: int,
        action: Any,
        info: ActionInfo,
        extras: dict[str, Any],
        now_s: float,
        receiver_status: Any | None,
    ) -> None:
        width = max(60, _terminal_width())
        hz_text = "-" if self._hz_ema is None else f"{self._hz_ema:.1f}"
        uptime_s = max(0.0, now_s - self._start_s)
        latency = getattr(info, "latency_ms", None)
        latency_text = "-" if latency is None else f"{float(latency):.2f}"
        status11 = extras.get("status11")
        status_text = _format_status_bits(status11)
        toggle_mask = int(extras.get("toggle_mask", 0) or 0)
        record_last = self._format_last_event("record_start", now_s)
        mark_last = self._format_last_event("mark", now_s)
        policy_last = self._format_last_event("policy_start", now_s)
        go_home_last = self._format_last_event("go_home", now_s)
        source = getattr(info, "source_id", "") or self.source_id
        lines = [
            "Remote Teleop Sender Monitor",
            (
                f"target={self.target} input={self.input_device} "
                f"source={self.source_id} configured_hz={self.rate_hz:.1f}"
            ),
            (
                f"seq={int(seq)} measured_hz={hz_text} uptime={uptime_s:.1f}s "
                f"joystick_latency_ms={latency_text}"
            ),
            f"action={_format_action(action)} source_info={source}",
            f"status11={status_text} toggle_mask=0x{toggle_mask:x}",
            _format_receiver_status(receiver_status),
            _format_receiver_policy_status(receiver_status),
            (
                "record_start: "
                f"pulse={_yes_no(extras.get('record_start_requested', False))} "
                f"count={self._event_counts['record_start']} last={record_last}"
            ),
            (
                "TASK ARM/MARK:"
                f"pulse={_yes_no(extras.get('mark_requested', False))} "
                f"count={self._event_counts['mark']} last={mark_last}"
            ),
            (
                "go_home:      "
                f"pulse={_yes_no(extras.get('go_home_requested', False))} "
                f"count={self._event_counts['go_home']} last={go_home_last}"
            ),
            (
                "policy_start: "
                f"pulse={_yes_no(extras.get('policy_start_requested', False))} "
                f"count={self._event_counts['policy_start']} last={policy_last}"
            ),
            (
                "other events: "
                f"toggle={self._event_counts['toggle']} "
                f"reset={self._event_counts['reset']} "
                f"discard={self._event_counts['discard']} "
                f"quit={self._event_counts['quit']}"
            ),
            "",
            "Recent events:",
        ]
        if self._event_history:
            lines.extend(f"  {item}" for item in self._event_history)
        else:
            lines.append("  -")
        rendered = "\n".join(_fit_line(line, width) for line in lines)
        sys.stdout.write("\033[H" + rendered + "\033[J")
        sys.stdout.flush()

    def _format_last_event(self, name: str, now_s: float) -> str:
        item = self._last_event_by_name.get(name)
        if item is None:
            return "-"
        seq, event_s = item
        return f"seq={seq} age={max(0.0, now_s - event_s):.1f}s"


def _terminal_width() -> int:
    try:
        return int(shutil.get_terminal_size((120, 20)).columns)
    except Exception:
        return 120


def _fit_line(text: str, width: int) -> str:
    if len(text) >= width:
        return text[: max(1, width - 1)]
    return text.ljust(width - 1)


def _format_status_bits(status: Any) -> str:
    if status is None:
        return "-"
    try:
        values = [int(bool(value)) for value in status]
    except TypeError:
        return "-"
    return "[" + "".join(str(value) for value in values) + "]"


def _yes_no(value: Any) -> str:
    return "YES" if bool(value) else "no"


def _format_receiver_status(receiver_status: Any | None) -> str:
    if receiver_status is None:
        return "receiver=waiting_for_status"
    payload = dict(getattr(receiver_status, "payload", {}) or {})
    receive_time_ns = int(getattr(receiver_status, "receive_time_ns", 0) or 0)
    age_ms = (
        "-"
        if receive_time_ns <= 0
        else f"{max(0.0, (time.time_ns() - receive_time_ns) / 1_000_000.0):.0f}"
    )
    mode = str(payload.get("receiver_mode", "-") or "-")
    control_mode = str(payload.get("control_mode", "") or "")
    model_control = int(payload.get("model_control", 0) or 0)
    if not control_mode:
        control_mode = "model" if model_control else "manual"
    recording = "yes" if int(payload.get("recording", 0) or 0) else "no"
    episode = int(payload.get("episode_idx", -1) or -1)
    saved = int(payload.get("saved", 0) or 0)
    steps = int(payload.get("record_steps", 0) or 0)
    health = "OK" if int(payload.get("receiver_health_ok", 1) or 0) else "ERR"
    health_error = str(payload.get("receiver_health_error", "") or "-")
    go_home = str(payload.get("go_home_result", "") or "-")
    message = str(payload.get("message", "") or "")
    saved_path = str(payload.get("saved_path", "") or "")
    saved_name = Path(saved_path).name if saved_path else ""
    suffix_parts = []
    if message:
        suffix_parts.append(f"msg={message}")
    if saved_name:
        suffix_parts.append(f"file={saved_name}")
    suffix = "" if not suffix_parts else " " + " ".join(suffix_parts)
    return (
        f"receiver_mode={mode} control={control_mode} "
        f"model_control={'yes' if model_control else 'no'} "
        f"recording={recording} episode={episode} "
        f"steps={steps} saved={saved} go_home={go_home} "
        f"health={health}:{health_error} status_age_ms={age_ms}{suffix}"
    )


def _format_receiver_policy_status(receiver_status: Any | None) -> str:
    if receiver_status is None:
        return "model_report=waiting_for_status"
    payload = dict(getattr(receiver_status, "payload", {}) or {})
    action = _status_vector(payload.get("policy_action"), width=4)
    assisted = _status_vector(payload.get("policy_assisted_action"), width=4)
    commanded = _status_vector(payload.get("commanded_action"), width=4)
    intent = _status_vector(payload.get("policy_intent_probabilities"), width=8)
    cycle_text = ""
    if int(payload.get("scripted_cycle_enabled", 0) or 0):
        goal_side = _format_script_side(payload.get("planner_target_side"))
        ready_side = _format_script_side(payload.get("scripted_cycle_ready_side"))
        cycle_text = (
            "scripted_cycle="
            f"{'active' if int(payload.get('scripted_cycle_active', 0) or 0) else 'waiting'} "
            f"cycle={int(payload.get('planner_cycle_index', -1))} "
            f"goal={goal_side} "
            f"ready={ready_side} "
            "task_state="
            f"{str(payload.get('scripted_cycle_task_state_stage', '-') or '-')} "
            "auto_pending="
            f"{str(payload.get('scripted_cycle_task_auto_pending_event', '-') or '-')} "
            "work_live="
            f"{int(payload.get('scripted_cycle_task_auto_work_liveness', 0) or 0)} "
            "ready_blockers="
            f"{str(payload.get('scripted_cycle_ready_blockers', '-') or '-')} "
            f"excursion={int(payload.get('scripted_cycle_excursion_observed', 0) or 0)} "
            f"review={int(payload.get('scripted_cycle_review_due', 0) or 0)} "
            f"event={str(payload.get('scripted_cycle_event', '-') or '-')} "
            f"fault={str(payload.get('scripted_cycle_fault', '-') or '-')} "
            "start_error="
            f"{str(payload.get('scripted_cycle_activation_rejected_reason', '-') or '-')} "
            "task_mark_error="
            f"{str(payload.get('scripted_cycle_task_state_advance_rejected_reason', '-') or '-')} "
        )
    if action is None:
        return cycle_text + "model_report=inactive"
    action_text = _format_action(action)
    assisted_text = "-" if assisted is None else _format_action(assisted)
    commanded_text = "-" if commanded is None else _format_action(commanded)
    if intent is None:
        intent_text = "-"
        bucket_text = "-"
    else:
        intent_text = "[" + ",".join(f"{value:.2f}" for value in intent) + "]"
        bucket_text = f"+{intent[6]:.2f}/-{intent[7]:.2f}"
    return (
        cycle_text + f"model_raw={action_text} assisted={assisted_text} "
        f"commanded={commanded_text} "
        f"intent8(sw+,sw-,bo+,bo-,st+,st-,bk+,bk-)={intent_text} "
        f"bucket_intent={bucket_text}"
    )


def _status_vector(value: Any, *, width: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _format_script_side(value: Any) -> str:
    side = str(value or "-")
    if side == "A":
        return "A(left)"
    if side == "B":
        return "B(right)"
    return side


if __name__ == "__main__":
    main()
