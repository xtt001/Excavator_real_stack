#!/usr/bin/env python3
"""Measure one-axis command response through the real bridge.

This tool is intentionally conservative:
- one axis at a time;
- small default amplitudes;
- automatic zero command before and after every pulse;
- hard amplitude cap unless the source is edited deliberately;
- JSONL output for later go-home tuning.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import socket
import time
from pathlib import Path
from typing import Any


AXIS_INDEX = {
    "swing": 0,
    "boom": 1,
    "stick": 2,
    "bucket": 3,
}
AXIS_NAMES = tuple(AXIS_INDEX)
ZERO_ACTION = [0.0, 0.0, 0.0, 0.0]
HARD_MAX_AMPLITUDE = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--axis",
        action="append",
        choices=sorted(AXIS_INDEX),
        help="Axis to test. Repeat for multiple axes. Default: all four axes.",
    )
    parser.add_argument(
        "--direction",
        choices=("positive", "negative", "both"),
        default="both",
    )
    parser.add_argument(
        "--amplitudes",
        default="0.03,0.05,0.07,0.10,0.12",
        help="Comma-separated normalized commands, each in (0, 0.20].",
    )
    parser.add_argument("--duration-s", type=float, default=0.45)
    parser.add_argument("--settle-s", type=float, default=0.80)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument(
        "--motion-qpos-threshold-rad",
        type=float,
        default=0.003,
        help="command-window qpos delta used to classify movement direction.",
    )
    parser.add_argument(
        "--motion-qvel-threshold-rad-s",
        type=float,
        default=0.010,
        help="qvel change relative to the zero-command baseline used to classify first movement.",
    )
    parser.add_argument(
        "--abort-delta-rad",
        type=float,
        default=0.08,
        help="Abort a pulse and zero if the active axis moves this far.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output. Default: artifacts/axis_response/<timestamp>.jsonl",
    )
    parser.add_argument(
        "--confirm-hardware-motion",
        action="store_true",
        help="Required before sending any non-zero command.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    axes = tuple(args.axis or AXIS_NAMES)
    amplitudes = _parse_amplitudes(args.amplitudes)
    directions = _directions(args.direction)
    _validate_args(args, amplitudes)
    output = args.output or _default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[axis-response] target bridge: {args.host}:{args.port}")
    print(f"[axis-response] axes={','.join(axes)} directions={args.direction}")
    print(f"[axis-response] amplitudes={','.join(f'{a:.3f}' for a in amplitudes)}")
    print(f"[axis-response] output={output}")

    summaries: list[dict[str, Any]] = []
    step_id = 0
    with socket.create_connection((args.host, args.port), timeout=args.timeout_s) as sock:
        sock.settimeout(args.timeout_s)
        try:
            _send_action(sock, ZERO_ACTION)
            for axis in axes:
                for sign in directions:
                    for amplitude in amplitudes:
                        command = float(sign * amplitude)
                        trial, step_id = _run_trial(
                            sock=sock,
                            axis=axis,
                            command=command,
                            step_id=step_id,
                            duration_s=args.duration_s,
                            settle_s=args.settle_s,
                            rate_hz=args.rate_hz,
                            motion_qpos_threshold_rad=args.motion_qpos_threshold_rad,
                            motion_qvel_threshold_rad_s=args.motion_qvel_threshold_rad_s,
                            abort_delta_rad=args.abort_delta_rad,
                        )
                        summaries.append(trial["summary"])
                        _append_jsonl(output, trial)
                        _print_summary(trial["summary"])
        finally:
            try:
                _send_action(sock, ZERO_ACTION)
                _request(sock, "close.request")
            except Exception as exc:  # pragma: no cover - best-effort hardware zero
                print(f"[axis-response] warning: failed final zero/close: {exc}")

    print("[axis-response] suggested dead-zone estimate:")
    for line in _recommendations(summaries):
        print(line)
    print("[axis-response] complete")
    return 0


def _run_trial(
    *,
    sock: socket.socket,
    axis: str,
    command: float,
    step_id: int,
    duration_s: float,
    settle_s: float,
    rate_hz: float,
    motion_qpos_threshold_rad: float,
    motion_qvel_threshold_rad_s: float,
    abort_delta_rad: float,
) -> tuple[dict[str, Any], int]:
    axis_idx = AXIS_INDEX[axis]
    period_s = 1.0 / rate_hz
    samples: list[dict[str, Any]] = []

    _send_action(sock, ZERO_ACTION)
    baseline = _read_state(sock, step_id)
    step_id += 1
    baseline_qpos = _vector4(baseline, "qpos")
    baseline_qvel = _vector4(baseline, "qvel")
    start_monotonic = time.monotonic()
    aborted = False
    first_motion_latency_s: float | None = None

    deadline = start_monotonic + duration_s
    while time.monotonic() < deadline:
        ack = _send_action(sock, _action(axis_idx, command))
        state = _read_state(sock, step_id)
        step_id += 1
        sample = _sample_record(
            phase="command",
            axis=axis,
            command=command,
            state=state,
            ack=ack,
            t0=start_monotonic,
            baseline_qpos=baseline_qpos,
            baseline_qvel=baseline_qvel,
        )
        samples.append(sample)

        active_delta = float(sample["qpos_delta"][axis_idx])
        active_qvel_delta = float(sample["qvel_delta"][axis_idx])
        if first_motion_latency_s is None and (
            abs(active_delta) >= motion_qpos_threshold_rad
            or abs(active_qvel_delta) >= motion_qvel_threshold_rad_s
        ):
            first_motion_latency_s = float(sample["t_s"])
        if abs(active_delta) >= abort_delta_rad:
            aborted = True
            break
        _sleep_until_next(period_s)

    _send_action(sock, ZERO_ACTION)
    settle_deadline = time.monotonic() + settle_s
    while time.monotonic() < settle_deadline:
        ack = _send_action(sock, ZERO_ACTION)
        state = _read_state(sock, step_id)
        step_id += 1
        samples.append(
            _sample_record(
                phase="settle",
                axis=axis,
                command=0.0,
                state=state,
                ack=ack,
                t0=start_monotonic,
                baseline_qpos=baseline_qpos,
                baseline_qvel=baseline_qvel,
            )
        )
        _sleep_until_next(period_s)

    summary = _summarize_trial(
        axis=axis,
        axis_idx=axis_idx,
        command=command,
        baseline=baseline,
        samples=samples,
        aborted=aborted,
        first_motion_latency_s=first_motion_latency_s,
        motion_qpos_threshold_rad=motion_qpos_threshold_rad,
    )
    trial = {
        "type": "axis_response_trial",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "summary": summary,
        "samples": samples,
    }
    return trial, step_id


def _summarize_trial(
    *,
    axis: str,
    axis_idx: int,
    command: float,
    baseline: dict[str, Any],
    samples: list[dict[str, Any]],
    aborted: bool,
    first_motion_latency_s: float | None,
    motion_qpos_threshold_rad: float,
) -> dict[str, Any]:
    command_samples = [s for s in samples if s["phase"] == "command"]
    all_samples = samples or command_samples
    baseline_qvel = _vector4(baseline, "qvel")
    if all_samples:
        final_qpos_delta = all_samples[-1]["qpos_delta"]
        peak_abs_qvel = max(abs(float(s["qvel"][axis_idx])) for s in all_samples)
        faults = sorted({str(s.get("fault_code") or "") for s in all_samples})
        ack_all = all(bool(s.get("ack", False)) for s in all_samples)
    else:
        final_qpos_delta = [0.0, 0.0, 0.0, 0.0]
        peak_abs_qvel = 0.0
        faults = []
        ack_all = False

    if command_samples:
        command_axis_delta = float(command_samples[-1]["qpos_delta"][axis_idx])
        max_abs_command_delta = max(
            abs(float(s["qpos_delta"][axis_idx])) for s in command_samples
        )
        peak_abs_qvel_delta = max(
            abs(float(s["qvel_delta"][axis_idx])) for s in command_samples
        )
        command_qvel_delta_mean = sum(
            float(s["qvel_delta"][axis_idx]) for s in command_samples
        ) / len(command_samples)
    else:
        command_axis_delta = 0.0
        max_abs_command_delta = 0.0
        peak_abs_qvel_delta = 0.0
        command_qvel_delta_mean = 0.0

    expected_sign = 0 if command == 0 else (1 if command > 0 else -1)
    observed_sign = (
        0
        if abs(command_axis_delta) < motion_qpos_threshold_rad
        else (1 if command_axis_delta > 0 else -1)
    )
    return {
        "axis": axis,
        "axis_index": axis_idx,
        "command": float(command),
        "duration_command_s": (
            float(command_samples[-1]["t_s"] - command_samples[0]["t_s"])
            if len(command_samples) > 1
            else 0.0
        ),
        "sample_count": len(samples),
        "baseline_qpos": _vector4(baseline, "qpos"),
        "baseline_qvel": baseline_qvel,
        "final_qpos_delta": final_qpos_delta,
        "active_axis_delta_rad": command_axis_delta,
        "command_axis_delta_rad": command_axis_delta,
        "max_abs_active_delta_rad": float(max_abs_command_delta),
        "max_abs_command_delta_rad": float(max_abs_command_delta),
        "peak_abs_active_qvel_rad_s": float(peak_abs_qvel),
        "peak_abs_active_qvel_delta_rad_s": float(peak_abs_qvel_delta),
        "command_qvel_delta_mean_rad_s": float(command_qvel_delta_mean),
        "first_motion_latency_s": first_motion_latency_s,
        "moved": bool(first_motion_latency_s is not None),
        "expected_sign": expected_sign,
        "observed_sign": observed_sign,
        "sign_matches": bool(observed_sign == 0 or observed_sign == expected_sign),
        "aborted": bool(aborted),
        "ack_all": bool(ack_all),
        "fault_codes": [f for f in faults if f],
    }


def _sample_record(
    *,
    phase: str,
    axis: str,
    command: float,
    state: dict[str, Any],
    ack: dict[str, Any],
    t0: float,
    baseline_qpos: list[float],
    baseline_qvel: list[float],
) -> dict[str, Any]:
    qpos = _vector4(state, "qpos")
    qvel = _vector4(state, "qvel")
    return {
        "phase": phase,
        "t_s": float(time.monotonic() - t0),
        "host_time_ns": time.time_ns(),
        "axis": axis,
        "command": float(command),
        "qpos": qpos,
        "qvel": qvel,
        "qpos_delta": [float(qpos[i] - baseline_qpos[i]) for i in range(4)],
        "qvel_delta": [float(qvel[i] - baseline_qvel[i]) for i in range(4)],
        "status": state.get("status", []),
        "ack": bool(ack.get("ack", False)),
        "fault_code": str(ack.get("fault_code", "") or ""),
        "raw_low_level_command": ack.get("raw_low_level_command", []),
        "commanded_action": ack.get("commanded_action", []),
    }


def _send_action(sock: socket.socket, action: list[float]) -> dict[str, Any]:
    response = _request(
        sock,
        "send_action.request",
        {"action": [float(x) for x in action], "send_time_ns": time.time_ns()},
    )
    payload = response.get("payload", {})
    if not isinstance(payload, dict):
        raise RuntimeError(f"send_action payload is not a dict: {payload!r}")
    return payload


def _read_state(sock: socket.socket, step_id: int) -> dict[str, Any]:
    response = _request(
        sock,
        "read_state.request",
        {"step_id": int(step_id), "request_time_ns": time.time_ns()},
    )
    payload = response.get("payload", {})
    if not isinstance(payload, dict):
        raise RuntimeError(f"read_state payload is not a dict: {payload!r}")
    joint = payload.get("joint", {})
    if isinstance(joint, dict):
        joint_payload = joint.get("payload", {})
        if isinstance(joint_payload, dict):
            return joint_payload
    raise RuntimeError(f"read_state joint payload missing: {payload!r}")


def _request(
    sock: socket.socket,
    msg_type: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message = {
        "version": 1,
        "type": msg_type,
        "payload": {} if payload is None else payload,
    }
    sock.sendall(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n")
    response = _read_line(sock)
    if response.get("type") != msg_type.replace(".request", ".response"):
        raise RuntimeError(f"unexpected response type: {response}")
    if response.get("ok") is not True:
        raise RuntimeError(f"bridge request failed: {response}")
    return response


def _read_line(sock: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(1)
        if not chunk:
            raise RuntimeError("bridge closed before sending a response")
        if chunk == b"\n":
            break
        chunks.append(chunk)
    return json.loads(b"".join(chunks).decode("utf-8"))


def _action(axis_idx: int, command: float) -> list[float]:
    action = [0.0, 0.0, 0.0, 0.0]
    action[axis_idx] = float(command)
    return action


def _vector4(state: dict[str, Any], key: str) -> list[float]:
    value = state.get(key)
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        raise RuntimeError(f"state {key} must contain four values, got {value!r}")
    out = [float(value[i]) for i in range(4)]
    if any(not math.isfinite(v) for v in out):
        raise RuntimeError(f"state {key} contains non-finite values: {out!r}")
    return out


def _parse_amplitudes(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("--amplitudes must contain at least one value")
    return values


def _directions(value: str) -> tuple[float, ...]:
    if value == "positive":
        return (1.0,)
    if value == "negative":
        return (-1.0,)
    return (1.0, -1.0)


def _validate_args(args: argparse.Namespace, amplitudes: tuple[float, ...]) -> None:
    if args.port <= 0:
        raise ValueError("--port must be positive")
    if not args.confirm_hardware_motion:
        raise RuntimeError("--confirm-hardware-motion is required before sending motion")
    if args.duration_s <= 0.0 or args.settle_s < 0.0 or args.rate_hz <= 0.0:
        raise ValueError("--duration-s/rate-hz must be positive; --settle-s must be >= 0")
    if args.motion_qpos_threshold_rad <= 0.0 or args.motion_qvel_threshold_rad_s <= 0.0:
        raise ValueError("motion thresholds must be positive")
    if args.abort_delta_rad <= 0.0:
        raise ValueError("--abort-delta-rad must be positive")
    for amplitude in amplitudes:
        if not (0.0 < amplitude <= HARD_MAX_AMPLITUDE):
            raise ValueError(
                f"amplitude {amplitude} is outside (0, {HARD_MAX_AMPLITUDE}]"
            )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")


def _print_summary(summary: dict[str, Any]) -> None:
    latency = summary["first_motion_latency_s"]
    latency_text = "-" if latency is None else f"{float(latency):.3f}s"
    faults = ",".join(summary["fault_codes"]) or "-"
    qpos_dir = _sign_label(int(summary["observed_sign"]))
    cmd_dir = _sign_label(int(summary["expected_sign"]))
    print(
        "[axis-response] "
        f"axis={summary['axis']} cmd={float(summary['command']):+.3f} "
        f"cmd_dir={cmd_dir} qpos_dir={qpos_dir} "
        f"moved={int(summary['moved'])} "
        f"cmd_delta={float(summary['command_axis_delta_rad']):+.4f}rad "
        f"peak_dqvel={float(summary['peak_abs_active_qvel_delta_rad_s']):.4f}rad/s "
        f"latency={latency_text} sign_ok={int(summary['sign_matches'])} "
        f"aborted={int(summary['aborted'])} ack={int(summary['ack_all'])} fault={faults}"
    )


def _recommendations(summaries: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for axis in AXIS_NAMES:
        axis_rows = [s for s in summaries if s["axis"] == axis]
        if not axis_rows:
            continue
        for sign_name, predicate in (
            ("positive", lambda v: v > 0),
            ("negative", lambda v: v < 0),
        ):
            rows = sorted(
                [s for s in axis_rows if predicate(float(s["command"]))],
                key=lambda s: abs(float(s["command"])),
            )
            moved = [
                s
                for s in rows
                if s["moved"]
                and s["observed_sign"] != 0
                and s["sign_matches"]
                and not s["fault_codes"]
            ]
            if moved:
                first = moved[0]
                lines.append(
                    "  "
                    f"{axis} {sign_name}: first responsive command "
                    f"{float(first['command']):+.3f}, "
                    f"qpos_dir={_sign_label(int(first['observed_sign']))}, "
                    f"peak_qvel={float(first['peak_abs_active_qvel_rad_s']):.4f}rad/s"
                )
            else:
                tested = ",".join(f"{float(s['command']):+.3f}" for s in rows)
                lines.append(f"  {axis} {sign_name}: no clear motion in [{tested}]")
    return lines


def _sign_label(sign: int) -> str:
    if sign > 0:
        return "increase"
    if sign < 0:
        return "decrease"
    return "none"


def _default_output_path() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "axis_response" / f"{stamp}.jsonl"


def _sleep_until_next(period_s: float) -> None:
    if period_s > 0.0:
        time.sleep(period_s)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[axis-response] failed: {exc}")
        raise SystemExit(1)
