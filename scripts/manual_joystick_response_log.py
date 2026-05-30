#!/usr/bin/env python3
"""Log live joystick commands and bridge state for manual response calibration."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


AXES = ("swing", "boom", "stick", "bucket")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("testbed/testbed/configs/teleop_real_v1.yaml"),
    )
    parser.add_argument("--host", default="192.168.100.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=45.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--log-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/manual_response"),
    )
    parser.add_argument(
        "--no-status-buttons",
        action="store_true",
        help="Do not forward joystick status toggle buttons 1/5/6/etc.",
    )
    parser.add_argument(
        "--confirm-real-control",
        action="store_true",
        help="Required because this command sends live joystick actions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_real_control:
        raise RuntimeError("--confirm-real-control is required")
    if args.port <= 0:
        raise ValueError("--port must be positive")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    cfg = _load_yaml(args.config)
    teleop_cfg = dict(cfg.get("teleop", {}) or {})
    joystick_cfg = dict(teleop_cfg.get("joystick", {}) or {})
    task_cfg = dict(cfg.get("task", {}) or {})
    default_dt = float(task_cfg.get("dt", 1.0 / args.rate_hz))

    from testbed.actions.gamepad import JoystickActionSource
    from testbed.backends.real.bridge_socket import JsonTcpBridgeClient

    source = JoystickActionSource.from_config(joystick_cfg, default_dt=default_dt)
    client = JsonTcpBridgeClient(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
        connect_on_init=True,
    )

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{stamp}.jsonl"
    csv_path = args.output_dir / f"{stamp}.csv"
    meta_path = args.output_dir / f"{stamp}.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "created": dt.datetime.now(dt.timezone.utc).isoformat(),
                "host": args.host,
                "port": args.port,
                "rate_hz": args.rate_hz,
                "duration_s": args.duration_s,
                "axis_order": AXES,
                "joystick": joystick_cfg,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    stop = False

    def _stop(_sig: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"[manual-response] bridge={args.host}:{args.port}")
    print(f"[manual-response] writing {jsonl_path}")
    print("[manual-response] move one physical control at a time; Ctrl+C stops and sends zero.")

    period_s = 1.0 / args.rate_hz
    step = 0
    start = time.monotonic()
    last_log = 0.0
    rows: list[dict[str, Any]] = []

    try:
        client.send_action(np.zeros(4, dtype=np.float32))
        with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as cf:
            writer = csv.DictWriter(cf, fieldnames=_csv_fields())
            writer.writeheader()
            while not stop:
                now = time.monotonic()
                elapsed = now - start
                if elapsed >= args.duration_s:
                    break

                samples = client.read_state(step_id=step)
                obs = dict(samples.joint.payload)
                raw_action, info = source.next_action(obs)
                extras = getattr(info, "extras", {}) or {}
                toggle_mask = int(extras.get("toggle_mask", 0) or 0)
                if toggle_mask and not args.no_status_buttons:
                    client.apply_status_toggle_mask(toggle_mask)

                result = client.send_action(raw_action, state=obs)
                qpos = _vec(obs.get("qpos"), 4)
                qvel = _vec(obs.get("qvel"), 4)
                status = _vec(obs.get("status"), 12, dtype=int)
                action = _vec(raw_action, 4)
                raw_low = _vec(result.raw_low_level_command, 8)
                record = {
                    "step": step,
                    "t_s": elapsed,
                    "host_time_ns": time.time_ns(),
                    "action": action,
                    "qpos": qpos,
                    "qvel": qvel,
                    "status": status,
                    "toggle_mask": toggle_mask,
                    "ack": bool(result.ack),
                    "fault_code": str(result.fault_code),
                    "raw_low_level_command": raw_low,
                    "source_id": str(getattr(info, "source_id", "")),
                    "latency_ms": float(getattr(info, "latency_ms", 0.0) or 0.0),
                }
                jf.write(json.dumps(record, separators=(",", ":")) + "\n")
                jf.flush()
                csv_row = _flatten_record(record)
                writer.writerow(csv_row)
                rows.append(record)

                if args.log_interval_s > 0 and now - last_log >= args.log_interval_s:
                    print(
                        "[manual-response] "
                        f"t={elapsed:5.1f}s "
                        f"a={_fmt(action)} q={_fmt(qpos)} v={_fmt(qvel)} "
                        f"ack={int(result.ack)} fault={result.fault_code or '-'}"
                    )
                    last_log = now

                step += 1
                _sleep_to_rate(start, step, period_s)
    finally:
        try:
            client.send_action(np.zeros(4, dtype=np.float32))
        except Exception as exc:
            print(f"[manual-response] warning: failed to send final zero: {exc}")
        try:
            client.close()
        except Exception:
            pass
        try:
            source.close()
        except Exception:
            pass

    print(f"[manual-response] complete: {step} samples")
    if rows:
        _print_summary(rows)
    print(f"[manual-response] jsonl={jsonl_path}")
    print(f"[manual-response] csv={csv_path}")
    return 0


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
    with path.open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def _vec(value: Any, size: int, *, dtype: Any = float) -> list[Any]:
    arr = np.asarray(value if value is not None else [], dtype=dtype).reshape(-1)
    out: list[Any] = []
    for i in range(size):
        if i < arr.size:
            v = arr[i].item() if hasattr(arr[i], "item") else arr[i]
            out.append(v)
        else:
            out.append(0 if dtype is int else 0.0)
    return out


def _csv_fields() -> list[str]:
    fields = ["step", "t_s", "host_time_ns", "ack", "fault_code", "toggle_mask"]
    for prefix, names in (
        ("action", AXES),
        ("qpos", AXES),
        ("qvel", AXES),
        ("raw_low", tuple(str(i) for i in range(8))),
        ("status", tuple(str(i) for i in range(12))),
    ):
        fields.extend(f"{prefix}_{name}" for name in names)
    return fields


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "step": record["step"],
        "t_s": record["t_s"],
        "host_time_ns": record["host_time_ns"],
        "ack": int(record["ack"]),
        "fault_code": record["fault_code"],
        "toggle_mask": record["toggle_mask"],
    }
    for prefix, values, names in (
        ("action", record["action"], AXES),
        ("qpos", record["qpos"], AXES),
        ("qvel", record["qvel"], AXES),
        ("raw_low", record["raw_low_level_command"], tuple(str(i) for i in range(8))),
        ("status", record["status"], tuple(str(i) for i in range(12))),
    ):
        for name, value in zip(names, values):
            row[f"{prefix}_{name}"] = value
    return row


def _fmt(values: list[Any]) -> str:
    return "[" + ",".join(f"{float(v):+.3f}" for v in values[:4]) + "]"


def _sleep_to_rate(start: float, step: int, period_s: float) -> None:
    deadline = start + step * period_s
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(min(remaining, period_s))


def _print_summary(rows: list[dict[str, Any]]) -> None:
    action = np.asarray([r["action"] for r in rows], dtype=float)
    qpos = np.asarray([r["qpos"] for r in rows], dtype=float)
    qvel = np.asarray([r["qvel"] for r in rows], dtype=float)
    print("[manual-response] summary:")
    for i, name in enumerate(AXES):
        print(
            "  "
            f"{name}: max|action|={np.max(np.abs(action[:, i])):.3f} "
            f"qpos_delta={qpos[-1, i] - qpos[0, i]:+.4f} "
            f"max|qvel|={np.max(np.abs(qvel[:, i])):.4f}"
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"manual_joystick_response_log failed: {exc}")
        raise SystemExit(1)
