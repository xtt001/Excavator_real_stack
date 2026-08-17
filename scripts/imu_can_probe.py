#!/usr/bin/env python3
"""Read-only IMU CAN address probe for the excavator high-speed ch1 protocol."""

from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _can_interfaces() -> list[str]:
    result = subprocess.run(
        ["ip", "-o", "link", "show"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    interfaces: list[str] = []
    for line in result.stdout.splitlines():
        match = re.match(r"^\d+:\s+(can\d+)", line)
        if match:
            interfaces.append(match.group(1))
    return interfaces


def _interface_is_up(interface: str) -> bool:
    result = subprocess.run(
        ["ip", "-details", "link", "show", interface],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return "LOWER_UP" in result.stdout or "state UP" in result.stdout


def _parse_candump_id(line: str, interface: str) -> int | None:
    parts = line.replace(")", " ").split()
    if interface not in parts:
        return None
    idx = parts.index(interface)
    if idx + 1 >= len(parts):
        return None
    token = parts[idx + 1].split("#", 1)[0]
    try:
        return int(token, 16)
    except ValueError:
        return None


def _capture_ids(interface: str, duration_s: float) -> list[int]:
    result = subprocess.run(
        ["timeout", f"{float(duration_s):.3f}", "candump", "-ta", interface],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ids: list[int] = []
    for line in result.stdout.splitlines():
        can_id = _parse_candump_id(line, interface)
        if can_id is not None:
            ids.append(can_id)
    return ids


def _summarize_interface(interface: str, duration_s: float) -> dict[str, object]:
    up = _interface_is_up(interface)
    ids = _capture_ids(interface, duration_s) if up else []
    imu_ids = [
        can_id
        for can_id in ids
        if can_id <= 0x7FF and ((can_id >> 6) & 0x1F) == 0x08
    ]
    id_counts = collections.Counter(imu_ids)
    addr_counts = collections.Counter(can_id & 0x07 for can_id in imu_ids)
    cmd_counts_by_addr: dict[str, dict[str, int]] = {}
    for addr in sorted(addr_counts):
        cmd_counts = collections.Counter(
            (can_id >> 3) & 0x07 for can_id in imu_ids if (can_id & 0x07) == addr
        )
        cmd_counts_by_addr[str(addr)] = {
            str(cmd): int(count) for cmd, count in sorted(cmd_counts.items())
        }
    return {
        "interface": interface,
        "up": up,
        "captured_frames": len(ids),
        "imu_highspeed_ch1_frames": len(imu_ids),
        "imu_highspeed_ch1_ids": {
            f"0x{can_id:03X}": int(count) for can_id, count in sorted(id_counts.items())
        },
        "raw_addr_counts": {
            str(addr): int(count) for addr, count in sorted(addr_counts.items())
        },
        "missing_raw_addr_0_to_3": [
            addr for addr in range(4) if int(addr_counts.get(addr, 0)) == 0
        ],
        "cmd_counts_by_raw_addr": cmd_counts_by_addr,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="", help="One SocketCAN interface, e.g. can5.")
    parser.add_argument("--all", action="store_true", help="Probe every canX interface.")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--require-four",
        action="store_true",
        help="Return non-zero unless raw addresses 0,1,2,3 are all observed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("candump") is None:
        raise RuntimeError("candump not found. Install can-utils on the target machine.")
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    if args.all:
        interfaces = _can_interfaces()
    else:
        interfaces = [args.interface or "can5"]
    if not interfaces:
        raise RuntimeError("no canX interfaces found")

    result = {
        "duration_s": float(args.duration_s),
        "interfaces": [
            _summarize_interface(interface, float(args.duration_s))
            for interface in interfaces
        ],
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")

    if args.require_four:
        ok = any(
            item["up"]
            and not item["missing_raw_addr_0_to_3"]
            and int(item["imu_highspeed_ch1_frames"]) > 0
            for item in result["interfaces"]
        )
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"imu_can_probe failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
