#!/usr/bin/env python3
"""Read-only SocketCAN probe for real-machine bring-up."""

from __future__ import annotations

import argparse
import json
import pathlib
import selectors
import shutil
import subprocess
import sys
import time
from typing import TextIO


DEFAULT_IDS = ("18F021F6", "18F022F6", "18F023F6")


def _normalize_can_id(raw: str) -> str:
    token = raw.strip().upper()
    if token.startswith("0X"):
        token = token[2:]
    if not token or not all(ch in "0123456789ABCDEF" for ch in token):
        raise ValueError(f"invalid CAN id: {raw!r}")
    return token


def _parse_candump_id(line: str, expected_interface: str) -> str | None:
    text = line.strip()
    if not text:
        return None
    if text.startswith("("):
        right = text.find(")")
        if right < 0:
            return None
        text = text[right + 1 :].strip()
    parts = text.split()
    if len(parts) < 2 or parts[0] != expected_interface:
        return None
    token = parts[1].split("#", 1)[0].upper()
    if not token or not all(ch in "0123456789ABCDEF" for ch in token):
        return None
    return token


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_ip_show(interface: str, output_dir: pathlib.Path) -> None:
    if shutil.which("ip") is None:
        print("warning: ip command not found; skipping interface details", file=sys.stderr)
        return
    result = subprocess.run(
        ["ip", "-details", "link", "show", interface],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    _write_text(output_dir / "ip-link-show.txt", result.stdout)
    print(result.stdout.strip())
    if result.returncode != 0:
        raise RuntimeError(f"CAN interface {interface!r} is not visible to ip link")


def _open_outputs(output_dir: pathlib.Path, ids: set[str]) -> tuple[TextIO, dict[str, TextIO]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = (output_dir / "candump.raw.log").open("a", encoding="utf-8")
    per_id = {
        can_id: (output_dir / f"{can_id}.log").open("a", encoding="utf-8")
        for can_id in sorted(ids)
    }
    return raw, per_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="can0", help="SocketCAN interface to listen on.")
    parser.add_argument("--duration-s", type=float, default=10.0, help="Read duration in seconds.")
    parser.add_argument(
        "--ids",
        nargs="+",
        default=list(DEFAULT_IDS),
        help="CAN IDs expected from the excavator control protocol.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/can_probe",
        help="Directory for raw candump logs, per-ID logs, and summary.json.",
    )
    parser.add_argument(
        "--require-all-ids",
        action="store_true",
        help="Return non-zero if any requested ID is not observed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    if shutil.which("candump") is None:
        raise RuntimeError("candump not found. Install can-utils on the target machine.")

    can_ids = {_normalize_can_id(can_id) for can_id in args.ids}
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_ip_show(args.interface, output_dir)

    counts = {can_id: 0 for can_id in can_ids}
    raw_file, per_id_files = _open_outputs(output_dir, can_ids)
    process = subprocess.Popen(
        ["candump", "-ta", args.interface],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    selector.register(process.stdout, selectors.EVENT_READ)
    start = time.monotonic()
    last_print = start
    observed_total = 0

    try:
        while time.monotonic() - start < args.duration_s:
            timeout = min(0.2, max(0.0, args.duration_s - (time.monotonic() - start)))
            for key, _ in selector.select(timeout):
                line = key.fileobj.readline()
                if not line:
                    break
                raw_file.write(line)
                raw_file.flush()
                observed_total += 1
                can_id = _parse_candump_id(line, args.interface)
                if can_id in counts:
                    counts[can_id] += 1
                    per_id_files[can_id].write(line)
                    per_id_files[can_id].flush()
            now = time.monotonic()
            if now - last_print >= 1.0:
                msg = " ".join(f"{can_id}:{counts[can_id]}" for can_id in sorted(counts))
                print(f"probe counts {msg}")
                last_print = now
    finally:
        selector.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        raw_file.close()
        for file_obj in per_id_files.values():
            file_obj.close()

    summary = {
        "interface": args.interface,
        "duration_s": args.duration_s,
        "ids": sorted(can_ids),
        "counts": counts,
        "observed_total_frames": observed_total,
        "missing_ids": [can_id for can_id in sorted(can_ids) if counts[can_id] == 0],
        "output_dir": str(output_dir),
    }
    _write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_all_ids and summary["missing_ids"]:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"can_probe failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
