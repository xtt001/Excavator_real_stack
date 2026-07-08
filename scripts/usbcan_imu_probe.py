#!/usr/bin/env python3
"""Read-only ZLG USBCAN IMU receive probe."""

from __future__ import annotations

import argparse
import collections
import ctypes
import json
import sys
import time
from datetime import datetime
from pathlib import Path


USBCAN_II = ctypes.c_uint32(4)
BAUD_BY_BITRATE = {
    1_000_000: 0x1400,
    500_000: 0x1C00,
    250_000: 0x1C01,
    125_000: 0x1C03,
}


class InitConfig(ctypes.Structure):
    _fields_ = [
        ("AccCode", ctypes.c_uint32),
        ("AccMask", ctypes.c_uint32),
        ("Reserved", ctypes.c_uint32),
        ("Filter", ctypes.c_ubyte),
        ("Timing0", ctypes.c_ubyte),
        ("Timing1", ctypes.c_ubyte),
        ("Mode", ctypes.c_ubyte),
    ]


class CanObj(ctypes.Structure):
    _fields_ = [
        ("ID", ctypes.c_uint32),
        ("TimeStamp", ctypes.c_uint32),
        ("TimeFlag", ctypes.c_ubyte),
        ("SendType", ctypes.c_ubyte),
        ("RemoteFlag", ctypes.c_ubyte),
        ("ExternFlag", ctypes.c_ubyte),
        ("DataLen", ctypes.c_ubyte),
        ("Data", ctypes.c_ubyte * 8),
        ("Reserved", ctypes.c_ubyte * 3),
    ]


class CanStatus(ctypes.Structure):
    _fields_ = [
        ("ErrInterrupt", ctypes.c_ubyte),
        ("regMode", ctypes.c_ubyte),
        ("regStatus", ctypes.c_ubyte),
        ("regALCapture", ctypes.c_ubyte),
        ("regECCapture", ctypes.c_ubyte),
        ("regEWLimit", ctypes.c_ubyte),
        ("regRECounter", ctypes.c_ubyte),
        ("regTECounter", ctypes.c_ubyte),
        ("Reserved", ctypes.c_uint32),
    ]


class ErrInfo(ctypes.Structure):
    _fields_ = [
        ("ErrCode", ctypes.c_uint32),
        ("Passive_ErrData", ctypes.c_ubyte * 3),
        ("ArLost_ErrData", ctypes.c_ubyte),
    ]


class BoardInfo(ctypes.Structure):
    _fields_ = [
        ("hw_Version", ctypes.c_ushort),
        ("fw_Version", ctypes.c_ushort),
        ("dr_Version", ctypes.c_ushort),
        ("in_Version", ctypes.c_ushort),
        ("irq_Num", ctypes.c_ushort),
        ("can_Num", ctypes.c_ubyte),
        ("str_Serial_Num", ctypes.c_char * 20),
        ("str_hw_Type", ctypes.c_char * 40),
        ("Reserved", ctypes.c_ushort * 4),
    ]


def _usbcan_devices() -> list[str]:
    devices: list[str] = []
    for path in Path("/sys/bus/usb/devices").glob("*"):
        vendor_path = path / "idVendor"
        product_path = path / "idProduct"
        if not vendor_path.is_file() or not product_path.is_file():
            continue
        try:
            vendor = vendor_path.read_text(encoding="ascii").strip().lower()
            product = product_path.read_text(encoding="ascii").strip().lower()
        except OSError:
            continue
        if vendor == "0471" and product == "1200":
            devices.append(str(path))
    return sorted(devices)


def _load_library(path: str) -> ctypes.CDLL:
    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:
        raise RuntimeError(f"failed to load {path}: {exc}") from exc
    _configure_library(lib)
    return lib


def _configure_library(lib: ctypes.CDLL) -> None:
    lib.VCI_OpenDevice.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.VCI_OpenDevice.restype = ctypes.c_uint32
    lib.VCI_CloseDevice.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    lib.VCI_CloseDevice.restype = ctypes.c_uint32
    lib.VCI_ReadBoardInfo.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(BoardInfo)]
    lib.VCI_ReadBoardInfo.restype = ctypes.c_uint32
    lib.VCI_InitCAN.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(InitConfig),
    ]
    lib.VCI_InitCAN.restype = ctypes.c_uint32
    lib.VCI_StartCAN.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.VCI_StartCAN.restype = ctypes.c_uint32
    lib.VCI_ResetCAN.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.VCI_ResetCAN.restype = ctypes.c_uint32
    lib.VCI_ClearBuffer.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.VCI_ClearBuffer.restype = ctypes.c_uint32
    lib.VCI_GetReceiveNum.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.VCI_GetReceiveNum.restype = ctypes.c_uint32
    lib.VCI_Receive.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(CanObj),
        ctypes.c_uint32,
        ctypes.c_int,
    ]
    lib.VCI_Receive.restype = ctypes.c_uint32
    lib.VCI_ReadCANStatus.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(CanStatus),
    ]
    lib.VCI_ReadCANStatus.restype = ctypes.c_uint32
    lib.VCI_ReadErrInfo.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ErrInfo),
    ]
    lib.VCI_ReadErrInfo.restype = ctypes.c_uint32


def _cstr(value: bytes) -> str:
    return bytes(value).split(b"\0", 1)[0].decode("ascii", "replace")


def _init_channel(
    lib: ctypes.CDLL,
    device_index: int,
    channel: int,
    bitrate: int,
    mode: int,
) -> None:
    if bitrate not in BAUD_BY_BITRATE:
        supported = ", ".join(str(k) for k in sorted(BAUD_BY_BITRATE))
        raise ValueError(f"unsupported bitrate {bitrate}; supported: {supported}")
    baud = BAUD_BY_BITRATE[bitrate]
    ret = lib.VCI_OpenDevice(USBCAN_II, ctypes.c_uint32(device_index), ctypes.c_uint32(0))
    if ret != 1:
        raise RuntimeError(f"VCI_OpenDevice failed: device_index={device_index}, ret={ret}")

    config = InitConfig(
        0,
        0xFFFFFFFF,
        0,
        1,
        baud & 0xFF,
        (baud >> 8) & 0xFF,
        mode,
    )
    ret = lib.VCI_InitCAN(
        USBCAN_II,
        ctypes.c_uint32(device_index),
        ctypes.c_uint32(channel),
        ctypes.byref(config),
    )
    if ret != 1:
        lib.VCI_CloseDevice(USBCAN_II, ctypes.c_uint32(device_index))
        raise RuntimeError(
            f"VCI_InitCAN failed: device_index={device_index}, channel={channel}, "
            f"bitrate={bitrate}, ret={ret}"
        )
    lib.VCI_ClearBuffer(USBCAN_II, ctypes.c_uint32(device_index), ctypes.c_uint32(channel))
    ret = lib.VCI_StartCAN(USBCAN_II, ctypes.c_uint32(device_index), ctypes.c_uint32(channel))
    if ret != 1:
        lib.VCI_CloseDevice(USBCAN_II, ctypes.c_uint32(device_index))
        raise RuntimeError(
            f"VCI_StartCAN failed: device_index={device_index}, channel={channel}, ret={ret}"
        )


def _close_channel(lib: ctypes.CDLL, device_index: int, channel: int) -> None:
    lib.VCI_ResetCAN(USBCAN_II, ctypes.c_uint32(device_index), ctypes.c_uint32(channel))
    lib.VCI_CloseDevice(USBCAN_II, ctypes.c_uint32(device_index))


def _capture(
    lib: ctypes.CDLL,
    device_index: int,
    channel: int,
    duration_s: float,
    batch_size: int,
    wait_ms: int,
) -> dict[str, object]:
    frames = (CanObj * batch_size)()
    id_counts: collections.Counter[int] = collections.Counter()
    imu_counts: collections.Counter[int] = collections.Counter()
    examples: dict[int, str] = {}
    total = 0
    remote_frames = 0
    extended_frames = 0
    max_available = 0
    last_available = 0
    deadline = time.monotonic() + duration_s

    while time.monotonic() < deadline:
        available = int(
            lib.VCI_GetReceiveNum(
                USBCAN_II,
                ctypes.c_uint32(device_index),
                ctypes.c_uint32(channel),
            )
        )
        max_available = max(max_available, available)
        last_available = available
        if available <= 0:
            time.sleep(0.005)
            continue
        want = min(available, batch_size)
        got = int(
            lib.VCI_Receive(
                USBCAN_II,
                ctypes.c_uint32(device_index),
                ctypes.c_uint32(channel),
                frames,
                ctypes.c_uint32(want),
                ctypes.c_int(wait_ms),
            )
        )
        if got <= 0:
            continue
        for idx in range(min(got, batch_size)):
            frame = frames[idx]
            can_id = int(frame.ID & 0x1FFFFFFF)
            total += 1
            id_counts[can_id] += 1
            if frame.RemoteFlag:
                remote_frames += 1
                continue
            if frame.ExternFlag:
                extended_frames += 1
                continue
            payload = bytes(frame.Data[: min(int(frame.DataLen), 8)])
            examples.setdefault(can_id, payload.hex(" "))
            if int(frame.DataLen) >= 8 and can_id <= 0x7FF and ((can_id >> 6) & 0x1F) == 0x08:
                imu_counts[can_id] += 1

    status = CanStatus()
    err_info = ErrInfo()
    status_ret = int(
        lib.VCI_ReadCANStatus(
            USBCAN_II,
            ctypes.c_uint32(device_index),
            ctypes.c_uint32(channel),
            ctypes.byref(status),
        )
    )
    err_ret = int(
        lib.VCI_ReadErrInfo(
            USBCAN_II,
            ctypes.c_uint32(device_index),
            ctypes.c_uint32(channel),
            ctypes.byref(err_info),
        )
    )
    addr_counts = collections.Counter(can_id & 0x07 for can_id in imu_counts.elements())
    return {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "duration_s": duration_s,
        "max_available": max_available,
        "last_available": last_available,
        "total_frames": total,
        "remote_frames": remote_frames,
        "extended_frames": extended_frames,
        "can_status": {
            "status_ret": status_ret,
            "err_ret": err_ret,
            "regStatus": int(status.regStatus),
            "err_interrupt": int(status.ErrInterrupt),
            "rx_error_counter": int(status.regRECounter),
            "tx_error_counter": int(status.regTECounter),
            "err_code": int(err_info.ErrCode),
        },
        "imu_highspeed_ch1_frames": sum(imu_counts.values()),
        "imu_highspeed_ch1_ids": {
            f"0x{can_id:03X}": int(count) for can_id, count in sorted(imu_counts.items())
        },
        "raw_addr_counts": {
            str(addr): int(count) for addr, count in sorted(addr_counts.items())
        },
        "top_ids": [
            {
                "id": f"0x{can_id:03X}",
                "count": int(count),
                "example": examples.get(can_id, ""),
            }
            for can_id, count in id_counts.most_common(24)
        ],
    }


def _print_summary(result: dict[str, object]) -> None:
    status = result.get("can_status", {})
    print(
        f"[{result['captured_at']}] "
        f"total={result['total_frames']} "
        f"imu={result['imu_highspeed_ch1_frames']} "
        f"ids={len(result['top_ids'])} "
        f"max_avail={result.get('max_available', '-')}"
    )
    if isinstance(status, dict):
        print(
            "  "
            f"status=0x{int(status.get('regStatus', 0)):02x} "
            f"rec={status.get('rx_error_counter', '-')} "
            f"tec={status.get('tx_error_counter', '-')} "
            f"err=0x{int(status.get('err_code', 0)):08x}"
        )
    top_ids = result["top_ids"]
    if isinstance(top_ids, list):
        for item in top_ids[:12]:
            print(f"  {item['id']} count={item['count']} data={item['example']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", default="libusbcan.so", help="Path/name of libusbcan.so.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--bitrate", type=int, default=250000)
    parser.add_argument("--mode", type=int, choices=[0, 1], default=0, help="0 normal, 1 listen-only.")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--wait-ms", type=int, default=1, help="VCI_Receive wait time in milliseconds.")
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of a short summary.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--watch", action="store_true", help="Repeat until Ctrl-C.")
    parser.add_argument("--interval-s", type=float, default=1.0, help="Sleep between --watch captures.")
    parser.add_argument("--require-data", action="store_true", help="Return non-zero when no frames are captured.")
    parser.add_argument(
        "--skip-usb-check",
        action="store_true",
        help="Skip the sysfs 0471:1200 enumeration check before VCI_OpenDevice.",
    )
    return parser.parse_args()


def run_once(args: argparse.Namespace) -> dict[str, object]:
    if not args.skip_usb_check and not _usbcan_devices():
        raise RuntimeError(
            "USBCAN USB device 0471:1200 is not enumerated; replug the adapter "
            "and check `lsusb -d 0471:1200` on the Jetson"
        )
    lib = _load_library(args.lib)
    _init_channel(lib, args.device_index, args.channel, args.bitrate, args.mode)
    try:
        return _capture(
            lib,
            args.device_index,
            args.channel,
            args.duration_s,
            args.batch_size,
            args.wait_ms,
        )
    finally:
        _close_channel(lib, args.device_index, args.channel)


def main() -> int:
    args = parse_args()
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    last_result: dict[str, object] | None = None
    while True:
        last_result = run_once(args)
        if args.json:
            print(json.dumps(last_result, indent=2, sort_keys=True))
        else:
            _print_summary(last_result)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(last_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.watch:
            break
        time.sleep(max(0.0, args.interval_s))

    if args.require_data and last_result is not None and int(last_result["total_frames"]) == 0:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"usbcan_imu_probe failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
