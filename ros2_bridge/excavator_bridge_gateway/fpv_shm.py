"""与 ros2_bridge/fpv_frame_store C++ 布局一致的 POSIX 共享内存读写。"""

from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass
from typing import Optional

MAGIC = 0x46505631
VERSION = 1
MUTEX_SIZE = 40  # Linux x86_64 glibc pthread_mutex_t
HEADER_SIZE = 8 + MUTEX_SIZE + 8 + 8 + 4 + 4 + 4 + 4
MAX_WIDTH = 640
MAX_HEIGHT = 480
MAX_BYTES = MAX_WIDTH * MAX_HEIGHT * 3
SHM_TOTAL = HEADER_SIZE + MAX_BYTES


@dataclass(frozen=True)
class FpvFrame:
    timestamp_ns: int
    receive_time_ns: int
    sequence: int
    width: int
    height: int
    rgb: bytes


def _shm_path(name: str) -> str:
    base = name.lstrip("/")
    return os.path.join("/dev/shm", base)


def _open_map(name: str, *, create: bool) -> mmap.mmap:
    path = _shm_path(name)
    if create:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, SHM_TOTAL)
    else:
        fd = os.open(path, os.O_RDWR)
    return mmap.mmap(fd, SHM_TOTAL, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)


class FpvShmWriter:
    def __init__(self, name: str = "excavator_fpv_v1") -> None:
        self._name = name
        self._mm = _open_map(name, create=True)
        self._mm[0:4] = struct.pack("<I", MAGIC)
        self._mm[4:8] = struct.pack("<I", VERSION)

    def write_rgb(
        self,
        rgb: bytes,
        width: int,
        height: int,
        timestamp_ns: int,
        receive_time_ns: int | None = None,
    ) -> bool:
        if width <= 0 or height <= 0 or width > MAX_WIDTH or height > MAX_HEIGHT:
            return False
        nbytes = width * height * 3
        if len(rgb) != nbytes:
            return False
        recv_ns = int(receive_time_ns if receive_time_ns is not None else time.time_ns())
        seq = struct.unpack_from("<I", self._mm, 64)[0] + 1
        struct.pack_into(
            "<QQIIII", self._mm, 48, int(timestamp_ns), recv_ns, seq, width, height, nbytes
        )
        self._mm[HEADER_SIZE : HEADER_SIZE + nbytes] = rgb
        return True

    def close(self) -> None:
        self._mm.close()


class FpvShmReader:
    def __init__(self, name: str = "excavator_fpv_v1") -> None:
        self._name = name
        try:
            self._mm = _open_map(name, create=False)
        except OSError:
            self._mm = None

    def read_latest(self) -> Optional[FpvFrame]:
        if self._mm is None:
            return None
        magic, version = struct.unpack_from("<II", self._mm, 0)
        if magic != MAGIC or version != VERSION:
            return None
        timestamp_ns, receive_ns, seq, width, height, nbytes = struct.unpack_from(
            "<QQIIII", self._mm, 48
        )
        if nbytes == 0 or width <= 0 or height <= 0:
            return None
        if nbytes > MAX_BYTES:
            return None
        rgb = bytes(self._mm[HEADER_SIZE : HEADER_SIZE + nbytes])
        return FpvFrame(
            timestamp_ns=int(timestamp_ns),
            receive_time_ns=int(receive_ns),
            sequence=int(seq),
            width=int(width),
            height=int(height),
            rgb=rgb,
        )

    def is_fresh(self, max_age_ms: int) -> bool:
        frame = self.read_latest()
        if frame is None:
            return False
        age_ns = max(0, time.time_ns() - frame.receive_time_ns)
        return age_ns <= int(max_age_ms) * 1_000_000
