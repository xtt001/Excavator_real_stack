#!/usr/bin/env python3
"""
JSON/TCP 网关：testbed 连此端口；控制请求转发到 excavator_real_bridge；
read_state 中按配置拼接 FPV/GMSL 图像，或在无相机诊断模式下返回空图像集。
不修改 bridge/src/excavator_real_bridge.cpp。
"""

from __future__ import annotations

import argparse
import base64
import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from testbed.backends.real.bridge_protocol import (
    BridgeProtocolError,
    decode_frame,
    encode_frame,
    response_message,
)
from testbed.backends.real.bridge_socket import JsonTcpBridgeClient

from excavator_bridge_gateway.fpv_shm import FpvShmReader

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CachedFpvSample:
    sequence: int
    sample: dict[str, Any]


def _placeholder_fpv(width: int, height: int, frame_id: int) -> dict[str, Any]:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = (frame_id * 5) % 255
    image[..., 1] = np.linspace(0, 255, width, dtype=np.uint8)
    image[..., 2] = np.linspace(255, 0, height, dtype=np.uint8)[:, None]
    return {
        "encoding": "raw_uint8",
        "shape": [height, width, 3],
        "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
    }


class FpvPayloadCache:
    """Encode FPV frames off the request path and cache the latest payload.

    JPEG encoding is intentionally keyed by the SHM frame sequence.  The
    recorder can call read_state faster than the camera updates, and encoding
    the same 640x480 frame repeatedly wastes enough CPU to disturb control.
    """

    def __init__(
        self,
        reader: FpvShmReader,
        *,
        fpv_source: str,
        fpv_encoding: str,
        jpeg_quality: int,
        max_encode_hz: float,
        sample_source: str = "ros2_compressed_fpv",
        thread_name: str = "fpv-jpeg-cache",
    ) -> None:
        self.reader = reader
        self.fpv_source = str(fpv_source)
        self.fpv_encoding = str(fpv_encoding).lower()
        self.jpeg_quality = int(jpeg_quality)
        self.max_encode_hz = float(max_encode_hz)
        self.sample_source = str(sample_source)
        self.thread_name = str(thread_name)
        self._lock = threading.Lock()
        self._latest: _CachedFpvSample | None = None
        self._last_sequence = -1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.fpv_encoding != "jpeg" or self.fpv_source not in {"auto", "shm"}:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=self.thread_name,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

    def latest(self, *, max_stale_ms: int) -> dict[str, Any] | None:
        with self._lock:
            cached = self._latest
        if cached is None:
            return None
        receive_time_ns = int(cached.sample.get("receive_time_ns", 0) or 0)
        if not _frame_is_fresh(receive_time_ns, max_stale_ms):
            return None
        return {
            "timestamp_ns": int(cached.sample["timestamp_ns"]),
            "source": str(cached.sample["source"]),
            "receive_time_ns": receive_time_ns,
            "payload": dict(cached.sample["payload"]),
            "metadata": dict(cached.sample.get("metadata", {}) or {}),
        }

    def _run(self) -> None:
        min_period_s = 0.0
        if self.max_encode_hz > 0.0:
            min_period_s = 1.0 / self.max_encode_hz
        last_encode_s = 0.0
        while not self._stop.is_set():
            frame = self.reader.read_latest()
            if frame is None:
                self._stop.wait(0.02)
                continue
            if int(frame.sequence) == self._last_sequence:
                self._stop.wait(0.005)
                continue
            now_s = time.monotonic()
            if min_period_s > 0.0 and now_s - last_encode_s < min_period_s:
                self._stop.wait(min_period_s - (now_s - last_encode_s))
                continue
            try:
                payload = _fpv_payload(
                    rgb=frame.rgb,
                    width=frame.width,
                    height=frame.height,
                    encoding="jpeg",
                    jpeg_quality=self.jpeg_quality,
                )
            except Exception:
                log.exception("FPV JPEG cache encode failed")
                self._stop.wait(0.1)
                continue
            sample = {
                "timestamp_ns": frame.timestamp_ns,
                "source": self.sample_source,
                "receive_time_ns": frame.receive_time_ns,
                "payload": payload,
                "metadata": _frame_metadata(frame),
            }
            with self._lock:
                self._latest = _CachedFpvSample(sequence=int(frame.sequence), sample=sample)
            self._last_sequence = int(frame.sequence)
            last_encode_s = time.monotonic()


def _fpv_sample_from_shm(
    reader: FpvShmReader,
    *,
    max_stale_ms: int,
    placeholder_width: int,
    placeholder_height: int,
    frame_id: int,
    fpv_source: str,
    fpv_encoding: str,
    jpeg_quality: int,
    sample_source: str = "ros2_compressed_fpv",
    stream_name: str = "fpv",
) -> dict[str, Any]:
    use_shm = fpv_source in {"auto", "shm"}
    allow_placeholder = fpv_source in {"auto", "placeholder"}

    if use_shm:
        frame = reader.read_latest()
        if frame is not None and _frame_is_fresh(frame.receive_time_ns, max_stale_ms):
            payload = _fpv_payload(
                rgb=frame.rgb,
                width=frame.width,
                height=frame.height,
                encoding=fpv_encoding,
                jpeg_quality=jpeg_quality,
            )
            return {
                "timestamp_ns": frame.timestamp_ns,
                "source": sample_source,
                "receive_time_ns": frame.receive_time_ns,
                "payload": payload,
                "metadata": _frame_metadata(frame),
            }

    if not allow_placeholder:
        raise BridgeProtocolError(f"{stream_name} shm unavailable and placeholder disabled")

    ts = time.time_ns()
    return {
        "timestamp_ns": ts,
        "source": "bridge_placeholder_fpv",
        "receive_time_ns": ts,
        "payload": _placeholder_fpv(placeholder_width, placeholder_height, frame_id),
        "metadata": {},
    }


def _frame_metadata(frame: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    sequence = getattr(frame, "sequence", None)
    if sequence is not None:
        metadata["sequence"] = int(sequence)
    v4l2_timestamp_ns = getattr(frame, "v4l2_timestamp_ns", None)
    if v4l2_timestamp_ns is not None:
        metadata["v4l2_timestamp_ns"] = int(v4l2_timestamp_ns)
    v4l2_flags = getattr(frame, "v4l2_flags", None)
    if v4l2_flags is not None:
        flags = int(v4l2_flags)
        metadata["flags"] = flags
        metadata["v4l2_error"] = int(bool(flags & 0x40))
        metadata["timestamp_clock"] = _v4l2_timestamp_clock(flags)
        metadata["timestamp_source"] = _v4l2_timestamp_source(flags)
    return metadata


def _gmsl_group_metadata(
    *,
    target_ns: int,
    skew_ns: int,
    valid: bool,
    camera_count: int,
) -> dict[str, Any]:
    return {
        "group_id": int(target_ns),
        "group_target_v4l2_timestamp_ns": int(target_ns),
        "group_skew_ns": int(skew_ns),
        "group_skew_ms": float(skew_ns) / 1_000_000.0,
        "group_valid": int(bool(valid)),
        "group_camera_count": int(camera_count),
        "group_source": "gmsl_v4l2_timestamp_latest_wait",
    }


def _v4l2_timestamp_clock(flags: int) -> str:
    masked = int(flags) & 0x0000E000
    if masked == 0x00002000:
        return "monotonic"
    if masked == 0x00004000:
        return "copy"
    return "unknown"


def _v4l2_timestamp_source(flags: int) -> str:
    masked = int(flags) & 0x00070000
    if masked == 0x00000000:
        return "eof"
    if masked == 0x00010000:
        return "soe"
    return "unknown"


def _fpv_payload(
    *,
    rgb: bytes,
    width: int,
    height: int,
    encoding: str,
    jpeg_quality: int,
) -> dict[str, Any]:
    encoding = str(encoding).lower()
    if encoding == "raw":
        return {
            "encoding": "raw_uint8",
            "shape": [height, width, 3],
            "data_b64": base64.b64encode(rgb).decode("ascii"),
        }
    if encoding != "jpeg":
        raise BridgeProtocolError(f"unsupported fpv encoding {encoding!r}")
    try:
        import cv2
    except ImportError as exc:
        raise BridgeProtocolError("cv2 is required for fpv jpeg encoding") from exc
    arr = np.frombuffer(rgb, dtype=np.uint8)
    try:
        rgb_image = arr.reshape((int(height), int(width), 3))
    except ValueError as exc:
        raise BridgeProtocolError("fpv RGB frame size does not match width/height") from exc
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    quality = int(max(1, min(100, int(jpeg_quality))))
    ok, encoded = cv2.imencode(".jpg", bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise BridgeProtocolError("cv2.imencode failed for fpv jpeg frame")
    return {
        "encoding": "jpeg",
        "shape": [height, width, 3],
        "data_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def _frame_is_fresh(receive_time_ns: int, max_age_ms: int) -> bool:
    age_ns = max(0, time.time_ns() - int(receive_time_ns))
    return age_ns <= int(max_age_ms) * 1_000_000


class BridgeGateway:
    def __init__(
        self,
        *,
        listen_host: str,
        listen_port: int,
        control_host: str,
        control_port: int,
        control_timeout_s: float,
        camera_source: str,
        fpv_source: str,
        fpv_shm_name: str,
        fpv_max_stale_ms: int,
        fpv_encoding: str,
        fpv_jpeg_quality: int,
        fpv_jpeg_cache_hz: float,
        gmsl_cameras: Mapping[str, str] | None,
        gmsl_max_group_skew_ms: float,
        gmsl_group_timeout_ms: float,
        placeholder_width: int,
        placeholder_height: int,
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = int(listen_port)
        self.control_host = control_host
        self.control_port = int(control_port)
        self.control_timeout_s = float(control_timeout_s)
        self.camera_source = str(camera_source)
        self.fpv_source = str(fpv_source)
        self.fpv_reader = FpvShmReader(fpv_shm_name)
        self.fpv_max_stale_ms = int(fpv_max_stale_ms)
        self.fpv_encoding = str(fpv_encoding).lower()
        self.fpv_jpeg_quality = int(fpv_jpeg_quality)
        self.gmsl_max_group_skew_ms = float(gmsl_max_group_skew_ms)
        self.gmsl_group_timeout_ms = float(gmsl_group_timeout_ms)
        self.fpv_cache = FpvPayloadCache(
            self.fpv_reader,
            fpv_source=self.fpv_source,
            fpv_encoding=self.fpv_encoding,
            jpeg_quality=self.fpv_jpeg_quality,
            max_encode_hz=float(fpv_jpeg_cache_hz),
        )
        if self.camera_source == "fpv":
            self.fpv_cache.start()
        self.gmsl_readers = (
            {
                str(camera_key): FpvShmReader(str(shm_name))
                for camera_key, shm_name in dict(gmsl_cameras or {}).items()
            }
            if self.camera_source == "gmsl"
            else {}
        )
        # GMSL frames must be grouped by V4L2 timestamp before payload encoding.
        # Per-camera JPEG caches are deliberately not used here because they can
        # combine payloads from different exposure batches.
        self.gmsl_caches: dict[str, FpvPayloadCache] = {}
        self.placeholder_width = int(placeholder_width)
        self.placeholder_height = int(placeholder_height)
        self._frame_id = 0
        self._upstream_lock = threading.Lock()
        self._upstream: JsonTcpBridgeClient | None = None

    def _drop_upstream(self) -> None:
        if self._upstream is not None:
            try:
                self._upstream.force_close()
            except Exception:
                pass
            self._upstream = None

    def _ensure_upstream(self) -> JsonTcpBridgeClient:
        if self._upstream is None:
            self._upstream = JsonTcpBridgeClient(
                host=self.control_host,
                port=self.control_port,
                timeout_s=self.control_timeout_s,
                connect_on_init=True,
            )
            log.info(
                "upstream bridge connected %s:%d",
                self.control_host,
                self.control_port,
            )
        return self._upstream

    def _upstream_response(self, request_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._upstream_lock:
            last_exc: Exception | None = None
            for _attempt in range(2):
                try:
                    return self._ensure_upstream()._request_response(request_type, payload)
                except (BridgeProtocolError, OSError, ConnectionError) as exc:
                    last_exc = exc
                    log.warning("upstream %s failed: %s", request_type, exc)
                    self._drop_upstream()
            assert last_exc is not None
            raise last_exc

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        msg_type = str(message.get("type", ""))
        payload = message.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if msg_type == "read_state.request":
            upstream_response = self._upstream_response("read_state", dict(payload))
            if not bool(upstream_response.get("ok", True)):
                return upstream_response
            upstream = upstream_response.get("payload", {})
            if not isinstance(upstream, dict):
                return response_message(
                    "read_state.response",
                    {},
                    ok=False,
                    error="upstream read_state payload must be a mapping",
                )
            self._frame_id += 1
            upstream["images"] = self._camera_samples()
            return response_message("read_state.response", upstream)

        if msg_type.endswith(".request"):
            base = msg_type[: -len(".request")]
            return self._upstream_response(base, dict(payload))

        return response_message("error.response", {}, ok=False, error=f"unsupported type {msg_type}")

    def _fpv_sample(self) -> dict[str, Any]:
        if self.fpv_encoding == "jpeg":
            cached = self.fpv_cache.latest(max_stale_ms=self.fpv_max_stale_ms)
            if cached is not None:
                return cached
        return _fpv_sample_from_shm(
            self.fpv_reader,
            max_stale_ms=self.fpv_max_stale_ms,
            placeholder_width=self.placeholder_width,
            placeholder_height=self.placeholder_height,
            frame_id=self._frame_id,
            fpv_source=self.fpv_source,
            fpv_encoding=self.fpv_encoding,
            jpeg_quality=self.fpv_jpeg_quality,
        )

    def _camera_samples(self) -> dict[str, Any]:
        if self.camera_source == "none":
            return {}
        if self.camera_source == "gmsl":
            return self._gmsl_group_samples()
        return {"fpv": self._fpv_sample()}

    def _gmsl_group_samples(self) -> dict[str, Any]:
        frames, group = self._read_gmsl_timestamp_group()
        samples: dict[str, Any] = {}
        for camera_key, frame in frames.items():
            metadata = _frame_metadata(frame)
            metadata.update(group)
            payload = _fpv_payload(
                rgb=frame.rgb,
                width=frame.width,
                height=frame.height,
                encoding=self.fpv_encoding,
                jpeg_quality=self.fpv_jpeg_quality,
            )
            samples[camera_key] = {
                "timestamp_ns": frame.timestamp_ns,
                "source": f"gmsl_preprocess_shm:{camera_key}:timestamp_group",
                "receive_time_ns": frame.receive_time_ns,
                "payload": payload,
                "metadata": metadata,
            }
        return samples

    def _read_gmsl_timestamp_group(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.gmsl_readers:
            raise BridgeProtocolError("GMSL camera source selected but no cameras are configured")
        max_skew_ns = int(max(0.0, self.gmsl_max_group_skew_ms) * 1_000_000)
        deadline_s = time.monotonic() + max(0.0, self.gmsl_group_timeout_ms) * 0.001
        best_frames: dict[str, Any] | None = None
        best_skew_ns: int | None = None
        best_target_ns = 0
        last_error = "no frames read"

        while True:
            frames: dict[str, Any] = {}
            missing: list[str] = []
            stale: list[str] = []
            for camera_key, reader in self.gmsl_readers.items():
                frame = reader.read_latest()
                if frame is None:
                    missing.append(camera_key)
                    continue
                if not _frame_is_fresh(frame.receive_time_ns, self.fpv_max_stale_ms):
                    stale.append(camera_key)
                    continue
                frames[camera_key] = frame

            if missing or stale:
                parts = []
                if missing:
                    parts.append("missing=" + ",".join(missing))
                if stale:
                    parts.append("stale=" + ",".join(stale))
                last_error = " ".join(parts)
            elif len(frames) == len(self.gmsl_readers):
                timestamps = [
                    int(getattr(frame, "v4l2_timestamp_ns", 0) or 0)
                    for frame in frames.values()
                ]
                if all(timestamp > 0 for timestamp in timestamps):
                    skew_ns = max(timestamps) - min(timestamps)
                    target_ns = max(timestamps)
                    if best_skew_ns is None or skew_ns < best_skew_ns:
                        best_frames = dict(frames)
                        best_skew_ns = int(skew_ns)
                        best_target_ns = int(target_ns)
                    if skew_ns <= max_skew_ns:
                        return dict(frames), _gmsl_group_metadata(
                            target_ns=target_ns,
                            skew_ns=skew_ns,
                            valid=True,
                            camera_count=len(frames),
                        )
                    last_error = f"skew_ns={skew_ns} exceeds max_skew_ns={max_skew_ns}"
                else:
                    last_error = "one or more GMSL frames are missing v4l2_timestamp_ns"

            if time.monotonic() >= deadline_s:
                if best_frames is not None and best_skew_ns is not None:
                    return best_frames, _gmsl_group_metadata(
                        target_ns=best_target_ns,
                        skew_ns=best_skew_ns,
                        valid=False,
                        camera_count=len(best_frames),
                    )
                raise BridgeProtocolError(f"GMSL timestamp group unavailable: {last_error}")
            time.sleep(0.001)

    def _gmsl_sample(self, camera_key: str) -> dict[str, Any]:
        return _fpv_sample_from_shm(
            self.gmsl_readers[camera_key],
            max_stale_ms=self.fpv_max_stale_ms,
            placeholder_width=self.placeholder_width,
            placeholder_height=self.placeholder_height,
            frame_id=self._frame_id,
            fpv_source="shm",
            fpv_encoding=self.fpv_encoding,
            jpeg_quality=self.fpv_jpeg_quality,
            sample_source=f"gmsl_preprocess_shm:{camera_key}",
            stream_name=camera_key,
        )

    def serve_forever(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.listen_host, self.listen_port))
        server.listen(8)
        log.info(
            "Bridge gateway on %s:%d -> control %s:%d",
            self.listen_host,
            self.listen_port,
            self.control_host,
            self.control_port,
        )
        while True:
            conn, addr = server.accept()
            threading.Thread(
                target=self._serve_client,
                args=(conn, addr),
                daemon=True,
            ).start()

    def _serve_client(self, conn: socket.socket, addr: Any) -> None:
        log.info("client %s", addr)
        buffer = b""
        try:
            with conn:
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line.strip():
                            continue
                        try:
                            message = decode_frame(line)
                            response = self.handle_message(message)
                        except Exception as exc:
                            log.exception("request failed")
                            response = response_message(
                                "error.response", {}, ok=False, error=str(exc)
                            )
                        conn.sendall(encode_frame(response))
                        if response.get("type") in {"close.response", "shutdown.response"}:
                            return
        except Exception:
            log.exception("client session error")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8766)
    parser.add_argument("--control-timeout", type=float, default=1.0)
    parser.add_argument("--camera-source", choices=["none", "fpv", "gmsl"], default="fpv")
    parser.add_argument("--fpv-source", choices=["auto", "shm", "placeholder"], default="auto")
    parser.add_argument("--fpv-shm-name", default="excavator_fpv_v1")
    parser.add_argument("--fpv-max-stale-ms", type=int, default=500)
    parser.add_argument("--fpv-encoding", choices=["raw", "jpeg"], default="raw")
    parser.add_argument("--fpv-jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--fpv-jpeg-cache-hz",
        type=float,
        default=30.0,
        help="Maximum JPEG encode rate for the background FPV cache.",
    )
    parser.add_argument(
        "--gmsl-camera",
        action="append",
        default=[],
        metavar="KEY=SHM_NAME",
        help="GMSL camera SHM mapping. Repeatable. Default: video4/video5/video6/video7.",
    )
    parser.add_argument(
        "--gmsl-max-group-skew-ms",
        type=float,
        default=5.0,
        help="Maximum accepted four-camera V4L2 timestamp skew before waiting.",
    )
    parser.add_argument(
        "--gmsl-group-timeout-ms",
        type=float,
        default=50.0,
        help="Maximum wait for a valid GMSL timestamp group before returning the best group.",
    )
    parser.add_argument("--placeholder-width", type=int, default=640)
    parser.add_argument("--placeholder-height", type=int, default=480)
    args = parser.parse_args()
    gmsl_cameras = _parse_gmsl_cameras(args.gmsl_camera)

    BridgeGateway(
        listen_host=args.host,
        listen_port=args.port,
        control_host=args.control_host,
        control_port=args.control_port,
        control_timeout_s=args.control_timeout,
        camera_source=args.camera_source,
        fpv_source=args.fpv_source,
        fpv_shm_name=args.fpv_shm_name,
        fpv_max_stale_ms=args.fpv_max_stale_ms,
        fpv_encoding=args.fpv_encoding,
        fpv_jpeg_quality=args.fpv_jpeg_quality,
        fpv_jpeg_cache_hz=args.fpv_jpeg_cache_hz,
        gmsl_cameras=gmsl_cameras,
        gmsl_max_group_skew_ms=args.gmsl_max_group_skew_ms,
        gmsl_group_timeout_ms=args.gmsl_group_timeout_ms,
        placeholder_width=args.placeholder_width,
        placeholder_height=args.placeholder_height,
    ).serve_forever()


def _parse_gmsl_cameras(values: list[str]) -> dict[str, str]:
    if not values:
        values = [
            "video4=excavator_gmsl_video4",
            "video5=excavator_gmsl_video5",
            "video6=excavator_gmsl_video6",
            "video7=excavator_gmsl_video7",
        ]
    cameras: dict[str, str] = {}
    for raw in values:
        key, sep, shm_name = str(raw).partition("=")
        if not sep or not key or not shm_name:
            raise SystemExit(f"invalid --gmsl-camera {raw!r}; expected KEY=SHM_NAME")
        cameras[key] = shm_name
    return cameras


if __name__ == "__main__":
    main()
