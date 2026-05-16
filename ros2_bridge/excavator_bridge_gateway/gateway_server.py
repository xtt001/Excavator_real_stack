#!/usr/bin/env python3
"""
JSON/TCP 网关：testbed 连此端口；控制请求转发到 excavator_real_bridge；
read_state 中用 ROS FPV 共享内存替换占位图。不修改 bridge/src/excavator_real_bridge.cpp。
"""

from __future__ import annotations

import argparse
import base64
import logging
import socket
import threading
import time
from typing import Any

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


def _fpv_sample_from_shm(
    reader: FpvShmReader,
    *,
    max_stale_ms: int,
    placeholder_width: int,
    placeholder_height: int,
    frame_id: int,
    fpv_source: str,
) -> dict[str, Any]:
    use_shm = fpv_source in {"auto", "shm"}
    allow_placeholder = fpv_source in {"auto", "placeholder"}

    if use_shm and reader.is_fresh(max_stale_ms):
        frame = reader.read_latest()
        if frame is not None:
            return {
                "timestamp_ns": frame.timestamp_ns,
                "source": "ros2_compressed_fpv",
                "receive_time_ns": frame.receive_time_ns,
                "payload": {
                    "encoding": "raw_uint8",
                    "shape": [frame.height, frame.width, 3],
                    "data_b64": base64.b64encode(frame.rgb).decode("ascii"),
                },
            }

    if not allow_placeholder:
        raise BridgeProtocolError("fpv shm unavailable and placeholder disabled")

    ts = time.time_ns()
    return {
        "timestamp_ns": ts,
        "source": "bridge_placeholder_fpv",
        "receive_time_ns": ts,
        "payload": _placeholder_fpv(placeholder_width, placeholder_height, frame_id),
    }


class BridgeGateway:
    def __init__(
        self,
        *,
        listen_host: str,
        listen_port: int,
        control_host: str,
        control_port: int,
        control_timeout_s: float,
        fpv_source: str,
        fpv_shm_name: str,
        fpv_max_stale_ms: int,
        placeholder_width: int,
        placeholder_height: int,
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = int(listen_port)
        self.control_host = control_host
        self.control_port = int(control_port)
        self.control_timeout_s = float(control_timeout_s)
        self.fpv_source = str(fpv_source)
        self.fpv_reader = FpvShmReader(fpv_shm_name)
        self.fpv_max_stale_ms = int(fpv_max_stale_ms)
        self.placeholder_width = int(placeholder_width)
        self.placeholder_height = int(placeholder_height)
        self._frame_id = 0
        self._upstream_lock = threading.Lock()

    def _upstream_request(self, request_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._upstream_lock:
            client = JsonTcpBridgeClient(
                host=self.control_host,
                port=self.control_port,
                timeout_s=self.control_timeout_s,
                connect_on_init=True,
            )
            try:
                return client._request(request_type, payload)
            finally:
                client.close()

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        msg_type = str(message.get("type", ""))
        payload = message.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if msg_type == "read_state.request":
            upstream = self._upstream_request("read_state", dict(payload))
            self._frame_id += 1
            upstream["images"] = {
                "fpv": _fpv_sample_from_shm(
                    self.fpv_reader,
                    max_stale_ms=self.fpv_max_stale_ms,
                    placeholder_width=self.placeholder_width,
                    placeholder_height=self.placeholder_height,
                    frame_id=self._frame_id,
                    fpv_source=self.fpv_source,
                )
            }
            return response_message("read_state.response", upstream)

        if msg_type.endswith(".request"):
            base = msg_type[: -len(".request")]
            upstream = self._upstream_request(base, dict(payload))
            return response_message(f"{base}.response", upstream)

        return response_message("error.response", {}, ok=False, error=f"unsupported type {msg_type}")

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
    parser.add_argument("--fpv-source", choices=["auto", "shm", "placeholder"], default="auto")
    parser.add_argument("--fpv-shm-name", default="excavator_fpv_v1")
    parser.add_argument("--fpv-max-stale-ms", type=int, default=500)
    parser.add_argument("--placeholder-width", type=int, default=640)
    parser.add_argument("--placeholder-height", type=int, default=480)
    args = parser.parse_args()

    BridgeGateway(
        listen_host=args.host,
        listen_port=args.port,
        control_host=args.control_host,
        control_port=args.control_port,
        control_timeout_s=args.control_timeout,
        fpv_source=args.fpv_source,
        fpv_shm_name=args.fpv_shm_name,
        fpv_max_stale_ms=args.fpv_max_stale_ms,
        placeholder_width=args.placeholder_width,
        placeholder_height=args.placeholder_height,
    ).serve_forever()


if __name__ == "__main__":
    main()
