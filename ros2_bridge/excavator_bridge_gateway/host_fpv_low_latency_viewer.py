#!/usr/bin/env python3
"""Low-latency host viewer with parallel receive, latest-only decode and GUI."""

from __future__ import annotations

import argparse
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


def _stamp_ns(msg: CompressedImage) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


class LatestCompressedViewer(Node):
    def __init__(
        self,
        topic: str,
        *,
        node_name: str = "host_fpv_low_latency_viewer",
    ) -> None:
        super().__init__(str(node_name))
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._condition = threading.Condition()
        self._latest: tuple[int, int, bytes] | None = None
        self._seq = 0
        self._received = 0
        self.create_subscription(CompressedImage, topic, self._on_image, qos)
        self.get_logger().info(f"viewing {topic} with best_effort depth=1")

    def _on_image(self, msg: CompressedImage) -> None:
        data = bytes(msg.data)
        stamp_ns = _stamp_ns(msg)
        with self._condition:
            self._seq += 1
            self._received += 1
            self._latest = (self._seq, stamp_ns, data)
            self._condition.notify()

    def wait_latest(
        self, after_seq: int, *, timeout_s: float
    ) -> tuple[int, int, bytes] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._latest is not None and self._latest[0] != after_seq,
                timeout=max(0.0, float(timeout_s)),
            )
            return self._latest

    def received_count(self) -> int:
        with self._condition:
            return self._received

    def wake_decoder(self) -> None:
        with self._condition:
            self._condition.notify_all()


class LatestJpegDecoder:
    """Decode outside the ROS and GUI threads, retaining only the newest frame."""

    def __init__(self, node: LatestCompressedViewer, *, scale: float) -> None:
        self._node = node
        self._scale = float(scale)
        self._lock = threading.Lock()
        self._latest: tuple[int, int, np.ndarray, float] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="host-fpv-jpeg-decoder",
            daemon=True,
        )
        self._decoded = 0
        self._input_skipped = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._node.wake_decoder()
        self._thread.join(timeout=1.0)

    def latest(self) -> tuple[int, int, np.ndarray, float] | None:
        with self._lock:
            return self._latest

    def stats(self) -> tuple[int, int]:
        with self._lock:
            return self._decoded, self._input_skipped

    def _run(self) -> None:
        last_seq = 0
        while not self._stop.is_set():
            latest = self._node.wait_latest(last_seq, timeout_s=0.1)
            if latest is None:
                continue
            seq, stamp_ns, data = latest
            if seq == last_seq:
                continue
            skipped = max(0, int(seq) - int(last_seq) - 1) if last_seq else 0
            last_seq = int(seq)

            decode_start_s = time.monotonic()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if self._scale != 1.0:
                frame = cv2.resize(
                    frame,
                    None,
                    fx=self._scale,
                    fy=self._scale,
                    interpolation=(
                        cv2.INTER_AREA if self._scale < 1.0 else cv2.INTER_LINEAR
                    ),
                )
            decode_ms = (time.monotonic() - decode_start_s) * 1000.0
            with self._lock:
                self._latest = (int(seq), int(stamp_ns), frame, decode_ms)
                self._decoded += 1
                self._input_skipped += skipped


def _spin(node: Node) -> None:
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic",
        default="/camera/color/image_raw/compressed",
        help="CompressedImage topic to view.",
    )
    parser.add_argument("--window-name", default="Excavator FPV")
    parser.add_argument("--max-display-hz", type=float, default=60.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--stats-every-s", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rclpy.init()
    node = LatestCompressedViewer(str(args.topic))
    spin_thread = threading.Thread(target=_spin, args=(node,), daemon=True)
    spin_thread.start()
    decoder = LatestJpegDecoder(node, scale=float(args.scale))
    decoder.start()

    cv2.namedWindow(str(args.window_name), cv2.WINDOW_NORMAL)
    min_period_s = 0.0 if args.max_display_hz <= 0 else 1.0 / float(args.max_display_hz)
    last_display_s = 0.0
    last_stats_s = time.monotonic()
    last_stats_received = 0
    last_stats_decoded = 0
    last_stats_skipped = 0
    last_seq = 0
    shown = 0

    try:
        while rclpy.ok():
            now_s = time.monotonic()
            if min_period_s > 0 and now_s - last_display_s < min_period_s:
                cv2.waitKey(1)
                time.sleep(0.001)
                continue

            latest = decoder.latest()
            if latest is not None:
                seq, stamp_ns, frame, decode_ms = latest
                if seq != last_seq:
                    cv2.imshow(str(args.window_name), frame)
                    shown += 1
                    last_seq = seq
                    last_display_s = now_s

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if args.stats_every_s > 0 and now_s - last_stats_s >= args.stats_every_s:
                recv = node.received_count()
                decoded, skipped = decoder.stats()
                dt = max(now_s - last_stats_s, 1e-6)
                recv_hz = (recv - last_stats_received) / dt
                decode_hz = (decoded - last_stats_decoded) / dt
                display_hz = shown / dt
                age_text = ""
                decode_text = ""
                latest_for_age = decoder.latest()
                if latest_for_age is not None:
                    if latest_for_age[1] > 0:
                        age_ms = (time.time_ns() - latest_for_age[1]) / 1_000_000.0
                        age_text = f" latest_age_ms={age_ms:.1f}"
                    decode_text = f" decode_ms={latest_for_age[3]:.1f}"
                node.get_logger().info(
                    f"recv={recv_hz:.1f} Hz decode={decode_hz:.1f} Hz "
                    f"display={display_hz:.1f} Hz skipped={skipped - last_stats_skipped}"
                    f"{decode_text}{age_text}"
                )
                last_stats_s = now_s
                last_stats_received = recv
                last_stats_decoded = decoded
                last_stats_skipped = skipped
                shown = 0
    finally:
        decoder.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
