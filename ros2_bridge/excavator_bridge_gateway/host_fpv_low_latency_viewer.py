#!/usr/bin/env python3
"""Low-latency host FPV viewer: subscribe compressed and display latest frame."""

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
    def __init__(self, topic: str) -> None:
        super().__init__("host_fpv_low_latency_viewer")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._lock = threading.Lock()
        self._latest: tuple[int, int, bytes] | None = None
        self._seq = 0
        self._received = 0
        self.create_subscription(CompressedImage, topic, self._on_image, qos)
        self.get_logger().info(f"viewing {topic} with best_effort depth=1")

    def _on_image(self, msg: CompressedImage) -> None:
        data = bytes(msg.data)
        stamp_ns = _stamp_ns(msg)
        with self._lock:
            self._seq += 1
            self._received += 1
            self._latest = (self._seq, stamp_ns, data)

    def latest(self) -> tuple[int, int, bytes] | None:
        with self._lock:
            return self._latest

    def received_count(self) -> int:
        with self._lock:
            return self._received


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

    cv2.namedWindow(str(args.window_name), cv2.WINDOW_NORMAL)
    min_period_s = 0.0 if args.max_display_hz <= 0 else 1.0 / float(args.max_display_hz)
    last_display_s = 0.0
    last_stats_s = time.monotonic()
    last_stats_frames = 0
    last_seq = 0
    shown = 0

    try:
        while rclpy.ok():
            now_s = time.monotonic()
            if min_period_s > 0 and now_s - last_display_s < min_period_s:
                cv2.waitKey(1)
                time.sleep(0.001)
                continue

            latest = node.latest()
            if latest is not None:
                seq, stamp_ns, data = latest
                if seq != last_seq:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        if args.scale != 1.0:
                            frame = cv2.resize(
                                frame,
                                None,
                                fx=float(args.scale),
                                fy=float(args.scale),
                                interpolation=cv2.INTER_AREA,
                            )
                        cv2.imshow(str(args.window_name), frame)
                        shown += 1
                        last_seq = seq
                        last_display_s = now_s

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if args.stats_every_s > 0 and now_s - last_stats_s >= args.stats_every_s:
                recv = node.received_count()
                dt = max(now_s - last_stats_s, 1e-6)
                recv_hz = (recv - last_stats_frames) / dt
                display_hz = shown / dt
                age_text = ""
                latest_for_age = node.latest()
                if latest_for_age is not None and latest_for_age[1] > 0:
                    age_ms = (time.time_ns() - latest_for_age[1]) / 1_000_000.0
                    age_text = f" latest_age_ms={age_ms:.1f}"
                node.get_logger().info(
                    f"recv={recv_hz:.1f} Hz display={display_hz:.1f} Hz{age_text}"
                )
                last_stats_s = now_s
                last_stats_frames = recv
                shown = 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
