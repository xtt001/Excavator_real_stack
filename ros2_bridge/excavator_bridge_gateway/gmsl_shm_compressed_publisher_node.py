#!/usr/bin/env python3
"""Publish the latest processed GMSL SHM frame as a low-latency JPEG topic."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from excavator_bridge_gateway.fpv_shm import FpvFrame, FpvShmReader


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Read one processed GMSL SHM stream and publish latest-only JPEG frames."
    )
    parser.add_argument("--shm-name", default="excavator_gmsl_video4")
    parser.add_argument(
        "--topic",
        default="/excavator/eye/video4/image_raw/compressed",
    )
    parser.add_argument("--frame-id", default="video4")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    # Keep this slightly above the measured 30.2 Hz source so timer jitter does
    # not turn a nominal 30 Hz stream into ~26 Hz. The SHM sequence still caps
    # publishing to one message per actual camera frame.
    parser.add_argument("--max-publish-hz", type=float, default=35.0)
    parser.add_argument("--poll-hz", type=float, default=120.0)
    parser.add_argument("--max-stale-ms", type=float, default=500.0)
    parser.add_argument("--stats-every-s", type=float, default=5.0)
    return parser.parse_known_args()


def _jpeg_from_rgb(frame: FpvFrame, *, quality: int) -> bytes:
    expected = int(frame.width) * int(frame.height) * 3
    if len(frame.rgb) != expected:
        raise ValueError(
            f"RGB payload size mismatch: got={len(frame.rgb)} expected={expected}"
        )
    rgb = np.frombuffer(frame.rgb, dtype=np.uint8).reshape(
        int(frame.height), int(frame.width), 3
    )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bounded_quality = max(1, min(100, int(quality)))
    ok, encoded = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), bounded_quality]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


class GmslShmCompressedPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("gmsl_video4_compressed_publisher")
        self._reader = FpvShmReader(str(args.shm_name))
        self._topic = str(args.topic)
        self._frame_id = str(args.frame_id)
        self._jpeg_quality = int(args.jpeg_quality)
        self._max_stale_ns = int(max(0.0, float(args.max_stale_ms)) * 1_000_000)
        self._min_publish_period_s = (
            0.0
            if float(args.max_publish_hz) <= 0.0
            else 1.0 / float(args.max_publish_hz)
        )
        self._stats_every_s = float(args.stats_every_s)
        self._last_sequence = -1
        self._last_publish_s = 0.0
        self._stats_start_s = time.monotonic()
        self._published = 0
        self._published_bytes = 0
        self._encode_ms_sum = 0.0
        self._stale_reads = 0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(CompressedImage, self._topic, qos)
        poll_hz = max(1.0, float(args.poll_hz))
        self.create_timer(1.0 / poll_hz, self._publish_latest)
        self.get_logger().info(
            f"latest-only GMSL preview: shm={args.shm_name} -> {self._topic}; "
            f"JPEG quality={self._jpeg_quality}, max={args.max_publish_hz:g} Hz, "
            "QoS=best_effort/depth1"
        )

    def _publish_latest(self) -> None:
        frame = self._reader.read_latest()
        if frame is None or int(frame.sequence) == self._last_sequence:
            self._maybe_log_stats()
            return

        now_s = time.monotonic()
        if (
            self._min_publish_period_s > 0.0
            and now_s - self._last_publish_s < self._min_publish_period_s
        ):
            return
        age_ns = max(0, time.time_ns() - int(frame.receive_time_ns))
        if self._max_stale_ns > 0 and age_ns > self._max_stale_ns:
            self._stale_reads += 1
            self._last_sequence = int(frame.sequence)
            self._maybe_log_stats()
            return

        encode_start_s = time.monotonic()
        try:
            jpeg = _jpeg_from_rgb(frame, quality=self._jpeg_quality)
        except Exception as exc:
            self.get_logger().warning(
                f"video4 JPEG encode failed: {exc}", throttle_duration_sec=2.0
            )
            return
        encode_ms = (time.monotonic() - encode_start_s) * 1000.0

        msg = CompressedImage()
        stamp_ns = max(0, int(frame.timestamp_ns))
        msg.header.stamp.sec = stamp_ns // 1_000_000_000
        msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
        msg.header.frame_id = self._frame_id
        msg.format = "jpeg"
        msg.data = jpeg
        self._publisher.publish(msg)

        self._last_sequence = int(frame.sequence)
        self._last_publish_s = now_s
        self._published += 1
        self._published_bytes += len(jpeg)
        self._encode_ms_sum += encode_ms
        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        now_s = time.monotonic()
        elapsed_s = now_s - self._stats_start_s
        if self._stats_every_s <= 0.0 or elapsed_s < self._stats_every_s:
            return
        hz = self._published / max(elapsed_s, 1e-6)
        bitrate_mbps = self._published_bytes * 8.0 / max(elapsed_s, 1e-6) / 1e6
        encode_ms = self._encode_ms_sum / max(self._published, 1)
        self.get_logger().info(
            f"publish={hz:.1f} Hz bitrate={bitrate_mbps:.2f} Mbps "
            f"encode={encode_ms:.2f} ms stale_reads={self._stale_reads}"
        )
        self._stats_start_s = now_s
        self._published = 0
        self._published_bytes = 0
        self._encode_ms_sum = 0.0
        self._stale_reads = 0


def main() -> None:
    args, ros_args = _parse_args()
    rclpy.init(args=ros_args)
    node = GmslShmCompressedPublisher(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
