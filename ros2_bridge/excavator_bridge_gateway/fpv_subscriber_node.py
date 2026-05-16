#!/usr/bin/env python3
"""订阅 Orbbec compressed 彩色图并写入 FPV 共享内存（独立进程，不修改 C++ bridge）。"""

from __future__ import annotations

import time

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

from excavator_bridge_gateway.fpv_shm import FpvShmWriter


def _stamp_ns(msg: CompressedImage) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


class FpvSubscriberNode(Node):
    def __init__(self) -> None:
        super().__init__("fpv_image_subscriber")
        topic = self.declare_parameter(
            "compressed_topic", "/camera/color/image_raw/compressed"
        ).value
        shm_name = self.declare_parameter("shm_name", "excavator_fpv_v1").value
        self._writer = FpvShmWriter(str(shm_name))
        self._bridge = CvBridge()
        self.create_subscription(
            CompressedImage, str(topic), self._on_image, qos_profile_sensor_data
        )
        self.get_logger().info(f"FPV subscriber: {topic} -> shm {shm_name}")

    def _on_image(self, msg: CompressedImage) -> None:
        try:
            bgr = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(
                f"decode failed: {exc}", throttle_duration_sec=2.0
            )
            return
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ok = self._writer.write_rgb(
            rgb.tobytes(),
            int(rgb.shape[1]),
            int(rgb.shape[0]),
            _stamp_ns(msg),
            time.time_ns(),
        )
        if not ok:
            self.get_logger().warning(
                f"reject frame shape={rgb.shape}", throttle_duration_sec=2.0
            )


def main() -> None:
    rclpy.init()
    node = FpvSubscriberNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
