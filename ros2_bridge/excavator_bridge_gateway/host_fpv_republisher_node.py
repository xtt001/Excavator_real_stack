#!/usr/bin/env python3
"""主端：订 /camera/color/image_raw/compressed，发布 /camera/color/image_raw 供 rqt。"""

from __future__ import annotations

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


class HostFpvRepublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("host_fpv_republisher")
        compressed = self.declare_parameter(
            "compressed_topic", "/camera/color/image_raw/compressed"
        ).value
        raw = self.declare_parameter("raw_topic", "/camera/color/image_raw").value
        self._bridge = CvBridge()
        self._pub = self.create_publisher(Image, str(raw), qos_profile_sensor_data)
        self.create_subscription(
            CompressedImage,
            str(compressed),
            self._on_compressed,
            qos_profile_sensor_data,
        )
        self._frames = 0
        self.create_timer(5.0, self._log_fps)
        self.get_logger().info(f"{compressed} -> {raw} (for rqt)")

    def _on_compressed(self, msg: CompressedImage) -> None:
        try:
            raw = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            out = self._bridge.cv2_to_imgmsg(raw, encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(
                f"decode failed: {exc}", throttle_duration_sec=2.0
            )
            return
        out.header = msg.header
        self._pub.publish(out)
        self._frames += 1

    def _log_fps(self) -> None:
        if self._frames:
            self.get_logger().info(f"republish ~{self._frames / 5.0:.1f} Hz")
        else:
            self.get_logger().warning("no frames from compressed yet")
        self._frames = 0


def main() -> None:
    rclpy.init()
    node = HostFpvRepublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
