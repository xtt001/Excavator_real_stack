"""独立启动 compressed 图像订阅节点，写入 FPV 共享内存。"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("excavator_ros2_bridge")
    default_params = os.path.join(pkg_share, "config", "fpv_subscriber.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument(
                "compressed_topic",
                default_value="/camera/color/image_raw/compressed",
            ),
            DeclareLaunchArgument("shm_name", default_value="excavator_fpv_v1"),
            Node(
                package="excavator_ros2_bridge",
                executable="fpv_image_subscriber",
                name="fpv_image_subscriber",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "compressed_topic": LaunchConfiguration("compressed_topic"),
                        "shm_name": LaunchConfiguration("shm_name"),
                    },
                ],
            ),
        ]
    )
