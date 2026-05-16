"""独立启动 Orbbec 相机（仅彩色 640x480@30，compressed 发布）。"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("excavator_ros2_bridge")
    orbbec_share = get_package_share_directory("orbbec_camera")
    default_config = os.path.join(pkg_share, "config", "orbbec_fpv.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_model",
                default_value="gemini330_series",
                description="Orbbec camera model yaml base name",
            ),
            DeclareLaunchArgument(
                "config_file_path",
                default_value=default_config,
                description="Excavator FPV overlay config",
            ),
            DeclareLaunchArgument("camera_name", default_value="camera"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(orbbec_share, "launch", "orbbec_camera.launch.py")
                ),
                launch_arguments={
                    "camera_model": LaunchConfiguration("camera_model"),
                    "config_file_path": LaunchConfiguration("config_file_path"),
                    "camera_name": LaunchConfiguration("camera_name"),
                }.items(),
            ),
        ]
    )
