"""主端：compressed(BEST_EFFORT) -> raw -> rqt。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    compressed = LaunchConfiguration("compressed_topic").perform(context)
    display = LaunchConfiguration("display_topic").perform(context)

    republish = Node(
        package="image_transport",
        executable="republish",
        name="host_fpv_compressed_to_raw",
        output="screen",
        parameters=[
            {
                "qos_overrides": {
                    compressed: {
                        "subscription": {
                            "reliability": "best_effort",
                            "history": "keep_last",
                            "depth": 1,
                        },
                    },
                },
            },
        ],
        arguments=["compressed", "raw"],
        remappings=[
            ("in/compressed", compressed),
            ("out", display),
        ],
    )

    rqt = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "/usr/bin/python3",
                    "/opt/ros/humble/lib/rqt_image_view/rqt_image_view",
                    display,
                ],
                output="screen",
            )
        ],
    )
    return [republish, rqt]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "compressed_topic",
                default_value="/camera/color/image_raw/compressed",
            ),
            DeclareLaunchArgument(
                "display_topic",
                default_value="/fpv/host_display/image_raw",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
