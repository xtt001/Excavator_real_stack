"""主端：compressed -> raw（Python）-> rqt。图源始终为 /camera/color/image_raw/compressed。"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration


def _resolve_stack_root() -> str:
    env_root = os.environ.get("EXCAVATOR_REAL_STACK_ROOT", "").strip()
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(launch_dir, "..", "..", ".."))


def _raw_topic(compressed: str) -> str:
    if compressed.endswith("/compressed"):
        return compressed[: -len("/compressed")]
    return "/camera/color/image_raw"


def _setup(context, *args, **kwargs):
    compressed = LaunchConfiguration("compressed_topic").perform(context)
    raw = _raw_topic(compressed)
    stack_root = _resolve_stack_root()
    py_script = os.path.join(
        stack_root, "ros2_bridge", "excavator_bridge_gateway", "host_fpv_republisher_node.py"
    )
    if not os.path.isfile(py_script):
        raise RuntimeError(f"host republisher not found: {py_script}")

    env = dict(os.environ)
    py_path = os.path.join(stack_root, "ros2_bridge")
    env["PYTHONPATH"] = os.pathsep.join([p for p in (py_path, env.get("PYTHONPATH", "")) if p])

    republisher = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            py_script,
            "--ros-args",
            "-p",
            f"compressed_topic:={compressed}",
            "-p",
            f"raw_topic:={raw}",
        ],
        output="screen",
        additional_env=env,
    )

    # rqt 只列出 sensor_msgs/Image，故下拉无 /compressed；选 /camera/color/image_raw
    rqt = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "/usr/bin/python3",
                    "/opt/ros/humble/lib/rqt_image_view/rqt_image_view",
                    raw,
                ],
                output="screen",
            )
        ],
    )
    return [republisher, rqt]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "compressed_topic",
                default_value="/camera/color/image_raw/compressed",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
