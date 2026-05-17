"""
从端 FPV：订阅 compressed 写 SHM（供 gateway / 从端 HDF5）。

默认不在从端开 rqt（主端用 scripts/start_host_fpv_rqt.sh 订同一 compressed 看图）。
单机调试可传: use_rqt:=true use_republish:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_stack_root() -> str:
    env_root = os.environ.get("EXCAVATOR_REAL_STACK_ROOT", "").strip()
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    # 开发态：launch 位于 ros2_bridge/excavator_ros2_bridge/launch/
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(launch_dir, "..", "..", ".."))


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory("excavator_ros2_bridge")
    default_params = os.path.join(pkg_share, "config", "fpv_subscriber.yaml")
    params_file = LaunchConfiguration("params_file").perform(context) or default_params

    compressed_topic = LaunchConfiguration("compressed_topic").perform(context)
    shm_name = LaunchConfiguration("shm_name").perform(context)
    display_topic_cfg = LaunchConfiguration("display_topic").perform(context).strip()
    subscriber_backend = LaunchConfiguration("subscriber_backend").perform(context)
    use_rqt = LaunchConfiguration("use_rqt").perform(context).lower()
    use_republish = LaunchConfiguration("use_republish").perform(context).lower()

    actions = []
    # 默认与 Orbbec 一致：compressed 旁路 raw 为 /camera/color/image_raw
    if display_topic_cfg:
        display_topic = display_topic_cfg
    elif compressed_topic.endswith("/compressed"):
        display_topic = compressed_topic[: -len("/compressed")]
    else:
        display_topic = "/camera/color/image_raw"

    if use_republish in ("1", "true", "yes", "on"):
        actions.append(
            Node(
                package="image_transport",
                executable="republish",
                name="fpv_compressed_to_raw",
                output="screen",
                parameters=[
                    {
                        "qos_overrides": {
                            compressed_topic: {
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
                    ("in/compressed", compressed_topic),
                    ("out", display_topic),
                ],
            )
        )

    if use_rqt in ("1", "true", "yes", "on"):
        # rqt 依赖系统 PyQt5；勿用 .venv 的 python3
        rqt_bin = "/opt/ros/humble/lib/rqt_image_view/rqt_image_view"
        actions.append(
            ExecuteProcess(
                cmd=["/usr/bin/python3", rqt_bin, display_topic],
                output="screen",
            )
        )

    if subscriber_backend == "cpp":
        actions.append(
            Node(
                package="excavator_ros2_bridge",
                executable="fpv_image_subscriber",
                name="fpv_image_subscriber",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "compressed_topic": compressed_topic,
                        "shm_name": shm_name,
                    },
                ],
            )
        )
    else:
        stack_root = _resolve_stack_root()
        py_script = os.path.join(
            stack_root,
            "ros2_bridge",
            "excavator_bridge_gateway",
            "fpv_subscriber_node.py",
        )
        if not os.path.isfile(py_script):
            raise RuntimeError(
                f"Python subscriber not found: {py_script}. "
                "Set EXCAVATOR_REAL_STACK_ROOT or use subscriber_backend:=cpp"
            )
        env = dict(os.environ)
        py_path = os.path.join(stack_root, "ros2_bridge")
        env["PYTHONPATH"] = os.pathsep.join([p for p in (py_path, env.get("PYTHONPATH", "")) if p])
        actions.append(
            ExecuteProcess(
                cmd=[
                    "/usr/bin/python3",
                    py_script,
                    "--ros-args",
                    "-p",
                    f"compressed_topic:={compressed_topic}",
                    "-p",
                    f"shm_name:={shm_name}",
                ],
                output="screen",
                additional_env=env,
            )
        )

    return actions


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
            DeclareLaunchArgument(
                "display_topic",
                default_value="",
                description="空则自动取 compressed 去掉 /compressed 后缀",
            ),
            DeclareLaunchArgument(
                "subscriber_backend",
                default_value="python",
                description="python（默认）或 cpp（需 colcon 编译 fpv_image_subscriber）",
            ),
            DeclareLaunchArgument("use_rqt", default_value="false"),
            DeclareLaunchArgument("use_republish", default_value="false"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
