#!/usr/bin/env bash
# 加载 ROS2 + excavator_ros2_bridge（不要求本机有 Orbbec 源码/安装）。
# 用于 FPV 订阅、主端 rqt；相机由 start_orbbec_fpv_camera.sh 在从端单独启动。
# 用法: source scripts/source_excavator_ros_ws.sh

_stack_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${EXCAVATOR_ROS2_MULTIHOST:-0}" == "1" ]]; then
  # shellcheck disable=SC1091
  source "${_stack_root}/scripts/ros2_multihost_env.sh"
fi

_excavator_ws="${EXCAVATOR_ROS_WS:-${HOME}/ros2_ws}"
_orbbec_ws="${EXCAVATOR_ORBBEC_WS:-${HOME}/orbbec_ws}"
_ros_distro="${ROS_DISTRO:-humble}"

if [[ ! -f "/opt/ros/${_ros_distro}/setup.bash" ]]; then
  echo "source_excavator_ros_ws: missing /opt/ros/${_ros_distro}/setup.bash" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "/opt/ros/${_ros_distro}/setup.bash"

_sourced=0
if [[ -f "${_excavator_ws}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${_excavator_ws}/install/setup.bash"
  _sourced=1
elif [[ -f "${_orbbec_ws}/install/setup.bash" ]]; then
  # excavator_ros2_bridge 常与 orbbec 同 workspace
  # shellcheck disable=SC1090
  source "${_orbbec_ws}/install/setup.bash"
  _sourced=1
fi

if [[ "${_sourced}" -eq 0 ]]; then
  echo "source_excavator_ros_ws: no install/setup.bash in:" >&2
  echo "  EXCAVATOR_ROS_WS=${_excavator_ws}" >&2
  echo "  EXCAVATOR_ORBBEC_WS=${_orbbec_ws}" >&2
  echo "  Build: ln -sf .../excavator_ros2_bridge <ws>/src/ && colcon build --packages-select excavator_ros2_bridge" >&2
  return 1 2>/dev/null || exit 1
fi

export EXCAVATOR_ROS_WS="${_excavator_ws}"
export EXCAVATOR_ORBBEC_WS="${_orbbec_ws}"
