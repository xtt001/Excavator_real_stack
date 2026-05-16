#!/usr/bin/env bash
# 加载 ROS2 + OrbbecSDK_ROS2（~/orbbec_ws）+ excavator_ros2_bridge 叠加工作空间。
# 用法: source scripts/source_ros_stack.sh
# 主从同域: export EXCAVATOR_ROS2_MULTIHOST=1 后再 source

_stack_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${EXCAVATOR_ROS2_MULTIHOST:-0}" == "1" ]]; then
  # shellcheck disable=SC1091
  source "${_stack_root}/scripts/ros2_multihost_env.sh"
fi

_orbbec_ws="${EXCAVATOR_ORBBEC_WS:-${HOME}/orbbec_ws}"
_excavator_ws="${EXCAVATOR_ROS_WS:-${HOME}/ros2_ws}"
_orbbec_src="${EXCAVATOR_ORBBEC_SRC:-${_orbbec_ws}/src/OrbbecSDK_ROS2}"
_ros_distro="${ROS_DISTRO:-humble}"

if [[ ! -f "/opt/ros/${_ros_distro}/setup.bash" ]]; then
  echo "source_ros_stack: missing /opt/ros/${_ros_distro}/setup.bash" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "/opt/ros/${_ros_distro}/setup.bash"

if [[ ! -d "${_orbbec_src}" ]]; then
  echo "source_ros_stack: Orbbec source not found: ${_orbbec_src}" >&2
  echo "  Clone OrbbecSDK_ROS2 into orbbec_ws/src, then: cd ${_orbbec_ws} && colcon build" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "${_orbbec_ws}/install/setup.bash" ]]; then
  echo "source_ros_stack: missing ${_orbbec_ws}/install/setup.bash" >&2
  echo "  Build: cd ${_orbbec_ws} && colcon build --symlink-install --packages-select orbbec_camera" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "${_orbbec_ws}/install/setup.bash"

if [[ -f "${_excavator_ws}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${_excavator_ws}/install/setup.bash"
else
  echo "source_ros_stack: warn: ${_excavator_ws}/install/setup.bash not found;" >&2
  echo "  symlink excavator_ros2_bridge into a colcon ws and build, or set EXCAVATOR_ROS_WS." >&2
fi

export EXCAVATOR_ORBBEC_WS="${_orbbec_ws}"
export EXCAVATOR_ORBBEC_SRC="${_orbbec_src}"
export EXCAVATOR_ROS_WS="${_excavator_ws}"
