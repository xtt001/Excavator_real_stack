#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# OrbbecSDK_ROS2: ~/orbbec_ws/src/OrbbecSDK_ROS2 -> 包名 orbbec_camera
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_fpv_env.sh"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/excavator_deploy_network.sh"
excavator_apply_slave_network_defaults
export EXCAVATOR_ORBBEC_WS="${EXCAVATOR_ORBBEC_WS:-${HOME}/orbbec_ws}"
export EXCAVATOR_ORBBEC_SRC="${EXCAVATOR_ORBBEC_SRC:-${EXCAVATOR_ORBBEC_WS}/src/OrbbecSDK_ROS2}"
export EXCAVATOR_ROS_WS="${EXCAVATOR_ROS_WS:-${EXCAVATOR_ORBBEC_WS}}"

set +u
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/source_ros_stack.sh"
set -u

exec ros2 launch excavator_ros2_bridge orbbec_fpv_camera.launch.py "$@"
