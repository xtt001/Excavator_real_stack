#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_fpv_env.sh"
export EXCAVATOR_ORBBEC_WS="${EXCAVATOR_ORBBEC_WS:-${HOME}/orbbec_ws}"
export EXCAVATOR_ORBBEC_SRC="${EXCAVATOR_ORBBEC_SRC:-${EXCAVATOR_ORBBEC_WS}/src/OrbbecSDK_ROS2}"
export EXCAVATOR_ROS_WS="${EXCAVATOR_ROS_WS:-${HOME}/ros2_ws}"

set +u
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/source_ros_stack.sh"
set -u

exec ros2 launch excavator_ros2_bridge fpv_image_subscriber.launch.py "$@"
