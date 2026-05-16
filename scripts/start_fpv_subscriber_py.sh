#!/usr/bin/env bash
# 从端：订阅彩色 compressed -> 写 SHM（供 gateway 录制）；不在从端起 rqt。
# 主端看图请用 ./scripts/start_host_fpv_rqt.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export EXCAVATOR_REAL_STACK_ROOT="${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_fpv_env.sh"
export EXCAVATOR_ORBBEC_WS="${EXCAVATOR_ORBBEC_WS:-${HOME}/orbbec_ws}"
export EXCAVATOR_ORBBEC_SRC="${EXCAVATOR_ORBBEC_SRC:-${EXCAVATOR_ORBBEC_WS}/src/OrbbecSDK_ROS2}"
export EXCAVATOR_ROS_WS="${EXCAVATOR_ROS_WS:-${HOME}/ros2_ws}"

set +u
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/source_ros_stack.sh"
set -u

exec ros2 launch excavator_ros2_bridge fpv_subscriber_with_rqt.launch.py \
  use_rqt:=false \
  use_republish:=false \
  "$@"
