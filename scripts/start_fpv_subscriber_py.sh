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
# 仅需 excavator_ros2_bridge；不要求本机 ~/orbbec_ws（相机可在从端另一进程）
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/source_excavator_ros_ws.sh"
set -u

echo "【从端】仅订阅 compressed 写 SHM，不在此机起 rqt。"
echo "【主端】在操作员 PC 上执行: ./scripts/start_host_fpv_rqt.sh"
echo "       （主端经 DDS 订从端 /camera/color/image_raw/compressed 并用 rqt 显示）"
echo

exec ros2 launch excavator_ros2_bridge fpv_subscriber_with_rqt.launch.py \
  use_rqt:=false \
  use_republish:=false \
  "$@"
