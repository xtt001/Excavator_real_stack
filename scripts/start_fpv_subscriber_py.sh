#!/usr/bin/env bash
# 启动 FPV 订阅（写 SHM）+ rqt_image_view 可视化
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export EXCAVATOR_REAL_STACK_ROOT="${ROOT_DIR}"
export EXCAVATOR_ORBBEC_WS="${EXCAVATOR_ORBBEC_WS:-${HOME}/orbbec_ws}"
export EXCAVATOR_ORBBEC_SRC="${EXCAVATOR_ORBBEC_SRC:-${EXCAVATOR_ORBBEC_WS}/src/OrbbecSDK_ROS2}"
export EXCAVATOR_ROS_WS="${EXCAVATOR_ROS_WS:-${HOME}/ros2_ws}"

# rqt 需系统 PyQt5；激活 .venv 时避免 env python3 覆盖 /usr/bin
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  export PATH="/usr/bin:/bin:${PATH}"
  unset VIRTUAL_ENV
fi

set +u
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/source_ros_stack.sh"
set -u

exec ros2 launch excavator_ros2_bridge fpv_subscriber_with_rqt.launch.py "$@"
