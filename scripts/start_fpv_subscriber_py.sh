#!/usr/bin/env bash
# 启动 FPV 订阅（写 SHM）+ rqt_image_view 可视化
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${EXCAVATOR_ROS_WS:-/home/yxc/ros2_ws}"

export EXCAVATOR_REAL_STACK_ROOT="${ROOT_DIR}"

# rqt 需系统 PyQt5；激活 .venv 时避免 env python3 覆盖 /usr/bin
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  export PATH="/usr/bin:/bin:${PATH}"
  unset VIRTUAL_ENV
fi

# colcon setup.bash 会引用未定义变量（如 COLCON_TRACE），与 set -u 冲突
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
if [[ -f "${WS}/install/setup.bash" ]]; then
  source "${WS}/install/setup.bash"
else
  echo "warn: ${WS}/install/setup.bash not found; build excavator_ros2_bridge first." >&2
fi
set -u

exec ros2 launch excavator_ros2_bridge fpv_subscriber_with_rqt.launch.py "$@"
