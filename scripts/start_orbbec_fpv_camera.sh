#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${EXCAVATOR_ROS_WS:-/home/yxc/ros2_ws}"

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
if [[ -f "${WS}/install/setup.bash" ]]; then
  source "${WS}/install/setup.bash"
else
  echo "Missing ${WS}/install/setup.bash — build excavator_ros2_bridge first." >&2
  exit 1
fi
set -u

exec ros2 launch excavator_ros2_bridge orbbec_fpv_camera.launch.py "$@"
