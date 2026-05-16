#!/usr/bin/env bash
# 主端：订阅从端 compressed（BEST_EFFORT）-> republish raw -> rqt
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/excavator_deploy_network.sh"
excavator_apply_host_network_defaults

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_fpv_env.sh"

COMPRESSED_TOPIC="${EXCAVATOR_FPV_COMPRESSED_TOPIC:-/camera/color/image_raw/compressed}"
DISPLAY_TOPIC="${EXCAVATOR_FPV_DISPLAY_TOPIC:-/fpv/host_display/image_raw}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  export PATH="/usr/bin:/bin:${PATH}"
  unset VIRTUAL_ENV
fi

set +u
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_multihost_env.sh"
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
set -u

ros2 daemon stop >/dev/null 2>&1 || true

LAUNCH_FILE="${ROOT_DIR}/ros2_bridge/excavator_ros2_bridge/launch/host_fpv_rqt.launch.py"
echo "Host FPV rqt: ${COMPRESSED_TOPIC} -> ${DISPLAY_TOPIC} (ROS_DOMAIN_ID=${ROS_DOMAIN_ID}, RMW=${RMW_IMPLEMENTATION})"
echo "QoS: subscription best_effort (match Orbbec sensor_data)"

exec ros2 launch "${LAUNCH_FILE}" \
  compressed_topic:="${COMPRESSED_TOPIC}" \
  display_topic:="${DISPLAY_TOPIC}"
