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
# rqt 看解码后的 /camera/color/image_raw（由 republish 从 compressed 生成）
RAW_TOPIC="${COMPRESSED_TOPIC%/compressed}"

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
echo "【主端】图源: ${COMPRESSED_TOPIC}（ros2 topic list 里能看到）"
echo "  解码发布: ${RAW_TOPIC}；rqt 下拉里只有 ${RAW_TOPIC}（无 compressed 是类型限制）"
echo "  ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW=${RMW_IMPLEMENTATION} peer=${EXCAVATOR_ROS_PEER_IP:-未设置}"
echo "  QoS: subscription best_effort（与 Orbbec sensor_data 一致）"
if [[ -z "${EXCAVATOR_ROS_PEER_IP:-}" ]]; then
  echo "warn: 未设置 EXCAVATOR_ROS_PEER_IP，主端可能发现不了从端话题；请先:" >&2
  echo "  export EXCAVATOR_SLAVE_IP=192.168.31.171" >&2
  echo "  source scripts/excavator_deploy_network.sh && excavator_apply_host_network_defaults" >&2
fi

exec ros2 launch "${LAUNCH_FILE}" compressed_topic:="${COMPRESSED_TOPIC}"
