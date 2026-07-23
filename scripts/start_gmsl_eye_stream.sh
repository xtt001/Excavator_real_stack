#!/usr/bin/env bash
# 从端：从现有 GMSL video4 SHM 旁路发布 eye_left JPEG，不二次打开相机。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_fpv_env.sh"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/excavator_deploy_network.sh"
excavator_apply_slave_network_defaults

set +u
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/source_excavator_ros_ws.sh"
set -u

SHM_NAME="${EXCAVATOR_EYE_SHM_NAME:-excavator_gmsl_video4}"
TOPIC="${EXCAVATOR_EYE_COMPRESSED_TOPIC:-/excavator/eye/video4/image_raw/compressed}"
JPEG_QUALITY="${EXCAVATOR_EYE_JPEG_QUALITY:-80}"
# video4 实测约 30.2 Hz；上限略高以免定时器相位把有效帧降到约 26 Hz。
MAX_HZ="${EXCAVATOR_EYE_STREAM_MAX_HZ:-35}"
SHM_PATH="/dev/shm/${SHM_NAME#/}"

if [[ ! -e "${SHM_PATH}" ]]; then
  echo "error: ${SHM_PATH} 不存在；请先启动 GMSL 四路采集。" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/ros2_bridge${PYTHONPATH:+:${PYTHONPATH}}"
echo "【从端 eye 预览】${SHM_PATH} -> ${TOPIC}"
echo "  独立进程旁路读取；JPEG quality=${JPEG_QUALITY} max_hz=${MAX_HZ}"
echo "  QoS: best_effort depth=1；不二次打开 /dev/video4"

exec /usr/bin/python3 \
  "${ROOT_DIR}/ros2_bridge/excavator_bridge_gateway/gmsl_shm_compressed_publisher_node.py" \
  --shm-name "${SHM_NAME}" \
  --topic "${TOPIC}" \
  --frame-id video4 \
  --jpeg-quality "${JPEG_QUALITY}" \
  --max-publish-hz "${MAX_HZ}" \
  "$@"
