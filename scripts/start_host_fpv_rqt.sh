#!/usr/bin/env bash
# 主端：订阅从端 /camera/color/image_raw/compressed，本机 republish 后 rqt 显示。
# 规则：从端发布 compressed + 落盘；主端仅可视化，不写 SHM、不录 HDF5。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_fpv_env.sh"
# 组播不通: export EXCAVATOR_ROS_PEER_IP=<从机 IP>

COMPRESSED_TOPIC="${EXCAVATOR_FPV_COMPRESSED_TOPIC:-/camera/color/image_raw/compressed}"
DISPLAY_TOPIC="${EXCAVATOR_FPV_DISPLAY_TOPIC:-/fpv/host_display/image_raw}"
WAIT_SEC="${EXCAVATOR_FPV_TOPIC_WAIT_SEC:-30}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  export PATH="/usr/bin:/bin:${PATH}"
  unset VIRTUAL_ENV
fi

set +u
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_multihost_env.sh"
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
set -u

_wait_topic() {
  local topic="$1"
  local deadline=$((SECONDS + WAIT_SEC))
  echo "Waiting for compressed topic: ${topic} (up to ${WAIT_SEC}s) ..."
  while (( SECONDS < deadline )); do
    if ros2 topic list 2>/dev/null | grep -Fx "${topic}" >/dev/null; then
      echo "Found: ${topic}"
      return 0
    fi
    sleep 1
  done
  return 1
}

if ! _wait_topic "${COMPRESSED_TOPIC}"; then
  echo "Missing ${COMPRESSED_TOPIC}. On slave:" >&2
  echo "  ./scripts/start_orbbec_fpv_camera.sh" >&2
  echo "Ensure both sides use scripts/ros2_fpv_env.sh (DOMAIN_ID=${EXCAVATOR_ROS_DOMAIN_ID})." >&2
  echo "If needed: export EXCAVATOR_ROS_PEER_IP=<slave-ip>" >&2
  exit 1
fi

echo "Host rqt: ${COMPRESSED_TOPIC} -> ${DISPLAY_TOPIC}"
ros2 run image_transport republish compressed raw \
  --ros-args \
  -r "in/compressed:=${COMPRESSED_TOPIC}" \
  -r "out:=${DISPLAY_TOPIC}" &
REPUBLISH_PID=$!

cleanup() {
  kill "${REPUBLISH_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2
exec /usr/bin/python3 /opt/ros/humble/lib/rqt_image_view/rqt_image_view "${DISPLAY_TOPIC}"
