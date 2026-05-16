#!/usr/bin/env bash
# 与 start_host_fpv_rqt.sh 相同 ROS 域，用于调试（避免 DOMAIN_ID 不一致）。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOPIC="${1:-/camera/color/image_raw/compressed}"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_fpv_env.sh"
set +u
source "${ROOT_DIR}/scripts/ros2_multihost_env.sh"
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
set -u
ros2 daemon stop >/dev/null 2>&1 || true

echo "echo ${TOPIC} @ ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
exec ros2 topic echo "${TOPIC}"
