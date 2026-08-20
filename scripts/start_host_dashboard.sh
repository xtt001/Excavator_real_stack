#!/usr/bin/env bash
# 主端现场 GUI：video4 + receiver/IMU/录制/QC + v2 数据事件控制。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/excavator_deploy_network.sh"
excavator_apply_host_network_defaults
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_fpv_env.sh"

if [[ -z "${ROS_DISTRO:-}" ]]; then
  for distro in jazzy humble iron rolling; do
    if [[ -f "/opt/ros/${distro}/setup.bash" ]]; then
      export ROS_DISTRO="${distro}"
      break
    fi
  done
fi
if [[ -z "${ROS_DISTRO:-}" || ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  echo "error: no ROS2 setup.bash found under /opt/ros." >&2
  exit 1
fi

_path_without_prefixes() {
  local value="$1"
  shift
  local out="" part prefix skip
  local -a parts
  IFS=: read -r -a parts <<< "${value}"
  for part in "${parts[@]}"; do
    [[ -z "${part}" ]] && continue
    skip=0
    for prefix in "$@"; do
      if [[ -n "${prefix}" && ( "${part}" == "${prefix}" || "${part}" == "${prefix}/"* ) ]]; then
        skip=1
        break
      fi
    done
    [[ "${skip}" -eq 1 ]] && continue
    out="${out:+${out}:}${part}"
  done
  echo "${out}"
}

_sanitize_host_gui_env() {
  local conda_root=""
  if [[ -n "${CONDA_EXE:-}" ]]; then
    conda_root="$(cd "$(dirname "${CONDA_EXE}")/.." 2>/dev/null && pwd -P || true)"
  fi
  local conda_prefix="${CONDA_PREFIX:-}"
  local clean_path
  clean_path="$(_path_without_prefixes "${PATH:-}" "${conda_prefix}" "${conda_root}")"
  export PATH="/usr/bin:/bin${clean_path:+:${clean_path}}"
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    local clean_ld
    clean_ld="$(_path_without_prefixes "${LD_LIBRARY_PATH}" "${conda_prefix}" "${conda_root}")"
    if [[ -n "${clean_ld}" ]]; then
      export LD_LIBRARY_PATH="${clean_ld}"
    else
      unset LD_LIBRARY_PATH
    fi
  fi
  unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
  unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
  unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_FONTDIR
}

_sanitize_host_gui_env
set +u
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_multihost_env.sh"
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

export PYTHONPATH="${ROOT_DIR}/testbed:${ROOT_DIR}/ros2_bridge${PYTHONPATH:+:${PYTHONPATH}}"
STATUS_PORT="${EXCAVATOR_HOST_STATUS_PORT:-8781}"
VIDEO_TOPIC="${EXCAVATOR_EYE_COMPRESSED_TOPIC:-/excavator/eye/video4/image_raw/compressed}"
CONFIG="${EXCAVATOR_TELEOP_CONFIG:-${ROOT_DIR}/testbed/testbed/configs/teleop_real_v1.yaml}"

echo "【主端现场 GUI】default_status=127.0.0.1:${STATUS_PORT} default_video=${VIDEO_TOPIC}"
echo "  GUI 不连接 8770、不发送摇杆动作；v2 模式下只向 8771 提交数据事件。"
echo "  Wayland 会自动通过 XWayland/xcb 启动，以保证‘保持窗口最前’由 Mutter 执行。"

cd "${ROOT_DIR}"
exec /usr/bin/python3 -m testbed.cli.host_dashboard \
  --config "${CONFIG}" \
  --status-port "${STATUS_PORT}" \
  --video-topic "${VIDEO_TOPIC}" \
  "$@"
