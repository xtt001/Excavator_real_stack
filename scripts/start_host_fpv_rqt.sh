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
LAUNCH_PID=""

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
  echo "  Ubuntu 24.04 host: install ROS2 Jazzy packages for optional rqt." >&2
  echo "  Ubuntu 22.04 slave: keep using ROS2 Humble." >&2
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
    if [[ -z "${out}" ]]; then
      out="${part}"
    else
      out="${out}:${part}"
    fi
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
    local clean_ld_library_path
    clean_ld_library_path="$(_path_without_prefixes "${LD_LIBRARY_PATH}" "${conda_prefix}" "${conda_root}")"
    if [[ -n "${clean_ld_library_path}" ]]; then
      export LD_LIBRARY_PATH="${clean_ld_library_path}"
    else
      unset LD_LIBRARY_PATH
    fi
  fi
  unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
  unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
  unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_FONTDIR
}

_cleanup_host_fpv_rqt() {
  local pid
  if [[ -n "${LAUNCH_PID}" ]]; then
    kill -TERM "-${LAUNCH_PID}" >/dev/null 2>&1 || true
    sleep 0.5
    kill -KILL "-${LAUNCH_PID}" >/dev/null 2>&1 || true
  fi
  while read -r pid; do
    [[ -z "${pid}" || "${pid}" == "$$" ]] && continue
    kill -TERM "${pid}" >/dev/null 2>&1 || true
  done < <(
    pgrep -f "host_fpv_republisher_node.py|rqt_image_view .*${RAW_TOPIC}|host_fpv_rqt.launch.py" || true
  )
  sleep 0.2
  while read -r pid; do
    [[ -z "${pid}" || "${pid}" == "$$" ]] && continue
    kill -KILL "${pid}" >/dev/null 2>&1 || true
  done < <(
    pgrep -f "host_fpv_republisher_node.py|rqt_image_view .*${RAW_TOPIC}|host_fpv_rqt.launch.py" || true
  )
}

_sanitize_host_gui_env

set +u
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_multihost_env.sh"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

ros2 daemon stop >/dev/null 2>&1 || true
_cleanup_host_fpv_rqt
trap _cleanup_host_fpv_rqt EXIT INT TERM

LAUNCH_FILE="${ROOT_DIR}/ros2_bridge/excavator_ros2_bridge/launch/host_fpv_rqt.launch.py"
echo "【主端】图源: ${COMPRESSED_TOPIC}（ros2 topic list 里能看到）"
echo "  解码发布: ${RAW_TOPIC}；rqt 下拉里只有 ${RAW_TOPIC}（无 compressed 是类型限制）"
echo "  ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW=${RMW_IMPLEMENTATION} peer=${EXCAVATOR_ROS_PEER_IP:-未设置}"
echo "  QoS: compressed subscription best_effort；raw publish reliable（兼容 rqt 默认订阅）"
if [[ -z "${EXCAVATOR_ROS_PEER_IP:-}" ]]; then
  echo "warn: 未设置 EXCAVATOR_ROS_PEER_IP，主端可能发现不了从端话题；请先:" >&2
  echo "  export EXCAVATOR_SLAVE_IP=192.168.100.1" >&2
  echo "  source scripts/excavator_deploy_network.sh && excavator_apply_host_network_defaults" >&2
fi

setsid ros2 launch "${LAUNCH_FILE}" compressed_topic:="${COMPRESSED_TOPIC}" &
LAUNCH_PID="$!"
wait "${LAUNCH_PID}"
