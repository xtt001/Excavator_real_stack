#!/usr/bin/env bash
# 主端低延迟看图：直接订阅 compressed，丢旧帧，不走 rqt/raw republish。
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

_sanitize_host_gui_env

set +u
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ros2_multihost_env.sh"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

export PYTHONPATH="${ROOT_DIR}/ros2_bridge${PYTHONPATH:+:${PYTHONPATH}}"
TOPIC="${EXCAVATOR_FPV_COMPRESSED_TOPIC:-/camera/color/image_raw/compressed}"
MAX_DISPLAY_HZ="${EXCAVATOR_FPV_VIEWER_MAX_HZ:-60}"
SCALE="${EXCAVATOR_FPV_VIEWER_SCALE:-1.0}"

echo "【主端低延迟看图】topic=${TOPIC}"
echo "  ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW=${RMW_IMPLEMENTATION} peer=${EXCAVATOR_ROS_PEER_IP:-未设置} bind=${EXCAVATOR_ROS_BIND_IP:-auto}"
echo "  QoS: best_effort depth=1；直接显示 compressed 最新帧"

exec /usr/bin/python3 "${ROOT_DIR}/ros2_bridge/excavator_bridge_gateway/host_fpv_low_latency_viewer.py" \
  --topic "${TOPIC}" \
  --max-display-hz "${MAX_DISPLAY_HZ}" \
  --scale "${SCALE}" \
  "$@"
