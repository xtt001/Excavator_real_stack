#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/excavator_deploy_network.sh"
excavator_apply_slave_network_defaults

if [[ -d "${ROOT_DIR}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

export PYTHONPATH="${ROOT_DIR}/ros2_bridge${PYTHONPATH:+:}${PYTHONPATH:-}"

camera_args=(--camera-source "${EXCAVATOR_CAMERA_SOURCE:-fpv}")
if [[ "${EXCAVATOR_CAMERA_SOURCE:-fpv}" == "gmsl" ]]; then
  IFS=',' read -r -a gmsl_cameras <<<"${EXCAVATOR_GMSL_GATEWAY_CAMERAS:-video4=excavator_gmsl_video4,video5=excavator_gmsl_video5,video6=excavator_gmsl_video6,video7=excavator_gmsl_video7}"
  for camera in "${gmsl_cameras[@]}"; do
    [[ -n "${camera}" ]] || continue
    camera_args+=(--gmsl-camera "${camera}")
  done
  camera_args+=(--gmsl-max-group-skew-ms "${EXCAVATOR_GMSL_MAX_GROUP_SKEW_MS:-5.0}")
  camera_args+=(--gmsl-group-timeout-ms "${EXCAVATOR_GMSL_GROUP_TIMEOUT_MS:-50.0}")
fi

exec python3 -m excavator_bridge_gateway.gateway_server \
  --host "${EXCAVATOR_GATEWAY_HOST:-127.0.0.1}" \
  --port "${EXCAVATOR_GATEWAY_PORT:-8765}" \
  --control-host "${EXCAVATOR_CONTROL_HOST:-127.0.0.1}" \
  --control-port "${EXCAVATOR_CONTROL_PORT:-8766}" \
  "${camera_args[@]}" \
  --fpv-encoding "${EXCAVATOR_FPV_ENCODING:-jpeg}" \
  --fpv-jpeg-quality "${EXCAVATOR_FPV_JPEG_QUALITY:-95}" \
  --fpv-jpeg-cache-hz "${EXCAVATOR_FPV_JPEG_CACHE_HZ:-30}" \
  "$@"
