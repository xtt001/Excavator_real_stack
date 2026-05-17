#!/usr/bin/env bash
# 主端手柄 + 从端落盘：先 ./scripts/mount_slave_dataset.sh，再本脚本
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/excavator_deploy_network.sh"
excavator_apply_host_network_defaults

LOCAL_MOUNT="${EXCAVATOR_SLAVE_DATASET_MOUNT:-${HOME}/mnt/slave_real_teleop}"
BRIDGE_HOST="${EXCAVATOR_BRIDGE_HOST:-${EXCAVATOR_SLAVE_IP:-192.168.31.171}}"
BRIDGE_PORT="${EXCAVATOR_BRIDGE_PORT:-8765}"

if [[ ! -d "${LOCAL_MOUNT}" ]] || ! mountpoint -q "${LOCAL_MOUNT}" 2>/dev/null; then
  echo "error: 请先挂载从端目录: ./scripts/mount_slave_dataset.sh" >&2
  exit 1
fi

if [[ -d "${ROOT_DIR}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

echo "主端手柄 -> gateway ${BRIDGE_HOST}:${BRIDGE_PORT}；HDF5 -> ${LOCAL_MOUNT} (从端磁盘)"
exec tb-record-real \
  --config testbed/configs/teleop_real_v1.yaml \
  --data-side host \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host "${BRIDGE_HOST}" \
  --bridge-port "${BRIDGE_PORT}" \
  --input joystick \
  --output-dir "${LOCAL_MOUNT}" \
  "$@"
