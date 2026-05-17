#!/usr/bin/env bash
# 主端：SSHFS 挂载从端数据集目录（手柄在主端、HDF5 写从端时用）
set -euo pipefail
if ! command -v sshfs >/dev/null 2>&1; then
  echo "error: 未安装 sshfs。主端执行: sudo apt install -y sshfs" >&2
  exit 1
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/excavator_deploy_network.sh"

SLAVE_IP="${EXCAVATOR_SLAVE_IP:-192.168.31.170}"
REMOTE_DIR="${EXCAVATOR_SLAVE_DATASET_DIR:-/data/real_teleop_v1}"
LOCAL_MOUNT="${EXCAVATOR_SLAVE_DATASET_MOUNT:-${HOME}/mnt/slave_real_teleop}"
SSH_USER="${EXCAVATOR_SLAVE_SSH_USER:-${USER}}"

mkdir -p "${LOCAL_MOUNT}"
if mountpoint -q "${LOCAL_MOUNT}" 2>/dev/null; then
  echo "已挂载: ${LOCAL_MOUNT}"
  exit 0
fi

echo "挂载 ${SSH_USER}@${SLAVE_IP}:${REMOTE_DIR} -> ${LOCAL_MOUNT}"
exec sshfs "${SSH_USER}@${SLAVE_IP}:${REMOTE_DIR}" "${LOCAL_MOUNT}" \
  -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
