#!/usr/bin/env bash
set -euo pipefail
LOCAL_MOUNT="${EXCAVATOR_SLAVE_DATASET_MOUNT:-${HOME}/mnt/slave_real_teleop}"
if mountpoint -q "${LOCAL_MOUNT}" 2>/dev/null; then
  fusermount -u "${LOCAL_MOUNT}"
  echo "已卸载: ${LOCAL_MOUNT}"
else
  echo "未挂载: ${LOCAL_MOUNT}"
fi
