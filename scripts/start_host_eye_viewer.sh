#!/usr/bin/env bash
# 主端：显示从端 video4 / eye_left，复用最新帧并行解码 viewer。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export EXCAVATOR_FPV_COMPRESSED_TOPIC="${EXCAVATOR_EYE_COMPRESSED_TOPIC:-/excavator/eye/video4/image_raw/compressed}"
# 略高于 video4 实测约 30.2 Hz，避免显示定时相位周期性跳过有效帧。
export EXCAVATOR_FPV_VIEWER_MAX_HZ="${EXCAVATOR_EYE_VIEWER_MAX_HZ:-35}"
export EXCAVATOR_FPV_VIEWER_SCALE="${EXCAVATOR_EYE_VIEWER_SCALE:-2.0}"

exec "${ROOT_DIR}/scripts/start_host_fpv_viewer.sh" \
  --window-name "Eye Left / video4" \
  "$@"
