#!/usr/bin/env bash
# Pro real teleop recording: keep the canonical action contract and write to a new USB dataset.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export EXCAVATOR_DATASET_DIR="${EXCAVATOR_DATASET_DIR:-/media/mundane/EXTERNAL_USB/pro_real_teleop}"
export EXCAVATOR_TELEOP_CONFIG="${EXCAVATOR_TELEOP_CONFIG:-${ROOT_DIR}/testbed/testbed/configs/teleop_real_v1.yaml}"
export EXCAVATOR_RECEIVER_INPUT="${EXCAVATOR_RECEIVER_INPUT:-policy_remote}"
export EXCAVATOR_RECEIVER_RECORD_MODE="${EXCAVATOR_RECEIVER_RECORD_MODE:-record}"
export EXCAVATOR_SESSION_ID="${EXCAVATOR_SESSION_ID:-pro_real_teleop}"
export EXCAVATOR_NUM_EPISODES="${EXCAVATOR_NUM_EPISODES:-1000000}"
export EXCAVATOR_MAX_STEPS="${EXCAVATOR_MAX_STEPS:-50000}"

echo "pro real teleop dataset=${EXCAVATOR_DATASET_DIR}"
echo "swing joystick mapping: forward -> action[0] negative; backward -> action[0] positive"
echo "HDF5 action contract: [swing, boom, stick, bucket] unchanged"

exec ./scripts/slave_real_stack.sh run --force --policy-remote "$@"
