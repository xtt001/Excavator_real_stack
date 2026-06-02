#!/usr/bin/env bash
# TEST ONLY: checkpoint/bundle preflight plus shadow_zero policy evaluation.
# This does not send policy actions to the machine and does not save training HDF5.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON="${ROOT}/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi
export PYTHON
export PYTHONPATH="${ROOT}/testbed${PYTHONPATH:+:${PYTHONPATH}}"

CU12_LIB="${ROOT}/.venv/lib/python3.10/site-packages/nvidia/cu12/lib"
if [[ -d "${CU12_LIB}" ]]; then
  export LD_LIBRARY_PATH="${CU12_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

CONFIG="${CONFIG:-testbed/testbed/configs/policy_real_one_dig_v1.yaml}"
BUNDLE_DIR="${BUNDLE_DIR:-policy_bundles/real_one_dig_v1}"
TEST_LOG_DIR="${TEST_LOG_DIR:-/media/mundane/EXTERNAL_USB/policy_control_tests}"
MAX_STEPS="${MAX_STEPS:-500}"
BRIDGE_HOST="${BRIDGE_HOST:-127.0.0.1}"
BRIDGE_PORT="${BRIDGE_PORT:-8765}"
BRIDGE_TIMEOUT="${BRIDGE_TIMEOUT:-2.0}"

echo "TEST ONLY: checking policy bundle and running shadow_zero."
"${PYTHON}" scripts/summarize_policy_test_log.py --bundle-dir "${BUNDLE_DIR}"

"${PYTHON}" -m testbed.cli.record_real \
  --config "${CONFIG}" \
  --data-side slave \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host "${BRIDGE_HOST}" \
  --bridge-port "${BRIDGE_PORT}" \
  --bridge-timeout "${BRIDGE_TIMEOUT}" \
  --input policy \
  --no-record \
  --policy-output-mode shadow_zero \
  --num-episodes 1 \
  --max-steps "${MAX_STEPS}" \
  --test-log-dir "${TEST_LOG_DIR}" \
  --live-action-line

"${PYTHON}" scripts/summarize_policy_test_log.py \
  --latest "${TEST_LOG_DIR}" \
  --expect-output-mode shadow_zero \
  --require-shadow-zero \
  --min-steps "${MAX_STEPS}" \
  --warmup-steps 1
