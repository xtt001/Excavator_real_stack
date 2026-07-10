#!/usr/bin/env bash
# SUPERVISED FIELD TEST ONLY: run E52 control with complete trace capture.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "${CONFIRM_HARDWARE_MOTION:-}" != "YES" ]]; then
  echo "Refusing E52 control: set CONFIRM_HARDWARE_MOTION=YES after the on-site operator confirms motion." >&2
  exit 2
fi

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

CONFIG="${CONFIG:-testbed/testbed/configs/policy_real_gmsl_eye2_e52_v1.yaml}"
BUNDLE_DIR="${BUNDLE_DIR:-policy_bundles/real_gmsl_eye2_e52_v1}"
LOG_ROOT="${LOG_ROOT:-/media/mundane/EXTERNAL_USB/policy_control_tests}"
MAX_STEPS="${MAX_STEPS:-4000}"
IMAGE_INTERVAL_STEPS="${IMAGE_INTERVAL_STEPS:-5}"
BRIDGE_HOST="${BRIDGE_HOST:-127.0.0.1}"
BRIDGE_PORT="${BRIDGE_PORT:-8765}"
BRIDGE_TIMEOUT="${BRIDGE_TIMEOUT:-2.0}"
OPERATOR_ID="${OPERATOR_ID:-}"
SESSION_ID="${SESSION_ID:-}"
NOTES="${NOTES:-}"
STAMP="$(date -u +%Y%m%dT%H%M%S.%NZ)"
SESSION_ROOT="${LOG_ROOT}/e52_control_trace_${STAMP}"
PREFLIGHT_REPORT="${SESSION_ROOT}/e52_bundle_preflight.json"

if ! [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_STEPS must be a positive integer, got ${MAX_STEPS}" >&2
  exit 2
fi
if ! [[ "${IMAGE_INTERVAL_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "IMAGE_INTERVAL_STEPS must be a positive integer, got ${IMAGE_INTERVAL_STEPS}" >&2
  exit 2
fi
mkdir -p "${SESSION_ROOT}"

echo "E52 supervised control trace"
echo "  config=${CONFIG}"
echo "  bundle=${BUNDLE_DIR}"
echo "  session=${SESSION_ROOT}"
echo "  max_steps=${MAX_STEPS} image_interval_steps=${IMAGE_INTERVAL_STEPS}"
echo "Use Ctrl+C, bridge stop, or physical emergency stop if the observed motion is wrong."

META_ARGS=()
[[ -n "${OPERATOR_ID}" ]] && META_ARGS+=(--operator-id "${OPERATOR_ID}")
[[ -n "${SESSION_ID}" ]] && META_ARGS+=(--session-id "${SESSION_ID}")
[[ -n "${NOTES}" ]] && META_ARGS+=(--notes "${NOTES}")

"${PYTHON}" scripts/verify_e52_runtime_bundle.py \
  --config "${CONFIG}" \
  --bundle-dir "${BUNDLE_DIR}" | tee "${PREFLIGHT_REPORT}"

set +e
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
  --policy-output-mode control \
  --num-episodes 1 \
  --max-steps "${MAX_STEPS}" \
  --test-log-dir "${SESSION_ROOT}" \
  --test-log-image-interval-steps "${IMAGE_INTERVAL_STEPS}" \
  "${META_ARGS[@]}" \
  --live-action-line
CONTROL_RC=$?
set -e

LATEST_STEPS="$(find "${SESSION_ROOT}" -mindepth 2 -maxdepth 2 -name steps.jsonl -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
if [[ -z "${LATEST_STEPS}" ]]; then
  echo "No steps.jsonl found under ${SESSION_ROOT}; control exit=${CONTROL_RC}" >&2
  exit 1
fi
RUN_DIR="${LATEST_STEPS%/steps.jsonl}"

"${PYTHON}" scripts/summarize_e52_control_trace.py \
  --run-dir "${RUN_DIR}" \
  --config "${CONFIG}" \
  --bundle-dir "${BUNDLE_DIR}" \
  --preflight-report "${PREFLIGHT_REPORT}"

echo "E52 trace run: ${RUN_DIR}"
echo "E52 trace analysis: ${RUN_DIR}/e52_trace_analysis"
exit "${CONTROL_RC}"
