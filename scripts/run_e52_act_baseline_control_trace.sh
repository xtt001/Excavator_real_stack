#!/usr/bin/env bash
# SUPERVISED FIELD TEST ONLY: run the E52 ACT action policy without learned gates.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "${CONFIRM_GO_HOME_DONE:-}" != "YES" ]]; then
  echo "Refusing ACT-only baseline: run go-home first and set CONFIRM_GO_HOME_DONE=YES." >&2
  exit 2
fi
if [[ "${CONFIRM_HARDWARE_MOTION:-}" != "YES" ]]; then
  echo "Refusing ACT-only baseline: set CONFIRM_HARDWARE_MOTION=YES after the operator is ready." >&2
  exit 2
fi
if [[ "${CONFIRM_ACT_ONLY_BASELINE:-}" != "YES" ]]; then
  echo "Refusing ACT-only baseline: set CONFIRM_ACT_ONLY_BASELINE=YES to confirm phase/snap/temporal/gohome gates are bypassed." >&2
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

BASE_CONFIG="${BASE_CONFIG:-testbed/testbed/configs/policy_real_gmsl_eye2_e52_v1.yaml}"
BUNDLE_DIR="${BUNDLE_DIR:-policy_bundles/real_gmsl_eye2_e52_v1}"
LOG_ROOT="${LOG_ROOT:-/media/mundane/EXTERNAL_USB/policy_control_tests}"
MAX_STEPS="${MAX_STEPS:-4000}"
IMAGE_INTERVAL_STEPS="${IMAGE_INTERVAL_STEPS:-5}"
BRIDGE_HOST="${BRIDGE_HOST:-127.0.0.1}"
BRIDGE_PORT="${BRIDGE_PORT:-8765}"
BRIDGE_TIMEOUT="${BRIDGE_TIMEOUT:-2.0}"
RECEIVER_PORT="${RECEIVER_PORT:-8770}"
OPERATOR_ID="${OPERATOR_ID:-}"
SESSION_ID="${SESSION_ID:-}"
NOTES="${NOTES:-E52 ACT-only baseline; runtime gates bypassed}"
STAMP="$(date -u +%Y%m%dT%H%M%S.%NZ)"
SESSION_ROOT="${LOG_ROOT}/e52_act_baseline_control_${STAMP}"
RUNTIME_CONFIG="${SESSION_ROOT}/act_only_runtime_config.yaml"
PREFLIGHT_REPORT="${SESSION_ROOT}/act_only_bundle_preflight.json"

if ! [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_STEPS must be a positive integer, got ${MAX_STEPS}" >&2
  exit 2
fi
if ! [[ "${IMAGE_INTERVAL_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "IMAGE_INTERVAL_STEPS must be a positive integer, got ${IMAGE_INTERVAL_STEPS}" >&2
  exit 2
fi

if "${PYTHON}" - "${RECEIVER_PORT}" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
then
  echo "Refusing ACT-only baseline: receiver port ${RECEIVER_PORT} is already listening." >&2
  echo "Stop policy_remote and run slave_real_stack.sh with --no-receiver first." >&2
  exit 2
fi
mkdir -p "${SESSION_ROOT}"

"${PYTHON}" - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
teleop = config.setdefault("teleop", {})
policy = teleop.setdefault("policy", {})
policy["runtime_gates"] = {"enabled": False}
policy["report_intent"] = True
assist = policy.setdefault("deadzone_assist", {})
assist["enabled"] = True
assist["axis_enabled"] = [True, True, True, True]
assist["trigger_fraction"] = [0.36, 0.50, 0.50, 0.375]
assist["min_consecutive_steps"] = 2
assist["margin"] = [0.02, 0.02, 0.02, 0.02]
policy["source_id"] = "policy:act:real_gmsl_eye2_e52_v1:act_only_baseline"
teleop.setdefault("metadata", {})["notes"] = (
    "E52 ACT-only supervised baseline; learned runtime gates bypassed"
)
config.setdefault("task", {})["task_name"] = (
    "real_gmsl_eye2_e52_act_only_baseline_control"
)
target.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

echo "E52 ACT-only supervised baseline"
echo "  base_config=${BASE_CONFIG}"
echo "  runtime_config=${RUNTIME_CONFIG}"
echo "  bundle=${BUNDLE_DIR}"
echo "  session=${SESSION_ROOT}"
echo "  max_steps=${MAX_STEPS} image_interval_steps=${IMAGE_INTERVAL_STEPS}"
echo "  gates=BYPASSED (phase, snap, temporal direction, automatic gohome)"
echo "ACT output is sent directly after clip/action_scale and existing backend guards."
echo "Use Ctrl+C, bridge stop, or the physical emergency stop if motion is wrong."

"${PYTHON}" scripts/verify_e52_runtime_bundle.py \
  --config "${RUNTIME_CONFIG}" \
  --bundle-dir "${BUNDLE_DIR}" \
  --act-only-baseline | tee "${PREFLIGHT_REPORT}"

META_ARGS=()
[[ -n "${OPERATOR_ID}" ]] && META_ARGS+=(--operator-id "${OPERATOR_ID}")
[[ -n "${SESSION_ID}" ]] && META_ARGS+=(--session-id "${SESSION_ID}")
[[ -n "${NOTES}" ]] && META_ARGS+=(--notes "${NOTES}")

set +e
"${PYTHON}" -m testbed.cli.record_real \
  --config "${RUNTIME_CONFIG}" \
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

set +e
"${PYTHON}" scripts/summarize_policy_test_log.py \
  --run-dir "${RUN_DIR}" \
  --bundle-dir "${BUNDLE_DIR}" \
  --expect-camera-names video4,video5 \
  --expect-output-mode control \
  --allow-stop-reason aborted \
  --min-steps 1 \
  --warmup-steps 0
SUMMARY_RC=$?
set -e

echo "E52 ACT-only baseline run: ${RUN_DIR}"
echo "E52 ACT-only preflight: ${PREFLIGHT_REPORT}"
if [[ "${CONTROL_RC}" -ne 0 ]]; then
  exit "${CONTROL_RC}"
fi
exit "${SUMMARY_RC}"
