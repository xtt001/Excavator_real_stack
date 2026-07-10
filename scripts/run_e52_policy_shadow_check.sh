#!/usr/bin/env bash
# TEST ONLY: portable E52 bundle preflight plus a shadow_zero field log check.
# Policy and gohome decisions are logged, but no policy command is emitted.
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

CONFIG="${CONFIG:-testbed/testbed/configs/policy_real_gmsl_eye2_e52_v1.yaml}"
BUNDLE_DIR="${BUNDLE_DIR:-policy_bundles/real_gmsl_eye2_e52_v1}"
TEST_LOG_DIR="${TEST_LOG_DIR:-/media/mundane/EXTERNAL_USB/policy_control_tests/e52_runtime_shadow}"
MAX_STEPS="${MAX_STEPS:-500}"
BRIDGE_HOST="${BRIDGE_HOST:-127.0.0.1}"
BRIDGE_PORT="${BRIDGE_PORT:-8765}"
BRIDGE_TIMEOUT="${BRIDGE_TIMEOUT:-2.0}"

echo "TEST ONLY: preflighting E52 bundle and running shadow_zero."
"${PYTHON}" scripts/verify_e52_runtime_bundle.py \
  --config "${CONFIG}" \
  --bundle-dir "${BUNDLE_DIR}"

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

LATEST_STEPS="$("${PYTHON}" - "${TEST_LOG_DIR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = sorted(root.glob("*/steps.jsonl"), key=lambda path: path.stat().st_mtime_ns)
if not candidates:
    raise SystemExit(f"no steps.jsonl found under {root}")
print(candidates[-1])
PY
)"
E53_REPORT="${LATEST_STEPS%/steps.jsonl}/e53_no_motion_report.json"

"${PYTHON}" scripts/e53_verify_no_motion_policy_log.py \
  "${LATEST_STEPS}" \
  --output-json "${E53_REPORT}" \
  --expected-output-mode shadow_zero \
  --min-policy-nonzero-steps 1 \
  --require-runtime-gate-diagnostics

echo "E52 shadow-zero verification report: ${E53_REPORT}"
