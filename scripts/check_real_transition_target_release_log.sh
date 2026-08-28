#!/usr/bin/env bash
# Review the newest planner-conditioned ACT field log without commanding hardware.
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
export PYTHONPATH="${ROOT}/testbed${PYTHONPATH:+:${PYTHONPATH}}"

BUNDLE_DIR="${BUNDLE_DIR:-policy_bundles/real_transition_target_release_v2}"
LOG_ROOT="${LOG_ROOT:-/media/mundane/EXTERNAL_USB/policy_control_tests}"
MODE="${MODE:-shadow}"

if [[ "${MODE}" == "shadow" ]]; then
  OUTPUT_MODE="shadow_zero"
else
  OUTPUT_MODE="control"
fi

ARGS=(
  --bundle-dir "${BUNDLE_DIR}"
  --checkpoint-name policy_accepted.ckpt
  --expect-camera-names video4,video5,video6,video7
  --latest "${LOG_ROOT}"
  --expect-output-mode "${OUTPUT_MODE}"
  --expect-policy-remote
  --expect-scripted-cycle
  --allow-stop-reason aborted
  --min-steps 1
  --warmup-steps 1
)
if [[ "${MODE}" == "shadow" ]]; then
  ARGS+=(--require-shadow-zero)
fi
exec "${PYTHON}" scripts/summarize_policy_test_log.py "${ARGS[@]}"
