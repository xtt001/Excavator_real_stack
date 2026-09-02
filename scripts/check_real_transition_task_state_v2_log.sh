#!/usr/bin/env bash
# Review the newest task-state-v2 field log without commanding hardware.
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
export REAL_TRANSITION_RUNTIME_BUNDLE_NAME="real_transition_task_state_v2_allow2"

source "${ROOT}/scripts/real_transition_target_release_paths.sh"
real_transition_resolve_runtime_paths "${ROOT}"
BUNDLE_DIR="${REAL_TRANSITION_BUNDLE_DIR}"
LOG_ROOT="${REAL_TRANSITION_LOG_ROOT}"
MODE="${MODE:-shadow}"
case "${MODE}" in
  shadow) OUTPUT_MODE="shadow_zero" ;;
  control) OUTPUT_MODE="control" ;;
  *) echo "MODE must be shadow or control, got ${MODE}" >&2; exit 2 ;;
esac

ARGS=(
  --bundle-dir "${BUNDLE_DIR}"
  --checkpoint-name policy_accepted.ckpt
  --expect-camera-names video4,video5,video6,video7
  --latest "${LOG_ROOT}"
  --expect-output-mode "${OUTPUT_MODE}"
  --expect-policy-remote
  --expect-scripted-cycle
  --expect-task-state-v2
  --allow-stop-reason aborted
  --min-steps 1
  --warmup-steps 1
)
if [[ "${MODE}" == "shadow" ]]; then
  ARGS+=(--require-shadow-zero)
fi
exec "${PYTHON}" scripts/summarize_policy_test_log.py "${ARGS[@]}"
