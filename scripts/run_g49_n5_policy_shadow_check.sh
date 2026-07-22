#!/usr/bin/env bash
# TEST ONLY: hash-pinned G49 N5 four-camera shadow-zero field preflight.
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

CONFIG="${CONFIG:-testbed/testbed/configs/policy_real_gmsl_fourcam_g49_n5_shadow_v1.yaml}"
BUNDLE_DIR="${BUNDLE_DIR:-policy_bundles/real_gmsl_fourcam_g49_n5_v1}"
EXPECT_CAMERA_NAMES="${EXPECT_CAMERA_NAMES:-video4,video5,video6,video7}"
TEST_LOG_DIR="${TEST_LOG_DIR:-/media/mundane/EXTERNAL_USB/policy_control_tests/g49_n5_shadow}"
MAX_STEPS="${MAX_STEPS:-400}"

"${PYTHON}" -m testbed.cli.preflight_act_shadow_deployment \
  --config "${CONFIG}" \
  --bundle-dir "${BUNDLE_DIR}"

export CONFIG BUNDLE_DIR EXPECT_CAMERA_NAMES TEST_LOG_DIR MAX_STEPS
exec "${ROOT}/scripts/run_policy_shadow_check.sh"
