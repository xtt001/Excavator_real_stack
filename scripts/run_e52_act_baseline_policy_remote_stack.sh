#!/usr/bin/env bash
# SUPERVISED FIELD TEST ONLY: E52 ACT-only baseline armed by the host sender.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "${CONFIRM_HARDWARE_MOTION:-}" != "YES" ]]; then
  echo "Refusing ACT-only policy_remote: set CONFIRM_HARDWARE_MOTION=YES after the operator is ready." >&2
  exit 2
fi
if [[ "${CONFIRM_ACT_ONLY_BASELINE:-}" != "YES" ]]; then
  echo "Refusing ACT-only policy_remote: set CONFIRM_ACT_ONLY_BASELINE=YES to confirm learned gates are bypassed." >&2
  exit 2
fi
if [[ "${CONFIRM_GO_HOME_BEFORE_POLICY:-}" != "YES" ]]; then
  echo "Refusing ACT-only policy_remote: acknowledge that go-home must finish before pressing policy button 4." >&2
  echo "Set CONFIRM_GO_HOME_BEFORE_POLICY=YES; this confirms the procedure, not that go-home is already complete." >&2
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
POLICY_REMOTE_MAX_STEPS="${E52_POLICY_REMOTE_MAX_STEPS:-50000}"
STAMP="$(date -u +%Y%m%dT%H%M%S.%NZ)"
SESSION_ROOT="${LOG_ROOT}/e52_act_baseline_policy_remote_${STAMP}"
RUNTIME_CONFIG="${SESSION_ROOT}/act_only_policy_remote_config.yaml"
PREFLIGHT_REPORT="${SESSION_ROOT}/act_only_bundle_preflight.json"

if ! [[ "${POLICY_REMOTE_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "E52_POLICY_REMOTE_MAX_STEPS must be a positive integer, got ${POLICY_REMOTE_MAX_STEPS}" >&2
  exit 2
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  echo "Ignoring generic MAX_STEPS=${MAX_STEPS}; use E52_POLICY_REMOTE_MAX_STEPS for this long-running receiver."
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
teleop.setdefault("policy_remote", {})["start_in_policy"] = False
teleop.setdefault("metadata", {})["notes"] = (
    "E52 ACT-only policy_remote baseline; host button 4 arms model control"
)
config.setdefault("task", {})["task_name"] = (
    "real_gmsl_eye2_e52_act_only_policy_remote"
)
target.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

"${PYTHON}" scripts/verify_e52_runtime_bundle.py \
  --config "${RUNTIME_CONFIG}" \
  --bundle-dir "${BUNDLE_DIR}" \
  --act-only-baseline | tee "${PREFLIGHT_REPORT}"

echo
echo "E52 ACT-only policy_remote baseline"
echo "  runtime_config=${RUNTIME_CONFIG}"
echo "  session=${SESSION_ROOT}"
echo "  receiver_max_steps=${POLICY_REMOTE_MAX_STEPS}"
echo "  receiver=127.0.0.1:8770, initial mode=manual"
echo "  learned gates=BYPASSED; action_scale=1; deadzone assist=all-axis"
echo
echo "MANDATORY BEFORE POLICY BUTTON 4:"
echo "  1. Connect the host teleop_remote sender."
echo "  2. Complete hardware remote/ignition/pilot enable."
echo "  3. Press go-home button 3 and wait for go_home_done."
echo "  4. Verify the arm has not fallen; do not arm ACT from a dropped stick pose."
echo "  5. Press policy button 4 once to enter model control."
echo "Press policy button 4 again to return to manual control."
echo "Ctrl+C here stops the managed stack and sends the terminal zero command."

export EXCAVATOR_TELEOP_CONFIG="${RUNTIME_CONFIG}"
export EXCAVATOR_RECEIVER_INPUT="policy_remote"
export EXCAVATOR_RECEIVER_RECORD_MODE="no-record"
export EXCAVATOR_POLICY_OUTPUT_MODE="control"
export EXCAVATOR_POLICY_ACTION_SCALE="1.0"
export EXCAVATOR_TEST_LOG_DIR="${SESSION_ROOT}"
export EXCAVATOR_NUM_EPISODES="1000000"
export EXCAVATOR_MAX_STEPS="${POLICY_REMOTE_MAX_STEPS}"

exec ./scripts/slave_real_stack.sh run --force --policy-remote
