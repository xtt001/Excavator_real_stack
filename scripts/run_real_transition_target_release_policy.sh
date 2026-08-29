#!/usr/bin/env bash
# Supervised field runner for planner-conditioned left/right ACT cycles.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODE="${MODE:-shadow}"
if [[ "${MODE}" != "shadow" && "${MODE}" != "control" ]]; then
  echo "MODE must be shadow or control, got ${MODE}" >&2
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

# This runner is independent of expert-recording sessions.  Operators often
# launch it from a shell that still exports ctxXX variables; never let those
# variables turn policy_remote back into a transition recorder.
unset EXCAVATOR_TRANSITION_SESSION_DIR
unset EXCAVATOR_TRANSITION_CONTROL_PORT
unset EXCAVATOR_TRANSITION_CONTROL_BIND_HOST

CU12_LIB="${ROOT}/.venv/lib/python3.10/site-packages/nvidia/cu12/lib"
if [[ -d "${CU12_LIB}" ]]; then
  export LD_LIBRARY_PATH="${CU12_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

BASE_CONFIG="${BASE_CONFIG:-testbed/testbed/configs/policy_real_transition_target_release_v2.yaml}"
CYCLE_SCRIPT="${CYCLE_SCRIPT:-}"
POLICY_REMOTE_MAX_STEPS="${REAL_TRANSITION_POLICY_MAX_STEPS:-50000}"

# A fresh git checkout does not contain the ignored model bundle. Prefer the
# unique accepted bundle on an inserted field drive, then fall back to a local
# bundle for development. Explicit BUNDLE_DIR/LOG_ROOT values still win.
source "${ROOT}/scripts/real_transition_target_release_paths.sh"
real_transition_resolve_runtime_paths "${ROOT}"
BUNDLE_DIR="${REAL_TRANSITION_BUNDLE_DIR}"
LOG_ROOT="${REAL_TRANSITION_LOG_ROOT}"
export BUNDLE_DIR LOG_ROOT

if [[ "${MODE}" == "control" ]]; then
  if [[ "${CONFIRM_HARDWARE_MOTION:-}" != "YES" ]]; then
    echo "Refusing control mode: set CONFIRM_HARDWARE_MOTION=YES." >&2
    exit 2
  fi
  if [[ "${CONFIRM_SCRIPT_REVIEWED:-}" != "YES" ]]; then
    echo "Refusing control mode: review the configured cycle script(s) and set CONFIRM_SCRIPT_REVIEWED=YES." >&2
    exit 2
  fi
fi

if ! [[ "${POLICY_REMOTE_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REAL_TRANSITION_POLICY_MAX_STEPS must be a positive integer." >&2
  exit 2
fi
for required in "${BASE_CONFIG}" "${BUNDLE_DIR}/policy_accepted.ckpt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file does not exist: ${required}" >&2
    exit 2
  fi
done
if [[ -n "${CYCLE_SCRIPT}" && ! -f "${CYCLE_SCRIPT}" ]]; then
  echo "Required cycle script does not exist: ${CYCLE_SCRIPT}" >&2
  exit 2
fi

OUTPUT_MODE="shadow_zero"
if [[ "${MODE}" == "control" ]]; then
  OUTPUT_MODE="control"
fi
STAMP="$(date -u +%Y%m%dT%H%M%S.%NZ)"
SESSION_ROOT="${LOG_ROOT}/real_transition_target_release_${MODE}_${STAMP}"
RUNTIME_CONFIG="${SESSION_ROOT}/runtime_config.yaml"
PREFLIGHT_REPORT="${SESSION_ROOT}/runtime_preflight.json"
mkdir -p "${SESSION_ROOT}"

"${PYTHON}" - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${BUNDLE_DIR}" "${CYCLE_SCRIPT}" "${SESSION_ROOT}" "${OUTPUT_MODE}" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
bundle = Path(sys.argv[3])
script_arg = str(sys.argv[4]).strip()
session = Path(sys.argv[5])
output_mode = sys.argv[6]
config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
teleop = config.setdefault("teleop", {})
policy = teleop.setdefault("policy", {})
policy["bundle_dir"] = str(bundle.resolve())
policy["ckpt_path"] = "policy_accepted.ckpt"
policy["output_mode"] = str(output_mode)
policy["action_scale"] = [1.0, 1.0, 1.0, 1.0]
policy.setdefault("deadzone_assist", {})["enabled"] = False
policy["reset_policy_on_goal"] = True
planner = policy.setdefault("cycle_planner", {})
planner.update({"enabled": True, "loop": False})
if script_arg:
    script = Path(script_arg)
    planner.pop("script_paths_by_initial_side", None)
    planner["script_path"] = str(script.resolve())
policy_remote = teleop.setdefault("policy_remote", {})
policy_remote["start_in_policy"] = False
policy_remote.setdefault("scripted_cycle", {})["enabled"] = True
teleop.setdefault("test_log", {})["output_dir"] = str(session.resolve())
teleop.setdefault("metadata", {})["notes"] = (
    f"planner-conditioned target-release ACT; output_mode={output_mode}; "
    f"script={Path(script_arg).resolve() if script_arg else 'auto-by-ready-side'}"
)
target.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

"${PYTHON}" scripts/verify_real_transition_target_release_runtime.py \
  --config "${RUNTIME_CONFIG}" \
  --bundle-dir "${BUNDLE_DIR}" \
  --expect-output-mode "${OUTPUT_MODE}" | tee "${PREFLIGHT_REPORT}"

echo
echo "Planner-conditioned Real Transition ACT"
echo "  mode=${MODE}"
echo "  runtime_config=${RUNTIME_CONFIG}"
echo "  bundle=${BUNDLE_DIR}/policy_accepted.ckpt"
echo "  bundle_source=${REAL_TRANSITION_BUNDLE_SOURCE}"
echo "  external_drive=${REAL_TRANSITION_DRIVE_ROOT:-none}"
echo "  cycle_script=${CYCLE_SCRIPT:-auto-by-ready-side}"
echo "  session=${SESSION_ROOT}"
echo "  receiver=127.0.0.1:8770, initial mode=manual"
echo
echo "Before pressing left-hand policy button 7:"
echo "  1. Connect the host teleop_remote sender and verify deadman/stop controls."
echo "  2. Press left-hand button 7 once to ARM automatic script selection."
echo "  3. Hold the machine stable in A or B for 0.5 s."
echo "  4. The matching finite script is selected and ACT starts automatically."
echo "During a run, left-hand button 7 returns to manual. Script completion/fault latches zero;"
echo "press left-hand button 7 once to acknowledge the latch and remain in manual mode."
if [[ "${MODE}" == "shadow" ]]; then
  echo "shadow_zero does not move the machine; it validates loading, first-goal commit and logs."
fi
if [[ "${DRY_RUN:-}" == "YES" ]]; then
  echo "DRY_RUN=YES: preflight complete; managed field stack was not started."
  exit 0
fi

export EXCAVATOR_TELEOP_CONFIG="${RUNTIME_CONFIG}"
export EXCAVATOR_RECEIVER_INPUT="policy_remote"
export EXCAVATOR_RECEIVER_RECORD_MODE="no-record"
export EXCAVATOR_POLICY_OUTPUT_MODE="${OUTPUT_MODE}"
export EXCAVATOR_POLICY_ACTION_SCALE="1.0"
export EXCAVATOR_TEST_LOG_DIR="${SESSION_ROOT}"
export EXCAVATOR_NUM_EPISODES="1000000"
export EXCAVATOR_MAX_STEPS="${POLICY_REMOTE_MAX_STEPS}"

exec ./scripts/slave_real_stack.sh run --force --policy-remote
