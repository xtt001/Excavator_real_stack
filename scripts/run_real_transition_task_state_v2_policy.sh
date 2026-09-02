#!/usr/bin/env bash
# Supervised shadow/control runner for the task-state-v2 ACT candidate.
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
export PYTHON PYTHONPATH="${ROOT}/testbed${PYTHONPATH:+:${PYTHONPATH}}"

unset EXCAVATOR_TRANSITION_SESSION_DIR
unset EXCAVATOR_TRANSITION_CONTROL_PORT
unset EXCAVATOR_TRANSITION_CONTROL_BIND_HOST

CU12_LIB="${ROOT}/.venv/lib/python3.10/site-packages/nvidia/cu12/lib"
if [[ -d "${CU12_LIB}" ]]; then
  export LD_LIBRARY_PATH="${CU12_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

BASE_CONFIG="${BASE_CONFIG:-testbed/testbed/configs/policy_real_transition_task_state_v2_allow2.yaml}"
CYCLE_SCRIPT="${CYCLE_SCRIPT:-}"
POLICY_REMOTE_MAX_STEPS="${REAL_TRANSITION_POLICY_MAX_STEPS:-50000}"
export REAL_TRANSITION_RUNTIME_BUNDLE_NAME="real_transition_task_state_v2_allow2"
source "${ROOT}/scripts/real_transition_target_release_paths.sh"
real_transition_resolve_runtime_paths "${ROOT}"
BUNDLE_DIR="${REAL_TRANSITION_BUNDLE_DIR}"
LOG_ROOT="${REAL_TRANSITION_LOG_ROOT}"
export BUNDLE_DIR LOG_ROOT

if [[ "${MODE}" == "control" ]]; then
  for confirmation in \
    CONFIRM_HARDWARE_MOTION \
    CONFIRM_SCRIPT_REVIEWED \
    CONFIRM_TASK_STATE_OPERATOR \
    CONFIRM_SHADOW_LOG_REVIEWED; do
    if [[ "${!confirmation:-}" != "YES" ]]; then
      echo "Refusing control mode: set ${confirmation}=YES after its named review." >&2
      exit 2
    fi
  done
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
SESSION_ROOT="${LOG_ROOT}/real_transition_task_state_v2_${MODE}_${STAMP}"
RUNTIME_CONFIG="${SESSION_ROOT}/runtime_config.yaml"
PREFLIGHT_REPORT="${SESSION_ROOT}/runtime_preflight.json"
mkdir -p "${SESSION_ROOT}"

"${PYTHON}" - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${BUNDLE_DIR}" "${CYCLE_SCRIPT}" "${SESSION_ROOT}" "${OUTPUT_MODE}" <<'PY'
import sys
from pathlib import Path

import yaml
from testbed.config_loader import load_yaml_config

source = Path(sys.argv[1])
target = Path(sys.argv[2])
bundle = Path(sys.argv[3])
script_arg = str(sys.argv[4]).strip()
session = Path(sys.argv[5])
output_mode = sys.argv[6]
config = load_yaml_config(source)
teleop = config.setdefault("teleop", {})
policy = teleop.setdefault("policy", {})
policy.update(
    {
        "bundle_dir": str(bundle.resolve()),
        "ckpt_path": "policy_accepted.ckpt",
        "output_mode": str(output_mode),
        "qvel_mode": "raw",
        "action_scale": [1.0, 1.0, 1.0, 1.0],
        "reset_policy_on_goal": True,
        "reset_policy_on_phase_change": True,
    }
)
policy.setdefault("deadzone_assist", {})["enabled"] = False
planner = policy.setdefault("cycle_planner", {})
planner.update({"enabled": True, "loop": False})
if script_arg:
    planner.pop("script_paths_by_initial_side", None)
    planner["script_path"] = str(Path(script_arg).resolve())
policy_remote = teleop.setdefault("policy_remote", {})
policy_remote["start_in_policy"] = False
scripted = policy_remote.setdefault("scripted_cycle", {})
scripted.update({"enabled": True, "auto_start_after_arm": True})
scripted["task_state_v2"] = {
    "enabled": True,
    "advance_source": "operator_mark",
    "require_excursion_before_work_complete": True,
}
joystick = teleop.setdefault("joystick", {})
joystick.update({"mark_button": 1, "policy_start_button": 6})
teleop.setdefault("test_log", {})["output_dir"] = str(session.resolve())
teleop.setdefault("metadata", {})["notes"] = (
    f"task-state-v2 allow2 ACT; output_mode={output_mode}; "
    f"script={Path(script_arg).resolve() if script_arg else 'auto-by-ready-side'}; "
    "task owner=explicit operator marks"
)
target.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

PREFLIGHT_ARGS=(
  --config "${RUNTIME_CONFIG}"
  --bundle-dir "${BUNDLE_DIR}"
  --expect-output-mode "${OUTPUT_MODE}"
  --load-model
)
if [[ -n "${PREFLIGHT_DEVICE:-}" ]]; then
  PREFLIGHT_ARGS+=(--device "${PREFLIGHT_DEVICE}")
fi
"${PYTHON}" scripts/verify_real_transition_task_state_v2_runtime.py \
  "${PREFLIGHT_ARGS[@]}" | tee "${PREFLIGHT_REPORT}"

echo
echo "Task-state-v2 Real Transition ACT"
echo "  mode=${MODE}"
echo "  runtime_config=${RUNTIME_CONFIG}"
echo "  bundle=${BUNDLE_DIR}/policy_accepted.ckpt"
echo "  bundle_source=${REAL_TRANSITION_BUNDLE_SOURCE}"
echo "  external_drive=${REAL_TRANSITION_DRIVE_ROOT:-none}"
echo "  cycle_script=${CYCLE_SCRIPT:-auto-by-ready-side}"
echo "  session=${SESSION_ROOT}"
echo "  receiver=0.0.0.0:8770, initial mode=manual"
echo
echo "Operator sequence:"
echo "  1. Start the host teleop sender with this runtime config."
echo "  2. Physical button 7 arms the finite script; stable A/B selects its side."
echo "  3. Complete digging/dumping and the positive swing excursion."
echo "  4. Press physical button 2 once: WORK_COMPLETE."
echo "  5. Press physical button 2 again: RETURN_COMMITTED; next target is exposed."
echo "  6. Button 7 returns to manual. Fault/completion latches zero until acknowledged."
echo
echo "Host sender example (replace FIELD_JETSON_IP):"
echo "  python -m testbed.cli.teleop_remote --config ${BASE_CONFIG} --host FIELD_JETSON_IP --input joystick --policy-start-button 7 --mark-button 2 --go-home-button 3 --confirm-remote-control"
if [[ "${MODE}" == "shadow" ]]; then
  echo "shadow_zero logs model output while the returned command remains zero."
fi
if [[ "${DRY_RUN:-}" == "YES" ]]; then
  echo "DRY_RUN=YES: static preflight complete; no field service was started."
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
export EXCAVATOR_SESSION_ID="real_transition_task_state_v2_${MODE}_${STAMP}"

exec ./scripts/slave_real_stack.sh run --force --policy-remote
