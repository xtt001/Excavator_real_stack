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

BASE_CONFIG="${BASE_CONFIG:-testbed/testbed/configs/policy_real_transition_target_release_v2.yaml}"
CYCLE_SCRIPT="${CYCLE_SCRIPT:-testbed/testbed/configs/real_transition_single_cycle_right_to_left_v1.json}"
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
    echo "Refusing control mode: review CYCLE_SCRIPT and set CONFIRM_SCRIPT_REVIEWED=YES." >&2
    exit 2
  fi
fi

if ! [[ "${POLICY_REMOTE_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REAL_TRANSITION_POLICY_MAX_STEPS must be a positive integer." >&2
  exit 2
fi
for required in "${BASE_CONFIG}" "${CYCLE_SCRIPT}" "${BUNDLE_DIR}/policy_accepted.ckpt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file does not exist: ${required}" >&2
    exit 2
  fi
done

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

source, target, bundle, script, session, output_mode = [Path(value) for value in sys.argv[1:6]] + [sys.argv[6]]
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
planner.update({"enabled": True, "script_path": str(script.resolve()), "loop": False})
policy_remote = teleop.setdefault("policy_remote", {})
policy_remote["start_in_policy"] = False
policy_remote.setdefault("scripted_cycle", {})["enabled"] = True
teleop.setdefault("test_log", {})["output_dir"] = str(session.resolve())
teleop.setdefault("metadata", {})["notes"] = (
    f"planner-conditioned target-release ACT; output_mode={output_mode}; "
    f"script={script.resolve()}"
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
echo "  cycle_script=${CYCLE_SCRIPT}"
echo "  session=${SESSION_ROOT}"
echo "  receiver=127.0.0.1:8770, initial mode=manual"
echo
echo "Before pressing policy button 4:"
echo "  1. Connect the host teleop_remote sender and verify deadman/stop controls."
echo "  2. Place the machine at the script initial side and hold it stable for 0.5 s."
echo "  3. Confirm the displayed script and first target."
echo "  4. Press policy button 4 once. A rejected initial-ready leaves manual control active."
echo "During a run, button 4 returns to manual. Script completion/fault latches zero;"
echo "press button 4 once to acknowledge the latch and remain in manual mode."
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
