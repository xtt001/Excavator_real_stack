#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_DIR="${EXCAVATOR_TRANSITION_SESSION_DIR:-}"

if [[ -z "${SESSION_DIR}" ]]; then
  printf '%s\n' \
    '[real-transition] EXCAVATOR_TRANSITION_SESSION_DIR is required.' \
    'It must point to a field-prepared session_<id> directory.' >&2
  exit 2
fi
if [[ "${SESSION_DIR}" != /* ]]; then
  printf '[real-transition] session directory must be absolute: %s\n' "${SESSION_DIR}" >&2
  exit 2
fi
for name in sequence_manifest.json split_manifest.json home_side_contract.json; do
  if [[ ! -f "${SESSION_DIR}/${name}" ]]; then
    printf '[real-transition] missing required artifact: %s/%s\n' \
      "${SESSION_DIR}" "${name}" >&2
    exit 2
  fi
done

session_name="$(basename "${SESSION_DIR}")"
if [[ "${session_name}" != session_* || "${session_name}" == session_ ]]; then
  printf '[real-transition] expected session_<id> directory, got: %s\n' \
    "${session_name}" >&2
  exit 2
fi

export EXCAVATOR_TELEOP_CONFIG="${ROOT_DIR}/testbed/testbed/configs/teleop_real_transition_v2_0_1.yaml"
export EXCAVATOR_TRANSITION_SESSION_DIR="${SESSION_DIR}"
export EXCAVATOR_DATASET_DIR="${SESSION_DIR}"
export EXCAVATOR_SESSION_ID="${session_name#session_}"
export EXCAVATOR_RECEIVER_INPUT=remote
export EXCAVATOR_RECEIVER_RECORD_MODE=record
export EXCAVATOR_NUM_EPISODES="${EXCAVATOR_NUM_EPISODES:-24}"
export EXCAVATOR_MAX_STEPS="${EXCAVATOR_MAX_STEPS:-12000}"
export EXCAVATOR_TRANSITION_CONTROL_PORT="${EXCAVATOR_TRANSITION_CONTROL_PORT:-8771}"

exec "${ROOT_DIR}/scripts/slave_real_stack.sh" run --force "$@"
