#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${EXCAVATOR_ENV_FILE:-${ROOT_DIR}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${ENV_FILE}"
  set +a
fi

ENV_NAME="${EXCAVATOR_ENV_NAME:-excavator-real-stack}"
BRIDGE_HOST="${EXCAVATOR_BRIDGE_HOST:-127.0.0.1}"
BRIDGE_PORT="${EXCAVATOR_BRIDGE_PORT:-9876}"
CAN_IF="${EXCAVATOR_CAN_IF:-can0}"
IMU_IF="${EXCAVATOR_IMU_IF:-can1}"
CAN_SIMULATION="${EXCAVATOR_CAN_SIMULATION:-true}"
IMU_SIMULATION="${EXCAVATOR_IMU_SIMULATION:-true}"
CAN_BUS_ENABLED="${EXCAVATOR_CAN_BUS_ENABLED:-false}"
CREATE_MAPPING="${EXCAVATOR_CREATE_MAPPING:-true}"
HEARTBEAT_TIMEOUT_MS="${EXCAVATOR_HEARTBEAT_TIMEOUT_MS:-500}"
READ_TIMEOUT_MS="${EXCAVATOR_READ_TIMEOUT_MS:-100}"
IMAGE_WIDTH="${EXCAVATOR_IMAGE_WIDTH:-160}"
IMAGE_HEIGHT="${EXCAVATOR_IMAGE_HEIGHT:-120}"
SMOKE_NUM_EPISODES="${EXCAVATOR_SMOKE_NUM_EPISODES:-1}"
SMOKE_MAX_STEPS="${EXCAVATOR_SMOKE_MAX_STEPS:-3}"
SMOKE_BUILD="${EXCAVATOR_SMOKE_BUILD:-1}"
ALLOW_REAL_CAN_SMOKE="${EXCAVATOR_ALLOW_REAL_CAN_SMOKE:-0}"
CONFIG_PATH="${EXCAVATOR_TELEOP_CONFIG:-testbed/testbed/configs/teleop_real_v1.yaml}"
BRIDGE_BIN="${EXCAVATOR_BRIDGE_BIN:-${ROOT_DIR}/bridge/build/excavator_real_bridge}"

timestamp="$(date +%Y%m%d-%H%M%S)"
SMOKE_OUTPUT_DIR="${EXCAVATOR_SMOKE_OUTPUT_DIR:-/tmp/excavator-real-stack-smoke-${timestamp}}"
SMOKE_LOG_DIR="${EXCAVATOR_SMOKE_LOG_DIR:-${SMOKE_OUTPUT_DIR}/logs}"
BRIDGE_LOG="${SMOKE_LOG_DIR}/excavator_real_bridge.log"

bool_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

conda_env_exists() {
  command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
}

run_in_env() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${ENV_NAME}" || -n "${VIRTUAL_ENV:-}" ]]; then
    "$@"
  elif conda_env_exists; then
    conda run -n "${ENV_NAME}" "$@"
  elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PATH="${ROOT_DIR}/.venv/bin:${PATH}" "$@"
  else
    "$@"
  fi
}

detect_cmake_prefix() {
  if [[ -n "${EXCAVATOR_CMAKE_PREFIX_PATH:-}" ]]; then
    printf '%s\n' "${EXCAVATOR_CMAKE_PREFIX_PATH}"
  elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    printf '%s\n' "${CONDA_PREFIX}"
  elif conda_env_exists; then
    conda run -n "${ENV_NAME}" python -c 'import os; print(os.environ.get("CONDA_PREFIX", ""))'
  else
    printf '\n'
  fi
}

wait_for_bridge() {
  local deadline=$((SECONDS + 10))
  while (( SECONDS < deadline )); do
    if run_in_env python -c "import socket; s=socket.create_connection(('${BRIDGE_HOST}', ${BRIDGE_PORT}), timeout=0.2); s.close()" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
  done
  echo "bridge did not become ready; log follows:" >&2
  sed -n '1,200p' "${BRIDGE_LOG}" >&2 || true
  return 1
}

BRIDGE_PID=""
cleanup() {
  if [[ -n "${BRIDGE_PID}" ]] && kill -0 "${BRIDGE_PID}" >/dev/null 2>&1; then
    kill "${BRIDGE_PID}" >/dev/null 2>&1 || true
    wait "${BRIDGE_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "${ROOT_DIR}"

if bool_true "${CAN_BUS_ENABLED}" && [[ "${ALLOW_REAL_CAN_SMOKE}" != "1" ]]; then
  cat >&2 <<EOF
Refusing to run smoke test with EXCAVATOR_CAN_BUS_ENABLED=${CAN_BUS_ENABLED}.
Set EXCAVATOR_ALLOW_REAL_CAN_SMOKE=1 only during supervised hardware bring-up.
EOF
  exit 2
fi

if bool_true "${SMOKE_BUILD}" || [[ ! -x "${BRIDGE_BIN}" ]]; then
  cmake_prefix="$(detect_cmake_prefix)"
  run_in_env cmake -S bridge -B bridge/build -DCMAKE_PREFIX_PATH="${cmake_prefix}"
  run_in_env cmake --build bridge/build --target excavator_real_bridge -j2
fi

mkdir -p "${SMOKE_OUTPUT_DIR}" "${SMOKE_LOG_DIR}"

"${BRIDGE_BIN}" \
  --host "${BRIDGE_HOST}" \
  --port "${BRIDGE_PORT}" \
  --can-if "${CAN_IF}" \
  --imu-if "${IMU_IF}" \
  --create-mapping "${CREATE_MAPPING}" \
  --can-bus-enabled "${CAN_BUS_ENABLED}" \
  --can-simulation "${CAN_SIMULATION}" \
  --imu-simulation "${IMU_SIMULATION}" \
  --heartbeat-timeout-ms "${HEARTBEAT_TIMEOUT_MS}" \
  --read-timeout-ms "${READ_TIMEOUT_MS}" \
  --image-width "${IMAGE_WIDTH}" \
  --image-height "${IMAGE_HEIGHT}" \
  >"${BRIDGE_LOG}" 2>&1 &
BRIDGE_PID=$!

wait_for_bridge

run_in_env tb-record-real \
  --config "${CONFIG_PATH}" \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host "${BRIDGE_HOST}" \
  --bridge-port "${BRIDGE_PORT}" \
  --input zero \
  --num-episodes "${SMOKE_NUM_EPISODES}" \
  --max-steps "${SMOKE_MAX_STEPS}" \
  --output-dir "${SMOKE_OUTPUT_DIR}"

run_in_env tb-dataset-qc --dataset-dir "${SMOKE_OUTPUT_DIR}" --profile real

run_in_env python scripts/smoke_bridge_protocol.py \
  --host "${BRIDGE_HOST}" \
  --port "${BRIDGE_PORT}" \
  --heartbeat-timeout-ms "${HEARTBEAT_TIMEOUT_MS}" \
  --shutdown

if ! grep -q "watchdog forced zero command" "${BRIDGE_LOG}"; then
  echo "watchdog did not log a zero-command event; log follows:" >&2
  sed -n '1,240p' "${BRIDGE_LOG}" >&2 || true
  exit 1
fi

wait "${BRIDGE_PID}"
BRIDGE_PID=""

cat <<EOF
Smoke test passed.
Dataset: ${SMOKE_OUTPUT_DIR}
QC:      ${SMOKE_OUTPUT_DIR}/qc
Log:     ${BRIDGE_LOG}
EOF
