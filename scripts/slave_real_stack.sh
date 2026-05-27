#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${EXCAVATOR_SLAVE_STACK_STATE_DIR:-/tmp/excavator_slave_stack}"
PID_DIR="${STATE_DIR}/pids"
LOG_ROOT="${EXCAVATOR_SLAVE_STACK_LOG_ROOT:-${ROOT_DIR}/artifacts/slave_stack}"
LOG_DIR_FILE="${STATE_DIR}/log_dir"

CONTROL_HOST="${EXCAVATOR_CONTROL_HOST:-127.0.0.1}"
CONTROL_PORT="${EXCAVATOR_CONTROL_PORT:-8766}"
GATEWAY_HOST="${EXCAVATOR_GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${EXCAVATOR_GATEWAY_PORT:-8765}"
RECEIVER_PORT="${EXCAVATOR_REMOTE_ACTION_PORT:-8770}"
CAN_IF="${EXCAVATOR_CAN_IF:-can2}"
IMU_IF="${EXCAVATOR_IMU_IF:-can3}"
CAN_BITRATE="${EXCAVATOR_CAN_BITRATE:-250000}"
USB_LABEL="${EXCAVATOR_USB_LABEL:-EXTERNAL_USB}"
USB_MOUNT="${EXCAVATOR_USB_MOUNT:-/media/${USER}/EXTERNAL_USB}"
DATASET_DIR="${EXCAVATOR_DATASET_DIR:-${USB_MOUNT}/real_teleop_v1}"
CONFIG_PATH="${EXCAVATOR_TELEOP_CONFIG:-${ROOT_DIR}/testbed/testbed/configs/teleop_real_v1.yaml}"
PID_YAML_PATH="${EXCAVATOR_PID_YAML:-${ROOT_DIR}/control/config/joint_pid.yaml}"
SESSION_ID="${EXCAVATOR_SESSION_ID:-remote_teleop_slave_record}"
MAX_STEPS="${EXCAVATOR_MAX_STEPS:-50000}"
BRIDGE_TIMEOUT="${EXCAVATOR_BRIDGE_TIMEOUT:-2.0}"
FPV_MAX_STALE_MS="${EXCAVATOR_FPV_MAX_STALE_MS:-1000}"
FPV_SHM_NAME="${EXCAVATOR_FPV_SHM_NAME:-excavator_fpv_v1}"
FPV_SHM_TIMEOUT_S="${EXCAVATOR_FPV_SHM_TIMEOUT_S:-45}"
STARTUP_TIMEOUT_S="${EXCAVATOR_STARTUP_TIMEOUT_S:-12}"
RECEIVER_STOP_TIMEOUT_S="${EXCAVATOR_RECEIVER_STOP_TIMEOUT_S:-${EXCAVATOR_RECORDER_STOP_TIMEOUT_S:-180}}"

SERVICES=(bridge orbbec fpv gateway receiver)
STOP_ORDER=(receiver gateway fpv orbbec bridge)
STARTED_SERVICES=()

usage() {
  cat <<'EOF'
Usage:
  scripts/slave_real_stack.sh start [--force] [--no-camera] [--no-receiver] [--skip-usb] [--skip-can] [--skip-pip-install]
  scripts/slave_real_stack.sh run [--force] [--no-camera] [--no-receiver] [--skip-usb] [--skip-can] [--skip-pip-install]
  scripts/slave_real_stack.sh stop [--force]
  scripts/slave_real_stack.sh restart [--force] [start options]
  scripts/slave_real_stack.sh status
  scripts/slave_real_stack.sh logs
  scripts/slave_real_stack.sh tail [service]

Default start services:
  bridge, Orbbec camera, FPV SHM subscriber, gateway, receiver.

Use "run" when you want one foreground terminal that shows logs and stops the
managed services on Ctrl+C.

Common environment overrides:
  EXCAVATOR_USB_MOUNT=/media/mundane/EXTERNAL_USB
  EXCAVATOR_DATASET_DIR=/media/mundane/EXTERNAL_USB/real_teleop_v1
  EXCAVATOR_ORBBEC_WS=/home/mundane/orbbec_ws
  EXCAVATOR_ROS_WS=/home/mundane/orbbec_ws
  EXCAVATOR_CAN_IF=can2 EXCAVATOR_IMU_IF=can3
  EXCAVATOR_PID_YAML=/media/mundane/D/Excavator_real_stack/control/config/joint_pid.yaml
  EXCAVATOR_RECEIVER_STOP_TIMEOUT_S=180

Compatibility:
  --no-recorder and "tail recorder" remain aliases for receiver.
EOF
}

log() {
  printf '[slave-stack] %s\n' "$*"
}

die() {
  printf '[slave-stack] error: %s\n' "$*" >&2
  exit 1
}

canonical_service() {
  case "$1" in
    recorder)
      printf 'receiver'
      ;;
    *)
      printf '%s' "$1"
      ;;
  esac
}

pid_file() {
  printf '%s/%s.pid' "${PID_DIR}" "$(canonical_service "$1")"
}

log_dir() {
  if [[ -s "${LOG_DIR_FILE}" ]]; then
    cat "${LOG_DIR_FILE}"
  else
    printf '%s/latest' "${LOG_ROOT}"
  fi
}

service_alive() {
  local name="$1"
  local file pid
  file="$(pid_file "${name}")"
  [[ -s "${file}" ]] || return 1
  pid="$(cat "${file}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

service_pid() {
  local file
  file="$(pid_file "$1")"
  [[ -s "${file}" ]] && cat "${file}" || true
}

port_pids() {
  local port="$1"
  ss -H -ltnp "sport = :${port}" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
    | sort -u
}

require_port_free() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(port_pids "${port}" | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  if [[ -n "${pids}" ]]; then
    die "${label} port ${port} is already in use by pid(s): ${pids}. Run: scripts/slave_real_stack.sh stop --force"
  fi
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local label="$3"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
      log "${label} is listening on ${host}:${port}"
      return 0
    fi
    sleep 0.3
  done
  die "${label} did not open ${host}:${port}. Check $(log_dir)/${label}.log"
}

wait_for_shm() {
  local shm_path="/dev/shm/${FPV_SHM_NAME#/}"
  local deadline=$((SECONDS + FPV_SHM_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if [[ -e "${shm_path}" ]]; then
      log "FPV SHM is present: ${shm_path}"
      return 0
    fi
    sleep 0.5
  done
  die "FPV SHM was not created: ${shm_path}. Check $(log_dir)/orbbec.log and $(log_dir)/fpv.log"
}

check_started() {
  local name="$1"
  sleep 0.7
  if ! service_alive "${name}"; then
    log "${name} exited during startup. Last log lines:"
    tail -40 "$(log_dir)/${name}.log" 2>/dev/null || true
    exit 1
  fi
}

start_service() {
  local name="$1"
  shift
  local file log_file pid
  file="$(pid_file "${name}")"
  log_file="$(log_dir)/${name}.log"
  if service_alive "${name}"; then
    log "${name} already running pid=$(service_pid "${name}")"
    return 0
  fi
  mkdir -p "${PID_DIR}" "$(log_dir)"
  {
    printf '\n=== %s start %s ===\n' "${name}" "$(date -Is)"
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
  } >>"${log_file}"
  setsid "$@" >>"${log_file}" 2>&1 &
  pid="$!"
  printf '%s\n' "${pid}" >"${file}"
  STARTED_SERVICES+=("${name}")
  log "started ${name} pid=${pid} log=${log_file}"
  check_started "${name}"
}

stop_pid_group() {
  local name="$1"
  local signal="$2"
  local timeout_s="$3"
  local pid="$4"
  local deadline
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  log "stopping ${name} pid=${pid} signal=${signal}"
  kill "-${signal}" "-${pid}" 2>/dev/null || kill "-${signal}" "${pid}" 2>/dev/null || true
  deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep 0.4
  done
  log "${name} did not exit after ${timeout_s}s; sending KILL"
  kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
}

stop_service() {
  local name="$1"
  local signal="${2:-TERM}"
  local timeout_s="${3:-8}"
  local file pid
  file="$(pid_file "${name}")"
  [[ -s "${file}" ]] || return 0
  pid="$(cat "${file}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]]; then
    stop_pid_group "${name}" "${signal}" "${timeout_s}" "${pid}"
  fi
  rm -f "${file}"
}

force_stop_stale() {
  local pids pid args
  log "force stop stale listeners on ${CONTROL_PORT}/${GATEWAY_PORT}/${RECEIVER_PORT}"
  pids="$(
    {
      port_pids "${CONTROL_PORT}"
      port_pids "${GATEWAY_PORT}"
      port_pids "${RECEIVER_PORT}"
    } | sort -u
  )"
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    [[ -n "${args}" ]] || continue
    log "force stopping pid=${pid}: ${args}"
    kill -TERM "${pid}" 2>/dev/null || true
  done <<<"${pids}"
  sleep 1
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    kill -0 "${pid}" 2>/dev/null || continue
    log "force killing pid=${pid}"
    kill -KILL "${pid}" 2>/dev/null || true
  done <<<"${pids}"
}

stop_stack() {
  local force="${1:-0}"
  stop_service receiver INT "${RECEIVER_STOP_TIMEOUT_S}"
  stop_service recorder INT "${RECEIVER_STOP_TIMEOUT_S}"
  if [[ -d "${DATASET_DIR}" ]]; then
    log "syncing dataset directory: ${DATASET_DIR}"
    sync "${DATASET_DIR}" 2>/dev/null || sync
  fi
  stop_service gateway TERM 8
  stop_service fpv INT 10
  stop_service orbbec INT 10
  stop_service bridge TERM 8
  if [[ "${force}" == "1" ]]; then
    force_stop_stale
  fi
}

mount_usb() {
  sudo mkdir -p "${USB_MOUNT}"
  if ! findmnt "${USB_MOUNT}" >/dev/null 2>&1; then
    log "mounting USB label ${USB_LABEL} at ${USB_MOUNT}"
    sudo mount -t ntfs3 -o "uid=$(id -u),gid=$(id -g),umask=022" \
      "/dev/disk/by-label/${USB_LABEL}" "${USB_MOUNT}" \
      || sudo mount -t ntfs-3g -o "uid=$(id -u),gid=$(id -g),umask=022" \
        "/dev/disk/by-label/${USB_LABEL}" "${USB_MOUNT}"
  fi
  mkdir -p "${DATASET_DIR}"
  [[ -w "${DATASET_DIR}" ]] || die "dataset directory is not writable: ${DATASET_DIR}"
  log "dataset directory writable: ${DATASET_DIR}"
}

setup_can() {
  log "setting up CAN ${CAN_IF}/${IMU_IF} bitrate=${CAN_BITRATE}"
  "${ROOT_DIR}/control/setup/setup_can.sh" "${CAN_IF}" "${CAN_BITRATE}"
  "${ROOT_DIR}/control/setup/setup_can.sh" "${IMU_IF}" "${CAN_BITRATE}"
}

prepare_start() {
  local run_id log_dir_path
  [[ -x "${ROOT_DIR}/bridge/build/excavator_real_bridge" ]] \
    || die "missing bridge binary: ${ROOT_DIR}/bridge/build/excavator_real_bridge"
  run_id="$(date +%Y%m%d_%H%M%S)"
  log_dir_path="${LOG_ROOT}/${run_id}"
  mkdir -p "${PID_DIR}" "${log_dir_path}"
  printf '%s\n' "${log_dir_path}" >"${LOG_DIR_FILE}"
  ln -sfn "${log_dir_path}" "${LOG_ROOT}/latest"
}

start_stack() {
  local force="$1"
  local no_camera="$2"
  local no_receiver="$3"
  local skip_usb="$4"
  local skip_can="$5"
  local skip_pip="$6"

  if [[ "${force}" == "1" ]]; then
    stop_stack 1
  fi

  require_port_free "${CONTROL_PORT}" bridge
  require_port_free "${GATEWAY_PORT}" gateway
  if [[ "${no_receiver}" != "1" ]]; then
    require_port_free "${RECEIVER_PORT}" receiver
  fi

  prepare_start
  export ROOT_DIR CONTROL_HOST CONTROL_PORT GATEWAY_HOST GATEWAY_PORT RECEIVER_PORT
  export CAN_IF IMU_IF DATASET_DIR CONFIG_PATH PID_YAML_PATH SESSION_ID MAX_STEPS BRIDGE_TIMEOUT
  export FPV_MAX_STALE_MS FPV_SHM_NAME
  export EXCAVATOR_ORBBEC_WS="${EXCAVATOR_ORBBEC_WS:-${HOME}/orbbec_ws}"
  export EXCAVATOR_ROS_WS="${EXCAVATOR_ROS_WS:-${EXCAVATOR_ORBBEC_WS}}"
  export EXCAVATOR_SKIP_PIP_INSTALL="${skip_pip}"

  if [[ "${skip_usb}" != "1" && "${no_receiver}" != "1" ]]; then
    mount_usb
  fi
  if [[ "${skip_can}" != "1" ]]; then
    setup_can
  fi

  start_service bridge bash -lc '
    cd "${ROOT_DIR}"
    exec ./bridge/build/excavator_real_bridge \
      --host "${CONTROL_HOST}" \
      --port "${CONTROL_PORT}" \
      --can-if "${CAN_IF}" \
      --imu-if "${IMU_IF}" \
      --can-bus-enabled true \
      --can-simulation false \
      --imu-simulation false \
      --create-mapping true \
      --pid-yaml "${PID_YAML_PATH}" \
      --heartbeat-timeout-ms 800
  '
  wait_for_port "${CONTROL_HOST}" "${CONTROL_PORT}" bridge

  if [[ "${no_camera}" != "1" ]]; then
    start_service orbbec bash -lc '
      cd "${ROOT_DIR}"
      exec ./scripts/start_orbbec_fpv_camera.sh
    '
    start_service fpv bash -lc '
      cd "${ROOT_DIR}"
      exec ./scripts/start_fpv_subscriber_py.sh
    '
    wait_for_shm
  fi

  start_service gateway bash -lc '
    cd "${ROOT_DIR}"
    exec ./scripts/start_bridge_gateway.sh \
      --fpv-source auto \
      --fpv-max-stale-ms "${FPV_MAX_STALE_MS}"
  '
  wait_for_port "${GATEWAY_HOST}" "${GATEWAY_PORT}" gateway

  if [[ "${no_receiver}" != "1" ]]; then
    start_service receiver bash -lc '
      cd "${ROOT_DIR}"
      if [[ -d .venv ]]; then
        source .venv/bin/activate
      fi
      if [[ "${EXCAVATOR_SKIP_PIP_INSTALL}" != "1" ]]; then
        python -m pip install --no-deps -e ./testbed
      fi
      exec tb-receiver-real \
        --config "${CONFIG_PATH}" \
        --data-side slave \
        --backend bridge_tcp \
        --state-reader bridge_tcp \
        --bridge-host "${GATEWAY_HOST}" \
        --bridge-port "${GATEWAY_PORT}" \
        --bridge-timeout "${BRIDGE_TIMEOUT}" \
        --input remote \
        --remote-port "${RECEIVER_PORT}" \
        --num-episodes 1 \
        --max-steps "${MAX_STEPS}" \
        --output-dir "${DATASET_DIR}" \
        --session-id "${SESSION_ID}" \
        --wait-for-record-start \
        --live-action-line
    '
    wait_for_port "${GATEWAY_HOST}" "${RECEIVER_PORT}" receiver
  fi

  log "started. log dir: $(log_dir)"
  log "host teleop can now connect to slave port ${RECEIVER_PORT}"
}

status_stack() {
  local name pid state args pids
  printf 'log_dir=%s\n' "$(log_dir)"
  for name in "${SERVICES[@]}"; do
    pid="$(service_pid "${name}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
      state="running"
    else
      args=""
      state="stopped"
    fi
    printf '%-8s %-8s pid=%s %s\n' "${name}" "${state}" "${pid:-"-"}" "${args}"
  done
  for name in "bridge:${CONTROL_PORT}" "gateway:${GATEWAY_PORT}" "receiver:${RECEIVER_PORT}"; do
    pids="$(port_pids "${name#*:}" | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
    printf 'port %-16s %s\n' "${name}" "${pids:-free}"
  done
  if findmnt "${USB_MOUNT}" >/dev/null 2>&1; then
    printf 'usb mounted: %s\n' "${USB_MOUNT}"
  else
    printf 'usb not mounted: %s\n' "${USB_MOUNT}"
  fi
  if [[ -e "/dev/shm/${FPV_SHM_NAME#/}" ]]; then
    printf 'fpv shm present: /dev/shm/%s\n' "${FPV_SHM_NAME#/}"
  else
    printf 'fpv shm missing: /dev/shm/%s\n' "${FPV_SHM_NAME#/}"
  fi
}

show_logs() {
  local dir
  dir="$(log_dir)"
  printf '%s\n' "${dir}"
  ls -1 "${dir}" 2>/dev/null || true
}

tail_logs() {
  local service="${1:-}"
  local dir
  dir="$(log_dir)"
  if [[ -n "${service}" ]]; then
    service="$(canonical_service "${service}")"
    tail -n 80 -f "${dir}/${service}.log"
    return
  fi
  tail -n 80 -f "${dir}"/*.log
}

run_stack() {
  local force="$1"
  local no_camera="$2"
  local no_receiver="$3"
  local skip_usb="$4"
  local skip_can="$5"
  local skip_pip="$6"

  trap 'log "interrupt received; stopping managed slave services"; stop_stack 0; exit 130' INT TERM
  start_stack "${force}" "${no_camera}" "${no_receiver}" "${skip_usb}" "${skip_can}" "${skip_pip}"
  log "following logs. Press Ctrl+C here to stop managed slave services."
  tail_logs
}

main() {
  local command="${1:-}"
  local force=0 no_camera=0 no_receiver=0 skip_usb=0 skip_can=0 skip_pip=0
  local tail_service=""
  if [[ -z "${command}" ]]; then
    usage
    exit 1
  fi
  if [[ "${command}" == "-h" || "${command}" == "--help" ]]; then
    usage
    exit 0
  fi
  shift || true

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --force)
        force=1
        ;;
      --no-camera)
        no_camera=1
        ;;
      --no-receiver|--no-recorder)
        no_receiver=1
        ;;
      --skip-usb)
        skip_usb=1
        ;;
      --skip-can)
        skip_can=1
        ;;
      --skip-pip-install)
        skip_pip=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        if [[ "${command}" == "tail" && -z "${tail_service}" ]]; then
          tail_service="$1"
        else
          die "unknown argument: $1"
        fi
        ;;
    esac
    shift
  done

  case "${command}" in
    start)
      start_stack "${force}" "${no_camera}" "${no_receiver}" "${skip_usb}" "${skip_can}" "${skip_pip}"
      ;;
    run)
      run_stack "${force}" "${no_camera}" "${no_receiver}" "${skip_usb}" "${skip_can}" "${skip_pip}"
      ;;
    stop)
      stop_stack "${force}"
      ;;
    restart)
      stop_stack "${force}"
      start_stack 0 "${no_camera}" "${no_receiver}" "${skip_usb}" "${skip_can}" "${skip_pip}"
      ;;
    status)
      status_stack
      ;;
    logs)
      show_logs
      ;;
    tail)
      tail_logs "${tail_service}"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
