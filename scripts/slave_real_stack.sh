#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${EXCAVATOR_SLAVE_STACK_STATE_DIR:-/tmp/excavator_slave_stack}"
PID_DIR="${STATE_DIR}/pids"
LOG_ROOT="${EXCAVATOR_SLAVE_STACK_LOG_ROOT:-${ROOT_DIR}/artifacts/slave_stack}"
LOG_DIR_FILE="${STATE_DIR}/log_dir"
CAMERA_MODE_FILE="${STATE_DIR}/camera_mode"
LOG_VIEW="${EXCAVATOR_LOG_VIEW:-dashboard}"

CONTROL_HOST="${EXCAVATOR_CONTROL_HOST:-127.0.0.1}"
CONTROL_PORT="${EXCAVATOR_CONTROL_PORT:-8766}"
GATEWAY_HOST="${EXCAVATOR_GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${EXCAVATOR_GATEWAY_PORT:-8765}"
RECEIVER_PORT="${EXCAVATOR_REMOTE_ACTION_PORT:-8770}"
CAN_IF="${EXCAVATOR_CAN_IF:-can2}"
IMU_IF="${EXCAVATOR_IMU_IF:-can5}"
CAN_BITRATE="${EXCAVATOR_CAN_BITRATE:-250000}"
IMU_CAN_BITRATE="${EXCAVATOR_IMU_CAN_BITRATE:-1000000}"
IMU_RAW_CAN_LOG="${EXCAVATOR_IMU_RAW_CAN_LOG:-1}"
IMU_RAW_CAN_LOG_IF="${EXCAVATOR_IMU_RAW_CAN_LOG_IF:-${IMU_IF}}"
JOINT_RPY_PROFILE="${EXCAVATOR_JOINT_RPY_PROFILE:-daoyuan_chain}"
BUCKET_IMU0_PROFILE="${EXCAVATOR_BUCKET_IMU0_PROFILE:-roll_ccw90}"
BUCKET_QPOS_SOURCE="${EXCAVATOR_BUCKET_QPOS_SOURCE:-daoyuan_chain}"
BUCKET_IMU0_REFERENCE_RAD="${EXCAVATOR_BUCKET_IMU0_REFERENCE_RAD:-0}"
BUCKET_IMU0_SIGN="${EXCAVATOR_BUCKET_IMU0_SIGN:-${EXCAVATOR_BUCKET_IMU0_GYRO_SIGN:-1}}"
DAOYUAN_STICK_POLICY_OFFSET_RAD="${EXCAVATOR_DAOYUAN_STICK_POLICY_OFFSET_RAD:-0.19801020488135143}"
DAOYUAN_BUCKET_POLICY_OFFSET_RAD="${EXCAVATOR_DAOYUAN_BUCKET_POLICY_OFFSET_RAD:--2.006833804661174}"
BUCKET_GRAVITY_HINGE_REFERENCE_RAD="${EXCAVATOR_BUCKET_GRAVITY_HINGE_REFERENCE_RAD:-2.0839045979023254}"
BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD="${EXCAVATOR_BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD:--2.025561263010988}"
BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW="${EXCAVATOR_BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW:-21}"
USB_LABEL="${EXCAVATOR_USB_LABEL:-EXTERNAL_USB}"
USB_MOUNT="${EXCAVATOR_USB_MOUNT:-/media/${USER}/EXTERNAL_USB}"
DATASET_DIR="${EXCAVATOR_DATASET_DIR:-${USB_MOUNT}/real_teleop_v1}"
CONFIG_PATH="${EXCAVATOR_TELEOP_CONFIG:-${ROOT_DIR}/testbed/testbed/configs/teleop_real_v1.yaml}"
RECEIVER_INPUT="${EXCAVATOR_RECEIVER_INPUT:-remote}"
RECEIVER_RECORD_MODE="${EXCAVATOR_RECEIVER_RECORD_MODE:-config}"
POLICY_OUTPUT_MODE="${EXCAVATOR_POLICY_OUTPUT_MODE:-}"
POLICY_ACTION_SCALE="${EXCAVATOR_POLICY_ACTION_SCALE:-}"
TEST_LOG_DIR="${EXCAVATOR_TEST_LOG_DIR:-}"
PID_YAML_PATH="${EXCAVATOR_PID_YAML:-${ROOT_DIR}/control/config/joint_pid.yaml}"
SESSION_ID="${EXCAVATOR_SESSION_ID:-remote_teleop_slave_record}"
NUM_EPISODES="${EXCAVATOR_NUM_EPISODES:-1000000}"
MAX_STEPS="${EXCAVATOR_MAX_STEPS:-50000}"
BRIDGE_TIMEOUT="${EXCAVATOR_BRIDGE_TIMEOUT:-2.0}"
CONTROL_MODE="${EXCAVATOR_CONTROL_MODE:-open_loop_motor_speed}"
FPV_MAX_STALE_MS="${EXCAVATOR_FPV_MAX_STALE_MS:-1000}"
FPV_SHM_NAME="${EXCAVATOR_FPV_SHM_NAME:-excavator_fpv_v1}"
FPV_SHM_TIMEOUT_S="${EXCAVATOR_FPV_SHM_TIMEOUT_S:-45}"
CAMERA_STACK="${EXCAVATOR_CAMERA_STACK:-gmsl}"
GMSL_SHM_PREFIX="${EXCAVATOR_GMSL_SHM_PREFIX:-excavator_gmsl_}"
GMSL_GATEWAY_CAMERAS="${EXCAVATOR_GMSL_GATEWAY_CAMERAS:-video4=${GMSL_SHM_PREFIX}video4,video5=${GMSL_SHM_PREFIX}video5,video6=${GMSL_SHM_PREFIX}video6,video7=${GMSL_SHM_PREFIX}video7}"
GMSL_PREPROCESS_BIN="${EXCAVATOR_GMSL_PREPROCESS_BIN:-${ROOT_DIR}/tools/gmsl_realtime_capture/build/gmsl_realtime_preprocess_probe}"
GMSL_PREPROCESS_CAMERA_ARGS="${EXCAVATOR_GMSL_PREPROCESS_CAMERA_ARGS:---camera video4=/dev/video4 --camera video5=/dev/video5 --camera video6=/dev/video6 --camera video7=/dev/video7}"
GMSL_PREPROCESS_MANIFEST="${EXCAVATOR_GMSL_PREPROCESS_MANIFEST:-${ROOT_DIR}/configs/camera_calibration/gmsl_h190ta_four_camera/preprocess_manifest.json}"
GMSL_INTRINSICS_MANIFEST="${EXCAVATOR_GMSL_INTRINSICS_MANIFEST:-${ROOT_DIR}/configs/camera_intrinsics/gmsl_h190ta/manifest.json}"
GMSL_BUFFERS="${EXCAVATOR_GMSL_BUFFERS:-8}"
STARTUP_TIMEOUT_S="${EXCAVATOR_STARTUP_TIMEOUT_S:-45}"
RECEIVER_STOP_TIMEOUT_S="${EXCAVATOR_RECEIVER_STOP_TIMEOUT_S:-${EXCAVATOR_RECORDER_STOP_TIMEOUT_S:-180}}"
EXCAVATOR_SKIP_PIP_INSTALL="${EXCAVATOR_SKIP_PIP_INSTALL:-1}"

if [[ "${CAMERA_STACK}" == "gmsl" ]]; then
  SERVICES=(canraw bridge gmsl gateway receiver)
  STOP_ORDER=(receiver gateway gmsl bridge canraw)
else
  SERVICES=(canraw bridge orbbec fpv gateway receiver)
  STOP_ORDER=(receiver gateway fpv orbbec bridge canraw)
fi
STARTED_SERVICES=()

usage() {
  cat <<'EOF'
Usage:
  scripts/slave_real_stack.sh start [--force] [--policy-remote] [--no-camera] [--no-receiver] [--skip-usb] [--skip-can] [--install-python-package]
  scripts/slave_real_stack.sh run [--force] [--policy-remote] [--no-camera] [--no-receiver] [--skip-usb] [--skip-can] [--install-python-package]
  scripts/slave_real_stack.sh stop [--force]
  scripts/slave_real_stack.sh restart [--force] [start options]
  scripts/slave_real_stack.sh status
  scripts/slave_real_stack.sh logs
  scripts/slave_real_stack.sh tail [service]

Default start services:
  bridge, GMSL four-camera preprocess publisher, gateway, receiver.

Use "run" when you want one foreground terminal that shows logs and stops the
managed services on Ctrl+C.

Common profiles:
  --policy-remote  Start the single field receiver for manual teleop, go-home,
                   recording, and policy control toggle.
  --no-camera      Do not start camera services; gateway read_state returns
                   empty images for read-only IMU/qvel diagnostics.

Common environment overrides:
  EXCAVATOR_USB_MOUNT=/media/mundane/EXTERNAL_USB
  EXCAVATOR_DATASET_DIR=/media/mundane/EXTERNAL_USB/real_teleop_v1
  EXCAVATOR_ORBBEC_WS=/home/mundane/orbbec_ws
  EXCAVATOR_ROS_WS=/home/mundane/orbbec_ws
  EXCAVATOR_CAN_IF=can2 EXCAVATOR_CAN_BITRATE=250000
  EXCAVATOR_IMU_IF=can5 EXCAVATOR_IMU_CAN_BITRATE=1000000  # set EXCAVATOR_IMU_IF=usbcan0 for ZLG USBCAN IMU input
  EXCAVATOR_IMU_RAW_CAN_LOG=1                 # set 0 to disable background candump
  EXCAVATOR_IMU_RAW_CAN_LOG_IF=can5           # defaults to EXCAVATOR_IMU_IF; SocketCAN only
  EXCAVATOR_JOINT_RPY_PROFILE=daoyuan_chain   # daoyuan_chain | legacy_diff
  EXCAVATOR_BUCKET_QPOS_SOURCE=daoyuan_chain  # daoyuan_chain | gravity_hinge | rpy | legacy_quaternion
  EXCAVATOR_BUCKET_IMU0_PROFILE=roll_ccw90    # legacy_y | roll_ccw90; current bucket IMU0 is rotated
  EXCAVATOR_BUCKET_IMU0_REFERENCE_RAD=0       # roll_ccw90 outer-pose native RPY reference
  EXCAVATOR_BUCKET_IMU0_SIGN=1                # set -1 if bucket qpos/qvel sign is reversed
  EXCAVATOR_DAOYUAN_STICK_POLICY_OFFSET_RAD=0.19801020488135143
  EXCAVATOR_DAOYUAN_BUCKET_POLICY_OFFSET_RAD=-2.006833804661174
  EXCAVATOR_BUCKET_GRAVITY_HINGE_REFERENCE_RAD=2.0839045979023254
  EXCAVATOR_BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD=-2.025561263010988
  EXCAVATOR_BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW=21
  EXCAVATOR_PID_YAML=/media/mundane/D/Excavator_real_stack/control/config/joint_pid.yaml
  EXCAVATOR_CONTROL_MODE=open_loop_motor_speed
  EXCAVATOR_CAMERA_STACK=gmsl                 # gmsl | orbbec
  EXCAVATOR_GMSL_PREPROCESS_BIN=/media/mundane/D/Excavator_real_stack/tools/gmsl_realtime_capture/build/gmsl_realtime_preprocess_probe
  EXCAVATOR_GMSL_GATEWAY_CAMERAS=video4=excavator_gmsl_video4,video5=excavator_gmsl_video5,video6=excavator_gmsl_video6,video7=excavator_gmsl_video7
  EXCAVATOR_RECEIVER_INPUT=remote
  EXCAVATOR_RECEIVER_RECORD_MODE=config       # config | record | no-record
  EXCAVATOR_POLICY_OUTPUT_MODE=control        # optional for policy/policy_remote
  EXCAVATOR_POLICY_ACTION_SCALE=1.0           # optional for policy/policy_remote
  EXCAVATOR_TEST_LOG_DIR=/media/mundane/EXTERNAL_USB/policy_control_tests
  EXCAVATOR_NUM_EPISODES=1000000
  EXCAVATOR_RECEIVER_STOP_TIMEOUT_S=180
  EXCAVATOR_SKIP_PIP_INSTALL=1
  EXCAVATOR_LOG_VIEW=dashboard              # dashboard | plain

Compatibility:
  --no-recorder and "tail recorder" remain aliases for receiver.
  --skip-pip-install remains accepted; receiver already skips pip by default.
EOF
}

log() {
  printf '[slave-stack] %s\n' "$*"
}

die() {
  printf '[slave-stack] error: %s\n' "$*" >&2
  exit 1
}

is_socketcan_if() {
  [[ "$1" =~ ^can[0-9]+$ ]]
}

apply_policy_remote_profile() {
  CONFIG_PATH="${EXCAVATOR_TELEOP_CONFIG:-${ROOT_DIR}/testbed/testbed/configs/policy_real_gmsl_four_camera_v1.yaml}"
  RECEIVER_INPUT="${EXCAVATOR_RECEIVER_INPUT:-policy_remote}"
  RECEIVER_RECORD_MODE="${EXCAVATOR_RECEIVER_RECORD_MODE:-record}"
  POLICY_OUTPUT_MODE="${EXCAVATOR_POLICY_OUTPUT_MODE:-}"
  POLICY_ACTION_SCALE="${EXCAVATOR_POLICY_ACTION_SCALE:-}"
  DATASET_DIR="${EXCAVATOR_DATASET_DIR:-${USB_MOUNT}/real_teleop_v1}"
  TEST_LOG_DIR="${EXCAVATOR_TEST_LOG_DIR:-${USB_MOUNT}/policy_control_tests}"
  NUM_EPISODES="${EXCAVATOR_NUM_EPISODES:-1000000}"
  MAX_STEPS="${EXCAVATOR_MAX_STEPS:-50000}"
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

wait_for_gmsl_shm() {
  local deadline=$((SECONDS + FPV_SHM_TIMEOUT_S))
  local mapping camera shm_name all_present
  local -a mappings
  while (( SECONDS < deadline )); do
    all_present=1
    IFS=',' read -r -a mappings <<<"${GMSL_GATEWAY_CAMERAS}"
    for mapping in "${mappings[@]}"; do
      [[ -n "${mapping}" ]] || continue
      camera="${mapping%%=*}"
      shm_name="${mapping#*=}"
      if [[ "${camera}" == "${mapping}" || -z "${shm_name}" || ! -e "/dev/shm/${shm_name#/}" ]]; then
        all_present=0
        break
      fi
    done
    if [[ "${all_present}" == "1" ]]; then
      log "GMSL SHM is present for: ${GMSL_GATEWAY_CAMERAS}"
      return 0
    fi
    sleep 0.5
  done
  die "GMSL SHM was not created for all cameras: ${GMSL_GATEWAY_CAMERAS}. Check $(log_dir)/gmsl.log"
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
  stop_service gmsl INT 12
  stop_service fpv INT 10
  stop_service orbbec INT 10
  stop_service bridge TERM 8
  stop_service canraw INT 5
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
  log "setting up CAN ${CAN_IF} bitrate=${CAN_BITRATE}; IMU ${IMU_IF} bitrate=${IMU_CAN_BITRATE}"
  "${ROOT_DIR}/control/setup/setup_can.sh" "${CAN_IF}" "${CAN_BITRATE}"
  if is_socketcan_if "${IMU_IF}"; then
    "${ROOT_DIR}/control/setup/setup_can.sh" "${IMU_IF}" "${IMU_CAN_BITRATE}"
  else
    log "skipping SocketCAN setup for IMU_IF=${IMU_IF}; bridge will open it directly"
  fi
}

prepare_start() {
  local no_camera="${1:-0}"
  local run_id log_dir_path
  [[ -x "${ROOT_DIR}/bridge/build/excavator_real_bridge" ]] \
    || die "missing bridge binary: ${ROOT_DIR}/bridge/build/excavator_real_bridge"
  if [[ "${CAMERA_STACK}" == "gmsl" && "${no_camera}" != "1" ]]; then
    [[ -x "${GMSL_PREPROCESS_BIN}" ]] \
      || die "missing GMSL preprocess binary: ${GMSL_PREPROCESS_BIN}. Build tools/gmsl_realtime_capture on the Jetson first."
  fi
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
  local skip_pip="${6:-${EXCAVATOR_SKIP_PIP_INSTALL}}"

  if [[ "${force}" == "1" ]]; then
    stop_stack 1
  fi

  require_port_free "${CONTROL_PORT}" bridge
  require_port_free "${GATEWAY_PORT}" gateway
  if [[ "${no_receiver}" != "1" ]]; then
    require_port_free "${RECEIVER_PORT}" receiver
  fi

  prepare_start "${no_camera}"
  if [[ "${no_camera}" == "1" ]]; then
    printf 'none\n' >"${CAMERA_MODE_FILE}"
  else
    printf '%s\n' "${CAMERA_STACK}" >"${CAMERA_MODE_FILE}"
  fi
  export ROOT_DIR CONTROL_HOST CONTROL_PORT GATEWAY_HOST GATEWAY_PORT RECEIVER_PORT
  export CAN_IF IMU_IF IMU_RAW_CAN_LOG_IF DATASET_DIR CONFIG_PATH RECEIVER_INPUT RECEIVER_RECORD_MODE
  export EXCAVATOR_JOINT_RPY_PROFILE="${JOINT_RPY_PROFILE}"
  export EXCAVATOR_BUCKET_IMU0_PROFILE="${BUCKET_IMU0_PROFILE}"
  export EXCAVATOR_BUCKET_QPOS_SOURCE="${BUCKET_QPOS_SOURCE}"
  export EXCAVATOR_BUCKET_IMU0_REFERENCE_RAD="${BUCKET_IMU0_REFERENCE_RAD}"
  export EXCAVATOR_BUCKET_IMU0_SIGN="${BUCKET_IMU0_SIGN}"
  export EXCAVATOR_BUCKET_IMU0_GYRO_SIGN="${BUCKET_IMU0_SIGN}"
  export EXCAVATOR_DAOYUAN_STICK_POLICY_OFFSET_RAD="${DAOYUAN_STICK_POLICY_OFFSET_RAD}"
  export EXCAVATOR_DAOYUAN_BUCKET_POLICY_OFFSET_RAD="${DAOYUAN_BUCKET_POLICY_OFFSET_RAD}"
  export EXCAVATOR_BUCKET_GRAVITY_HINGE_REFERENCE_RAD="${BUCKET_GRAVITY_HINGE_REFERENCE_RAD}"
  export EXCAVATOR_BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD="${BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD}"
  export EXCAVATOR_BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW="${BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW}"
  export POLICY_OUTPUT_MODE POLICY_ACTION_SCALE TEST_LOG_DIR
  export PID_YAML_PATH SESSION_ID NUM_EPISODES MAX_STEPS BRIDGE_TIMEOUT CONTROL_MODE
  export EXCAVATOR_SKIP_PIP_INSTALL
  export EXCAVATOR_NO_CAMERA="${no_camera}"
  export FPV_MAX_STALE_MS FPV_SHM_NAME
  export CAMERA_STACK GMSL_SHM_PREFIX GMSL_GATEWAY_CAMERAS GMSL_PREPROCESS_BIN
  export GMSL_PREPROCESS_CAMERA_ARGS GMSL_PREPROCESS_MANIFEST GMSL_INTRINSICS_MANIFEST GMSL_BUFFERS
  export EXCAVATOR_ORBBEC_WS="${EXCAVATOR_ORBBEC_WS:-${HOME}/orbbec_ws}"
  export EXCAVATOR_ROS_WS="${EXCAVATOR_ROS_WS:-${EXCAVATOR_ORBBEC_WS}}"
  export EXCAVATOR_SKIP_PIP_INSTALL="${skip_pip}"

  if [[ "${skip_usb}" != "1" && "${no_receiver}" != "1" ]]; then
    mount_usb
  fi
  if [[ "${skip_can}" != "1" ]]; then
    setup_can
  fi
  log "joint rpy profile=${JOINT_RPY_PROFILE} bucket qpos source=${BUCKET_QPOS_SOURCE} imu0_profile=${BUCKET_IMU0_PROFILE} reference_rad=${BUCKET_IMU0_REFERENCE_RAD} sign=${BUCKET_IMU0_SIGN} daoyuan_stick_offset=${DAOYUAN_STICK_POLICY_OFFSET_RAD} daoyuan_bucket_offset=${DAOYUAN_BUCKET_POLICY_OFFSET_RAD} gravity_ref=${BUCKET_GRAVITY_HINGE_REFERENCE_RAD} gravity_offset=${BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD} median=${BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW}"

  if [[ "${IMU_RAW_CAN_LOG}" == "1" ]]; then
    if ! is_socketcan_if "${IMU_RAW_CAN_LOG_IF}"; then
      log "skipping raw IMU candump for non-SocketCAN interface: ${IMU_RAW_CAN_LOG_IF}"
    else
      command -v candump >/dev/null 2>&1 || die "candump not found; install can-utils or set EXCAVATOR_IMU_RAW_CAN_LOG=0"
      start_service canraw bash -lc '
        printf "[slave-stack] recording raw IMU CAN: candump -ta %s\n" "${IMU_RAW_CAN_LOG_IF}" >&2
        exec candump -ta "${IMU_RAW_CAN_LOG_IF}"
      '
    fi
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
      --control-mode "${CONTROL_MODE}" \
      --pid-yaml "${PID_YAML_PATH}" \
      --heartbeat-timeout-ms 800
  '
  wait_for_port "${CONTROL_HOST}" "${CONTROL_PORT}" bridge

  if [[ "${no_camera}" != "1" ]]; then
    case "${CAMERA_STACK}" in
      gmsl)
        start_service gmsl bash -lc '
          cd "${ROOT_DIR}"
          read -r -a camera_args <<<"${GMSL_PREPROCESS_CAMERA_ARGS}"
          exec "${GMSL_PREPROCESS_BIN}" \
            "${camera_args[@]}" \
            --manifest "${GMSL_INTRINSICS_MANIFEST}" \
            --preprocess-manifest "${GMSL_PREPROCESS_MANIFEST}" \
            --frames 0 \
            --buffers "${GMSL_BUFFERS}" \
            --publish-shm \
            --shm-prefix "${GMSL_SHM_PREFIX}" \
            --detail-frames 0
        '
        wait_for_gmsl_shm
        ;;
      orbbec)
        start_service orbbec bash -lc '
          cd "${ROOT_DIR}"
          exec ./scripts/start_orbbec_fpv_camera.sh
        '
        start_service fpv bash -lc '
          cd "${ROOT_DIR}"
          exec ./scripts/start_fpv_subscriber_py.sh
        '
        wait_for_shm
        ;;
      *)
        die "invalid EXCAVATOR_CAMERA_STACK=${CAMERA_STACK}; expected gmsl or orbbec"
        ;;
    esac
  fi

  start_service gateway bash -lc '
    cd "${ROOT_DIR}"
    if [[ "${EXCAVATOR_NO_CAMERA}" == "1" ]]; then
      export EXCAVATOR_CAMERA_SOURCE=none
    elif [[ "${CAMERA_STACK}" == "gmsl" ]]; then
      export EXCAVATOR_CAMERA_SOURCE=gmsl
      export EXCAVATOR_GMSL_GATEWAY_CAMERAS="${GMSL_GATEWAY_CAMERAS}"
    else
      export EXCAVATOR_CAMERA_SOURCE=fpv
    fi
    exec ./scripts/start_bridge_gateway.sh \
      --fpv-source "${EXCAVATOR_FPV_SOURCE:-auto}" \
      --fpv-max-stale-ms "${FPV_MAX_STALE_MS}"
  '
  wait_for_port "${GATEWAY_HOST}" "${GATEWAY_PORT}" gateway

  if [[ "${no_receiver}" != "1" ]]; then
    start_service receiver bash -lc '
      cd "${ROOT_DIR}"
      if [[ -d .venv ]]; then
        source .venv/bin/activate
      fi
      export PYTHONPATH="${ROOT_DIR}/testbed${PYTHONPATH:+:${PYTHONPATH}}"
      CU12_LIB="${ROOT_DIR}/.venv/lib/python3.10/site-packages/nvidia/cu12/lib"
      if [[ -d "${CU12_LIB}" ]]; then
        export LD_LIBRARY_PATH="${CU12_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      fi
      if [[ "${EXCAVATOR_SKIP_PIP_INSTALL}" != "1" ]]; then
        python -m pip install --no-build-isolation --no-deps -e ./testbed
      fi
      extra_args=()
      case "${RECEIVER_RECORD_MODE}" in
        config|"")
          ;;
        record)
          extra_args+=(--record)
          ;;
        no-record)
          extra_args+=(--no-record)
          ;;
        *)
          printf "[slave-stack] error: invalid EXCAVATOR_RECEIVER_RECORD_MODE=%s\n" "${RECEIVER_RECORD_MODE}" >&2
          exit 2
          ;;
      esac
      if [[ -n "${POLICY_OUTPUT_MODE}" ]]; then
        extra_args+=(--policy-output-mode "${POLICY_OUTPUT_MODE}")
      fi
      if [[ -n "${POLICY_ACTION_SCALE}" ]]; then
        extra_args+=(--policy-action-scale "${POLICY_ACTION_SCALE}")
      fi
      if [[ -n "${TEST_LOG_DIR}" ]]; then
        extra_args+=(--test-log-dir "${TEST_LOG_DIR}")
      fi
      exec python -m testbed.cli.record_real \
        --config "${CONFIG_PATH}" \
        --data-side slave \
        --backend bridge_tcp \
        --state-reader bridge_tcp \
        --bridge-host "${GATEWAY_HOST}" \
        --bridge-port "${GATEWAY_PORT}" \
        --bridge-timeout "${BRIDGE_TIMEOUT}" \
        --input "${RECEIVER_INPUT}" \
        --remote-port "${RECEIVER_PORT}" \
        --num-episodes "${NUM_EPISODES}" \
        --max-steps "${MAX_STEPS}" \
        --output-dir "${DATASET_DIR}" \
        --session-id "${SESSION_ID}" \
        --wait-for-record-start \
        --live-action-line \
        "${extra_args[@]}"
    '
    wait_for_port "${GATEWAY_HOST}" "${RECEIVER_PORT}" receiver
  fi

  log "started. log dir: $(log_dir)"
  if [[ "${no_receiver}" == "1" ]]; then
    log "receiver disabled; host teleop port ${RECEIVER_PORT} is not listening"
  else
    log "host teleop can now connect to slave port ${RECEIVER_PORT}"
  fi
}

status_stack() {
  local name pid state args pids mapping shm_name camera_mode
  local -a mappings
  if [[ -s "${CAMERA_MODE_FILE}" ]]; then
    camera_mode="$(cat "${CAMERA_MODE_FILE}")"
  else
    camera_mode="${CAMERA_STACK}"
  fi
  printf 'log_dir=%s\n' "$(log_dir)"
  printf 'camera_mode=%s\n' "${camera_mode}"
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
  if [[ "${camera_mode}" == "none" ]]; then
    printf 'camera disabled: read_state images are empty\n'
  elif [[ "${camera_mode}" == "gmsl" ]]; then
    IFS=',' read -r -a mappings <<<"${GMSL_GATEWAY_CAMERAS}"
    for mapping in "${mappings[@]}"; do
      [[ -n "${mapping}" ]] || continue
      shm_name="${mapping#*=}"
      if [[ -e "/dev/shm/${shm_name#/}" ]]; then
        printf 'gmsl shm present: /dev/shm/%s\n' "${shm_name#/}"
      else
        printf 'gmsl shm missing: /dev/shm/%s\n' "${shm_name#/}"
      fi
    done
  elif [[ -e "/dev/shm/${FPV_SHM_NAME#/}" ]]; then
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

dashboard_logs() {
  local dir="$1"
  local service_list
  service_list="$(IFS=,; printf '%s' "${SERVICES[*]}")"
  EXCAVATOR_DASH_PID_DIR="${PID_DIR}" \
  EXCAVATOR_DASH_CAN_IF="${CAN_IF}" \
  EXCAVATOR_DASH_IMU_IF="${IMU_IF}" \
  EXCAVATOR_DASH_CONTROL_PORT="${CONTROL_PORT}" \
  EXCAVATOR_DASH_GATEWAY_PORT="${GATEWAY_PORT}" \
  EXCAVATOR_DASH_RECEIVER_PORT="${RECEIVER_PORT}" \
  python3 - "${dir}" "${service_list}" <<'PY'
from __future__ import annotations

import glob
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


log_dir = Path(sys.argv[1])
services = [item for item in sys.argv[2].split(",") if item]
pid_dir = Path(os.environ.get("EXCAVATOR_DASH_PID_DIR", ""))
can_if = os.environ.get("EXCAVATOR_DASH_CAN_IF", "-")
imu_if = os.environ.get("EXCAVATOR_DASH_IMU_IF", "-")
ports = (
    os.environ.get("EXCAVATOR_DASH_CONTROL_PORT", "-"),
    os.environ.get("EXCAVATOR_DASH_GATEWAY_PORT", "-"),
    os.environ.get("EXCAVATOR_DASH_RECEIVER_PORT", "-"),
)

log_files = sorted(glob.glob(str(log_dir / "*.log")))
if not log_files:
    print(f"[slave-stack] no log files found under {log_dir}", flush=True)
    raise SystemExit(0)

ansi_re = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
field_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\[[^\]]*\]|[^ ]*)")
timestamp_re = re.compile(r"(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9:,]+ .*)")

receiver_status = ""
last_event = "waiting for log events"
current_service = "-"
top_rows = 8


def term_size() -> tuple[int, int]:
    size = shutil.get_terminal_size((160, 40))
    return max(12, size.lines), max(40, size.columns)


def fit(text: str, width: int) -> str:
    text = text.replace("\t", " ")
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "~"


def fields(text: str) -> dict[str, str]:
    return {key: value for key, value in field_re.findall(text)}


def service_state(name: str) -> str:
    pid_file = pid_dir / f"{name}.pid"
    try:
        pid = pid_file.read_text(encoding="utf-8").strip()
    except OSError:
        return f"{name}:stopped"
    if not pid.isdigit():
        return f"{name}:badpid"
    if Path(f"/proc/{pid}").exists():
        return f"{name}:run({pid})"
    return f"{name}:dead({pid})"


def receiver_lines(width: int) -> tuple[str, str, str]:
    if not receiver_status:
        return ("receiver: waiting for first live status", "", "")
    data = fields(receiver_status)
    line1 = (
        "receiver "
        f"mode={data.get('mode', '-')} control={data.get('control', '-')} "
        f"health={data.get('health', '-')} err={data.get('err', '-')} "
        f"imu={data.get('imu', '-')} hz={data.get('hz', '-')} "
        f"ctl_ms={data.get('ctl_ms', '-')} step={data.get('step', '-')} "
        f"ack={data.get('ack', '-')} fault={data.get('fault', '-')}"
    )
    line2 = (
        "action   "
        f"raw={data.get('raw', '-')} send={data.get('send', '-')} "
        f"remote_ms={data.get('remote_ms', '-')} stale={data.get('stale', '-')} "
        f"drop={data.get('drop', '-')} age={data.get('age', '-')} "
        f"guard={data.get('guard', '-')}"
    )
    home = ""
    if "home" in data:
        home = (
            "go_home "
            f"home={data.get('home', '-')} t={data.get('t', '-')} "
            f"maxerr={data.get('maxerr', '-')} rawmaxerr={data.get('rawmaxerr', '-')} "
            f"maxaxis={data.get('maxaxis', '-')} target={data.get('target', '-')} "
            f"wrongdir={data.get('wrongdir', '-')} hrawerr={data.get('hrawerr', '-')}"
        )
    return fit(line1, width), fit(line2, width), fit(home, width)


def draw_top() -> None:
    rows, cols = term_size()
    service_text = " ".join(service_state(name) for name in services)
    recv1, recv2, recv3 = receiver_lines(cols)
    lines = [
        "Excavator slave stack dashboard | Ctrl+C stops managed services",
        f"log_dir={log_dir}",
        f"can={can_if} imu={imu_if} ports bridge/gateway/receiver={ports[0]}/{ports[1]}/{ports[2]}",
        f"services {service_text}",
        recv1,
        recv2,
        recv3,
        f"last_event {last_event}",
    ]
    rows_used = min(top_rows, rows - 3)
    sys.stdout.write("\0337")
    for row in range(rows_used):
        sys.stdout.write(f"\033[{row + 1};1H\033[K{fit(lines[row], cols)}")
    sys.stdout.write(f"\033[{rows_used + 1};{rows}r")
    sys.stdout.write("\0338")
    sys.stdout.flush()


def scroll_event(text: str) -> None:
    global last_event
    clean = fit(text, term_size()[1])
    last_event = clean
    rows, _ = term_size()
    sys.stdout.write(f"\033[{rows};1H\033[K{clean}\n")
    sys.stdout.flush()


def service_from_header(text: str) -> str | None:
    match = re.match(r"==> (.*) <==", text)
    if not match:
        return None
    name = Path(match.group(1)).name
    return name[:-4] if name.endswith(".log") else name


def emit(raw: bytes) -> None:
    global current_service, receiver_status
    text = raw.decode(errors="replace")
    text = ansi_re.sub("", text).strip()
    if not text:
        return
    header_service = service_from_header(text)
    if header_service is not None:
        current_service = header_service
        return
    if text.startswith("mode="):
        receiver_status = text
        event_match = timestamp_re.search(text)
        if event_match:
            scroll_event(f"[receiver] {event_match.group(1)}")
        draw_top()
        return
    scroll_event(f"[{current_service}] {text}")
    draw_top()


proc = subprocess.Popen(
    ["tail", "-n", "80", "-F", *log_files],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=0,
)

def terminate(_signum: int, _frame: object) -> None:
    proc.terminate()
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)

sys.stdout.write("\033[?25l\033[2J\033[H")
draw_top()
rows, _ = term_size()
sys.stdout.write(f"\033[{min(top_rows, rows - 3) + 1};1H")
sys.stdout.flush()

buffer = bytearray()
next_draw = 0.0
try:
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    while True:
        now = time.monotonic()
        if now >= next_draw:
            draw_top()
            next_draw = now + 1.0
        ready, _, _ = select.select([fd], [], [], 0.2)
        if not ready:
            if proc.poll() is not None:
                break
            continue
        chunk = os.read(fd, 4096)
        if not chunk:
            if proc.poll() is not None:
                break
            continue
        for byte in chunk:
            if byte in (10, 13):
                emit(bytes(buffer))
                buffer.clear()
            else:
                buffer.append(byte)
finally:
    if proc.poll() is None:
        proc.terminate()
    sys.stdout.write("\033[r\033[?25h\n")
    sys.stdout.flush()
PY
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
  if [[ "${LOG_VIEW}" == "dashboard" && -t 1 ]] && command -v python3 >/dev/null 2>&1; then
    dashboard_logs "${dir}"
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
  local force=0 no_camera=0 no_receiver=0 skip_usb=0 skip_can=0 skip_pip="${EXCAVATOR_SKIP_PIP_INSTALL}"
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
      --policy-remote)
        apply_policy_remote_profile
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
      --install-python-package)
        skip_pip=0
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
