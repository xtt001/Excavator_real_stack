#!/usr/bin/env bash
set -uo pipefail

PYTHON="${PYTHON:-$HOME/miniforge3/envs/excavator-real-stack/bin/python}"
SSH_HOST="${SSH_HOST:-slave-jetson}"
SSH_USER="${SSH_USER:-mundane}"
REMOTE_DIR="${REMOTE_DIR:-/media/mundane/EXTERNAL_USB/real_teleop_v1}"
DAY="${DAY:-$(date +%F)}"
BASE_DIR="${BASE_DIR:-}"
POLL_S="${POLL_S:-10}"
STABLE_INTERVAL_S="${STABLE_INTERVAL_S:-5}"
MIN_MTIME_AGE_S="${MIN_MTIME_AGE_S:-10}"
HISTORY_FROM="${HISTORY_FROM:-0}"
SYNC_REMOTE_TIME="${SYNC_REMOTE_TIME:-1}"
MAX_TIME_SKEW_S="${MAX_TIME_SKEW_S:-5}"
TIME_SYNC_WARN_INTERVAL_S="${TIME_SYNC_WARN_INTERVAL_S:-60}"
SCAN_ALL_ON_TIME_SYNC_FAIL="${SCAN_ALL_ON_TIME_SYNC_FAIL:-1}"
INTERACTIVE_REMOTE_SUDO="${INTERACTIVE_REMOTE_SUDO:-1}"
LOG_FILE="${LOG_FILE:-}"
LAST_TIME_SYNC_WARN_EPOCH=0
LAST_FALLBACK_WARN_EPOCH=0
LOCK_DIR=""
LOCK_HELD=0
ACTIVE_CHILD_PID=""
RUN_CAPTURE_OUTPUT=""
STOPPING=0
TIME_SYNC_OK=1
REMOTE_TIME_WAS_CORRECTED=0
ACTIVE_TMP_PATH=""

usage() {
  cat <<'EOF'
Usage:
  scripts/watch_today_qc_from_slave.sh [options]

Options:
  --ssh-host HOST              SSH host alias or IP. Default: slave-jetson
  --ssh-user USER              SSH user. Default: mundane
  --remote-dir DIR             Slave-side episode directory.
  --day YYYY-MM-DD             Host-side date to watch. Default: today.
  --base-dir DIR               Host output base. Default: data/qc_today_<day>
  --poll-s SEC                 Poll interval. Default: 10
  --stable-interval-s SEC      Delay between two remote stat checks. Default: 5
  --min-mtime-age-s SEC        Skip very fresh files. Default: 10
  --history-from N             Only print per-episode history from episode_N. Default: 0
  --no-sync-remote-time        Do not try to sync slave time before scans.
  --max-time-skew-s SEC        Sync slave time when skew exceeds this. Default: 5
  --no-interactive-remote-sudo
                              Do not prompt for the slave sudo password.
  --no-scan-all-on-time-sync-fail
                              Keep date-filtered scans even if time sync fails.
  --log-file FILE              Also append watcher output to FILE.
  -h, --help                   Show this help.

Environment:
  PYTHON can override the Python executable.
  SYNC_REMOTE_TIME=0 disables remote time sync.
  INTERACTIVE_REMOTE_SUDO=0 disables the slave sudo password prompt.
  SCAN_ALL_ON_TIME_SYNC_FAIL=0 disables the bad-clock scan fallback.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ssh-host)
      SSH_HOST="$2"
      shift 2
      ;;
    --ssh-user)
      SSH_USER="$2"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --day)
      DAY="$2"
      shift 2
      ;;
    --base-dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --poll-s)
      POLL_S="$2"
      shift 2
      ;;
    --stable-interval-s)
      STABLE_INTERVAL_S="$2"
      shift 2
      ;;
    --min-mtime-age-s)
      MIN_MTIME_AGE_S="$2"
      shift 2
      ;;
    --history-from)
      HISTORY_FROM="$2"
      shift 2
      ;;
    --no-sync-remote-time)
      SYNC_REMOTE_TIME=0
      shift
      ;;
    --max-time-skew-s)
      MAX_TIME_SKEW_S="$2"
      shift 2
      ;;
    --no-interactive-remote-sudo)
      INTERACTIVE_REMOTE_SUDO=0
      shift
      ;;
    --no-scan-all-on-time-sync-fail)
      SCAN_ALL_ON_TIME_SYNC_FAIL=0
      shift
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$BASE_DIR" ]; then
  BASE_DIR="data/qc_today_${DAY}"
fi
EP_DIR="${BASE_DIR}/episodes"
REPORT_DIR="${BASE_DIR}/report"
REMOTE_TARGET="${SSH_USER}@${SSH_HOST}"
PID_FILE="${BASE_DIR}/qc_watch.pid"
LOCK_DIR="${BASE_DIR}/qc_watch.lock"

log() {
  printf '%(%F %T)T %s\n' -1 "$*"
}

remote_shell_quote() {
  printf '%q' "$1"
}

episode_number() {
  local episode_id="$1"
  episode_id="${episode_id#episode_}"
  episode_id="${episode_id%.hdf5}"
  printf '%s' "$episode_id"
}

cleanup() {
  if [ -n "$ACTIVE_TMP_PATH" ]; then
    rm -f "$ACTIVE_TMP_PATH"
    ACTIVE_TMP_PATH=""
  fi
  if [ "$LOCK_HELD" = 1 ]; then
    rm -rf "$LOCK_DIR"
    rm -f "$PID_FILE"
    LOCK_HELD=0
  fi
}

stop_watcher() {
  trap - INT TERM HUP QUIT EXIT
  STOPPING=1
  if [ -n "$ACTIVE_CHILD_PID" ]; then
    kill "$ACTIVE_CHILD_PID" 2>/dev/null || true
    wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
    ACTIVE_CHILD_PID=""
  fi
  cleanup
  log "today QC watcher stopping" || true
  exit 130
}

run_command() {
  "$@" &
  ACTIVE_CHILD_PID=$!
  wait "$ACTIVE_CHILD_PID"
  local rc=$?
  ACTIVE_CHILD_PID=""
  if [ "$rc" -ge 128 ] && [ "$STOPPING" = 0 ]; then
    stop_watcher
  fi
  return "$rc"
}

run_capture() {
  local tmp rc
  tmp="$(mktemp)"
  "$@" >"$tmp" 2>&1 &
  ACTIVE_CHILD_PID=$!
  wait "$ACTIVE_CHILD_PID"
  rc=$?
  ACTIVE_CHILD_PID=""
  RUN_CAPTURE_OUTPUT="$(cat "$tmp")"
  rm -f "$tmp"
  if [ "$rc" -ge 128 ] && [ "$STOPPING" = 0 ]; then
    stop_watcher
  fi
  return "$rc"
}

run_tty_command() {
  "$@" </dev/tty &
  ACTIVE_CHILD_PID=$!
  wait "$ACTIVE_CHILD_PID"
  local rc=$?
  ACTIVE_CHILD_PID=""
  if [ "$rc" -ge 128 ] && [ "$STOPPING" = 0 ]; then
    stop_watcher
  fi
  return "$rc"
}

interruptible_sleep() {
  run_command sleep "$1"
}

is_live_watcher_pid() {
  local pid="$1"
  local stat args

  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi

  stat="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
  args="$(ps -o args= -p "$pid" 2>/dev/null || true)"
  if [[ "$stat" == *Z* ]]; then
    return 1
  fi
  [[ "$args" == *"watch_today_qc_from_slave.sh"* ]]
}

acquire_lock() {
  local existing_pid

  if [ ! -d "$LOCK_DIR" ] && [ -f "$PID_FILE" ]; then
    existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if is_live_watcher_pid "$existing_pid"; then
      log "another QC watcher is already running for ${BASE_DIR}: pid=${existing_pid}"
      return 1
    fi
    log "remove stale QC watcher pid: ${PID_FILE} pid=${existing_pid:-unknown}"
    rm -f "$PID_FILE"
  fi

  if mkdir "$LOCK_DIR" 2>/dev/null; then
    LOCK_HELD=1
    printf '%s\n' "$$" > "${LOCK_DIR}/pid"
    printf '%s\n' "$$" > "$PID_FILE"
    return 0
  fi

  existing_pid="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
  if is_live_watcher_pid "$existing_pid"; then
    log "another QC watcher is already running for ${BASE_DIR}: pid=${existing_pid}"
    return 1
  fi

  log "remove stale QC watcher lock: ${LOCK_DIR}"
  rm -rf "$LOCK_DIR"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    LOCK_HELD=1
    printf '%s\n' "$$" > "${LOCK_DIR}/pid"
    printf '%s\n' "$$" > "$PID_FILE"
    return 0
  fi

  log "failed to create QC watcher lock: ${LOCK_DIR}"
  return 1
}

sync_remote_time() {
  if [ "$SYNC_REMOTE_TIME" != 1 ]; then
    TIME_SYNC_OK=1
    return 0
  fi

  local host_epoch remote_epoch skew abs_skew sync_cmd sync_output sync_rc interactive_attempted
  interactive_attempted=0
  host_epoch="$(date +%s)"
  if run_capture ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_TARGET" "date +%s"; then
    remote_epoch="$RUN_CAPTURE_OUTPUT"
  else
    remote_epoch=""
  fi
  if ! [[ "$remote_epoch" =~ ^[0-9]+$ ]]; then
    log "remote time check failed; skip time sync"
    TIME_SYNC_OK=0
    return 1
  fi

  skew=$((remote_epoch - host_epoch))
  abs_skew="${skew#-}"
  if [ "$abs_skew" -le "$MAX_TIME_SKEW_S" ]; then
    TIME_SYNC_OK=1
    return 0
  fi

  if [ "$INTERACTIVE_REMOTE_SUDO" = 1 ] && [ -r /dev/tty ] && [ -w /dev/tty ]; then
    log "remote time skew=${skew}s; enter sudo password for ${REMOTE_TARGET} if prompted"
    interactive_attempted=1
    host_epoch="$(date +%s)"
    sync_cmd="SECONDS=0; sudo -v && sudo timedatectl set-ntp false && adjusted_epoch=\$(( ${host_epoch} + SECONDS )) && sudo date -u -s @\${adjusted_epoch} >/dev/null && (sudo hwclock --systohc 2>/dev/null || true) && sudo timedatectl set-ntp true && date '+%F %T %z'"
    if run_tty_command ssh -tt -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_TARGET" "$sync_cmd"; then
      log "remote time synced with interactive sudo: skew=${skew}s"
      TIME_SYNC_OK=1
      REMOTE_TIME_WAS_CORRECTED=1
      return 0
    fi
  fi

  sync_cmd="sudo -n timedatectl set-ntp false && sudo -n date -u -s @${host_epoch} >/dev/null && (sudo -n hwclock --systohc 2>/dev/null || true) && sudo -n timedatectl set-ntp true && date '+%F %T %z'"
  run_capture ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_TARGET" "$sync_cmd"
  sync_rc=$?
  sync_output="$RUN_CAPTURE_OUTPUT"
  if [ "$sync_rc" -eq 0 ]; then
    log "remote time synced: skew=${skew}s new_time=${sync_output//$'\n'/; }"
    TIME_SYNC_OK=1
    REMOTE_TIME_WAS_CORRECTED=1
    return 0
  fi

  TIME_SYNC_OK=0
  if [ $((host_epoch - LAST_TIME_SYNC_WARN_EPOCH)) -ge "$TIME_SYNC_WARN_INTERVAL_S" ]; then
    log "remote time sync failed rc=${sync_rc}: skew=${skew}s output=${sync_output//$'\n'/; }"
    if [ "$interactive_attempted" = 1 ]; then
      log "remote time sync failed after interactive sudo; passwordless sudo for timedatectl/date also failed on ${REMOTE_TARGET}"
    else
      log "remote time sync requires passwordless sudo for timedatectl/date on ${REMOTE_TARGET}"
    fi
    LAST_TIME_SYNC_WARN_EPOCH="$host_epoch"
  fi
  return "$sync_rc"
}

build_remote_find_cmd() {
  local start="$1"
  local end="$2"
  local now_epoch

  if { [ "$TIME_SYNC_OK" = 1 ] && [ "$REMOTE_TIME_WAS_CORRECTED" != 1 ]; } || [ "$SCAN_ALL_ON_TIME_SYNC_FAIL" != 1 ]; then
    printf "find %s -maxdepth 1 -type f -name 'episode_*.hdf5' -newermt %s ! -newermt %s -printf '%%f\\t%%s\\t%%T@\\n' | sort -V" \
      "$(remote_shell_quote "$REMOTE_DIR")" \
      "$(remote_shell_quote "$start")" \
      "$(remote_shell_quote "$end")"
    return 0
  fi

  now_epoch="$(date +%s)"
  if [ $((now_epoch - LAST_FALLBACK_WARN_EPOCH)) -ge "$TIME_SYNC_WARN_INTERVAL_S" ]; then
    if [ "$REMOTE_TIME_WAS_CORRECTED" = 1 ]; then
      log "remote time was corrected in this watcher; fallback to scanning all remote episode_*.hdf5 files and filtering episode number >= ${HISTORY_FROM}" >&2
    else
      log "remote time is not reliable; fallback to scanning all remote episode_*.hdf5 files and filtering episode number >= ${HISTORY_FROM}" >&2
    fi
    LAST_FALLBACK_WARN_EPOCH="$now_epoch"
  fi
  printf "find %s -maxdepth 1 -type f -name 'episode_*.hdf5' -printf '%%f\\t%%s\\t%%T@\\n' | sort -V" \
    "$(remote_shell_quote "$REMOTE_DIR")"
}

print_qc_status() {
  local status_text
  status_text="$(REPORT_DIR="$REPORT_DIR" HISTORY_FROM="$HISTORY_FROM" "$PYTHON" - <<'PY'
import csv
import json
import os
from pathlib import Path

report_dir = Path(os.environ["REPORT_DIR"])
history_from = int(os.environ.get("HISTORY_FROM", "0"))
summary_path = report_dir / "summary.json"
episodes_path = report_dir / "episodes.csv"

if not summary_path.exists() or not episodes_path.exists():
    print("QC UNKNOWN: report missing")
    raise SystemExit

summary = json.loads(summary_path.read_text())
warnings = summary.get("warnings", {})
warning_parts = []
if isinstance(warnings, dict):
    for key, value in warnings.items():
        if isinstance(value, dict) and value:
            warning_parts.append(f"{key}={','.join(map(str, value.keys()))}")
        elif isinstance(value, list) and value:
            warning_parts.append(f"{key}={','.join(map(str, value))}")
        elif value not in (None, [], {}, "", False):
            warning_parts.append(f"{key}={value}")

with episodes_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

bad_episode_ids = []
episode_parts = []
for row in rows:
    episode_id = row.get("episode_id", "?")
    try:
        episode_num = int(str(episode_id).split("_", 1)[1])
    except Exception:
        episode_num = -1
    if episode_num < history_from:
        continue

    ok = row.get("success") in {"1", "true", "True"}
    warnings_text = row.get("warnings", "")
    error_text = row.get("error", "")
    label = "OK" if ok and not warnings_text and not error_text else "BAD"
    if label != "OK":
        bad_episode_ids.append(episode_id)

    detail = f"{episode_id}:{label}:steps={row.get('n_steps', '?')}"
    if warnings_text:
        detail += f":warnings={warnings_text}"
    if error_text:
        detail += f":error={error_text}"
    episode_parts.append(detail)

n_episodes = summary.get("n_episodes", len(rows))
n_success = summary.get("n_success", "?")
success_rate = summary.get("success_rate", "?")
overall_ok = n_success == n_episodes and not warning_parts and not bad_episode_ids
status = "OK" if overall_ok else "BAD"
warning_text = "none" if not warning_parts else ";".join(warning_parts)

print(f"QC {status}: success={n_success}/{n_episodes} rate={success_rate} warnings={warning_text}")
if episode_parts:
    print("QC episodes: " + "; ".join(episode_parts))
PY
)" || status_text="QC UNKNOWN: failed to read report"

  while IFS= read -r line; do
    log "$line"
  done <<< "$status_text"
}

mkdir -p "$EP_DIR" "$REPORT_DIR"
if [ -n "$LOG_FILE" ]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

trap stop_watcher INT TERM HUP QUIT
trap cleanup EXIT

if ! acquire_lock; then
  exit 1
fi

log "today QC watcher started: day=${DAY} remote=${REMOTE_TARGET}:${REMOTE_DIR} episodes=${EP_DIR} report=${REPORT_DIR} history_from=${HISTORY_FROM} sync_remote_time=${SYNC_REMOTE_TIME}"
print_qc_status

while true; do
  sync_remote_time || true

  start="${DAY} 00:00:00"
  end="$(date -d "${DAY} +1 day" +%F) 00:00:00"
  remote_find_cmd="$(build_remote_find_cmd "$start" "$end")"

  run_capture ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_TARGET" "$remote_find_cmd"
  scan_rc=$?
  scan_output="$RUN_CAPTURE_OUTPUT"
  if [ "$scan_rc" -ne 0 ]; then
    log "remote scan failed rc=${scan_rc}: ${scan_output}"
    interruptible_sleep "$POLL_S"
    continue
  fi

  episodes=()
  if [ -n "$scan_output" ]; then
    mapfile -t episodes <<< "$scan_output"
  fi

  if [ "${#episodes[@]}" -eq 0 ]; then
    log "no ${DAY} episodes found yet"
    print_qc_status
    interruptible_sleep "$POLL_S"
    continue
  fi

  changed=0
  for line in "${episodes[@]}"; do
    if [[ "$line" != episode_*.hdf5$'\t'* ]]; then
      log "skip unrecognized remote line: ${line}"
      continue
    fi

    IFS=$'\t' read -r name _size _mtime <<< "$line"
    episode_num="$(episode_number "$name")"
    if { [ "$TIME_SYNC_OK" != 1 ] || [ "$REMOTE_TIME_WAS_CORRECTED" = 1 ]; } && [ "$SCAN_ALL_ON_TIME_SYNC_FAIL" = 1 ]; then
      if [[ "$episode_num" =~ ^[0-9]+$ ]] && [ "$episode_num" -lt "$HISTORY_FROM" ]; then
        continue
      fi
    fi

    remote_path="${REMOTE_DIR%/}/${name}"
    local_path="${EP_DIR}/${name}"
    stat_cmd="stat -c '%s %Y' $(remote_shell_quote "$remote_path")"

    if run_capture ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_TARGET" "$stat_cmd"; then
      stat1="$RUN_CAPTURE_OUTPUT"
    else
      stat1=""
    fi
    interruptible_sleep "$STABLE_INTERVAL_S"
    if run_capture ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_TARGET" "$stat_cmd"; then
      stat2="$RUN_CAPTURE_OUTPUT"
    else
      stat2=""
    fi

    if [ -z "$stat1" ] || [ "$stat1" != "$stat2" ]; then
      log "skip unstable ${name}: ${stat1} -> ${stat2}"
      continue
    fi

    remote_size="${stat2%% *}"
    remote_mtime="${stat2##* }"
    now_epoch="$(date +%s)"
    age_s=$((now_epoch - remote_mtime))
    if [ "$age_s" -lt "$MIN_MTIME_AGE_S" ]; then
      log "skip fresh ${name}: mtime_age=${age_s}s"
      continue
    fi

    if [ -f "$local_path" ] && [ "$(stat -c %s "$local_path")" = "$remote_size" ]; then
      continue
    fi

    tmp_path="${EP_DIR}/.${name}.tmp.$$"
    rm -f "$tmp_path"
    ACTIVE_TMP_PATH="$tmp_path"
    log "copy ${name} (${remote_size} bytes)"
    if run_command rsync -a --partial --protect-args "${REMOTE_TARGET}:${remote_path}" "$tmp_path"; then
      mv "$tmp_path" "$local_path"
      ACTIVE_TMP_PATH=""
      changed=1
      log "copied ${name}"
    else
      rc=$?
      rm -f "$tmp_path"
      ACTIVE_TMP_PATH=""
      log "rsync failed for ${name}: rc=${rc}"
    fi
  done

  if [ "$changed" = 1 ]; then
    log "run QC for ${EP_DIR}"
    if run_command env MPLCONFIGDIR=/tmp/excavator_mpl "$PYTHON" -m testbed.cli.dataset_qc --dataset-dir "$EP_DIR" --output-dir "$REPORT_DIR"; then
      log "QC finished"
    else
      log "QC command failed"
    fi
  else
    log "scan complete: ${#episodes[@]} ${DAY} episodes, no new stable copy"
  fi
  print_qc_status

  interruptible_sleep "$POLL_S"
done
