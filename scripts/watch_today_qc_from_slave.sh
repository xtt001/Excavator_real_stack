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
  -h, --help                   Show this help.

Environment:
  PYTHON can override the Python executable.
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
log "today QC watcher started: day=${DAY} remote=${REMOTE_TARGET}:${REMOTE_DIR} episodes=${EP_DIR} report=${REPORT_DIR} history_from=${HISTORY_FROM}"
print_qc_status

while true; do
  start="${DAY} 00:00:00"
  end="$(date -d "${DAY} +1 day" +%F) 00:00:00"
  remote_find_cmd="find $(remote_shell_quote "$REMOTE_DIR") -maxdepth 1 -type f -name 'episode_*.hdf5' -newermt $(remote_shell_quote "$start") ! -newermt $(remote_shell_quote "$end") -printf '%f\\t%s\\t%T@\\n' | sort -V"

  scan_output="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_TARGET" "$remote_find_cmd" 2>&1)"
  scan_rc=$?
  if [ "$scan_rc" -ne 0 ]; then
    log "remote scan failed rc=${scan_rc}: ${scan_output}"
    sleep "$POLL_S"
    continue
  fi

  episodes=()
  if [ -n "$scan_output" ]; then
    mapfile -t episodes <<< "$scan_output"
  fi

  if [ "${#episodes[@]}" -eq 0 ]; then
    log "no ${DAY} episodes found yet"
    print_qc_status
    sleep "$POLL_S"
    continue
  fi

  changed=0
  for line in "${episodes[@]}"; do
    if [[ "$line" != episode_*.hdf5$'\t'* ]]; then
      log "skip unrecognized remote line: ${line}"
      continue
    fi

    IFS=$'\t' read -r name _size _mtime <<< "$line"
    remote_path="${REMOTE_DIR%/}/${name}"
    local_path="${EP_DIR}/${name}"
    stat_cmd="stat -c '%s %Y' $(remote_shell_quote "$remote_path")"

    stat1="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_TARGET" "$stat_cmd" 2>/dev/null || true)"
    sleep "$STABLE_INTERVAL_S"
    stat2="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_TARGET" "$stat_cmd" 2>/dev/null || true)"

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
    log "copy ${name} (${remote_size} bytes)"
    if rsync -a --partial --protect-args "${REMOTE_TARGET}:${remote_path}" "$tmp_path"; then
      mv "$tmp_path" "$local_path"
      changed=1
      log "copied ${name}"
    else
      rc=$?
      rm -f "$tmp_path"
      log "rsync failed for ${name}: rc=${rc}"
    fi
  done

  if [ "$changed" = 1 ]; then
    log "run QC for ${EP_DIR}"
    if MPLCONFIGDIR=/tmp/excavator_mpl "$PYTHON" -m testbed.cli.dataset_qc --dataset-dir "$EP_DIR" --output-dir "$REPORT_DIR"; then
      log "QC finished"
    else
      log "QC command failed"
    fi
  else
    log "scan complete: ${#episodes[@]} ${DAY} episodes, no new stable copy"
  fi
  print_qc_status

  sleep "$POLL_S"
done
