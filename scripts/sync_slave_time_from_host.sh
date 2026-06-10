#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${EXCAVATOR_SLAVE_HOST:-${EXCAVATOR_SLAVE_SSH_HOST:-slave-jetson}}"
SSH_USER="${EXCAVATOR_SLAVE_USER:-${EXCAVATOR_SLAVE_SSH_USER:-mundane}}"
MAX_SKEW_S="${MAX_TIME_SKEW_S:-5}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-8}"
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_slave_time_from_host.sh [options]

Set the Jetson slave clock from the current host clock over SSH. Run this on
the host before starting scripts/slave_real_stack.sh in the Jetson terminal.

Options:
  --ssh-host HOST        Slave SSH host. Default: slave-jetson
  --ssh-user USER        Slave SSH user. Default: mundane
  --max-skew-s SEC       Skip setting time when skew is within this. Default: 5
  --force                Set slave time even when current skew is small.
  -h, --help             Show this help.
EOF
}

log() {
  printf '[sync-slave-time] %s\n' "$*"
}

die() {
  printf '[sync-slave-time] error: %s\n' "$*" >&2
  exit 1
}

abs_int() {
  local value="$1"
  printf '%s' "${value#-}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-host)
      SSH_HOST="$2"
      shift 2
      ;;
    --ssh-user)
      SSH_USER="$2"
      shift 2
      ;;
    --max-skew-s)
      MAX_SKEW_S="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "${MAX_SKEW_S}" =~ ^[0-9]+$ ]] || die "invalid --max-skew-s: ${MAX_SKEW_S}"

REMOTE_TARGET="${SSH_USER}@${SSH_HOST}"

remote_epoch() {
  ssh -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" \
    "${REMOTE_TARGET}" "date +%s"
}

host_epoch="$(date +%s)"
if ! slave_epoch="$(remote_epoch 2>/dev/null)"; then
  die "failed to read slave time from ${REMOTE_TARGET}"
fi
[[ "${slave_epoch}" =~ ^[0-9]+$ ]] || die "invalid slave epoch: ${slave_epoch}"

skew=$((slave_epoch - host_epoch))
abs_skew="$(abs_int "${skew}")"
log "current skew=${skew}s host=$(date '+%F %T %z') slave_epoch=${slave_epoch}"

if [[ "${FORCE}" != "1" && "${abs_skew}" -le "${MAX_SKEW_S}" ]]; then
  log "slave time is close enough; no update needed"
  exit 0
fi

log "setting ${REMOTE_TARGET} clock from host; enter the Jetson sudo password if prompted"
host_epoch="$(date +%s)"
remote_cmd="SECONDS=0; sudo -v && sudo timedatectl set-ntp false && adjusted_epoch=\$(( ${host_epoch} + SECONDS )) && sudo date -u -s @\${adjusted_epoch} >/dev/null && (sudo hwclock --systohc 2>/dev/null || true) && (sudo timedatectl set-ntp true || true) && date '+%F %T %z %s'"
ssh -tt -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}" \
  "${REMOTE_TARGET}" "${remote_cmd}"

host_epoch="$(date +%s)"
slave_epoch="$(remote_epoch)"
[[ "${slave_epoch}" =~ ^[0-9]+$ ]] || die "invalid slave epoch after sync: ${slave_epoch}"
skew=$((slave_epoch - host_epoch))
abs_skew="$(abs_int "${skew}")"
if [[ "${abs_skew}" -gt "${MAX_SKEW_S}" ]]; then
  die "slave time still differs from host by ${skew}s after sync"
fi

log "verified slave time: skew=${skew}s"
