#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH_HOST="${EXCAVATOR_SLAVE_HOST:-${EXCAVATOR_SLAVE_SSH_HOST:-slave-jetson}}"
SSH_USER="${EXCAVATOR_SLAVE_USER:-${EXCAVATOR_SLAVE_SSH_USER:-mundane}}"
REMOTE_ROOT="${EXCAVATOR_REMOTE_ROOT:-/media/mundane/D/Excavator_real_stack}"
CONFIG_REL="testbed/testbed/configs/policy_real_gmsl_four_camera_v1.yaml"
BRIDGE_PORT="${EXCAVATOR_CONTROL_PORT:-8766}"
SAMPLES=20
INTERVAL_S=0.15
SSH_RETRIES=5
RETRY_SLEEP_S=2
SSH_CONNECT_TIMEOUT=5
REMOTE_PYTHON="${REMOTE_PYTHON:-python3}"
LOCAL_PYTHON="${PYTHON:-python3}"
START_STACK=1
FORCE_START=0
SETUP_CAN=0
KEEP_RUNNING=0

usage() {
  cat <<'EOF'
Usage:
  scripts/calibrate_home_pose_from_current.sh [options]

Read the current 4-axis qpos from the slave bridge and write it as the home
pose in the configured real runtime YAML locally and on the slave.

Options:
  --ssh-host HOST        Slave host. Default: slave-jetson
  --ssh-user USER        Slave SSH user. Default: mundane
  --remote-root PATH     Slave repo root. Default: /media/mundane/D/Excavator_real_stack
  --config RELPATH       Config path relative to this repo.
                         Default: testbed/testbed/configs/policy_real_gmsl_four_camera_v1.yaml
  --bridge-port PORT     Slave bridge control port. Default: 8766
  --samples N            Number of qpos samples. Default: 20
  --interval-s SEC       Delay between samples. Default: 0.15
  --ssh-retries N        SSH/read retries before failing. Default: 5
  --setup-can            Let slave_real_stack.sh set up CAN before sampling.
  --force-start          Force-stop stale slave listeners before starting bridge.
  --keep-running         Leave bridge/gateway running if this script started them.
  --no-start             Do not start bridge/gateway; only read an existing bridge.
  -h, --help             Show this help.

Before running, manually move the excavator to the desired home pose.
EOF
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
    --remote-root)
      REMOTE_ROOT="$2"
      shift 2
      ;;
    --config)
      CONFIG_REL="$2"
      shift 2
      ;;
    --bridge-port)
      BRIDGE_PORT="$2"
      shift 2
      ;;
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    --interval-s)
      INTERVAL_S="$2"
      shift 2
      ;;
    --ssh-retries)
      SSH_RETRIES="$2"
      shift 2
      ;;
    --setup-can)
      SETUP_CAN=1
      shift
      ;;
    --force-start)
      FORCE_START=1
      shift
      ;;
    --keep-running)
      KEEP_RUNNING=1
      shift
      ;;
    --no-start)
      START_STACK=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${CONFIG_REL}" = /* ]]; then
  printf 'error: --config must be a repo-relative path, got: %s\n' "${CONFIG_REL}" >&2
  exit 1
fi

CONFIG_PATH="${ROOT_DIR}/${CONFIG_REL}"
SSH_TARGET="${SSH_USER}@${SSH_HOST}"
SSH_OPTS=(
  -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT}"
  -o "ServerAliveInterval=2"
  -o "ServerAliveCountMax=3"
)
STARTED_STACK=0
SAMPLE_JSON="$(mktemp)"

q() {
  printf '%q' "$1"
}

remote_run() {
  ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "$@"
}

remote_repo_cmd() {
  local cmd
  cmd="cd $(q "${REMOTE_ROOT}") &&"
  while [[ $# -gt 0 ]]; do
    cmd+=" $(q "$1")"
    shift
  done
  printf '%s' "${cmd}"
}

cleanup() {
  rm -f "${SAMPLE_JSON}"
  if [[ "${STARTED_STACK}" == "1" && "${KEEP_RUNNING}" != "1" ]]; then
    printf '[home-calib] stopping calibration bridge/gateway on slave...\n'
    remote_run "$(remote_repo_cmd ./scripts/slave_real_stack.sh stop)" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

remote_bridge_open() {
  local py
  py="import socket; s=socket.create_connection(('127.0.0.1', ${BRIDGE_PORT}), 1.0); s.close()"
  remote_run "$(q "${REMOTE_PYTHON}") -c $(q "${py}")" >/dev/null 2>&1
}

start_remote_stack_if_needed() {
  if [[ "${START_STACK}" != "1" ]]; then
    return 0
  fi
  if remote_bridge_open; then
    printf '[home-calib] slave bridge already listens on 127.0.0.1:%s; will only sample it.\n' "${BRIDGE_PORT}"
    return 0
  fi

  local args=(
    start
    --no-camera
    --no-receiver
    --skip-usb
    --skip-pip-install
  )
  if [[ "${FORCE_START}" == "1" ]]; then
    args+=(--force)
  fi
  if [[ "${SETUP_CAN}" != "1" ]]; then
    args+=(--skip-can)
  fi

  printf '[home-calib] starting slave bridge/gateway for calibration...\n'
  remote_run "$(remote_repo_cmd ./scripts/slave_real_stack.sh "${args[@]}")"
  STARTED_STACK=1
}

remote_sample_once() {
  ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" \
    "cd $(q "${REMOTE_ROOT}") && PYTHONPATH=testbed $(q "${REMOTE_PYTHON}") - $(q "${BRIDGE_PORT}") $(q "${SAMPLES}") $(q "${INTERVAL_S}")" <<'PY'
import json
import sys
import time

import numpy as np

from testbed.backends.real.bridge_socket import JsonTcpBridgeClient

port = int(sys.argv[1])
sample_count = int(sys.argv[2])
interval_s = float(sys.argv[3])
deadline = time.monotonic() + max(20.0, sample_count * interval_s + 30.0)
client = None
rows = []
last_error = ""


def close_client() -> None:
    global client
    if client is not None:
        try:
            client.force_close()
        except Exception:
            pass
        client = None


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


try:
    for step_id in range(sample_count):
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(f"timed out reading bridge state: {last_error}")
            try:
                if client is None:
                    client = JsonTcpBridgeClient(
                        host="127.0.0.1",
                        port=port,
                        timeout_s=1.0,
                        connect_on_init=True,
                    )
                sample = client.read_state(step_id=step_id)
                payload = dict(sample.joint.payload)
                qpos = np.asarray(payload["qpos"], dtype=np.float64).reshape(-1)[:4]
                qvel = np.asarray(
                    payload.get("qvel", np.zeros(4)),
                    dtype=np.float64,
                ).reshape(-1)[:4]
                if qpos.shape != (4,) or qvel.shape != (4,):
                    raise RuntimeError(f"bad qpos/qvel shape: {qpos.shape}/{qvel.shape}")
                if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
                    raise RuntimeError("non-finite qpos/qvel")
                rows.append(
                    {
                        "timestamp_ns": int(sample.joint.timestamp_ns),
                        "qpos": qpos.tolist(),
                        "qvel": qvel.tolist(),
                        "status": jsonable(payload.get("status")),
                        "health": {
                            str(k): jsonable(v)
                            for k, v in payload.items()
                            if str(k).startswith("imu") or str(k) == "state_loop_tick"
                        },
                    }
                )
                time.sleep(interval_s)
                break
            except Exception as exc:
                last_error = repr(exc)
                close_client()
                time.sleep(1.0)

    qpos_stack = np.asarray([row["qpos"] for row in rows], dtype=np.float64)
    qvel_stack = np.asarray([row["qvel"] for row in rows], dtype=np.float64)
    result = {
        "sample_count": len(rows),
        "qpos_mean": qpos_stack.mean(axis=0).tolist(),
        "qpos_std": qpos_stack.std(axis=0).tolist(),
        "max_abs_qvel": np.max(np.abs(qvel_stack), axis=0).tolist(),
        "first": rows[0],
        "last": rows[-1],
    }
    print(json.dumps(result, sort_keys=True))
finally:
    close_client()
PY
}

sample_current_pose() {
  local attempt
  for attempt in $(seq 1 "${SSH_RETRIES}"); do
    printf '[home-calib] reading current qpos from slave, attempt %s/%s...\n' "${attempt}" "${SSH_RETRIES}" >&2
    if remote_sample_once >"${SAMPLE_JSON}"; then
      return 0
    fi
    sleep "${RETRY_SLEEP_S}"
  done
  return 1
}

update_local_config() {
  "${LOCAL_PYTHON}" - "${CONFIG_PATH}" "${SAMPLE_JSON}" "${ROOT_DIR}/testbed" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
sample_path = Path(sys.argv[2])
testbed_path = Path(sys.argv[3])

data = json.loads(sample_path.read_text(encoding="utf-8"))
pose = [float(x) for x in data["qpos_mean"]]
pose_text = "[" + ", ".join(f"{x:.6f}" for x in pose) + "]"

lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
stack: list[tuple[str, int]] = []
found = {
    "go_home_enabled": False,
    "go_home_pose": False,
    "phase_pose": False,
}
has_phase_labeling = False
updated: list[str] = []

for line in lines:
    stripped = line.strip()
    if stripped and not stripped.startswith("#"):
        indent = len(line) - len(line.lstrip(" "))
        while stack and indent <= stack[-1][1]:
            stack.pop()
        path = [key for key, _indent in stack]
        key_part = stripped.split("#", 1)[0].rstrip()

        if key_part.startswith("enabled:") and path == ["teleop", "recording", "go_home"]:
            line = " " * indent + "enabled: true\n"
            found["go_home_enabled"] = True
        elif key_part.startswith("home_pose_rad:") and path == ["teleop", "recording", "go_home"]:
            line = " " * indent + f"home_pose_rad: {pose_text}\n"
            found["go_home_pose"] = True
        elif key_part.startswith("home_pose_rad:") and path == ["phase_labeling"]:
            line = " " * indent + f"home_pose_rad: {pose_text}\n"
            found["phase_pose"] = True

        if key_part.endswith(":") and ":" not in key_part[:-1]:
            if path == [] and key_part[:-1] == "phase_labeling":
                has_phase_labeling = True
            stack.append((key_part[:-1], indent))

    updated.append(line)

if not has_phase_labeling:
    found["phase_pose"] = True

missing = [name for name, ok in found.items() if not ok]
if missing:
    raise RuntimeError(f"config keys not found: {', '.join(missing)}")

config_path.write_text("".join(updated), encoding="utf-8")

sys.path.insert(0, str(testbed_path))
try:
    import yaml
except Exception as exc:  # pragma: no cover - operator environment guard.
    raise RuntimeError("PyYAML is required to validate the updated config") from exc

from testbed.backends.real.go_home import GoHomeConfig

cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
go_home = GoHomeConfig.from_mapping(cfg["teleop"]["recording"]["go_home"])
if go_home is None:
    raise RuntimeError("updated go_home config is still disabled or invalid")

print(f"[home-calib] home_pose_rad={pose_text}")
print(
    "[home-calib] qpos_std=["
    + ", ".join(f"{float(x):.6f}" for x in data["qpos_std"])
    + "]"
)
print(
    "[home-calib] max_abs_qvel=["
    + ", ".join(f"{float(x):.6f}" for x in data["max_abs_qvel"])
    + "]"
)
PY
}

sync_config_to_slave() {
  printf '[home-calib] syncing updated config to slave...\n'
  tar -C "${ROOT_DIR}" -cf - "${CONFIG_REL}" \
    | ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "cd $(q "${REMOTE_ROOT}") && tar -xf -"
}

main() {
  [[ -f "${CONFIG_PATH}" ]] || {
    printf 'error: config not found: %s\n' "${CONFIG_PATH}" >&2
    exit 1
  }
  printf '[home-calib] target slave: %s\n' "${SSH_TARGET}"
  start_remote_stack_if_needed
  sample_current_pose
  update_local_config
  sync_config_to_slave
  printf '[home-calib] done.\n'
}

main "$@"
