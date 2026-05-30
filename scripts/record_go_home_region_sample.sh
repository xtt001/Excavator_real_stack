#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH_HOST="${EXCAVATOR_SLAVE_HOST:-${EXCAVATOR_SLAVE_SSH_HOST:-slave-jetson}}"
SSH_USER="${EXCAVATOR_SLAVE_USER:-${EXCAVATOR_SLAVE_SSH_USER:-mundane}}"
REMOTE_ROOT="${EXCAVATOR_REMOTE_ROOT:-/media/mundane/D/Excavator_real_stack}"
CONFIG_REL="testbed/testbed/configs/teleop_real_v1.yaml"
BRIDGE_PORT="${EXCAVATOR_CONTROL_PORT:-8766}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python3}"
LOCAL_PYTHON="${PYTHON:-python3}"
OUTPUT_REL="artifacts/go_home_region_samples/go_home_region_samples.jsonl"
LABEL=""
NOTE=""
SAMPLES=20
INTERVAL_S=0.10
SSH_CONNECT_TIMEOUT=5
START_STACK=1
FORCE_START=0
SETUP_CAN=0
KEEP_RUNNING=0

usage() {
  cat <<'EOF'
Usage:
  scripts/record_go_home_region_sample.sh --label LABEL [options]

Read the current 4-axis qpos from the slave bridge and append a labeled sample
for calibrating the go-home start region. This script only reads state; it does
not send motion commands.

Recommended labels:
  home, dig_above, dump_above, return_near_home, unsafe_too_far

Options:
  --label LABEL         Required sample label.
  --note TEXT           Optional free-form note for this sample.
  --output RELPATH      JSONL output path relative to this repo.
                        Default: artifacts/go_home_region_samples/go_home_region_samples.jsonl
  --ssh-host HOST       Slave host. Default: slave-jetson
  --ssh-user USER       Slave SSH user. Default: mundane
  --remote-root PATH    Slave repo root. Default: /media/mundane/D/Excavator_real_stack
  --config RELPATH      Config path relative to this repo.
  --bridge-port PORT    Slave bridge control port. Default: 8766
  --samples N           Number of qpos samples. Default: 20
  --interval-s SEC      Delay between samples. Default: 0.10
  --setup-can           Let slave_real_stack.sh set up CAN before sampling.
  --force-start         Force-stop stale slave listeners before starting bridge.
  --keep-running        Leave bridge/gateway running if this script started them.
  --no-start            Do not start bridge/gateway; only read an existing bridge.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)
      LABEL="$2"
      shift 2
      ;;
    --note)
      NOTE="$2"
      shift 2
      ;;
    --output)
      OUTPUT_REL="$2"
      shift 2
      ;;
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

if [[ -z "${LABEL}" ]]; then
  printf 'error: --label is required\n' >&2
  usage >&2
  exit 1
fi
if [[ "${OUTPUT_REL}" = /* || "${CONFIG_REL}" = /* ]]; then
  printf 'error: --output and --config must be repo-relative paths\n' >&2
  exit 1
fi

OUTPUT_PATH="${ROOT_DIR}/${OUTPUT_REL}"
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
    printf '[region-sample] stopping temporary bridge/gateway on slave...\n'
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
    printf '[region-sample] slave bridge already listens on 127.0.0.1:%s; will only sample it.\n' "${BRIDGE_PORT}"
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

  printf '[region-sample] starting slave bridge/gateway for read-only sampling...\n'
  remote_run "$(remote_repo_cmd ./scripts/slave_real_stack.sh "${args[@]}")"
  STARTED_STACK=1
}

sample_current_pose() {
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
    print(
        json.dumps(
            {
                "sample_count": len(rows),
                "qpos_mean": qpos_stack.mean(axis=0).tolist(),
                "qpos_std": qpos_stack.std(axis=0).tolist(),
                "max_abs_qvel": np.max(np.abs(qvel_stack), axis=0).tolist(),
                "samples": rows,
            },
            sort_keys=True,
        )
    )
finally:
    close_client()
PY
}

append_and_summarize() {
  mkdir -p "$(dirname "${OUTPUT_PATH}")"
  "${LOCAL_PYTHON}" - \
    "${OUTPUT_PATH}" \
    "${SAMPLE_JSON}" \
    "${CONFIG_PATH}" \
    "${LABEL}" \
    "${NOTE}" \
    "${SSH_TARGET}" <<'PY'
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

output_path = Path(sys.argv[1])
sample_path = Path(sys.argv[2])
config_path = Path(sys.argv[3])
label = sys.argv[4]
note = sys.argv[5]
ssh_target = sys.argv[6]

sample = json.loads(sample_path.read_text(encoding="utf-8"))
record = {
    "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    "label": label,
    "note": note,
    "ssh_target": ssh_target,
    **sample,
}
with output_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

records = []
with output_path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

home_pose = None
try:
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    home_pose = np.asarray(
        cfg["teleop"]["recording"]["go_home"]["home_pose_rad"],
        dtype=np.float64,
    ).reshape(4)
except Exception:
    home_pose = None

print(f"[region-sample] appended label={label!r} output={output_path}")
print(
    "[region-sample] qpos_mean=["
    + ", ".join(f"{float(x):+.6f}" for x in record["qpos_mean"])
    + "]"
)
print(
    "[region-sample] qpos_std=["
    + ", ".join(f"{float(x):.6f}" for x in record["qpos_std"])
    + "]"
)
print(
    "[region-sample] max_abs_qvel=["
    + ", ".join(f"{float(x):.6f}" for x in record["max_abs_qvel"])
    + "]"
)

labels = sorted({str(r["label"]) for r in records})
print("[region-sample] samples by label:")
for item_label in labels:
    group = [r for r in records if str(r["label"]) == item_label]
    qpos = np.asarray([r["qpos_mean"] for r in group], dtype=np.float64)
    print(
        f"  {item_label}: n={len(group)} "
        "min=["
        + ", ".join(f"{float(x):+.3f}" for x in qpos.min(axis=0))
        + "] max=["
        + ", ".join(f"{float(x):+.3f}" for x in qpos.max(axis=0))
        + "]"
    )

if home_pose is not None:
    allowed = [
        r
        for r in records
        if str(r["label"]) not in {"unsafe", "unsafe_too_far", "reject"}
    ]
    if allowed:
        qpos = np.asarray([r["qpos_mean"] for r in allowed], dtype=np.float64)
        max_error = np.max(np.abs(qpos - home_pose.reshape(1, 4)), axis=0)
        suggested = max_error + 0.05
        print(
            "[region-sample] max_abs_error_from_config_home=["
            + ", ".join(f"{float(x):.3f}" for x in max_error)
            + "]"
        )
        print(
            "[region-sample] suggested_near_tolerance_rad_with_0.05_margin=["
            + ", ".join(f"{float(x):.3f}" for x in suggested)
            + "]"
        )
PY
}

main() {
  [[ -f "${CONFIG_PATH}" ]] || {
    printf 'error: config not found: %s\n' "${CONFIG_PATH}" >&2
    exit 1
  }
  printf '[region-sample] target slave: %s\n' "${SSH_TARGET}"
  printf '[region-sample] label: %s\n' "${LABEL}"
  start_remote_stack_if_needed
  sample_current_pose >"${SAMPLE_JSON}"
  append_and_summarize
}

main "$@"
