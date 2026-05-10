#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${EXCAVATOR_ENV_NAME:-excavator-real-stack}"

cd "${ROOT_DIR}"

missing=0

note() {
  printf '%s\n' "$*"
}

check_cmd() {
  local name="$1"
  local required="${2:-required}"
  if command -v "${name}" >/dev/null 2>&1; then
    note "ok: ${name} -> $(command -v "${name}")"
    return 0
  fi
  if [[ "${required}" == "required" ]]; then
    note "missing: ${name}"
    missing=1
  else
    note "warning: ${name} not found"
  fi
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

note "== command checks =="
check_cmd python3 required
check_cmd cmake required
check_cmd c++ required
check_cmd ip required
check_cmd candump optional
check_cmd cansend optional

note
note "== python package checks =="
if ! run_in_env python -c "import numpy, h5py, cv2, torch, testbed; print('ok: python imports')"; then
  missing=1
fi

note
note "== bridge binary =="
if [[ -x bridge/build/excavator_real_bridge ]]; then
  note "ok: bridge/build/excavator_real_bridge"
else
  note "warning: bridge/build/excavator_real_bridge is not built"
  note "build with: cmake -S bridge -B bridge/build -DCMAKE_PREFIX_PATH=\"\${CONDA_PREFIX:-}\" && cmake --build bridge/build --target excavator_real_bridge"
fi

note
note "== repository smoke command =="
note "safe smoke: scripts/smoke_real_bridge.sh"
note "read-only CAN probe: python scripts/can_probe.py --interface can0 --duration-s 10"

exit "${missing}"
