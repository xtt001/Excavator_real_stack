#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${EXCAVATOR_ENV_NAME:-excavator-real-stack}"
MODE="${1:-auto}"

usage() {
  cat <<'EOF'
Usage: scripts/setup_env.sh [auto|conda|venv]

Creates or updates the development/runtime environment from repo-owned files.

Modes:
  auto   Prefer conda when available, otherwise use .venv
  conda  Create/update conda env from environment.yml
  venv   Create/update .venv with pip requirements.txt

Optional:
  EXCAVATOR_ENV_NAME=<name>  Override conda env name
EOF
}

if [[ "${MODE}" == "--help" || "${MODE}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${MODE}" != "auto" && "${MODE}" != "conda" && "${MODE}" != "venv" ]]; then
  usage >&2
  exit 2
fi

cd "${ROOT_DIR}"

setup_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found; use scripts/setup_env.sh venv or install Miniconda." >&2
    exit 1
  fi

  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Updating conda env: ${ENV_NAME}"
    conda env update -n "${ENV_NAME}" -f environment.yml --prune
  else
    echo "Creating conda env: ${ENV_NAME}"
    conda env create -n "${ENV_NAME}" -f environment.yml
  fi

  echo
  echo "Environment ready."
  echo "Activate with: conda activate ${ENV_NAME}"
  echo "Build bridge with: cmake -S bridge -B bridge/build -DCMAKE_PREFIX_PATH=\"\${CONDA_PREFIX}\" && cmake --build bridge/build --target excavator_real_bridge"
}

setup_venv() {
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

  echo
  echo "Python venv ready at ${ROOT_DIR}/.venv"
  echo "Activate with: source .venv/bin/activate"
  echo
  if ! pkg-config --exists eigen3 2>/dev/null && [[ ! -d /usr/include/eigen3 ]]; then
    echo "Note: .venv does not install C++ Eigen headers."
    echo "Install Eigen3 separately for bridge builds, for example:"
    echo "  sudo apt-get install -y libeigen3-dev"
    echo "or use: scripts/setup_env.sh conda"
  fi
}

if [[ "${MODE}" == "conda" ]]; then
  setup_conda
elif [[ "${MODE}" == "venv" ]]; then
  setup_venv
elif command -v conda >/dev/null 2>&1; then
  setup_conda
else
  setup_venv
fi
