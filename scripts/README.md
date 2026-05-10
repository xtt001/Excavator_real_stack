# Scripts

This directory holds full-stack bring-up and deployment helpers.

## Environment Setup

Create or update the recommended environment from repository-owned files:

```bash
scripts/setup_env.sh
```

By default the script uses conda when available and installs:

- Python package dependencies from `requirements.txt` / `testbed/pyproject.toml`.
- CMake and Eigen from `environment.yml`, so the C++ bridge can build.

Useful variants:

```bash
scripts/setup_env.sh conda
scripts/setup_env.sh venv
EXCAVATOR_ENV_NAME=excavator-real-stack scripts/setup_env.sh conda
```

The `venv` mode installs Python packages only; Eigen3 still needs a system or
conda install before building `bridge/excavator_real_bridge`.

Target-machine prerequisite check:

```bash
scripts/check_target_prereqs.sh
```

This does not touch hardware. It checks common commands, Python imports, CAN
tool availability, and whether the bridge binary has already been built.

## Safe Bridge Smoke Test

Copy `.env.example` to `.env` if you need to adjust host, port, CAN interface
names, image size, or heartbeat timing. The checked-in defaults keep real CAN
writes disabled.

```bash
cp .env.example .env
scripts/smoke_real_bridge.sh
```

The smoke script:

- Builds `bridge/excavator_real_bridge` when needed.
- Starts the bridge with `can_bus_enabled=false`, `can_simulation=true`, and
  `imu_simulation=true` unless overridden.
- Records a short `bridge_tcp` episode with zero input.
- Runs `tb-dataset-qc --profile real`.
- Sends invalid JSON, missing action, wrong-dimension action, and a valid action
  followed by a watchdog timeout check.
- Shuts the bridge down and prints the dataset/QC/log paths.

It refuses to run with `EXCAVATOR_CAN_BUS_ENABLED=true` unless
`EXCAVATOR_ALLOW_REAL_CAN_SMOKE=1` is set explicitly for supervised hardware
bring-up.

## Target-Machine CAN Probe

After the CAN adapter is connected but before sending any control command, run a
read-only probe:

```bash
python scripts/can_probe.py \
  --interface "${EXCAVATOR_CAN_IF:-can0}" \
  --duration-s "${EXCAVATOR_CAN_PROBE_DURATION_S:-10}" \
  --ids 18F021F6 18F022F6 18F023F6 \
  --output-dir artifacts/can_probe
```

The probe uses `candump` only. It writes raw logs, per-ID logs, `ip link`
details, and a `summary.json`; it does not transmit CAN frames.

## One-Axis Bring-Up Client

When the bridge is already running under supervised hardware conditions, use the
one-axis client to send a tiny command and automatic zeros:

```bash
python scripts/one_axis_bringup.py \
  --host "${EXCAVATOR_BRIDGE_HOST:-127.0.0.1}" \
  --port "${EXCAVATOR_BRIDGE_PORT:-9876}" \
  --axis swing \
  --amplitude 0.03 \
  --duration-s 0.5 \
  --confirm-hardware-motion
```

The confirmation flag is required for non-zero commands even in simulation, so
the same command line is deliberate when moved to hardware.

## Data Collection Wrapper

There is intentionally no single "start real collection" script yet. Data
collection should wait until `docs/real_machine_bringup_checklist.md` is
complete for the target machine, real camera frames have replaced the
placeholder `fpv`, and a short real episode passes QC.
