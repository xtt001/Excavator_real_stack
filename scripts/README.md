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

## Axis Response Calibration

Before increasing go-home speed or timeout, measure the hydraulic dead zone and
latency one axis at a time. Run this on the slave while the bridge is already
running:

```bash
python3 scripts/calibrate_axis_response.py \
  --host 127.0.0.1 \
  --port 8766 \
  --axis boom \
  --direction both \
  --amplitudes 0.03,0.05,0.07,0.10,0.12 \
  --duration-s 0.45 \
  --settle-s 0.80 \
  --abort-delta-rad 0.05 \
  --confirm-hardware-motion
```

The script writes JSONL trials under `artifacts/axis_response/` and prints the
first responsive command per direction. Use those measurements to tune
go-home `min_action`, `max_action`, and `p_gain`; do not tune them from timeout
alone.

Go-home completion uses `success_tolerance_rad` plus stable velocity and
`dwell_s`; `center_tolerance_rad` remains the tighter control target for fine
positioning. If a joint consistently stalls just outside center, widen only that
joint's success tolerance and keep the center band as the preferred target.

## GMSL Camera Bring-Up

The GMSL bring-up script defaults to the four-camera field set:

```bash
scripts/bring_up_gmsl_cameras.sh
```

By default this configures `/dev/video4`, `/dev/video5`, `/dev/video6`, and
`/dev/video7`. To override the GMSL video ids explicitly, set
`GMSL_VIDEO_DEVICES`. Valid ids are `0` through `7`; the script rejects anything
outside that range.

```bash
GMSL_VIDEO_DEVICES="4 5 6 7" scripts/bring_up_gmsl_cameras.sh
```

To check the resolved configuration without loading modules or touching
devices:

```bash
GMSL_PRINT_CONFIG_ONLY=1 scripts/bring_up_gmsl_cameras.sh
```

## Data Collection Wrapper

On the slave, use the managed stack script for the current real-machine
host/slave collection path:

```bash
scripts/slave_real_stack.sh run
```

`run` starts the real CAN bridge, Orbbec camera, FPV SHM subscriber, gateway,
and remote receiver, then follows all logs in the foreground. Press `Ctrl+C` in
that terminal to stop the managed services in the safe order.
The receiver starts from the already installed local environment by default; add
`--install-python-package` only when the editable testbed package must be
reinstalled on the slave.

Useful variants:

```bash
scripts/slave_real_stack.sh start
scripts/slave_real_stack.sh status
scripts/slave_real_stack.sh tail receiver
scripts/slave_real_stack.sh stop
scripts/slave_real_stack.sh restart --force
```

Compatibility aliases remain available: `tail recorder` and `--no-recorder`
map to the receiver service.

The CAN monitor is intentionally separate so the operator can open a dedicated
`candump` terminal while the managed stack keeps running.

## Host-Side Live QC

When episodes are written on the slave disk, run live QC from the host instead
of the Jetson:

```bash
tb-dataset-qc-watch-ssh \
  --ssh-host "${EXCAVATOR_SLAVE_SSH_HOST:-slave-jetson}" \
  --ssh-user "${EXCAVATOR_SLAVE_SSH_USER:-mundane}" \
  --remote-dir "${EXCAVATOR_SLAVE_DATASET_DIR:-/media/mundane/EXTERNAL_USB/real_teleop_v1}" \
  --cache-dir data/qc_cache
```

The watcher uses SSH for directory/stat checks and copies each completed HDF5
episode to a host cache before running QC locally. It does not require sshfs
and does not run Python QC on the slave.
