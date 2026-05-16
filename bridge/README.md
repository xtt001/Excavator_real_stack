# Bridge

This directory contains the first real-machine JSON/TCP bridge process.

The bridge connects:

```text
testbed RealBackend / RealBridgeClient
        |
        v
excavator_real_bridge
        |
        v
control C++ library, ROS nodes, CAN devices, camera/state streams
```

## Current V1

`excavator_real_bridge` is a C++ newline-delimited JSON server that speaks the
existing `testbed` `bridge_tcp` protocol:

```text
send_action.request  -> send_action.response
send_status.request  -> send_status.response  (toggle_mask -> applyStatusToggleMask)
read_state.request   -> read_state.response
reset.request        -> reset.response
close.request        -> close.response
shutdown.request     -> shutdown.response
```

It receives normalized 4D commands `[swing, boom, stick, bucket]`, maps them to
the lower 8-axis `SpeedScalarCmd`, returns controller acknowledgement/fault
diagnostics, and returns timestamped joint/status samples plus an internal RGB
placeholder `fpv` image.

This first bridge does not implement ROS, real camera transport, or OEM remote
decoding. The placeholder image exists only so recorder/QC smoke tests can run
before the real low-latency video path is connected.

## Build

Recommended environment setup from the repository root:

```bash
scripts/setup_env.sh
conda activate excavator-real-stack
```

Alternatively, install Eigen3 on the target machine:

```bash
sudo apt-get update
sudo apt-get install -y libeigen3-dev
```

Then build:

```bash
cmake -S bridge -B bridge/build -DCMAKE_PREFIX_PATH="${CONDA_PREFIX:-}"
cmake --build bridge/build --target excavator_real_bridge
```

## Safe Smoke Test

The bridge defaults to simulation and real CAN disabled. Keep those defaults
for first smoke tests. The recommended path is the repository smoke script:

```bash
cp .env.example .env
scripts/smoke_real_bridge.sh
```

The script builds the bridge, starts it with safe defaults, records a short
`bridge_tcp` episode, runs QC, checks protocol errors, verifies watchdog logging,
and shuts the bridge down.

For manual debugging, start the bridge directly:

```bash
./bridge/build/excavator_real_bridge \
  --host 127.0.0.1 \
  --port 8765 \
  --can-bus-enabled false \
  --can-simulation true \
  --imu-simulation true
```

In another shell:

```bash
tb-record-real \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host 127.0.0.1 \
  --bridge-port 8765 \
  --input zero \
  --num-episodes 1 \
  --max-steps 3

tb-dataset-qc --dataset-dir data/real_teleop_v1 --profile real
```

Only enable real CAN with explicit operator safety checks, verified interface
names, and one-axis low-speed testing.

## Hardware Bring-Up Hooks

Before starting the bridge with real CAN writes enabled, confirm the target bus
with the read-only probe:

```bash
python scripts/can_probe.py \
  --interface can0 \
  --duration-s 10 \
  --ids 18F021F6 18F022F6 18F023F6 \
  --output-dir artifacts/can_probe
```

After the bus, e-stop, manual override, pilot/remote mode, and operator
supervision are confirmed, start the bridge with explicit real-CAN flags and
test one axis at a time:

```bash
python scripts/one_axis_bringup.py \
  --host 127.0.0.1 \
  --port 9876 \
  --axis swing \
  --amplitude 0.03 \
  --duration-s 0.5 \
  --confirm-hardware-motion
```

The first training dataset must not use the built-in placeholder image. Replace
the `fpv` sample with a real low-latency camera path and source timestamps
before collecting data for ACT.
