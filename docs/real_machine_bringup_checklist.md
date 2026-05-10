# Real-Machine Bring-Up Checklist

Use this checklist on the target machine before any data collection. The current
software baseline already supports environment setup, C++ bridge build, safe
simulation smoke tests, read-only CAN probing, and a supervised one-axis command
client. The remaining work is hardware validation.

## 0. Software Baseline Before Hardware

- Create the environment from repository-owned files:

  ```bash
  scripts/setup_env.sh
  conda activate excavator-real-stack
  ```

- Run the target prerequisite check:

  ```bash
  scripts/check_target_prereqs.sh
  ```

- Build the C++ bridge:

  ```bash
  cmake -S bridge -B bridge/build -DCMAKE_PREFIX_PATH="${CONDA_PREFIX:-}"
  cmake --build bridge/build --target excavator_real_bridge
  ```

- Run the safe smoke test with real CAN disabled:

  ```bash
  cp .env.example .env
  scripts/smoke_real_bridge.sh
  ```

- Confirm the smoke script passes recorder, QC, protocol-error handling,
  watchdog zero command, and shutdown.
- Keep `.env` defaults safe until the supervised hardware phase:
  `EXCAVATOR_CAN_BUS_ENABLED=false`, `EXCAVATOR_CAN_SIMULATION=true`,
  `EXCAVATOR_IMU_SIMULATION=true`.

## 1. Target-Machine CAN Readiness

- Install target-machine system tools: `iproute2`, `can-utils`, CMake, compiler,
  and either conda Eigen or `libeigen3-dev`.
- Verify interface names with `ip -details link show`; record whether the
  excavator bus is `can0`, `can1`, or another name.
- Configure bitrate only when the adapter, machine bus, and required bitrate are
  confirmed. The legacy helper is:

  ```bash
  control/setup/setup_can.sh can0 250000
  ```

- Run read-only CAN probing before any command transmission:

  ```bash
  python scripts/can_probe.py \
    --interface can0 \
    --duration-s 10 \
    --ids 18F021F6 18F022F6 18F023F6 \
    --output-dir artifacts/can_probe
  ```

- Archive `artifacts/can_probe/summary.json`, `candump.raw.log`, and per-ID logs.
- Do not continue to command tests until the observed CAN IDs, bitrate, and bus
  health are understood.

## 2. Contract Checks To Confirm On Hardware

- Action order at the testbed boundary:
  `[swing, boom, stick, bucket]`.
- Lower command order inside the C++ control API:
  `[swing, boom, stick, bucket, left_track, right_track, boom_offset, chassis_dozer]`.
- CAN frame IDs currently implemented:
  `18F021F6` at 10 Hz, `18F022F6` at 50 Hz, `18F023F6` at 50 Hz.
- Confirm for each axis:
  zero command, positive direction, negative direction, physical joint name,
  sign convention, low-speed response, and stop behavior.
- Confirm qpos/qvel units, zero points, sign, limits, and whether IMU-derived
  states match the training contract.
- Confirm status semantics for deadman, e-stop, remote mode, pilot, stale
  sensor, manual override, and any OEM safety interlock.

## 3. Supervised One-Axis Motion Test

Only start this phase with an operator at the machine, e-stop ready, manual
override verified, people clear of the work envelope, and the machine in a low
energy state.

- Start the bridge with explicit hardware flags only after the previous phases
  pass. Example shape:

  ```bash
  ./bridge/build/excavator_real_bridge \
    --host 127.0.0.1 \
    --port 9876 \
    --can-if can0 \
    --imu-if can1 \
    --can-simulation false \
    --imu-simulation false \
    --can-bus-enabled true \
    --heartbeat-timeout-ms 200
  ```

- Send one tiny command on one axis and automatic zeros:

  ```bash
  python scripts/one_axis_bringup.py \
    --host 127.0.0.1 \
    --port 9876 \
    --axis swing \
    --amplitude 0.03 \
    --duration-s 0.5 \
    --confirm-hardware-motion
  ```

- Repeat one axis at a time: `swing`, `boom`, `stick`, `bucket`.
- Stop immediately if the wrong axis moves, direction is reversed, motion is
  jerky, status bits disagree, watchdog does not zero, or manual override/e-stop
  does not behave as expected.
- Update `.env`, docs, and any axis mapping only after the physical result is
  confirmed.

## 4. Data Collection Readiness

Do not collect training data until all items below are true:

- The one-axis tests have confirmed all four first-stage axes.
- The bridge returns real qpos/qvel/status with validated units and timestamps.
- The real `fpv` camera path is connected; the C++ placeholder image is no
  longer used for training data.
- Camera frames carry source timestamps and use a low-latency/latest-frame path.
- HDF5 records include raw action, guarded action, command send time,
  controller acknowledgement time, joint timestamp, image timestamp,
  `sync_max_skew_ns`, status, and fault diagnostics.
- The chosen operator input is available:
  joystick/keyboard for first supervised collection, or OEM remote reader if
  demonstrations must use the manufacturer remote.
- `tb-dataset-qc --profile real` passes on a short real episode.
- A human review confirms images, qpos/qvel, action signs, and status/fault
  fields match the real motion.

## 5. First Real Data Collection

- Record short episodes first: low speed, simple motions, one operator, clear
  notes, and no autonomous policy.
- Use `tb-record-real` with `--backend bridge_tcp --state-reader bridge_tcp`
  against the validated bridge.
- Run QC immediately after each batch and keep bad episodes separate instead of
  mixing them into the training set.
- Only move to ACT training after metadata, diagnostics, timestamps, video, and
  action semantics are stable.
- Do not run closed-loop ACT/autonomous commands on the real machine until
  offline evaluation and shadow prediction have been reviewed.
