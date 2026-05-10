# Excavator Real Stack

Integrated real-machine stack for excavator imitation-learning data collection,
low-level control, and future ROS/CAN bridge deployment.

This repository combines the two previously separate codebases into one
deployment-oriented monorepo while keeping clear internal boundaries:

```text
Excavator_real_stack/
  control/   C++ low-level control, CAN, PID, machine status, safety plumbing
  testbed/   Python data collection, RealBackend, HDF5, ACT training, QC
  bridge/    Future ROS/CAN bridge process connecting testbed and control
  configs/   Machine deployment configs, axis signs, limits, topics, CAN names
  scripts/   Bring-up, smoke checks, and deployment helpers
  docs/      Integration notes and real-machine checklists
```

## Boundary

The stack should be deployed together on the real-machine development system,
but modules should not collapse into one layer:

- `testbed/` owns operator input, action guards, observations, recording,
  dataset QC, offline ACT training, and bridge-facing contracts.
- `control/` owns CAN, PID/control logic, hydraulic/motor mapping, hardware
  status, and safety-critical machine details.
- `bridge/` owns the runtime connection between both sides. It should translate
  the testbed command/state protocol to ROS/CAN/control-library calls.

In the first hardware bring-up phase, keep using mock/noop/bridge_mock paths in
`testbed/` until the real bridge process is ready.

## Current Source Snapshot

- `testbed/` imported from `excavator_testbed`
  branch `tx/v1-baseline-realworld`, commit `2709f2f`.
- `control/` imported from `excavator`
  branch `main`, commit `1ab8eba`.

## Immediate Next Steps

1. Implement a bridge process under `bridge/` that can call `control/` and
   speak the JSON/TCP or ROS-facing contract expected by `testbed/`.
2. Add real-machine configs under `configs/` for CAN interface names, axis
   signs, limits, topic names, and timestamp policy.
3. Validate command/state mapping with mock bridge first, then with CAN
   disabled, then with one low-speed axis on hardware.
4. Keep camera transport low-latency and record source timestamps for camera,
   joint state, command sample, command send, and controller acknowledgement.

## Useful Commands

From `testbed/`:

```bash
tb-record-real --config testbed/configs/teleop_real_v1.yaml --backend bridge_mock --state-reader bridge_mock --input zero --num-episodes 1
tb-bridge-mock-server --port 8765
tb-record-real --config testbed/configs/teleop_real_v1.yaml --backend bridge_tcp --state-reader bridge_tcp --bridge-port 8765
```

From `control/`, use the existing CMake flow and hardware setup docs/scripts
after the target machine has the required CAN environment.
