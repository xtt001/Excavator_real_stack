# Real-Machine Configs

This directory should hold deployment-level configuration that spans both
`testbed/` and `control/`.

Planned config groups:

- CAN interface names and enable flags.
- Axis order, sign, scale, and limits.
- Safety limits, heartbeat timeouts, and fault behavior.
- ROS topic names or JSON/TCP bridge host/port.
- Timestamp policy for joint state, camera frames, command sample, command
  send, and controller acknowledgement.
- Camera transport settings for low-latency teleoperation.

Keep experiment/training configs under `testbed/testbed/configs/`. Keep
low-level PID and machine-control configs under `control/config/` unless they
need to be shared by the full deployment.
