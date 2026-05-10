# Real-Machine Bring-Up Checklist

Use this checklist before the first hardware-connected run.

## Software

- Verify `testbed/` can run mock and bridge_mock recording.
- Verify the bridge process starts without importing ROS/CAN from testbed.
- Verify `control/` builds on the target machine.
- Verify CAN interface names and simulation/enable flags are explicit.

## Contract

- Confirm action order: `[swing, boom, stick, bucket]`.
- Confirm lower command order:
  `[swing, boom, stick, bucket, left_track, right_track, boom_offset, chassis_dozer]`.
- Confirm qpos/qvel units, zero points, signs, and limits.
- Confirm status bits for deadman, e-stop, remote mode, pilot, stale sensor,
  and manual override.

## Timing

- Use source timestamps for joint state and camera frames.
- Record command sample time, command send time, controller acknowledgement
  time, joint timestamp, and image timestamp.
- Check observation skew before using data for training.

## Hardware Safety

- Start with CAN bus disabled or control simulation enabled.
- Test one axis at low command scale.
- Keep operator e-stop and manual override active and verified.
- Use heartbeat timeout to force zero command on bridge failure.
- Do not run autonomous ACT control until offline prediction and shadow checks
  are complete.
