# Bridge

This directory is reserved for the real-machine bridge process.

The bridge should connect:

```text
testbed RealBackend / RealBridgeClient
        |
        v
bridge process
        |
        v
control C++ library, ROS nodes, CAN devices, camera/state streams
```

Expected responsibilities:

- Receive normalized 4D testbed commands: `[swing, boom, stick, bucket]`.
- Map commands into the lower control stack's 8-axis command contract.
- Return command acknowledgement, fault code, controller timestamp, and raw
  lower-level command diagnostics.
- Publish or return timestamped joint state, status, and camera samples.
- Keep ROS/CAN imports and hardware access out of the Python testbed import
  path until the bridge is explicitly launched.

For local development without hardware, use the mock bridge already available
inside `testbed/testbed/backends/real/`.
