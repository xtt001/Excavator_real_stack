# Configs

This branch keeps only real-excavator configs.

| Goal | Config | Entry |
|---|---|---|
| Safe teleop recording | `testbed/configs/teleop_real_v1.yaml` | `tb-record-real` |
| Offline ACT training | `testbed/configs/act_real_v1.yaml` | `tb-train` |

## `teleop_real_v1.yaml`

Defines:

- backend mode: `mock`, `noop`, `bridge_mock`, or `bridge_tcp`
- state-reader mode: `mock`, `bridge_mock`, or `bridge_tcp`
- optional JSON/TCP bridge host, port, and timeout
- control rate and mock image size
- joystick/keyboard teleop mapping
- sync and low-latency video metadata
- dataset output directory
- safety guard limits and timeout
- operator/session metadata fields

The default output is `data/real_teleop_v1/`.

For local bridge development, start `tb-bridge-mock-server --port 8765`, then
run `tb-record-real --backend bridge_tcp --state-reader bridge_tcp
--bridge-port 8765`. The same values can also live in `real.bridge` inside the
YAML config.

## `act_real_v1.yaml`

Defines:

- dataset directory and episode shape
- `equipment_model: real_excavator`
- camera list, currently `fpv`
- policy class, currently `ACT`
- low-dimensional inputs, currently `qpos + qvel`
- training schedule, split, AMP, and checkpoint directory

The default checkpoint directory is `runs/ckpts/real_excavation_act_v1/`.

## Data Contract

```text
order  = [swing, boom, stick, bucket]
qpos   = calibrated joint angle, rad
qvel   = joint velocity, rad/s
action = normalized command, [-1, 1]
```

Metadata should include `is_real=true`, `platform=real_excavator`,
`qpos_units=rad`, `qvel_units=rad/s`, and
`action_semantics=normalized_teleop_cmd_v1`.
