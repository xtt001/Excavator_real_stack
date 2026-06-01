# Configs

This branch keeps only real-excavator configs.

| Goal | Config | Entry |
|---|---|---|
| Safe teleop receiver/record | `testbed/configs/teleop_real_v1.yaml` | `tb-receiver-real` |
| One-dig policy shadow/control | `testbed/configs/policy_real_one_dig_v1.yaml` | `tb-receiver-real --input policy` |
| Offline ACT training | `testbed/configs/act_real_v1.yaml` | `tb-train` |

## `teleop_real_v1.yaml`

Defines:

- backend mode: `mock`, `noop`, `bridge_mock`, or `bridge_tcp`
- state-reader mode: `mock`, `bridge_mock`, or `bridge_tcp`
- optional JSON/TCP bridge host, port, and timeout
- control rate and mock image size
- joystick/keyboard teleop mapping
- optional `teleop.recording.go_home` near-home feedback settings; keep
  `enabled: false` until `home_pose_rad` is field-calibrated
- optional `phase_labeling` ranges for offline coarse phase labels
- sync and low-latency video metadata
- receiver health gate defaults and failed-record quarantine
- dataset output directory
- safety guard limits and timeout
- operator/session metadata fields

The default output is `data/real_teleop_v1/`.

For local bridge development, start `tb-bridge-mock-server --port 8765`, then
run `tb-receiver-real --backend bridge_tcp --state-reader bridge_tcp
--bridge-port 8765`. The same values can also live in `real.bridge` inside the
YAML config.

`tb-record-real` is kept as a compatibility alias for the same receiver logic.

`tb-dataset-qc-watch-ssh` is the preferred field watcher when HDF5 is written
on the slave. It lists and copies completed `episode_*.hdf5` files over SSH,
then runs QC on the host-side cache so the Jetson does not spend CPU on QC.

## `policy_real_one_dig_v1.yaml`

Defines a policy-backed receiver for the real one-dig checkpoint. The expected
local bundle is `policy_bundles/real_one_dig_v1/` with:

- `policy_best.ckpt`
- `dataset_stats.pkl`
- `resolved_config.yaml`
- optional `run_metadata.json`

The default `teleop.recording.enabled` is `false`, so policy tests do not create
HDF5 training episodes and do not depend on the record-start button. Lightweight
test logs are written under `teleop.test_log.output_dir`.

The default `teleop.policy.output_mode` is `shadow_zero`, so the model output is
written to JSONL logs while the command sent to the backend remains zero. After
shadow checks pass, use `output_mode: control` for the full one-dig test window.
Normal data collection remains on `teleop_real_v1.yaml`; use
`--record` only when you intentionally want HDF5 sessions from this config.

Local dry run:

```bash
tb-receiver-real \
  --config testbed/testbed/configs/policy_real_one_dig_v1.yaml \
  --backend mock \
  --state-reader mock \
  --input policy \
  --num-episodes 1 \
  --max-steps 5 \
  --test-log-dir /tmp/real_one_dig_policy_shadow_smoke
```

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
