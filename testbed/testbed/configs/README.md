# Configs

This branch keeps only real-excavator configs.

| Goal | Config | Entry |
|---|---|---|
| Safe teleop receiver/record | `testbed/configs/teleop_real_v1.yaml` | `tb-receiver-real` |
| E52 eye-only gated policy no-motion shadow | `testbed/configs/policy_real_gmsl_eye2_e52_v1.yaml` | `scripts/run_e52_policy_shadow_check.sh` |
| GMSL four-camera policy shadow/control | `testbed/configs/policy_real_gmsl_four_camera_v1.yaml` | `tb-receiver-real --input policy` |
| Legacy one-dig FPV policy shadow/control | `testbed/configs/policy_real_one_dig_v1.yaml` | `tb-receiver-real --input policy` |
| Offline ACT training | `testbed/configs/act_real_v1.yaml` | `tb-train` |
| Offline ACT training, repaired 20Hz data | `testbed/configs/act_real_20hz_v1.yaml` | `tb-train` |
| Offline ACT training, GMSL four-camera qpos baseline | `testbed/configs/act_real_gmsl_four_camera_qpos_v1.yaml` | `tb-train` |
| Offline ACT training, GMSL four-camera qpos+qvel comparison | `testbed/configs/act_real_gmsl_four_camera_qpos_qvel_v1.yaml` | `tb-train` |
| Build online QC reference bundle | HDF5 dataset + manifest inputs | `tb-build-online-qc-reference` |

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
- online training-usability QC defaults under `receiver.online_qc`
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

`receiver.online_qc` is a per-frame training-usability gate. It writes
`diagnostics/train_exclude_mask=1` for local soft failures. Immediate hard
failures are reserved for frame-corrupting or physically discontinuous cases:
FPV decode failure, black/repeated FPV frames, large qpos jumps, and sustained
policy-vs-raw-IMU qpos divergence when a valid raw IMU qpos stream is present.
Qpos distribution outliers and FPV drift default to mask/review rather than
episode-level hard failure; set `qpos_distribution_hard_fail` or
`fpv_drift_hard_fail` only after backtesting the reference against known good
episodes. At episode save time, the final gate quarantines records under
`failed/` when total steps, healthy steps, healthy fraction, bucket reference
range, or bucket trajectory semantics indicate the episode is not train-ready.
Leave `reference_path` empty until an `online_qc_reference.json` has been
generated; without a reference, only reference-free checks run and distribution
drift is reported as unknown rather than treated as train-ready.

Use `--qc-dashboard` during recording when a human needs a stable terminal view
of QC state. It refreshes a `top`-style table with fixed rows for receiver
health, qpos distribution, bucket reference, bucket semantics, IMU/qpos
consistency, FPV frame health, FPV drift, and episode-final usability. Dynamic
fields such as warning code, mask count, healthy fraction, train-ready
candidate, semantic decision, and recent WARN/FAIL events stay in the same
columns so they can be scanned while the receiver loop runs. Add
`--qc-event-log /path/to/events.jsonl` to persist the de-duplicated WARN/FAIL
events shown by the dashboard for later review.

Build a reference bundle from train-ready data:

```bash
tb-build-online-qc-reference \
  --dataset-dir /path/to/real_teleop_v1_repaired_20hz_v1 \
  --manifest /path/to/train_ready_manifest.json \
  --training-qc-summary /path/to/training_qc_summary.json \
  --output /path/to/online_qc_reference.json
```

The reference bundle includes train-ready qpos/FPV statistics plus bucket qpos
limits copied from `training_qc_summary.json` and bucket semantic statistics
computed from strict-PASS episodes. Rebuild the bundle whenever the training QC
reference set changes.

## `policy_real_gmsl_four_camera_v1.yaml`

Defines the current four-camera GMSL policy receiver. The expected local bundle
is `policy_bundles/real_gmsl_four_camera_v1/` with:

- `policy_best.ckpt`
- `dataset_stats.pkl`
- `resolved_config.yaml`
- optional `run_metadata.json`

The bundle `resolved_config.yaml` must contain:

```yaml
task:
  camera_names: [video4, video5, video6, video7]
```

Use `scripts/run_policy_shadow_check.sh` for the default field preflight. It
checks the bundle files and the camera contract before starting `shadow_zero`.

## GMSL four-camera ACT training configs

Use these after GMSL HDF5 has been recorded, resampled to 20Hz, and checked by
training QC under
`/media/mundane/EXTERNAL_USB/real_gmsl_four_camera_20hz_v1`.

- `act_real_gmsl_four_camera_qpos_v1.yaml` is the qpos-only baseline.
- `act_real_gmsl_four_camera_qpos_qvel_v1.yaml` is the qpos+qvel comparison.

Both configs use the same `train_ready_manifest.json` and camera order
`[video4, video5, video6, video7]`. Recording can keep qvel in HDF5 even when
the training run uses only qpos.

## v2.0.1 Real Transition conditioned training

`act_real_transition_v2_0_1_b1.yaml` consumes the immutable materializer
output directly. The loader requires `real_transition_condition_v1[2]` and the
20-step `conditions/valid_mask`, keeps the condition at its fixed `-1/+1` and
`0/1` scale, computes qpos/action statistics from train episodes only, and uses
`split_manifest.json` as the source-block split authority. `locked_test`
episodes are recorded in run metadata but are not opened by the train or
validation loaders.

The B1 config also preserves the successful one-cycle N5 deadzone contract:
four-camera role encoding, direct-policy-output mechanical thresholds,
`expert_transition_window` promotion, and 50% state-hold transition sampling
with a 20-step horizon. It does not relabel every sub-deadzone action to zero
and does not add a suppressive runtime gate.

Condition is enforced with a train-and-validation, reward-shaped counterfactual
loss. For the last chunk-safe effective swing transition in each cycle, the
loader reuses the same image/qpos with the target side flipped. The
recorded-goal branch is rewarded for crossing the mechanical threshold; the
flipped branch is penalized if it keeps the old-goal direction above half
deadzone; and the two branches must differ by at least one direction-specific
deadzone. In addition, the recorded side label is emitted at every active
cycle observation and the action-class loss is evaluated on both the recorded
and same-observation flipped condition. This prevents a visual-only shortcut
from satisfying the condition objective. These are offline differentiable
surrogates, not online RL or physical-effect proof.
Training fails closed if any train or validation cycle lacks the required
terminal swing anchor. The same validation loss participates in checkpoint
selection, so a lower BC loss cannot hide a condition-ignored checkpoint.
Anchor coverage and the historical threshold artifact must be checked again
after the real materialized dataset arrives.

After copying a materialized root to the training machine, run:

```bash
PYTHONPATH=testbed python -m testbed.cli.train \
  --config testbed/testbed/configs/act_real_transition_v2_0_1_b1.yaml \
  --dataset-dir /path/to/materialized/episodes \
  --train-ready-manifest /path/to/materialized/train_ready_manifest.json \
  --split-manifest /path/to/materialized/split_manifest.json \
  --warm-start /path/to/n5/policy_best.ckpt
```

The warm-start path expands only the two ACT robot-state projection matrices:
the original four qpos columns are copied exactly and the two new condition
columns are initialized to zero. Use `--resume` only for an exact-shape 6D
checkpoint; the CLI rejects using resume and warm-start together.

For the fixed qvel utilisation ablation, use
`act_real_transition_v2_0_1_b1_qvel.yaml`. It keeps the same train-ready
manifest, source-block split, seed, epoch budget, deadzone/state-hold and
condition objectives, but sets `low_dim_keys` to
`[qpos, qvel, real_transition_condition_v1]` and expands the state dimension to
10. The warm-start copies the four qpos columns and leaves the new qvel and
condition columns at zero before training.

## Goal-only script planner

The conditioned ACT bundle can be composed by an upper-layer planner. The
planner owns only the target-side order; ACT remains the action owner. The
compact `ABBABABA` form is only shorthand. For a variable-length, variable-
order script, use [act_cycle_script_example.yaml](act_cycle_script_example.yaml)
or write JSON/YAML such as:

```yaml
schema: act_cycle_script_v1
script_id: strip_early
initial_side: A
loop: true
steps:
  - step_id: dig_b1
    target_side: B
  - step_id: dig_a1
    target_side: A
    label: short
  - target_side: A
  - target_side: B
    metadata: {workface: wf_02}
```

The first controlled field check uses two finite one-cycle scripts instead of
the multi-cycle example:

- `real_transition_single_cycle_right_to_left_v1.json` expresses `B -> A`;
- `real_transition_single_cycle_left_to_right_v1.json` expresses `A -> B`.

Both set `loop=false` and stop after one accepted target-ready boundary.

Then generate 16 goals from it:

```bash
PYTHONPATH=testbed python -m testbed.cli.act_cycle_planner \
  --script /path/to/cycle_script.yaml \
  --cycles 16 \
  --bundle-dir /data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_ctx04_b1_release_epoch499 \
  --output /path/to/abbababa_plan.json
```

For the compact form, the first character is the initial ready side; the
script form names `initial_side` explicitly and treats `steps` as the exact
target order. Each committed goal exposes only
`real_transition_condition_v1=[target_side_code,1]` to ACT. The planner
advances after a verified target-ready boundary and refuses a side mismatch;
it never sends actuator commands. The generated manifest is immutable by
default and can be consumed by a runner or attached to an observation with
`ABCyclePlanner.apply_condition()`.

For a runner that owns a `PolicyActionSource`, enable the same planner under
`teleop.policy.cycle_planner`. Call `source.commit_cycle_goal()` only after
the initial ready gate, pass each observation to `source.next_action()`, and
call `source.mark_cycle_target_ready(actual_side)` only after the existing
ready-evidence checks succeed. If a goal has not been committed, the policy
source fails closed instead of producing an action.

## Data-calibrated mock closed-loop check

Before a field run, `tb-real-transition-mock-eval` can exercise the same
backend → policy → controller → state-reader path without a bridge or
actuator. It fits the action-to-qvel response from **train-only** HDF5 rows,
derives A/B endpoint bands from train-ready cycle endpoints, and retrieves
real held-out camera frames by the predicted qpos state rather than by a fixed
time index (qvel remains in the fitted plant and ready/stability gate). A
separate train-only qpos bank supplies the support distance. The reader stops
at a data-support or safe-swing boundary, so an
unsupported surrogate state is reported as `UNSUPPORTED` rather than being
counted as model completion. The fitted reader also reuses the direct-output
deadzone table and the one-cycle state-hold rule: an axis below its positive or
negative threshold holds qpos and clears residual qvel.

```bash
PYTHONPATH=testbed python -m testbed.cli.real_transition_mock_eval \
  --bundle-dir /data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_ctx04_b1_release_epoch499 \
  --dataset-dir /path/to/materialized/episodes \
  --train-ready-manifest /path/to/materialized/train_ready_manifest.json \
  --ready-contract /path/to/real_transition_raw_v2/ready_contract.json \
  --cycle-manifest /path/to/materialized/cycle_manifest.jsonl \
  --output /path/to/mock_closed_loop_eval.json
```

The reported target bands are data diagnostics: the gate range covers the
observed train-split endpoint min/max (the current profile uses 60 train
episodes) with a small empirical margin, while
the q05–q95 band is retained for typical-pose inspection. The image support
warning and hard-stop limits are the train-only cross-episode p95 and p99
state distances. This is a causal observation-retrieval surrogate for
plumbing and bounded action checks; it is not hydraulic, soil, or field-effect
evidence.

For full planner capability checks use the reference replay below instead of
the legacy fitted-integrator check. The ctx04 records first swing through a
common 1.6–1.9 rad dump excursion and then a negative return; the A/B target is
primarily the return release/stop region. The ready safe range therefore applies
at cycle boundaries, not as a per-tick work envelope. A fitted linear qvel
integrator is retained for experiments, but it must first reproduce recorded
expert actions before it can be used as a model acceptance plant.

```bash
PYTHONPATH=testbed python -m testbed.cli.planner_open_loop_replay \
  --bundle-dir /path/to/conditioned-act-bundle \
  --dataset-dir /path/to/materialized/episodes \
  --train-ready-manifest /path/to/materialized/train_ready_manifest.json \
  --cycle-manifest /path/to/materialized/cycle_manifest.jsonl \
  --ready-contract /path/to/real_transition_raw_v2/ready_contract.json \
  --deadzone-thresholds /path/to/direct_policy_output_mechanical_deadzone.json \
  --split validation --mode continuous \
  --output /path/to/planner_open_loop_replay.json
```

This reference replay drives the planner through complete held-out 3/4/5-cycle
source runs, preserves ACT state across goal commits in `continuous` mode, and
reports dig-positive, return-negative, target-geometry, release-stop and
same-observation target-flip metrics separately. The recorded qpos/qvel/images
are the observation stream; policy actions do not update that stream, so the
result is an open-loop action-capability diagnostic rather than a physical
completion claim.

## `policy_real_one_dig_v1.yaml`

Defines the legacy single-FPV receiver for the real one-dig checkpoint. The
expected local bundle is `policy_bundles/real_one_dig_v1/` with:

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

Local dry run for the legacy FPV bundle:

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
