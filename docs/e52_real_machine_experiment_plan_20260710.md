# E52 Real-Machine Experiment Plan

Status: prepared; no hardware motion has been authorized or executed by this
plan.

Deployment baseline audited on 2026-07-10:

- repository: `/home/pingfan/Excavator_real_stack`;
- branch: `fs/online-qc-dev`;
- commit: `5f433bff38f7eb7bb67f2e7148b27657ec84a256`;
- upstream relation: local and `origin/fs/online-qc-dev` are `0/0`;
- E52 config: `testbed/testbed/configs/policy_real_gmsl_eye2_e52_v1.yaml`;
- E52 bundle verifier: `scripts/verify_e52_runtime_bundle.py`;
- E52 shadow entrypoint: `scripts/run_e52_policy_shadow_check.sh`.

The local repository contains unrelated existing changes. Do not include or
overwrite them when preparing this experiment. Jetson reachability was not
available during this audit, so Jetson commit, bundle presence, and bundle
preflight remain mandatory field checks.

## 1. Objective and claim boundary

The experiment must answer these questions in order:

1. Does the deployed E52 stack compute stable raw ACT and gated actions from
   live GMSL observations without commanding motion?
2. Do verified short commands produce the expected qpos/qvel direction and a
   measurable bounded response on the active task axes?
3. After a dedicated bounded-motion guard exists, does an E52 action window
   produce the expected physical progress without wrong-axis motion, support
   exit, early gohome, or tail motion?
4. Only after the previous gates pass, can a longer task segment be attempted.

This plan does not treat shadow output as physical success. It also does not
use failed HDF5 demonstrations as clean physical-correctness labels.

Task-specific invariant:

- `stick` is intentionally inactive and must remain below its effective
  deadzone. Do not schedule a policy stick-motion test.

## 2. Deployment capability matrix

| ID | Capability | Latest code | Entry point | Verdict |
| --- | --- | --- | --- | --- |
| D00 | Verify E52 ACT/gate files and SHA-256 | Supported | `verify_e52_runtime_bundle.py` | Ready after Jetson bundle check |
| D01 | Read-only IMU/CAN and camera/bridge bring-up | Supported | `imu_can_probe.py`, `slave_real_stack.sh` | Ready |
| D02 | Live E52 inference with zero command | Supported | `run_e52_policy_shadow_check.sh` | Ready |
| D03 | Verify commanded action stayed zero and gate diagnostics exist | Supported | `e53_verify_no_motion_policy_log.py` | Ready |
| D04 | Bounded scripted single-axis response with explicit confirmation and qpos abort | Supported | `calibrate_axis_response.py` | Ready after operator approval |
| D05 | E52 policy `control` output in generic receiver | Mechanically present | `record_real --policy-output-mode control` | Not an accepted field entrypoint |
| D06 | E52-specific bounded policy motion with arm delay, explicit qpos/qvel envelope, gohome suppression, and automatic verdict | Not implemented | none | Blocked |
| D07 | Full autonomous segment/cycle | Generic control can run, but required E52 safeguards are incomplete | none | Blocked |
| D08 | Near-home stationary timeout gohome fallback | Documented design only | none | Blocked |

The E52 deployment config and preflight intentionally require
`output_mode: shadow_zero`. Do not bypass this by editing the YAML or directly
passing `--policy-output-mode control` during the first field session.

## 3. Experiment task table

The machine-readable checklist is
`docs/e52_real_machine_experiment_tasks_20260710.csv`.

| Phase | Tasks | Machine motion | Exit gate |
| --- | --- | --- | --- |
| P0 provenance | T00-T03 | No | Exact code/bundle/config identities recorded |
| P1 platform preflight | T10-T14 | No | Sensors, cameras, bridge, zero command, and operator abort path pass |
| P2 E52 shadow | T20-T26 | No policy motion | E53 verdict passes at home and manually selected task states |
| P3 action response | T30-T36 | Scripted, one axis at a time | Active-axis response direction and boundedness are measured |
| P4 bounded E52 motion | T40-T47 | Short policy window | Blocked until D06 is implemented and reviewed |
| P5 task segment | T50-T55 | One supervised segment | Requires P4 repeated pass |
| P6 full cycle | T60-T65 | Full supervised cycle | Requires P5 pass and gohome decision |

## 4. P0: provenance lock

Record all of the following in the run sheet before starting field processes:

- host and Jetson repository commit and `git status --short`;
- config path and SHA-256;
- candidate manifest path, candidate id, and verification result;
- checkpoint, dataset stats, resolved config, and every gate artifact SHA-256;
- camera names and low-dimensional keys;
- deadzone artifact path and SHA-256;
- operator, observer, machine area, weather/lighting, soil condition, and abort
  authority;
- USB log root and available free space.

Required Jetson command after connectivity is restored:

```bash
cd /media/mundane/D/Excavator_real_stack
git rev-parse HEAD
git status --short
export PYTHONPATH="$PWD/testbed"
.venv/bin/python scripts/verify_e52_runtime_bundle.py \
  --config testbed/testbed/configs/policy_real_gmsl_eye2_e52_v1.yaml \
  --bundle-dir policy_bundles/real_gmsl_eye2_e52_v1
```

P0 fails on a commit mismatch, missing artifact, hash mismatch, wrong camera
set, wrong low-dimensional contract, enabled deadzone assist, or non-shadow E52
config.

## 5. P1: no-motion platform preflight

1. Establish a clear exclusion zone and assign one operator to the physical
   stop/override path.
2. Confirm the physical emergency stop, deadman, and manual override on the
   actual machine. A configured boolean is not proof of physical operation.
3. Run the four-IMU read-only probe and verify all required cameras are live.
4. Start the bridge/gateway for shadow as documented in
   `docs/host_slave_start_commands.md`.
5. Verify zero commands remain zero before loading the policy.

Abort P1 on stale cameras, invalid IMU attitude, bridge health failure,
unexpected CAN fault, non-zero command at idle, or an unverified physical stop
path.

## 6. P2: E52 shadow matrix

Run `scripts/run_e52_policy_shadow_check.sh` first at the home state. Repeat
shadow observation after the operator manually places the machine at a small
set of safe anchors; policy output remains zero-commanded throughout.

| Anchor | Purpose | Required review |
| --- | --- | --- |
| S0 home/idle | false-motion and early-gohome check | all gated actions below effective motion when task should be idle |
| S1 pre-swing | expected swing direction | raw and E52 direction, gate suppression, latency |
| S2 pre-boom | expected boom direction | explicit boom sign contract and qvel validity |
| S3 pre-bucket | expected bucket direction | bucket direction, magnitude, and feature-support concern |
| S4 post-action/tail | stop behavior | zero effective tail action and no early gohome |
| S5 visually shifted soil/lighting | visual robustness stress | persistent intent, wrong-axis activity, disagreement with operator label |

For every anchor, save `steps.jsonl`, camera frames, E53 report, operator
expected axis/direction annotation, and the exact start/end timestamps.

P2 pass requirements:

- E53 no-motion verdict is OK;
- `commanded_action`, `safe_action`, and returned action remain zero;
- raw ACT and all E52 gate stages are present in diagnostics;
- policy inference is live rather than constantly zero or constantly failing;
- stick remains below effective deadzone;
- no gohome request occurs outside the reviewed tail region;
- no persistent wrong-axis/wrong-direction output appears at a labeled anchor.

## 7. P3: bounded action-response identification

P3 collects the physical information missing from the offline evaluation. It
does not execute the policy. Use `scripts/calibrate_axis_response.py`, one axis
and one direction at a time, with explicit operator confirmation, measured
amplitudes, fixed duration/settle time, and `--abort-delta-rad`.

Active task axes for this plan are swing, boom, and bucket. Stick remains an
inactive invariant. Determine amplitudes from the existing deadzone artifact
and prior calibration evidence; do not invent or globally scale them during the
session.

Each trial must record:

- axis, direction, amplitude, duration, settle time, and abort delta;
- initial/final qpos and qvel, peak delta, response latency, and direction;
- `commanded_action`, controller acknowledgment, fault codes, and whether the
  automatic abort fired;
- operator observation and any hydraulic/load/soil condition note.

P3 passes per axis/direction only when repeated trials have the expected qpos
direction, bounded displacement, valid acknowledgments, and no cross-axis or
post-command motion requiring intervention.

## 8. P4: required bounded E52 motion capability

P4 is not executable with the accepted E52 entrypoints at the audited commit.
Before P4, implement and test a dedicated owner with all of these properties:

- explicit `--confirm-hardware-motion` equivalent;
- mandatory finite maximum control steps with no unlimited default;
- shadow warm-up before arming and a separate operator arm event;
- qpos delta envelope per axis, shortest-angle swing handling, and automatic
  zero/stop when crossed;
- qvel and sensor-validity stop conditions based on measured P3 evidence;
- gohome requests disabled for the first bounded policy windows;
- zero command on normal completion, guard trip, health failure, Ctrl+C, and
  process exception;
- complete raw ACT, every E52 stage, safe action, commanded action, qpos/qvel,
  guard reason, and stop-reason logging;
- a post-run verifier that refuses PASS if provenance or terminal zero evidence
  is missing.

Thresholds must be explicit experiment inputs sourced from P3; they must not be
new checked-in defaults inferred from offline imitation metrics.

## 9. P4-P6 evaluation sequence

Once D06 exists, use this order:

1. one short E52 window from a reviewed S1/S2/S3 anchor;
2. repeat the same anchor and direction enough times to distinguish repeatable
   response from one-off hydraulic/load variation;
3. compare observed qpos/qvel effect with the P3 response envelope and the
   operator's expected axis/direction;
4. extend duration only after every previous run terminates at zero without a
   guard, health, or operator abort;
5. attempt one action segment before any full cycle;
6. keep gohome disabled until action-side segment behavior passes and the
   gohome acceptance path has its own reviewed test.

Do not physically A/B raw ACT and E51 until matched-state reset and separate
bounded-run provenance are available. Logging raw ACT while executing E51 is a
counterfactual comparison, not a physical raw-ACT trial.

## 10. Global stop conditions

Immediately command zero and stop the current phase on:

- wrong axis or wrong direction;
- effective stick action;
- qpos/qvel envelope crossing;
- unexpected continued motion after zero command;
- early gohome request or any automatic gohome motion during P0-P4;
- camera, IMU, bridge, receiver-health, timestamp, or artifact-provenance
  failure;
- disagreement between logged `safe_action` and actual `commanded_action`;
- deadman, emergency stop, or manual override not behaving as verified;
- operator or observer calling stop.

After a stop, preserve logs and record the exact stop reason. Do not resume by
loosening a threshold in the same run.

## 11. Required outputs

Use a fresh run directory under:

```text
/media/mundane/EXTERNAL_USB/policy_control_tests/e52_field_trial_<timestamp>/
```

Required contents per task:

```text
run_manifest.json
task_record.json
steps.jsonl
summary.json
camera_frames/
operator_annotation.json
artifact_hashes.txt
```

P3 additionally requires the native `calibrate_axis_response.py` report. P4+
must add the bounded-motion verifier report once D06 is implemented.

## 12. Current readiness verdict

- P0-P2: deployment code supports execution after Jetson synchronization and
  bundle preflight are confirmed.
- P3: supported by the existing scripted action-response tool, subject to
  physical stop verification and operator approval.
- P4-P6: blocked. The generic policy control switch is not sufficient for this
  plan because the accepted E52-specific bounded-motion guard does not exist.
- No code or documentation result should be interpreted as authorization to
  move the machine without the on-site operator's explicit confirmation.
