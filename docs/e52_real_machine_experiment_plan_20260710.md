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
3. During a supervised E52 control trace, what action intent was produced, how
   did each gate modify it, and what physical qpos/qvel response followed?
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
| D05 | Supervised E52 policy control with trace capture | Supported | `run_e52_policy_control_trace.sh` | Ready after P0-P2 |
| D06 | Terminal zero receipt, provenance, summary, timeline, and context | Supported | `termination.json`, `summarize_e52_control_trace.py` | Ready |
| D07 | Supervised segment/cycle | Mechanically supported after shorter reviewed runs | same control-trace entrypoint | Evidence-gated |
| D08 | Near-home stationary timeout gohome fallback | Documented design only | none | Blocked |

The checked-in E52 deployment config remains `output_mode: shadow_zero`. Do not
edit that default. Supervised motion uses the dedicated trace entrypoint, which
verifies the shadow configuration and applies a request-local control override
after explicit `CONFIRM_HARDWARE_MOTION=YES`.

## 3. Experiment task table

The machine-readable checklist is
`docs/e52_real_machine_experiment_tasks_20260710.csv`.

| Phase | Tasks | Machine motion | Exit gate |
| --- | --- | --- | --- |
| P0 provenance | T00-T03 | No | Exact code/bundle/config identities recorded |
| P1 platform preflight | T10-T14 | No | Sensors, cameras, bridge, zero command, and operator abort path pass |
| P2 E52 shadow | T20-T26 | No policy motion | E53 verdict passes at home and manually selected task states |
| P3 action response | T30-T36 | Scripted, one axis at a time | Active-axis response direction and boundedness are measured |
| P4 supervised E52 trace | T40-T47 | Human-observed policy motion | Complete trace and operator review |
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

## 8. P4: supervised E52 control trace

P4 intentionally uses human supervision instead of a second automatic motion
state machine:

```bash
cd /media/mundane/D/Excavator_real_stack
export CONFIRM_HARDWARE_MOTION=YES
export MAX_STEPS=4000
export IMAGE_INTERVAL_STEPS=5
./scripts/run_e52_policy_control_trace.sh
```

The operator watches motion continuously and stops with Ctrl+C, bridge
shutdown, physical emergency stop, or power removal when necessary. The
existing receiver/control pump sends zero on normal completion and Ctrl+C. The
trace extension records that terminal zero command and controller receipt.

The entrypoint does not add online qpos/qvel envelopes or silently alter gate
semantics. It only owns explicit motion confirmation, bundle preflight, a fresh
USB trace root, request-local control override, a finite maximum step limit,
configurable FPV capture frequency, and post-run trace analysis.

Trace completeness is not a physical-success verdict. The operator and offline
reviewer decide whether direction, duration, and resulting state were correct.

## 9. P4-P6 evaluation sequence

Use this order:

1. one supervised E52 run from a reviewed S1/S2/S3 anchor;
2. repeat the same anchor and direction enough times to distinguish repeatable
   response from one-off hydraulic/load variation;
3. compare observed qpos/qvel effect with the P3 response evidence and the
   operator's expected axis/direction;
4. extend duration only after every previous run has a complete trace and the
   stop/zero event is understood;
5. attempt one action segment before any full cycle;
6. treat any gohome request or transition as a separately reviewed event; do
   not count a run containing unexpected gohome motion as a P4 action pass.

Do not physically A/B raw ACT and E51 until matched-state reset and separate
bounded-run provenance are available. Logging raw ACT while executing E51 is a
counterfactual comparison, not a physical raw-ACT trial.

## 10. Global stop conditions

Immediately command zero and stop the current phase on:

- wrong axis or wrong direction;
- effective stick action;
- qpos/qvel leaving the operator-reviewed range for the current trial;
- unexpected continued motion after zero command;
- unexpected gohome request or transition during a P4 action trial;
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
/media/mundane/EXTERNAL_USB/policy_control_tests/e52_control_trace_<timestamp>/
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
also requires:

```text
termination.json
e52_trace_analysis/run_manifest.json
e52_trace_analysis/trace_summary.json
e52_trace_analysis/trace_context.json
e52_trace_analysis/trace_timeline.png
```

## 12. Current readiness verdict

- P0-P2: deployment code supports execution after Jetson synchronization and
  bundle preflight are confirmed.
- P3: supported by the existing scripted action-response tool, subject to
  physical stop verification and operator approval.
- P4: supported by the E52 control-trace entrypoint after Jetson sync and P0-P2
  review. Human observation is the primary stop decision.
- P5-P6: not blocked by missing control code, but remain evidence-gated on
  shorter-run trace review and the separate gohome decision.
- No code or documentation result should be interpreted as authorization to
  move the machine without the on-site operator's explicit confirmation.
