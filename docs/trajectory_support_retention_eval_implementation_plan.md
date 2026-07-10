# Trajectory Support Retention Evaluation Implementation Plan

Status: active research plan

Priority: this is the highest-priority source of truth for the
`codex/trajectory-support-retention-eval` worktree. Read and update this document
before starting another research or implementation slice in this worktree.

Base commit: `31eab44a2c52911eccd3dc501e9e3a743b0deb80`

Branch: `codex/trajectory-support-retention-eval`

## 1. Research question

The current offline policy evaluation is teacher-forced: the policy sees the
recorded expert observations at every step, so policy actions never change the
next qpos, qvel, or camera observation. This can show action similarity and
deadzone effectiveness, but it cannot show whether the policy would keep the
machine inside the part of state-observation space where the policy remains
predictive.

The research hypothesis is:

> A policy that accumulates effective action intent in a direction and amount
> consistent with expert trajectories should produce state progress that stays
> inside the expert trajectory support region for longer. A policy that
> accumulates missing, wrong-direction, wrong-axis, or over-duration intent
> should leave that support region earlier and become increasingly unreliable.

This is a hypothesis to test, not an assumption to encode into the score.

## 2. Required interpretation boundary

Do not treat raw joystick-action integration as physical state displacement.
The real mapping includes directional deadzones, hydraulic delay, load,
cross-axis coupling, saturation, and controller behavior. The evaluation must
separate three claims:

1. **Intent consistency**: under recorded observations, the policy accumulates
   effective axis-direction commands similar to the expert.
2. **Short-horizon effect consistency**: a transition estimator validated on
   held-out real data predicts that those commands produce expert-like qpos/qvel
   change over a bounded horizon.
3. **Support retention**: an explicitly approximate recursive rollout stays in
   the held-out expert support region while making forward task progress.

Passing an earlier level must not be reported as passing a later level. Only a
bounded real-machine closed-loop test can confirm actual physical retention.

## 3. Confirmed reference base

The planner and all future bounded implementation slices must reflect against:

- the user-provided research hypothesis above;
- base commit `31eab44a2c52911eccd3dc501e9e3a743b0deb80`;
- `docs/policy_model_effect_eval_protocol.md` for current open-loop replay,
  deadzone, local-window, full-dataset, and fixed-qpos/multi-FPV contracts;
- `testbed/testbed/cli/offline_policy_eval.py` for the current replay artifact
  contract;
- `testbed/testbed/policies/deadzone_eval.py` for directional deadzone semantics;
- actual HDF5 structures, manifests, resolved configs, deadzone artifacts, and
  evaluation outputs inspected in the active slice;
- episode-level held-out validation and data-driven thresholds rather than
  guessed constants.

If these references conflict with live data or code, record the conflict here
and stop the affected implementation slice until the source of truth is clear.

## 4. Target lock and non-goals

Owned research area:

- offline trajectory-support metrics;
- offline action-effect calibration and validation;
- held-out support-region estimation;
- deterministic report artifacts and plots;
- focused tests and this plan document.

Non-goals until explicitly promoted:

- no real-machine command transmission;
- no Jetson synchronization or remote diagnosis;
- no changes to checked-in training defaults;
- no changes to policy runtime behavior or safety behavior;
- no claim that an offline transition estimator is a real closed loop;
- no large visual world model or image generator;
- no replacement of the existing offline evaluation protocol.

The new evaluation is an additional gate above existing open-loop checks.

Task-specific invariant confirmed by the user on 2026-07-10:

- `stick` command is expected to remain zero in the current task design;
- Level 1 must treat policy stick impulse as unwanted action leakage, not as
  missing task progress;
- Level 2 must not fit or require a stick action-to-state response model;
- stick qpos/qvel may remain in support-state diagnostics, but commanded stick
  motion is not a required trajectory milestone.

## 5. Core quantities

For axis `j`, let `d_pos[j]` and `d_neg[j]` be the verified directional
deadzone thresholds. Define separate positive and negative effective command
magnitudes:

```text
effective_pos = max((action - d_pos) / (1 - d_pos), 0)
effective_neg = max((-action - d_neg) / (1 - d_neg), 0)
```

Both channels remain non-negative and must remain separate so opposite commands
cannot cancel and hide oscillation. For a horizon of `H` steps at measured
sample period `dt`:

```text
intent_impulse[j, dir, t, H] =
    sum(effective_command[j, dir, t:t+H]) * dt
```

Required horizons are derived from the verified sample period. For 20 Hz data,
the initial comparison set is `5, 10, 20, 40` steps (`0.25, 0.5, 1.0, 2.0 s`).
The implementation must read or receive `dt`; it must not silently assume 20 Hz.

## 6. Evaluation levels

### Level 0: existing action-effectiveness baseline

Reuse or preserve existing definitions for:

- same-axis same-direction effective action;
- extra-axis or wrong-direction effective action;
- startup delay and eventual liveness;
- idle false-active behavior;
- tail effective action and early gohome behavior.

This level remains teacher-forced and makes no state-retention claim.

### Level 1: effective-intent integral consistency

Inputs:

- expert and policy action arrays from the same full-episode replay;
- verified directional deadzone table;
- verified sample period;
- explicit episode and semantic-window identities.

Per axis, direction, window, and horizon, compute:

- expert and policy intent impulse;
- impulse magnitude ratio with explicit zero-denominator handling;
- missing expert impulse;
- extra or wrong-direction impulse;
- positive/negative cancellation ratio;
- cumulative-impulse trajectory error over time;
- onset delay and release overshoot;
- eight-channel impulse direction cosine only when both norms are meaningful.

Level 1 must compare cumulative curves, not only endpoint integrals. A policy
that first moves in the wrong direction and later cancels the error must not
look equivalent to a consistently correct policy.

### Level 2: short-horizon state-effect consistency

Build the smallest transition estimator that can pass held-out validation. The
first candidates should be deterministic local methods, such as regularized
linear regression and k-nearest recorded transitions. Inputs may include:

- current qpos and qvel;
- a short history of effective action channels;
- directional intent impulse;
- verified lag features;
- explicit axis-coupling terms only when real data supports them.

Outputs should initially be bounded-horizon qpos and qvel deltas, not generated
images. Evaluate horizons independently. Do not recursively apply this model in
Level 2.

Required validity checks:

- leave-one-episode-out or fixed held-out episode splits;
- expert-action prediction error by axis and horizon;
- sign accuracy relative to measured state change;
- calibration or neighbor-distance uncertainty;
- comparison with constant-state and constant-velocity baselines;
- failure by action magnitude, direction, phase, and multi-axis concurrency.

If the estimator cannot beat simple baselines or its error is comparable to the
candidate-policy effect difference, Level 2 is `inconclusive` and Level 3 is
blocked.

### Level 3: approximate recursive support retention

This level is conditional on a valid Level 2 estimator and a defensible source
of next observations. The preferred first approach is a held-out transition
graph or kNN retrieval from real episodes, not generated imagery.

At each recursive step:

1. evaluate the policy on a retrieved real observation;
2. apply the candidate's full offline action transformation;
3. estimate or retrieve the next state using only allowed held-out donor data;
4. retrieve a compatible next real observation without using the target
   episode as a donor;
5. compute support distance, progress, and estimator uncertainty;
6. stop on support exit, unavailable neighbor, excessive uncertainty, safety
   boundary, or the requested horizon.

Support cannot be defined by nearest qpos alone. At minimum it must include
normalized qpos, qvel, local progress or phase evidence, and recent effective
intent. Visual embeddings are optional and may be added only after their
distance behavior is validated.

Support thresholds must come from leave-one-episode-out expert-to-expert
distance distributions. There must be separate reasons for:

- `policy_support_exit`;
- `no_valid_transition_neighbor`;
- `transition_uncertainty_too_high`;
- `progress_stall`;
- `safety_boundary`;
- `completed_horizon`.

### Level 4: bounded physical closure

This is not part of the initial implementation. If Levels 0-3 are useful, the
first physical validation must use short, supervised, abortable action windows
and must compare predicted qpos/qvel effects with observed effects. Shadow-zero
alone cannot validate state retention because it does not apply policy actions.

## 7. Progress and support must both be measured

A zero-action policy may remain near a familiar start state indefinitely. It
must not pass by avoiding movement. Every support-retention result therefore
requires both:

- **retention**: state remains inside a calibrated expert support tube;
- **progress**: state advances through the expected local trajectory or reaches
  the expected semantic milestone.

Initial progress candidates are projection onto a held-out expert trajectory,
ordered local action phases, and data-supported bucket trajectory semantics.
The selected progress definition must be validated against expert episodes
before it is used to rank policies.

## 8. Mandatory controls and ablations

Every reported metric must include enough controls to demonstrate sensitivity:

- expert action upper-reference;
- zero action;
- one-step and five-step delayed expert action;
- sign-flipped action;
- axis-shuffled action;
- action scales `0.5` and `1.5` when those values are used only as evaluation
  controls;
- raw ACT action;
- each added gate stage when artifacts exist;
- the full candidate action.

Expected ordering is a validation target, not a forced scoring rule. If obvious
negative controls are not worse than expert or plausible candidates, the metric
or transition estimator is invalid.

## 9. Split and leakage rules

- The policy-training split, transition-estimator split, threshold-calibration
  split, and final test split must be recorded separately.
- The target test episode must never be a kNN or transition-graph donor.
- Thresholds, feature normalization, support radii, progress mapping, and model
  selection must be fitted without final-test episodes.
- Results must aggregate all eligible episodes and retain per-episode rows.
- New-domain or stress episodes must be reported separately and must not be
  mixed into threshold selection.
- Bootstrap confidence intervals must resample episodes, not individual frames.
- Poor transition coverage or observation matching yields `inconclusive`, not a
  forced pass or fail.

## 10. Artifact contract

The final evaluation package should contain:

```text
run_manifest.json
data_split.json
transition_model_metadata.json          # Level 2+
transition_validation_by_episode.csv    # Level 2+
intent_integral_by_episode.csv
intent_integral_aggregate.csv
support_rollout_by_anchor.csv            # Level 3+
support_survival_summary.json            # Level 3+
summary.json
plots/
  cumulative_intent_by_axis.png
  intent_error_by_horizon.png
  predicted_state_effect.png             # Level 2+
  support_survival_curve.png              # Level 3+
  progress_vs_support.png                 # Level 3+
```

Every summary must record dataset, manifest, episode set, policy artifact,
deadzone artifact, sample period, action source (`policy_action`, transformed
candidate action, or control), git commit, and exact evaluator arguments.

## 11. Proposed ownership

Keep mathematical owners separate from CLI/report orchestration:

```text
testbed/testbed/policies/trajectory_support_eval.py
    Effective intent channels, cumulative metrics, support distances, and
    typed result helpers without filesystem or plotting ownership.

scripts/trajectory_support_eval.py
    Artifact loading, split validation, candidate/control orchestration,
    aggregation, and report writing.

testbed/tests/test_trajectory_support_eval.py
    Synthetic contract tests, zero-denominator cases, cancellation detection,
    horizon alignment, and negative-control ordering fixtures.
```

Transition estimation may become a separate focused module only after the
Level 1 owner and real-data contract are stable. Do not place transition-model
training inside `offline_policy_eval.py` or `deadzone_eval.py`.

## 12. Implementation order and gates

### Phase A: provenance and feasibility audit

- Locate the actual candidate action artifacts, datasets, manifests, deadzone
  table, and qpos/qvel arrays available from the base commit environment.
- Verify episode coverage, control rate, action/qpos/qvel alignment, missing
  fields, and split provenance.
- Measure action-to-qpos/qvel lag and local response sample counts.
- Record a feasibility verdict for Levels 1, 2, and 3 separately.

Exit gate: no implementation constant is guessed; all required sources are
recorded, or the blocked level is marked `inconclusive`.

### Phase B: Level 1 core and tests

- Implement directional effective-command magnitudes and cumulative impulses.
- Implement horizon/window metrics and control generators.
- Add focused synthetic tests.
- Run one smoke artifact and inspect cumulative curves manually.

Exit gate: expert beats negative controls on deterministic fixtures; opposite
commands cannot hide through net cancellation; provenance appears in output.

### Phase C: full Level 1 candidate comparison

- Run all eligible episodes for raw ACT and available transformed candidates.
- Produce aggregate tables, plots, and episode bootstrap confidence intervals.
- Identify windows where endpoint similarity hides trajectory-shape failure.

Exit gate: metrics are stable under episode resampling and produce actionable
candidate differences beyond existing frame-level metrics.

### Phase D: Level 2 transition estimator

- Fit only the smallest validated estimator.
- Compare against constant-state and constant-velocity baselines.
- Report coverage and uncertainty by horizon, phase, direction, and axis.

Exit gate: held-out error is materially below baseline and below the policy
effect differences that the evaluator is expected to resolve.

### Phase E: Level 3 support-retention prototype

- Implement only after Phase D passes.
- Validate expert self-rollouts and all mandatory negative controls first.
- Compare support survival and progress for raw ACT and the full candidate.

Exit gate: expert and plausible candidates outrank negative controls without
rewarding zero-action stalling; unsupported retrievals are explicit.

## 13. Stop conditions

Stop and update this plan before broadening the implementation when:

- actual data paths or schemas differ from recorded assumptions;
- a deadzone threshold lacks verified provenance;
- alignment or sample rate is ambiguous;
- transition prediction fails the Level 2 validity gate;
- support distance cannot distinguish expert from mandatory negative controls;
- future observations would require invented or generated image content;
- the work would require changing runtime control, training defaults, public
  schemas, or Jetson state;
- an apparent improvement depends on final-test threshold tuning.

## 14. Current bounded slice

Phases A-C and the Phase D target-contract/baseline diagnostic are complete.
The next accepted slice is the **Phase D candidate-effect resolvability audit**:
apply only held-out-fitted transition models to raw ACT and E51 action features
at identical expert state anchors, measure expert-feature support/coverage, and
compare the predicted candidate-effect difference with held-out transition
residuals and episode-bootstrap uncertainty. Break results down by horizon,
axis, direction, phase, action magnitude, and multi-axis concurrency. A small
predicted difference relative to model error is `inconclusive`, not evidence of
equivalence. Do not begin recursive-rollout code.

## 15. Research log

### 2026-07-10: plan initialization

- Worktree created from the user-selected base commit.
- Current repo evaluation confirmed to be teacher-forced open-loop replay.
- Existing directional deadzone and fixed-qpos/multi-FPV contracts are retained.
- No runtime, training-default, remote, or Jetson changes are authorized.
- Phase A evidence audit was pending at initialization.

### 2026-07-10: Phase A provenance and feasibility audit

Verified sources:

```text
dataset:
  /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/
  real_teleop_v1_episodes_72_104_20hz
manifest:
  <dataset>/qc_batch_ref_72_87/train_ready_manifest.json
candidate package:
  /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/
  runs/policy_packages/e52_e51_causal_temporal_gate_candidate
candidate replay:
  /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/
  runs/offline_eval/e51_full_act_causal_temporal_gate_smoke_all_train_ready
deadzone artifact:
  /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/
  deadzone_policy_raw_for_runtime_scale.json
```

Observed facts:

- candidate package verification reports `ok=true`, 23 declared and 23 checked
  artifacts, with no errors;
- the manifest contains 24 train-ready episodes;
- the dataset contains 16,529 aligned steps, with 590 to 841 steps per episode;
- HDF5 `action`, `observations/qpos`, and `observations/qvel` lengths and
  `(T, 4)` shapes match in every train-ready episode;
- raw, phase-gated, snapped, and temporal-direction replay arrays match the
  HDF5 lengths for all 24 episodes;
- replay `expert_action` exactly matches the aligned HDF5 `action` arrays;
- replay timestamps have median `dt=0.05 s` in every episode;
- the deadzone artifact has explicit provenance as runtime-config deadzone
  thresholds adjusted back to raw policy-action scale; it is not a new direct
  physical response calibration.

Short-response feasibility observations using expert commands above the current
deadzone artifact:

| Axis/direction | Samples | Main observation |
| --- | ---: | --- |
| swing pos | 2,216 | strong same-sign qvel/qpos response through 0-4 step lag |
| swing neg | 2,263 | strong same-sign qvel/qpos response through 0-4 step lag |
| boom pos | 2,035 | qvel follows action sign, but qpos delta has the opposite sign |
| boom neg | 1,109 | qvel follows action sign, but qpos delta has the opposite sign |
| stick pos/neg | 0 | expected task behavior: commanded stick action remains zero |
| bucket pos | 1,113 | strong same-sign qvel/qpos response |
| bucket neg | 1,665 | strong same-sign qvel/qpos response, weaker magnitude correlation |

Feasibility verdict:

- **Level 1: feasible now.** Required actions, deadzone provenance, alignment,
  and sample period are present.
- **Level 2: partially feasible for the task-active axes.** Swing and bucket
  have substantial response samples. Stick is intentionally excluded from the
  action-effect estimator and retained as a zero-action leakage invariant. Boom
  still requires explicit resolution of the qvel/qpos sign contract before
  target construction.
- **Level 3: blocked.** It remains conditional on a held-out Level 2 estimator
  passing simple baselines and on a defensible next-observation retrieval
  contract.

### 2026-07-10: Level 1 mathematical owner and smoke

Implemented:

```text
testbed/testbed/policies/trajectory_support_eval.py
testbed/tests/test_trajectory_support_eval.py
```

The owner provides directional post-deadzone magnitudes, explicit-`dt`
cumulative impulses, cancellation-aware impulse metrics, and rolling horizon
rows. It has no filesystem, plotting, policy-runtime, or transition-model
responsibility.

Focused verification:

```text
PYTHONPATH=testbed pytest -q \
  testbed/tests/test_trajectory_support_eval.py \
  testbed/tests/test_deadzone_eval.py
```

Result: `10 passed`.

A read-only all-episode smoke compared expert, zero, delayed expert,
sign-flipped expert, raw ACT, and the E51 temporal candidate. At the 20-step
(`1.0 s`) horizon:

| Candidate | Mean channel L1 | Mean cumulative-path L1 | Missing impulse | Extra impulse | Mean valid cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| expert | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 1.00000 |
| zero | 0.17331 | 0.09098 | 0.17331 | 0.00000 | unavailable |
| expert delay 5 | 0.03577 | 0.02055 | 0.01790 | 0.01787 | 0.97829 |
| expert sign-flipped | 0.36984 | 0.19414 | 0.17331 | 0.19653 | 0.00000 |
| raw ACT | 0.11241 | 0.06128 | 0.02453 | 0.08788 | 0.97266 |
| E51 temporal | 0.10656 | 0.05804 | 0.02511 | 0.08145 | 0.97786 |

Initial interpretation, limited to teacher-forced Level 1:

- the metric correctly separates expert, zero, delayed, and sign-flipped
  controls;
- E51 reduces mean channel and cumulative-path errors by about 5% relative to
  raw ACT;
- E51 reduces extra policy impulse by about 7.3%;
- E51 increases missing expert impulse by about 2.3%, showing the expected
  suppression-versus-recall tradeoff rather than an unconditional improvement;
- this result does not yet show qpos/qvel effect consistency or recursive
  support retention.

### 2026-07-10: Level 1 report and full paired comparison

Implementation commit:

```text
f260d84857c54ef25dd8270c76b50804c31ed01a
```

Implemented:

```text
scripts/trajectory_support_eval.py
testbed/tests/test_trajectory_support_eval_report.py
```

The report owner validates identical expert/time contracts across candidates,
generates all mandatory controls, aggregates rolling windows to episode rows,
computes episode-level bootstrap confidence intervals, computes paired
candidate-versus-baseline bootstrap deltas, and writes CSV, JSON, and plots.
The output directory must be empty to prevent accidental artifact overwrite.

Focused verification result: `13 passed` across report, mathematical-owner, and
existing deadzone tests.

Formal artifact:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/
runs/offline_eval/e61_trajectory_support_level1_e51_all_train_ready
```

Coverage and provenance:

- evaluator git commit: `f260d84857c54ef25dd8270c76b50804c31ed01a`;
- 24 episodes and 16,529 steps;
- horizons: 5, 10, 20, and 40 steps;
- stride: 5 steps;
- 2,000 episode-level bootstrap samples with seed `20260710`;
- baseline: raw ACT;
- stages: phase-gated, snapped, and E51 causal temporal action;
- controls: expert, zero, delayed expert, sign-flipped expert, axis-shuffled
  expert, and expert scales 0.5/1.5;
- claim boundary: `teacher_forced_level1_only`.

Paired E51-versus-raw results:

| Horizon | Channel L1 change | Episode improvement | Extra impulse change | Missing impulse change | Cosine change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.25 s | -5.39% | 24/24 | -7.47% | +0.92% | +0.00357 |
| 0.50 s | -5.38% | 24/24 | -7.48% | +1.26% | +0.00295 |
| 1.00 s | -5.27% | 24/24 | -7.47% | +2.32% | +0.00528 |
| 2.00 s | -5.33% | 23/24 | -7.54% | +3.67% | +0.00460 |

At 1.0 s, the paired channel-L1 delta is `-0.00593` with 95% bootstrap CI
`[-0.00810, -0.00427]`; the cumulative-path error change is `-5.35%`. At 2.0 s,
the paired channel-L1 delta remains below zero with 95% CI
`[-0.01527, -0.00790]`.

Interpretation:

- Level 1 now gives stable episode-paired evidence that E51 reduces accumulated
  extra intent and cumulative action-path error relative to raw ACT;
- the benefit is not free: missing expert impulse increases monotonically with
  horizon, so later state-effect evaluation must test whether this suppression
  stalls progress;
- phase gating provides most of the reduction, snap changes aggregate Level 1
  metrics only slightly, and the temporal gate adds a smaller consistent gain;
- all real candidate stages retain zero effective stick impulse; the
  axis-shuffled negative control produces non-zero stick leakage, validating the
  zero-stick invariant;
- mandatory negative controls are separated in the expected direction, but
  this remains teacher-forced evidence and does not prove support retention.

### 2026-07-10: Phase D target contract and held-out baseline diagnostic

Implementation commit:

```text
eabebb4a584c56abf240797df545a284ab14d55b
```

Implemented:

```text
testbed/testbed/policies/trajectory_transition_eval.py
scripts/trajectory_transition_baseline_probe.py
testbed/tests/test_trajectory_transition_eval.py
testbed/tests/test_trajectory_transition_baseline_probe.py
```

The mathematical owner constructs bounded-horizon future-qpos targets,
shortest-angle swing deltas, explicitly signed initial-qvel displacement, and
directional post-deadzone action impulse. The report fits each linear model on
23 episodes and scores it on the omitted episode. Stick is never fitted with an
action model; it retains the user-confirmed constant-state invariant.

Resolved state contract:

- qpos order is `swing, boom, stick, bucket`;
- qvel is gyro-derived and is not assumed to be a qpos finite difference;
- qvel-to-qpos and action-to-qpos signs are `[+1, -1, +1, +1]`;
- the boom `-1` sign is explicit and agrees with all 24 episodes;
- mapped qvel versus measured qpos-rate global correlations are `0.902` for
  swing, `0.765` for boom, `-0.104` for stick, and `0.875` for bucket;
- swing, boom, and bucket correlations are positive in 24/24 episodes, while
  stick is positive in only 1/24, consistent with the inactive-axis contract.

Formal artifact:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/
runs/offline_eval/e62_trajectory_transition_baseline_all_train_ready
```

Coverage and provenance:

- evaluator git commit: `eabebb4a584c56abf240797df545a284ab14d55b`;
- 24 leave-one-episode-out folds at `dt=0.05 s`;
- horizons: 5, 10, 20, and 40 steps;
- stride: 5 steps;
- 2,000 episode-level bootstrap samples with seed `20260710`;
- claim boundary: `held_out_expert_transition_probe_only`.

Held-out `qvel_action_linear` MAE relative to constant state:

| Horizon | Swing | Boom | Bucket | Episodes better |
| --- | ---: | ---: | ---: | ---: |
| 0.25 s | -83.90% | -48.15% | -55.20% | 24/24 on every active axis |
| 0.50 s | -85.36% | -55.32% | -49.39% | 24/24 on every active axis |
| 1.00 s | -85.15% | -63.07% | -42.01% | 24/24 on every active axis |
| 2.00 s | -85.85% | -70.53% | -37.86% | 24/24 on every active axis |

At 1.0 s, held-out MAE is `0.01497 rad` for swing, `0.00664 rad` for boom,
and `0.11829 rad` for bucket. At 2.0 s it is `0.02895`, `0.01077`, and
`0.26109 rad`, respectively. Stick constant-state MAE remains only
`0.00030-0.00158 rad` across the evaluated horizons, while initial-qvel
extrapolation is substantially worse.

Interpretation and gate status:

- a small causal linear estimator materially beats constant-state on all
  task-active axes and every held-out episode, so the target/sign contract is
  usable for the next Level 2 diagnostic;
- the result validates expert-transition prediction only; it has not yet shown
  that raw ACT and E51 action features are in support or that their predicted
  effect difference is larger than estimator error;
- therefore Phase D is not complete and Level 3 remains blocked pending the
  candidate-effect resolvability and uncertainty audit.
