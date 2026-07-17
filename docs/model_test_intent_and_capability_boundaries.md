# Model-test intent and capability boundaries

Last reviewed: 2026-07-16. Current experiment generation: G50 data-volume
ablation plus evaluation-semantics v2 and train-reference task-sequence
compatibility v1 plus short-horizon rollout trace v1. Review this file whenever
a test identity or report schema changes.

This file is the durable interpretation boundary for offline model and data
tests. A numeric result is not self-describing. Every new test must state its
intent and claim boundary before its result is used to choose a model, training
target, runtime mechanism, or data-collection plan.

This document is the source of truth for **what a test result is allowed to
mean**. `docs/policy_model_effect_eval_protocol.md` owns execution procedures;
individual experiment reports own their frozen data, checkpoint, metric, and
result facts. When those documents disagree about interpretation, this file's
claim boundary wins until it is explicitly revised.

## Evidence-strength rule

Tests in this repository are instruments, not interchangeable scores. Use the
following evidence order:

1. **Artifact and data integrity** can invalidate a run but cannot rank policy
   behavior.
2. **Representation and teacher-forced tests** can justify or reject a narrower
   training hypothesis but cannot promote a controller.
3. **Synthetic closed-loop tests** can veto a candidate by exposing a declared
   software failure mode. Passing them does not prove machine success.
4. **Observed command-response tests** provide real response evidence only for
   the commands, states, loads, and terrain actually represented in the data.
5. **Controlled real-machine tests** are required for claims about real motion,
   but still do not establish terrain or operating-condition generalization
   unless those conditions are explicitly held out and covered.

A result may move **down** this ladder as a falsification signal, but it may not
be promoted upward by interpretation. In particular, failure to reproduce one
held-out demo target under state hold is not a generic policy deadlock, and
reproduction is not proof that the machine moves.

## Breaking semantic migration: single-demo evidence is not ground truth

The current report contract intentionally has no compatibility aliases for the
old judgmental fields. A held-out expert trajectory is one valid task execution
sample. It is not the unique correct action sequence.

All current evaluators must separate four evidence layers:

1. `single_demo_similarity`: exact vector/set, direction recall, and timing
   relative to this recording. Descriptive only.
2. `single_demo_local_support` or `single_demo_event_support`: directions seen
   in this recording and declared horizon. This is not task-wide support.
3. executable proposal/liveness: whether a deadzone-effective proposal exists
   and remains stable under the declared synthetic intervention.
4. task validity/safety: only estimable from an explicit task, physical, or
   safety label. If absent, reports must emit `estimable=false` rather than
   infer invalidity from demo disagreement.

Removed public semantics include `required`, `exact_expert`, `unsupported`,
`unexpected_effective`, `target_recovered`, and `state_hold_deadlocked` when
their only reference was one demo. Existing artifacts using those schemas are
historical evidence only and must be regenerated before current comparison.

## Registry and maintenance policy

Every new test or materially changed test must update this document in the same
work slice. The update is required before its result enters a model-selection,
training, runtime, or field-test decision.

Each registry entry has two independent states:

- `evidence_status`: `valid`, `valid_with_scope_limitations`, `inconclusive`, or
  `invalid`;
- `lifecycle`: `current`, `superseded`, or `historical_only`.

`superseded` does not mean the old artifact is deleted. It means its numeric
ranking or dataset-specific conclusion must not govern the current experiment.
The failure mechanism may still remain relevant and must be stated separately.

For every new entry, maintain all of the following:

- the exact decision question and falsifiable failure;
- the intervention or synthetic update rule introduced by the test;
- the capability measured directly, the capability represented only by a
  proxy, and the capability not covered;
- permitted and forbidden interpretations;
- dataset/split, checkpoint, camera order, transforms, deadzone/action domain,
  temporal settings, and hashes;
- whether the test is a veto, a diagnostic, a ranking signal, or a promotion
  gate;
- lifecycle, replacement test, and expiry trigger;
- known distribution holes such as missing axis/direction, terrain, load,
  release, tail, or multi-axis examples.

Changing any of the following requires a new test identity rather than silently
overwriting the old result: label semantics, anchor extraction, state update,
deadzone domain, camera set/order, split, temporal aggregation, assist/runtime
path, pass/fail rule, or field-control mode.

## Current capability coverage map

Legend: `D` = directly measured under the declared test world, `P` = proxy or
partial evidence, `-` = not covered. A `D` does not escape that row's declared
world; for example state-hold directly measures one demo target's reproduction
in its frozen-state world, not generic liveness or behavior on the excavator.

| Test family | Data integrity | Single-demo similarity | Executable proposal | Synthetic proposal liveness | Single-demo relation | Representation content | Camera contribution | Observed machine response | Hydraulic/load dynamics | Terrain OOD generalization | Real closed-loop task |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| QC / training preflight | D | - | - | - | - | - | - | - | - | - | - |
| Training/validation loss | P | P | - | - | - | - | - | - | - | - | - |
| Teacher-forced open-loop replay | P | D | P | - | P | - | P | - | - | - | - |
| Deadzone-derived active/idle metrics | - | P | D | - | D | - | - | - | - | - | - |
| State-hold demo-target counterfactual | - | D | D | - | D | - | - | - | - | - | - |
| Armed natural-startup activation | P | P | D | D | P | - | P | - | - | - | - |
| Mechanical-assist A/B in replay | - | P | D | P | D | - | - | - | - | - | - |
| Frozen query-0 intent probe | P | - | - | - | - | D | P | - | - | - | - |
| Temporal-aggregation ablation | - | P | P | P | P | - | - | - | - | - | - |
| Matched eye2/four-camera/role A/B | P | P | P | P | P | P | D | - | - | - | - |
| Recorded command-response audit/probe | P | - | - | - | - | P | P | D | P | - | - |
| Fixed-state / image-swap sensitivity | P | P | P | - | P | P | D | - | - | P | - |
| Shadow-zero runtime test | P | - | D | - | D | - | P | - | - | - | - |
| Controlled command-on-machine test | P | - | D | P | D | - | P | D | D | P | P |
| Terrain-held-out controlled task suite | P | P | D | P | D | P | D | D | D | D | D |

The last row is the coverage target, not a completed test. G49 currently has
strong evidence in the middle columns and no direct evidence in the final four
columns.

The primary table predates the train-reference sequence dimension. Its explicit
coverage extension is:

| Test family | Train-cohort task-sequence compatibility | Exact task correctness | Self-generated state progression | Real closed-loop task |
| --- | :---: | :---: | :---: | :---: |
| Train-reference task-sequence compatibility | D | - | - | - |
| Short-horizon rollout trace audit | - | - | D* | D* |

`D` here is confined to deadzone-thresholded semantic direction changes in
saved teacher-forced trajectories. It must not be promoted into a task-success
claim. `D*` is conditional on a valid causal trace in the declared world. The
current teacher-forced negative control earns neither self-generated nor real
closed-loop evidence.

## Current test-family boundaries

### QC and training preflight

- **Intent:** prove that files, shapes, camera presence/order, split membership,
  exclusions, action domain, and experiment config agree.
- **Directly covers:** artifact/data/config integrity.
- **Does not cover:** label correctness, demonstration quality, motion intent,
  generalization, or controller behavior.
- **Use:** hard validity gate only. A pass means the experiment is executable,
  not useful.

### Training and validation loss

- **Intent:** monitor optimization of the declared supervised objective and
  select a checkpoint under the frozen validation procedure.
- **Directly covers:** only that objective on sampled expert chunks.
- **Does not cover:** executable onset, idle safety, self-generated states,
  terrain shift, hydraulic response, or task completion.
- **Misleading pattern:** calling the lowest loss the best controller. G49 N3
  has the best continuous MAE while failing startup liveness.

### Teacher-forced single-demo replay

- **Intent:** compare policy actions with one recorded expert sample while
  advancing through that recording's observations.
- **Directly covers:** single-demo action similarity, per-axis output
  distributions, and declared window metrics.
- **Proxy only:** whether the output crosses a configured command deadzone.
- **Does not cover:** whether that command moves the machine or whether the
  policy reaches the next observation under its own actions.
- **Use:** descriptive similarity and regression evidence within one matched
  protocol. Exact match is not correctness and is never a closed-loop promotion
  proof.

### Train-reference task-sequence compatibility

- **Intent:** ask whether a complete saved model trajectory follows a task
  sequence represented by the training expert cohort without treating one
  validation demonstration as the unique answer.
- **Intervention:** threshold actions with the frozen asymmetric direct-output
  deadzone; ignore leading idle and exact timing; collapse same-direction
  reactivation across idle gaps; emit a new token only on the first effective
  direction or an axis sign change.
- **Directly covers:** nearest train-expert semantic-sequence similarity,
  train-supported bigrams, aggregate direction repertoire, and coverage of
  directions present in at least 90% of training expert episodes.
- **Calibration requirement:** always report the held-out expert distribution
  plus reversed-sequence and first-event-only controls. Never judge a model from
  one low episode or a single exact-demo comparison.
- **Does not cover:** self-generated observations, command response, task
  completion, safety, or terrain generalization. A high score can still hide a
  missing phase; ordering and core-direction coverage remain separate.
- **Use:** cohort-level task-grammar diagnostic and model tradeoff analysis. It
  is not a promotion or safety gate.
- **Expiry trigger:** split, deadzone/action domain, semantic-token extraction,
  same-direction collapse rule, reference cohort, or camera/checkpoint changes.

### Short-horizon rollout trace audit

- **Intent:** distinguish a plausible policy sequence shown on recorded frames
  from observations actually produced after the policy's own sent commands.
- **Directly covers:** rollout authority, camera/qpos/qvel trace integrity,
  raw-to-returned-to-safe-to-commanded action-chain logging, bounded abort
  contract, and explicit previous-command-to-next-observation causal links.
- **Self-generated rule:** a post-initial observation counts only when it names
  the preceding acknowledged sent command, occurs after its send timestamp, and
  comes from `live_policy_on`, `simulator`, or an explicitly labelled synthetic
  dynamics world. Teacher-forced and state-hold observations never qualify.
- **Evidence levels:** noncausal none, incomplete trace, zero-command stream,
  synthetic proxy, simulator-direct, or physical short-horizon direct. The
  world remains part of every claim.
- **Safety:** the auditor is passive. `live_policy_on` requires an explicit
  bounded-control contract with per-axis command, delta, qpos, qvel, direction,
  deadman, acknowledgement, timing, and camera limits. No limits are inferred
  from data or defaults.
- **Does not cover:** task correctness, task completion, global safety, terrain
  generalization, or any unobserved period after the short horizon.
- **Use:** validity gate for future short rollout evidence. A trace-integrity or
  contract failure invalidates the run; a valid teacher-forced trace still has
  no self-generated-state evidence.
- **Expiry trigger:** trace schema, causal-parent rule, state-origin taxonomy,
  action-chain owner, safety contract, camera order, sampling rate, or horizon
  changes.

### Deadzone-derived executable and single-demo relation metrics

- **Intent:** reduce continuous command output to per-axis
  `negative / idle / positive` relative to one frozen asymmetric deadzone table.
- **Directly covers:** command-domain threshold crossing/direction and whether
  a proposal matches the same axis/direction at the same frame of one demo.
- **Required wording:** use `outside_single_demo_frame`, never
  `extra_or_wrong`. The current data cannot distinguish an alternative valid
  action from an invalid action.
- **Does not cover:** pressure build-up, hysteresis, temperature, load, coupled
  actuation, or physical displacement.
- **Expiry trigger:** a new action domain, action scale, machine calibration, or
  load-dependent response model.

### State-hold single-demo target counterfactual

- **Intent:** hold image/qpos at one demonstrated ineffective-to-effective
  anchor, zero qvel, and ask whether the policy reproduces that demonstrated
  axis/direction within a fixed horizon.
- **Directly covers:** `demo_target_reproduced` versus
  `demo_target_not_reproduced`, reproduction delay, teacher-forced concealment
  of that reproduction, anchor-relative extra directions, opposite-to-demo
  target ticks, and direction flips in the frozen-observation world.
- **Does not cover:** a unique correct action, generic deadlock/liveness, task
  validity, safety, physical hydraulics, changing pressure, image vibration,
  load response, terrain deformation, or task progress.
- **Use:** diagnostic of conditional target reproduction and temporal/software
  sensitivity. It is not a promotion or veto gate by itself.
- **Required wording:** `state_hold_demo_target_not_reproduced`; never shorten
  this to `deadlocked`. `anchor_extra_effective` means absent from the one demo
  anchor, not wrong or unsafe.
- **Misleading pattern:** translating `14/20 demo targets reproduced` into a
  70% startup rate or treating the other six as six policy deadlocks.

### Armed natural-startup activation

- **Intent:** distinguish a policy that merely emitted an effective command at
  some earlier recorded frame from one that still emits any effective command
  when the final expert-ineffective observation is explicitly armed and held.
- **Intervention:** advance policy state through recorded observations before
  the arm reference while suppressing commands, then repeat the observation at
  `single_demo_first_onset_step - 1` with qvel and previous command zero. No startup
  axis is required.
- **Directly covers:** raw command-domain liveness on any axis and delay under
  this observe-only-warmup plus frozen-observation software world.
- **Proxy only:** natural readiness near the recorded expert onset. The arm frame
  is a reproducible reference, not a ground-truth go signal; single-demo
  direction overlap and local support are descriptive only.
- **Does not cover:** premature-motion safety, held-idle false activation,
  physical motion, hydraulic response, task success, terrain generalization, or
  self-generated closed-loop observations.
- **Use:** diagnostic and relative ranking signal. A failure exposes absence or
  loss of effective intent in this synthetic world; a pass is not a promotion
  gate.
- **Expiry trigger:** any change to event extraction, arm reference, warmup state
  update, deadzone/action domain, camera order, temporal aggregation, qvel or
  previous-command handling, validation split, or checkpoint generation.

### Mechanical-assist A/B

- **Intent:** determine whether the fixed monotonic assist transform rescues a
  sub-deadzone proposal without changing its proposed sign.
- **Directly covers:** the software transform and its synthetic replay effects.
- **Does not cover:** improved learned policy understanding or guaranteed real
  response.
- **Use:** report raw and assisted paths separately. Assist recovery must never
  be attributed to the raw model.

### Frozen query-0 intent probe

- **Intent:** ask whether a frozen inference-time decoder feature contains
  linearly accessible per-axis ternary intent information.
- **Directly covers:** supervised linear separability under the frozen split,
  feature layer, and probe training rule.
- **Does not cover:** calibrated probabilities, safe argmax projection,
  temporal execution, causal camera use, terrain understanding, liveness, or
  deployability.
- **Use:** decide whether a head design is worth testing. It is not a controller
  score.

### Temporal-aggregation ablation

- **Intent:** attribute output differences to the current chunk aggregation
  rule by executing query 0 without aggregation.
- **Directly covers:** that software-path difference during expert replay.
- **Does not cover:** self-generated online observations or real response.
- **Use:** causal attribution inside the replay implementation, not a general
  claim that memory helps or hurts the machine.

### Camera-count and camera-role A/B

- **Intent:** under matched data/training/evaluation, measure whether changing
  camera availability or explicit role identity changes policy/feature results.
- **Directly covers:** contribution of those streams under the tested fusion
  architecture.
- **Does not cover:** semantic scene understanding, calibrated geometry,
  terrain causality, or usefulness under a different fusion design.
- **Misleading pattern:** interpreting a failed naive four-camera model as proof
  that video6/video7 are harmful or redundant.

### Recorded command-response audits and probes

- **Intent:** use causally aligned recorded commands and later qpos/qvel to ask
  whether a response target is observable or predictable across episodes.
- **Directly covers:** associations present in the recorded operator-command
  distribution.
- **Proxy only:** hydraulic/load response, because operator choice, state, load,
  and terrain are confounded.
- **Does not cover:** the counterfactual response to a policy command that was
  never sent, retry correctness, or closed-loop policy success.

### Fixed-state / image-swap sensitivity

- **Intent:** perturb visual input while keeping a declared low-dimensional
  trajectory fixed to detect camera/domain shortcuts.
- **Directly covers:** prediction sensitivity to that constructed image swap and
  mapping quality.
- **Does not cover:** physically possible multi-view observations or real
  terrain generalization when the swapped image and state are inconsistent.
- **Use:** a large change is shortcut evidence; a small change is not proof of
  visual robustness.

### Shadow-zero and controlled field tests

- **Shadow-zero intent:** verify the live observation/inference/safety/logging
  chain while commanding physical zero. It does not test machine motion.
- **Controlled command intent:** observe real response under bounded authorized
  commands and explicit abort rules. It directly covers only sampled states,
  loads, directions, and terrain.
- **Generalization requirement:** claims about terrain diversity require an
  explicit terrain taxonomy, held-out terrain groups, repeated axis/direction
  coverage, and real response/task outcomes. Random or source-mixed episode
  validation is insufficient.

## G49 current registry snapshot

| Test ID | Evidence status | Lifecycle | Role | Permitted conclusion | Forbidden conclusion |
| --- | --- | --- | --- | --- | --- |
| `g49_training_validation_n0_n4` | valid_with_scope_limitations | current | checkpoint selection/optimization diagnostic | objective behavior on the fixed 120/20 split | best real controller or terrain generalization |
| `g49_open_loop_val20` | valid_with_scope_limitations | current | matched imitation/ranking signal | N3 is the closest continuous expert-path regressor; N1 is more transition-responsive | N3 or N1 is the best real-machine policy |
| `g49_train_reference_task_sequence_v1` | valid_with_scope_limitations | current | train-cohort task-grammar diagnostic | all N0--N5 outputs are much closer to forward expert task sequences than reversed or first-event-only controls; N2 best preserves emitted phase order, N5 best covers the full eight-direction repertoire, and N1 is the most balanced | that one held-out demo is uniquely correct, that high similarity proves a complete task, or that any model is closed-loop/field ready |
| `short_horizon_rollout_trace_v1` | valid | current | causal trace validity gate | only an acknowledged sent command explicitly linked to the next observation can establish self-generated state in the declared world | that plausible actions on recorded observations are closed loop, or that a short causal trace proves task success or safety |
| `g49_n5_shadow_deployment_preflight_v1` | valid | current | bundle identity and no-motion runtime compatibility gate | the packaged N5 checkpoint, stats, resolved config, camera order/roles, identity action domain, 20 Hz policy time base, 50 Hz zero-hold pump, and shadow-only settings are mutually consistent | that Jetson performance is sufficient, that live cameras are healthy, that N5 is selective or safe, or that bounded control is authorized |
| `g49_n5_teacher_forced_short_rollout_negative_control_v1` | valid | current | noncausal negative control | the 20-tick N5 trace is structurally valid and shows a plausible proposal motif, but has 0/19 causal state transitions and correctly receives no self-generated-state evidence | physical response, task progress, task success, or failure of N5 closed-loop control |
| `g49_deadzone_intent_val20` | valid_with_scope_limitations | current | command-domain diagnostic | relative active/idle/opposite output under the frozen table | actual movement or load-aware deadzone behavior |
| `g49_single_demo_open_loop_similarity_v4` | valid_with_scope_limitations | current | first-candidate readiness and single-demo similarity diagnostic | when the saved policy first becomes executable and how its first direction set compares with one recorded event | correctness, task-wide support, a required startup axis, unsafe early motion, closed-loop response, or task success |
| `g49_state_hold_demo_target_val20_h20` | valid_with_scope_limitations | current | conditional demo-target reproduction diagnostic | models differ in which demonstrated targets they reproduce under a frozen observation | generic deadlock/liveness, invalidity of anchor-extra actions, real startup probability, or hydraulic recovery rate |
| `g49_startup_activation_val20_h20_v1` | valid_with_scope_limitations | current | natural-startup persistence diagnostic and relative ranking signal | N1 retains selective any-axis intent in 18/20; N5 reaches 20/20 but demonstrates that this metric alone rewards globally aggressive output | a required startup axis, false-active safety, physical startup probability, terrain generalization, or deployment readiness |
| `g49_n5_fourcam_role_transition_cross` | valid_with_scope_limitations | current | missing factorial training/evaluation cell | transition supervision transfers to the tested four-camera role path mainly as broader activation, not selective target startup | stick cameras are harmful, pair-aware fusion will fail, or N5 is field-ready |
| `g50_n5_data_volume_ablation` | valid_with_scope_limitations | current | fixed-step practical data-volume learning curve | N5 broad natural activation is saturated at 20 episodes; 20--120 episodes do not monotonically improve single-demo similarity, while outside-single-demo-event directions decrease with more data | that outside-demo means invalid, less data is generally better, 20 episodes cover terrain generalization, or sample-count causality is isolated from subset composition |
| `g49_frozen_query0_probe` | valid_with_scope_limitations | current | representation hypothesis diagnostic | startup information is linearly readable, including in N3 four-camera features | strict argmax is safe or cameras understand terrain |
| `g49_no_temporal_aggregation` | valid_with_scope_limitations | current | replay-path attribution | temporal aggregation is not the main cause of the observed G49 startup gap | online memory is solved or irrelevant |
| `g49_all_axis_response_envelope_v1` | valid_with_scope_limitations | current | historical command-response calibration | train-derived command-to-qvel signs, clean from-rest latency, validation support coverage, and explicit response unknowns for all four axes | that an unsent model command moved the machine, terrain causality, or closed-loop task success |
| `g49_policy_episode_action_and_response_evidence_v2` | valid_with_scope_limitations | current | orthogonal episode-match and response-evidence coverage | N5 first-event commands are mostly advanced current-episode actions: 39/41 effective axes occur in the same episode and all 41 occur in training events; 30/41 separately lack enough similar-condition response samples | that insufficient response evidence is a model error, that one expert timing is uniquely correct, or that historically supported commands guarantee task success |
| `g49_camera_count_role_ab` | valid_with_scope_limitations | current | architecture-specific camera attribution | extra cameras improve some regression/safety signals; additive role identity is insufficient | extra cameras reduce confidence in general or pair-aware fusion will fail |
| `g49_metric_validity_audit` | valid | current | interpretation correction | current headline metrics are useful diagnostics but require episode/axis macro, exact-vector, safety, and delay-curve companions | any single current headline number is a complete promotion score |
| `g49_assist_ab` | inconclusive | current | missing runtime-transform attribution | assist was not evaluated in the current batch | any raw-versus-assist benefit or safety claim |
| `g49_held_idle_safety` | inconclusive | current | missing synthetic idle evidence | transition-anchor state-hold is not an idle-hold test | stable idle behavior under repeated inference |
| `g49_release_tail_safety` | inconclusive | current | missing phase coverage | current startup evidence says nothing about stopping or releasing | safe release, no lingering command, or task-tail stability |
| `g49_image_swap_sensitivity` | valid_with_scope_limitations | current | fixed-qpos cross-style visual-sensitivity diagnostic | N5's reset-step-0 direction set changes in 21/24 old-image versus nearest-qpos new-image swaps, so its output is materially image-sensitive under this construction | correct old-task interpretation, physically consistent swapped observations, visual robustness, or terrain causality |
| `g49_n5_old_fourcam_cross_style_24` | valid_with_scope_limitations | current | old-style transfer diagnostic | N5 produces executable output on old four-camera observations and partially follows later old-task phases, while failing the old expert's uniform bucket-positive startup convention | task-level cross-style generalization, closed-loop transfer, or field readiness |
| `g49_real_command_response` | inconclusive | current | missing promotion evidence | no current G49 real-response conclusion | any physical or terrain-generalization claim |
| `g49_controlled_field_generalization` | inconclusive | current | missing promotion evidence | no current G49 field conclusion | deployment readiness |

Current G49 startup validation is task-consistent and capability-coverage
limited: 18/20 first-crossing anchors are stick-positive and 2/20 are
boom-negative. This concentration reflects a repeated expert task sequence,
not random demonstration behavior. Swing, bucket, opposite stick, positive
boom, alternative initial task sequences, terrain-held-out, load-held-out,
release/tail, and real-response startup claims remain uncovered by that subset.
It also does not provide a held-idle false-activation test or balanced
joint-action onset coverage.

The current metric audit further narrows the headline interpretation. The 20
startup anchors are one first-crossing observation per validation episode, not a
statistically designed or axis-balanced sample. Expert first action occurs 27--53
ticks after recording begins in this validation block (median 31.5 ticks, or
1.575 seconds at 20 Hz). This preparation interval is not idle ground truth.
The accepted startup diagnostic therefore uses only the policy's first
deadzone-effective output and treats exact-anchor, overlap, single-demo local
support, and opposite-direction fields as descriptive similarity. There is
no required startup axis and none of these fields is a promotion or safety gate.
See `docs/g49_evaluation_metric_audit_20260715.md`.

## Historical-result validity audit

| Historical evidence | Lifecycle now | What remains valid | What must not be reused as current |
| --- | --- | --- | --- |
| Original teacher-forced replay rankings | historical_only | expert-path imitation can reveal regression and local action shape | closed-loop liveness or live readiness |
| Old 19/5 state-hold matrices | historical_only | frozen-observation target-reproduction differences and traces remain useful | the old `deadlock/recovery/unexpected` judgment, numeric ranking of G49 checkpoints, or new-data startup rates |
| Old H3 factorized hard projection | historical_only | hard projection can amplify a classification error into an effective wrong command | claim that factorized intent must fail on the 120-episode dataset |
| Old frozen probe with five bucket-positive startup anchors | superseded | query-0 representation probing is a valid diagnostic design | the five-anchor distribution, stick non-estimability, or its model ranking for G49 |
| Eye2-only visual response probes | historical_only | generic frozen eye features were unstable for that response target | conclusion that all four-camera visual information is useless |
| Four-camera frozen response probe | historical_only | video6/video7 changed episode-level response predictions under that probe | terrain recognition, causal load inference, or policy benefit |
| Legacy joystick-scaled deadzone/action interpretation | superseded | joystick scaling remains a separate field-interface concern where configured | compressing direct model output or reusing scaled deadzones in current G49 evaluation |
| G43 temporal-history rejection | historical_only | longer/causal history can improve some segments while worsening extra output and hidden anchors | conclusion that all temporal input or new-data temporal learning is harmful |

The durable rule is to preserve **failure mechanisms**, not automatically
preserve old model rankings. Dataset-specific counts, thresholds, and pass/fail
numbers expire when their split, action domain, camera set, label support, or
runtime path changes.

## Required experiment contract

Each test artifact directory must contain an `experiment_contract.json` written
before promotion decisions. It must record:

1. `test_id` and status: `valid`, `valid_with_scope_limitations`,
   `inconclusive`, or `invalid`;
2. intent, decision question, and falsifiable hypothesis;
3. observable inputs, controlled variables, target, and reference base;
4. source-data inclusion, exclusions, provenance, and leakage boundaries;
5. validation isolation and predeclared pass/fail or stop conditions;
6. capabilities that the test actually measures;
7. forbidden interpretations and non-goals;
8. runtime/control changes, if any;
9. artifact paths and hashes.

In addition, every report must include this compact capability card:

```text
Test identity:
Lifecycle: current | superseded | historical_only
Decision role: validity gate | diagnostic | ranking signal | veto | promotion gate
Test world / intervention:
Direct coverage:
Proxy coverage:
Not covered:
Permitted claims:
Forbidden claims:
Known distribution holes:
Expiry trigger / replacement:
```

The author of a new test owns adding or updating its row in the current registry
and coverage map. A reviewer must reject a promotion claim when the artifact has
no capability card, when its required hash/config identity is incomplete, or
when the claim belongs to a stronger evidence level than the test.

The contract precedes the metrics. If the metric definition is later found to
be invalid, keep the artifact, change its status to `invalid`, and state why. Do
not silently replace it with a corrected result under the same test identity.

## Capability levels that must not be collapsed

- **Label/data audit**: checks whether a target can be constructed consistently.
- **Response identifiability**: checks whether observations predict a measured
  response under an observational dataset.
- **Teacher-forced/open-loop imitation**: checks predictions along expert
  observations; it does not execute policy actions.
- **Counterfactual/state-hold replay**: stresses a declared synthetic state
  update; it is only as real as that update rule.
- **Real closed-loop control**: executes policy commands and observes the actual
  machine/environment response.

Evidence from an earlier level cannot be described as evidence from a later
level. In particular, visual features are not terrain labels, operator actions
are not optimal-control targets, and an observational command-response model is
not a causal hydraulic model.

## Current record: `fresh_action_response_probe_20260714`

### Intent

Starting directly from the aligned HDF5 observations, test whether existing
episodes contain episode-generalizable information about the *incremental*
hydraulic response after an action crosses the configured deadzone. Previous
response sidecars, terrain clusters, policy predictions, and prior experiment
verdicts are excluded from label generation.

### Data and isolation

- input set: every structurally aligned 20 Hz HDF5 found in the existing
  `episode_72..104` directory;
- invalid and `train_exclude_mask` windows are removed directly from each file;
- event: a per-axis transition from inactive/opposite direction to an effective
  direct-domain command;
- target: the positive increase in same-direction qvel peak over the preceding
  eight-tick baseline, observed in the following eight ticks;
- outer validation: leave one complete episode out;
- inner ridge selection: episode-grouped only;
- forbidden heldout `105..109`: not read.

### Feature-family question

Compare current action only, action plus qpos/qvel, eight causal history ticks,
current video4/video5 generic frozen visual features, their combination, and a
noncausal future-action diagnostic ceiling. The future-action feature is never
a proposed runtime input.

### Capability boundary

This test can support only:

- response-label construction review;
- episode-held-out response-identifiability comparison;
- deciding whether a bounded response-aware training experiment has enough
  signal to be worth attempting.

It cannot establish:

- closed-loop policy success or deadlock recovery;
- causal terrain/load effects;
- terrain recognition or terrain-domain coverage;
- expert-action optimality;
- live-control or deployment readiness;
- that visual information is generally useful or useless.

### Invalid predecessor retained for audit

The first probe used future peak qvel and short-horizon qpos displacement. It is
`invalid`: current velocity contaminated the peak-qvel target, and the qpos
relative-error denominator was unstable at the observed scale. Its artifact is
retained only to prevent future reuse of those metrics.

### Current scoped result

The corrected **eye2-only** probe has 285 transitions across 30 structurally aligned
episodes: swing 61, boom 94, bucket 130, stick 0. On incremental response gain,
the episode-macro relative MAE is 0.831 for action only, 0.721 after adding
current qpos/qvel, 0.723 with causal history, 0.771 with current generic visual
features, and 0.712 for the noncausal future-action ceiling. This is evidence
that current state contains some reusable response information. It is not
evidence that history or four-camera vision has no information: this probe used
only video4/video5, the event count is small, the visual encoder is generic and
frozen, and several held-out episodes remain poorly predicted.

The next permissible slice is a bounded auxiliary response-target feasibility
test. It must not alter runtime action selection, and it must retain independent
open-loop, state-hold, and hidden-action gates before any policy claim.

## Follow-up record: `fresh_response_mlp_probe_v2_20260714`

### Intent and fixed stop rule

This follow-up asked whether a fixed small nonlinear model could reveal
episode-generalizable history or visual interactions missed by the ridge
probe. It used a 64-32 Softplus MLP, three-seed ensemble, five outer episode
groups, and inner validation containing complete training episodes. No ACT
policy was trained and no runtime path changed.

Promotion required history or vision to improve episode-macro relative MAE over
`action_state`, improve a majority of held-out episodes, and avoid a large
worst-episode regression.

### Result

| Feature family | Macro relative MAE | Better than median baseline | Head-to-head vs `action_state` |
| --- | ---: | ---: | ---: |
| action only | 0.793 | 24/30 | n/a |
| action + state | 0.740 | 26/30 | reference |
| + causal history | 0.702 | 26/30 | better 14/30, worse 16/30 |
| + current generic vision | 0.816 | 27/30 | better 10/30, worse 20/30 |
| + history + vision | 0.780 | 24/30 | better 13/30, worse 17/30 |
| noncausal future-action ceiling | 0.668 | 26/30 | better 17/30, worse 13/30 |

History's macro mean improved, but its median head-to-head episode delta was
`+0.033` and its worst regression was `+0.238`. Vision's worst regression was
`+0.568`. Neither passed the predeclared stability requirement. The failed
first launch is retained separately as `fresh_response_mlp_probe_20260714` with
status `invalid_execution`; it failed before model training and has no metrics.

### Decision and boundary

Do not attach this response target or these generic visual features to ACT on
the current evidence. The valid conclusion is narrow: current state carries
some reusable response signal, while the added history and generic visual
features are not stable across the existing episodes under this probe. This is
not proof that history or vision is intrinsically useless. Revisit only with
independently stronger response/load observations or a new predeclared
validation argument.

## Correction record: four-camera response probes

The existing HDF5s contain four synchronized encoded streams:
`video4/video5/video6/video7`. The preceding visual probes used only video4 and
video5. Their negative visual result is therefore an **eye2 result**, not a
statement about the available visual dataset.

Two new tests retained the same 285 events, targets, episode isolation, model
settings, and random seeds, changing only the visual input from two to four
cameras.

| Probe | action + state | eye2 vision | four-camera vision | Four-camera vs eye2 |
| --- | ---: | ---: | ---: | ---: |
| ridge macro relative MAE | 0.721 | 0.771 | 0.787 | worse |
| nonlinear MLP macro relative MAE | 0.740 | 0.816 | 0.773 | better |

For the nonlinear probe, four-camera vision beat eye2 vision on `21/30`
episodes. This is evidence that video6/video7 contain additional usable
information. Four-camera vision still beat the state-only reference on only
`13/30` episodes, regressed on `17/30`, and had a worst head-to-head regression
of `+0.299`; it therefore failed the predeclared promotion condition.

The corrected interpretation is:

- eye2 discards information that can matter for response prediction;
- naive frozen-feature concatenation is not a sufficient multi-view fusion
  method;
- this result supports a separately declared learned four-camera policy/fusion
  A/B;
- it does not establish terrain recognition, hydraulic causality, closed-loop
  recovery, or deployment readiness.

## Model record: matched eye2 vs four-camera ACT state-hold

The existing baseline bundles permit a direct camera-count A/B. Both use
qpos-only ACT, 2000 epochs, no image transform, and the same 19 train / 5
validation episode IDs. The fresh evaluation used identity action scale,
current direct-output deadzones, horizon 20, complete traces, and the same 48
validation anchors.

| Model | Raw demo target reproduced | Raw reproduction hidden | Assist demo target reproduced | Assist reproduction hidden | Assist anchor-extra effective |
| --- | ---: | ---: | ---: | ---: | ---: |
| eye2 ACT | 35/48 | 5 | 43/48 | 3 | 7 |
| four-camera ACT | 25/48 | 9 | 43/48 | 2 | 7 |

The old predeclared promotion condition is retired because it treated one demo
target as the required answer. Descriptively, assist reproduction did not
increase and raw reproduction was ten anchors lower. The differences were
complementary: four-camera ACT reproduced two targets eye2 did not, eye2
reproduced two targets four-camera did not, and three targets were reproduced
by neither. Thus video6/video7 change decisions, but this test cannot call one
decision set more correct or more generally live.

The current ACT image path applies `backbones[0]` to every camera and
concatenates the resulting maps along the image-width dimension. This path has
no explicit camera-identity embedding or learned view selector. That is an
implementation fact; the hypothesis that it causes the observed regression is
not yet proven.

The next valid model test is therefore a camera-aware fusion A/B with explicit
camera identity and a reversible fusion owner. It must compare against both
matched baselines, retain executable-proposal, single-demo-relation, and
inference-latency diagnostics without converting them into correctness gates.
Simply changing `camera_names` to four again is not a new experiment.

## Model record: additive eye/stick camera-role encoding

This experiment tested the narrowest camera-aware change. The four cameras
were declared by their physical roles: `video4/video5 = eye` and
`video6/video7 = stick`. ACT received a learned identity vector for each camera
plus a learned vector shared by cameras in the same role. These vectors were
added to each camera's spatial position encoding before the existing
transformer. Both tables were zero-initialized, making enablement an exact
functional rollback before training.

The candidate used the same 24 episodes, 19/5 split, qpos-only input, four
cameras, image transform, seed, optimizer settings, and 2000-epoch budget as
the matched four-camera baseline. The formal training completed with best
epoch 1705. Teacher-forced validation loss improved from `0.11784` to
`0.08590`, but this was a secondary metric.

| Model | Raw demo target reproduced | Raw reproduction hidden | Assist demo target reproduced | Assist reproduction hidden | Assist anchor-extra effective |
| --- | ---: | ---: | ---: | ---: | ---: |
| eye2 baseline | 35/48 | 5 | 43/48 | 3 | 7 |
| naive four-camera baseline | 25/48 | 9 | 43/48 | 2 | 7 |
| four-camera + camera/role identity | 31/48 | 7 | 40/48 | 6 | 8 |

The old promotion-gate conclusion is retired. Descriptively, raw demo-target
reproduction rose by six anchors relative to naive four-camera concatenation,
hidden reproduction fell by two, and anchor-extra effective anchors fell from
five to three. It remained four raw target reproductions below eye2 and changed
assist reproduction and anchor-relative outputs in the other direction. These
facts establish different conditional behavior, not a correctness ranking.

The durable conclusion is not that camera roles are irrelevant. The learned
role parameters moved away from zero and the raw anchor decisions changed in a
useful direction, so explicit view identity carries learnable signal. The
failure shows that adding identity to an otherwise unchanged global spatial
concatenation is insufficient to control how eye and stick evidence competes.

### Capability boundary and next valid test

This experiment does not test calibrated geometry, role-specific backbones,
group-level attention, camera dropout, terrain labels, temporal history, or
new deadzone targets. Its failure only rejects additive identity encoding under
the fixed dataset and budget.

A stronger follow-up must change the fusion operation itself: form one eye
token and one stick token with learned within-role pooling, then use a small
cross-role gate or attention block to combine them with qpos. It should include
pair dropout during training so the model cannot solve the task by always
trusting one fixed camera pair. The same state-hold gates and matched split must
remain unchanged.

## Model record: frozen query-0 executable-intent separability

### Intent and capability boundary

This test asks one narrower question before changing ACT training or camera
fusion: does the existing inference-time decoder representation already contain
linearly separable evidence for each axis's executable `neg / idle / pos`
intent? It is not a repeat of the rejected hard factorized-output test. Probe
predictions never enter the continuous ACT action, temporal aggregation,
state-hold command, or runtime gate.

The feature is the 512-dimensional decoder hidden state at query 0 immediately
before `action_head`. Extraction calls DETRVAE with `actions=None`; therefore
the VAE latent is the inference-time zero latent and cannot encode a future
expert action chunk. A scoped forward hook captures the feature without
changing DETRVAE or ACTAdapter behavior. Parameter-and-buffer hashes were
bitwise identical before and after both train and validation extraction for all
three models.

The protocol used the matched 19-train / 5-validation episode split and the
natural validation distribution: 13,288 train frames and 3,241 validation
frames. Labels came directly from the current asymmetric mechanical deadzones
in the `neg, idle, pos` order. The only probe was a fixed 512-to-12 linear head,
reshaped to four independent 3-logit axes and trained for 50 epochs. Class
weights were derived from train counts only as inverse frequency, normalized
over present classes. No validation threshold or epoch tuning was performed.

The exact train / validation counts for `neg, idle, pos` were:

| Axis | Train | Validation |
| --- | ---: | ---: |
| swing | 1,848 / 9,698 / 1,742 | 415 / 2,352 / 474 |
| boom | 902 / 10,863 / 1,523 | 207 / 2,522 / 512 |
| stick | 0 / 13,288 / 0 | 0 / 3,241 / 0 |
| bucket | 2,234 / 8,953 / 2,101 | 612 / 2,141 / 488 |

Stick is non-estimable for active intent under this dataset. Its idle-only
accuracy must not be interpreted as a passed stick result.

### Verified results

| Frozen ACT representation | All-frame active recall | Idle false-active | Macro F1 | Transition anchors | Startup bucket+ | Mid-cycle anchors | Double-axis preserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eye2 baseline | 90.7% | 7.8% | 86.5% | 31/48 | 1/5 | 30/43 | 75.5% |
| naive four-camera | 90.0% | 6.8% | 87.4% | 31/48 | 2/5 | 29/43 | 65.6% |
| four-camera + camera/role identity | 89.9% | 7.9% | 86.1% | 34/48 | 3/5 | 31/43 | 63.0% |

All three probes had zero opposite-direction predictions on validation active
frames and on the 48 transition anchors. All five startup anchors were verified
as bucket-positive before metrics were accepted. The role model's transition
recall was `11/16` boom, `14/21` bucket, and `9/11` swing. Its three recovered
startup episodes were 91, 84, and 92; episodes 94 and 74 were classified idle.
The mean startup confidence/margin was only `0.686 / 0.380`, versus
`0.819 / 0.641` on its mid-cycle anchors.

### Interpretation and limits

The verified fact is that query-0 decoder features are strongly separable on
ordinary active frames and partially separable at transitions. The role model
contains more linearly accessible transition evidence than either matched
baseline, including three of the five startup decisions, even though its
continuous ACT output recovered none of those five in the prior state-hold
test. This supports the inference that part of the failure lies in how the
continuous action head uses an available representation, rather than proving
that the representation is fully sufficient.

The same result also rejects a stronger claim: frozen features do not make
startup solved. Two startup observations remain confidently idle, and both
four-camera representations preserve simultaneous active axes less often than
eye2. This test does not establish a safe argmax action mapping, calibrated
probabilities, terrain generalization, role competition, pair-dropout benefit,
closed-loop recovery, or deployability. The historical rejection of hard
factorized action projection therefore remains valid.

The complete run is
`/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/frozen_intent_probe_query0_linear_20260714`.
Its `experiment.json` SHA-256 is
`200c3f1f6ff315c4015f10637ee19a33c5f04fe0135b6b67925e467028b25faa`;
the directory also contains the exact feature-cache identities, checkpoint and
episode hashes, probe weights, JSON/CSV metrics, anchor predictions, and compact
PNG reviews.

## Current replacement record: G49 new-data evaluation

`evidence_status=valid_with_scope_limitations`, `lifecycle=current`.

The current matched evidence uses the new-data view with 120 train and 20
validation episodes. It includes N0 eye2 continuous, N1 eye2 transition, N2
eye2 effective-action auxiliary, N3 naive four-camera continuous, N4
four-camera additive camera/role identity, and N5 N4-plus-transition models.
The sealed test data was not used.

The accepted current findings are deliberately split by capability:

- N3 is the best continuous expert-path regressor in open-loop validation; this
  is an imitation result only.
- N1 is the strongest frozen-transition responsiveness reference in raw
  state-hold; this is a synthetic-liveness result only.
- Under the descriptive first-candidate readiness view, N1 becomes executable
  before or at the recorded expert onset in 20/20 validation episodes. N0, N2,
  N3, and N4 do so in 11/20, 14/20, 3/20, and 7/20 respectively. These are
  timing observations relative to the recording, not startup success rates.
- Under the stricter armed natural-startup diagnostic, which warms policy state
  on the recorded preparation prefix and then freezes the final
  expert-ineffective observation, N0/N1/N2/N3/N4 emit an effective command on
  any axis within 20 ticks in 12/18/15/3/7 episodes. N1's two remaining episodes
  emitted effective commands only much earlier in the preparation prefix and no
  longer did so at the arm reference. This corrects the optimistic interpretation
  of N1's earlier 20/20 first-candidate number without declaring the early output
  unsafe.
- N5 reaches armed any-axis liveness in 20/20 episodes at zero delay and is also
  deadzone-effective at recording step 0 in all 20. It retains local expert
  single-demo local support in 16/20 armed rows. The other four rows are not
  unseen action
  combinations: all occur in the 120 training episodes, three introduce a
  swing-positive direction used later in the same validation episode, and one
  applies a common training startup motif that is absent from that validation
  episode. The old report said target startup state-hold `5/20`; under the
  current contract this is `5/20 single-demo targets reproduced`, not 15
  deadlocks. Its old `unexpected_effective` field is removed and replaced by
  `anchor_extra_effective`, which cannot be translated into wrong, unsafe, or
  globally unsupported starts. N5 remains the
  counterexample showing that natural liveness needs selectivity and
  idle/release companions, not proof of indiscriminate unsafe activation.
- Among live armed-startup rows, the first direction set is wholly within the
  event's 40-tick single-demo support in 11/12, 17/18, 14/15, 2/3, and 6/7 episodes
  for N0--N4. All five models have zero opposite-to-anchor rows. These remain
  descriptive expert-data similarity facts, not gates.
- N1's first direction set lies wholly within the event's 40-tick single-demo support
  in 17/20 episodes. Exact first-axis matching is intentionally not required;
  this figure describes similarity to the recorded task sequence only.
- N3 and N4 frozen query-0 representations retain linearly accessible startup
  information even when their continuous executed startup output is weak; this
  is a representation result only.
- Disabling temporal aggregation barely changes startup demo-target reproduction and worsens
  MAE/idle false activation; this attributes the observed replay failure mainly
  away from the current aggregation rule, not toward a real-machine cause.
- No N0--N5 candidate has current real command-response, terrain-held-out, or
  controlled closed-loop evidence. None is promoted to live control.

The complete current result and hashes are in
`docs/g49_new_data_first_batch_offline_results_20260715.md`. The G49 frozen
probe artifact is
`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/frozen_intent_probe_formal_train120_val20/experiment.json`,
SHA-256
`15a2755c22d6ab8aa657824d21603c8ce5181b2d6ea980dae1e8015aa751195f`.

The descriptive startup artifact is
`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/expert_intent_eval_n0_n4_val20_h40_v3/expert_intent_eval_report.json`,
SHA-256
`b92b12ab222ccc8ebd77368c2bdc875c17c0f8b11d4dfa474461d339694263cc`.

The armed natural-startup artifacts are the five directories
`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/startup_activation_n{0,1,2,3,4}_val20_h20`.
Their report SHA-256 values in N0--N4 order are
`93b527c0a0fe21e65d7fd1e7749b7c818f8fa2815546862161e194c8cadcb152`,
`007660de13b57fb5dc2baff13913a8f18aa12c4e415f070291f7605a0ce7db37`,
`ce1c2a1923c34190f0b430147149091d0290d2ad0278d94952691d926f9351ea`,
`a3b2e65ac6221dfe70e11f6bfdbb12dba5c20f8fb9d006e95e7f86d6fdea1496`,
and `a88f2ea7dc4695ab5881549146edaa54822e4cc9e45bcb0955f5a165258c763f`.
Each source manifest records raw inference, identity action scale, assist and
runtime gates disabled, no command sent, and no sealed-test read.

The N5 follow-up artifacts are:

- training bundle
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/training_runs/n5_new_fourcam_role_transition_boundary_2000/ckpt`,
  best checkpoint SHA-256
  `0c9b755447f1c06a893394fb1111b9365eb47a8670523b6eeaef8b2df7e13b0e`;
- open-loop summary
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/n5_open_loop_val20/collection_summary.json`,
  SHA-256
  `c399d5bb6a5d8b075e92c77c2ba6f4e59759e55ae12ab45e42afe22c26d7d7e1`;
- descriptive intent report
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/expert_intent_eval_n5_val20_h40_v3/expert_intent_eval_report.json`,
  SHA-256
  `2091c5798681ba86eb88aaee513bcf118b3b2a4c2aca896b54d44339cda71132`;
- armed startup report
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/startup_activation_n5_val20_h20/startup_activation_report.json`,
  SHA-256
  `4c3a634a5fd8153dd1ded105f4f7ed96fbe63ec4fed75720efcd7acff9e2a245`;
- full state-hold summary
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/n5_state_hold_raw_val20_h20/assist_disabled/state_hold_summary.json`,
  SHA-256
  `98b90b4e1709532aa3b172497fbac03248a426d23c62efd574834cfaab56b545`.

This record supersedes only the old probe's dataset-specific startup inventory,
stick-label support, and model ranking. It does not erase the old H3 safety
counterexample or the general state-hold/open-loop interpretation boundaries.

## G49 N5 old-four-camera cross-style probe

`evidence_status=valid_with_scope_limitations`, `lifecycle=current`.

This diagnostic replays N5, without retraining, on all 24 formal old-style
four-camera episodes (`16,529` steps). N5 was trained only on the new-style 120
episode block, but the old data influenced earlier project development and is
therefore a cross-style diagnostic rather than an untouched generalization
set. The sealed old episodes `105..109` remain unread.

The direct old-observation replay shows that N5 does produce command-domain
intent: all 24 episodes eventually cross a deadzone, 17 do so at recording step
0, and 19 do so before or at the old expert's first onset. However, the old
expert starts bucket-positive in 24/24 episodes, while N5's first direction set
is boom-negative in 15, boom-negative plus stick-positive in 6, and one each of
bucket-negative, boom-negative plus bucket-positive, and stick-positive. Among
the 19 pre-or-at-onset candidates, none is wholly inside the first event's
40-tick single-demo support and only one overlaps the anchor. This establishes
low similarity to those recordings, not semantically invalid startup transfer
or absence of intent.

Later task phases transfer better than startup: across 227 old expert events,
anchor-current single-demo direction recall is `0.7413`, exact-set similarity
is `0.4978`, outside-single-demo-event-support rate is `0.2687`, and
opposite-to-single-demo-anchor rate is `0.0176`. These are teacher-forced
single-demo relation metrics; the outside-support value is not task-wide
unsupported behavior and none establishes closed-loop capability.

The fixed-qpos image swap keeps each old episode's step-0 qpos/qvel exactly
fixed and replaces all four old images with the nearest-qpos frame drawn from
N5's 120 new-training episodes. The nearest-frame normalized qpos RMS distance
has median `0.00679` and p95 `0.02205`. The deadzone direction set changes in
21/24 pairs; mean per-axis absolute action change is `0.1974` with p95 `0.2953`.
Old images yield 17/24 effective step-0 outputs, dominated by boom-negative,
whereas matched new images yield 21/24 and are dominated by stick-positive.
This directly establishes image sensitivity under the constructed swap and
rejects a qpos-only explanation. It does not show that either swapped
observation is physically consistent or that N5 understands terrain.

The detailed report is
`docs/g49_n5_old_fourcam_cross_style_probe_20260715.md`. Immutable artifacts:

- old-style open-loop collection summary
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/n5_old_fourcam_cross_style_24_open_loop/collection_summary.json`,
  SHA-256 `99056d1b00ef8e3e7db903ebc08351631da06403dec077d95cbd118b6b10bca6`;
- expert-intent report
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/n5_old_fourcam_cross_style_24_intent_eval_h40/expert_intent_eval_report.json`,
  SHA-256 `aa4ccd600b8fe5af9888a49a4b608cb4f1bad9c8c114a4311e73c5b0f93ce9d8`;
- fixed-qpos image-swap report
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/n5_old_fourcam_fixed_qpos_image_swap_step0/cross_style_image_swap_report.json`,
  SHA-256 `487a2dc45f7b66f0559b154e69d94a7a5fc05f8243f292776ecacae87833417f`.
