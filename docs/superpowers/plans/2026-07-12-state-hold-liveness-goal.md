# State-Hold Liveness Closed-Loop Goal

Date: 2026-07-12

Status: blocked on new on-policy execution-response data and controlled field
collection authorization

## [Goal: State-Hold Liveness] Objective

Find and verify the best replacement for the E52 suppressive gate under the
direct-policy-output action contract. The selected candidate must improve on
the current raw ACT plus mechanical-assist baseline without trading liveness
for wrong motion, tail motion, or unsafe gohome behavior.

## [Goal: State-Hold Liveness] Target lock

- Worktree: `/home/pingfan/Excavator_real_stack_e52_deadlock_eval`
- Branch: `codex/e52-offline-deadlock-gate`
- Base HEAD: `0fab67eda7e449b70622b65afc7ada01142f56e5`
- Training data: train-ready episodes from `72..104` only
- Held-out decision set: `episode_105..109`; never use these episodes for
  training, fine-tuning, calibration fitting, or threshold selection
- Runtime action scale for policy output: identity `[1, 1, 1, 1]`
- Deadzone source: direct mechanical calibration from the field config
- No Jetson access, live control, checked-in runtime-default promotion, or
  field deployment without explicit user authorization

### [Goal: State-Hold Liveness] Human-review heading rule

Every executor prompt, experiment note, callback audit, reflection entry, and
user-facing phase update must identify its current objective in the heading:

```text
[Execution target G<slice>/H<hypothesis>: <bounded objective title>]
```

Goal-wide sections use `[Goal: State-Hold Liveness]`. Do not use ambiguous
headings such as `current status`, `next step`, or `results` without the target
prefix. Preserve the same execution-target title from dispatch through callback
audit so a human reviewer can search and align the full slice.

## [Goal: State-Hold Liveness] Confirmed reference base

The current 45-anchor held-out state-hold matrix is the decision baseline:

| Pipeline | Recovered | Deadlocked | Startup | Hidden by teacher forcing |
| --- | ---: | ---: | ---: | ---: |
| raw ACT | 33 / 45 | 12 / 45 | 5 / 5 | 3 |
| raw ACT + mechanical assist | 40 / 45 | 5 / 45 | 5 / 5 | 0 |
| E52 gate | 15 / 45 | 30 / 45 | 0 / 5 | 22 |
| E52 gate + mechanical assist | 35 / 45 | 10 / 45 | 5 / 5 | 5 |

Additional confirmed facts:

- E52 converts 18 raw-ACT recovered anchors into deadlocks and recovers no
  raw-ACT deadlock.
- All 18 induced failures have an inactive phase gate; snap never activates.
- Fourteen of the 18 induced failures are bucket transitions.
- All 20 held-out bucket transition anchors are below the legacy raw-policy
  bucket thresholds that were adjusted for the removed `0.75` action scale.
- Raw ACT plus assist leaves five intrinsic held-out failures; their best raw
  actions are only about `1.7%..25.7%` of the direct mechanical threshold.

## [Goal: State-Hold Liveness] Acceptance contract

A candidate may replace the reference baseline only if all of the following are
reported from fresh artifacts:

1. Held-out state-hold liveness is greater than `40 / 45`; the target is
   `45 / 45`.
2. Startup remains `5 / 5`.
3. `raw recovered -> candidate deadlock` is exactly zero.
4. Teacher-forced-hidden deadlocks are exactly zero.
5. Wrong/extra effective motion and tail effective motion do not regress beyond
   the accepted raw-ACT-plus-assist reference.
6. Gohome pre-tail false positives do not regress.
7. Action-domain provenance, checkpoint, config, split, deadzone table, and
   per-anchor traces are preserved.

MAE is a secondary metric. A lower MAE cannot compensate for a liveness or
safety-contract failure.

## [Goal: State-Hold Liveness] Hypothesis ladder

Explore one semantic layer at a time in this order. A later branch may start
only after the previous branch has a factual callback and planner audit.

### [Execution target H1: Direct-Action-Domain Relabeling]

Keep the E16 architecture and training procedure fixed. Replace only the legacy
raw-scaled deadzone labels with the direct mechanical deadzone contract. Verify
the split, run a training smoke, train the bounded candidate, and evaluate raw
and assist variants on the held-out state-hold matrix.

### [Execution target H2: Transition State-Hold Training Objective]

Starting from the best verified H1/reference checkpoint, add transition-anchor
sampling, frozen-observation augmentation, and a same-axis/same-direction
sequence liveness margin. Keep stop/tail losses separate so the new objective
cannot pass by suppressing all motion.

### [Execution target H3: Factorized Intent and Effective-Effort Output]

Predict per-axis `negative / idle / positive` intent plus conditional effort,
then reconstruct direct joystick output outside the calibrated deadzone. Apply
temporal aggregation before the final deadzone projection rather than averaging
raw joystick actions across the discontinuity.

### [Execution target H4: Execution-Feedback Input]

Add only verified signals needed to observe failed actuation: qvel or qpos
delta, previous commanded action, and a bounded same-direction stalled-duration
feature. Train with state-hold augmentation and preserve the low-dimensional
contract explicitly.

### [Execution target H5: Monotonic Liveness Governor]

If learned policies still leave intrinsic deadlocks, add a focused module that
may promote persistent, intent-supported ineffective motion but may not
attenuate an already effective raw action below the mechanical deadzone unless
an independently verified safety veto is active. Keep gohome ownership
separate.

## [Goal: State-Hold Liveness] Backtracking and recovery rules

- Every slice starts from the latest accepted checkpoint or the raw-ACT-plus-
  assist reference. Experimental outputs never silently become defaults.
- A failed smoke, training failure, split leak, missing artifact, metric
  regression, or induced deadlock is a factual callback, not a reason to widen
  the current slice.
- On failure, record the exact config, checkpoint, command, artifact paths, and
  failed acceptance clauses; then restore the latest accepted reference for the
  next independent hypothesis.
- Do not tune on `105..109`. If a threshold or weight decision needs those
  episodes, reject that decision path and derive it from training-only folds.
- After three accepted callbacks, or immediately after a failed/misaligned
  callback, run a deep reflection against this reference base and change branch
  order if the evidence warrants it.
- Stop when one candidate satisfies the acceptance contract with reproducible
  artifacts, or when every independent hypothesis is factually exhausted and a
  new user decision or data collection is required.

## [Goal: State-Hold Liveness] Experiment ledger

| Slice | Hypothesis | Status | Reference | Result | Next action |
| --- | --- | --- | --- | --- | --- |
| G1 | H1 direct-action-domain relabeling | accepted | E16 one-epoch smoke | exact-diff, split-disjoint, CUDA smoke, and artifact audits passed; manifest `09dd094...c197344` | launch bounded formal H1 training |
| G2 | H1 direct-action-domain relabeling | failed | accepted G1 smoke | formal config/split audits passed, but detached `nohup` PID exited before first check with an empty log; manifest `8b967c1...e49710` | recover launch through an owned unified exec session |
| G2R | H1 direct-action-domain relabeling | accepted | accepted G1 smoke and audited G2 config | persistent exec session `1575` completed `2000/2000`; metadata status `completed`; best epoch `1999`, val loss `0.0688679572`; final artifacts and hashes audited | evaluate H1 raw and assist variants on held-out `105..109` |
| G3 | H1 raw/no-gate evaluation adapter | accepted | audited state-hold diagnostic and `PolicyActionSource` | explicit raw pipeline strips runtime gates, forces identity scale/control/no fail-safe, preserves mechanical assist A/B and provenance; `32` focused tests pass, ruff passes | run full 45-anchor held-out state-hold A/B |
| G4 | H1 held-out state-hold evaluation | rejected | raw ACT + assist `40/45` | H1 raw `32/45`, startup `3/5`, hidden `3`; H1+assist `39/45`, startup `5/5`, hidden `1`; strict comparisons show raw induced/recovered `4/3`, assist induced/recovered `1/0` | backtrack to raw ACT + assist and start H2 |
| G4C | reusable state-hold comparison | accepted | matched 45-anchor JSONL reports | strict key/invariant validation, four transition classes, CSV/JSON/provenance; `12` tests and ruff pass | reuse for H2-H5 candidate audits |
| G5 | H1 acceptance audit and backtrack | accepted | goal acceptance contract | H1 fails clauses 1 and 4 and worsens the accepted assist reference; continuous safety replay is skipped because liveness already rejects promotion | preserve H1 as negative evidence; make raw ACT + assist the H2 reference |
| G6 | H2 transition state-hold objective | accepted | raw ACT + assist `40/45` | focused transition sampler, full-horizon tail bound, dataset mask, exact held ACT aggregation loss, and model-only initialization implemented; `78` relevant tests pass | run bounded smoke and fine-tune |
| G7S | H2 one-epoch integration smoke | accepted | E16 raw action checkpoint | model-only init with fresh optimizer; exact split; non-zero validation held-sequence loss; metadata completed; best epoch `0` | run fixed 200-epoch training-only fine-tune |
| G7F | H2 200-epoch fine-tune | accepted | E16 raw action checkpoint, SHA `0f2515...ec8f` | completed `200/200`; best epoch `110`, val loss `0.160300`; endpoint epoch `199`; hashes and metadata audited | compare best and endpoint on validation state-hold |
| G7V | H2 validation checkpoint selection | accepted | E16 validation assist `44/48` | both H2 checkpoints reach assist `45/48`, induce `0`, recover `1`, startup `5/5`; best epoch110 selected by lower val loss and shorter raw worst delay | freeze epoch110 and run held-out once |
| G8 | H2 held-out state-hold evaluation | rejected | held-out raw ACT + assist `40/45` | H2 raw `39/45`, induced `0`, recovered `6`, hidden `0`; H2+assist remains exactly `40/45`, same five deadlocks, induced/recovered `0/0`, startup `5/5` | reject H2 for promotion; backtrack and start H3 |
| G9 | H2 acceptance audit and backtrack | accepted | goal acceptance contract | H2 improves raw liveness but does not strictly beat the accepted assist reference; safety replay skipped | preserve H2 as training evidence; H3 must address direction/effort observability |
| G10 | H3 factorized intent plus effort | accepted | raw ACT + assist `40/45` | adversarial review plus executable mathematical/safety contract | focused owner; no new model head; hard-argmax remains falsification-only |
| G11 | H3 focused implementation and smoke | accepted | locked G10 contract | `84` focused tests, E16 strict model-only init, one-epoch `72..104` smoke, episode97 inference | proceed with one fixed formal candidate; retain `15.83%` recorded-future conflict as release-risk diagnostic |
| G12 | H3 fixed candidate and old-validation audit | rejected | E16 assist `44/48`; H2 assist `45/48` | best145 and endpoint199 both `45/48`; best hidden `0`, endpoint hidden `1`; H3 adds unexpected/opposite crossings | reject hard-argmax H3 without held-out access or tuning; backtrack to H2/assist |
| G13 | H4 causal execution-feedback contract and smoke | accepted | H2 best epoch110 | causal raw-send sidecars, train-only feedback normalization, strict 4D-to-12D zero-column expansion, recursive evaluator, and counterfactual branch all passed focused tests and smoke | run exactly one fixed 200-epoch H4 candidate |
| G14 | H4 fixed training and old-validation gate | rejected | H2 assist `45/48` | best40 raw/assist `41/48`,`45/48`; endpoint199 `40/48`,`45/48`; best induces `episode_92:249`, worsens unexpected motion `4/8 -> 7/29`, and fails locked `46/48` gate | reject without held-out access; backtrack to H2+assist and start H5 |
| G15A | H5 frozen train-q01 confidence governor | rejected | H2 best + mechanical assist | old-val `46/48`, but hidden `1` and a new wrong swing promotion at `episode_92:249`; wrong evidence dominates desired recovery on every scalar score | do not scan scalar thresholds; audit a structurally independent active-axis-capacity constraint |
| G15B | H5 capacity-constrained cross-head consensus | rejected | H2 best + mechanical assist | train-only replay promotes 1570/13288 ticks but only 11 match expert effective directions (`0.70%` precision); 649 occur in expert-zero frames and 153 after the last effective frame | reject before implementation/old-val; audit frozen temporal-direction evidence once |
| G15C | H5 frozen E52 temporal-direction eligibility | rejected | H2 best + assist; E52 temporal artifact | final model trained all 24 including old val and uses legacy scaled labels; OOF boom+ probabilities on all three intrinsic old failures are only `.0070/.0174/.0352` vs frozen `.5` | terminate governor route; current data lacks a safe positive direction signal |
| G16 | final closure and data requirement | in progress | all H1-H5 callbacks | verify code/artifacts and document why offline demos cannot identify failed-actuation retries safely | require new on-policy execution-response data before another learned candidate |

### [Execution target G1/H1: Direct-Relabel Smoke] Callback audit and reflection

- Accepted callback: request-local artifacts only; no checked-in runtime/config
  change and no held-out leakage.
- Planner audit: verified the live manifest structure, direct mechanical
  thresholds, exact four-path config diff, 24-episode split proof, completed
  one-epoch metadata, checkpoint epochs, and artifact hashes.
- Alignment: the slice advanced H1 without changing another semantic layer.
- Efficiency: the smoke was necessary configuration-risk reduction; it did not
  duplicate a held-out evaluation or claim model improvement.
- Current accepted reference remains raw ACT plus assist at `40 / 45`; the H1
  smoke is not a promoted policy candidate.

### [Execution target G2/H1: Direct-Relabel Formal Training Launch] Failure audit and deep reflection

- Failure boundary: formal E16 source completion, G1 integrity, exact config
  diff, direct deadzone provenance, and held-out split isolation all passed.
- Failure fact: the detached `nohup` target PID disappeared before the first
  liveness check, with a zero-byte log and no training artifacts.
- Planner audit: no differently owned training process survived and no CUDA
  process was present; the hypothesis and training configuration were not
  exercised.
- Deep-reflection verdict: this is an execution-environment ownership failure,
  not negative H1 model evidence. Do not backtrack to H2 yet.
- Recovery: reuse the already audited request-local config and split, but keep
  the training command as the foreground process of a persistent unified exec
  session so process lifetime and output remain observable.
- Efficiency: retrying the same audited config through a different process
  owner removes the actual blocker; regenerating configs or changing model
  semantics would be drift.

### [Execution target G2R/H1: Persistent Direct-Relabel Formal Training] Callback audit and reflection

- Accepted callback: the foreground persistent training session completed all
  `2000` epochs in about `25m32s`; `run_metadata.json` records status
  `completed` and completion time `2026-07-11T17:54:07.668897`.
- Training result: best epoch `1999`, best validation loss
  `0.06886795721948147`. This is a training-integrity result only, not evidence
  that H1 improves closed-loop liveness.
- Artifact root:
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_hold_liveness_20260712/h1_direct_relabel_formal/ckpt`.
- Audited hashes: `policy_best.ckpt`
  `b6848de3331788b43ccf81d2081ff70876305e05a51e67b0242bc8a09a4c5ec1`;
  `policy_latest.ckpt`
  `7518f23f3ad191392ca7410d5c6810c9d82bf8a686867d6a5ccf0c19e819b6d1`;
  `dataset_stats.pkl`
  `973f5f72a7011e3920119156481ab39f3851dd9f14faefbbbce9b443f3f78a82`;
  `resolved_config.yaml`
  `e1f36a8694a6980940a6dee13e8c8ae6e2550adcc86829cd7005ba43ca557c15`.
- Planner audit: the completion preserved the audited `72..104` split and did
  not read or train on held-out `105..109`. No runtime/default config or Jetson
  state changed.
- Reflection verdict: H1 is now ready for falsification. The accepted policy
  reference remains raw ACT plus assist at `40/45` until fresh state-hold and
  safety artifacts show that H1 satisfies the acceptance contract.

### [Execution target G3/H1: Raw/Assist Held-Out Evaluation Adapters] Callback audit and reflection

- Boundary decision: keep raw/gated selection as thin orchestration in the
  existing state-hold CLI; keep inference, assist state, and action semantics in
  `PolicyActionSource`.
- Added an explicit `pipeline_mode=raw` contract. It requires no candidate
  manifest, replaces any configured runtime gate with `enabled: false`, forces
  direct identity action scale, control output, and exception visibility, and
  retains the field mechanical-assist parameters unchanged.
- Backward behavior lock: the default remains gated E52 and still requires and
  verifies the E52 candidate manifest and artifacts.
- Verification: `32` focused policy-action/state-hold/CLI tests passed; ruff and
  Python compilation passed. Lazy HDF5 access loaded both configured cameras
  for every held-out episode.
- Efficiency decision: do not implement the continuous assist safety-replay
  adapter until H1 first exceeds the `40/45` liveness gate. A liveness failure
  would reject H1 before those safety metrics can promote it.

### [Execution target G4/H1: Full Held-Out State-Hold A/B] Launch contract

- Checkpoint: formal H1 `policy_best.ckpt`, best epoch `1999`, SHA-256
  `b6848de3331788b43ccf81d2081ff70876305e05a51e67b0242bc8a09a4c5ec1`.
- Evaluation-only data: all `3368` steps and all `45` transition anchors from
  `episode_105..109`; no training, calibration fitting, or threshold selection.
- Pipeline: raw/no gate, identity action scale, direct mechanical deadzones,
  `qvel_mode=raw`, state-hold horizon `20`, assist disabled and enabled.
- Artifact root:
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/runs/goal_state_hold_liveness_20260712/h1_direct_relabel_formal_state_hold_all105_109_direct_h20`.
- Estimated wall time from the reference run: approximately `20..22` minutes.

### [Execution target G4/H1: Full Held-Out State-Hold A/B] Callback audit and rejection

- Terminal status: exit `0`; both modes emitted exactly `45` anchor rows and
  preserved explicit raw/no-gate, identity-scale, direct-deadzone provenance.
- H1 raw: `32/45` recovered, `13/45` deadlocked, startup `3/5`, and `3`
  teacher-forcing-hidden deadlocks. The raw ACT reference is `33/45`, startup
  `5/5`, hidden `3`.
- H1 plus assist: `39/45` recovered, `6/45` deadlocked, startup `5/5`, and `1`
  teacher-forcing-hidden deadlock. The accepted raw ACT plus assist reference is
  `40/45`, startup `5/5`, hidden `0`.
- Strict raw-to-raw comparison: H1 induced four deadlocks and recovered three
  reference deadlocks. Induced anchors are `episode_105:644 swing-`,
  `episode_106:92 bucket+`, `episode_107:40 bucket+`, and
  `episode_109:461 swing-`.
- Strict assist-to-assist comparison: H1 induced one new deadlock at
  `episode_109:275 swing+` and recovered no reference deadlock.
- Strict raw-reference to H1-plus-assist comparison reports zero induced
  deadlocks and six recovered raw deadlocks, but this does not override the
  failure to beat the accepted assist reference.
- All six H1-plus-assist deadlocks remained below the target-axis assist trigger
  for all held ticks. Their best signed raw actions were only about
  `0.34%..29.42%` of the mechanical deadzone (`0.69%..58.84%` of the half-
  deadzone trigger target).
- Artifact root:
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/runs/goal_state_hold_liveness_20260712/h1_direct_relabel_formal_state_hold_all105_109_direct_h20`.

### [Execution target G4C/H1: Reusable State-Hold Candidate Comparator] Callback audit and reflection

- Added a focused comparison owner and thin CLI with strict duplicate, anchor
  key-set, semantic invariant, status, delay, and teacher-forcing validation.
- Outputs preserve source paths and SHA-256 hashes and write non-overwriting
  per-anchor CSV/JSON plus aggregate JSON.
- Verification: `12` focused tests, ruff, format, compilation, and CLI help
  passed. The three H1 comparisons are under the G4 artifact root's
  `comparisons/` directory.
- Reflection: this closes the earlier ad-hoc comparison gap and is reusable for
  every later hypothesis without changing evaluation semantics.

### [Execution target G5/H1: Acceptance Audit and Backtrack] Deep reflection

- Verdict: reject H1. It fails the required `>40/45` liveness clause, produces
  a teacher-forcing-hidden assist deadlock, and worsens the accepted assist
  reference. No ordinary safety replay can promote a candidate that already
  fails the primary liveness gate, so that work is deliberately skipped.
- Training-only census: the `72..104` training split contains `180` transition
  anchors in `13288` steps (`1.3546%`). Uniform start sampling yields only
  `0.2594` transition-start samples per full epoch across 19 training episodes,
  or approximately `2.9` exact-anchor exposures over 2000 epochs.
- Causal interpretation: H1 changed the threshold source for an auxiliary
  intent head, while the action head remained dominated by ordinary chunk L1.
  It did not make the frozen observation or the held temporal action sequence a
  supervised event. The negative result therefore falsifies direct relabeling
  alone, not the broader use of calibrated direct-output thresholds.
- Backtrack point: raw ACT plus mechanical assist at `40/45`, startup `5/5`,
  hidden `0`. H2 must improve from this reference and may not silently inherit
  H1 as an accepted checkpoint.
- H2 requirement: oversample transition starts from training data only and add
  a same-axis/same-direction loss on the held ACT temporal sequence, with stop
  and wrong-motion behavior evaluated separately. Do not repeat E56's generic
  per-query same-direction promotion, which previously reduced startup/main
  coverage under teacher-forced evaluation.

### [Execution target G6/H2: Transition Oversampling and Held-Sequence Objective] Callback audit and reflection

- Boundary split: a new pure NumPy data owner resolves transition-sampling
  config, derives direction-specific inactive-to-effective anchors from the
  existing deadzone-label semantics, intersects valid starts, enforces the full
  hold-horizon tail bound, and samples with an explicit probability. Dataset
  code only wires this owner into HDF5 loading.
- ACT adapter remains the training-loss owner. It reconstructs the exact
  repeated-observation temporal-aggregation prefixes using the same `k=0.01`
  source as inference, then penalizes the last two same-direction prefixes when
  they remain below `0.5 * mechanical_deadzone + 0.02`.
- This objective is assist-aware by design: the learned action must express
  stable intent strongly enough for the existing two-tick mechanical assist to
  cross the calibrated deadzone. It does not promote every effective query and
  does not add stop/wrong suppression to the same loss.
- Added model-only `init_ckpt` semantics so H2 starts from the accepted E16
  weights with a fresh optimizer, epoch zero, and a fresh best-validation
  baseline rather than inheriting optimizer/min-loss state.
- Training-only census recorded `179` full-horizon eligible transition starts
  across the 24 episodes; one of the earlier `180` raw anchors is excluded by
  the explicit 20-step tail boundary.
- Verification: `78` relevant dataset, label, sampler, ACT-loss, checkpoint,
  and runtime-action tests passed; compilation and diff checks passed. Ruff
  passes for all new focused files and passes modified legacy owners after
  excluding their independently verified pre-existing lint findings.

### [Execution target G7S/H2: One-Epoch Integration Smoke] Callback audit and reflection

- Smoke root:
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_hold_liveness_20260712/h2_transition_state_hold_smoke`.
- Status `completed`; initialized model weights from accepted E16 checkpoint
  SHA-256 `0f2515cdaaad4a541b542d9423508b0885cada029feaaa1bd2957a3dbf87ec8f`
  while leaving optimizer state behind.
- Split remained train 19 / validation 5 with no `105..109` overlap. Dataset
  stats SHA-256 remained
  `973f5f72a7011e3920119156481ab39f3851dd9f14faefbbbce9b443f3f78a82`.
- Validation emitted non-zero `state_hold_pos_shortfall_loss=0.0179` and
  historical field now named `demo_target_hold_loss=0.0018`, proving the full config-to-mask-to-loss
  chain is active. A zero aggregate training shortfall in this single random
  epoch is acceptable because sampled anchors may already satisfy the target;
  unit gradient tests separately prove failed anchors receive the correct sign.
- Verdict: accept integration, not policy quality. Proceed to the predeclared
  200-epoch fine-tune without changing probability, weight, margin, horizon, or
  assist trigger semantics.

### [Execution target G7F/H2: Fixed 200-Epoch Training-Only Fine-Tune] Launch contract

- Config root:
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_hold_liveness_20260712/h2_transition_state_hold_finetune_e16_200`.
- Formal config SHA-256:
  `6adf8f0fdfda9867375a0171e28a63fbaf3b45051b1a31bd5c12476e2fbe6a25`;
  direct deadzone SHA-256:
  `3a892aeee4f75ef903b93e7afe201211ebaad48d97c552756b8cdf65abf10867`.
- Exact smoke semantics are locked: transition probability `0.5`, hold horizon
  `20`, loss weight `0.1`, assist trigger fraction `0.5`, margin `0.02`, and two
  consecutive terminal prefixes. Only task/output paths, epochs, and artifact
  cadence differ from the accepted smoke.
- The fixed endpoint and best checkpoint will first be tested on validation
  episodes `94, 91, 84, 74, 92`. Held-out `105..109` remains untouched until a
  validation-selected checkpoint shows a credible liveness improvement.

### [Execution target G7F/H2: Fixed 200-Epoch Training-Only Fine-Tune] Callback audit and reflection

- Terminal status `completed`; `200/200` epochs finished in about `1m17s`.
  Best epoch `110`, best validation loss `0.16029975563287735`; fixed endpoint
  epoch `199`.
- Best checkpoint SHA-256:
  `689961b492d8b38a9a7688663c8a2fe3ca5ac792062560aefee3e151f8495135`;
  endpoint SHA-256:
  `f5d25eb3f54afaeac7053089a5201bb972ed8a3bd4cf0606e0a021665f9f38ea`.
- Dataset stats remain byte-identical to E16/H1. Run metadata preserves the
  exact 24-episode eligible-start census and contains no held-out episode.
- The ordinary validation loss is noisy because each episode supplies a random
  transition/uniform start. It is therefore only a checkpoint proposal signal;
  selection requires the deterministic full validation state-hold matrix.

### [Execution target G7V/H2: Old-Batch Validation State-Hold Selection] Callback audit and reflection

- Decision set: validation episodes `94, 91, 84, 74, 92`, `48` anchors,
  direct mechanical thresholds, raw/no gate, horizon `20`, both assist modes.
- E16 reference: raw `36/48`, hidden `4`; assist `44/48`, hidden `1`; startup
  `5/5` in both modes.
- H2 best epoch110: raw `40/48`, hidden `1`; assist `45/48`, hidden `1`;
  startup `5/5`.
- H2 endpoint epoch199: raw `40/48`, hidden `0`; assist `45/48`, hidden `1`;
  startup `5/5`.
- Strict assist comparisons are identical for best and endpoint: zero induced
  deadlocks and one recovered E16 deadlock at `episode_92:249 bucket-`.
- Both raw comparisons recover five and induce one at
  `episode_91:187 boom+`; because the selected pipeline includes mechanical
  assist, that raw-only trade remains diagnostic but cannot be ignored in the
  final safety audit.
- Checkpoint selection: freeze best epoch110 before held-out access. It has the
  lower training validation loss and a shorter raw worst recovery delay
  (`9` ticks versus endpoint `17`) while matching endpoint assist liveness.
- Validation artifact root:
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_hold_liveness_20260712/h2_transition_state_hold_finetune_e16_200/validation_state_hold`.

### [Execution target G8/H2: Frozen Epoch110 Held-Out State-Hold A/B] Launch contract

- Frozen checkpoint: H2 best epoch110, SHA-256
  `689961b492d8b38a9a7688663c8a2fe3ca5ac792062560aefee3e151f8495135`.
- No H2 weight, sampling probability, threshold, margin, horizon, assist, or
  checkpoint choice changed after the old-batch validation result.
- Decision set: held-out `105..109`, all `45` anchors, raw/no gate, identity
  action scale, direct mechanical deadzones, horizon `20`, assist disabled and
  enabled.
- Promotion still requires `>40/45`, startup `5/5`, zero induced deadlocks,
  zero hidden deadlocks, followed by ordinary wrong/extra, tail, and gohome
  safety checks. A liveness failure immediately backtracks without safety
  replay.

### [Execution target G8/H2: Frozen Epoch110 Held-Out State-Hold A/B] Callback audit and rejection

- Terminal status: exit `0`; both reports contain exactly `45` anchors and the
  frozen checkpoint/direct-output provenance.
- H2 raw: `39/45`, startup `5/5`, hidden `0`. Relative to raw E16 `33/45`,
  strict comparison shows zero induced deadlocks and six recovered deadlocks.
- H2 plus assist: `40/45`, startup `5/5`, hidden `0`. Relative to the accepted
  raw ACT plus assist reference, the strict comparison is exactly unchanged:
  zero induced, zero recovered, and the same five deadlocked anchors.
- The five remaining anchors are `episode_105:99 bucket+`,
  `episode_105:409 swing+`, `episode_106:348 boom+`,
  `episode_107:306 bucket-`, and `episode_107:508 boom+`.
- On those five anchors, H2's best signed raw action reaches only about
  `-1.30%..23.75%` of the mechanical threshold. The bucket-negative anchor is
  wrong-sign for every held tick; the other four remain far below the
  half-deadzone assist trigger.
- Artifact root:
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/runs/goal_state_hold_liveness_20260712/h2_best_epoch110_state_hold_all105_109_direct_h20`.

### [Execution target G9/H2: Acceptance Audit and Backtrack] Deep reflection

- Verdict: reject H2 for promotion because `40/45` does not satisfy the strict
  `>40/45` contract. Do not run ordinary safety replay and do not tune H2
  probability, loss weight, or margin after observing held-out failures.
- Positive evidence: explicit transition oversampling and held-sequence loss
  materially improved the underlying raw policy (`+6` recoveries, no induced
  or hidden deadlocks). This validates the offline diagnostic and the training
  direction, even though the existing assist had already recovered those six
  easier cases.
- Remaining failure structure: four anchors express very weak same-direction
  effort and one expresses the wrong direction under a frozen observation.
  A continuous action regression head plus temporal averaging still treats
  direction confidence and effort magnitude as one quantity, so uncertain
  predictions collapse toward zero or cross sign.
- Backtrack point remains raw ACT plus mechanical assist at `40/45`. H3 is an
  independent structural hypothesis: factor direction/idle intent from
  conditional effort and aggregate intent/effort before the discontinuous
  deadzone projection.

### [Execution target G10R/H3: Factorized Contract Adversarial Review] Callback audit and reflection

- Read-only adversarial review accepts reuse of the existing eight intent
  logits but rejects the initial incomplete wording. Existing signed action L1
  is not conditional effort, query-zero intent is not temporally aggregated,
  the legacy zero-sentinel buffer cannot represent valid idle predictions, and
  held persistence has no automatic release guarantee.
- Historical E16/E20/E42/E43/E47 evidence makes false motion the symmetric risk
  to E52 deadlock: a small probability advantage under hard selection can
  become a full `deadzone + 0.02` action. Therefore target-axis recovery alone
  cannot promote H3; wrong/opposite/extra crossing, multi-axis motion, flips,
  and release/tail behavior are hard rejection criteria.
- Responsibility boundary: a new focused `factorized_action` owner will hold
  config validation, tri-state semantics, magnitude loss, temporal probability
  and effort aggregation, projection, and diagnostics. The ACT adapter remains
  responsible only for model forward/lifecycle and calls that owner; runtime
  config wiring remains opt-in. No DETR-VAE head shape changes are required.
- Leakage boundary remains unchanged: class census and all derived constants
  may use only the 19 training IDs from `72..104`; the five old validation IDs
  may only select between the predeclared best-loss and epoch199 checkpoints;
  `105..109` is inaccessible until the old-validation acceptance gate passes.

### [Execution target G10/H3: Factorized Intent + Conditional Effort Contract] Locked single candidate

- Per-axis logits are exactly
  `[intent_neg, fixed_idle_zero, intent_pos]`, where the existing checkpoint
  layout is axis-major `[pos, neg]`. Softmax makes direction and idle mutually
  exclusive; non-finite values are errors and exact winner ties select idle.
- Direct-output expert labels use the calibrated asymmetric mechanical
  deadzones: `action <= -neg` is negative, `action >= pos` is positive, and the
  open interval is idle. All thresholds must be strictly positive; padding is
  excluded. The training split has no effective stick-direction labels, so H3
  makes no claim that it learns stick motion.
- The action head is reinterpreted and trained as magnitude: first unnormalize
  both prediction and expert into the direct policy-output domain, then compare
  their absolute values. Signed action L1 is replaced, not retained; direction
  supervision comes only from tri-state CE.
- At time `t`, every populated historical ACT chunk that covers `t` contributes
  its per-query tri-state probabilities and non-negative direct magnitude under
  the frozen oldest-to-newest `exp(-0.01 i)` weights. Occupancy is explicit,
  probabilities and effort are aggregated separately, and projection occurs
  once after aggregation. `temporal_agg=true` and chunk/horizon `20` are part of
  the executable contract.
- Selection is one fixed `strict_argmax_idle_on_tie` falsification candidate;
  no probability threshold, hysteresis, margin, or checkpoint sweep is allowed.
  Projection is `idle -> 0`, `pos -> +clip(max(effort, pos_deadzone+0.02))`,
  `neg -> -clip(max(effort, neg_deadzone+0.02))`, with clip `1.0` and identity
  external action scale.
- H2 transition sampling is reused at probability `0.5` with full horizon `20`.
  H2 continuous shortfall loss is disabled. Held-prefix NLL targets the anchor
  direction only at delays `18,19`, using the exact inference aggregation; its
  weight is the previously frozen `0.1`.
- The mutually exclusive CE replaces the E16 BCE objective. The single
  candidate inherits classification weight `0.05` and fixed class weights
  `[8,1,8]` from the prior E16 imbalance contract. Magnitude L1 weight is `1.0`.
  These values are frozen before training and are not calibrated on validation
  or held-out data.
- Training diagnostics must expose label counts, class/magnitude/held losses,
  and recorded-future versus held-target conflict. Inference diagnostics must
  expose aggregated `4x3` probabilities, selected class, winner margin,
  aggregated effort, projection floors/output, source count/query ages/weights,
  and legacy signed aggregate as a non-executed comparison.
- G11 first proves order, tie, asymmetric projection, unnormalize-before-abs,
  held-prefix parity, explicit zero occupancy, reset, single projection,
  gradient direction, and strict E16 checkpoint loading. A one-epoch smoke must
  produce finite nonzero CE, magnitude, and held losses with a fresh optimizer.
- G12 trains exactly one fixed 200-epoch candidate from accepted E16 weights.
  Old-validation advancement requires H3-alone at least `46/48`, startup `5/5`,
  zero reference-recovered to candidate-deadlock, zero hidden deadlock, no new
  wrong/opposite/extra effective crossing before recovery, zero tail effective
  frames, and H3 output invariant to mechanical-assist enablement. Failure of
  both predeclared checkpoints rejects hard-argmax H3 without post-hoc tuning.

### [Execution target G11/H3: Focused Implementation and One-Epoch Smoke] Callback audit and reflection

- Added the focused `factorized_action` owner. It validates the opt-in contract,
  maps checkpoint logits to tri-state probabilities, replaces signed L1 with
  unnormalize-before-absolute magnitude L1, computes weighted CE and exact held
  prefix NLL, maintains an explicit-occupancy temporal buffer, aggregates
  probability/effort separately, performs one final projection, and emits the
  locked provenance/selection diagnostics. The generic ACT model is unchanged.
- ACT wiring forces `intent_dim=8` for H3 even with legacy BCE disabled, rejects
  simultaneous signed/BCE auxiliary objectives, forbids H3 from the runtime-gate
  intent API, requires temporal aggregation for prediction, and leaves all
  existing behavior unchanged while H3 is disabled. Policy action diagnostics
  now carry the H3 trace only when the opt-in adapter exposes it.
- Focused verification passed `84` tests covering class order, asymmetric
  labels/projection, exact-tie idle, non-finite failure, direct magnitude
  semantics, CE/held gradients, held aggregation parity, zero-effort occupancy,
  reset/growth, single projection, H2 sampler, checkpoint init, and policy action
  integration. Ruff and compilation checks pass for the focused owners.
- Training-only census is unchanged at `13288` steps and counts
  swing `1848/9698/1742`, boom `902/10863/1523`, stick `0/13288/0`, and bucket
  `2234/8953/2101` in `[neg,idle,pos]` order. The `179` eligible horizon-20
  transitions create `360` delay-18/19 target slots; `57` (`15.83%`) have
  recorded idle future labels and none are opposite. This is not used to tune
  the frozen loss but is retained as an explicit persistence/release risk.
- One-epoch smoke completed in about four training seconds and strictly loaded
  the accepted E16 model-only checkpoint. Validation emitted finite nonzero
  magnitude `0.3301`, class NLL `0.1407`, and held NLL `0.2270`; training emitted
  `0.1162`, `0.1486`, and `0.2013`. Legacy intent and H2 shortfall losses stayed
  exactly zero, proving objective replacement rather than stacking.
- A training-episode97 raw/no-gate inference replay completed with zero policy
  errors, full per-tick factorized diagnostics, identity action scale, and exact
  action-trace equality between assist-disabled and assist-enabled modes. The
  smoke policy recovered `9/10`, including startup, but quality is deliberately
  non-decisional after one epoch.
- Artifact root:
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_hold_liveness_20260712/h3_factorized_intent_effort_smoke`.

### [Execution target G12F/H3: Fixed 200-Epoch Training-Only Candidate] Launch contract

- Freeze the exact accepted G11 semantics and initialize from E16 SHA-256
  `0f2515cdaaad4a541b542d9423508b0885cada029feaaa1bd2957a3dbf87ec8f`.
  Split SHA-256 is
  `09fe85bdab539ca2a12b5b4613f507ea009706cb38077b46e168f5171da59a3d`;
  direct deadzone SHA-256 is
  `3a892aeee4f75ef903b93e7afe201211ebaad48d97c552756b8cdf65abf10867`.
- Train exactly `200` epochs with transition probability `0.5`, horizon `20`,
  magnitude weight `1.0`, class weight `0.05`, class vector `[8,1,8]`, held
  weight `0.1`, target delays `18,19`, exponential `k=0.01`, margin `0.02`, and
  no legacy intent/deadzone/release/H2 action-shortfall objective.
- Only best ordinary validation loss and the fixed epoch199 endpoint may enter
  the deterministic old-validation matrix. No class/probability/margin/weight,
  threshold, hysteresis, or checkpoint sweep is permitted.

### [Execution target G12F/H3: Fixed 200-Epoch Training-Only Candidate] Callback audit and reflection

- Training completed `200/200`; best ordinary validation loss was
  `0.1647819281` at epoch `145`. Best bundle SHA-256 is
  `3c90e58a446b8c90cc817fa41ec08a20e52fc5466c8ad7928df89b016989500e`;
  endpoint epoch199 SHA-256 is
  `19a7588f62cb7e1fb9f76ee240300cff28969fe910a1f6179d560ae477e83989`.
- The formal config differs from the accepted smoke only in task/output paths,
  `200` epochs, and artifact cadence. Its SHA-256 is
  `091431176b19d1624fd14ca1fa22df42339a915b1d2d9f20b244d8b7e60f2eff`.

### [Execution target G12V/H3: Deterministic Old-Validation Gate] Callback audit and rejection

- Decision set remained the five old validation episodes `94,91,84,74,92`,
  `48` anchors, direct mechanical thresholds, raw/no-gate identity-scale
  output, horizon `20`, with assist disabled and enabled. No `105..109` data or
  result was read.
- Best epoch145 recovered `45/48`, startup `5/5`, hidden `0`; endpoint epoch199
  recovered `45/48`, startup `5/5`, hidden `1`. Assist-enabled and disabled
  action traces are identical for both checkpoints, as required by projection.
- Relative to E16+assist `44/48`, best H3 recovers one reference deadlock and
  induces none. Relative to H2+assist `45/48`, best and endpoint have exactly
  the same three recovery/deadlock statuses and recover no additional anchor.
  All three shared failures are boom-positive at `episode_94:474` and
  `episode_74:198,208`; idle remains the dominant class for all 20 held ticks.
- Full-anchor held diagnostics further reject hard projection on safety:
  relative to the complete expert action at each frozen anchor, best H3 has
  unexpected effective directions on `7/48` anchors, one expert-opposite tick,
  and one direction flip; endpoint has `9/48`, one opposite tick, and one flip.
  H2+assist has `4/48`, zero opposite ticks, and zero flips under the same
  diagnostic. These are falsification diagnostics, not claims about live risk.
- Verdict: reject H3 before held-out access. It fails the predeclared `46/48`
  liveness gate and worsens the old-validation wrong-direction surface. Do not
  scan class weight, confidence threshold, margin, hysteresis, or later epochs.
- Positive evidence: separating direction/effort eliminates sub-deadzone output
  and H3 matches H2 liveness without assist, but static hard classification
  cannot infer three ambiguous boom transitions and amplifies occasional class
  errors into effective commands. H4 must add causal retry evidence rather than
  another static threshold.
- Artifact root:
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_hold_liveness_20260712/h3_factorized_intent_effort_e16_200`.

### [Execution target G13R/H4: Execution-Feedback Candidate Adversarial Review] Callback audit and reflection

- Static qpos+qvel is rejected as the H4 hypothesis because identical fixed
  qpos/zero-qvel observations cannot distinguish “not tried yet” from “command
  issued but no response.” Historical full-budget qpos+qvel also increased
  end80 extra/wrong from `18.25%` to `31.5%` on the earlier live-like set.
- A learned retry token is deferred because it adds unlabeled recursive state,
  reset/duration semantics, and another exposure mismatch at once. The minimal
  genuine feedback candidate is B: `qpos[4] + qvel[4] + previous final commanded
  action[4]`, with the continuous signed H2 output and no new gate.
- The resampled diagnostic `commanded_action[t-1]` is not causally aligned:
  resampling uses separate observation and action indices. The input must be
  rebuilt from each raw source episode by selecting the latest command whose
  raw `action_send_timestamp_ns` is strictly earlier than the resampled source
  observation timestamp. Episode starts and excluded/gap segments reset to
  physical zero; sidecars retain source indices, timestamps, hashes, and reset
  provenance rather than mutating training HDF5.
- `/action[t-1]` is causal but is only an expert safe-action label, not the final
  commanded action. It may not be silently renamed as H4 evidence. Likewise,
  target-axis synthetic previous commands are prohibited because they encode
  the supervised direction into the input.
- Existing normalization computes stats over train plus validation after the
  split. H4 must not extend that leakage: inherit the accepted H2/E16 qpos and
  action stats for parity, but compute qvel and causal-command stats from the 19
  train IDs only, with IDs and hashes in provenance.

### [Execution target G13/H4: Causal Previous-Command Retry Contract] Locked single candidate

- Input order is exactly qpos, raw qvel, causal previous final command (`12D`).
  Base model is H2 best epoch110, expanded from `4D` by copying both ACT qpos
  projection matrices' first four columns and zero-initializing the new eight;
  all other parameters load strictly and the optimizer is fresh. Zero-feature
  inference must match the source model before training.
- Normal ACT/H2 training always consumes the real causal sidecar command. On a
  sampled transition only, a second counterfactual branch constructs two
  previous-command variants from a stable hash of episode ID, timestep, and
  fixed seed: one positive and one negative command on a hash-selected axis,
  with target-independent magnitude below that direction's mechanical
  deadzone, all other axes zero, and qvel zero. The generator has no target
  action parameter, so direction cannot leak from supervision.
- The symmetric pair shares the same frozen observation and target chunk; its
  losses are averaged and enter with one fixed branch weight `1.0`. The normal
  causal sample remains in every batch. This teaches “a weak command was tried
  without response; infer the correct retry from image/qpos,” not “repeat the
  sign supplied in the input.” No variant, axis, magnitude, or weight sweep is
  allowed.
- Preserve H2/E16 continuous signed action L1, KL `10`, intent BCE weight
  `0.05`/positive weight `8`, transition probability `0.5`, horizon `20`, held
  weight `0.1`, assist trigger fraction `0.5`, margin `0.02`, and two terminal
  held prefixes. H3, release, generic deadzone, and window objectives remain off.
- Recursive state-hold is the decision path. Warmup and teacher-forced replay
  use each observation's causal recorded command. Held tick zero uses the
  anchor's recorded previous command; after each prediction, the selected
  offline arm's final returned action (post-assist in the assist arm) becomes
  the next held input while image/qpos stay fixed and qvel is zero. Teacher-
  forced success cannot override recursive failure.
- Train one fixed `200`-epoch candidate. Only ordinary-val best and epoch199 may
  enter old validation. Advancement requires recursive raw/no-assist at least
  `46/48`, startup `5/5`, zero induced/hidden deadlocks relative to H2+assist,
  and no unexpected/opposite, direction-flip, ordinary-window, or tail
  regression. Zero-command and zero-qvel ablations are attribution-only and may
  not tune the candidate.
- Model action scale remains identity throughout. Joystick `action_scale` is
  neither an H4 input nor a permitted model-action compression/fix.

### [Execution target G13/H4: Causal Feedback Implementation and Smoke] Callback audit and reflection

- Responsibility boundary: causal raw-send alignment, sidecar validation,
  target-independent symmetric retry variants, and feedback normalization live
  in the focused `execution_feedback` data owner. Dataset and trainer only wire
  the opt-in payload/second loss branch. The state-hold owner performs recursive
  feedback, and ACT checkpoint expansion is isolated in a strict two-key owner.
- The sidecar manifest covers exactly the locked 19 train and five validation
  IDs, never `105..109`. It contains 24 episode records, `16505` valid aligned
  observations, episode-start-only resets, and strictly-prior command ages from
  `3967` to `201223029` ns. Full source/sidecar hash and causal realignment
  validation passed. Manifest SHA-256 is
  `a76c14124a718ab74040040b7bb3ceddfd9d30ca537be6dc10f00f849bffd957`.
- Feedback normalization inherits H2 qpos/action statistics byte-for-byte and
  computes qvel/previous-command statistics from the 19 train IDs and `13288`
  included steps only. The recursive offline CLI requires a feedback manifest
  exactly when the bundle consumes `previous_final_command`, verifies all
  hashes, loads the per-episode causal command, and feeds each held output back
  as the next command input.
- Strict 4D-to-12D initialization grows only
  `input_proj_robot_state.weight` and `encoder_joint_proj.weight`, copies their
  first four columns, and zeros the appended eight. Every other tensor is
  exact. Three real observations (`episode_97:0,198`, `episode_94:474`) produce
  bit-identical action and intent outputs before training; both maximum
  differences are `0.0`. The proof artifact is
  `h4_causal_previous_command_h2_200/zero_init_functional_parity.json`.
- Focused verification passed `81` execution-feedback, checkpoint-expansion,
  dataset/trainer, policy-action, recursive state-hold, and CLI tests. The
  one-epoch CUDA smoke completed with a fresh optimizer; both train and
  validation activated the counterfactual branch with finite nonzero loss, and
  the console expansion report showed exactly the two `4 -> 12` projections.
- Verdict: accept integration and provenance only. Launch the single locked
  200-epoch candidate without changing counterfactual seed, branch weight,
  transition probability, deadzone, horizon, or H2 objectives.

### [Execution target G14/H4: Fixed Training and Recursive Old-Validation Gate] Callback audit and rejection

- Formal training completed `200/200` in about `2m13s`. Ordinary validation
  selected epoch `40` at loss `0.2090457901`; epoch `199` remained the fixed
  endpoint. Stable best checkpoint SHA-256 is
  `c841f83c02528d3160e952181ac471fda7274d327bafd4139f0d2a23e2e6cf14`;
  endpoint SHA-256 is
  `164f5ae9b5fd3d5baf4b48bf779e52ced068656025e60959f891f0e09c8b6978`;
  feedback-aware stats SHA-256 is
  `dc1c8defd77f92b7a592d88e90d95a2ab1f303a7900a3da16e8b355dcb3cab3f`.
- Decision set remained old validation episodes `94,91,84,74,92`, all `48`
  anchors, identity scale, direct deadzones, horizon `20`, and recursive
  previous-final-command feedback. Neither checkpoint accessed `105..109`.
- Best epoch40: raw `41/48`, startup `5/5`, hidden `1`; assist `45/48`, startup
  `5/5`, hidden `1`. Relative to H2, raw recovers two and induces one; assist
  recovers `episode_74:208 boom+` but induces
  `episode_92:249 bucket-`, for zero net liveness gain.
- Endpoint epoch199: raw `40/48`, startup `5/5`, hidden `0`; assist `45/48`,
  startup `5/5`, hidden `1`. Its assist recovery/deadlock statuses are exactly
  H2's and therefore provide no improvement.
- Safety falsification rejects best epoch40 independently of its liveness
  failure. H2+assist has unexpected effective directions on `4/48` anchors and
  `8` held ticks; H4 best+assist worsens this to `7/48` and `29` ticks. Both
  have zero target-opposite ticks and zero direction flips, but the unexpected
  regression violates the locked contract.
- At the newly induced `episode_92:249 bucket-` anchor, the initial causal
  command already contains boom+ and bucket- evidence. H4 tick zero discards
  both and emits swing+ `0.520`; recursive feedback grows swing to about
  `0.848`. Assist snaps swing at tick zero, yielding unexpected swing motion on
  all `20/20` held ticks while bucket never recovers. H2+assist instead snaps
  bucket negative at tick zero.
- At `episode_74:208 boom+`, H4 raw remains below the boom deadzone for all 20
  ticks; only assist snaps it at delay `16`. This is not H4-alone recovery.
  At `episode_74:198`, teacher forcing eventually supplies a stronger recorded
  command, while recursive feedback stays near `0.05` and deadlocks, exposing
  the exact hidden closed-loop failure H4 was meant to remove.
- Root cause: the fixed counterfactual branch covers only sub-deadzone weak
  commands with zero qvel. Real recursive and assist paths also contain an
  effective or post-assist command with zero response. That unsupported state
  lets the full 4D command act as an erroneous phase/axis cue and creates an
  attractor. This falsifies the fixed H4 hypothesis; it is not an expansion or
  action-scale error.
- Verdict: reject H4 before held-out access. Do not scan epochs, branch weight,
  weak-command magnitude, or extend the same candidate post hoc. Backtrack to
  H2 best plus mechanical assist. H5 must be externally monotonic: it may add a
  bounded same-direction promotion from independent persistent evidence, but
  must never attenuate an already effective policy command or feed its own full
  action vector back into the learned phase representation.
- Artifact root:
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_hold_liveness_20260712/h4_causal_previous_command_h2_200`.

### [Execution target G15A/H5: Frozen Train-Confidence Monotonic Governor] Callback audit and rejection

- The first H5 candidate was frozen before full old-validation replay. It used
  the H2 intent head's 19-train-anchor target-direction `q01=0.357304`, required
  target probability greater than the opposite direction, four consecutive
  same-sign ineffective raw predictions, train-only per-axis qvel limits
  `[0.07045,0.02980,0.02067,0.07763]`, and at most one maximum-probability
  promotion. Promotion only increased the existing raw sign to the calibrated
  deadzone plus margin; H2 mechanical assist remained intact.
- Exact warmup and state carry were preserved across all 48 old-validation
  anchors. The candidate reached `46/48`, startup `5/5`, with zero induced
  liveness deadlocks and recovered `episode_74:208 boom+` at delay `2`.
- It still has one teacher-forcing-hidden deadlock at `episode_74:198 boom+`:
  the teacher path later sees boom-positive intent `0.611` and promotes, while
  the held observation stays at `0.187` and never qualifies.
- It also creates a new wrong effective swing at `episode_92:249`. Governor
  state carried six qualifying swing ticks from exact warmup; at the anchor,
  swing raw fraction `0.306`, probability `0.959`, and direction margin `0.951`
  cause immediate promotion even though the full expert action is boom-positive
  plus bucket-negative. Unexpected effective motion changes from H2's
  `4 anchors / 8 ticks` to `5 anchors / 5 ticks`; opposite and flip counts stay
  zero.
- Scalar dominance proves that threshold scanning cannot repair this candidate:
  the desired `episode_74:208 boom+` evidence is weaker on every score
  (probability `0.367`, margin `0.287`, raw fraction `0.164`, counter `4`) than
  the wrong swing. Raising probability, margin, raw-fraction, or persistence
  rejects the desired recovery before the wrong one; train `q05=0.8408` is one
  concrete example.
- Verdict: reject G15A. Do not tune its scalar thresholds or reset warmup state
  artificially. A follow-up may proceed only as a structural hypothesis using
  a train-derived maximum-effective-axis capacity plus cross-head direction
  agreement, because that constraint can distinguish an unsupported third
  axis from a missing second axis without knowing the anchor target.

### [Execution target G15B/H5: Capacity-Constrained Cross-Head Consensus] Train-only callback audit and rejection

- The independent G15B structure removes the absolute intent threshold, adds
  the train-supported maximum of two simultaneous effective axes, clears weak
  counters whenever capacity is full, requires raw-sign intent greater than
  its opposite, and uses the exact training target raw-fraction minimum
  `0.0282211416`. Persistence stays `4`; one maximum-evidence direction may be
  promoted to deadzone plus `0.02`; no action is attenuated or sign-changed.
- Qvel stall limits are reproducibly train-only. From the 19 train IDs, select
  713 non-excluded center ticks whose complete centered nine-step expert-action
  window is all-zero (`abs(action)<1e-6`). Per axis, use
  `max(3 * population_std, 0.006)` rad/s, yielding swing `0.07044806`, boom
  `0.02980213`, stick `0.02067453`, bucket `0.07763434`. Future samples are used
  only to estimate offline stationary noise, never as a runtime feature.
- The active-axis cap is also strictly train-supported: all 180 transition
  anchors contain one or two expert-effective axes (`135/45`); all 13288 valid
  train steps contain zero, one, or two (`3865/8496/927`), never more than two.
- Nevertheless, full sequential H2 replay on the same 13288 train steps
  rejects the rule before implementation or old-validation access. It promotes
  `1570` ticks in `195` runs (`11.82%`); only `11` promotions match the full
  same-frame expert-effective direction set and `1559` are unexpected, for
  `0.70%` precision. It promotes `649` expert-zero frames (`16.79%` of those
  frames) and `153` frames after each episode's last expert-effective action.
- The false motion is not confined to one ambiguous axis: promotion ticks are
  boom-negative `491`, boom-positive `516`, bucket-negative `4`,
  bucket-positive `189`, stick-negative `64`, swing-negative `101`, and
  swing-positive `205`. Training contains no effective stick labels, so 64
  stick promotions are a direct semantic counterexample.
- At the 179 unique transition steps, only 16 steps promote; three promotions
  belong to the full expert direction set and 13 are unexpected. Only `3/180`
  per-axis anchor targets receive the promotion. The cap correctly limits the
  final count to two but cannot make the first or second direction correct.
- Verdict: reject G15B on train-only evidence. Do not implement it, run old
  validation, or tune raw fraction, intent, qvel, or persistence. One final
  independent H5 evidence source may be audited: the already frozen causal
  E52 temporal-direction model used only as positive eligibility, never for
  phase or direction attenuation. Its training provenance and train-only
  promotion precision must pass before any old-validation run.

### [Execution target G15C/H5: Frozen Temporal-Direction Eligibility] Callback audit and rejection

- The final E52 temporal-direction artifact is not an admissible independent
  old-validation signal. It was fitted on all 24 episodes, including the five
  old-validation episodes, and its target labels use the legacy runtime-scaled
  deadzone domain rather than the locked direct mechanical domain. Using it to
  select or qualify H5 would violate both the split and action-domain contract.
- The only split-safe evidence available is the previously frozen out-of-fold
  direction probability. Against the frozen `0.5` eligibility threshold, its
  boom-positive probabilities at the three intrinsic old-validation failures
  are only `0.0070` (`episode_94:474`), `0.0174`
  (`episode_74:198`), and `0.0352` (`episode_74:208`). It therefore cannot
  recover any of the three failures, regardless of the downstream monotonic
  promotion magnitude.
- The out-of-fold swing probability for the known wrong G15A promotion at
  `episode_92:249` is `0.0297`. That negative is directionally useful but does
  not rescue a signal that also abstains on every desired recovery.
- Verdict: reject G15C before implementation, old-validation replay, or any
  held-out access. Do not lower the frozen threshold, refit on the old
  validation episodes, or translate legacy thresholds post hoc. H5 is
  exhausted: the current offline demonstrations contain no split-safe,
  high-precision positive direction signal for a liveness governor.

### [Execution target G16: Closed-Loop Goal Closure and Data Requirement] Final reflection

The best verified current pipeline remains raw ACT plus the existing
mechanical deadzone assist at `40/45` held-out anchors and startup `5/5`. H2 is
useful training evidence because it improves raw ACT from `33/45` to `39/45`
without induced or teacher-forcing-hidden deadlocks, but H2 plus assist remains
exactly `40/45`; it does not satisfy the strict replacement contract. H3-H5
were rejected on the old-validation or train-only gates and never consumed the
held-out set.

| Candidate layer | Decision evidence | Liveness result | Safety/provenance result | Verdict |
| --- | --- | ---: | --- | --- |
| E52 gate | held-out 45 anchors | `15/45`; startup `0/5` | induces 18 raw recoveries; hidden 22 | remove from command path |
| raw ACT | held-out 45 anchors | `33/45`; startup `5/5` | identity action scale | reference only |
| raw ACT + mechanical assist | held-out 45 anchors | `40/45`; startup `5/5` | hidden 0 | best verified current pipeline |
| H1 direct relabel + assist | held-out 45 anchors | `39/45` | induces one reference failure; hidden 1 | reject |
| H2 transition objective + assist | held-out 45 anchors | `40/45` | zero induced/hidden, but no strict gain | retain as training evidence, do not promote |
| H3 factorized projection | old validation 48 anchors | `45/48` | unexpected/opposite/flip regression | reject before held-out |
| H4 causal previous command + assist | old validation 48 anchors | `45/48` | induces one failure; unexpected ticks `8 -> 29` | reject before held-out |
| H5 governors | train/old-validation gates | best G15A `46/48` | hidden 1 and wrong swing; G15B precision `0.70%`; G15C split/signal invalid | reject before held-out |

The original offline replay missed the live deadlock because it advances to
the next expert observation even when the evaluated command would not move the
machine. This breaks the causal loop: a suppressed command is followed by an
image/qpos state produced by the expert's successful command, so the policy is
given phase progress it did not earn. The new state-hold diagnostic freezes the
anchor image and qpos, zeros qvel, recursively feeds the selected pipeline's
final command state, and checks all 20 ticks against the direct mechanical
deadzone. Its teacher-forced comparison exposes failures hidden by recorded
state progression, and its opt-in full-horizon trace records wrong/extra motion
even after the target direction first recovers.

This diagnostic can falsify software liveness, but it cannot prove real
actuation. Two histories can have the same current image/qpos/qvel: no command
has yet been tried, or a command was sent and the excavator failed to respond.
The successful demonstration data contains the first history but not a
reliably labeled sample of the second. Any model using only the current inputs
must produce the same output for both. H4 added the previous command but still
lacked the observed execution outcome; the command vector became an ambiguous
phase cue and created the wrong-axis attractor at `episode_92:249`. This is an
identifiability limit, not a remaining loss-weight or scalar-threshold search.

The next admissible architecture is an execution-aware retry controller, not a
new suppressive phase gate:

1. Keep a continuous raw ACT proposal with identity model action scale. H2's
   transition objective is the best starting checkpoint family, but it remains
   experimental until a future candidate passes the complete contract.
2. Keep the mechanical deadzone assist as a deterministic actuator-interface
   module: same sign only, monotonic magnitude promotion, no phase ownership,
   and no compression of already-effective model output.
3. Train a separate response/effect model from actual sent-command histories,
   command timestamps, qpos/qvel deltas over a calibrated response window, and
   acknowledgements. It estimates whether the commanded axis/direction
   produced motion; it must not infer failed actuation from visual phase alone.
4. Train a high-precision, abstaining direction-eligibility signal from
   on-policy retries and operator corrections. A retry may only increase the
   already proposed sign after a real command has been sent and no response is
   observed. Ambiguous direction must abstain and return control to replanning
   or the operator.
5. Train the recursive path with on-policy aggregation or DAgger-style
   correction data and scheduled model-command histories. Teacher-forced expert
   command histories alone are not an acceptable training substitute.
6. Preserve independent safety and gohome ownership. An execution retry may
   not attenuate an already-effective safe command, invent an unsupported
   direction, hold motion through release/tail, or issue gohome.

The minimum new field record must causally align: raw policy action and intent,
assist input/output, final sent command and send timestamp, observation source
timestamps, qpos/qvel before and throughout the response window, command
acknowledgement or suppression reason, operator override/correction, reset/gap
boundaries, and release/tail/gohome labels. Collection must cover startup and
mid-cycle weak-command/no-response retries, effective responses, wrong-sign
corrections, multi-axis motion, releases, and idle/tail negatives for every
supported axis/direction. The current train set has no effective stick labels,
so any learned stick retry must abstain until dedicated evidence exists.

Evaluation remains staged: session/terrain/operator-disjoint training and
validation; recursive and teacher-forced full-horizon state-hold; ordinary
window, wrong/extra, tail, and gohome gates; one frozen held-out run; then
`shadow_zero`; finally a bounded supervised field A/B. Offline state-hold is a
required falsification gate, not a claim of true-machine success.

Closure is blocked, rather than complete: every admissible H1-H5 route has been
factually exhausted, and another learned candidate now requires new on-policy
execution-response data plus explicit authorization for a controlled field
collection. No runtime default, Jetson state, or live command path was changed.

### [Execution target G19: Existing-Data Execution-Response Audit] Callback and boundary

- The current `72..104` 20 Hz resampled dataset was processed without changing
  any source HDF5. The additive sidecar owner is
  `testbed/testbed/data/execution_response.py`, with CLI entrypoint
  `python -m testbed.cli.audit_execution_response` and focused tests.
- The audit covers all 30 available episodes, `20,964` steps and `20,934`
  valid causally aligned observations. It uses the direct mechanical deadzone
  table and identity policy scale `[1,1,1,1]`; stick is explicitly marked
  `task_not_applicable` rather than treated as a missing label.
- It produces `285` effective command onset windows across swing, boom and
  bucket. At horizon `20`, `281` windows show same-direction qvel response;
  `42` windows also contain opposite-direction qvel above the stationary noise
  threshold (often a residual/inertia observation, not an automatic wrong-label);
  `3` windows show neither same-direction nor opposite-direction response:
  `episode_82:258 boom+`, `episode_87:308 bucket-`, and
  `episode_98:543 boom+`.
- These three are conservative review candidates only. They are all close to
  the calibrated deadzone and there is no policy intent or operator-correction
  label in the source HDF5. Do not train a retry governor from them as if they
  were confirmed hydraulic failures. The artifact is a response/latency audit
  and sidecar-pipeline proof, not a replacement for on-policy failure data.
- Artifact root:
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/execution_response_existing_data_20260712`.
- Manifest SHA-256:
  `7d367faf43604d3584774609a083f12e9cd38cdd75524f153c72f95d570f72a5`.

### [Execution target G25/H2: Split-Safe Execution-Monitor Replay] Callback and boundary

- The target worktree is writable again. A focused owner was added at
  `testbed/testbed/policies/execution_monitor.py`; it observes final sent
  commands and causal qpos/qvel feedback but has no `action_scale`, hard intent
  argmax, suppressive gate, or zero-action fallback.
- `testbed/testbed/policies/execution_monitor_eval.py` provides an immutable
  sidecar replay adapter. `testbed/testbed/cli/evaluate_execution_monitor.py`
  evaluates only the locked 19 train and five validation IDs from
  `train_val_split.yaml`; six extra episodes present in the broad response
  audit are explicitly excluded. Episodes `105..109` are rejected if they
  appear in the split.
- The monitor owner and replay adapter have `12` focused tests passing; Ruff
  and `git diff --check` pass. The existing monitor tests cover causal
  timestamps, responded/stalled/unknown, reset/gap/safety, structural stick,
  same-direction retry token, and abstention on opposite/new directions.
- Replay report:
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/execution_monitor_eval_g25_split_20260712/execution_monitor_eval.json`
  with SHA-256
  `6c46434a7fd0c29e147dff4fd71a6c0588fca2cabe855343b7a791caaf4a2a2a`.
- Train-only response replay: `19` episodes, `180` onset events,
  `176 responded`, `4 stalled candidates`, `0 unknown`, and `0` monitor/sidecar
  response mismatches. Validation: `5` episodes, `48` events, `48 responded`,
  `0` stalled, and `0` mismatches. The locked identity action scale and
  direct-domain deadzone provenance are recorded in the report.
- This is response-label consistency, not retry precision. The report marks
  `retry_precision_estimable=false` and does not select a retry policy because
  the existing teleop sidecars have no policy-on intent, operator correction,
  or confirmed failed-actuation labels. No source HDF5 was modified and no
  external USB was required.
