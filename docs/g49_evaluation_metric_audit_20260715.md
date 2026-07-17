# G49 evaluation-metric audit

Date: 2026-07-15

Semantic contract revised: 2026-07-16. The original report schema is
superseded. All numbers below are relations to one demonstrated trajectory;
they do not label alternative actions wrong, unsupported task-wide, unsafe, or
deadlocked.

## Decision

The current G49 metrics are useful diagnostics, but the headline table is not a
sufficient model-selection or promotion benchmark. In particular:

- the 20 startup anchors are a property of the 20-episode chronological
  validation split, not a statistically designed sample;
- micro-averaged single-demo active recall hides severe axis imbalance,
  especially stick;
- state-hold demo-target reproduction tests only one demonstrated
  axis/direction and cannot establish correctness;
- the current multi-axis metric is single-demo similarity; additional active
  axes are anchor-relative differences, not errors;
- 261 per-axis anchors are clustered within 20 episodes and must not be treated
  as 261 independent trials.

The existing results remain valid as diagnostic observations. Their permitted
interpretation is narrowed below; no checkpoint is promoted or rejected solely
by a replacement scalar score.

## Sources and frozen scope

This audit reads only the already-open G49 train/validation data and existing
N0--N4 evaluation artifacts. It does not inspect the sealed test episodes.

- Dataset view:
  `/data/pingfan/Excavator_real_stack_data/g48_new_trainval_view_v1`
- Split:
  `/data/pingfan/Excavator_real_stack_data/g48_new_trainval_view_v1/train_val_split.yaml`
- Validation source episodes: `135..155` with unavailable/excluded IDs omitted,
  mapped to composite IDs `10120..10139`
- Evaluation root:
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation`
- Direct-output deadzones: positive `[0.661, 0.259, 0.500, 0.408]`, negative
  `[0.721, 0.357, 0.500, 0.508]`
- Sampling: 20 Hz

## Why there are 20 startup anchors

`extract_should_move_anchors()` first finds the earliest frame in each episode
where any expert axis/direction crosses the fixed deadzone. Every axis/direction
that transitions from ineffective to effective on that same earliest frame is
labelled `startup`; all later per-axis transitions are labelled `mid_cycle`.

The implementation does not impose one startup axis per episode. The current
validation data happens to have exactly one newly effective axis on the earliest
frame of each of its 20 episodes, so it produces exactly 20 startup anchors.
The 120 training episodes produce 122 startup-axis anchors because two episodes
have simultaneous first transitions.

Therefore `20` means “one observed first crossing in each current validation
episode”, not “20 representative startup conditions” and not “the number needed
for a reliable success-rate estimate”.

## Startup is task-consistent but not capability-complete

| Split | Episodes | Startup-axis anchors | stick+ | boom- | swing+ | swing- | bucket+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 120 | 122 | 79 | 37 | 4 | 1 | 1 |
| Validation | 20 | 20 | 18 | 2 | 0 | 0 | 0 |

The concentration is evidence of a consistent expert task sequence, not random
operator behavior. Within 20 ticks after the first crossing, 15/20 validation
episodes contain the same `stick+ / boom- / bucket+` direction set; within 40
ticks, 19/20 contain that set. The exact first-crossing label records which axis
reaches the command threshold first on that consistent sequence.

The chronological validation block is therefore useful for measuring this
specific later-recording task style. It is not a balanced four-axis capability
benchmark. Its startup result can rank candidates on predominantly `stick+`
starts, but cannot establish startup capability for swing, bucket, stick-,
boom+, or different initial task sequences. “Coverage-limited” must not be
misread as “demonstration-inconsistent”.

The absolute uncertainty is also large. Wilson 95% intervals for held-startup
demo-target reproduction are:

| Model | Demo target reproduced | Wilson 95% interval |
| --- | ---: | ---: |
| N0 | 3/20 | 5.2%--36.0% |
| N1 | 14/20 | 48.1%--85.5% |
| N2 | 7/20 | 18.1%--56.7% |
| N3 | 1/20 | 0.9%--23.6% |
| N4 | 5/20 | 11.2%--46.9% |

Because all candidates use the same anchors, paired evidence is stronger than
those absolute intervals. N1 beats N0 on 11 startup anchors and loses on none
(two-sided exact paired p=0.00098). This is credible evidence that N1 reproduces
this validation block's demonstrated targets more often; it is not evidence
that its alternatives are more correct or that it covers unobserved conditions.

## Anchor quality: mostly real transitions, but boundary-sensitive

The 261 validation anchors are not dominated by one-frame threshold chatter:

- only 1/261 effective runs lasts fewer than 4 ticks;
- only 10/261 last fewer than 10 ticks;
- median preceding same-axis idle duration is 48 ticks;
- only 1/261 has a preceding idle duration of 3 ticks or fewer.

This supports using the anchors as sustained expert command transitions.
However, the expert onset values sit close to the threshold: onset/deadzone
ratio P10 is `1.0049` and the median is `1.0335`. A threshold change therefore
changes the event identity:

| Deadzone scale | Per-axis anchors | Unique episode/step events | Startup-axis anchors |
| ---: | ---: | ---: | ---: |
| 0.95 | 261 | 256 | 21 |
| 1.00 | 261 | 255 | 20 |
| 1.05 | 252 | 246 | 20 |

The fixed deadzone is acceptable for a command-domain diagnostic, but a single
threshold result must not be treated as a calibrated physical truth. Future
reports should include at least one predeclared deadzone sensitivity band.

## The startup instant fragments a coordinated onset

At the exact first-crossing frame all 20 validation episodes have one active
startup axis. This changes rapidly when an onset window is considered:

| Window from first crossing | Episodes with two or more active directions |
| ---: | ---: |
| 0 ticks | 0/20 |
| 1 tick | 2/20 |
| 5 ticks | 5/20 |
| 10 ticks | 9/20 |
| 20 ticks | 18/20 |

Thus the exact first tick is useful for asking “what moves first”, but it is not
a complete startup-intent label. Future joint-intent metrics must group nearby
axis transitions into a predeclared onset window instead of assuming startup is
intrinsically single-axis.

It is also not an idle ground-truth boundary. In the validation block, the
expert first crossing occurs 27--53 ticks after recording begins, with a median
of 31.5 ticks (1.575 seconds at 20 Hz). The interval may contain operator
recording and observation preparation that is not observable as a task-level
wait requirement. A policy output before the expert crossing must therefore not
be called premature or unsafe from timing alone.

The current `single_demo_open_loop_similarity_v4` startup view searches only for
the policy's first deadzone-effective saved output. Once that output appears,
later teacher-forced observations are excluded from the startup interpretation,
because a real effective command could have changed the subsequent state. It
reports time-to-first executable output and descriptive similarity of that first
direction set to the single-demo anchor and 40-tick event support. It imposes no
required startup axis and defines no pass/fail gate.

## The 261-anchor denominator is clustered

The validation set contains 261 per-axis transition anchors but only 255 unique
episode/step event locations. Each episode contributes 4--19 anchors, with a
median of 13.5. The aggregate therefore weights episodes with more transitions
more heavily, and anchors in the same episode share terrain, camera appearance,
mechanical state, and operator style.

Report the existing micro aggregate for traceability, but use episode-macro and
axis/direction-macro results for model ranking. Confidence intervals or paired
comparisons must resample/cluster by episode, not by anchor row.

## Demo-target reproduction is not policy correctness

State hold asks whether one newly demonstrated axis/direction appears at least
once within 20 repeated ticks. This is conditional reproduction of one sample,
not a unique-correct-intent or generic-liveness test.

Recomputing the existing traces gives:

| Model | Startup demo target reproduced | Exact demo-anchor set appears at any tick | Demo-target reproduction with no anchor-extra/opposite/flip |
| --- | ---: | ---: | ---: |
| N0 | 3/20 | 1/20 | 0/20 |
| N1 | 14/20 | 5/20 | 5/20 |
| N2 | 7/20 | 1/20 | 0/20 |
| N3 | 1/20 | 0/20 | 0/20 |
| N4 | 5/20 | 0/20 | 0/20 |

For N1, 9 of the 14 target reproductions also emit a direction not active at
the exact demo anchor during the held horizon. Many are later `boom-` and
`bucket+` directions from the consistent expert task motif. They are not
evidence that the expert is inconsistent. Under the state-hold intervention,
the physical state has not progressed, so this is a useful state-timing
disagreement flag; demo timing alone does not prove that the alternative
startup direction is unsafe or invalid. N1 is the strongest demo-target
reproduction reference, but neither `14/20` reproduction nor `5/20` exact
demo-anchor similarity is a task-success score.

Current state-hold reports place these descriptive outcomes side by side:

1. demo-target reproduction;
2. demonstrated-effective direction preservation;
3. exact single-demo ternary-vector similarity;
4. anchor-extra, opposite-to-demo-target, or direction-flip events.

None of items 2--4 is a correctness or safety label without independent task
or physical supervision.

## A single 20-tick demo-target number hides delay

Twenty ticks equals one second and matches the current ACT chunk horizon. Older
recorded response audits also found that many axis responses appeared by 20
ticks, but G49 has no current command-response calibration establishing 20 as a
physical acceptance boundary.

The existing held-startup demo-target reproduction curves are:

| Model | <1 tick | <3 ticks | <5 ticks | <10 ticks | <20 ticks |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0 | 2 | 2 | 2 | 3 | 3 |
| N1 | 10 | 11 | 11 | 12 | 14 |
| N2 | 6 | 6 | 6 | 6 | 7 |
| N3 | 0 | 0 | 0 | 0 | 1 |
| N4 | 4 | 4 | 4 | 4 | 5 |
| N5 | 5 | 5 | 5 | 5 | 5 |

The 20-tick endpoint is a reproducible observation horizon, not a generic
deadlock veto. It must be reported as a curve. In particular, N3's sole target
reproduction appears at tick 18 and should not look equivalent to immediate
reproduction.

## Micro active recall hides the weak axis

The current headline active recall pools all active axis/frame labels. Swing and
bucket dominate the numerator, so the number hides stick behavior:

| Model | Headline micro active recall | Axis-macro active recall | Stick active recall |
| --- | ---: | ---: | ---: |
| N0 | 81.1% | 67.8% | 5.9% |
| N1 | 82.9% | 72.7% | 25.2% |
| N2 | 80.3% | 68.7% | 15.6% |
| N3 | 77.9% | 65.8% | 9.1% |
| N4 | 77.2% | 65.1% | 9.2% |
| N5 | 80.6% | 70.6% | 24.2% |

Micro recall remains useful for workload-weighted imitation. Axis macro, the
eight axis/direction cells, and episode macro must be primary whenever the
claim is general four-axis capability.

## Current multi-axis metric rewards over-activation

The reported `multi-axis all-directions preserved` checks whether all expert
active directions are present but does not require expert-idle axes to remain
idle. On the same 3,844 multi-axis validation frames:

| Model | Active directions preserved | Exact ternary vector including no extras |
| --- | ---: | ---: |
| N0 | 54.5% | 50.5% |
| N1 | 56.7% | 43.1% |
| N2 | 51.8% | 50.2% |
| N3 | 50.4% | 47.5% |
| N4 | 46.9% | 44.1% |
| N5 | 53.8% | 45.6% |

N1 appears better than N0 under recall-only preservation but worse under exact
joint intent because it emits more extra axes. Both metrics are valid; neither
may be reported alone.

## Baseline and metric verdict

The matched candidate baselines are structurally appropriate:

- N0 versus N1/N2 isolates training-objective changes under eye2;
- N0 versus N3 measures the tested eye2-to-naive-four-camera change;
- N3 versus N4 isolates additive camera/role identity.
- N4 versus N5 isolates transition supervision under the same four-camera role
  path; N1 versus N5 exposes the remaining camera-path and interaction effect
  under the same transition objective.

The current headline indicators are not appropriate as a single benchmark.
Use the following minimum scorecard for the next candidate generation:

1. **Data/split:** preserve the chronological validation result, but add a
   separately frozen, terrain/initial-geometry and axis/direction-stratified
   evaluation panel. The current HDF5 metadata has no terrain taxonomy, so
   terrain generalization cannot be recovered from episode IDs alone.
2. **Event definition:** require a sustained expert onset, a minimum preceding
   idle dwell, and a predeclared nearby-transition grouping window. Report exact
   first-axis and grouped joint-onset results separately.
3. **Aggregation:** report micro, episode macro, axis macro, and all eight
   axis/direction cells. Use episode-clustered paired uncertainty.
4. **Executable proposal:** for natural startup, report time to the first
   executable action without requiring a demo-selected axis. Retain target and
   exact-vector curves only as single-demo similarity diagnostics. Keep 20
   ticks as an observation horizon, not a deadlock or physical boundary.
5. **Validity and safety:** do not infer either from demo disagreement. Report
   anchor-extra, opposite-to-demo-target, and direction flips descriptively;
   obtain independent held-idle, release/tail, task, and physical labels before
   using pass/fail language.
6. **Continuous quality:** retain MAE as a secondary active-magnitude/trajectory
   metric, not as the primary policy rank.
7. **No scalar collapse:** do not combine liveness and safety into one weighted
   score until real-machine costs justify those weights. A candidate must expose
   its Pareto trade-off.

Under this corrected reading, N1 is evidence that transition supervision helps
the current `stick+`-heavy startup distribution, while N3 is evidence that the
current four-camera continuous model is quieter and fits expert magnitudes
better. Neither result establishes a generally executable or field-ready
policy.

## Armed natural-startup follow-up

The follow-up `startup_activation_v1` test removes the required target axis. It
warms policy state on the recorded preparation prefix, suppresses commands, and
then holds the final expert-ineffective observation for 20 ticks. Its question is
only whether the raw policy still enters any deadzone-effective axis/direction.

| Model | Live by 1 tick | Live by 20 ticks | 95% Wilson interval | Effective during warmup | First direction within single-demo local support among live |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0 | 11/20 | 12/20 | 38.7%--78.1% | 11/20 | 11/12 |
| N1 | 18/20 | 18/20 | 69.9%--97.2% | 20/20 | 17/18 |
| N2 | 14/20 | 15/20 | 53.1%--88.8% | 14/20 | 14/15 |
| N3 | 3/20 | 3/20 | 5.2%--36.0% | 3/20 | 2/3 |
| N4 | 6/20 | 7/20 | 18.1%--56.7% | 6/20 | 6/7 |
| N5 | 20/20 | 20/20 | 83.9%--100% | 20/20 | 16/20 |

N1 remains the strongest tested startup reference, but the 20-episode panel is
not large enough to treat every adjacent ranking as settled: paired discordance
is 7 wins versus 1 loss against N0 and 4 versus 1 against N2. In contrast, N1's
advantage over N3 and N4 is large on this panel. N4 improves over N3 but remains
far behind N1, so additive camera-role identity does not recover the startup
behavior lost by the tested four-camera continuous training path.

The main correction is semantic. N1's two non-live arm cases had emitted
effective output earlier in the preparation prefix and then stopped. Therefore
the earlier 20/20 first-candidate metric measured that the model had expressed
an executable proposal at least once; the new 18/20 metric measures whether that
proposal is present or can reappear at the arm reference. Early output is not
called unsafe because this dataset has no ground-truth idle/go annotation.

This test still cannot rank held-idle false activation, physical response,
hydraulic sufficiency, task completion, or terrain generalization. It is a
diagnostic/ranking signal, not a promotion gate.

## Selected N5 experiment and result

The selected training cell was a four-camera camera-role model with N1's
transition-boundary supervision, while keeping the G49 split, seed, ACT size,
camera order, qpos-only low-dimensional input, checkpoint schedule, and all other
training settings fixed. This closes the missing factorial comparison between the
strongest tested startup objective and the four-camera role input path.

Do not add a hard argmax intent head in the same run. N1 already shows that the
continuous policy can retain executable startup intent in 18/20 arm references,
while the current evidence has no held-idle ground truth. Combining both changes
would prevent attribution and could amplify false activation without measuring
it.

The result must be interpreted by the declared paired views rather than its
20/20 any-axis liveness number:

1. N5 versus N4 raises any-axis armed startup from 7/20 to 20/20. It reproduces
   5/20 startup demo targets and emits anchor-extra effective directions at
   20/20 anchors versus 8/20 for N4. Anchor-extra means absent from one exact
   demo frame; it is not a wrong, unsafe, or task-wide unsupported label.
2. N5 is effective at recording step 0 in all 20 episodes. This shows that its
   liveness gain is not evidence of learning the recorded preparation-to-start
   boundary.
3. N5 open-loop MAE is `0.1803`, worse than N4 `0.1561`, N1 `0.1767`, and N3
   `0.1507`; idle false-active is `18.4%`, also the worst of N0--N5.
4. N5 reproduces 169/261 state-hold demo targets, versus N4's 139 and N1's 186,
   with 102 anchor-extra-effective anchors and 40 opposite-to-demo-target ticks.

The subsequent task-support audit shows that 16/20 armed N5 starts are wholly
inside the same episode's 40-tick single-demo support. All remaining direction
combinations occur in the 120 training episodes; three introduce swing-positive
before its later use in the same validation episode, while one applies a common
training startup motif absent from that episode. Therefore the original
exact-anchor counts cannot support a blanket aggressiveness or wrong-axis claim.
They still justify the narrower conclusion that scene-conditioned selectivity,
held-idle safety, and release behavior are not yet established.

The missing cross cell therefore rejects the simple hypothesis that transition
supervision alone converts the current four-camera role representation into
selective startup behavior. Do not tune its weight on this validation set. If
four-camera work continues, test pair-aware fusion and whole-pair dropout, and
add held-idle/release evidence before any hard activation projection.
