# Intent-command-response validation design

Date: 2026-07-15; semantics revised 2026-07-16

## Goal

Build an evidence chain that can answer four different questions without
collapsing them into one offline score:

1. What executable behavior is demonstrated by this expert sequence, and what
   broader behavior support is actually estimable?
2. What intent did the model explicitly predict?
3. What final command crossed the controller boundary?
4. Did the machine respond, and did the task state progress?

No offline replay can directly answer the fourth question for a model command
that was never sent. The code must represent `unknown/out_of_support` instead of
converting missing counterfactual evidence into success or failure.

## Evidence chain

```text
observation
  -> single-demo intent event / same-demo event support
  -> model intent logits and onset timing
  -> raw continuous proposal
  -> projected/safe/final sent command
  -> controller acknowledgement
  -> aligned qvel/qpos response
  -> phase/task progress
```

Each arrow has its own metric and artifact. A later stage must never be inferred
from an earlier-stage pass.

## 1. A demonstration is an event sample, not unique ground truth

The current new-data expert is highly consistent. On the G49 validation block,
15/20 episodes contain `stick+ / boom- / bucket+` within 20 ticks after first
crossing and 19/20 contain the set within 40 ticks. The label owner should use
this consistency rather than reduce intent to one thresholded frame.

The focused `single_demo_intent_events_v2` owner derives, without modifying source
HDF5:

- `episode_id`, event/phase ID, onset and release bounds;
- immediate intent set at 0--1 ticks;
- near-onset intent set at 2--5 and 6--10 ticks;
- ordered direction motif and time-to-onset for each axis;
- persistence, pre-idle dwell, and later same-demo directions;
- qpos/qvel context and source hashes.

This permits three separate descriptive relations:

- **immediate demonstrated intent:** what is active now in this recording;
- **near-future demonstrated intent:** what follows in this recording;
- **outside single-demo event support:** absent from the declared horizon in
  this recording. This is not task-wide unsupported behavior.

State hold can distinguish “a same-demo direction appearing earlier than its
recorded onset” from “a direction outside this demo's event horizon”. Both are
descriptive diagnostics; neither proves that an alternative is incorrect,
unsafe, or unsupported by the task.

## 2. The model must expose intent explicitly

A continuous action alone cannot prove what the model intended. Add a public,
non-hidden inference trace rather than inferring latent intent after the fact:

```text
PolicyDecisionTrace
  intent_logits[axis, neg/idle/pos]
  onset_logits[axis, now/soon/later/none]
  intent_confidence[axis]
  continuous_proposal[axis]
  projected_action[axis]
  safe_action[axis]
  final_command[axis]
  temporal_state_id / query source
```

Initially the head may remain diagnostic-only. It must not control motion until
intent calibration and independently labelled task/safety tests pass.

Required intent metrics:

- per-axis/direction precision, recall, calibration, and abstention coverage;
- immediate and near-future intent reported separately;
- exact single-demo joint-vector similarity and anchor-extra-axis rate;
- event-sequence consistency and onset-timing error;
- eye-only, stick-only, four-camera, and pair-dropout agreement;
- frozen-feature probe versus end-to-end head, so representation and execution
  failures remain distinguishable.

## 3. State-hold needs two worlds and no correctness shortcut

Retain state-hold as a software falsification, but split it into:

1. **Perception snapshot:** reset/freeze temporal execution and evaluate the
   explicit immediate intent from the anchor observation.
2. **Runtime repeated-state:** restore the real policy/aggregation state and
   repeatedly present the unchanged observation to expose conditional
   demo-target non-reproduction or open-loop schedule leakage.

For the repeated-state world, report:

- demo-target reproduction curve at 1/3/5/10/20 ticks;
- exact immediate single-demo vector similarity;
- later same-demo directions whose recorded onset has not occurred;
- directions outside single-demo event support;
- opposite direction and direction flips;
- held-idle and release/tail behavior only when independently labelled.

This preserves the valuable no-response counterfactual without accusing a
consistent expert sequence of being random.

## 4. Current data can build a response envelope, not policy-on proof

The existing HDF5 contains causally alignable final operator commands and later
qpos/qvel. `testbed.data.execution_response` already provides the core
alignment pattern, but its old default supported axes omit stick because it was
written for the previous task. The new-data response owner must:

- support all four axes and both directions;
- recalibrate qvel noise and response horizons from train only;
- condition response envelopes on command magnitude, qpos/qvel state, and an
  explicit visual/terrain or load stratum where available;
- validate latency/probability on held-out episodes;
- return `supported`, `weak_support`, or `out_of_support` for each model command;
- keep opposite residual motion, no response, transport/safety suppression, and
  incomplete horizons separate.

A model proposal may then be compared with the historical response envelope:

- within support: estimated response feasibility and latency distribution;
- outside support: unknown, requiring a controlled probe;
- never: “the machine responded to this model”, because the model command was
  not actually sent.

## 5. Direct response proof requires policy-on causal logs

The final evidence layer must record the exact model decision and the exact
command actually sent:

```text
intent_event_id
model intent logits
raw policy action
projected action
safe action
commanded action + send timestamp
controller ack / suppression reason
qpos/qvel samples + observation timestamps
response status and latency
phase/task progress
```

Use the existing downstream `ExecutionMonitor` contract as the starting point.
Run it in observe-only mode first: no automatic retry and no command rewriting.
Controlled field validation should progress through bounded, authorized stages:

1. shadow inference and trace integrity;
2. safe single-axis/direction response probes for response-envelope coverage;
3. supervised event-level commands with abort limits;
4. short closed-loop task segments;
5. terrain/initial-condition held-out task trials.

Only stages 2--5 provide direct physical response evidence. Shadow-zero proves
the logging and inference chain, not response.

## Proposed code owners

- `testbed.data.expert_intent_events`: persistent immediate/near-future event
  labels and sequence motifs;
- `testbed.policies.intent_trace`: stable `PolicyDecisionTrace` schema;
- `testbed.policies.intent_eval`: intent calibration, joint vector, onset timing,
  and camera-intervention metrics;
- extend `testbed.data.execution_response` with a new all-axis response-envelope
  contract rather than changing old artifacts silently;
- extend `testbed.policies.execution_monitor` logging with `intent_event_id` and
  observe-only response reports;
- `testbed.cli.evaluate_intent_command_response`: one aggregator that keeps
  intent, command realization, response support, and task progress in separate
  report sections.

## Minimum next slice

Before training another model:

1. derive and review the G49 train/validation `ExpertIntentEvent` sidecar;
2. rerun N0--N4 outputs against immediate, near-future, and unsupported intent;
3. add the diagnostic `PolicyDecisionTrace` interface without changing commands;
4. build an all-axis new-data response envelope from train and validate it on
   the current chronological validation block;
5. report model commands outside the envelope as unknown;
6. only then decide the bounded policy-on data needed to close the remaining
   causal gap.

This design makes current data maximally useful while preserving the hard
boundary: expert demonstrations can validate task intent consistency and
historical command response, but only policy-on execution can directly validate
the response to a model command.

## Implementation progress

### Accepted slice 1: ExpertIntentEvent sidecar

Implemented focused owners:

- `testbed/testbed/data/expert_intent_events.py`;
- `testbed/testbed/cli/audit_expert_intent_events.py`;
- `testbed/tests/test_expert_intent_events.py`.

The full G49 train/validation run uses the frozen 120/20 split, the existing
direct-output deadzone artifact, and a declared 40-tick task-support horizon.
It produced 1,737 transition events without reading sealed test episodes.

Artifact root:
`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/expert_intent_events_h40_train120_val20`

Manifest SHA-256:
`e7d42c81a00be82483f658759d5c78f87f756270f2a69fafe025ac0739e7b52f`

The validation first-event summary directly confirms expert consistency:
19/20 episodes have task-supported set `boom- / stick+ / bucket+` within 40
ticks; the remaining episode has `stick+ / stick-`. The exact first-crossing
anchor remains 18 `stick+` and 2 `boom-`, which is now represented separately
from the later supported task sequence.

### Accepted slice 2: descriptive first-candidate startup evaluation

Implemented focused owners:

- `testbed/testbed/policies/intent_eval.py`;
- `testbed/testbed/cli/evaluate_expert_intent.py`;
- `testbed/tests/test_intent_eval.py`.

The current schema is `expert_intent_open_loop_eval_v3`. It evaluates only
saved teacher-forced continuous outputs and explicitly imposes no required
startup axis. The expert first crossing is treated as a recorded timing
reference, not a go signal or idle ground truth. Startup uses only the policy's
first deadzone-effective output; all later teacher-forced outputs are excluded
from the initial-readiness interpretation.

Exact anchor, anchor overlap, event-local support, and opposite direction are
descriptive expert-data similarity only. They are neither promotion gates nor
safety gates. The formal N0--N4 artifact is
`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/expert_intent_eval_n0_n4_val20_h40_v3/expert_intent_eval_report.json`,
SHA-256
`b92b12ab222ccc8ebd77368c2bdc875c17c0f8b11d4dfa474461d339694263cc`.

N1 produces its first effective output before or at the recorded expert onset
in 20/20 validation episodes; N0/N2/N3/N4 do so in 11/14/3/7 episodes. This
supports a relative startup-latency comparison only. It does not prove that an
early command was safe, moved the machine, or completed the task.

### Accepted slice 3: armed natural-startup activation

Implemented focused owners:

- `testbed/testbed/policies/startup_activation.py`;
- `testbed/testbed/cli/offline_startup_activation.py`;
- `testbed/tests/test_startup_activation.py`;
- `testbed/tests/test_offline_startup_activation_cli.py`.

The `startup_activation_v1` diagnostic first advances policy state through the
recorded preparation prefix with commands suppressed. It then explicitly arms
the final expert-ineffective observation, freezes image and qpos, zeros qvel and
previous command, and asks whether the raw policy enters any axis/direction's
active command region within 20 ticks. The expert onset defines a reproducible
reference only; it is not asserted to be the unique physical go signal.

N0--N4 natural liveness is 12/20, 18/20, 15/20, 3/20, and 7/20. N1 had effective
warmup output in all 20 episodes, but in episodes 10130 and 10133 that output
ended well before the arm reference and did not recover during the frozen
20-tick horizon. This is why the earlier first-candidate result of 20/20 and the
new persistent-at-arm result of 18/20 are both true. Neither number is a real
startup probability.

The first effective direction is not constrained to the expert's first axis.
Exact anchor, overlap, local support, outside support, and opposite direction are
reported only as single-demo similarity. The formal report directories are
`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/startup_activation_n{0,1,2,3,4}_val20_h20`;
their exact hashes are registered in
`docs/model_test_intent_and_capability_boundaries.md`.

N5 later supplied an important counterexample to interpreting natural liveness
alone. It reaches 20/20 at zero post-arm delay, but it is also effective at
recording step 0 in all 20 episodes and has outside-single-demo-local-support
directions in 4/20 armed rows. It reproduces 5/20 startup demo targets, and every
startup anchor has an anchor-extra effective direction. These are relations to
one demo, not 15 deadlocks or 20 wrong starts. The natural-startup test detects
willingness to enter any active region, not task correctness or idle safety.

### Accepted slice 4: train-calibrated all-axis response envelope

Implemented focused owners:

- `testbed/testbed/data/execution_response_envelope.py`;
- `testbed/testbed/cli/evaluate_response_envelope.py`;
- `testbed/tests/test_execution_response_envelope.py`.

The historical `direct_command_qvel_response_v1` sidecar was not changed.  The
new `all_axis_response_envelope_v1` contract consumes its causal command
alignment, calibrates qvel sign and stationary noise from the 120 training
episodes only, then validates train response cells on the unchanged 20-episode
chronological validation block.  Sealed test episodes remain unread.

The train calibration found qvel response signs `[+1, +1, -1, +1]` for
`[swing, boom, stick, bucket]`.  In particular, positive stick command maps to
negative measured stick qvel.  The old response sidecar avoided this error by
declaring stick unsupported; the new contract makes the mapping explicit and
train-derived instead of silently applying command sign to qvel sign.

After requiring four preceding same-axis idle ticks, four sustained command
ticks, a stationary baseline, and complete causal horizons, the train set has
438 clean from-rest onsets and validation has 84.  Of the 84 validation events,
68 fall in train cells with at least 10 events, 11 have weak support, and 5 are
out of support.  Train-cell response-probability Brier score falls from
`0.2285` at one tick to `0.1022` at five ticks and `0.0381` at 10/20 ticks.
Three validation events in otherwise supported cells still show no response by
20 ticks, concentrated in source episode 140's near-threshold stick/swing
commands.  This is response evidence that a fixed deadzone crossing alone
cannot represent.

Artifact root:
`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/response_envelope_all_axis_train120_val20_v1`.
The manifest records split/deadzone hashes, every source episode, artifact
hashes, and `sealed_test_read=false`.  Manifest SHA-256:
`97e722a86d1d4810b96540e86c674fab79042eb49cb1ae0f43995ea312c68d53`.

### Accepted slice 5: unsent policy-command support comparison

Implemented focused owners:

- `testbed/testbed/policies/response_support_eval.py`;
- `testbed/testbed/cli/evaluate_policy_response_support.py`;
- `testbed/tests/test_response_support_eval.py`.

This evaluator queries saved teacher-forced N0/N1/N5 commands at expert event
observations against the train-derived response cells.  Version 2 deliberately
keeps episode-action relation and historical physical-response evidence as two
orthogonal fields.  Insufficient similar-condition response data is never a
predicted failure or success, and it is excluded from model action-accuracy
failure counts because the model command was not sent.

At the 20 first validation events:

| Model | events with effective command | wholly within 40-tick local event | wholly within current episode actions | wholly within training event directions | effective command axes | sufficient / weak / insufficient response evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N0 | 11/20 | 10/20 | 10/20 | 11/20 | 21 | 5 / 0 / 16 |
| N1 | 18/20 | 17/20 | 17/20 | 18/20 | 33 | 11 / 0 / 22 |
| N5 | 20/20 | 16/20 | 19/20 | 20/20 | 41 | 7 / 4 / 30 |

N5's 41 first-event effective command axes decompose as 5 current-frame matches,
1 immediate 0--1 tick match, 3 near 2--5 tick matches, 5 near 6--10 tick
matches, 22 later-within-40-tick matches, 3 actions used in another phase of
the same episode, and 2 actions seen only in other training episodes.  Thus
39/41 axes occur in the current episode and all 41 occur in training events.
There are zero train-unseen directions.  This supports describing N5 as mostly
advancing known current-episode actions rather than inventing unknown actions.

The separate response-evidence axis says that 30/41 N5 commands lack enough
historical samples in the same magnitude/qpos/from-rest cell.  That is a data
coverage statement, not a model mismatch and not an evaluation failure.  It
only prevents claiming that those unsent commands have a known physical
response probability.  Direct policy-on causal logging remains necessary to
resolve the sparse response cells without requiring the model to imitate one
recorded timing exactly.

Current comparison artifact:
`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/response_envelope_all_axis_train120_val20_v1/policy_episode_action_and_response_evidence_n0_n1_n5_v2.json`,
SHA-256
`2b03fdbc1caac67e3f27bb48799d7511b365f7789dd5ae35a13c032a94dfee3e`.
The earlier `policy_response_support_n0_n1_n5.json` v1 artifact remains for
provenance but is superseded because its single `out_of_support` wording could
be misread as an action mismatch.
