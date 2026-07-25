# SimVerify G4-v2 Transition-Stitch Contract

Status: `method_frozen_before_transition_bank_build`

Evidence scope: `recorded-observation/offline empirical rollout`

Closed-loop execution: `false`

Held-out test read: `false`

This is a new versioned experiment. It does not modify or overturn the
completed G4-v1 `revise_condition` decision. Its purpose is to test whether a
conditioned policy can accumulate observable progress when each emitted action
selects the next supported recorded transition instead of receiving the next
teacher-forced observation from one fixed cycle.

## Capability boundary

The transition stitcher is a non-parametric verifier, not a simulator.

It can test:

- whether policy actions stay inside recorded state-action support;
- whether supported one-step transitions accumulate observable phase progress;
- whether current/next condition interventions produce different supported
  paths under identical initial states and retrieval randomness;
- whether a policy deadlocks, backtracks, or exhausts offline support.

It cannot prove:

- environmental response outside the recorded transition bank;
- soil contact, payload, or physical excavation success;
- Unity/AGX, real-machine, shadow, or control readiness;
- closed-loop success.

## Split and provenance lock

- transition bank: accepted train cycles only;
- emulator calibration: train source-episode leave-one-out;
- emulator confirmation and condition experiments: validation only;
- held-out episodes `1`, `13`, `25`, and `33`: locked unread;
- original HDF5: read-only;
- every derived artifact: immutable, SHA-bound, and Git-provenanced.

Validation results from G4-v1 are treated as development evidence for this new
method. No G4-v2 threshold may be changed after held-out is opened.

## Transition unit

Each bank row is one adjacent 20 Hz transition:

```text
(observable_state_t, recorded_action_t) -> observable_state_t+1
```

Rows are included only when both ticks belong to one accepted cycle. No
cross-cycle boundary is inserted into the bank.

The retriever may select a row from any train source episode except the source
episode owning the current stitched node. This prevents a rollout from
degenerating into teacher-forced continuation of the selected episode.

## Retrieval inputs

The observable state vector contains:

- source-domain qpos `[4]`;
- source-domain qvel `[4]`;
- frozen eye-pair cosine similarity to the five observable event prototypes;
- frozen stick-pair cosine similarity to the five observable event prototypes;
- frozen eye-pair cosine similarity to the three sector prototypes.

The action vector is the four-axis source-domain action:

- expert self-replay: recorded action;
- policy experiment: future-runtime-safe policy action.

All scalar dimensions are centered and scaled using train-only median and IQR.
If a train dimension has zero IQR, its train standard deviation is used; if
both are zero, the scale is one and the constant dimension contributes zero.

State and action each contribute one RMS-normalized group:

```text
distance^2
  = mean(square(standardized_state_delta))
  + mean(square(standardized_action_delta))
```

No fitted scalar weight is introduced between the two groups.

## Forbidden retrieval inputs

The following may not affect nearest-neighbor selection:

- current or next condition token;
- recorded current/next sector outcome;
- desired target sector;
- cycle phase or progress label;
- candidate successor state;
- terrain, bucket mass, exact tip position, planner state, or other privilege;
- held-out data.

Cycle phase, progress, sector, and successor error are post-retrieval scoring
fields only. Selecting a candidate because its successor looks closer to the
desired outcome is forbidden future leakage.

## Frozen visual representation

The visual representation reuses the M0 local ImageNet ResNet-18 checkpoint,
camera mapping, image transform, pair ordering, feature normalization, and
event/sector prototypes. No network download, fine-tuning, or result-driven
feature selection is permitted.

The transition package stores compact prototype similarities and their
provenance. It does not introduce a learned dynamics model.

## Expert self-replay prerequisite

Before any policy is evaluated:

1. query every train transition against the train bank while excluding its
   source episode;
2. generate train leave-one-episode-out distributions for retrieval distance,
   successor observable-state error, progress-delta error, and phase-delta
   agreement;
3. query validation expert transitions against the train-only bank;
4. compare validation source-episode distributions with the frozen train
   envelopes;
5. run cumulative validation expert stitching from ready start while excluding
   the current stitched node's source episode at every step.

Support radius and successor-error limits are the corresponding train
leave-one-episode-out `q97.5` values. Phase agreement uses the train
leave-one-episode-out `q02.5` lower bound. No subjective success percentage is
inserted.

If the expert stitcher fails, policy testing stops with
`offline_emulator_invalid`. This is not a policy failure.

## Unsupported transition policy

A rollout stops with `offline_support_exhausted` when:

- no cross-source-episode candidate exists;
- nearest distance exceeds the frozen train support radius;
- candidate successor error cannot be bounded by the expert calibration;
- a non-finite feature/action appears.

Unsupported steps remain visible and are not imputed. They are neither policy
success nor physical failure.

## Condition experiment

Only after expert self-replay passes, start from the same supported validation
anchor and run paired empirical rollouts with identical retrieval tie-breaking:

- B1 base condition;
- B1 one-field current-sector swap;
- B1 one-field next-sector swap;
- B2 with the same condition requests;
- B1 masked-condition control using the most frequent train condition;
- B1 semantic-permutation scoring over every non-identity sector mapping.

Masked condition is excluded from retrieval just like every other condition.
Semantic permutation changes scoring only; it never changes transition
selection.

## Condition metrics

- supported rollout horizon;
- accumulated observable progress;
- event-order preservation;
- backward-progress and deadlock counts;
- endpoint observable sector;
- current-sector signed progress/sector effect;
- next-sector signed progress/sector effect;
- intended-window minus off-window phase specificity;
- B1-minus-B2 paired effect;
- B1-minus-masked paired effect;
- identity semantic score minus permutation-null score;
- source-episode bootstrap interval;
- transition-distance and branch uncertainty.

Action MAE remains auxiliary.

## Decision logic

The experiment may say the condition is understood offline only when both
current and next interventions:

- remain inside expert-calibrated transition support;
- produce a signed target-related effect above same-checkpoint repeat noise;
- exceed B2 and masked controls in source-episode paired bootstrap;
- exceed the exact semantic-permutation null;
- act primarily in their frozen intended phase window;
- preserve observable event order and non-target-axis semantics.

Failure to distinguish the null returns `revise_condition`. Emulator failure
returns `offline_emulator_invalid`. Neither result authorizes held-out,
deployment, or a closed-loop claim.
