# SimVerify Gate Design Audit — 2026-07-25

Status: `G0_G2_passed_M2_contract_implementation_in_progress`

Evidence scope: `recorded-observation/offline`

This audit explains why each Gate exists, what it measures, where its
threshold comes from, and what a failure means. It does not introduce a new
fixed success percentage, read held-out test data, start training, or claim
closed-loop performance.

## Audit conclusion

The current data have not failed an arbitrary numerical requirement.

- G0, G1, and G2 passed on the immutable M0 package.
- M1 import smoke passed.
- G3, G4, and G5 do not yet have finite final numerical thresholds because the
  required B0 repeated-replay noise and B2 shuffled-condition null artifacts
  do not yet exist.
- M2 may freeze questions, trace schemas, intervention anchors, event
  extraction semantics, and runtime scheduling semantics. It may not fill in
  the missing model-derived distributions or inspect held-out test data.
- G6 remains pending because no candidate checkpoint or deployment-equivalence
  artifact exists.

The only current data limitation relevant to later Gates is a declared support
limitation: accepted train and validation cycles cover all 9/9
`current -> next` transitions, while individual counterfactual anchors may
still be unsupported in similar observable state. Unsupported anchors must be
reported and excluded from a supported-counterfactual success denominator;
they are not converted into failures.

## Gate dependency chain

```text
G0 boundary isolation
  -> G1 source/data contract
    -> G2 observable annotation reliability
      -> M2 evaluation contract
        -> G3/B0 observation-only baseline
          -> G4/B1 vs B0/B2 condition causality
            -> G5 two-cycle continuity
              -> G6 runtime equivalence and bundle safety
```

This ordering prevents a downstream model result from hiding an upstream
contract, label, or evaluation error.

## G0 — Repository and input boundary

### Design reason

G0 protects the meaning of every later result. If policy code imports PACT,
simulator runtime, or privileged state, an apparently successful replay can no
longer be attributed to the four cameras, source qpos/qvel, and explicit
condition that a deployable policy would actually receive.

### Gate operands

- policy, training, checkpoint, and evaluation semantics live in real stack;
- no PACT or simulator runtime import;
- no privileged observation in policy input or main evaluation;
- source HDF5 remains immutable;
- derived artifacts have provenance and checksums.

### Threshold source

This is a categorical contract Gate: violation count must be zero. Zero is not
a fitted performance percentage; one forbidden dependency is enough to
invalidate the claimed observable-only experiment.

### Failure meaning

A failure means the experiment boundary is contaminated. It says nothing
about whether the recorded trajectories are good or bad.

### Current status

Passed for M0/M1. M2 continues to enforce the same boundary.

## G1 — Canonical data contract

### Design reason

Before measuring policy behavior, the same row must mean the same physical
source sample everywhere. Camera role, action axis, state representation,
simulation time, resampling, and condition alignment cannot be inferred after
training without creating leakage or silent target changes.

### Gate operands

- frozen physical camera roles;
- source-domain qpos/qvel and `actuator_speed_cmd`;
- `step_id * dt` simulation time and 20 Hz export;
- zero action offset;
- same source row for four images, qpos, qvel, action, and condition;
- episode-level split isolation;
- source and derived SHA-256 inventories.

### Threshold source

Schema and alignment checks are exact. Resampling loss is evaluated by a
task-relevant preservation rule: every source action-sign segment lasting at
least one 20 Hz control period must survive. Shorter missed segments remain
reported but do not become an invented model-performance cutoff.

### Failure meaning

A failure means the training/evaluation operands are not reproducible or are
misaligned. It does not mean ACT is incapable of the task.

### Current status

Passed. The formal QC found 11292 valid action-sign segments, preserved 11229,
and missed 63; all 63 misses were shorter than 50 ms, so durable misses were
zero. Maximum preserved onset delay was below one 20 Hz tick.

## G2 — Observable annotation reliability

### Design reason

Conditioned-cycle evaluation requires cycle boundaries, ready envelopes,
events, and sectors that can be recovered from deployable observations. If the
labels require terrain grids, exact tip position, bucket mass, or planner
state, the policy would be trained or evaluated against information it cannot
observe.

### Gate operands

- action/qpos/qvel numerical candidates;
- frozen eye/stick visual confirmation;
- interval and confidence output rather than hidden point certainty;
- source-episode bootstrap stability;
- eye-sector discrimination against an episode-mapping null;
- qpos/eye sector agreement;
- train and validation transition inventory;
- physical privilege isolation.

### Threshold source

The thresholds are generated from source-episode resampling, null
distributions, confidence intervals, cluster separation, and Wilson bounds.
The v2 failure audit removed an unmatched absolute-cosine cutoff and corrected
the transition count unit before the independent v3 rerun. No fixed accuracy
percentage was chosen to make the data pass.

### Failure meaning

A failure would mean the labels are not identifiable or stable under source
episode changes. The correct response would be `revise_annotation`, not
training around the problem.

### Current status

Passed on the immutable v3 package. Accepted rows are separate from 350
review/ambiguous cycles; held-out test remains unread.

## G3 — Unconditioned cycle baseline

### Design reason

G3 asks whether recorded observation alone contains enough information for ACT
to emit the major task phases. Without B0, a failed conditioned model is
ambiguous: the problem could be observation insufficiency, model/training
failure, or condition design. B0 provides the required causal baseline.

### Gate operands

- complete train/validation recorded paths;
- raw normalized policy chunk;
- raw direct-unit policy chunk;
- temporal-aggregation action;
- future runtime-safe action;
- observable task-event extraction;
- effective-axis and direction semantics;
- repeated replay of the same B0 checkpoint.

### Threshold source

The expert train/validation event and duration envelope is fitted first.
Repeated B0 replay then measures checkpoint/runtime noise. Final G3
thresholds are frozen only after both inputs exist and are source-episode
bootstrapped.

### Failure meaning

A failure means the unconditioned observation-only baseline cannot reproduce
the major recorded-path action semantics. It does not prove a physical
closed-loop failure and does not authorize blaming condition.

### Current status

Pending. M2 freezes the evaluator; M3 B0 training/replay has not started.

## G4 — Condition response

### Design reason

Better action MAE cannot establish that the model used condition. G4 therefore
requires a controlled intervention: hold the recorded observation history
fixed and change exactly one condition field. B1 must show stable,
target-related response relative to both B0 and a shuffled-condition B2 null.

### Gate operands

- one-field current-sector or next-sector token swap;
- supported similar-observation anchors only in the success denominator;
- action effect, direction, latency, repeat consistency, and current/next
  sensitivity;
- phase preservation and unexpected/opposite effective-axis rates;
- B1 paired against B0 and B2 by source episode.

### Threshold source

B2 defines the null distribution for “condition present but not used.”
B0/B1/B2 source-episode paired bootstrap defines effect uncertainty. The final
finite thresholds cannot be generated before these artifacts exist.

### Failure meaning

- B1 approximately equals B2: `condition_ignored`;
- response exists but is unstable or task-destructive:
  `revise_condition` or model revision;
- unsupported anchor: evidence unavailable, not automatic model failure.

### Current status

Not authorized until B0 evidence exists. M2 only freezes intervention anchors
and denominator rules.

## G5 — Two-cycle continuity

### Design reason

A model can look plausible inside isolated windows yet fail at the ready
boundary, retain the old condition, or enter an uncontrolled action. G5 tests
the stateful transition between two consecutive accepted cycles without
claiming environmental response.

### Gate operands

- consecutive accepted-cycle recorded paths;
- necessary events in both cycles;
- ready-boundary action discontinuity;
- second-cycle condition activation;
- camera, state-hold, and delay/latest-wins diagnostics;
- source-episode aggregation rather than a single showcase trajectory.

### Threshold source

Expert train/validation ready-boundary distributions define task-compatible
envelopes; B0 repeated noise and B2 null complete the model comparison. The
final Gate is source-episode bootstrapped.

### Failure meaning

A failure means recorded-path two-cycle semantics or boundary handling are not
reliable. It is not evidence that a physical excavator completed or failed two
cycles.

### Current status

Pending B0/B1/B2. M2 freezes valid two-cycle anchors only.

## G6 — Runtime equivalence and bundle safety

### Design reason

Passing an offline model Gate is insufficient if compilation, scheduling,
stale-action handling, or packaging changes task semantics. G6 separates model
evidence from deployment-path equivalence and keeps sim-domain checkpoints
physically barred from real control.

### Gate operands

- reference FP32 versus compiled FP32 semantic equivalence;
- delay, stale offset, timeout, repeat-last, and latest-wins traces;
- bundle preflight and artifact SHA;
- explicit sim-domain and real-control-disabled metadata.

### Threshold source

Equivalence tolerances must be derived from frozen effective-action/event
semantics and measured replay variability. Safety metadata and forbidden
promotion are exact categorical checks.

### Failure meaning

A failure means optimization or runtime packaging altered the evaluated
semantics, or the bundle is unsafe to hand off. It does not retroactively
change G3–G5 model evidence.

### Current status

Pending. No checkpoint promotion or deployment is authorized.

## Final threshold-generation contract

`gate_thresholds_v1.json` may be generated only after all of the following
exist:

1. frozen source-episode split and immutable M0/M2 manifests;
2. expert train/validation envelopes;
3. B0 repeated-checkpoint replay/noise artifact;
4. B1 conditioned replay artifact;
5. B2 shuffled-condition null artifact;
6. source-episode paired-bootstrap calculation with inputs and seeds recorded.

The file, method, sample counts, quantiles, and SHA-256 are then frozen before
held-out test is opened. Held-out results may select only pass, reject, or a
new versioned experiment; they may not tune the frozen thresholds.

## M2 audit corrections applied

The M2 implementation was changed during this audit to preserve the Gate
logic:

- token swaps now change exactly one of `current_sector` or `next_sector`;
- unchanged condition fields no longer require artificial counterfactual
  support;
- unsupported swaps are retained for visibility but excluded from the success
  denominator;
- M2 fails unless the M1 report is passing, offline-only, training-free, and
  held-out-free;
- raw normalized chunk, raw direct chunk, temporal aggregation action, and
  future runtime-safe action must be independently stored without memory
  aliasing;
- delay/latest-wins uses explicit issue tick, ready tick, action age, stale
  offset, and strict timeout-to-zero semantics.

These changes constrain the experiment; they do not lower any model Gate.

