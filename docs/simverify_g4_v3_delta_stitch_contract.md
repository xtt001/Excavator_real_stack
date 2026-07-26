# SimVerify G4-v3 Novel-Transition Delta-Stitch Contract

Status: `method_frozen_before_v3_validation`

Evidence scope: `recorded-observation/offline empirical rollout`

Closed-loop execution: `false`

Held-out test read: `false`

G4-v2 and G4-v2.1 remain immutable `offline_emulator_invalid` results. This
version tests a narrower interpretation of per-step stitching: each selected
recorded transition contributes only its local observable annotation delta.
The rollout never adopts the selected row's absolute cycle progress.

## Failure addressed

G4-v2 selected a locally supported transition and then replaced the rollout's
progress with that row's absolute progress. Ready-start and ready-end are
observationally similar, so this allowed one-step jumps to apparent completion
and caused unrelated absolute phases to alternate. G4-v2.1 added raw one-tick
history but did not remove the absolute-progress jump and exhausted its
high-dimensional history support.

The v3 method was selected using train-only leave-one-source-episode
development. No v3 validation rollout was inspected before this contract was
frozen.

## Immutable inputs

- observable state and action representation: G4-v2;
- transition bank: accepted train cycles only;
- one-step support radius: frozen G4-v2 train leave-one-source-episode-out
  threshold;
- validation queries: accepted validation cycles only;
- held-out episodes `1`, `13`, `25`, and `33`: locked unread.

The original HDF5 remains read-only. V3 additionally reads the recorded action
at `tick + 1` so expert calibration can issue a real source action after a
selected terminal-cycle transition. That value is stored with source episode,
tick, dataset, and checksum provenance. It is not used as a policy action.

## Per-step retrieval

At each step, the query contains only:

- current observable state;
- current executed action.

State and action use the frozen G4-v2 train-only normalization and equal
group-weight distance. A candidate must:

- come from a source episode different from the current donor episode;
- be inside the frozen one-step support radius;
- not have contributed to this rollout before.

The exact nearest remaining candidate is selected. Excluding an already-used
row is an evidence-accounting rule: one recorded transition may contribute at
most once to one rollout. It does not inspect phase, progress, condition, or
successor outcome.

## Delta integration

After selection:

```text
state_(t+1) = selected recorded successor state
action_(t+1) = recorded successor action       # expert prerequisite only
accumulated_progress += selected local (progress_(i+1) - progress_i)
```

The local delta is post-retrieval scoring metadata. Absolute candidate
progress, phase, successor sector, condition, and desired outcome cannot affect
selection. A rollout completes when accumulated local progress reaches five
ready-to-ready event intervals.

The rollout budget is the maximum accepted train-cycle transition count. It is
data-derived and frozen before validation. Unsupported retrieval, a non-finite
value, or budget exhaustion remains visible.

## Action-null prerequisite

Completion alone is insufficient because every accepted annotation has
positive local progress. Every expert rollout therefore has a paired
`median_action_null`:

- identical initial observable state;
- identical bank, support radius, novelty rule, and rollout budget;
- action query replaced at every step by the train median action (zero in the
  frozen standardized representation).

The expert path must complete inside train-derived envelopes, while the paired
action-null must not. This establishes that accumulated progress depends on the
recorded control input rather than merely consuming arbitrary positive
transitions.

## Gate generation

Train source-episode aggregates generate, without hand-entered performance
numbers:

- lower envelope for expert completion;
- lower envelope for paired expert-minus-null completion;
- upper envelope for median-action-null completion;
- upper envelope for expert completion steps;
- upper envelope for expert maximum retrieval distance.

Validation must satisfy every frozen envelope, and the inherited G4-v2
one-step validation Gate must remain passed. Otherwise the result is
`offline_emulator_invalid_v3`.

## Capability boundary

Passing authorizes only a separately versioned B1.4 offline policy-stitch
experiment. It does not prove physical response, soil interaction, simulation
closed loop, real-machine success, or deployment readiness. Policy retrieval
must still exclude condition, phase, progress, successor identity, future
state, and privilege.
