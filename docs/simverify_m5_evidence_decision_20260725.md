# SimVerify M5 Evidence Decision — 2026-07-25

Decision: `revise_condition`

Evidence scope: `recorded-observation/offline`

This is a terminal evidence decision for the current frozen experiment
version, not a closed-loop result and not a checkpoint promotion.

## Gate path

| Gate | Result | Consequence |
| --- | --- | --- |
| G0 boundary isolation | pass | observable-only experiment boundary retained |
| G1 source/data contract | pass | canonical 20 Hz source-domain data retained |
| G2 observable annotation | pass | observable labels and support index retained |
| G3 B0 observation baseline | pass, offline only | B1/B2 authorized |
| G4 condition response | `revise_condition` | G5/G6 and held-out remain locked |
| G5 two-cycle continuity | not entered | prerequisite G4 did not pass |
| G6 runtime/deployment equivalence | not entered | no promotable candidate exists |

## Why the decision is not a data rejection

Both current- and next-sector condition swaps produced a B1 action effect
above the paired B2 null and measured repeat uncertainty. Current-sector
direction, latency, and phase preservation also passed. The support minimum
was met.

The terminal issue is causal identifiability of the condition design:

- next-sector direction did not separate from B2 under source-episode paired
  bootstrap;
- shuffled-condition B2 was itself token-sensitive, so it did not provide an
  ignored-condition null for `condition_ignored_rate`.

The correct label is therefore `revise_condition`, not `reject` and not
`sim_observable_only`.

## Explicit non-claims

- no Unity/AGX or real closed-loop execution occurred;
- no held-out test episode was read;
- no real-control, Jetson, shadow, or deployment configuration changed;
- no checkpoint is authorized for real hardware;
- no `control_candidate` exists;
- offline replay is not claimed as closed-loop success.

## Post-decision bounded follow-up

The fixed-observation causal v2 and per-step transition-stitch calibration are
recorded in
`docs/simverify_transition_and_condition_causal_evidence_20260725.md`.
They do not reopen G5 or held-out test. The additional evidence keeps the
terminal decision at `revise_condition`: current-sector understanding passes,
while next-sector semantic identifiability and response-phase specificity do
not.

The subsequent B1.1 hard pre-dump next-sector randomization is recorded in
`docs/simverify_b1_1_phase_randomization_evidence_20260726.md`. It partially
improved next-sector phase timing but weakened semantic evidence and regressed
current-sector evidence. B1.1 is rejected as a revision candidate; the terminal
decision remains `revise_condition`.
