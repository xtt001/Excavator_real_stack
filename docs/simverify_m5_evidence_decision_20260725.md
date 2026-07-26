# SimVerify M5 Evidence Decision — 2026-07-25 to 2026-07-26

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

The subsequent B1.2 pre-dump counterfactual-consistency experiment is recorded
in `docs/simverify_b1_2_pre_dump_consistency_evidence_20260726.md`. It retained
the true expert-supervised condition and applied the false next token only to a
deterministic inference-pair consistency term. B1.2 produced the best mean
next-sector phase specificity of B1/B1.1/B1.2, but reduced next-sector action
effect from `0.028491` to `0.003686` and signed semantic margin from `0.062237`
to `0.005483` relative to B1. Both current and next factors failed the complete
causal v2 criteria. B1.2 is rejected as a revision candidate.

## Final immutable M5 package

The terminal decision is materialized at:

```text
/data/pingfan/Excavator_real_stack_data/simverify_m5_decision_v2
```

Builder Git commit:
`157eb3f3147d3a16783e2ff85918d001add13307`

| Artifact | SHA-256 |
| --- | --- |
| `decision.json` | `c73b13f3b92a604b97b4bbe4fb111e4277fc94fdf4691df1496ddef2abfe8fab` |
| `m5_manifest.json` | `c09b2c9c17624a4700a2a0d48a30fda9434cf8950faa46004b3661d101feced6` |
| `checksums.sha256` | `e34aa7b0e531c1fd562dccb729b1ab7e3e97c73293dbcaab05b144548a279d90` |

Independent `sha256sum -c checksums.sha256` verification passed.

The builder reverified:

- all 55 M0 checksum entries;
- the M1 report and M0 manifest linkage;
- all nine M2 checksum entries and the M0/M1 linkage;
- the passing G3 package and its four replay packages;
- B1, B1.1, and B1.2 causal v2 packages;
- 17 unique replay packages referenced across G3 and G4;
- exact train/validation-only episode boundaries and zero held-out overlap.

The machine-readable terminal fields are:

```text
decision                       = revise_condition
terminal_for_experiment_version = true
evidence_scope                 = recorded-observation/offline
held_out_test_read             = false
held_out_test_authorized       = false
closed_loop_execution          = false
gate_thresholds_v1_generated   = false
sim_observable_only            = false
real_finetune_candidate        = false
control_candidate              = false
real_control_allowed           = false
G5                             = not_entered
G6                             = not_entered
```

This closes the current frozen experiment version at M5. A future experiment
may start only after freezing a new one-factor condition representation or
routing contract. It cannot reuse this M5 result to unlock held-out, G5/G6,
shadow, or real control.
