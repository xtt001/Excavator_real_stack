# SimVerify transition-stitch and condition-causal evidence — 2026-07-25

Decision: `revise_condition`

Evidence scope: `recorded-observation/offline`

This bounded follow-up evaluates two separate questions:

1. can recorded one-step transitions be composed into a credible offline
   environment response; and
2. with the observation fixed, does B1 use the declared semantics and timing of
   `cycle_condition_v1`?

Neither experiment is closed-loop execution. Held-out episodes 1, 13, 25, and
33 were not read.

## Per-step transition stitching

The first retriever implements the proposed approach directly: at every tick it
selects the nearest recorded train transition and advances to that transition's
recorded successor. Retrieval excludes condition, phase/progress labels, and
successor information. Thresholds are calibrated on train source-episode
leave-one-out distributions before validation is evaluated.

The expert one-step prerequisite passed all four validation checks:

| Check | Allowed | Observed |
| --- | ---: | ---: |
| retrieval distance q97.5 | <= 0.821554 | 0.533525 |
| successor-state error q97.5 | <= 0.898185 | 0.578079 |
| progress-delta error q97.5 | <= 0.102189 | 0.045673 |
| phase-delta agreement | >= 0.971986 | 0.974885 |

But the same locally valid transitions did not compose:

- train: 13/111 expert cycles completed and 98/111 deadlocked;
- validation: 3/31 completed and 28/31 deadlocked.

A single-factor v2.1 retriever added only the preceding observable
state/action delta. Its one-step prerequisite also passed, but every cumulative
rollout left calibrated support:

- train: 111/111 `offline_support_exhausted`;
- validation: 31/31 `offline_support_exhausted`.

Therefore both gates are `offline_emulator_invalid` and
`authorizes_condition_rollout=false`. The finding is not that nearest
transitions are useless. They are valid for local one-step support and progress
diagnostics. The unsupported claim is that independently nearest local
successors form a valid cumulative excavator trajectory.

Artifacts:

- `/data/pingfan/Excavator_real_stack_data/simverify_transition_stitch_calibration_v1`
  - manifest SHA-256:
    `94bef97e0c593076d32e8487d628d56b303f97df9249126dbc7cfebfd49e6966`
  - checksums SHA-256:
    `eba9b591ee6b2baa9622418898d35baf8d34b94ce6a6ff1bb089075129e09ef3`
- `/data/pingfan/Excavator_real_stack_data/simverify_history_stitch_calibration_v1`
  - manifest SHA-256:
    `da5f8fd1a3ab25da69ebc903f5f172acb1761eeea2a63c1a98bbc85b5b33b3ef`
  - checksums SHA-256:
    `725fc42bba9a13e66d010390156f573ed0a08a8e887d2cd41ef8f786963e4c61`

Both checksum inventories were reverified.

## Fixed-observation condition causal v2

The v2 contract was frozen in
`docs/simverify_condition_causal_v2_contract.md` before the decision artifact
was computed. It adds two controls missing from a simple token-swap test:

- exact B1 masked replay, where both requested conditions are replaced with the
  same train-only canonical token; and
- all five non-identity semantic permutations of left/center/right.

The experiment used 45 supported validation anchors and source-episode
bootstrap with 100,000 repetitions.

| Criterion | current sector | next sector |
| --- | --- | --- |
| action effect > identical-token mask | pass | pass |
| signed semantic margin > shuffled-condition B2 | pass | pass |
| identity beats all five wrong semantic mappings | pass | fail, 3/5 |
| effect concentrated in the declared response phase | pass | fail |
| task event coverage/order preserved | pass | pass |
| factor decision | **pass** | **fail** |

### What passed

`current_sector` passed every frozen criterion across source episodes 12 and
34. Its mean B1 action effect was 0.030324 versus exact zero under masking. The
paired signed semantic-margin lower bound versus B2 was 0.009550, above repeat
noise 0.0000001405. Its phase-specific response was positive in both source
episodes and exceeded B2.

`next_sector` was not ignored. Its mean B1 action effect was 0.028491 versus
exact zero under masking, and its signed semantic margin separated from B2.
Thus the evidence does not support the statement that B1 simply discards the
next-sector token.

### What failed

The next-sector response was not identified with the declared semantics and
phase:

- only three of five wrong semantic mappings were rejected; for two mappings,
  the source-episode bootstrap lower bound of identity advantage was zero;
- phase specificity was negative in source episodes 12 and 34
  (`-0.011604`, `-0.022368`) and positive only in episode 20 (`0.025112`);
- the phase-specificity bootstrap lower bound versus the exact masked null was
  `-0.022368`, so the response was not concentrated from observable dump-end
  through ready-end;
- it also did not exceed shuffled-condition B2 phase specificity.

Validation support is narrow: next-sector evidence spans three source episodes,
and episode 20 contributes one supported anchor. Accordingly this is
`condition_understanding_not_established`, not proof that the network can never
learn the next-sector meaning.

Artifact:

- `/data/pingfan/Excavator_real_stack_data/simverify_condition_causal_v2_validation_v1`
  - manifest SHA-256:
    `6445a3b07eb971528ad3ed91a5d96ae74754be51e6ea271d1540fd9c8696b1c3`
  - checksums SHA-256:
    `5c4421c11234da32f5af7a750e8383356da86831de02ab894e11dbf329a8a8b2`

The checksum inventory was reverified.

## Plain-language conclusion

The model understands **where it is digging now** well enough for this offline
test. It notices **where it should go next**, but the evidence does not show
that it uses that information at the right time or with uniquely identifiable
left/center/right meaning. The per-step nearest-neighbor idea is useful as a
local plausibility checker, but current recorded transitions cannot be safely
chained into a substitute simulator.

The frozen M5 decision therefore remains `revise_condition`. This evidence does
not authorize held-out test, simulation closed loop, real fine-tuning, or
deployment.
