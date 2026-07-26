# SimVerify B1.4 next-sector-only condition contract

Status: `frozen_before_support_artifact_and_policy_implementation`

Evidence scope: `recorded-observation/offline`

Closed-loop execution: `false`

Held-out test read: `false`

## Question

Can the unchanged ACT backbone learn `next_sector` as the sole controllable
condition when the already validated observable router enables it only after
`dump_end_proxy`, while `current_sector` remains a hindsight annotation and is
not read by the policy?

B1.4 follows the terminal B1.3 `revise_condition` decision. It does not
overwrite B1.3 evidence.

## Why the contract changes

`current_sector` is a hindsight description of where the recorded current
cycle occurred. At a fixed recorded observation it is coupled to qpos and
images. Swapping it while keeping the observation fixed creates a contradictory
input that the present demonstrations do not identify as a command.

`next_sector` describes a future ready target. B1.3 already established
next-sector action sensitivity, signed semantic margin above the shuffled
null, intended-window phase specificity, and task-envelope preservation. Its
semantic-identifiability failure was caused by applying some wrong semantic
permutations to source episodes in which those permutations did not change the
expected direction.

B1.4 therefore makes one primary change: only `next_sector` is a policy
condition and causal intervention target. `current_sector` remains in the
immutable M0 annotation and transition inventory but has zero policy influence.

## Frozen policy structure

B1.4 retains the B1.3 dataset, source episodes, episode split,
normalization, image transform, observable phase router, ACT transformer,
action head, action chunk, optimizer, learning rate, batch size, seed, epoch
count, KL/deadzone losses, validation schedule, checkpoint selection, and
inference precision.

The low-dimensional source row remains the 14-dimensional
`qpos[4] + qvel[4] + cycle_condition_v1[6]` record for dataset and checkpoint
compatibility. The policy projection is:

```text
state_projection(qpos, qvel)
+ I(route == next) * next_projection(next_sector_onehot3)
```

The current-sector slice is not read. In `current` and `neutral`, condition
influence is exact zero. The VAE encoder continues to receive qpos, qvel, and
actions only.

## Support prerequisite

Support is frozen before B1.4 training from the existing M2 counterfactual
anchor registry.

1. Only locally supported `next_sector` anchors are considered.
2. The minimum supported-anchor count per source episode is the ceiling of the
   train source-episode q02.5 count.
3. Validation source episodes below that train-derived minimum are excluded
   before any model metric or bootstrap is computed.
4. For each of the five non-identity sector permutations, only anchors for
   which that permutation changes the expected train-derived swing direction
   are informative.
5. Every permutation must have informative anchors in at least two eligible
   validation source episodes. Two is the minimum that permits between-source
   bootstrap variation.
6. Sector swing centers and all numeric support thresholds are fit from train
   annotations only.
7. Held-out episodes `1,13,25,33` remain unread.

Failure stops before policy training with
`insufficient_next_condition_semantic_support`.

## Candidate and null

- B1.4 uses the observed hindsight `next_sector` association.
- B2.4 uses the same architecture and deterministic train-valid-start
  condition shuffle. Because current-sector has zero policy influence, the
  effective changed factor is only the next-sector association.

No other training or model factor may differ.

## Fixed-observation semantic Gate

After completed B1.4 and B2.4 training:

1. run three requested-condition B1.4 validation replays;
2. run one exact masked B1.4 replay;
3. run one requested-condition B2.4 replay;
4. select only support-prerequisite-eligible source episodes;
5. compute action sensitivity, signed semantic margin, phase specificity, and
   task-envelope preservation at source-episode level;
6. for each wrong semantic permutation, compute identity-minus-permutation
   only from its predeclared informative anchors;
7. require B1.4 to exceed exact mask, B2.4, and repeat noise as applicable.

All five semantic permutations and every other criterion must pass.
Sensitivity alone is not condition understanding.

## Transition-stitch authorization

Only a complete next-sector semantic Gate pass authorizes a separately frozen
expert-only support calibration for per-step transition stitching. Stitching
may not use condition, phase, progress, successor identity, future state, or
privilege as a retrieval shortcut. Support exhaustion stops the rollout.

## Terminal decisions

- support prerequisite fails:
  `insufficient_next_condition_semantic_support`;
- next-sector semantic Gate fails: `revise_condition`;
- semantic Gate passes but no expert-valid stitcher exists:
  `condition_understanding_established_offline_stitch_unavailable`;
- semantic Gate and supported stitching pass:
  `condition_understanding_and_supported_progress_established_offline`.

None is a control candidate. None proves simulator or real closed-loop
execution, and none authorizes Jetson, real hardware, fine-tuning, deployment,
or checkpoint promotion.
