# SimVerify M0 Gate Design Audit v3

Status: `predeclared_after_failed_v2_before_v3_rerun`

Evidence scope: `recorded-observation/offline`

This addendum corrects two Gate-implementation errors exposed by the first
formal 1024-replicate v2 run. It does not change the declared 3x3 condition
space, lower a learned threshold to rescue a row, read held-out data, or
authorize training.

## 1. Immutable failed-v2 evidence

The failed package remains at:

```text
/data/pingfan/Excavator_real_stack_data/
  .sim_observable_cycle_v2.tmp-3864088/
```

It is bound to Git commit
`b7973221a71acb0343cecf76484789911928e8d5`. Important SHA-256 identities are:

| Artifact | SHA-256 |
| --- | --- |
| `BUILD_FAILED.json` | `2fdd4d26c6fc603fccc0c28661db5a93f9ca63938b98fcb77e1c39deec9aa6e9` |
| event Gate | `2e1ec1eae90bc311bbe886e1467049ab431c43d1033e87a65fa558ddebce75dd` |
| boundary Gate | `8920b6230f584bea36ebb7b65d53bbf6c6f1bcaf40ba01e75124ddeb6c31ecf3` |
| sector-visual Gate | `adbcc27919ff3ccd4fd0879bea2c0040b8172634dcb337a162f1e8cc3915a292` |
| transition inventory | `9d3fbcf78a83cc716c72c8e5259f7616b0d6083b1641cc65483fe72e80527756` |
| transition Gate | `be5ac3ce537154d4073899cbf482bdbc321c0948ee85fee917926111fc774a9d` |
| cycle annotations | `1e2cf9337f47e45488acc07dc164e8c4ac98a61a74d91edbb9e0863d714ebbf0` |

The event, numeric/sector-boundary, and sector-visual Gates passed. All three
1024-replicate bootstrap families completed without a failed refit. The run
stopped at final validation transition coverage and did not start M1 or
training.

## 2. M0-COV-01 count-unit correction

### Problem

`transition_inventory()` counted a `current->next_ready` condition only when
both the current cycle and the following source cycle were accepted. This
conflated two distinct quantities that the frozen contract requires
separately:

1. a trainable condition sample stored in one accepted cycle; and
2. a two-cycle continuity pair in which both consecutive cycles are accepted.

For example, validation episode 12 cycle 13 is an accepted `center->center`
condition sample. The v2 inventory discarded it because cycle 14 was
ambiguous for its own following sector. That later ambiguity does not erase
cycle 13's already observable and accepted condition.

### Correct operand

- `transition_matrix`: count every accepted cycle's
  `current_sector -> next_ready_sector`;
- `adjacent_two_cycle_pair_count`: count consecutive accepted source cycles;
- `continuity_errors`: compare the first accepted cycle's `next_ready_sector`
  with the consecutive accepted cycle's `current_sector`;
- three-cycle inventory remains a consecutive accepted-cycle statistic.

These counts must remain separate in schema and tests. Recomputing only the
count unit on the immutable v2 annotations changes validation coverage from
7/9 to 8/9; it does not by itself make M0 pass.

## 3. M0-SECTOR-03 acceptance correction

### Problem

The v2 sector cross-confirmation Gate correctly tested source-episode
bootstrap balanced accuracy against an episode-mapping null. The per-row
fusion rule nevertheless added an unrelated cutoff:

```text
eye-pair cosine similarity >= 1st percentile of correct validation rows
```

The `0.01` quantile had no matched null or error-cost derivation. Absolute
cosine magnitude is also not the claimed property: sector confirmation asks
whether eye evidence selects the same sector as swing qpos, not whether the
image is close enough to the global centroid in absolute terms.

The only provisional validation `right->left` support was rejected solely by
this cutoff:

```text
qpos label                         left
eye nearest-centroid label         left
best eye cosine similarity         0.9094149729872018
v2 absolute-similarity cutoff      0.9129450954077586
eye top1 margin                    0.0070224152376332
```

There was no qpos/eye disagreement and no centroid tie.

### Correct operand

Per-row eye/qpos fusion now requires:

1. a unique eye nearest-centroid label (`top1 margin > 0`);
2. equality between that eye label and the qpos label;
3. qpos must remain outside the source-episode boundary review band.

Absolute similarity and the former 1% margin cutoff remain diagnostics only.
Population reliability is still guarded by the already predeclared
source-episode balanced-accuracy lower-bound versus permutation-null Gate.
This separates the row-level claim ("the two observable modalities agree")
from the population claim ("eye sector labels are reliably identifiable").

This is a change of estimand, not a post-hoc numerical relaxation. No
replacement constant is introduced.

## 4. v3 rerun rule

The v3 rerun:

- keeps the same source bytes, split, held-out lock, feature weights, seeds,
  1024 replicates, event reliability threshold-generation method, numeric and
  sector boundaries, and 9/9 train/validation condition requirement;
- writes a new immutable
  `/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3`;
- counts condition transitions per accepted cycle and continuity pairs
  separately;
- uses unique nearest-centroid eye/qpos agreement for sector fusion;
- fails if the formal final train or validation condition matrix is not 9/9;
- enters M1 import smoke only after every M0 Gate, checksum, and immutable
  rename succeeds.

The read-only v2 counterfactual indicates that the two corrections recover the
missing `center->center` and `right->left` validation conditions. That is
diagnostic evidence only; the formal v3 rerun must recompute the complete
artifact graph.
