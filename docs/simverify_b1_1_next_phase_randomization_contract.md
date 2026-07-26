# SimVerify B1.1 next-sector phase-randomization contract

Status: frozen before implementation and training.

Evidence scope: `recorded-observation/offline`

## Hypothesis

B1 responds to `next_sector`, but the response is not concentrated in the
frozen `dump_end_proxy -> ready_end` window. The candidate cause is a
source-episode correlation between the next-sector token and observations from
earlier phases of the same cycle.

## One primary factor

B1.1 is identical to B1 except for deterministic train-only randomization of
the next-sector one-hot field on chunk-safe pre-dump training starts.

For a training sample beginning at target tick `t0`, with ACT chunk length
`Q=20` and the cycle's observable dump-end target tick `d`, randomization is
eligible only when:

```text
t0 + Q <= d
```

Thus every supervised action in the sampled chunk is before the declared
next-sector response window. Starts whose action chunk crosses `d`, all starts
at or after `d`, all validation rows, and all normalization statistics retain
the recorded condition.

Only `condition[3:6]` is reassigned. `condition[0:3]`, images, qpos, qvel,
actions, masks, episode split, optimizer, architecture, loss, seed, and
training length are unchanged.

## Randomization and provenance

- scope: train valid starts only;
- field: `cycle_condition_v1.next_sector`;
- seed: `20260726`;
- assignment: deterministic exact-marginal-preserving categorical
  derangement when the observed marginal admits one;
- annotation source:
  `/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3/cycle_annotations.jsonl`;
- annotation SHA-256:
  `bb531a1f8283362b4841f99f603e19d1725cd61cf571a75865c1bee3c8aca5e5`;
- phase boundary:
  accepted train annotation
  `observable_events.dump_end_proxy.representative_target_tick`;
- held-out episodes `1,13,25,33` remain unread.

The immutable training metadata must record eligible-row counts, source and
randomized next-sector marginals, changed fraction, mapping SHA, annotation
SHA, action chunk size, and phase rule.

## Gate

After completed B1.1 training:

1. run three requested-condition validation replays;
2. run one identical-token masked B1.1 replay;
3. reuse the frozen fixed-observation causal v2 Gate with B2 as the shuffled
   condition null;
4. compare B1.1 with the already frozen B1 result.

B1.1 succeeds only if:

- current-sector passes every existing causal criterion;
- next-sector passes action sensitivity, signed semantic margin, all five
  semantic permutations, phase specificity, and task-envelope preservation;
- the result is repeat-stable at source-episode bootstrap level.

Failure does not authorize a threshold change or held-out read. It routes to a
separate data-support revision, not a larger model or a new condition-token
architecture.

## Non-claims

This experiment does not use Unity/AGX, PACT imports, Jetson, real hardware,
privileged state, or closed-loop execution. No resulting checkpoint is
deployable.
