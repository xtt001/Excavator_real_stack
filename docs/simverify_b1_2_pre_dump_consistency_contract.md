# SimVerify B1.2 pre-dump counterfactual-consistency contract

Status: frozen before implementation and training.

Evidence scope: `recorded-observation/offline`

## Hypothesis

B1.1 showed that removing the early next-sector association can improve timing,
but hard supervision under a false next token weakens semantics and also
regresses current-sector evidence. B1.2 tests a narrower intervention: retain
the true supervised sample and use a false next token only in an auxiliary
pre-dump invariance comparison.

## One primary factor

B1.2 is identical to frozen B1 except for one auxiliary paired-consistency
term on chunk-safe pre-dump train samples.

For training start `t0`, ACT chunk length `Q=20`, and observable dump-end target
tick `d`, the auxiliary pair is eligible only when:

```text
t0 + Q <= d
```

The pair holds images, qpos, qvel, current-sector, checkpoint state, and action
chunk fixed. One branch receives the recorded next-sector and the other
receives a deterministic exact-marginal derangement. Expert action supervision
is applied only to the recorded-condition branch.

The auxiliary comparison runs both branches in ACT inference mode with the
same deterministic zero latent. It minimizes mean absolute difference across
all 20 normalized action queries and four action axes.

## Loss-scale contract

The auxiliary term uses coefficient `1.0`. This is not selected from validation:
it uses the same normalized-action L1 unit and mean reduction as the primary
imitation term, so coefficient one is the identity scale. No sweep, held-out
read, or result-dependent adjustment is allowed.

Validation receives only recorded conditions and has zero auxiliary loss.
Checkpoint selection therefore remains based on the original validation
objective.

## Fixed provenance

- source configuration: `simverify_b1_conditioned_v1`;
- annotation registry:
  `/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3/cycle_annotations.jsonl`;
- annotation SHA-256:
  `bb531a1f8283362b4841f99f603e19d1725cd61cf571a75865c1bee3c8aca5e5`;
- derangement seed: `20260726`;
- phase boundary:
  `dump_end_proxy.representative_target_tick`;
- held-out episodes: `1,13,25,33`, locked unread.

Architecture, source episodes, split, normalization, image transform, optimizer,
KL/deadzone losses, seed, epoch count, and inference condition remain identical
to B1.

## Gate

After completed training, B1.2 must run:

1. three requested-condition validation replays;
2. one identical-token masked replay;
3. the frozen fixed-observation causal v2 Gate with B2 as null;
4. a source-episode comparison against frozen B1 and rejected B1.1.

Both current and next factors must pass action sensitivity, signed semantic
margin, all five semantic permutations, phase specificity, and task-envelope
preservation. Failure keeps `revise_condition`, held-out, G5, and deployment
locked.

## Non-claims

This is not closed-loop execution. It does not use privilege, PACT imports,
Unity/AGX, Jetson, real hardware, or real-control defaults. No checkpoint from
this experiment is deployable.
