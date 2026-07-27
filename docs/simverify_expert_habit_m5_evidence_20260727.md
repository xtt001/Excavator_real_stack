# SimVerify Expert-Habit M5 Evidence — 2026-07-27

## Terminal decision

```text
terminal_decision=sim_observable_only
decision_scope=technical_capability_classification_only
deployment_authorized=false
held_out_test_read=false
physical_effect_validated=false
real_finetune_candidate=false
control_candidate=false
```

The B1 executor has demonstrated the requested fixed-scenario observable
capability in live AGX feedback: it can complete a ready-to-ready cycle, use a
dump-end committed condition to choose `left` or `center`, and continue into a
second cycle. This is not a physical digging-effect, unseen-scene, real-domain,
or deployment result.

## Why the previous AGX endpoint conclusion was invalid

The first formal AGX branches never armed the accepted-v11 ready detector.
B1 uses the simple low-dimensional condition and does not expose the optional
internal `condition_route` diagnostic, but the detector incorrectly required
`condition_route=next`. Every post-commit row therefore remained in
`searching_return_activation`; “no ready endpoint” was not a data or policy
result.

Commit `d031d82` corrected the existing contract:

- a valid nonzero `cycle_condition_v1` is the causal committed target;
- return motion arms the ready detector;
- an internal route, when present, remains an additional fail-closed check;
- no privilege or future observation was added.

Commit `07fdbdf` similarly corrected continuous B1 lifecycle handling. The
external scenario state clears and recommits the low-dimensional condition;
it does not call the phase-router reset API that B1 does not own.

The older no-endpoint AGX directories remain immutable diagnostic history, but
they are not endpoint-failure evidence.

## Paired condition intervention

All three corrected branches use:

- B1 checkpoint SHA-256
  `dde585a8a52dbf35ab3154bfc0883640a0a5b1a01ad1521d6a697478bf60acd2`;
- environment seed 0 and policy seed 0;
- deterministic FP32 inference and legacy temporal aggregation;
- the same checksum-bound 156-tick actual-sent-action prefix;
- the same initial state and causal dump-end branch point.

Only the committed target differs.

| Branch | Scripted target | Realized v11 ready | Completion tick |
| --- | --- | --- | ---: |
| reference | left | left | 217 |
| same-condition repeat | left | left | 219 |
| treatment | center | center | 233 |

The treatment-to-repeat mean absolute action difference ratio is
`12.64 / 4.85 / 3.55 / 3.75` over swing/boom/stick/bucket. The corresponding
qpos ratio is `9.49 / 5.33 / 3.54 / 7.46`. The condition effect exceeds
same-condition repeat variability on every axis.

At the center completion, numeric sector, visual sector, and ready visual
classifier all agree on `center`; absolute swing speed is `0.13379`, below the
frozen v11 threshold `0.14331`.

Immutable paired package:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_habit_b1_v2_agx_branch_eval_left_stay_vs_center_seed0_readyfix_v2
```

## Continuous fixed scenarios

### repeat_same

Scenario: `left -> left -> left`.

- condition commits: ticks `156` and `364`;
- first left-ready: tick `219`;
- second left-ready: tick `428`;
- requested/completed cycles: `2/2`;
- full policy reset count: `1` at run start only;
- temporal aggregation was not reset at the cycle boundary.

### move_adjacent then stay

Scenario: `left -> center -> center`.

- condition commits: ticks `156` and `343`;
- first center-ready after the adjacent move: tick `231`;
- second center-ready: tick `421`;
- requested/completed cycles: `2/2`;
- full policy reset count: `1` at run start only;
- temporal aggregation was not reset at the cycle boundary.

Each accepted endpoint simultaneously satisfies target qpos sector, low swing
speed, frozen ready visual classification, and frozen visual sector
classification. All five run directories pass their own SHA-256 inventories.

## Offline Gate is preserved

The immutable v2 offline Gate remains:

```text
basic_capability_established_offline=true
condition_understanding_established_offline=false
decision=condition_understanding_not_established_offline
```

Its two MAE-advantage criteria remain failed and were not rewritten. The AGX
paired intervention answers a different, higher-tier question: whether the
condition changes feedback execution and reaches the intended observable
endpoint despite non-trivial pointwise action error. It does, for the tested
fixed scene and targets.

## Evidence boundary

The current conclusion does establish:

- basic observable dig/dump/return execution in the tested AGX scene;
- causal target use after dump-end;
- left and center observable target-ready completion;
- two continuous `stay` cycles;
- an adjacent move followed by a second continuous cycle.

It does not establish:

- bucket fill, retained mass, dumped mass, or other physical-effect success;
- held-out source-episode or unseen-terrain generalization;
- right-target closed-loop completion;
- robustness across multiple Unity scene seeds;
- real camera/state/action domain transfer;
- real finetuning, shadow, deployment, or control readiness.

PACT and Unity remained read-only but dirty. Their commit, status SHA, and
working-diff SHA are bound into every run, so the result is auditable but
explicitly non-promotable.

## Immutable M5 package

```text
/data/pingfan/Excavator_real_stack_data/simverify_expert_habit_m5_v1

decision.json
  8659da0954ab2febe3eb5541af727ef6f831555dae45b1d1be347d1e8bbbc85e
m5_manifest.json
  a75150e6c62bb3c0dd68ae86493faea0c05093d60d27bf13d6bc9e859728815b
checksums.sha256
  f00978e653004941423e2ae41bdb55ed9fa27a39334d2c07d09f28b4732d436a
```

Checksum verification: `2/2`, zero failures.

## Plain-language conclusion

The earlier worry that the sliced data lacked a slowdown-to-ready phase was
wrong. The data and trained policy do contain that behavior; the evaluator was
not looking for it because it waited on an internal signal B1 never produces.

After fixing only the evaluation/lifecycle contract, the same checkpoint:

1. returned to left and stopped in left-ready when told `stay`;
2. returned to center and stopped in center-ready when told `step_right`;
3. repeated a second same-sector cycle without restarting the policy.

In plain language: **the model can do the basic simulated job and can read the
left/center condition in this fixed scene.** What remains unknown is whether
the bucket did the physical work well, whether this survives other terrain and
seeds, and whether any of it transfers to the real excavator.
