# SimVerify Expert-Habit B0/B1/B2 Evidence — 2026-07-27

## 1. Decision

```text
basic_capability_established_offline=true
condition_understanding_established_offline=false
decision=condition_understanding_not_established_offline
evidence_scope=recorded-observation/offline
teacher_forced_replay=true
closed_loop_execution=false
observable_cycle_completed_by_policy=false
physical_effect_validated=false
held_out_test_read=false
```

The first ready-to-ready training experiment establishes that the ACT executor
reproduces the observable action grammar on recorded validation observations.
It does not establish the full conditioned executor:

- B1 reacts to a target intervention with the correct swing direction much more
  often than the shuffled-target B2 null;
- B1 does not obtain a source-episode-stable post-commit action-error advantage
  over B2;
- B1 is significantly worse than unconditioned B0 in the post-commit window.

This is evidence of a learned target-direction signal, but not sufficient
evidence that the condition improves execution. The held-out test therefore
remains locked.

## 2. Frozen definition and dataset

Definition package:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_habit_cycle_definition_v5
```

- decision: `accept`;
- definition manifest SHA-256:
  `3e784dfc91990c49d786d595f488a6891679d9fe45cdab74e428744e8986565c`;
- held-out observation read count: `0`.

Derived dataset:

```text
/data/pingfan/Excavator_real_stack_data/
  sim_expert_habit_ready_cycle_v1
```

- one immutable HDF5 per full ready-to-ready cycle;
- 54 cycles: 37 train and 17 validation;
- 15,410 total 20 Hz rows;
- train intents: 13 `stay`, 11 `step_left`, 13 `step_right`;
- validation intents: 9 `stay`, 4 `step_left`, 4 `step_right`;
- dataset manifest SHA-256:
  `2b6876a53c1ea316ebf09dee3fd6c3d9d710c36a1fa196277f99569ad5a24bc8`;
- split SHA-256:
  `fde7880560491149f1fba22965512a2981c858a4d95541a43cb58dd98b815a50`;
- frozen scenario manifest SHA-256:
  `5eaedc4d30a6ce4cd1625563dff9891d8d59779d3b4ab73d9477442a5264ad69`.

The condition is inactive zeros through dump end. Starting at the first 20 Hz
row strictly after dump end, it atomically becomes current-sector one-hot plus
committed-target one-hot. The historical target source remains explicit
hindsight; no recorded command is claimed.

## 3. Matched training

All three runs use seed 0, 2000 epochs, batch size 4, four cameras, qpos/qvel,
20-step ACT chunks, the same backbone, optimizer, temporal aggregation
configuration, source-episode split, and checkpoint-selection rule.

| Baseline | Condition association | Best epoch | Best validation loss | Best checkpoint SHA-256 |
| --- | --- | ---: | ---: | --- |
| B0 | absent | 1870 | 0.237142 | `319c72586724ad87ce244e35656da88b9f838d8c63cbd03ffc6a99d2ec14273c` |
| B1 | correct hindsight target after dump end | 1230 | 0.263701 | `01a672354165685034683657e190a4b9adcd332b9e538dd79b9d59c55805d0ec` |
| B2 | train-only matched target shuffle | 1970 | 0.264875 | `ef219c744f4c4f4d69877ebe208f23025a9b20e03d5f534ab2a7b82ce8796596` |

B2 preserves current sector, pre-dump zeros, committed-row count, and exact
target-label marginals. Its frozen mapping changes 1,634 of 1,784 committed
train starts (`91.59%`) and has SHA-256
`b398dbf49d35ff935563488f1928ce97d1e7cabee595ad4710ac70b6c4057c98`.

The three `run_metadata.json` files record `status=completed`. B1 and B2 share
the same normalization stats SHA-256
`e97ab8f770b9e386bb64825c4c44e95a0e1b8a1193705e4494d1e575d8cb5f69`.

## 4. Validation replay

Immutable package:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_habit_validation_v1
```

- replay manifest SHA-256:
  `81bdf75002a8fe8b2c82461f3afa20fdec7a6795d9e83309259718f2aab2277d`;
- checksums file SHA-256:
  `e404009af596f4d4ae206f1d8d10ad06c1629d29767e7db5147ae75e1150f2b9`;
- checksum verification: `89/89`, zero failures;
- 17 validation cycles from source episodes 12, 20, and 34;
- 85 replay traces;
- held-out source episodes 1, 13, 25, and 33 were not read.

Each trace separately preserves:

1. normalized raw policy chunk;
2. source-domain raw policy chunk;
3. temporal-aggregation action;
4. future runtime-safe action.

The current runtime-safe action equals temporal aggregation in this offline
experiment, but it is stored as an independent array and is not aliased.

### 4.1 Basic action capability

| Metric | B0 | B1 | B2 | Expert |
| --- | ---: | ---: | ---: | ---: |
| full-cycle episode-macro MAE | 0.16909 | 0.16477 | 0.16726 | — |
| pre-dump episode-macro MAE | 0.18311 | 0.17650 | 0.17893 | — |
| post-commit episode-macro MAE | 0.09357 | 0.10303 | 0.10490 | — |
| required-event coverage | 99.02% | 100.00% | 100.00% | 95.10% |
| event-order valid rate | 100.00% | 100.00% | 100.00% | — |
| effective-action direction recall | 89.25% | 90.63% | 90.49% | 100% by definition |

The basic gate compares B1 against a zero-action null using source-episode
bootstrap, not a hand-selected percentage:

- event-coverage advantage: estimate `0.66667`, 95% CI
  `[0.66667, 0.66667]`;
- effective-action recall advantage: estimate `0.90368`, 95% CI
  `[0.89131, 0.91098]`.

Both pass. This is recorded-observation action-grammar evidence, not policy
cycle completion.

### 4.2 Condition intervention

The intervention keeps every recorded image, qpos, qvel, and policy history
fixed and changes only the committed target one-hot after dump end.

| Metric | B1 | B2 |
| --- | ---: | ---: |
| supported anchors | 17 | 17 |
| post-commit action effect L1 | 0.01811 | 0.01047 |
| pre-dump action effect L1 max | 0.0 | 0.0 |
| swing semantic-direction correct | 15/17 (88.24%) | 2/17 (11.76%) |

The swing action-to-qpos sign was fitted on 7,732 active train samples using
only observable action and qvel. Its median action-times-qvel is `0.29268` and
the fitted sign is `+1`.

Semantic source-episode bootstrap criteria pass:

- B1 direction versus 50% chance: estimate `0.32222`, 95% CI
  `[0.16667, 0.50000]`;
- B1 direction advantage versus B2: estimate `0.67407`, 95% CI
  `[0.33333, 0.88889]`;
- pre-dump causal localization: observed `0.0`, numerical threshold `1e-7`.

Execution-utility criteria fail:

- B0 minus B1 post-commit MAE: estimate `-0.01076`, 95% CI
  `[-0.02631, -0.00191]`;
- B2 minus B1 post-commit MAE: estimate `0.00054`, 95% CI
  `[-0.00282, 0.00540]`.

The first criterion shows B1 is consistently worse than B0. The second shows
that B1's small mean advantage over B2 is not stable across source episodes.

## 5. Post-hoc early-window diagnostic

This diagnostic does not alter the frozen gate. It checks whether averaging the
whole return phase hid an early condition benefit.

| First post-commit window | B0 MAE | B1 MAE | B2 MAE |
| --- | ---: | ---: | ---: |
| 5 ticks / 0.25 s | 0.17435 | 0.20068 | 0.18381 |
| 10 ticks / 0.5 s | 0.13538 | 0.15567 | 0.14387 |
| 20 ticks / 1.0 s | 0.10699 | 0.12120 | 0.11645 |
| 40 ticks / 2.0 s | 0.09333 | 0.10313 | 0.10404 |
| all committed rows | 0.09357 | 0.10303 | 0.10490 |

B1 is worse than both controls during the first 1 second in every validation
source episode. The failed execution-utility gate is therefore not an artifact
of late-window dilution.

## 6. Frozen gate

Immutable package:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_habit_gate_v1
```

- `gate_thresholds_v1.json` SHA-256:
  `8968f2860866a036c58be0cdebbb0dfb67d32e2755a45e0fcc4dd5a176365ac4`;
- gate manifest SHA-256:
  `6626a8fb7382ae41d1f636322793cb1c4649251ebcf1f61ba7efd27e70721dbf`;
- checksums SHA-256:
  `809dab17ca05ab10818fbccbc87af464b5669ec27663ae02966a27e01ab5fda5`;
- checksum verification: `3/3`, zero failures.

The gate uses 100,000 source-episode bootstrap draws and natural zero/chance,
B0, and B2 nulls. It does not use a manually selected success percentage.

## 7. Plain-language conclusion and next boundary

The model has learned the basic excavation action sequence on recorded
observations. It also recognizes the left/center/right target well enough that
changing only the target usually changes swing in the correct direction.

However, the conditioned model currently executes the demonstrated return
actions less accurately than the unconditioned model, and its small advantage
over the shuffled-target model is not stable across source episodes. In plain
language: **it can read the signpost, but the signpost has not yet made it drive
the route better.**

Consequently:

- do not unlock held-out test observations;
- do not claim fixed-scenario completion;
- do not promote to nearest-neighbor stitching, AGX closed loop, real shadow,
  or real control based on this result;
- preserve B0/B1/B2 and the failed gate as the baseline for a separately
  preregistered, one-factor condition-execution experiment.

Any next training change must explain why it should improve correct-target
execution without merely increasing target sensitivity. It must not replace or
rewrite this failed gate after seeing the result.

## 8. User-authorized bounded AGX closed-loop diagnostic

After the frozen offline Gate, the user explicitly chose to inspect actual AGX
execution because action MAE is not the task objective and the simulated
machine tolerates non-trivial action error. This diagnostic does not rewrite
the failed Gate and is not promotable evidence.

Real Stack commits:

```text
519db99 simverify: gate closed-loop condition at dump end
7a7045c simverify: make paired AGX inference deterministic
9043149 simverify: add shared-prefix AGX condition branches
d8e363d simverify: evaluate paired AGX condition branches
```

The runtime kept PACT and Unity read-only. It used the live Unity step-ack
service at 50 Hz and the Real Stack B1 policy at 20 Hz. The B1 condition was
all-zero before an observable dump release ended and became the committed
current-plus-target vector at policy tick 234 (11.7 seconds).

Independent full rollouts were not accepted as a causal condition pair. Even
with fixed seeds and deterministic Torch kernels, a first-action numerical
difference of approximately `2e-6` grew through deformable-terrain feedback.
The final paired experiment therefore replayed one checksum-bound 234-tick
actual-sent-action prefix in all branches, then released policy control at the
same causal dump-end state:

- reference: `left -> left`;
- same-condition repeat: `left -> left`;
- treatment: `left -> center`.

Immutable paired result:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_habit_b1_agx_shared_prefix_branch_eval_v1
```

Key evidence:

| Quantity | Result |
| --- | ---: |
| condition commit / branch tick | 234 |
| bounded horizon | 400 policy ticks |
| condition-effect / repeat-noise action ratio by axis | 10.27 / 13.66 / 38.47 / 3.78 |
| condition-effect / repeat-noise qpos ratio by axis | 41.51 / 42.15 / 78.91 / 6.45 |
| `left` reference final swing qpos | 0.39674 |
| `left` repeat final swing qpos | 0.39648 |
| `center` treatment final swing qpos | 0.42434 |
| `left` reference terminal mean absolute swing command | 0.55351 |
| `center` treatment terminal mean absolute swing command | 0.80617 |

All condition-effect ratios exceed the same-condition repeat variability, so
the target input causally changes the closed-loop action and state trajectory.
It is not merely an offline-MAE artifact.

However, both targets pass through their requested swing region and continue
left. The two `left` branches end at the left travel boundary while still
commanding left swing. The `center` branch also ends in the left sector and
continues commanding left swing. None holds a quiet target endpoint, and no
visual dig-ready confirmation is obtained. The immutable decision is therefore:

```text
condition_signal_detected_but_not_converted_to_target_endpoint
observable_cycle_completed=false
physical_effect_validated=false
task_success_claimed=false
real_control_candidate=false
```

Camera evidence shows the bucket traversing the dig area, reaching the fixed
dump bin, releasing visible soil, and returning toward the work area. This is
useful phase-execution evidence, but it is not promoted to
`physical_effect_validated` because no frozen physical-effect evaluator was
run.

In plain language: **the policy can perform the broad dig-and-dump sequence and
it notices the condition, but it does not yet use that condition to brake and
settle at the requested next digging position.** The next one-factor training
experiment should target endpoint/progress execution rather than condition
sensitivity or aggregate MAE.
