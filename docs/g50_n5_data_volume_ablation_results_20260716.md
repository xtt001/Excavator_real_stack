# G50 N5 data-volume ablation results

Date: 2026-07-16

## Question and capability boundary

This experiment asks how N5 behavior changes when the number of new-style
training episodes changes while model structure, four-camera input, transition
sampling/loss, seed, validation episodes, and optimizer-step budget remain
fixed. It does not establish terrain generalization, physical response,
closed-loop task success, or a generally optimal dataset size.

The nested train sizes are 20, 40, 80, and 120 episodes. All runs use seed 0,
the same 20 chronological validation episodes, 240,000 episode samples, and
approximately 60,000 optimizer steps. Epoch counts are 12,000, 6,000, 3,000,
and 2,000 respectively. N120 reuses the completed G49 N5 run.

The 20/40/80 subsets are deterministic and time-balanced, but their startup
mixtures are not identical. Stick-positive first crossings account for 17/20,
29/40, 55/80, and 79/120 training episodes, while validation is 18/20
stick-positive. Therefore this is a useful practical learning-curve slice, not
a pure composition-invariant estimate of sample-count causality.

## Results on the fixed validation block

| Train episodes | Single-demo MAE | Single-demo active recall | Active on demo-idle frames | Axis-macro demo recall | Stick demo recall | Single-demo exact vector | Natural startup | Startup demo target reproduced | All demo targets reproduced | Anchor-extra effective |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.1809 | 80.6% | 18.3% | 74.3% | 45.3% | 38.8% | 20/20 | 16/20 | 165/261 | 92 |
| 40 | 0.1754 | 81.4% | 16.9% | 72.7% | 31.7% | 48.7% | 20/20 | 17/20 | 165/261 | 97 |
| 80 | 0.1743 | 81.6% | 17.9% | 73.8% | 37.1% | 46.0% | 20/20 | 17/20 | 169/261 | 99 |
| 120 | 0.1803 | 80.6% | 18.4% | 70.6% | 24.2% | 45.6% | 20/20 | 5/20 | 169/261 | 102 |

All four models are already deadzone-effective during the recorded preparation
prefix in 20/20 episodes and activate at zero delay under the armed natural
startup diagnostic. This shows that N5's broad activation is primarily a
property of transition-promote training, not a capability that appears only at
120 episodes.

More data does not produce a monotonic improvement on this validation block.
Continuous single-demo MAE improves from 20 to 80 episodes, then regresses at
120. Overall demo-target reproduction saturates around 80 episodes. The
120-episode model reproduces the fewest startup demo targets in this
comparison and has the most anchor-relative extra effective actions. Neither
metric establishes correctness or generic liveness.

The same-demo event comparison provides an important counterweight. The
outside-single-demo-event-support rate decreases from 12.5% at N20 to 8.2% at
N120, and the outside-current-demo-window rate decreases from 32.2% to 29.8%.
This shows closer coverage of held-out recordings with more data. It does not
establish broader task support or label the remaining directions invalid.

## Interpretation

The result rejects the claim that N5's current 20/20 natural liveness is caused
mainly by having 120 training episodes. Twenty representative episodes are
already enough for that diagnostic under the current transition objective.

It does not prove that less data is generally better. The smaller nested sets
match the stick-positive-heavy validation startup distribution more closely,
all data comes from one recording session, and only one subset path and one
training seed were tested. The full dataset contains more alternative startup
motifs; the current continuous policy can express those alternatives as early
multi-axis activation rather than selecting a scene-specific intent.

The current evidence therefore separates two effects:

1. transition-promote supervision determines whether the policy is broadly
   willing to cross the deadzone;
2. dataset amount and composition affect single-demo action closeness and which
   startup motif the continuous head applies, but current metrics cannot decide
   whether an alternative motif is correct for the task.

The next pure data-count estimate should use multiple nested subset seeds with
startup-motif/action-distribution stratification. It should report mean and
variance and compare matched optimizer-step checkpoints as well as each run's
validation-selected checkpoint.

## Artifacts

Experiment root:
`/data/pingfan/Excavator_real_stack_data/runs/g50_n5_data_volume_ablation_20260715`

Each `evaluation/n{20,40,80}` directory contains historical-schema open-loop,
armed-startup, state-hold, and expert-intent reports. They must be regenerated
with the v2 single-demo semantics before future automated comparison. The N120
reference remains under
`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation`.
