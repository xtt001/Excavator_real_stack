# [Execution target G30/H1: Data-backed ACT failure-mode analysis]

Date: 2026-07-12
Scope: formal train IDs `19`, validation IDs `5`, G28 checkpoint, no held-out
105..109, no source HDF5 modification.

## [Execution target G30/H1: Direct answer]

The data does **not** support calling the three deadlocks simple visual OOD.
The stronger evidence is:

1. G28 maps deadlock observations to nearby, repeated train modes rather than
   to an empty visual region;
2. the formal train fold has no `boom+|bucket+` command at all, while the
   validation episode 94 contains 29 such frames;
3. in two episode-74 deadlocks, the intent head eventually indicates `boom+`
   but the continuous boom effort remains below the mechanical deadzone;
4. teacher-forced qpos progression makes the boom action cross the deadzone,
   while state-hold qpos keeps the policy in a bucket-only attractor.

This is a combination of action-support/representation failure and a
closed-loop phase/observability failure. It is not evidence for a pure
“the camera is OOD” explanation.

## [Execution target G30/H1: Visual-support and OOD evidence]

For each of 48 validation state-hold anchors, the analysis extracted the
G28 `goal_context_proj` 512-dimensional representation. It compared anchors
to 3,330 formal-train frames sampled every four ticks.

| representation used for nearest-neighbor search | deadlock NN distance | recovered NN distance | deadlock top-10 action-label entropy | recovered entropy |
| --- | ---: | ---: | ---: | ---: |
| vision + qpos | 0.202 | 0.234 | 0.241 bits | 0.729 bits |
| vision + qpos + qvel | 0.208 | 0.243 | 0.241 bits | 0.696 bits |
| vision + qpos + qvel + short action history | 0.244 | 0.257 | 0.294 bits | 0.595 bits |

Deadlock anchors are not farther from the train support; they are slightly
closer. Their neighbors are also more mode-pure, not more ambiguous. The
target future direction appears in only 6.7% of the top-10 neighbors for
deadlocks versus 73.3% for recovered anchors. Adding qvel/history raises the
deadlock figure only to 10.0%.

All 48 anchor frames had valid video4/video5 image groups and `online_qc`
status `PASS`; camera-group skew p95 was `0.034 ms`. This does not prove
perfect perception, but it gives no support for a sensor-corruption/OOD
primary cause.

The cross-episode train-neighbor analysis does show that multimodality exists
globally: at top-10, 18.3% of train frames have neighbors containing both a
boom-positive and a bucket direction. The failing anchors, however, are
locally dominated by one wrong mode (`bucket+` for episode 94 and `bucket-`
for episode 74), rather than by a balanced local mixture.

## [Execution target G30/H1: Action-mode support in the actual data]

Direct effective-action census (thresholds are the reviewed mechanical
deadzones, stick excluded as structural zero):

| source | `boom+|bucket+` | `boom+|bucket-` | `boom+` | `bucket+` | `bucket-` |
| --- | ---: | ---: | ---: | ---: | ---: |
| formal train, 13,288 frames | **0** | 867 | 637 | 2,094 | 1,337 |
| formal validation, 3,241 frames | 29 | 314 | 169 | 459 | 298 |
| all available episodes 72–104 | 88 | 1,448 | 1,044 | 3,115 | 2,092 |

The missing joint mode is a split-support problem, not a lack of data in the
machine record: episode 77 alone has 59 `boom+|bucket+` frames, while episode
94 validation has 29. Episode 77 was not in the frozen 19/5 formal split.

The episode-94 deadlock target is exactly the unsupported joint mode:

```text
expert:  [0.000, +0.264, 0.000, +0.745]
policy:  [0.005, +0.046, 0.000, +0.550]
effective policy axes: bucket+
```

This is not a symmetric midpoint between independent boom and bucket modes;
it is a dominant bucket mode with the boom component attenuated below the
deadzone. The formal train fold contains no positive-boom/positive-bucket
example from which a continuous head could learn this joint effort pattern.

## [Execution target G30/H1: Intent versus continuous effort]

The G28 intent probabilities were recomputed at all 48 anchors in the same
qpos-only, identity-scale inference contract. Target directions above 0.5:

- all anchors: `45/48 = 93.75%`;
- recovered anchors: `44/45 = 97.78%`;
- deadlocked anchors: `1/3 = 33.33%`.

The three deadlocks separate into two different mechanisms:

| anchor | target intent probabilities | continuous action | interpretation |
| --- | --- | --- | --- |
| episode 94:474 | boom+ `0.030`, bucket+ `0.974` | boom `0.046`, bucket `0.550` | intent and effort both choose bucket-only; unseen joint mode/phase cue is missing |
| episode 74:198 | boom+ `0.241`, bucket- `0.981` | boom `-0.039`, bucket `-0.698` | intent initially weak; state-hold remains in bucket-only attractor |
| episode 74:208 | boom+ `0.590`, bucket- `0.987` | boom `0.065`, bucket `-0.710` | intent sees target, but continuous boom effort is still sub-deadzone |

For episode 74, teacher-forced qpos progression changes the outcome:

- anchor 198: state-hold boom max `0.007`, teacher-forced boom max `0.279`;
- anchor 208: state-hold boom max `0.112`, teacher-forced boom max `0.279`.

The reviewed boom-positive threshold is `0.259`. Thus an intent/effort
factorization has a real opportunity at anchor 208, but intent alone cannot
fix anchor 94 or the initial weak intent at anchor 198.

## [Execution target G30/H1: Root-cause judgment]

| hypothesis | evidence | judgment |
| --- | --- | --- |
| visual OOD / corrupted camera | nearest model representation is in train support; all anchor QC passes | not primary |
| partial observability / phase attractor | qpos-only policy; teacher-forced versus state-hold divergence; qvel/history does not fully repair the three cases | real contributor |
| continuous action representation / mode averaging | formal train has zero positive-boom/positive-bucket joint mode; target boom effort is attenuated while bucket remains strong | strong contributor |
| local discrete intent supervision | high accuracy on recovered anchors; detects boom at episode 74:208 while action misses it | promising auxiliary, not sufficient alone |

The precise description is therefore: **a qpos/image-conditioned policy enters a
known bucket mode; when the desired joint mode is absent or phase-dependent,
the continuous head drops the boom effort below the mechanical deadzone.**

## [Execution target G30/H1: Existing-data next experiment]

No new acquisition is required for the first falsifiable follow-up. Rebuild a
train-only split that adds episode 77 (59 positive-boom/positive-bucket frames)
while keeping episode 94 as validation, then compare:

1. a soft intent-conditioned effort head (intent supplies eligibility,
   continuous effort remains the action source);
2. the original continuous head with the added joint-mode support;
3. teacher-forced/state-hold gap and the three known deadlock anchors.

Do not use hard argmax projection, previous-command phase cues, or action
suppression. The experiment must be judged by target-axis recovery and
hidden-deadlock/extra-motion gates, not by MAE alone. Held-out 105..109 remain
untouched until the new validation contract passes.

## [Execution target G30/H1: Evidence artifacts]

Summary manifest:

`/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g30_failure_mode_analysis/failure_mode_summary.json`

Supporting artifacts:

- `analysis_rows.json` — model-representation nearest-neighbor rows;
- `intent_anchor_results.json` — anchor-level intent/action inference;
- `feature_cache.npz` — train/anchor representation and qvel/history arrays.

Source SHA-256 values are recorded in the summary manifest. Held-out
`105..109` were not read or used.
