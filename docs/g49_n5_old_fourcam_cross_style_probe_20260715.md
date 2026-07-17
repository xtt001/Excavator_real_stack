# G49 N5 old-four-camera cross-style intent probe

## Decision question and scope

This test asks whether the new-style-only N5 policy produces executable intent
when it receives the older four-camera observations, and whether that output is
compatible with the old expert task sequence. It also asks whether the result
is materially affected by image content when qpos/qvel are held fixed.

It directly measures deadzone-thresholded policy output during teacher-forced
replay and prediction sensitivity under one constructed image swap. It does not
measure latent intent, physically realized motion, hydraulic response,
self-generated closed-loop observations, task success, or terrain
generalization.

## Frozen inputs

- Policy: G49 N5 four-camera camera/role ACT plus transition supervision.
- Checkpoint SHA-256:
  `0c9b755447f1c06a893394fb1111b9365eb47a8670523b6eeaef8b2df7e13b0e`.
- Cameras and order: `video4, video5, video6, video7`.
- Low-dimensional input: qpos only; qvel is present in the observation mapping
  but is not consumed by this policy bundle.
- Action domain: direct policy output, identity scale.
- Deadzone source: the frozen asymmetric G49 direct-output table.
- Old source: all 24 formal train-ready episodes from the `72..104` old-style
  20 Hz dataset, totaling `16,529` frames.
- The model was not trained on these old episodes. They are not nevertheless an
  untouched project-level test set because earlier model development used this
  data family.
- Old sealed episodes `105..109` were not read.

## A. Complete old-observation replay

N5 produces a deadzone-effective output at some point in all `24/24` episodes.
It is effective at recording step 0 in `17/24`, and first becomes effective
before or at the old expert's first onset in `19/24`. The remaining five become
effective only after the expert trajectory has already progressed.

The key startup comparison is not favorable:

| First direction set | N5 episodes |
| --- | ---: |
| `boom-` | 15 |
| `boom- + stick+` | 6 |
| `bucket-` | 1 |
| `boom- + bucket+` | 1 |
| `stick+` | 1 |

The old expert first anchor is `bucket+` in `24/24` episodes. Among the 19 N5
outputs that occur before or at that onset:

- `0/19` are wholly within the old first event's 40-tick support;
- `1/19` overlaps the old expert anchor;
- `0/19` is opposite to the old expert anchor.

The absence of opposite startup among the scored 19 should not hide the main
fact: N5 has transferred executable activation but not the old task's uniform
bucket-positive startup convention.

Across all 227 old expert events, transfer is materially better than at startup:

- required-direction recall at the current anchor: `0.7413`;
- exact direction-set rate: `0.4978`;
- unsupported-direction event rate: `0.2687`;
- opposite-to-anchor event rate: `0.0176`;
- continuous-action MAE over all 16,529 frames: `0.1949`.

These later-event numbers show partial expert-path transfer. They remain
teacher-forced and cannot prove that N5 would reach those phases under its own
earlier commands.

## B. Same-qpos old/new image swap at reset step 0

For each old episode step-0 observation:

1. keep the old qpos and qvel exactly fixed;
2. run reset-step-0 inference with the original four old images;
3. find the nearest qpos frame among every frame of N5's 120 new-training
   episodes, using RMS per-axis distance normalized by N5 training qpos std;
4. replace all four images with that matched new frame and reset/infer again.

The match is numerically close in qpos space: median normalized RMS distance is
`0.00679`, p95 is `0.02205`. Under this intervention:

- old images are effective in `17/24` pairs;
- matched new images are effective in `21/24` pairs;
- direction set changes in `21/24` pairs;
- mean per-axis absolute action change is `0.1974`, p95 `0.2953`.

Old-image outputs are dominated by `boom-` (`9`) and `boom- + stick+` (`6`),
with seven idle rows. Matched new-image outputs are dominated by `stick+`
(`13`), with three idle rows.

This rejects the explanation that the old-data result is only a fixed qpos
prior: image content/domain materially changes N5's direction decision. It does
not prove correct camera semantics. The swap is a diagnostic intervention and
may combine images and state that were never physically observed together.

## Conclusion

N5 can produce intent from old four-camera observations, and the visual input
materially influences that intent. The stronger generalization claim fails at
startup: the model interprets the old visual/state distribution mainly as a
new-style boom-negative or boom/stick startup, while the old expert consistently
starts bucket-positive.

The appropriate conclusion is therefore:

> N5 has cross-style executable and visual sensitivity, plus partial later-phase
> transfer, but it has not learned a task-invariant startup interpretation.

This result motivates an explicit old/new task-style discrimination or
task-phase representation test before treating N5 as a generalized controller.
It does not motivate consuming the sealed `105..109` set yet.

## Artifacts

- Open-loop summary:
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/n5_old_fourcam_cross_style_24_open_loop/collection_summary.json`,
  SHA-256 `99056d1b00ef8e3e7db903ebc08351631da06403dec077d95cbd118b6b10bca6`.
- Expert-intent report:
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/n5_old_fourcam_cross_style_24_intent_eval_h40/expert_intent_eval_report.json`,
  SHA-256 `aa4ccd600b8fe5af9888a49a4b608cb4f1bad9c8c114a4311e73c5b0f93ce9d8`.
- Expert-event manifest:
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/old_fourcam_cross_style_24_expert_intent_events_h40/expert_intent_events_manifest.json`,
  SHA-256 `6f2f6cffbd862b54b5b59c19cb139ebec7a30da5d0f9ad06e3b19e1e2cf18d84`.
- Fixed-qpos image-swap report:
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/n5_old_fourcam_fixed_qpos_image_swap_step0/cross_style_image_swap_report.json`,
  SHA-256 `487a2dc45f7b66f0559b154e69d94a7a5fc05f8243f292776ecacae87833417f`.
