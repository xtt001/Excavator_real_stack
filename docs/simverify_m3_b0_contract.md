# SimVerify M3/B0 Unconditioned Baseline Contract

Status: `batch_preflight_passed_training_not_started`

Evidence scope: `recorded-observation/offline`

M2 authorizes only the B0 observation-only baseline. This contract freezes B0
before any training so B1 can later change exactly one primary factor:
`cycle_condition_v1` input.

## Why B0 exists

B0 asks whether the four recorded camera views plus source-domain qpos/qvel
are sufficient for ACT to emit the major action phases of an accepted cycle.
If B0 fails, a later conditioned failure cannot be assigned to condition
design.

B0 is not a deployment candidate and cannot prove environment response.

## Frozen inputs

- dataset:
  `/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3/episodes`;
- cameras: `video4, video5, video6, video7`;
- camera roles:
  `eye_left, eye_right, stick_down, stick_up`;
- low-dimensional input: source-domain qpos then qvel, eight dimensions;
- condition input: absent;
- action: source-domain `actuator_speed_cmd`;
- chunk: 20 ticks at 20 Hz;
- image transform: the M0 materialized image, with no additional transform;
- accepted-cycle sampling mask: `conditions/valid_mask`;
- normalization: accepted rows from train episodes only;
- split: the M0 train/validation assignment, minus episodes with no valid
  20-tick accepted-cycle start.

The checkpoint metadata must remain:

```text
domain=sim
deployment_status=offline_evaluation_only
real_control_allowed=false
jetson_allowed=false
```

## Split-preserving sample exclusion

M0 assigns 16 train and four validation source episodes. The B0 batch preflight
found:

| Episode | M0 split | accepted rows | valid 20-tick starts |
| --- | --- | ---: | ---: |
| 19 | train | 0 | 0 |
| 23 | validation | 0 | 0 |

These episodes remain in their original M0 split and are not reassigned. They
are excluded from B0/B1 sampling by the predeclared rule “at least one complete
20-tick action chunk must lie inside an accepted cycle.” The resulting B0
training split has 15 episodes and validation has three. Held-out episodes
`1, 13, 25, 33` remain unread.

This is a sample-existence rule, not a fitted model-performance threshold.

## Inherited model/training factors

The first B0 config uses the existing four-camera ACT structure and the
N5-derived training settings:

- ResNet-18 shared image backbone;
- camera-role encoding mechanism;
- hidden dimension 512 and feedforward dimension 3200;
- KL weight 10;
- learning rate `1e-5`;
- 2000 epochs, seed 0, batch size four;
- the same effective-action deadzone objective family and 0.05 source
  deadzone from the M0 state/action contract.

These values are experiment factors inherited before B0 results exist. They
are not G3 pass percentages. B1/B2 must keep them unchanged.

## Batch preflight result

The read-only preflight loaded all 18 usable train/validation exports and
reported:

| Item | Result |
| --- | ---: |
| train episodes | 15 |
| validation episodes | 3 |
| train valid 20-tick starts | 31519 |
| validation valid 20-tick starts | 8199 |
| proprio dimension | 8 |
| camera tensor | `(B,4,3,216,384)` |
| source domain is real | false |
| condition in policy input | false |
| held-out read | false |
| training started | false |

## Next bounded action

After focused tests, a clean commit, and push, run a one-epoch B0 smoke in a
new output directory. The smoke must prove model construction, camera-role
encoding, deadzone loss, checkpoint sim-domain metadata, train-only stats, and
run provenance. It is not a G3 result. Formal 2000-epoch B0 training may start
only after that smoke passes.

