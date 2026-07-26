# SimVerify E04 Camera Counterfactual Evidence

Date: 2026-07-26

Decision: `e04_camera_counterfactual_robustness_not_established`

Evidence scope: `recorded-observation/offline teacher-forced development`

Closed-loop execution: `false`

Held-out test read: `false`

## Input and implementation identity

Formal artifact:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_e04_camera_counterfactual_development_v1
```

- builder Git commit:
  `83122f8b1c3878f581833f8e07b02282b6ae1667`;
- supported validation pairs: `4`;
- source episodes: `12`, `34`;
- camera variants: `12`;
- trace count: `104`;
- checksum entries verified: `110/110`;
- manifest SHA256:
  `c5be2be8b9e80453fc9d3db5b5f744fab3fe8cb901527d5618238a3891f19d84`;
- Gate SHA256:
  `fff2939a2491058680f73c27170d0e39252e16bd7baa763153491bc25f9487f0`;
- checksum-package SHA256:
  `784dfe2e04f0bcd0162960130d8ead65c2afa082da9f568362dfdc96adb89053`.

Focused implementation verification passed `49` tests before the formal run.
The later task-semantic reproduction helper increased the focused suite to
`18` passing tests.

## Reproduction amendment evidence

Two pre-artifact runs stopped before producing a Gate:

1. cross-process bitwise equality failed with maximum action delta
   `7.200241088867188e-05`;
2. a one-cycle cross-process raw maximum of
   `1.6319751739501953e-04` did not upper-bound the longer two-cycle trace,
   whose next run reached `1.8209218978881836e-04`.

Neither failure changed camera results or read held-out data. The final frozen
reproduction rule uses the existing M2 effective deadzone and task semantics:

- observed final old-versus-new maximum action delta:
  `1.0895729064941406e-04`;
- frozen effective deadzone upper: `0.05`;
- effective action-signature mismatches: `0`;
- condition-route mismatches: `0`;
- within-process four-camera repeat delta: `0`.

This is task-semantic replay equivalence, not a claim of GPU bitwise
determinism.

## Data-generated thresholds

| Threshold | Value |
| --- | ---: |
| condition-effect lower | 0.00229103 |
| two-cycle phase-coverage lower | 0.972917 |
| perturbation failure-rate upper | 0.0 |
| G5 ready-boundary discontinuity upper | 0.151286 |

Thresholds use the matched four-camera validation replay and its exact repeat.
Unsupported condition switches do not enter the denominator.

## Variant results

| Variant | Gate | Minimum source semantic margin | Main interpretation |
| --- | --- | ---: | --- |
| four cameras | pass | 0.015225 | G5.1 baseline reproduced |
| drop video4 / eye_left | pass | 0.017044 | this probe retained response |
| drop video5 / eye_right | pass | 0.017229 | this probe retained response |
| drop video6 / stick_down | fail | 0.016573 | one supported anchor left matched retention |
| drop video7 / stick_up | fail | -0.019529 | direction and phase failed |
| eye only | fail | -0.025497 | direction flipped in both source aggregates |
| stick only | fail | 0.017520 | mean direction remained positive, but matched anchor retention failed |
| swap eye pair | pass | 0.015225 | output was almost unchanged |
| swap stick pair | pass | 0.015224 | output was almost unchanged |
| swap cross-role pairs | pass | 0.015270 | output was almost unchanged |
| fixed trace-start frames | fail | -0.028733 | phase coverage and direction failed |
| one-tick image lag | fail | 0.014614 | direction survived, condition-effect retention did not |

All variants retained event order, ready-boundary continuity, router restart,
and exactly one condition-cycle reset. The failure is therefore not the earlier
router lifecycle defect.

## Engineering interpretation

The model does respond to the next-sector condition with the complete recorded
four-camera stream. That response is not camera-robust:

- removing the stick-up view or removing both stick views can reverse the
  requested swing direction;
- freezing images also reverses direction and loses task phases;
- a one-tick image lag already falls below the frozen matched-response
  envelope;
- swapping camera roles barely changes the policy output, so explicit
  camera-role sensitivity is not established even though the checkpoint
  contains enabled, learned nonzero camera-role embeddings.

This is not evidence that the data or observable annotation is invalid. It is
evidence that the current conditioned candidate couples its semantic response
to a narrow visual configuration and does not satisfy the frozen G5 camera
robustness contract.

## Gate consequence

`authorizes_e05 = false`.

The current experiment version must stop before E05 state-hold, E06
delay/latest-wins, held-out test, G6 runtime equivalence, real fine-tuning,
shadow, control, or deployment. The appropriate M5 terminal class is
`revise_condition`, not `sim_observable_only`, `real_finetune_candidate`, or
`control_candidate`.
