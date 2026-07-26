# SimVerify E04 Camera Counterfactual Contract

Status: `method_amended_after_two_numeric_reproduction_probes_before_successful_run`

Evidence scope: `recorded-observation/offline teacher-forced development`

Closed-loop execution: `false`

Held-out test read: `false`

## Question

E04 asks whether the G5.1 next-condition response and observable two-cycle task
phases survive declared camera interventions, and whether the policy depends on
a fixed image or an incorrect camera-role shortcut.

It does not test physical response, real-camera generalization, changed
intrinsics/extrinsics, or closed-loop digging.

## Immutable inputs

- the passing G5.1 two-cycle development package;
- the M0 camera mapping and read-only 20 Hz episode exports;
- M2 supported two-cycle condition-switch anchors and expert event envelope;
- the B1.4 sim-domain checkpoint;
- the three immutable B1.4 validation repeat packages used to measure
  cross-process numerical replay variation;
- held-out episodes `1`, `13`, `25`, and `33` remain unread.

Only supported changed-next-target validation pairs enter the success
denominator. There are four such pairs from source episodes 12 and 34.
Unsupported pairs remain reported and cannot be imputed.

## Single-factor interventions

Every replay uses the same source ticks, qpos, qvel, condition, policy weights,
normalization, router lifecycle reset, and temporal-aggregation lifecycle.
Only the four recorded RGB images may change.

The physical-role lock is:

- `video4 = eye_left`;
- `video5 = eye_right`;
- `video6 = stick_down`;
- `video7 = stick_up`.

The frozen variants are:

| Variant | Image intervention |
| --- | --- |
| `four_camera` | unchanged recorded images |
| `eye_only` | replace video6/video7 with raw-RGB zeros |
| `stick_only` | replace video4/video5 with raw-RGB zeros |
| `drop_video4` ... `drop_video7` | replace exactly one role with raw-RGB zeros |
| `swap_eye_pair` | exchange video4 and video5 |
| `swap_stick_pair` | exchange video6 and video7 |
| `swap_cross_role_pairs` | exchange video4/video6 and video5/video7 |
| `fixed_trace_start` | repeat each camera's recorded first trace frame |
| `lag_one_tick` | use the preceding recorded tick, clamped at trace start |

Zero masking is an explicit out-of-distribution sensor-loss probe, not an
estimate of a physically plausible camera image. Swap variants preserve the
four recorded images and change only their declared policy roles. Hold and lag
variants never synthesize pixels.

## Replay and repeat-noise lock

For each eligible pair and variant, replay both:

- `switched`: atomically deliver the second condition at the shared ready
  boundary;
- `unchanged`: retain the first condition.

The policy resets once at trace start. The condition-cycle router resets once
at the shared ready boundary. Temporal aggregation does not reset there.

`four_camera` is run twice. Repeat 0 must be task-semantically equivalent to the
immutable G5.1 trace: its maximum action delta must remain below the frozen M2
minimum effective deadzone, its deadzone-effective action signature must match
exactly, and its condition-route sequence must match exactly. Repeat 0 versus
repeat 1 generates the same-process action and metric noise floor. No
hand-entered tolerance is allowed.

The first attempted build correctly stopped because it required bitwise
equality across separate GPU processes. The observed old-versus-new maximum
action delta was `7.200241088867188e-05`, while the pre-existing three-repeat
B1.4 envelope was `1.6319751739501953e-04`. The within-process E04 repeat delta
was exactly zero. The amendment replaces an invalid bitwise premise with this
independently recorded data-derived envelope; it does not change any camera
intervention or success metric.

The second attempted build showed why a one-cycle raw maximum is not a valid
hard upper bound for a longer temporally aggregated two-cycle trace. Its
old-versus-new maximum was `1.8209218978881836e-04`, slightly above the
one-cycle envelope, while mean absolute delta stayed below
`8.085956324066501e-06`, action signs never differed, and the same-process
repeat remained exact. The final reproduction Gate therefore uses the already
frozen M2 effective deadzone (`0.05` on every axis), exact effective-signature
equality, and exact router equality. The three cross-process repeats remain
recorded as numerical context, not as an invalid duration-independent maximum.

## Metrics and aggregation

Every trace retains raw normalized chunks, raw source-domain chunks, temporal
aggregation actions, future-runtime-safe actions, delivered conditions, route
diagnostics, and intervention provenance.

For each supported pair, compute:

- two-cycle required-event coverage and event order;
- ready-boundary discontinuity;
- switched-versus-unchanged action effect;
- route-2 swing semantic margin;
- condition-effect retention relative to matched four-camera replay;
- phase-coverage retention relative to matched four-camera replay;
- direction flip and unexpected-effective-axis indicators.

Aggregate paired anchors within source episode before comparing source
episodes. The single-camera dropout result uses the worst of the four
single-role dropouts. Pair-swap and cross-role-swap failure rates count an
anchor when condition effect falls below the repeat-noise floor, route-2
semantic direction becomes non-positive, required event order fails, or phase
coverage leaves the unperturbed envelope.

## Development thresholds

Thresholds are generated from the matched unperturbed B1.4 validation replay
and its exact repeat, following the frozen M0 Gate contract:

- retention lower bound = unperturbed B1 validation q02.5 minus the
  same-checkpoint repeat absolute-delta q97.5;
- perturbation failure upper bound = unperturbed B1 validation q97.5 plus the
  same-checkpoint repeat disagreement q97.5;
- ready discontinuity and task-event coverage remain bounded by the frozen G5
  expert envelope;
- semantic margin must remain strictly positive.

The package records every estimator input, source-episode count, quantile,
formula, and SHA. This is a development Gate because validation has already
participated in method design. It cannot authorize held-out access by itself.
Anchor failure classification uses its matched four-camera value minus that
anchor's exact repeat delta; source-level retention still uses the frozen
q02.5 aggregation. This prevents a source-distribution quantile from being
misapplied as a per-anchor threshold.

## Decision

`e04_camera_counterfactual_robustness_established_development` requires each
variant's declared source-episode distribution to pass the frozen q02.5/q97.5
bounds. Source rows remain visible, and semantic direction, event order, ready
continuity, and router lifecycle must pass in every source episode:

- eye-only condition and phase retention;
- stick-only condition and phase retention;
- worst-single-camera-dropout condition and phase retention;
- within-pair swap failure-rate bounds;
- cross-role swap failure-rate bounds;
- fixed-frame and one-tick-lag diagnostic bounds;
- G5 expert event, ready-boundary, router lifecycle, and positive semantic
  direction invariants.

Otherwise the decision is
`e04_camera_counterfactual_robustness_not_established`. Failure identifies the
specific camera dependency and does not authorize hidden threshold relaxation,
held-out access, retraining, real control, or a closed-loop claim.
