# SimVerify B1.5 E04 Result

Status: `completed_failed_prerequisite`

Decision: `e04_camera_counterfactual_robustness_not_established`

Evidence scope: `recorded-observation/offline teacher-forced development`

Closed-loop execution: `false`

Held-out test read: `false`

Authorizes E05: `false`

## Immutable result

The B1.5 E04 package was built from clean Git commit
`4e0177c950f10ede73a873d4eb9f219ffd8c0c6d`. It contains 104 replay traces
covering the four supported validation pairs, all 12 frozen camera variants,
both switched and unchanged condition modes, and the repeated four-camera
control.

The package is:

`/home/pingfan/Excavator_real_stack_artifacts/simverify_b1_5_e04_camera_counterfactual_development_v1`

Its immutable identities are:

- manifest SHA-256:
  `7f206f1954be525a240b5e4e0ed3ab89e9c31ea38aa3bd0f833b8e32f864afa6`;
- Gate SHA-256:
  `8ff1340093185cdf2d0c070a896ff0828325531798f4517d1be3a7651aa5f004`;
- checksum-file SHA-256:
  `ab4d1ee6f82f5ea24d44295821d76759f082c6793ebe2bb94b6cab8a68ae5a99`;
- verified files: 110, with zero checksum failures;
- four-camera reproduction maximum action delta: `0.0`;
- four-camera effective-signature mismatch count: `0`;
- four-camera router mismatch count: `0`.

## What the single-factor revision fixed

The declared B1.5 factor was train-only 25% `video7 = stick_up` zero masking.
It fixed the main qualitative B1.4 failure under the matching `drop_video7`
probe:

| Source episode | B1.4 semantic margin | B1.5 semantic margin |
| --- | ---: | ---: |
| 12 | -0.0195292532 | 0.0407956764 |
| 34 | -0.0001536193 | 0.0308997345 |

Thus the old direction reversal is absent in both eligible source episodes.
Event order, ready continuity, route activation, and the exactly-once
condition-cycle reset also pass. B1.5 has a strictly positive source semantic
margin under every one of the 12 camera variants, not only `drop_video7`.

This does not by itself establish camera robustness. It establishes only that
the requested next-sector response keeps the expected swing direction on the
fixed recorded observations.

## Why the frozen targeted requirement still failed

The E04 per-anchor classifier uses the matching four-camera value minus exact
same-process repeat noise. The repeat noise was zero, so the frozen
non-inferiority margin is also zero: an intervention must not reduce either
condition effect or phase coverage at all for that anchor.

Two of the three episode-34 `drop_video7` anchors were slightly below their
matching four-camera condition-effect value:

| Anchor | `drop_video7` effect | Frozen lower bound | Relative change |
| --- | ---: | ---: | ---: |
| episode 34, anchor 2 | 0.0036128860 | 0.0036423297 | -0.81% |
| episode 34, anchor 3 | 0.0030127941 | 0.0030787569 | -2.14% |

Both anchors retained positive semantic margins (`0.0391135849` and
`0.0281064380`) and full phase coverage. Nevertheless, the predeclared E04
failure envelope is zero, so episode 34 has a `drop_video7` failure rate of
`2/3`; the source q97.5 failure statistic is `0.65 > 0.0`. The targeted
video7 requirement therefore does not pass, and the threshold is not relaxed
after observing the result.

## Complete E04 result

Five variants pass the unchanged complete Gate:

- `four_camera`;
- `drop_video4`;
- `drop_video5`;
- `stick_only`;
- `swap_stick_pair`.

Seven variants fail:

- `drop_video6`: condition-effect and phase-retention failures;
- `drop_video7`: exact per-anchor condition-effect retention failure;
- `eye_only`: condition-effect and phase-retention failures;
- `fixed_trace_start`: phase-retention failures;
- `lag_one_tick`: exact condition-effect retention failures;
- `swap_eye_pair`: exact per-anchor condition-effect retention failures;
- `swap_cross_role_pairs`: exact per-anchor condition-effect retention failure.

All variants retain positive source-mean semantic direction. The complete Gate
fails because semantic direction is necessary but not sufficient: the frozen
contract also requires exact matched effect/phase retention and zero failed
anchors.

## Comparison package

The prior B1.4 E04 package remains immutable at:

`/data/pingfan/Excavator_real_stack_data/simverify_e04_camera_counterfactual_development_v1`

Its manifest, Gate, and checksum-file SHA-256 values are respectively:

- `c5be2be8b9e80453fc9d3db5b5f744fab3fe8cb901527d5618238a3891f19d84`;
- `fff2939a2491058680f73c27170d0e39252e16bd7baa763153491bc25f9487f0`;
- `784dfe2e04f0bcd0162960130d8ead65c2afa082da9f568362dfdc96adb89053`.

That package also verifies with zero checksum failures.

## Decision and next admissible work

E05 is not authorized. No held-out test, real fine-tuning, shadow execution,
control, deployment, or closed-loop claim follows from this result.

The next admissible slice is a new, predeclared revision study rather than a
post-hoc E04 threshold edit. It should separate:

1. condition-semantic robustness, using source-episode bootstrap and the
   matched shuffled-condition null;
2. task-phase preservation, using a train-derived or otherwise independently
   frozen non-inferiority margin;
3. temporal-vision reliance diagnostics, where fixed-frame and lag probes are
   interpreted as sensitivity tests rather than automatically requiring
   invariance.

Any new margin must be frozen before another validation run. B1.5 remains a
sim-domain, offline-only checkpoint and is not promotable.
