# SimVerify B1.5/B2.5 Video7-Dropout Revision Contract

Status: `method_frozen_before_implementation_or_training`

Evidence scope: `recorded-observation/offline development`

Closed-loop execution: `false`

Held-out test read: `false`

## Causal question

E04 showed that B1.4 passes supported next-condition semantics with all four
cameras, but dropping `video7 = stick_up` reverses the source-episode semantic
margin and breaks the frozen camera Gate.

B1.5 asks whether one train-only sensor-loss factor can remove this brittle
dependence without changing the condition representation, router, ACT
architecture, split, action loss, or canonical validation input.

## Single changed factor

Relative to B1.4, B1.5 adds only:

```yaml
camera_loss_augmentation:
  enabled: true
  scope: train_only
  target_camera: video7
  probability: 0.25
  seed: 20260727
  mask_rgb: [0, 0, 0]
  decision_key: [seed, source_episode_id, source_tick]
```

`0.25` is frozen before training because `video7` is one of the four canonical
camera roles: one quarter of eligible train samples become the declared
single-sensor-loss case, while three quarters retain the canonical four-camera
observation. It is not tuned from validation.

The selection is SHA-256 keyed by source episode and source tick. It is
independent of DataLoader worker order, epoch order, and global NumPy RNG.
Repeated requests for the same source row receive the same camera decision.

The augmentation:

- changes only the raw RGB array supplied as `video7`;
- does not alter source HDF5, qpos, qvel, action, condition, phase route, sample
  validity, normalization, or loss labels;
- is physically the same zero-mask sensor-loss probe used by E04;
- is disabled for validation, replay, and every held-out source.

## Matched candidate and null

- B1.5 inherits B1.4 and retains the observed next-sector condition.
- B2.5 inherits B2.4 and applies the identical video7 augmentation while
  retaining the frozen deterministic shuffled-condition association.

Thus B1.5 versus B2.5 differs only in condition-label association. B1.5 versus
B1.4 differs only in the train-only video7-loss factor.

Both runs start from their declared initialization path and use the same seed,
split, epoch count, optimizer, mixed precision, camera-role encoding, router,
deadzone loss, and checkpoint selection rule as their parent.

## Required provenance

Before training, the loader must emit:

- resolved augmentation schema and parameters;
- train eligible-row count and selected-row count;
- selected fraction;
- SHA-256 of the sorted selected `(episode_id, source_tick)` keys;
- train and validation source episode IDs;
- validation `enabled=false` and selected count `0`;
- B1.4/E04/M5 v3 input SHAs;
- Git SHA and dirty state.

No held-out episode may appear in the manifest.

## Evaluation order

1. train and complete B1.5;
2. train and complete matched B2.5;
3. repeat the frozen G4 next-condition causal Gate;
4. repeat G5.1 two-cycle core replay;
5. repeat the unchanged E04 camera Gate;
6. stop at the first failed prerequisite.

No threshold may be changed after observing B1.5/B2.5 validation.

The frozen next-condition replay and Gate may be parameterized only to replace
the baseline identity pair `B1.4/B2.4` with `B1.5/B2.5`. This is an evaluator
provenance change, not a Gate change: replay rows, support, formulas, bootstrap
unit/seed/repetitions, criteria, and thresholds remain unchanged. Cross-pairs
such as `B1.5/B2.4` are invalid.

The same identity-only parameterization applies to the frozen G5.1 builder.
Candidate/null metric field names are role-based so B1.5 results are never
mislabelled as B1.4. Router lifecycle, two-cycle anchors, expert thresholds,
support rule, trace construction, condition-switch formula, and pass criteria
remain unchanged.

The same identity-only parameterization applies to the frozen E04 builder.
The E04 manifest, trace rows, passing G5.1 input, candidate bundle, and
cross-process replay packages must all declare B1.5. Camera interventions,
matched-pair metrics, threshold estimators, source aggregation, and complete
E04 pass criteria remain unchanged.

## Targeted causal Gate

The revision is useful only if:

- four-camera G4 and G5.1 remain established;
- `drop_video7` semantic margin is strictly positive in every eligible source
  episode;
- `drop_video7` failure rate returns inside the frozen E04 envelope;
- B1.5 exceeds the matched B2.5 semantic null;
- four-camera phase coverage and ready-boundary continuity do not regress.

Eye-only, fixed-frame, and lag results remain required E04 diagnostics, but
they cannot be attributed to this single video7-only factor. A targeted pass
does not authorize E05 unless the complete unchanged E04 Gate also passes.

Neither pass nor failure authorizes held-out test, real fine-tuning, shadow,
control, deployment, or a closed-loop claim.
