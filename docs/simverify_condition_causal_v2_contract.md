# SimVerify fixed-observation condition causal contract v2

Status: frozen before computing the v2 decision artifact.

## Question

Does the B1 policy use `cycle_condition_v1` with the declared semantics, rather
than merely reacting to a six-dimensional token?

This is a recorded-observation/offline causal test. It does not execute a
closed loop and cannot authorize simulation or real deployment.

## Intervention

For every validation anchor, the four-camera history, qpos, qvel, checkpoint,
temporal aggregation, and runtime-safe action transform are fixed. Only one
condition factor is changed:

- `current_sector`, with the other three-vector fixed; or
- `next_sector`, with the other three-vector fixed.

Unsupported anchors remain in the inventory but are excluded from the success
denominator. The held-out episodes remain unread.

## Controls

- B1 requested-token replay: trained with the correct condition.
- B2 requested-token replay: trained with source-episode-shuffled condition.
- B1 masked replay: both requested tokens are replaced by the same canonical
  train-only token. Its paired action difference must be exactly null.
- Three deterministic B1 repeats: define numerical repeat noise.
- Exact semantic permutations: all five non-identity bijections of
  `left/center/right` are evaluated without rerunning inference. A policy that
  understands the declared labels must favor the identity interpretation.

## Metrics and source-episode unit

All anchor metrics are averaged within source episode before inference.
Bootstrap resampling uses source episode, never frames or anchors.

1. **Action sensitivity**: mean absolute action difference in the declared
   factor window. B1 must exceed the masked control beyond repeat noise.
2. **Signed semantic margin**:
   `expected_swing_action_sign * mean_swing_action_delta`.
   B1 must exceed B2 beyond repeat noise.
3. **Semantic identifiability**: the identity signed margin must exceed every
   non-identity semantic permutation beyond repeat noise. Failure of any
   permutation means the declared label meaning is not identified.
4. **Phase specificity**: mean per-tick action effect inside the declared
   factor window minus its mean outside that window. It must be positive beyond
   repeat noise and must exceed B2 beyond repeat noise.
5. **Task-envelope preservation**: event coverage must not decrease and event
   order must remain valid. This is diagnostic preservation, not closed-loop
   success.

The paired 95% bootstrap lower bound must be greater than the corresponding
97.5% same-checkpoint repeat-noise bound. A factor passes only if every
criterion passes. Overall condition understanding is established only if both
factors pass.

## Decision

- both factors pass: `condition_understanding_established_offline`;
- otherwise: `condition_understanding_not_established`, with terminal research
  recommendation `revise_condition`.

No result from this contract may be described as closed-loop success.
