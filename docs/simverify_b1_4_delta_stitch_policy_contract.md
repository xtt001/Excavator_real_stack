# SimVerify B1.4 Delta-Stitch Policy Development Contract

Status: `method_frozen_before_policy_stitch_run`

Evidence scope: `recorded-observation/offline development`

Closed-loop execution: `false`

Independent validation: `false`

This experiment is authorized only by the G4-v3.1 expert development
prerequisite. It tests whether B1.4 next-sector condition interventions alter a
causal sequence of supported recorded transitions more semantically than the
matched shuffled-condition B2.4 control.

## Inputs and locks

- transition bank and normalization: immutable G4-v2 package;
- retrieval rule and local-delta integration: immutable G4-v3 implementation;
- prerequisite authorization: immutable G4-v3.1 audit;
- checkpoints: completed B1.4 and B2.4 sim-domain bundles;
- anchors: M2 validation `next_sector` counterfactual anchors that are supported
  and belong to the B1.4 support Gate's eligible source episodes;
- held-out episodes `1`, `13`, `25`, and `33`: locked unread.

Validation has already served method development. Results cannot be called an
independent confirmation.

## Causal rollout

Each paired rollout starts at the same recorded validation cycle-start
observation. The ACT temporal aggregator is reset, then receives the stitched
observation sequence at 20 Hz.

For every tick:

1. the policy receives qpos, qvel, four source images, and one fixed requested
   next-sector condition;
2. the future-runtime-safe temporal-aggregation action is emitted;
3. retrieval matches current observable state and that action against the
   train transition bank;
4. the exact nearest cross-current-donor episode, not-yet-used transition is
   selected inside the frozen support radius;
5. the next policy observation becomes that transition's recorded successor;
6. only the selected local annotation delta is added to accumulated progress.

Condition, phase, progress, target sector, candidate successor, successor
identity, future state, and privilege are not retrieval inputs. Condition may
affect selection only through the policy's emitted action.

Every trace separately stores:

- raw normalized policy chunk;
- raw direct policy chunk;
- temporal-aggregation action;
- future-runtime-safe action;
- selected transition identity and retrieval distance;
- accumulated local progress and donor observation identity.

## Paired factors

For each anchor, run identical initial observation and deterministic retrieval:

- B1.4 with the recorded base next-sector condition;
- B1.4 with the supported target next-sector condition;
- B2.4 with the same base request;
- B2.4 with the same target request.

Only the delivered next-sector condition changes inside each base/target pair.

## Observable endpoint semantic score

The endpoint uses the frozen eye-sector prototype similarities already present
in the observable state. For base sector `b` and target sector `t`:

```text
score =
  0.5 * (
    endpoint_base_similarity[b] - endpoint_base_similarity[t]
    + endpoint_target_similarity[t] - endpoint_target_similarity[b]
  )
```

Positive score means the two requested conditions separate endpoints in the
declared semantic direction. Endpoint sector labels or progress never affect
retrieval.

## Development Gate

The result is
`next_condition_supported_path_effect_established_development` only when:

- all B1.4 and B2.4 base/target rollouts complete inside expert support;
- B1.4 base/target paths diverge for every eligible source episode;
- B1.4 endpoint semantic score is positive for every eligible source episode;
- the paired B1.4-minus-B2.4 semantic score is positive for every eligible
  source episode;
- selected transitions are never reused and held-out data remains unread.

Failure returns
`next_condition_supported_path_effect_not_established`. Either result remains
offline development evidence and cannot authorize held-out test, closed-loop
claims, simulation, real control, fine-tuning, or deployment.
