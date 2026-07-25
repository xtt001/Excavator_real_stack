# SimVerify G4-v2.1 History-Stitch Amendment

Status: `method_frozen_before_v2_1_build`

Evidence scope: `recorded-observation/offline empirical rollout`

This amendment is motivated by a specific emulator failure, not by policy
results. G4-v2 one-step expert retrieval passed every train-derived validation
envelope, but cumulative expert stitching deadlocked in 98 of 111 train
rollouts because a current-state-only nearest neighbor alternated between
similar recorded states.

G4-v2 and its immutable `offline_emulator_invalid` result remain preserved.
No held-out episode or policy condition result was inspected.

## Single method change

G4-v2.1 adds one tick of observable transition history:

- current observable state;
- previous-to-current observable-state delta;
- current executed action;
- previous-to-current executed-action delta.

Each of the four groups is independently train-standardized and contributes
one RMS-normalized distance group:

```text
distance^2
  = mean(current_state_z_delta^2)
  + mean(state_history_z_delta^2)
  + mean(current_action_z_delta^2)
  + mean(action_history_z_delta^2)
```

At a cycle start both history deltas are zero. After a stitched transition,
the deltas are computed from the actual previous stitched node and the newly
selected recorded successor.

## Unchanged prohibitions

Retrieval still cannot use:

- condition;
- target sector;
- phase or progress;
- candidate successor;
- privilege;
- held-out data.

No support threshold or pass value is changed. All one-step and cumulative
envelopes are regenerated from train source-episode leave-one-out under the new
distance representation. Condition rollouts remain forbidden unless the
history-stitch expert prerequisite passes.
