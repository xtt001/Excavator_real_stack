# SimVerify M3 Condition-Swap Replay Contract

Status: `implementation_frozen_replays_pending`

Evidence scope: `recorded-observation/offline`

The replay consumes M2 `condition_counterfactual_anchors_v1`. On each supported
anchor it holds the complete recorded observation history fixed and changes
exactly one of current sector or next sector. Unsupported anchors remain in the
result inventory without policy success metrics.

For each unique cycle the base condition is replayed once. Each supported
target condition receives an independent replay with policy and temporal
aggregation reset. Both traces preserve normalized raw chunk, direct raw
chunk, temporal aggregation, future runtime-safe copy, expert action, and
condition arrays.

## Response windows

- current-sector response: ready start through observable carry-transition
  proxy;
- next-sector response: observable dump-end proxy through ready end.

These windows are generated from frozen observable annotations. They are not
selected after seeing B1/B2 results.

The target direction is generated from train/validation sector swing-qpos
medians and the observable source action-to-qvel sign. No privilege field is
used.

## Metrics

Each supported anchor records:

- deadzone-effective L1 action effect;
- swing action delta and target-direction match;
- non-target-axis disturbance;
- per-tick effect for later repeat-noise-derived response latency;
- base and target task-event coverage/order;
- phase-coverage delta.

The replay does not itself decide G4. A later source-episode calibration must
compare B1 to B2, include repeated B1 noise, generate thresholds from the
frozen formulas, and only then freeze `gate_thresholds_v1.json`.

No held-out test, PACT runtime, Unity, AGX, Jetson, field network, or real
control is involved.
