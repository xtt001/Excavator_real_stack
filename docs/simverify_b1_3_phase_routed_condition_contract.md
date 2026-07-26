# SimVerify B1.3 causal observable phase-routed condition contract

Status: `frozen_before_phase_router_artifact_and_policy_implementation`

Evidence scope: `recorded-observation/offline`

Closed-loop execution: `false`

Held-out test read: `false`

## Question

Can the same ACT backbone learn the declared `current_sector` and
`next_sector` meanings when the two fields no longer share one undifferentiated
low-dimensional projection, and when a causal observable router permits each
field to affect only its declared task phase?

This experiment follows the completed B1/B1.1/B1.2 `revise_condition`
decision. It does not reopen or overwrite those immutable results.

## One primary factor

B1.3 keeps the B1 dataset, source episodes, episode-level split,
normalization, image transform, ACT backbone, action chunk, optimizer,
learning rate, batch size, seed, epoch count, KL/deadzone losses, validation
schedule, checkpoint selection, and inference precision unchanged.

The only primary factor is the condition representation and routing:

- qpos/qvel retain their existing state projection;
- `current_sector` and `next_sector` receive separate learned projections;
- a frozen causal three-state router gates the projections;
- `current_sector` is enabled in `current`;
- both condition projections are disabled in `neutral`;
- `next_sector` is enabled in `next`.

The routed embeddings are summed with the qpos/qvel state embedding before the
unchanged ACT transformer. They are not added as new Transformer source tokens.
The ACT action head and action objective are unchanged.

The VAE action encoder receives qpos/qvel only. Condition cannot bypass the
router through the training-only latent path.

## Frozen route semantics

The three route states are defined from the already frozen observable M0 event
contract:

```text
current = ready_start through carry_transition_proxy, inclusive
neutral = after carry_transition_proxy and before dump_end_proxy
next    = dump_end_proxy through ready_end, inclusive
```

These are observable proxy phases. They do not claim soil contact, payload, or
physical excavation success.

## Causal observable router

Runtime inputs are only the current source-domain qpos `[4]`, current
source-domain qvel `[4]`, and the router's own past state. The router may not
read:

- images from a future tick;
- current or future expert action;
- current/next condition;
- cycle outcome or target sector;
- event/phase/progress labels;
- candidate successor state;
- terrain, bucket mass, exact tip position, planner state, or other privilege.

At an explicit cycle reset, the router starts in `current`. A train-only
diagonal-Gaussian classifier predicts one of the three route states from the
current qpos/qvel. A monotonic state machine permits only:

```text
current -> neutral -> next
```

It never moves backward. A transition requires a fixed number of consecutive
predictions of the immediately following state.

Classifier location/scale, class means/variances, and dwell are generated
before policy training. Dwell candidates are the predeclared integer range
`1..10` ticks, corresponding to `0.05..0.50 s` at the frozen 20 Hz rate.
Selection uses train source-episode leave-one-out only: maximize the mean
source-episode balanced route accuracy, with the smaller dwell winning an
exact tie. Validation cannot change the selected dwell or classifier.

The same frozen router parameters and state-machine implementation must
generate:

- train/validation route assignments used by the dataset loader;
- runtime routes used by recorded-observation replay;
- route diagnostics stored with replay traces.

Missing assignments, a cycle-id mismatch, non-finite qpos/qvel, a backward
route request, or router provenance mismatch fails closed.

## Router prerequisite Gate

Before B1.3 training, the immutable router artifact must:

1. fit only accepted train cycles;
2. audit train generalization by source-episode leave-one-out;
3. evaluate validation once with the all-train frozen classifier;
4. keep held-out episodes `1,13,25,33` unread;
5. store per-source-episode accuracy, balanced accuracy, confusion, and
   boundary offset;
6. store every train/validation `(episode_id, cycle_id, tick) -> route`
   assignment and its SHA;
7. prove runtime recomputation is exactly equal to the stored validation
   assignments.

Train-derived thresholds are:

- validation source-episode balanced accuracy must not fall below the train
  leave-one-episode-out `q02.5`;
- validation per-source-episode `q97.5` absolute route-boundary offset must not
  exceed the maximum corresponding train leave-one-episode-out value;
- every validation cycle must reach `neutral` and `next` in order;
- no forbidden field may enter classifier fitting or runtime routing.

Failure stops before policy training with `revise_condition_router`.

## Condition-understanding controls

A same-architecture shuffled-label null, B2.3, is required. B2.3 differs from
B1.3 only by the already frozen deterministic train-valid-start condition
shuffle. Comparing B1.3 only with the older shared-projection B2 would confound
condition routing with label association.

After completed B1.3 and B2.3 training:

1. run three requested-condition B1.3 validation replays;
2. run one identical-token masked B1.3 replay;
3. run one requested-condition B2.3 replay;
4. apply the fixed-observation causal v2 metrics using source-episode
   bootstrap and all five non-identity semantic permutations;
5. preserve task event coverage/order and the raw/aggregated/runtime-safe
   action separation.

Both `current_sector` and `next_sector` must pass:

- action sensitivity above the exact masked null;
- signed semantic margin above B2.3 and repeat noise;
- identity semantics above all five wrong permutations;
- positive intended-window phase specificity above mask and B2.3;
- task-envelope preservation.

Router prediction accuracy alone is not condition understanding. Action
sensitivity alone is not semantic understanding.

## Transition-stitch authorization

Only a complete fixed-observation condition Gate pass authorizes the next
offline step. The existing G4-v2 and G4-v2.1
`offline_emulator_invalid` artifacts remain immutable and cannot be called
policy failures.

Any later per-step transition-stitch method must first pass expert-only
train/validation cumulative support calibration. If expert transitions cannot
compose inside the frozen support envelope, policy stitching stops with
`offline_emulator_invalid`; it may not force an arbitrary nearest neighbor.

## Terminal decisions

- router prerequisite fails: `revise_condition_router`;
- router passes but either condition factor fails: `revise_condition`;
- condition Gate passes but no expert-valid stitcher exists:
  `condition_understanding_established_offline_stitch_unavailable`;
- condition Gate and an expert-valid supported stitcher pass:
  `condition_understanding_and_supported_progress_established_offline`.

None of these decisions is `control_candidate`. None authorizes held-out,
Unity/AGX closed loop, Jetson, real hardware, real fine-tuning, or deployment.

