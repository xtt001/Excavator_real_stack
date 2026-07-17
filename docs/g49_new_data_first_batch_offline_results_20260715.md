# G49 New-Data First-Batch Offline Results

Date: 2026-07-15

Semantic contract revised: 2026-07-16. The original artifacts remain factual
traces, but the old `recovered/deadlocked/unexpected/false-active` judgmental
schema is superseded. The values below are single-demo relations unless an
independent task or physical label is named.

## Test intent

This evaluation asks whether a policy trained on the new-data view can emit an
executable action in the expert axis and direction, especially when an
observation is held fixed at an ineffective-to-effective transition.

It does **not** claim hydraulic fidelity, closed-loop digging success, or field
readiness. It treats the measured direct-output mechanical deadzone as a hard
inactive/active boundary.

## Frozen evaluation contract

- Dataset view:
  `/data/pingfan/Excavator_real_stack_data/g48_new_trainval_view_v1`
- Split:
  `/data/pingfan/Excavator_real_stack_data/g48_new_trainval_view_v1/train_val_split.yaml`
- Split SHA256:
  `4b796da6520e1469ec4cd0d1d94af7a9886c1fde8e1a16b123ea36a474aa3ebc`
- Validation composite episode IDs: `10120..10139` (20 episodes).
- Test episodes remain excluded.
- Checkpoints: `policy_best.ckpt` only.
- Open-loop: temporal aggregation enabled, no image transform, all 20 validation
  episodes, 11,284 frames.
- State-hold: raw direct policy output, runtime gates disabled, assist disabled,
  qvel zero after hold, horizon 20, full horizon traced after demo-target
  reproduction.
- State-hold anchors: every per-axis ineffective-to-effective transition in the
  20 validation episodes: 261 anchors, including 20 startup and 241 mid-cycle.
- Direct-output deadzones:
  positive `[0.661, 0.259, 0.500, 0.408]`, negative
  `[0.721, 0.357, 0.500, 0.508]` for swing, boom, stick, bucket.

The validation startup set reflects a consistent later-recording task sequence:
18 first-crossing anchors are `stick-positive` and 2 are `boom-negative`.
Within 40 ticks, 19/20 episodes contain the same
`stick-positive / boom-negative / bucket-positive` direction set. This is not
evidence of random expert behavior, but it does mean startup results do not
establish behavior for other initial task sequences or axis/direction coverage.

## Candidates

| ID | Cameras | Training idea | Best checkpoint SHA256 |
|---|---|---|---|
| N0 | eye2 | continuous ACT baseline | `b27cc12126373bd0dd7fe2429b82b960d306632f107ffbc5281a41be59ab28ff` |
| N1 | eye2 | transition/deadzone-boundary losses | `189ff57c3f2770d82656b095e2bcfe1157f3d5579766f0fd79d431e076dfdcb5` |
| N2 | eye2 | effective-action auxiliary semantics | `8bd19b39bd4e75cc9a4fc250c5bb4fdcb405a1de7fde4e996c00a3672a36cd17` |
| N3 | fourcam | naive four-camera continuous ACT | `18cab59f7372e2ead79d53b068fa51d6d1898f6c0bebba53a25119f9446d4e0c` |
| N4 | fourcam | camera-role identity encoding | `674c86f54ff0cfbd29de90500bbbb4e6e68d31a091bae2ce89d90f1c5ca1ca99` |
| N5 | fourcam | N4 camera-role identity plus N1 transition supervision | `0c9b755447f1c06a893394fb1111b9365eb47a8670523b6eeaef8b2df7e13b0e` |

## Primary results

Interpretation amendment: the table below remains the original factual
diagnostic summary, but it is not a complete benchmark. The subsequent metric
audit found that the 20 startup anchors are not axis-balanced, demo-target
reproduction can coexist with anchor-extra effective axes, micro single-demo
recall hides stick weakness, and the multi-axis column is only single-demo
similarity. Use the corrected companion metrics in
`docs/g49_evaluation_metric_audit_20260715.md` before ranking a candidate.

| ID | Single-demo MAE | Single-demo active recall | Active on demo-idle frames | Exact demo-anchor startup | Demo multi-axis directions preserved | Startup demo target reproduced | All demo targets reproduced | Anchor-extra effective anchors | Opposite-to-demo-target ticks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N0 | 0.1650 | 81.1% | 14.0% | 2/20 | 54.5% | 3/20 | 159/261 | 90 | 40 |
| N1 | 0.1767 | **82.9%** | 17.5% | **10/20** | **56.7%** | **14/20** | **186/261** | 102 | 40 |
| N2 | 0.1630 | 80.3% | 14.0% | 6/20 | 51.8% | 7/20 | 164/261 | 90 | 35 |
| N3 | **0.1507** | 77.9% | **8.5%** | 0/20 | 50.4% | 1/20 | 134/261 | 67 | **0** |
| N4 | 0.1561 | 77.2% | 11.9% | 4/20 | 46.9% | 5/20 | 139/261 | **66** | 20 |
| N5 | 0.1803 | 80.6% | 18.4% | 5/20 | 53.8% | 5/20 | 169/261 | 102 | 40 |

`Single-demo active recall` and `active on demo-idle frames` are per-axis/frame measurements after
applying the same deadzones to expert and raw policy actions. `Exact startup at
anchor` checks the open-loop action at the transition frame. `Held startup`
allows demo-target reproduction during the 20-tick state-hold horizon.

## State-hold demo-target reproduction by axis

| ID | Swing (44) | Boom (91) | Stick (65) | Bucket (61) |
|---|---:|---:|---:|---:|
| N0 | 86.4% | 64.8% | 7.7% | 93.4% |
| N1 | **95.5%** | **68.1%** | **35.4%** | **96.7%** |
| N2 | 79.5% | 61.5% | 24.6% | 93.4% |
| N3 | 61.4% | 46.2% | 10.8% | 95.1% |
| N4 | 70.5% | 50.5% | 7.7% | 93.4% |

## What the tests establish

1. **Regression quality and executable onset are different objectives.** N3 is
   the best continuous regressor, improving MAE by 8.6% over N0, but it is the
   lowest startup demo similarity: zero exact startup actions and one held
   startup demo target reproduced. N1 has the worst MAE, but reproduces 14 of
   20 held startup demo targets. This is not a correctness ranking.

2. **Transition supervision changes conditional target reproduction.** N1
   raises held demo-target reproduction from 159/261 to 186/261 and stick-target
   reproduction from 7.7% to 35.4%. The observation-held counterfactual verifies
   that difference, but cannot establish which alternative action is correct.

3. **The N1 difference is multi-dimensional.** Relative to N0, activity on
   demo-idle frames rises by 3.5 points and anchor-extra-effective anchors rise
   from 90 to 102. Opposite-to-demo-target ticks remain 40. These facts show
   lower single-demo selectivity, not independently labelled unsafe behavior.

4. **Four cameras contain useful information, but the current continuous fusion
   uses it conservatively.** N3 improves continuous MAE, bucket MAE, idle
   false-active rate, and opposite-direction safety. At the same time it loses
   onset recall, particularly for stick. More visual evidence did not merely
   add noise; it improved steady-state/magnitude estimation while making the
   ambiguous start decision collapse toward inactivity.

5. **Camera identity alone is insufficient.** N4 recovers more startup anchors
   than N3 (5 versus 1), but remains far behind N1 and loses N3's zero-opposite
   result. A learned role identity tells the transformer which camera supplied
   a token, but does not by itself enforce pair-level evidence extraction or an
   axis-specific decision rule.

6. **N2 is not a test of the proposed hard argmax execution structure.** Its
   phase head is auxiliary during training; normal ACT inference still executes
   the continuous action head. N2 only tests whether auxiliary effective-action
   semantics reshape the continuous representation. Its modest gains therefore
   do not falsify a per-axis tri-state intent head with structural projection
   beyond the deadzone.

7. **N5 transfers task-level activation; scene selectivity is still unresolved.**
   Combining N4 camera-role identity with N1 transition supervision makes the
   raw policy deadzone-effective under the armed any-axis startup diagnostic in
   20/20 episodes. Starting at recording step 0 is permitted because the
   preparation prefix is not idle ground truth. Sixteen of the 20 armed starts
   are wholly within the same episode's 40-tick single-demo support; all four
   remaining direction combinations occur in the 120 training episodes. It
   reproduces 5/20 startup demo targets, open-loop MAE is the worst of N0--N5,
   and activity on demo-idle frames is 18.4%. The anchor-extra label cannot turn
   20/20 starts into 20 wrong actions. The liveness result is not a
   promotion result because held-idle, release, self-generated state, and real
   response evidence remain missing.

## Capability boundary

- Open-loop replay advances through expert observations and cannot reveal all
  execution-induced distribution shift.
- State-hold is a single-demo-target counterfactual with frozen images/qpos and
  zero qvel. Target non-reproduction is not generic deadlock; anchor-extra is
  not invalidity. It does not simulate dynamics, terrain, or hydraulics.
- The validation set is new-data validation, not the sealed final test set and
  not a field trial.
- Startup composition is heavily skewed to stick-positive, so the 14/20 N1
  result must not be quoted as generic four-axis startup accuracy.
- Unexpected-active counts are anchor-level safety warnings; because each
  anchor traces 20 ticks, they must be reviewed together with direction and
  axis traces before any live use.

## Frozen query-0 feature probe

The previous probe CLI contained an obsolete formal-run assertion that the
validation set must contain exactly five bucket-positive startup anchors. It
was replaced by a model-independent contract derived from each validation
HDF5's expert action plus the same direct-output deadzones. Every model's
reported startup rows must exactly match that inventory. The real G49 inventory
contains 20 rows: 18 stick-positive and 2 boom-negative.

The formal probe used all 73,157 train frames and 11,284 validation frames. It
froze each ACT model, captured the inference-time decoder query-0 feature before
the action head, and trained only a fixed `Linear(512, 12)` per-axis ternary
head with train-only inverse-frequency class weights.

| Feature source | All-frame active recall | Idle false-active | Transition recall | Startup recall | Double-axis active axes preserved |
|---|---:|---:|---:|---:|---:|
| N0 | 86.3% | 20.5% | 61.7% | 7/20 | 64.9% |
| N1 | **87.6%** | 22.1% | **69.3%** | **16/20** | **66.4%** |
| N3 | 85.7% | **14.9%** | 53.3% | 11/20 | 61.6% |
| N4 | 85.2% | 18.5% | 53.6% | 12/20 | 60.9% |

This establishes that startup information is present in the frozen
representation. The strongest evidence is N3: its executed continuous action
has zero exact startup recoveries, while a linear head reads 11 of 20 startup
directions from the same query-0 representation. The extra cameras therefore
did not erase startup evidence.

It does not establish that strict argmax projection is safe. The linear probes
raise idle false-active rates to 14.9--22.1%, and double-axis exact intent-vector
rates remain only 38.6--47.2%. A hard projection would turn those class errors
into guaranteed effective commands.

Formal probe artifact:
`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/frozen_intent_probe_formal_train120_val20/experiment.json`
(SHA256
`15a2755c22d6ab8aa657824d21603c8ce5181b2d6ea980dae1e8015aa751195f`).

## Temporal aggregation isolation

Open-loop replay was repeated for N0/N1/N3/N4 with temporal aggregation
disabled, executing only the current query-0 action. This did not recover the
startup gap:

| ID | Startup default / query-0 | MAE default / query-0 | Idle false-active default / query-0 |
|---|---:|---:|---:|
| N0 | 2 / 2 | 0.1650 / 0.1709 | 14.0% / 15.0% |
| N1 | 10 / 11 | 0.1767 / 0.1839 | 17.5% / 19.7% |
| N3 | 0 / 1 | 0.1507 / 0.1575 | 8.5% / 10.5% |
| N4 | 4 / 4 | 0.1561 / 0.1607 | 11.9% / 12.9% |

Temporal aggregation improves continuous MAE and suppresses false activation;
it is not the primary source of the startup failure. The main information loss
occurs between the decoder representation and the continuous action decision.

## Decision and next slice

- Keep N1 as the current **selective behavioral reference** for onset/liveness.
- Keep N3 as the current **continuous-regression and conservative-safety
  reference**.
- Do not promote any N0-N5 checkpoint to live control.
- The frozen-intent probe repair and formal N0/N1/N3/N4 run are complete.
- Startup intent is linearly readable, but the probe's false-active surface
  prevents immediate strict-argmax promotion. The next eye2 candidate must be a
  predeclared safety-controlled revisit of factorized intent/effort, not a
  repeat of the old H3 experiment or a validation-tuned threshold sweep.
- The old H3 result remains relevant: hard projection amplified occasional
  classification errors into wrong effective commands. The new-data revisit is
  justified only by two changed facts: 120 training episodes now contain
  effective labels for every axis, including 8,548 stick-active frames, and the
  frozen N1 representation reaches 16/20 startup recall. Its hard gates must
  include idle false-active, unexpected axes, opposite direction, multi-axis
  preservation, and release/tail behavior in addition to recovery.
- Only after the eye2 execution head is validated, add the same head to a
  four-camera model with pair-aware evidence extraction and per-axis fusion.
  Camera pairs should be jointly available evidence, not a forced global
  eye-versus-stick winner.
- N5 closes the simple four-camera role plus transition-supervision cross cell.
  Its failure mode rules out treating more active output as better startup. A
  further four-camera candidate must change evidence extraction itself and must
  be paired with held-idle and release/tail diagnostics rather than another
  transition-weight sweep on the same validation set.

## Artifacts

Evaluation root:
`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation`

Each `nN_open_loop_val20` directory contains per-episode `actions.npz`, plots,
`episode_metrics.csv`, and `collection_summary.json`. Each
`nN_state_hold_raw_val20_h20` directory contains full anchor JSONL/CSV traces,
the direct-output deadzone artifact, `state_hold_summary.json`, and
`run_summary.json`.

Collection/state-hold summary SHA256 pairs:

- N0: `c529dd67d1a10be7fcf19ffa9e97327e31778640278c46a630cd763193dcb5d0` /
  `27b24a30644b57220067164d87f0da8e44aa11b91bd7f719256b7b22189a134a`
- N1: `55c77122da56c4c7b3acf3eb27ed4625274a4b6a804570eee6701485f8e65e63` /
  `4c36f0d09647da3f166a99ed3fd85665d4ae30c60a35a48f4a2491c1cbf92f57`
- N2: `08647636dca6b503f5e6906740484d62e52b03ae6871278e2f142bef2011f9e8` /
  `86e0128a2df356e6eda1ea0be3c21e5ee66002b9e8bfc9c805fc730d40788091`
- N3: `44cddf2ba9ec4301adf7a20eba6e252538b986234113b00a02c85327485b95e5` /
  `9e088ec6d41f6b97bfb92c1ca3ac898c29163f2e2cf3e7240f0efd76830a6a03`
- N4: `3a47cfaf1539cd7f37d750491d2847a438298f5f9b01a9b34c1000bbc9fe6510` /
  `77917adfba739d5bb6a77c8d955980d12be3420ca4774b3f540e3a0b259cb5ce`
- N5: `c399d5bb6a5d8b075e92c77c2ba6f4e59759e55ae12ab45e42afe22c26d7d7e1` /
  `98b90b4e1709532aa3b172497fbac03248a426d23c62efd574834cfaab56b545`
