# G49 new-data first-batch ACT training contract

## Status and capability boundary

This batch trains offline ACT candidates on the new-style dataset. It does not
read the frozen new-style test block, change a field/runtime policy, claim
terrain-causal understanding, or make an argmax intent head executable.

The batch may answer only:

1. how an eye2 continuous ACT behaves on the new distribution;
2. whether G38-style transition supervision helps on that distribution;
3. whether G42-style effective-action semantics help without a runtime gate;
4. whether video6/video7 add value under the existing naive fusion path;
5. whether additive camera/role identity improves that existing four-camera path.
6. whether N1 transition supervision transfers through the N4 four-camera role
   path without changing the remaining training contract.

It cannot establish that four-camera eye/stick pooling, role competition,
terrain labels, cross-session generalization, hydraulic response modeling, or
real closed-loop control is solved.

Current execution status: N0--N4 and the selected N5 follow-up are complete. N5
ran 2000 epochs on 2026-07-15, completed in 55 minutes 27 seconds, and selected
epoch 290 with validation loss `0.3761702061`. Completion is established by
`run_metadata.json` status `completed`, not by checkpoint presence alone.

## Frozen data contract

- view: `/data/pingfan/Excavator_real_stack_data/g48_new_trainval_view_v1`
- train: 120 chronological new-style episodes
- validation: 20 chronological new-style episodes
- test: zero episodes linked into the training view
- source IDs `105..109`: forbidden from this batch
- manifest SHA-256:
  `93d16983e70b0a614908cdf232d8f5879946b5e6b6175ae7dee6dfc5199e0348`
- split SHA-256:
  `4b796da6520e1469ec4cd0d1d94af7a9886c1fde8e1a16b123ea36a474aa3ebc`
- action domain: direct policy output with identity scale `[1,1,1,1]`

All candidates use qpos, a 20-query ACT chunk, seed 0, no image transform,
2000 epochs, and the same validation block. No old policy checkpoint initializes
this batch.

## Candidate matrix

| ID | Cameras | Training difference | Intended comparison |
|---|---|---|---|
| N0 | video4/video5 | ordinary continuous ACT | new-style reference |
| N1 | video4/video5 | transition-window promote loss plus transition sampling | N1 vs N0: transition supervision |
| N2 | video4/video5 | effective-action target, ternary auxiliary head, raw continuous tie-breaker | N2 vs N0: target semantics |
| N3 | video4..video7 | ordinary continuous ACT with existing naive spatial concatenation | N3 vs N0: extra cameras |
| N4 | video4..video7 | N3 plus additive camera and eye/stick role identity | N4 vs N3: role identity |
| N5 | video4..video7 | N4 camera/role identity plus N1 transition-window promote loss and transition sampling | N5 vs N4: transition supervision under fourcam; N5 vs N1: remaining camera-path effect |

N4 and N5 do not implement eye/stick group pooling or a learned cross-role
fusion gate. Describing either as that stronger architecture is forbidden.

## New-train-only effective-action class audit

The N2 ternary class weights are derived only from the 120 training episodes.
After the existing train-exclude mask, the global axis-row counts are:

- neutral: 204,941
- positive: 44,973
- negative: 42,550

Neutral-normalized inverse-frequency ratios are
`[1.0, 4.556979, 4.816475]`. These replace the historical fixed
`[1,4,4]`; validation and test do not participate in the estimate.

## Performance and persistence contract

Every candidate uses four DataLoader workers, prefetch factor 2, persistent
workers, pinned memory, resume checkpoint cadence 100, model-only periodic
checkpoint cadence 500, and periodic retention 3. Best and periodic candidates
are inference-only; latest is resume-capable.

## Selection and stop conditions

Validation loss is secondary. Every candidate must later be compared on:

- per-episode first effective main-axis startup;
- all transitions and per-axis active recall;
- idle false-active and wrong/opposite direction;
- two-axis and three-axis preservation;
- deadzone hit and recovery delay;
- release/tail behavior;
- recursive state-hold raw and assist views;
- a frozen query-0 intent probe that never enters action execution.

Any non-finite loss, missing camera, split/hash drift, forbidden source episode,
non-empty output collision, or failed run metadata stops the batch. The frozen
new-style test block remains unread until candidates, checkpoints, thresholds,
and evaluation code are frozen.

## Prepared artifacts

- configs: `testbed/testbed/configs/*g49_n*_2000.yaml`
- preflight report:
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/training_preflight_report.json`
- preflight result: N0--N4 all `ready`, each with 120 train / 20 validation
- N5 preflight:
  `/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/training_runs/n5_new_fourcam_role_transition_boundary_2000/preflight.json`
- N5 config SHA-256:
  `41a498b4048051880da2e0bdff2b0de4dfc808150669515ae5fb70ccfaae597a`
- N5 best-checkpoint SHA-256:
  `0c9b755447f1c06a893394fb1111b9365eb47a8670523b6eeaef8b2df7e13b0e`

## N5 result and stop decision

N5 answered the intended cross comparison but did not win the original
exact-target/continuous-error ranking. It raised any-axis armed natural startup
to 20/20, and every validation episode already had a deadzone-effective output
at recording step 0. This is allowed by the training and recording contract: the
recorded preparation prefix is not idle ground truth. Target-axis startup at the
expert onset and target state-hold startup both remained 5/20, identical to N4.
All 20 startup state-hold anchors emitted a direction absent from the exact
anchor frame, but that metric is anchor-relative rather than a global wrong-axis
label.

N5 also worsened open-loop MAE to `0.1803` and idle false-active to `18.4%`,
compared with N4's `0.1561` and `11.9%`. It recovered 169/261 target state-hold
anchors, above N4's 139 but below N1's 186, while unexpected-effective anchors
rose to 102 from N4's 66. The later task-support audit changes the interpretation:
16/20 armed starts are wholly within the same episode's 40-tick expert support,
and every remaining direction combination exists in the 120 training episodes.
Transition supervision therefore transferred executable task-level activation;
whether it is sufficiently scene-selective remains unproven rather than
falsified by the exact-anchor metric.

Stop this training line here. Do not tune the transition weight against these 20
validation episodes and do not add a hard argmax projection. The next model
change, if pursued, must isolate pair-aware eye/stick evidence extraction and
whole-pair dropout, with held-idle and release/tail diagnostics declared before
training.
