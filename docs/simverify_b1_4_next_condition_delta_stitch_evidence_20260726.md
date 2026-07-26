# SimVerify B1.4 next-condition and delta-stitch evidence — 2026-07-26

Current bounded-slice decision:
`next_condition_supported_path_effect_established_development`

Evidence scope: `recorded-observation/offline development`

Independent validation: `false`

Closed-loop execution: `false`

Held-out test read: `false`

This slice answers two separate questions:

1. does the next-only B1.4 checkpoint use the declared next-sector condition
   with identifiable semantics on fixed recorded observations; and
2. when every policy action selects a supported recorded transition, does that
   condition effect accumulate into different observable paths?

The answer to both development questions is yes. This is not a simulator or a
physical task-success result.

## Fixed-observation semantic Gate

B1.4 removes current-sector input and keeps only next-sector as the policy
condition. The matched B2.4 control shuffles that condition during training.

The immutable support-aware Gate passed all frozen criteria on eligible
validation source episodes 12 and 34:

- action sensitivity exceeded the exact masked-condition null;
- signed semantic margin exceeded B2.4;
- response was concentrated in the declared next-condition phase;
- identity semantics beat all five wrong sector permutations;
- observable event coverage and order were preserved.

Artifact:

`/data/pingfan/Excavator_real_stack_data/simverify_b1_4_next_condition_causal_validation_v1`

- decision:
  `next_condition_understanding_established_offline`;
- manifest SHA-256:
  `119ff2805a160b8ed16739f8bd6d64a7f3d8ba28dee80acb20e0f15b03589e4e`;
- Gate SHA-256:
  `5447674efbfe174fc073e47289c74906669aacd5fe56f373da50a2d5267baa75`;
- checksums SHA-256:
  `98edb4f4c32ee71cb5aba50e37eaf99bcf005e54e243662c6f1a46ad6cae9bc2`.

## Per-step transition method

G4-v2 used the nearest recorded state-action transition at every step, but
replaced rollout progress with the selected row's absolute cycle progress.
Ready-start and ready-end are observably similar, so unrelated phase values
jumped and alternated. G4-v2.1 added raw one-tick history but retained this
absolute-progress defect and exhausted support.

G4-v3 implements the intended cumulative test:

- retrieve from current observable state plus current action;
- exclude the current donor source episode;
- never reuse one recorded transition in one rollout;
- advance to the selected recorded successor;
- accumulate only that transition's local progress delta;
- never use absolute progress, phase, condition, target, successor, future
  state, or privilege for retrieval.

Every expert rollout has a paired median-action null. This prevents completion
from being explained by merely consuming arbitrary positive annotation deltas.

### Immutable v3 result

Artifact:

`/data/pingfan/Excavator_real_stack_data/simverify_transition_delta_stitch_calibration_v1`

- train expert: 111/111 completed;
- validation expert: 31/31 completed;
- train median-action null: 0/111 completed;
- validation median-action null: 0/31 completed;
- v3 decision: `offline_emulator_invalid_v3`;
- manifest SHA-256:
  `7607825587aa06c53be9b3299f23baa05b3f2ab2aa69102b8ea9782f3fd6321b`;
- Gate SHA-256:
  `431284d5efb8ea75834215e340b0885953cef29bb11e781ebb7e178054a2b176`;
- checksums SHA-256:
  `28da0de39bd8e797afe1c1400ee4debee504ff293faabbb47fe70796e39593a8`.

The fail is preserved. Its only failed criterion was validation
source-episode q97.5 completion steps `377.95` versus frozen train limit
`376.9425`, a difference of about 1.01 ticks.

### v3.1 Gate-design audit

The old step statistic compares groups with unequal cycle counts: train source
episodes contain 4–11 cycles, while validation episode 12 contains 15. A q97.5
from the larger group lies closer to the same observed maximum. Nested
train leave-one-source-episode audit rejects 1/15 train episodes under the same
rule.

V3.1 therefore preserves v3's fail but treats speed similarity as diagnostic,
while retaining hard completion, action-null, retrieval-support, uniqueness,
and frozen-budget requirements. Validation had already been inspected, so this
is explicitly development evidence.

Artifact:

`/data/pingfan/Excavator_real_stack_data/simverify_transition_delta_stitch_gate_audit_v1`

- decision:
  `pass_expert_delta_stitch_development_prerequisite_v3_1`;
- manifest SHA-256:
  `1a73404fb1d45faa13b1397e4162e53f4b45af4a34b27bd66a961544bf78f544`;
- Gate SHA-256:
  `f299ec1303f22474d7db737edb02427149eea2e2826bd06bca37c046997a8c0c`;
- checksums SHA-256:
  `35e69493814dce21855f98d10c13734470364abfb337572463f0f5fcc502bfc6`.

## Conditioned policy delta-stitch

The final development experiment used 21 supported next-sector anchors:

- source episode 12: 8;
- source episode 34: 13.

For each anchor it ran B1.4 base/target and B2.4 base/target conditions from the
same initial observation, for 84 total causal offline paths. Every tick used the
policy's future-runtime-safe temporal-aggregation action for retrieval.

All 84 paths completed inside the expert support radius:

| Metric | Episode 12 | Episode 34 |
| --- | ---: | ---: |
| B1.4 base/target completion | 100% | 100% |
| B2.4 base/target completion | 100% | 100% |
| B1.4 path divergence | 100% | 100% |
| B1.4 endpoint semantic score | 0.011483 | 0.016354 |
| B2.4 endpoint semantic score | 0.001613 | -0.009495 |
| B1.4 minus B2.4 | 0.009870 | 0.025849 |

Across all paths:

- B1.4 steps: min 245, median 285.5, max 402;
- B2.4 steps: min 264, median 334, max 419;
- maximum retrieval distance: 0.580109 versus support radius 0.821554;
- transition reuse violations: zero.

Artifact:

`/data/pingfan/Excavator_real_stack_data/simverify_b1_4_condition_delta_stitch_development_v1`

- decision:
  `next_condition_supported_path_effect_established_development`;
- manifest SHA-256:
  `c5043c389589aed1b83c96694326a101de817016096cd38dde35d350f8f203df`;
- Gate SHA-256:
  `ec1561b93cb4a5fcf5d178b4f3b3c99fb057b3908c25739817a2e1c36e7ae44d`;
- checksums SHA-256:
  `76a8d34fca6dc2d811faec95f7666f25d7ee5156bdd0ec877628d9d11493b46b`;
- checksum inventory: 89/89 files verified.

Every trace separately preserves the raw normalized chunk, raw direct chunk,
temporal-aggregation action, future-runtime-safe action, selected transition
identity, retrieval distance, accumulated local progress, and donor observation
identity.

## Plain-language conclusion

The new B1.4 model is no longer merely “reacting when a token changes.” On the
available recorded observations, it distinguishes the requested next sector at
the declared time, rejects wrong left/center/right meanings, emits different
actions, and those actions select different supported data paths whose
observable endpoints move in the requested semantic direction more strongly
than the shuffled-condition model.

The nearest-transition idea is useful after one correction: accumulate each
selected transition's local increment; do not copy its absolute cycle
position. The action-null result shows that the completion is action-dependent,
not automatic accumulation from arbitrary rows.

This still does **not** mean the excavator completed a closed-loop dig. The
environment response at each tick was borrowed from recorded data. It cannot
represent an unseen soil response, payload change, contact failure, or action
outside recorded support.

## M5 status and next Gate

This slice removes the current `revise_condition` technical blocker in
development evidence, but it does not issue a final M5 promotion. G5 two-cycle
continuity, G6 runtime equivalence, the frozen held-out decision, and
real-transfer audit remain incomplete.

The next bounded Gate should be G5:

1. freeze an expert-only two-cycle delta-stitch prerequisite;
2. integrate the first ready-to-ready boundary without resetting the policy;
3. change only the next-sector condition at that boundary;
4. require both cycles' observable events, bounded ready discontinuity, and
   second-cycle path initiation;
5. compare B1.4 with B2.4 and an unchanged-condition null;
6. keep held-out locked until the complete Gate threshold file is frozen.

No checkpoint is promoted to `real_finetune_candidate`, shadow, control, or
deployment by this result.
