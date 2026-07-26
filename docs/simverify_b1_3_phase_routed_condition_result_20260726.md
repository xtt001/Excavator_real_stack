# SimVerify B1.3 phase-routed condition result

Status: `terminal_for_b1_3_experiment`

Decision: `revise_condition`

Evidence scope: `recorded-observation/offline`

Closed-loop execution: `false`

Held-out test read: `false`

Control candidate: `false`

## Plain-language conclusion

The observable phase router works well enough to separate when the two
condition fields are allowed to act. The trained B1.3 policy also changes its
actions when the requested condition changes, and those changes are
concentrated in the intended phase.

That is not yet sufficient to say that the policy understands the condition
meaning. The `current_sector` signed response did not reliably beat the
same-architecture shuffled-condition B2.3 null. The `next_sector` signed
response did beat B2.3, but it did not reject all five wrong label-to-sector
semantic permutations across validation source episodes. Therefore the frozen
condition-understanding Gate fails for both factors.

In plain language: the new routing fixed much of the **when**, but the evidence
does not yet establish the complete and robust **what**. Transition stitching
was consequently not entered.

## Frozen question and one-factor change

B1.3 asked whether the unchanged ACT backbone could learn the declared
`current_sector` and `next_sector` meanings when:

- the two one-hot fields receive separate learned projections;
- a causal observable router enables `current_sector` only in the current-cycle
  phase;
- both condition projections are disabled in the neutral phase;
- `next_sector` is enabled only in the return/next-ready phase.

Dataset, source episodes, episode-level split, normalization, image transform,
ACT backbone, action chunk, optimizer, learning rate, batch size, seed, epoch
count, losses, validation schedule, checkpoint selection, and inference
precision remained frozen. B2.3 changed only the deterministic training
condition association and used the same phase-routed architecture.

## Why each Gate criterion exists

The Gate intentionally distinguishes five claims:

1. **Action sensitivity versus exact mask** asks whether changing the requested
   condition changes policy output at all. It rejects a condition-ignored
   policy, but cannot establish meaning by itself.
2. **Signed semantic margin versus B2.3** asks whether the action change points
   toward the requested sector more strongly than a model trained with shuffled
   condition associations. This rejects architecture-only or routing-only
   effects.
3. **Semantic permutation rejection** asks whether the declared
   left/center/right meaning fits better than each of the five other one-to-one
   label mappings. This rejects arbitrary or swapped label semantics.
4. **Phase specificity** asks whether the effect is larger in the declared
   routed window than outside it, and larger than both mask and B2.3. This
   distinguishes the intended phase contract from general action disturbance.
5. **Task-envelope preservation** prevents a condition response from passing by
   breaking the recorded observable event coverage or event order.

Both factors had to pass every criterion. Source episode, rather than anchor,
was the bootstrap unit so that many nearby anchors from one recorded episode
could not masquerade as independent evidence. Three B1.3 repeats bounded
inference noise. Held-out episodes `1,13,25,33` remained unread.

## Router prerequisite

Immutable artifact:

```text
/data/pingfan/Excavator_real_stack_data/simverify_observable_phase_router_v1
```

- decision: `pass_observable_phase_router_prerequisite`
- selected dwell: `2` ticks at 20 Hz
- validation minimum source-episode balanced accuracy: `0.963587`
- train-derived lower bound: `0.955373`
- validation source-episode boundary absolute offset q97.5: `7` ticks
- train-derived upper bound: `8` ticks
- every validation cycle reached neutral and next in order
- stored assignment and runtime recomputation parity: exact
- forbidden runtime inputs: absent
- manifest SHA256:
  `53b8b6c9d381b9365c87e0419861f98c11fc59e4e312846789bac1ab46de4163`
- checksums SHA256:
  `f715b5a08f511bd6a8e54f485f61a4629d2b705aa328a0a1186838f08c24c749`

The router reads only current source-domain qpos/qvel and its own past route
state. It does not read condition, expert action, future observation, event
label, progress, successor state, or privilege.

## Training evidence

Candidate B1.3:

```text
/data/pingfan/Excavator_real_stack_data/simverify_b1_3_phase_routed_condition_v1_seed0
```

- status: `completed`
- best epoch: `1999`
- best validation loss: `0.2082674205303192`
- checkpoint SHA256:
  `6a367af4527a726161e5017d26db55f5025eff96de1bc7e4ffc53c4a71de0857`

Shuffled-condition null B2.3:

```text
/data/pingfan/Excavator_real_stack_data/simverify_b2_3_phase_routed_shuffled_condition_v1_seed0
```

- status: `completed`
- best epoch: `1530`
- best validation loss: `0.24961721897125244`
- checkpoint SHA256:
  `2aab7ce6ca1b1018d7f1bd12d30386a37855cc0a0a44a6d2b01569baf123a2a3`

Both bundles record git commit
`f7ee73f4aaade7eafaf11268f65ef75c4022faea`, a clean
`v2.0.0-simVerify` worktree, sim-domain `actuator_speed_cmd`, and
`deployment_status=offline_evaluation_only`. Their dataset statistics SHA256
is identical:
`2f600ef381c636b3723afb5a89ae660e1f57b148934c8a2b24930b45260d39aa`.

Validation loss is diagnostic only and was not used as condition-understanding
proof.

## Fixed-observation replay inventory

Each replay contains 124 anchors, of which 45 satisfy the frozen local support
contract. It stores raw policy chunks, temporally aggregated actions, and
runtime-safe actions separately. No output was written back into an
environment.

| Package | Mode | Repeat | Checksums SHA256 |
| --- | --- | ---: | --- |
| `simverify_b1_3_condition_replay_validation_repeat0_v1` | requested B1.3 | 0 | `4595d18d41ff63db65c60989d3bc853b3449e785063e9f37902fd5731acd55c4` |
| `simverify_b1_3_condition_replay_validation_repeat1_v1` | requested B1.3 | 1 | `033b30c78acd8a0c40f5691542924eab51a76473a4c805cf393af4da18eceb30` |
| `simverify_b1_3_condition_replay_validation_repeat2_v1` | requested B1.3 | 2 | `815965cbc33de399d7ea52e70983c5d9e6405aa690a2a1651f7be06d38552b04` |
| `simverify_b1_3_masked_condition_replay_validation_repeat0_v1` | exact masked B1.3 | 0 | `aa61ff8716a8ff5f094bad984154514e97dc353f49834deb552e913ea0291acb` |
| `simverify_b2_3_condition_replay_validation_repeat0_v1` | requested B2.3 | 0 | `87a938efdcff7275dc6dc81716e862614ce7d4f802576af71f04a1a80436c04c` |

All checksum inventories were independently reverified after generation.

## Condition Gate result

Immutable artifact:

```text
/data/pingfan/Excavator_real_stack_data/simverify_b1_3_condition_causal_v2_validation_v1
```

- decision: `condition_understanding_not_established`
- recommended terminal status: `revise_condition`
- current-sector factor pass: `false`
- next-sector factor pass: `false`
- bootstrap: 100,000 nonparametric source-episode draws
- manifest SHA256:
  `f8124475f35bf6d3953cbc728f1c572f4642a46abf85ef414fdaea83555fa87a`
- checksums SHA256:
  `f2cda68c4dd3d2f7a88cf04c19417e1a5320bd15678d13bd0d52b8b84a1d73b3`

### Current-sector result

Passed:

- action sensitivity versus exact mask;
- intended-window phase specificity versus mask and B2.3;
- all five semantic permutation comparisons;
- task-envelope preservation.

Failed:

- signed semantic margin versus B2.3.

The two validation source-episode candidate-minus-null signed-margin deltas
were `+0.0136245` and `-0.0137431`. Their bootstrap mean was approximately
zero (`-0.0000593`), with q02.5 `-0.0137431`. The candidate therefore did not
reliably beat the shuffled-label null.

### Next-sector result

Passed:

- action sensitivity versus exact mask;
- signed semantic margin versus B2.3;
- intended-window phase specificity versus mask and B2.3;
- task-envelope preservation.

Failed:

- complete semantic identifiability.

Three of five wrong semantic permutations were rejected. Two permutations had
a source-episode lower bound of exactly zero and therefore could not be
rejected. The affected validation source episode had only one supported
next-sector anchor, so the result is not robust enough to claim full
left/center/right meaning even though the signed response and timing are
encouraging.

## Authorization and next experiment

The frozen contract authorizes transition stitching only after both condition
factors pass. Therefore:

- transition stitching: `not_entered`;
- held-out test: `not_authorized`;
- simulator closed loop: `not_authorized`;
- Jetson or real hardware: `not_authorized`;
- deployment or checkpoint promotion: `not_authorized`.

The next authorized action is a new, separately frozen one-factor condition
revision. It should target semantic association rather than add another timing
mechanism: the observable router prerequisite already passed, action
sensitivity is present, and phase specificity passed. A future revision must
be trained with a same-architecture shuffled null and must pass the same
fixed-observation semantic controls before any supported transition-stitch
test is reconsidered.

This result does not say that the recorded data are unusable, and it does not
say that the policy would fail or succeed in closed loop. It says only that
the present offline evidence is insufficient to establish robust
current/next-sector condition understanding.
