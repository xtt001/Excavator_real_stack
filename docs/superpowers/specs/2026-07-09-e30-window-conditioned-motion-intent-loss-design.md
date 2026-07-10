# E30 Window-Conditioned Motion Intent Loss Design

## Context

The current best offline candidate is E22b: E16 eye2 weighted-intent ACT plus a learned phase gate and inactive action scale 0.5. It is the best original-image tradeoff so far, but it is still a post-hoc gate. It does not make the ACT action head itself learn when to move or stop, and it does not address visual-domain robustness.

The next training experiment should target the deadzone intent contract directly:

- When the expert intends motion, the policy should cross the same directional runtime deadzone on the same axis.
- When the expert is idle or the sample is in tail / automation ownership, the policy should stay below all runtime deadzones.
- When the expert moves on one axis / direction, policy crossings on other axes or the opposite direction are wrong intent.

This is a training-objective problem, not another phase-gate threshold sweep.

## Evidence From Current Repo

Current ACT training batches contain only:

```text
image_data, proprio_data, action_data, is_pad
```

`EpisodicDataset.__getitem__` samples a random `t0`, pads actions, and returns this four-tuple. `ACTTrainer._forward` unpacks the same four-tuple and calls `ACTAdapter.forward_loss`. Therefore the adapter currently cannot see episode id, source timestep, `tail_idle_mask`, `action_loss_mask`, or any phase/window labels.

Current `deadzone_loss` supports only `same_dir_window: all` or `same_dir_window: expert_transition_window`. E14a/E17/E18a showed that this family is not enough: it can improve MAE or quietness while suppressing startup should-move behavior.

The data pipeline already has related handoff labels:

- `handoff/gohome_eligible_label`
- `handoff/gohome_loss_mask`
- `handoff/tail_idle_mask`
- `handoff/owner_automation`
- `handoff/action_loss_mask`

These labels are useful, but they are not currently part of ACT training batches.

## Proposed Experiment

E30 introduces a window-conditioned motion intent loss. The first implementation target is eye2; the four-view variant should run only after eye2 proves the objective is mechanically sound.

Planned variants:

- `E30-eye2`: video4/video5, E16-style weighted intent head, plus window-conditioned deadzone intent loss.
- `E30-four`: video4/video5/video6/video7, same objective, only after E30-eye2 smoke and gate results.

E30 should compare against:

- E16: base eye2 weighted-intent ACT.
- E22b: current best phase-gated eye2 candidate.
- E28/E29b: four-view weighted-intent and four-view phase-gated baselines.

Use the same train/val episode split as E16/E28 for comparability:

```text
train: 97, 78, 100, 85, 86, 75, 102, 80, 104, 76, 99, 83, 73, 93, 87, 98, 90, 82, 79
val:   94, 91, 84, 74, 92
```

Do not reuse the old split file directly. Existing E16/E28 split files record the original train-ready dataset path, and `load_data` rejects them when the dataset directory changes. E30 needs a new split file with the same episode IDs and the handoff eligibility dataset path.

## Data Contract

Add optional per-action-step masks to ACT training samples. The masks align with the padded action chunk after `t0` and use `is_pad` for invalid suffixes.

Required masks:

- `deadzone_move_mask`: shape `(T, 4, 2)`. True where the expert action crosses the runtime deadzone for axis/direction. Direction order is `[pos, neg]`.
- `deadzone_stop_mask`: shape `(T,)` or `(T, 1)`. True where policy should stay below all runtime deadzones. This includes expert-idle steps and should also include tail idle / automation ownership when those labels exist.
- `deadzone_wrong_mask`: shape `(T, 4, 2)`. True where a policy crossing would be wrong. In expert-effective steps this is every non-expert axis/direction. In stop steps this may be equivalent to all directions.

The masks must be optional. Existing configs without the new training flag should keep the current four-tuple batch contract and old loss behavior.

E30 should use the derived handoff eligibility dataset as its training source:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/handoff_eligibility_20hz_dwell10
```

The main train-ready 20Hz dataset has `metadata.excluded_go_home=True` and does not contain `handoff/*` masks. The handoff eligibility dataset preserves the same train-ready episode set, includes the post-request automation tail, and provides the masks required to separate human action imitation from stop / handoff intent. Its aggregate summary reports 24 episodes, 24,747 output steps, 519 tail-idle frames, and 16,025 action-loss-valid frames.

Sampling must avoid automation-tail dominance. The current E16/E28 ACT configs use `chunk_size: 20`; if a random `t0` lands deep inside the automation-owned tail, the whole chunk may contain no human-action imitation targets. E30 should either require each sampled chunk to contain at least one `action_loss_mask == true` step, or use explicit sampling weights that keep human-action chunks dominant while still exposing tail / stop windows. Do not let the 8,203 automation-owned frames become ordinary uniformly sampled action targets.

Action normalization must also be mask-aware. Over the 24 train-ready episodes, the original train-ready dataset action mean is about `[-0.0092, 0.0197, -0.0017, -0.0404]`. The full handoff dataset shifts this to about `[-0.0149, -0.0355, -0.0012, -0.1141]` because it includes automation-owned tail actions. Restricting action stats to `action_loss_mask == true` gives about `[-0.0093, 0.0203, -0.0018, -0.0417]`, which matches the original human-action distribution. Therefore E30 should compute `action_mean` / `action_std` from `action_loss_mask == true` frames while still allowing stop-window deadzone losses to see tail / automation observations.

## Label Derivation

Use runtime-scaled deadzone thresholds from:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/deadzone_policy_raw_for_runtime_scale.json
```

For each action step:

- `expert_pos[axis] = action[axis] >= pos_threshold[axis]`
- `expert_neg[axis] = action[axis] <= -neg_threshold[axis]`
- `expert_effective = any(expert_pos or expert_neg)`

Base masks:

- `move_mask = expert_pos/expert_neg`
- `stop_mask = not expert_effective`
- `wrong_mask = all axis-directions except active expert directions when expert_effective`

If handoff labels are available:

- Force `stop_mask = true` for `tail_idle_mask` and `owner_automation`.
- Clear `move_mask` for `tail_idle_mask`, `owner_automation`, and `action_loss_mask == false`.
- Treat request/automation-owned regions as stop-intent for action training, not as ordinary action imitation.
- Apply `action_loss_mask` to the ordinary ACT L1 action imitation loss. Otherwise E30 would train the continuous action head on gohome automation commands, which is explicitly out of scope.

## Loss Contract

The E30 deadzone loss has three terms:

1. `window_deadzone_same_dir_loss`
   Promotes policy action above threshold plus optional margin for active `move_mask` axis/directions.

2. `window_deadzone_stop_loss`
   Penalizes policy crossings above threshold in `stop_mask` steps.

3. `window_deadzone_wrong_loss`
   Penalizes policy crossings in active `wrong_mask` directions.

Acceptance must not be based on the weighted sum alone. Log all three unweighted terms, their weighted total, and counts for active move/stop/wrong labels.

## Code Shape

Keep responsibilities narrow:

- `testbed.data.dataset.EpisodicDataset`
  Optionally reads and returns aligned mask chunks when config enables E30 labels. Its valid-start logic should account for `action_loss_mask` so that pure automation chunks are not sampled as normal imitation examples.

- New focused helper, likely `testbed.data.deadzone_intent_labels`
  Builds action-aligned masks from action arrays, deadzone thresholds, and optional handoff datasets.

- `testbed.data.dataset.get_norm_stats`
  Needs an optional action-stat mask for E30 so automation tail actions do not shift action normalization.

- `testbed.policies.act.trainer.ACTTrainer`
  Supports both four-tuple legacy batches and extended batches with masks.

- `testbed.policies.act.adapter.ACTAdapter`
  Adds a new optional window-conditioned loss path and accepts an optional action imitation mask. Existing `deadzone_loss` behavior remains available for old experiments.

Avoid embedding episode-level phase heuristics inside `ACTAdapter`; it should consume masks, not infer cycle phase from a local chunk.

## Tests

Minimum tests before launching E30:

- Dataset returns legacy four-tuples by default.
- Dataset returns extended masks only when the E30 mask flag is enabled.
- Mask alignment test: sampled `t0`, real action chunk, padding, and mask suffix agree.
- L1 action imitation ignores `action_loss_mask == false` frames while still allowing stop-intent deadzone suppression on tail / automation frames.
- Action normalization test: handoff full-action stats differ from original, while `action_loss_mask` stats match the original human-action distribution.
- ACT loss promotes should-move crossings and does not fire same-dir promotion in stop windows.
- ACT loss suppresses stop-window crossings.
- ACT loss penalizes wrong-axis / wrong-direction crossings during expert-effective steps.
- A one-epoch smoke train emits the new loss terms and produces `policy_latest.ckpt`.

## Evaluation

Use the existing full train-ready offline replay and runtime-scaled gates:

- Replay `collection_summary.json`: MAE, RMSE, per-axis metrics.
- `startup_first_expert_effective_40_*`: startup effective, same-dir, extra/wrong.
- `deadzone_window_*`: start40, longest expert-effective segment, full window.
- `tail_stability_*`: tail effective frames and tail max action.

E30-eye2 should be rejected unless it improves or preserves the E22b tradeoff:

- Startup same-dir and effective motion should not regress materially from E22b.
- Main-motion same-dir should remain close to E22b.
- Main extra/wrong should stay low.
- Tail effective frames should stay at or near zero.
- MAE/RMSE should remain in the E16/E22b band; MAE alone cannot pass the model.

## Non-Goals

- Do not train on gohome automation trajectory as ordinary policy action.
- Do not promote stick deadzone motion from this batch unless expert data actually crosses runtime-scaled stick thresholds.
- Do not treat four-view success as proof of visual-domain generalization.
- Do not deploy from E30 without visual-domain and offline gate evidence.

## Immediate Next Steps

1. Add focused tests for mask generation, `action_loss_mask` handling, and extended batch handling.
2. Implement the optional extended batch path against the handoff eligibility dataset.
3. Create an E30 split file that preserves the E16/E28 train/val IDs but points at the handoff eligibility dataset.
4. Run a one-epoch E30-eye2 smoke.
5. Launch full E30-eye2 only if smoke emits sane mask counts and loss terms.
6. Evaluate E30-eye2 against E16/E22b with the standard replay and runtime-scaled gates.
