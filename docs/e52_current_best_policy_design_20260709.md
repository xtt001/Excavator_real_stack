# E52/E51 current best policy design notes

Date: 2026-07-09

Status: design candidate, offline-evaluated. This is not yet a field-ready runtime
policy until the runtime gate wiring and dry-run logging checks are completed.

## 1. Conclusion

The current best candidate is the **E52 packaged E51 causal temporal gate stack**.
It is not a new pure ACT checkpoint. It is a conservative composition around the
best available ACT action policy:

1. an eye-only ACT action policy trained with weighted intent supervision;
2. a learned phase gate for should-move / should-stop behavior;
3. an intent-targeted near-deadzone snap rule;
4. a learned causal temporal direction gate;
5. a learned two-stage gohome request gate;
6. a proposed, not-yet-implemented timeout fallback for the specific case where
   gohome should have happened but the gate did not trigger.

The design target is:

- **Action closeness**: keep predicted actions close to human demonstrations.
- **Action liveness**: when the task state really requires motion, the policy
  should output an action that can cross the real machine deadzone in the correct
  direction, eventually rather than necessarily immediately.
- **Stop behavior**: when the cycle has reached the end / handoff region, the
  policy should stop producing effective joystick motion.
- **Gohome conservatism**: gohome request may be late, but must not be early in a
  way that creates collision risk.

The current evidence supports using E52/E51 as the best candidate so far, but it
does not prove field success. The remaining critical step is runtime integration
and a no-motion shadow dry run that records every intermediate gate output.

## 2. Why MAE alone is not enough

For this excavator policy, a low action MAE can still fail on the machine.

The joystick/control stack has a deadzone. If the model predicts the correct
direction but with amplitude below the deadzone, the real machine sees the
command as zero. This creates a deadlock failure mode: the model appears close to
the demonstration numerically, but the machine does not start moving.

The real evaluation question is therefore not only:

> Is the action close to the human action?

It is also:

> When the human intends motion, does the model output a same-direction action
> that is strong enough to move the machine?

This is why the current best design uses extra offline metrics:

- same-direction effective action recall;
- eventual liveness within a short horizon;
- missed expert-motion ratio;
- idle false-active ratio;
- tail/end effective-action count;
- gohome request recall and pre-tail false positives.

## 3. High-level data flow

```text
GMSL eye cameras + qpos/qvel
        |
        v
ACT action policy
        |
        +--> raw predicted action
        +--> intent probabilities
        |
        v
phase gate
        |
        v
intent-targeted deadzone snap
        |
        v
causal temporal direction gate
        |
        v
runtime safety/command path
```

Gohome is a separate request signal:

```text
GMSL eye cameras + qpos/qvel + policy/intent features
        |
        v
tail candidate gate
        |
        v
gohome eligibility gate
        |
        v
go_home_requested=True
        |
        v
GoHomeController takes over
```

The action policy should not imitate the automatic gohome trajectory. It should
learn the human-operated part of the cycle and emit a request when the cycle has
reached the eligible handoff state.

## 4. Base action policy

The action policy used by the E52 candidate package is:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_weighted_intent_head_eye2/policy_best.ckpt
```

This is an eye-only policy variant rather than a four-view policy. The main
reason is not that two cameras are intrinsically better, but that this specific
run gave the best current balance between action quality and executable motion
metrics in the available offline tests.

The role of the ACT policy is still core imitation:

- read images and low-dimensional robot state;
- predict the human joystick action;
- expose intent-like probabilities that later gates can use.

The gates do not replace ACT. They constrain ACT's output where the observed
failure modes are known: weak near-deadzone actions, wrong-direction or
over-duration actions, and unsafe/late gohome handoff.

## 5. Phase gate

Selected operating point:

```text
simple_0.15_s0.50
```

Interpretation:

- the gate estimates whether the current frame should be treated as active
  motion or stop/near-stop;
- threshold: `0.15`;
- inactive action scale: `0.50`.

The important point is that this is not a hand-written terrain-texture rule. It
is a lightweight learned gate trained from existing demonstrations. The label is
derived from whether expert action crosses the deadzone / effective-motion
criterion in the data.

The phase gate addresses two problems:

1. It suppresses weak motion where the human is effectively not operating.
2. It keeps an explicit should-move signal available so later logic can avoid
   treating all small actions as equally unimportant.

This matches the current understanding of "deadzone aware": the policy should
learn both when to move and when not to move. It should not only suppress idle
motion.

## 6. Intent-targeted near-deadzone snap

Selected settings:

```text
snap_margin: 0.02
snap_intent_threshold: 0.70
```

This is the most deterministic part of the stack. Its purpose is narrow:

- if the policy already indicates a clear action intent;
- and the predicted action is close to but below the deadzone;
- then snap that axis/direction across the deadzone by a small margin.

This is different from globally increasing all action amplitudes. A global boost
would make idle or uncertain states more dangerous. The snap only applies near
the deadzone and only when the intent signal is strong enough.

What it helps:

- avoids the "correct direction but no machine motion" failure;
- improves startup/liveness behavior;
- keeps action changes bounded.

What it does not solve:

- wrong intent;
- wrong visual interpretation;
- unsafe timing;
- field distribution shift.

## 7. Causal temporal direction gate

Selected operating point:

```text
tdir_t50_s75
direction_threshold: 0.50
direction_inactive_scale: 0.75
```

This gate is learned. It uses only current and past context, not future frames.
The evaluated temporal offsets are causal, for example:

```text
[-10, -5, -2, -1, 0]
```

The purpose is to reduce frame-level jitter and wrong/weak direction decisions
without adding non-causal information that would be unavailable at runtime.

This matters because the ACT policy may output a plausible but weak or
wrong-duration motion on individual frames. The temporal direction gate asks:

> Given the recent trajectory of state and intent, is this direction still
> consistent with the task phase?

It is intentionally not a full planner. It is a small learned filter around the
action policy.

## 8. Gohome request gate

Selected operating point:

```text
learned_tail_t0.97_tc10_e0.80_ec3
```

Interpretation:

- tail candidate threshold: `0.97`;
- tail consecutive count: `10`;
- gohome eligibility threshold: `0.80`;
- eligibility consecutive count: `3`.

The gohome decision is split into two learned stages:

1. **Tail candidate**: are we near the end of a cycle where gohome could be
   considered?
2. **Eligibility**: is it actually acceptable to request gohome now?

This split matches the task semantics discussed during analysis:

- gohome must happen after a cycle attempt;
- the bucket should be near the dig/home-side region;
- the bucket should be effectively empty / safe enough;
- early request is safety-critical and should be avoided;
- late request is undesirable but less dangerous if the action stack has stopped
  moving.

The policy does not need to output the automatic gohome trajectory. It should
emit one request. After the request, `GoHomeController` owns the automatic return
motion.

## 9. What happens if gohome is missed

Current runtime behavior, based on the inspected `record_real.py` path:

- if `go_home_requested` becomes true, the runtime can start `GoHomeController`;
- if it remains false, the normal policy/control mode continues;
- the E52 action-side tail gate is designed to output no effective joystick
  action at the end, so the likely failure is stationary waiting rather than a
  large unsafe motion.

This is why a simple fallback is attractive:

> If the system has been near home / handoff pose, qpos is almost stationary, and
> no meaningful policy motion has occurred for 10 seconds, emit one timeout
> gohome request.

This fallback is recorded as a design requirement but is not implemented in this
document. It should be added only with explicit logging and a one-request or
one-to-two-request limit per cycle.

## 10. Learned parts vs calibrated constants

The current design is not just a pile of hard-coded texture constants.

Learned components:

- ACT action policy;
- phase gate model;
- tail candidate model;
- gohome eligibility model;
- causal temporal direction gate model.

Calibrated deterministic components:

- selected thresholds and consecutive-count requirements;
- near-deadzone snap margin;
- snap intent threshold;
- inactive scaling factors;
- proposed 10-second timeout fallback condition.

The constants are operating points selected from offline scans and replay
metrics. They are still empirical. They should be treated as calibration values,
not as universal truths. If the hardware deadzone, joystick mapping, camera
setup, or task semantics change, these operating points must be revalidated.

## 11. Why this may help generalization

The original concern was that the model might overfit sand texture details. The
current E52 design does not directly solve all visual-domain generalization. It
helps indirectly by separating responsibilities:

- ACT handles visual imitation.
- Phase and direction gates enforce task-motion consistency.
- Gohome gates enforce conservative handoff semantics.
- Snap handles machine deadzone mechanics without asking the visual model to
  learn the exact output amplitude by itself.

This reduces the chance that a texture change alone causes an obviously wrong
machine command, because the final command also has to pass low-dimensional and
temporal consistency checks.

However, it is not a complete domain-general solution:

- if ACT's visual interpretation is wrong, the gates may only partially correct
  it;
- if a new sand texture changes both appearance and qpos/visual timing, the
  learned gates can still shift;
- the new 105-109 data is useful but too small to prove broad generalization.

The correct interpretation is:

> E52/E51 is the best current executable-policy candidate, and it has better
> safeguards against known failure modes than raw ACT. It is not proof that
> texture generalization is solved.

## 12. Offline evidence

### 12.1 Candidate package integrity

Package:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/policy_packages/e52_e51_causal_temporal_gate_candidate/candidate_package_manifest.json
```

Verification:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/policy_packages/e52_e51_causal_temporal_gate_candidate/candidate_package_verify.json
```

Observed verification result:

```text
ok: true
declared_artifacts: 23
checked_artifacts: 23
errors: []
```

### 12.2 Original train-ready set replay

Summary artifact:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e51_full_act_causal_temporal_gate_smoke_all_train_ready/full_act_temporal_gate_smoke_summary.json
```

Key metrics:

| Metric | Value |
| --- | ---: |
| Episodes | 24 |
| Temporal-direction action MAE | 0.04083 |
| Temporal-direction action RMSE | 0.08645 |
| Gohome event recall | 23 / 24 = 95.8% |
| Gohome pre-tail false-positive episodes | 0 |
| Gohome pre-tail active frames | 0 |
| ACT p95 runtime | 20.29 ms |
| Gate p95 runtime | 0.0075 ms |

Important interpretation:

- action closeness is acceptable relative to the current baseline family;
- gohome is mostly recalled;
- the key safety sign is zero pre-tail gohome false positives in this replay;
- gate compute cost is negligible compared with ACT inference.

### 12.3 New 105-109 liveness check

Summary artifact:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/runs/offline_eval/e52_e51_full_act_temporal_gate_new105_109_liveness_test/liveness_metrics/new105_109_liveness_summary.json
```

Key metrics:

| Metric | Value |
| --- | ---: |
| Episodes | 5 |
| Eventual same-direction liveness | 5 / 5 |
| Same-direction within 5 steps | 5 / 5 |
| Max same-direction delay at 20 Hz | 0.20 s |
| Mean same-direction frame recall | 88.7% |
| Mean missed expert-motion ratio | 10.2% |
| Mean idle false-active ratio | 18.9% |
| Temporal-direction action MAE | 0.04911 |
| Gohome event recall | 4 / 5 = 80.0% |
| Gohome pre-tail false-positive episodes | 0 |

Important interpretation:

- the most important startup/deadlock concern did not appear in these five new
  episodes: each episode eventually produced same-direction effective motion;
- gohome missed one of five episodes, so the timeout fallback remains valuable;
- pre-tail gohome false positives were still zero in this small new set.

### 12.4 Semantic 3/4-cycle crop experiment

Summary artifact:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/runs/offline_eval/main_segment_semantic_crop_e59_vs_e16_executable_metrics/summary.json
```

The semantic crop experiment was useful, but it is not the current best
executable policy candidate.

Observed pattern:

- MAE improved versus the earlier baseline on selected comparisons;
- idle false-active ratio improved strongly;
- same-direction/missed-motion metrics did not clearly improve and regressed on
  the stress subset.

Interpretation:

> Semantic cropping is promising for reducing unnecessary motion, but the current
> evidence does not justify replacing the E52/E51 action-liveness candidate.

## 13. Runtime integration status

The current repository already has the right high-level runtime surfaces:

- `PolicyActionSource` wraps ACT inference and deadzone assist behavior;
- `record_real.py` can receive `go_home_requested` from policy extras;
- if accepted, `record_real.py` starts `GoHomeController`;
- diagnostics already have fields for policy/gate stack ids and gohome-related
  values.

But the E52 package is still an offline candidate. Before field use, the runtime
needs explicit wiring for:

- phase-gated action;
- snapped action;
- temporal-direction-gated action;
- final returned policy action;
- gohome request output;
- gate probabilities and thresholds;
- timeout fallback state, if implemented.

Required logged fields for the first integration test:

- raw `policy_action`;
- `phase_gate_prob` and phase-gated action;
- snap active flags and snapped action;
- temporal direction gate probabilities and final gated action;
- `policy_returned_action`;
- runtime `safe_action`;
- final commanded action;
- `go_home_requested`;
- gohome request acceptance/rejection reason;
- timeout fallback timer and trigger state.

The first runtime test should be a no-motion dry run:

```text
output_mode = shadow_zero
```

Expected behavior in that dry run:

- policy and gate outputs are computed and logged;
- final commanded machine action remains zero;
- gohome request decisions are visible in logs but should not move the machine
  unless the dry-run mode explicitly allows that path.

## 14. Main failure modes

### 14.1 Early gohome request

Severity: high.

Reason: early gohome can create collision or unsafe automatic motion.

Mitigation in current design:

- learned tail candidate gate;
- learned eligibility gate;
- high tail threshold and consecutive-count requirement;
- pre-tail false-positive checks in replay.

Remaining risk:

- offline data may not cover every unsafe near-tail state.

### 14.2 Missed gohome request

Severity: medium if action-side stop works; high if the policy continues moving.

Current observation:

- E52/E51 missed one new 105-109 gohome event;
- tail action-side behavior is designed to stay below effective motion;
- likely failure is stationary waiting, not aggressive motion.

Planned mitigation:

- near-home + stationary-qpos + no effective policy motion for 10 seconds emits a
  one-time timeout gohome request.

### 14.3 Correct direction but too weak to move

Severity: high for task completion.

Mitigation:

- intent-targeted near-deadzone snap;
- liveness tests on old and new data;
- same-direction effective action metrics.

Remaining risk:

- if model confidence is low in a new visual domain, snap may not activate.

### 14.4 Wrong or over-duration action

Severity: medium to high depending on state.

Mitigation:

- phase gate;
- causal temporal direction gate;
- idle false-active metrics;
- tail effective-action checks.

Remaining risk:

- wrong visual understanding can still produce plausible but wrong intent.

### 14.5 Texture/domain shift

Severity: unknown.

Mitigation:

- use low-dimensional state and causal temporal consistency in gates;
- evaluate on newly recorded 105-109 episodes;
- keep semantic crop and visual-domain clustering as follow-up experiments.

Remaining risk:

- current new-data sample size is too small for strong generalization claims.

## 15. Acceptance gates before field trial

Before running this as a real moving policy, the following should pass:

1. Candidate package verification is clean.
2. Offline replay on selected old and new episodes shows no pre-tail gohome false
   positives.
3. Offline liveness shows eventual same-direction effective motion on
   should-move episodes.
4. Tail/end segments show no effective joystick motion after handoff-eligible
   stop.
5. `shadow_zero` runtime dry run logs every intermediate action/gate value.
6. Runtime dry run confirms final commanded action is zero while policy outputs
   are still visible.
7. Gohome request logging shows whether requests are accepted, rejected, missed,
   or timeout-triggered.
8. A human reviewer inspects a small set of video/log overlays around startup,
   tail, and gohome request frames.

Only after these checks should the stack be promoted from offline candidate to a
controlled field trial.

## 16. Artifact index

Candidate package:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/policy_packages/e52_e51_causal_temporal_gate_candidate/
```

Base action checkpoint:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_weighted_intent_head_eye2/policy_best.ckpt
```

Base action config:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_weighted_intent_head_eye2/resolved_config.yaml
```

Phase gate:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e22b_phase_gate_soft_scale_probe/phase_gate_model.pt
```

Tail candidate gate:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e33_two_stage_gohome_gate_probe/tail_candidate_model.pt
```

Gohome eligibility gate:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e32_gohome_eligibility_probe/gohome_eligibility_model.pt
```

Temporal direction gate:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e49_causal_temporal_direction_gate_probe_s75/temporal_direction_gate_model.pt
```

Original train-ready replay summary:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e51_full_act_causal_temporal_gate_smoke_all_train_ready/full_act_temporal_gate_smoke_summary.json
```

New 105-109 liveness summary:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/runs/offline_eval/e52_e51_full_act_temporal_gate_new105_109_liveness_test/liveness_metrics/new105_109_liveness_summary.json
```

Semantic crop comparison:

```text
/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/runs/offline_eval/main_segment_semantic_crop_e59_vs_e16_executable_metrics/summary.json
```

Experiment ledger:

```text
docs/superpowers/plans/2026-07-08-policy-gate-experiments.md
```

## 17. Final recommendation

Submit E52/E51 as the current best **design candidate**, with the following
wording:

> The current best approach is a gated ACT policy: use the best eye-only ACT
> action checkpoint for imitation, then add learned phase and causal temporal
> direction gates for executable motion consistency, an intent-targeted
> deadzone snap to avoid no-motion deadlock, and a conservative two-stage learned
> gohome request gate. Add a 10-second near-home stationary timeout as a fallback
> for missed gohome requests. This is the best offline-supported design so far,
> but it still requires runtime gate wiring and shadow-zero validation before
> real machine motion.
