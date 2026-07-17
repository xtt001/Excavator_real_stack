# [Execution target G28/H1: Goal/effect ACT complete offline results]

Date: 2026-07-12
Decision: **reject G28 for promotion; keep raw ACT + mechanical assist**

## [Execution target G28/H1: End-to-end result]

The candidate trained successfully and the new auxiliary losses learned on the
training fold, but the closed-loop validation gate failed:

| Pipeline on validation 48 anchors | Recovered | Deadlocked | Hidden by teacher forcing | Unexpected anchors |
| --- | ---: | ---: | ---: | ---: |
| G28 raw, no assist | 40 | 8 | 2 | 3 |
| G28 raw + mechanical assist | **45** | **3** | **2** | **7** |
| H2 best raw + assist reference | **45** | **3** | **1** | not traced in old artifact |

The strict matched comparison is decisive: G28 assist has zero induced or
recovered anchors relative to H2, but it adds one hidden deadlock. It therefore
does not satisfy the zero-hidden contract and is not allowed onto 105..109.

## [Execution target G28/H1: Complete offline artifact]

Report root:

`/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g28_goal_effect_act_formal`

The single report is:

`complete_offline_report/complete_offline_report.json`

It joins:

- full open-loop replay over all 24 formal episodes (16,529 steps);
- raw/no-assist and mechanical-assist state-hold recursive/full-horizon traces;
- startup, deadlock, hidden, unexpected/extra, opposite, and flip counts;
- fixed-qpos/multi-FPV replay (`episode_74` qpos + `episode_91` images and the reciprocal `episode_92`/`episode_74` pair);
- release and 80-step tail safety census;
- gohome estimability check;
- formal execution-monitor response replay;
- split, deadzone, and source-artifact hashes.

Held-out `105..109` is explicitly recorded as forbidden and unevaluated.

## [Execution target G28/H1: Open-loop and safety callback]

Open-loop G28 all-24 metrics:

- overall MAE `0.04843` (train-fold weighted `0.04705`, validation `0.05410`);
- policy p95 absolute action `0.8347`, policy max `1.0025`;
- 15 policy clip violations (`|action| > 1`) and no non-finite values;
- full-window extra/wrong effective frames `15.61%` on average;
- start40 policy effective `86.77%` while expert effective is `5.83%`, with
  `80.94%` extra/wrong frames and 23/24 episodes above the 20% extra threshold;
- tail 80-step extra effective frames: `60` across 24 episodes;
- release-window extra effective frames: `338` across 167 release windows;
- maximum simultaneous effective axes: 2.

This is a safety/phase regression, not a liveness win. The future-effect
objective made the action stream more active before the demonstrated command
and is rejected even independently of the hidden-deadlock failure.

Fixed-qpos/multi-FPV replay produced MAE `0.0918` for episode 74 with episode
91 images and `0.0889` for episode 92 with episode 74 images. These are
diagnostic only because the validation contract already failed.

## [Execution target G28/H1: Execution-monitor and gohome boundary]

The formal causal sidecar replay remains internally consistent:

- train: 180 events, 176 responded, 4 stalled candidates, 0 unknown,
  0 mismatches;
- validation: 48 events, 48 responded, 0 unknown;
- all selected: 228 events, 224 responded, 4 stalled candidates;
- retry precision is **not estimable** because existing sidecars are teleop
  heuristics without policy-on intent, operator correction, or confirmed
  failed-actuation labels.

The 24 formal HDF5 episodes have `excluded_go_home=true` and no complete
`handoff/gohome_eligible_label`/`handoff/tail_idle_mask`. Gohome false-positive
recall is therefore reported as `not_estimable`, not as a pass.

## [Execution target G28/H1: Verification]

- `423 passed, 16 warnings, 4 subtests passed` with explicit target-root
  `PYTHONPATH`.
- Targeted Ruff checks for every changed/new owner passed.
- `git diff --check` passed.
- Smoke training completed for one epoch before the fixed 200-epoch run.

## [Execution target G28/H1: Root-cause reflection]

The new branch is a valid structural change, but the existing demonstrations
do not supply enough independent causal support to turn the auxiliary future
forecast into a trustworthy confidence signal. The effect head is trained on
expert-trajectory outcomes, while the failing field situation is a policy
proposal that was not executed. The open-loop start40 and tail regressions show
that future-motion supervision can amplify a plausible trajectory without
proving that the command is safe or executable at that instant.

Therefore the data can support offline target construction and representation
learning, but it cannot support promotion of an action-confidence/retry policy
without on-policy sent-command/feedback/correction labels. The retained
reference is raw continuous ACT + mechanical assist, with the independent
causal execution monitor kept diagnostic-only until such data exists.
