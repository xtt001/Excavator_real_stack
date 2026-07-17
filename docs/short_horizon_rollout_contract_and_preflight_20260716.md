# Short-horizon rollout contract and offline preflight

Date: 2026-07-16

## Outcome

The first short-horizon closed-loop slice is implemented without changing the
runtime action path or sending a machine command.

Focused owners:

- `testbed/testbed/policies/short_horizon_rollout.py` owns rollout authority,
  per-tick trace semantics, causal command-to-observation linkage, abort
  boundaries, and capability labels;
- `testbed/testbed/cli/audit_short_horizon_rollout.py` is a passive JSON trace
  auditor;
- the existing `ExecutionMonitor` remains the downstream sent-command/qvel
  response owner and is not used to synthesize feedback;
- `record_real.py` is unchanged in this slice.

The auditor never sends, clips, retries, or rewrites a command.

## Contract

A trace declares one state world:

- `teacher_forced`;
- `state_hold`;
- `learned_dynamics`;
- `hybrid_lowdim`;
- `simulator`;
- `live_policy_on`.

`teacher_forced` and `state_hold` require `observe_only` authority.
`live_policy_on` requires `bounded_control` and has no implicit safety values.
It must explicitly provide:

- per-axis absolute command and per-tick delta limits;
- per-axis qvel abort limits;
- per-axis qpos lower and upper limits;
- allowed positive/negative directions;
- deadman and controller-ack policy;
- observation-gap and camera-age limits;
- camera order;
- checkpoint and resolved-config SHA-256 identities.

The maximum accepted horizon is two seconds. The exact tick count and sampling
rate are part of each immutable contract.

Each tick records:

- four camera timestamps and frame identities;
- qpos/qvel and observation timestamp;
- raw policy proposal, returned action, guarded action, and actual commanded
  action;
- command ID, send timestamp, controller acknowledgement, and safety state;
- the command ID that generated the next observation.

## Hard evidence rule

An observation after tick zero counts as self-generated only when all of these
hold:

1. the previous command was actually sent;
2. the controller acknowledged it;
3. the next observation timestamp is later than the send timestamp;
4. the next observation explicitly names the previous command ID as its causal
   parent;
5. the observation source is a declared generated-state world, not
   `teacher_forced` or `state_hold`.

This yields distinct evidence labels:

- `none_noncausal_observation_source`;
- `incomplete_causal_trace`;
- `zero_command_state_stream_only`;
- `synthetic_state_progression_proxy`;
- `direct_in_declared_simulator`;
- `direct_physical_short_horizon`.

None of these labels by itself proves task success, safety, or terrain
generalization.

## N5 teacher-forced negative control

The first formal preflight uses existing G49 N5 saved policy outputs and real
four-camera/qpos/qvel observations from validation episode 10120, steps 0--19.
It deliberately sends no command and labels the observation source
`teacher_forced`.

Artifact root:

`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/short_horizon_rollout_v1_n5_teacher_forced_negative_control`

Report SHA-256:

`52ded8fe2fd4a15eca1893a7b4a6db3f7c140606f9c981d9aa2bc2466195de9a`

Result:

- trace integrity: valid;
- contract compliance: true;
- 20 observations over 0.957 seconds;
- N5 proposal nonzero on 20/20 ticks;
- proposal motif: joint `boom- / stick+ / bucket+`;
- actual commands sent: 0;
- causal command-linked state transitions: 0/19;
- self-generated state evidence:
  `none_noncausal_observation_source`;
- physical response, task progress, and task success: not estimable.

This is the required negative result. It proves that the new auditor does not
promote four-camera teacher-forced replay into short closed-loop evidence merely
because the model emits a plausible action sequence and qpos/qvel change in the
recorded observations.

The preflight also exposed and fixed a real trace-schema bug: nanosecond
timestamps around `1.78e18` were initially checked through floating point and
lost integer precision. Timestamp parsing is now integer-only and has a
regression test beyond the IEEE-754 exact-integer range.

## Not yet authorized or implemented

- no `record_real.py` live rollout arming path;
- no policy command sent from this worktree;
- no automatic retry;
- no learned or hybrid state synthesis;
- no simulator claim;
- no task-progress label;
- no field safety limits chosen from guesses.

Before a real short rollout, the field side must supply reviewed per-axis
command, delta, qpos, qvel, and direction limits from the current machine and
test pose. The current generic action clip is not a substitute for this
experiment-specific authorization. The live slice must also enable all-axis
response monitoring explicitly; the historical `ExecutionMonitor` default that
omits stick belongs to an older task contract.
