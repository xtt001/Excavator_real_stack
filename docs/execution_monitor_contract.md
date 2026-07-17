# [Execution target G24/H1: causal execution monitor contract]

## [Execution target G24/H1: ownership boundary]

The continuous ACT proposal remains the first action source of truth. Mechanical
deadzone assist remains the deterministic actuator-interface owner. The monitor
is downstream of the actual controller send and owns only response state,
diagnostics, and an explicit retry permission. It has no `action_scale`, no
hard intent argmax, no phase gate, and no zero-action fallback.

## [Execution target G24/H1: inputs]

- `SentCommand`: final `command` after assist/safety, monotonic
  `send_timestamp_ns`, transport acknowledgement, safety-block state, and an
  optional retry token.
- `FeedbackSample`: monotonic observation timestamp, qpos/qvel, and explicit
  reset/gap/safety boundaries.
- `ExecutionMonitorConfig`: direct-domain deadzones, qvel response thresholds,
  response-window length, high-confidence direction threshold, supported axes,
  and a bounded retry count. Stick is unsupported by default because it is a
  structural zero axis for this task.

## [Execution target G24/H1: outputs and safety]

The monitor reports per-axis `inactive`, `pending`, `responded`, `stalled`, or
`unknown` status. `stalled` is emitted only after a complete finite causal
window with an acknowledged command and no same-direction qvel response.
Reset/gap/safety/non-finite/transport interruptions become `unknown`; opposite
qvel is retained as a diagnostic peak and is not a wrong-direction label.

`request_retry(candidate_action, direction_confidence)` never sends or rewrites
the candidate. It returns permission only when every effective candidate axis
is an already-stalled axis with high-confidence same sign, and the caller must
attach the returned token to the actual sent command. Candidate actions that
introduce an effective axis or opposite sign abstain. A retry is bounded by
`max_retries_per_event`.

## [Execution target G24/H1: training/evaluation handoff]

The existing `direct_command_qvel_response_v1` sidecar remains the offline data
owner. It can calibrate response-window summaries, but its teleoperation data
does not justify a learned retry governor: policy intent, operator correction,
and confirmed failed-actuation labels are absent. Future train/validation data
must be session-disjoint and on-policy, then run recursive state-hold and full
window safety before any held-out 105..109 evaluation.

`replay_execution_monitor_trace` is the thin evaluation adapter: it accepts an
interleaved sequence of `SentCommand` and `FeedbackSample` objects and returns
monitor snapshots without synthesizing feedback. A future learned direction
head supplies `direction_confidence` to `request_retry`; it does not replace
the continuous action head.
