# SimVerify G5 Core Two-Cycle Recorded-Path Contract

Status: `method_frozen_before_g5_core_run`

Evidence scope: `recorded-observation/offline teacher-forced development`

Closed-loop execution: `false`

Independent validation: `false`

Held-out test read: `false`

This bounded G5 core experiment asks whether the conditioned ACT remains
stateful and semantically coherent across one shared ready boundary. It does
not generate environmental response and does not replace the later camera,
state-hold, delay/latest-wins, or runtime-equivalence operands required by the
complete G5/G6 route.

## Why recorded-path replay is the main operand

The per-step delta-stitch verifier is useful for local action-supported path
diagnostics. A stricter two-cycle audit showed that forcing a different source
episode at every tick reached two real observable ready boundaries in only
14/58 train and 2/21 validation expert paths. Allowing coherent same-donor
continuation improved this to 42/58 and 19/21, but one median-action null also
produced a false two-boundary completion.

Therefore transition stitching is not a valid G5 environment. The frozen
research plan's E07 recorded-path replay is used instead. Its claim is narrower:
given a real recorded observation sequence, does the policy preserve internal
state and update condition correctly across the boundary?

## Inputs and split lock

- M2 `two_cycle_anchors_v1.jsonl`;
- accepted train pairs for expert threshold generation;
- accepted validation pairs for development replay;
- completed B1.4 and B2.4 sim-domain bundles;
- M2 train-generated event templates and effective deadzone;
- held-out episodes `1`, `13`, `25`, and `33` locked unread.

The original HDF5 is read-only. Validation has already been used for prior
method development, so this result is not independent confirmation.

## Continuous replay

For each adjacent pair:

```text
ready_i -> cycle_i -> ready_i+1 -> cycle_i+1 -> ready_i+2
```

The policy is reset exactly once at `ready_i`. Every recorded observation from
the first start through the second end is then delivered in source order at
20 Hz. Temporal aggregation is never reset at the shared boundary.

Condition update is atomic:

- ticks before the shared ready boundary receive `first_condition`;
- the shared ready boundary tick and all later ticks receive
  `second_condition`.

The boundary tick is stored once in the trace. The second-cycle condition
continuity contract must satisfy:

```text
first.next_ready_sector == second.current_sector
```

## Single-factor nulls

Each checkpoint receives two traces from the identical observation history:

- `switched`: condition changes atomically at the shared ready boundary;
- `unchanged`: the first condition is retained through the second cycle.

The only changed factor is the delivered condition. B2.4 is the matched
shuffled-condition training null.

## Preserved outputs

Every trace stores separately:

- raw normalized ACT chunk;
- raw direct ACT chunk;
- temporal-aggregation action;
- future-runtime-safe action;
- expert action;
- delivered condition;
- condition route diagnostics;
- observation tick and boundary index.

## Expert-generated continuity envelopes

Train source episodes generate:

- lower envelope for two-cycle required-event coverage;
- lower envelope for two-cycle event-order validity;
- upper envelope for ready-boundary action discontinuity.

The aggregation unit is adjacent pair, then source episode. Validation expert
paths must remain inside the frozen train envelope before model evidence is
interpreted.

## Model metrics

All validation adjacent pairs contribute:

- first- and second-cycle required-event coverage;
- first- and second-cycle event order;
- ready-boundary future-runtime-safe action discontinuity;
- second-cycle route-2 activation;
- switched-versus-unchanged action effect.

Pairs whose first and second `next_ready_sector` differ additionally contribute:

- route-2 swing-action semantic margin in the requested sector direction;
- B1.4-minus-B2.4 paired semantic margin;
- source-episode aggregation.

Sector order and action-to-qpos sign are fit from accepted train annotations and
train episode actions only.

## G5 core development decision

`g5_core_two_cycle_condition_continuity_established_development` requires:

- validation expert paths pass every train-generated continuity envelope;
- B1.4 switched traces preserve two-cycle required events and order under the
  frozen adjacent-pair-then-source-episode expert envelopes;
- B1.4 ready-boundary discontinuity stays inside the expert envelope for every
  validation source episode;
- B1.4 activates the declared second-cycle next-condition route;
- switched and unchanged B1.4 traces differ on every changed-condition source
  episode;
- B1.4 route-2 semantic margin is positive on every changed-condition source
  episode;
- B1.4 semantic margin exceeds B2.4 on every changed-condition source episode.

Failure returns `g5_core_two_cycle_condition_continuity_not_established`.

Neither result authorizes held-out test, closed-loop claims, real fine-tuning,
shadow, control, or deployment. Passing this core Gate only authorizes the
remaining G5 robustness operands.
