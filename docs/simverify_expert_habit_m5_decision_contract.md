# SimVerify Expert-Habit M5 Decision Contract

Status: frozen after the user-authorized closed-loop diagnostic design and
before packaging the terminal evidence decision.

## Decision rule

The expert-habit experiment may end at `sim_observable_only` only when all of
the following checksum-bound facts hold:

1. the accepted v11 observable ready definition and v2 ready-cycle dataset
   report zero held-out reads and no privilege input;
2. B0, B1, and B2 are matched completed runs, and the frozen offline Gate is
   preserved without rewriting its failed MAE criteria;
3. a shared-prefix B1 intervention changes only the committed target, exceeds
   same-condition repeat variability on every action and qpos axis, and all
   branches reach their scripted v11 ready endpoint;
4. one continuous `repeat_same` scenario completes two observable cycles;
5. one continuous `move_adjacent` then `stay` scenario completes both
   observable cycles;
6. no input claims held-out generalization, physical excavation validation,
   real transfer, deployment readiness, or real-control permission.

This decision classifies the demonstrated technical capability. Because the
live PACT and Unity checkouts are dirty, read-only external providers, the AGX
evidence remains explicitly non-promotable. `sim_observable_only` therefore
does not authorize held-out access, checkpoint deployment, real finetuning,
shadow execution, or control.

The other terminal flags must be false:

```text
reject=false
revise_annotation=false
revise_condition=false
real_finetune_candidate=false
control_candidate=false
```

The offline Gate remains
`condition_understanding_not_established_offline`; the higher-tier AGX result
does not retroactively change its thresholds or result.
