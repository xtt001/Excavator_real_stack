# SimVerify G4-v3.1 Delta-Stitch Gate Audit

Status: `audit_contract_frozen_after_v3_development_result`

Evidence scope: `recorded-observation/offline development`

Independent validation: `false`

G4-v3 remains immutable with decision `offline_emulator_invalid_v3`. It missed
one frozen criterion: validation source-episode q97.5 completion steps was
`377.95`, while the train-derived q97.5-of-source-episode-q97.5 limit was
`376.9425`.

This audit does not rewrite that result. It asks whether the failed statistic
is a valid emulator prerequisite.

## Gate defect

The compared source episodes contain different numbers of cycles:

- train source episodes: 4 to 11 cycles;
- validation episode 12: 15 cycles.

With the same underlying maximum of 379 steps, a q97.5 estimate from 15 values
lies closer to the maximum than estimates from smaller train groups. The
statistic therefore changes with group size, not only with emulator behavior.
In nested train leave-one-source-episode audit, the same rule rejects 1 of 15
train episodes. It is not a stable hard support criterion.

Completion speed remains a diagnostic. The hard budget is already frozen from
data as the maximum accepted train-cycle duration, 838 ticks. A candidate that
does not reach five integrated intervals within that budget is incomplete.

## Audited development prerequisite

V3.1 may authorize an offline B1.4 policy-stitch development experiment only
when:

- every train and validation expert cycle completes inside the frozen budget;
- every paired train and validation median-action null remains incomplete;
- paired completion separation is one in every source episode;
- expert retrieval distance remains inside the inherited one-step support
  radius;
- every other frozen v3 criterion passed.

No numeric tolerance is added to the failed q97.5 value.

## Evidence status

Validation was inspected by v3 before this audit was written. V3.1 is therefore
development evidence, not a new independent validation result. It cannot
authorize held-out test, simulation closed loop, real-machine use, or
deployment. Its only possible authorization is the separately versioned B1.4
offline policy-stitch development experiment.
