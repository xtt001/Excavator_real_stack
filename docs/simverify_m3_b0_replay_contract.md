# SimVerify M3 B0 Recorded-Observation Replay Contract

Status: `implementation_frozen_replays_pending`

Evidence scope: `recorded-observation/offline`

This replay is the G3 evidence generator for the unconditioned B0 checkpoint.
It measures what the checkpoint emits along accepted recorded cycle paths. It
does not generate physical state, execute a simulator, or establish closed-loop
task success.

## Inputs and exclusions

- immutable M0 package `sim_observable_cycle_v3`;
- immutable M2 evaluator package
  `sim_observable_cycle_v3_m2_contract_v1`;
- completed B0 bundle `simverify_b0_unconditioned_v1_seed0`;
- train or validation source episodes from the checked-in B0 split;
- four physical camera roles, source-domain qpos/qvel, and recorded expert
  `actuator_speed_cmd`;
- no condition input to B0;
- no privilege input;
- no held-out test episode before `gate_thresholds_v1.json` is frozen;
- no PACT, Unity, AGX, Jetson, real-control, or field-network dependency.

The replay validates both the adjacent training metadata and the contract
embedded inside the checkpoint. The checkpoint must identify baseline B0,
condition input absent, domain `sim`, and both real-control and Jetson use as
forbidden.

## Replay window and policy state

Each accepted cycle is replayed independently. Policy and temporal-aggregation
state are reset at the cycle start. The replay includes both endpoints of
`target_steps_20hz`, so the shared `ready_end` boundary is present for event
extraction. The next cycle is not included in the same trace; two-cycle
state-continuity is a later G5 operand.

At every recorded 20 Hz tick, the policy receives:

- `eye_left -> video4`;
- `eye_right -> video5`;
- `stick_down -> video6`;
- `stick_up -> video7`;
- source-domain qpos and qvel.

The annotation condition is saved only as an evaluation sidecar. It is never
passed into B0.

## Required action stages

Every cycle trace stores independent arrays for:

1. normalized raw ACT chunk;
2. direct source-domain raw ACT chunk;
3. temporal-aggregation action;
4. future runtime-safe action;
5. recorded expert action.

For this offline slice, stage 4 is an independent identity copy of stage 3. It
is not a claim that a deployment safety transform has been implemented or
validated. Array aliasing, non-finite values, dtype drift, or shape mismatch
fails the replay.

## Per-cycle measurements

Task events use the train-generated M2 effective-axis templates and timing
envelopes. Each cycle records:

- required event coverage;
- event order validity and missing-phase rate;
- expert-effective axis recall with matching direction;
- opposite direction count divided by expert-effective axis ticks;
- unexpected effective axis count divided by policy-effective axis ticks;
- action MAE as an auxiliary diagnostic only.

Ready boundaries are observable annotations. Their inclusion in event coverage
does not mean the excavator physically reached ready.

## Repetitions and threshold status

Each run is identified by split and `repeat_id`. Replaying the same checkpoint
under the same FP32 implementation estimates numerical/runtime repeat noise.
Zero observed repeat variance is a valid result; it is not replaced by an
invented tolerance.

G3 remains pending after any single replay. A later calibration step must:

1. aggregate by source episode;
2. compare validation policy distributions to the frozen expert envelopes;
3. incorporate repeated B0 replay noise;
4. generate bootstrap uncertainty without reading held-out test;
5. preserve the fact that the B2 null needed for G4/G5 is still unavailable.

Only after the research-plan threshold-generation sequence is complete may
`gate_thresholds_v1.json` be frozen and hashed.

## Immutability and provenance

The output directory must not already exist. It contains one compressed trace
per accepted cycle, `cycle_metrics.jsonl`, `replay_manifest.json`, and
`checksums.sha256`. The manifest records repository state, checkpoint and input
artifact SHAs, split, input ranges, evidence scope, and all no-claim flags.
Failed builds remain in a hidden temporary directory with
`BUILD_FAILED.json`; they are not valid replay artifacts.
