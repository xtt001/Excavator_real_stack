# SimVerify M2 Offline-Evaluation Contract

Status: `implementation_pending_formal_immutable_build`

Evidence scope: `recorded-observation/offline`

M2 freezes the evaluator before B0 training starts. It does not load a policy,
train a model, read held-out test data, or generate
`gate_thresholds_v1.json`.

## Entrypoint

After the implementation commit is clean:

```bash
PYTHONPATH=testbed python -m testbed.cli.build_simverify_m2 \
  --repo-root /home/pingfan/Excavator_real_stack_v2.0.0-simVerify
```

The default immutable output is:

```text
/data/pingfan/Excavator_real_stack_data/
  sim_observable_cycle_v3_m2_contract_v1/
```

The builder consumes only the immutable M0 train/validation exports and the
passing M1 report. It verifies input SHA-256 identities and fails if the
worktree is dirty, held-out annotations appear, or M1 is not an offline-only
passing smoke.

## Frozen M2 artifacts

- `test_intent_registry_v1.json`: E00–E07 question, observable inputs,
  intervention, metrics, proof boundary, and stop conditions;
- `replay_trace_schema_v1.json`: independent raw-normalized, raw-direct,
  temporal-aggregation, and future-runtime-safe action stages plus provenance;
- `expert_event_envelope_v1.json`: train-fitted observable action-event
  templates and validation expert distribution;
- `condition_counterfactual_anchors_v1.jsonl`: one-field token swaps with
  explicit support and denominator status;
- `state_hold_anchors_v1.jsonl`: fixed-observation anchors requiring policy
  snapshot/restore and forbidding generated physical state;
- `two_cycle_anchors_v1.jsonl`: consecutive accepted-cycle paths with condition
  continuity;
- `delay_latest_wins_contract_v1.json`: issue/ready tick scheduling, stale
  offset, repeat-last, timeout, and latest-wins semantics;
- `m2_authorization_report_v1.json`: authorization boundary for the next
  stage;
- `m2_manifest.json` and `checksums.sha256`: provenance and immutable artifact
  identities.

Cycle condition ownership remains the M0 half-open interval `[start, end)`.
Observable event extraction additionally includes tick `end`, because M0
defines every `ready_end` as the shared ready boundary. Two-cycle replay must
deduplicate that shared boundary rather than move it into either condition.

`ready_start` and `ready_end` are observable boundary anchors, not fixed action
signatures. For each interior action event and axis, the train split supplies a
sign constraint only when the one-sided 95% Wilson lower bound for the modal
effective sign exceeds majority. Validation expert coverage is recorded as a
distribution; it is not tuned into a hand-entered pass percentage.

## Three-stage action rule

Every later B0/B1/B2 replay must save:

1. the entire raw ACT chunk in normalized network units;
2. the entire raw ACT chunk in frozen source-action units;
3. the action selected by temporal aggregation;
4. a separately materialized future runtime-safe action.

The last two may be numerically equal in an offline scenario, but they may not
share storage or be described as the same semantic stage.

## Authorization boundary

A passing M2 package may authorize only M3 B0 baseline work. It does not
authorize B1, B2 threshold finalization, held-out access, real control,
simulator closed-loop execution, or any closed-loop success claim.
