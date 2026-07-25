# SimVerify M2 Evidence Record — 2026-07-25

Status: `M2_completed_M3_B0_authorized_not_started`

Evidence scope: `recorded-observation/offline`

M2 freezes the offline evaluator and intervention anchors. It does not contain
a policy replay result, model metric, finite G3/G4/G5 threshold, held-out
result, simulator run, or closed-loop evidence.

## Immutable package

```text
/data/pingfan/Excavator_real_stack_data/
  sim_observable_cycle_v3_m2_contract_v1/
```

M2 implementation Git commit:
`c9c8ab733dedaea3c11b4de8d27dde520edf3b2e`

| Artifact | SHA-256 |
| --- | --- |
| `m2_manifest.json` | `edb8720c4573e59017991c62a1b473826b6f4c595545fa5e750c15de943265a0` |
| `checksums.sha256` | `382d9d4356e4cf3b611fabdf2f08fdb75c4f9166c9abc5d45c0735a95551e33e` |
| `test_intent_registry_v1.json` | `30f5fac215bd5d897f61728bc4e71c8dc43a061a70de7908ad0c7770b05d8b5e` |
| `replay_trace_schema_v1.json` | `e623eeb498a0ab7b8d9c2730d0f9d8251da61f8c1ffb1a72960f102732ff8680` |
| `expert_event_envelope_v1.json` | `1bcfa3933a4736cd92267efd671d8258c0d79ee217a18409c3a7d15a6bd88304` |
| `condition_counterfactual_anchors_v1.jsonl` | `caf8512df5a29d5bced62df9cf59d8e6f573b6c089fd45fa214748e35d08004c` |
| `state_hold_anchors_v1.jsonl` | `11aeca0c733a71e1591956a1ceea8aa88427d7a32ff5f8cad96b150fc0fa6763` |
| `two_cycle_anchors_v1.jsonl` | `efe68d42131f1072285b24adae7716ddc8e88c647fefd6fc56f1ac150e8f4576` |
| `delay_latest_wins_contract_v1.json` | `43c10a96604f2790960710f266228f417ba06964b86e3b10b46e93e2f598ddf6` |
| `m2_authorization_report_v1.json` | `8c5dad9b7c849529d24d203d7551edbddd30d1a3cfd14e0342a92cb146721fd7` |

Independent `sha256sum -c checksums.sha256` and the repository checksum
verifier both passed all nine inventoried files.

## Provenance and isolation

- Git branch: `v2.0.0-simVerify`;
- Git dirty state at build: false;
- M0 dataset manifest SHA-256:
  `3ee02fa711581b25803c04524c3bfa6bd96464c1c69e3d7c8a269f00111ecc62`;
- M1 report SHA-256:
  `9e50b0bae20412f5d67313e5ab3786966a3bf899aa2f8b83e8d71b64416b2091`;
- source splits read: train and validation only;
- input exported episodes: 20;
- held-out status: `locked_unread`;
- held-out episode access count: zero;
- training started: false;
- closed-loop execution: false.

## Frozen evaluator evidence

The E00–E07 registry contains eight complete HR-12 test intents. Every intent
records its observable inputs, intervention, metrics, proof boundary, stop
conditions, privilege prohibition, held-out lock, and offline evidence scope.

The trace schema requires independently materialized:

- raw normalized ACT chunk;
- raw source-domain ACT chunk;
- temporal-aggregation action;
- future runtime-safe action.

Its validator passed a synthetic schema smoke and rejects action stages that
share storage.

## Expert event-envelope self-check

The event rules were fit on accepted train cycles only and applied to 31
accepted validation cycles.

| Metric | Result |
| --- | ---: |
| event coverage minimum | 0.8333333 |
| event coverage p02.5 | 0.8333333 |
| event coverage median | 1.0 |
| event coverage p97.5 | 1.0 |
| event-order violation rate | 0.0 |

These values characterize evaluator compatibility with expert validation
actions. They are not a policy threshold. Final G3/G4/G5 thresholds remain
unavailable until B0 repeated-replay noise and B2 shuffled-condition null
artifacts exist.

## Anchor inventory

| Anchor type | Count |
| --- | ---: |
| one-field condition swaps | 568 |
| supported one-field swaps | 234 |
| state-hold anchors | 852 |
| adjacent accepted two-cycle paths | 79 |

All 568 token-swap rows change exactly one primary condition field.
Unsupported swaps remain visible but have
`included_in_success_denominator=false`.

## Authorization

M2 passed as an evaluation-contract stage and authorizes only M3 B0
unconditioned-baseline work. It does not authorize:

- conditioned B1 promotion;
- B2-based final threshold generation before the required replay artifacts;
- held-out test access;
- real-machine or simulator closed-loop execution;
- checkpoint deployment or a closed-loop success claim.

