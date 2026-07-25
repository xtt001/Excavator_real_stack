# SimVerify M0/M1 Evidence Record — 2026-07-25

Status: `m1_import_smoke_passed_m2_pending`

Evidence scope: `recorded-observation/offline`

This record closes the bounded M0 plus M1 slice. It does not authorize
training, held-out test access, simulator execution, real-machine execution,
checkpoint promotion, or a closed-loop claim.

## M0 immutable package

```text
/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3
```

M0 code Git commit:
`a59fe690ed8a0b727c07e2dd58cdbf86e95d95ec`

| Artifact | SHA-256 |
| --- | --- |
| `dataset_manifest.json` | `3ee02fa711581b25803c04524c3bfa6bd96464c1c69e3d7c8a269f00111ecc62` |
| `checksums.sha256` | `384025726e73299c19653004023bdd56db2f992ee0df9078efba972126c9b808` |
| `annotation_thresholds_v2.json` | `35b06b68e30885de47c3b53fe03cde3f4b2fc5b577960a2a89ec24c8df7babb0` |
| `cycle_annotations.jsonl` | `bb531a1f8283362b4841f99f603e19d1725cd61cf571a75865c1bee3c8aca5e5` |

Independent `sha256sum -c checksums.sha256` verification passed for every main
artifact and all 24 materialized episode HDF5 files.

### M0 Gate results

- event selector: 1024/1024 source-episode refits completed;
- eye balanced-accuracy p02.5 `0.7886336` exceeded null p95 `0.3600793`;
- stick balanced-accuracy p02.5 `0.8389600` exceeded null p95 `0.3632156`;
- generated interval-confirmation threshold: `891/1024 = 0.8701171875`;
- one-sided 95% Wilson lower bound: `0.8518572`, above majority;
- dump and both sector boundary CIs remained strictly between adjacent
  cluster-center CIs;
- sector-eye balanced-accuracy p02.5 `0.8571429` exceeded null p95
  `0.4018438`;
- final train and validation condition matrices both retained 9/9 transitions;
- continuity errors: zero in train and validation;
- held-out transition status: `locked_unread`;
- privilege scan: 24/24 episodes passed, no oracle dependency;
- accepted cycles: 142; review/ambiguous cycles: 350.

Final accepted-cycle condition matrices:

| train current \ next | left | center | right |
| --- | ---: | ---: | ---: |
| left | 17 | 16 | 5 |
| center | 7 | 18 | 14 |
| right | 5 | 9 | 20 |

| validation current \ next | left | center | right |
| --- | ---: | ---: | ---: |
| left | 5 | 3 | 2 |
| center | 3 | 1 | 6 |
| right | 1 | 3 | 7 |

### 20 Hz export QC

- time basis: `step_id * metadata.dt`;
- wall-clock `step_ns` used: false;
- action offset: `0.0 s`;
- same source row for images/qpos/qvel/action/condition: true;
- valid action-sign segments: 11292;
- preserved segments: 11229;
- missed segments: 63;
- missed segments lasting at least 50 ms: zero;
- maximum preserved onset delay: `0.0399999991 s`.

The 63 sub-50-ms misses remain reported. They are not hidden or treated as
closed-loop evidence.

## M1 bounded import smoke

Report:

```text
/data/pingfan/Excavator_real_stack_data/
  sim_observable_cycle_v3_m1_import_smoke.json
```

Report SHA-256:
`9e50b0bae20412f5d67313e5ab3786966a3bf899aa2f8b83e8d71b64416b2091`

M1 code Git commit:
`f994b6bdf153f308b21c45787a42be8556552e15`

The smoke selected exactly:

| split | episode | steps | decoded JPEGs | valid condition rows |
| --- | ---: | ---: | ---: | ---: |
| train | 3 | 8000 | 32000 | 2738 |
| validation | 12 | 8000 | 32000 | 4758 |

It verified:

- bounded input checksums against the M0 checksum inventory;
- exact camera order `video4,video5,video6,video7`;
- frozen transform ID, RGB contract, and decoded shape `216x384x3`;
- finite float32 qpos/qvel/action with shape `(T,4)`;
- exact observation/action source-index equality and monotonicity;
- sim-time basis and zero action offset;
- condition schema, binary one-hot invariants, valid mask, cycle constancy, and
  exact annotation-sidecar materialization;
- no external or virtual HDF5 datasets;
- privilege scan passed;
- no held-out, oracle audit, source HDF5, PACT package, simulator backend, or
  ACT access.

M1 proves importability only. The report explicitly records:

```text
closed_loop_execution=false
training_started=false
training_authorized=false
held_out_test_read=false
```

## Next Gate

The next allowed stage is M2 offline-evaluation skeleton work. M2 must preserve
the held-out lock until a finite SHA-bound `gate_thresholds_v1.json` exists.
No M3/M4 training may start from this M0/M1 result alone.
