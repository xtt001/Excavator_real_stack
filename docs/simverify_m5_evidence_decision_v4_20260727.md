# SimVerify M5 Evidence Decision v4 — 2026-07-27

Decision: `revise_condition`

Experiment version: `B1.5_G4_G5.1_E04`

Terminal for experiment version: `true`

Evidence scope: `recorded-observation/offline`

Closed-loop execution: `false`

Held-out test read: `false`

## Plain-language conclusion

B1.5 can use the supported next-sector condition on the recorded observations.
It also carries that condition across two adjacent cycles with the observable
condition-router lifecycle.

The train-only `video7 = stick_up` dropout revision fixed the old qualitative
failure: removing video7 no longer reverses the requested swing direction in
either eligible source episode. In fact, all 12 E04 camera variants retain a
positive source-mean condition semantic direction.

The model is still not robust enough to advance. The frozen `drop_video7`
non-inferiority rule fails on two episode-34 anchors, and complete E04 passes
only 5 of 12 camera variants. E05, E06, G6, and held-out test therefore remain
unentered.

The correct terminal class is `revise_condition`, not `reject`,
`sim_observable_only`, or `real_finetune_candidate`.

## Verified Gate path

| Gate | Result | Consequence |
| --- | --- | --- |
| G0/G1/G2/M1/M2/G3 | pass, inherited through checksum-verified M5 v3 | observable-only foundation retained |
| G4 B1.5/B2.5 | next-condition understanding established offline development | G5.1 evaluation retained |
| G5 v1 | frozen failed lifecycle predecessor | exact historical defect remains bound |
| G5.1 B1.5/B2.5 | two-cycle core continuity established development | E04 authorized |
| E04 B1.5 | camera-counterfactual robustness not established | stop |
| E05 | not entered | E04 did not authorize |
| E06 | not entered | E04 stop retained |
| G6 | not entered | no robust promotable candidate |
| held-out test | locked unread | no final-test claim |

## E04 result carried into M5

Passing variants:

- `four_camera`;
- `drop_video4`;
- `drop_video5`;
- `stick_only`;
- `swap_stick_pair`.

Failing variants:

- `drop_video6`;
- `drop_video7`;
- `eye_only`;
- `fixed_trace_start`;
- `lag_one_tick`;
- `swap_cross_role_pairs`;
- `swap_eye_pair`.

The two `drop_video7` source metrics retained by the machine-readable decision
are:

| Source episode | Semantic margin | Condition effect | Phase coverage | Failure rate |
| --- | ---: | ---: | ---: | ---: |
| 12 | 0.0407956764 | 0.0030675244 | 1.0000000000 | 0.0000000000 |
| 34 | 0.0308997345 | 0.0028204053 | 0.9722222222 | 0.6666666667 |

This means the condition direction is repaired while the exact frozen
retention requirement remains unmet. The completed E04 threshold and decision
are not changed.

## Immutable M5 v4 package

Path:

```text
/home/pingfan/Excavator_real_stack_artifacts/simverify_m5_decision_v4
```

Builder Git commit:
`9f8e8fa6cff11b3a6abf690cb0e6aff6a3aaf6b5`.

| Artifact | SHA-256 |
| --- | --- |
| `decision.json` | `416ed43d48954ab00c9a7f9ca4a45cc27793195f61359239683bfaa254e58c47` |
| `m5_manifest.json` | `c0f10097aef3d907139caccf2c71b11c13323da8dbfe4d05812114c3c555feb0` |
| `checksums.sha256` | `35ef26b28da4a5d49a09aad7cefe4a6a06d75cd43dac1d7675a99e8ccbf613d1` |

Independent checksum verification passed with zero failures.

The builder reverified:

- 2 prior M5 v3 files;
- 4 B1.5/B2.5 G4 files;
- 93 frozen G5 v1 files;
- 94 B1.5/B2.5 G5.1 files;
- 110 B1.5 E04 files;
- 303 checksum-bound input files in total;
- exact B1.5/B2.5 baseline and checkpoint continuity;
- exact G5 v1 -> G5.1 -> E04 manifest linkage;
- B1.5 linkage to the supplied prior M5 v3 package;
- zero held-out overlap and no closed-loop or real-control permission.

## Machine-readable terminal state

```text
decision                         = revise_condition
terminal_for_experiment_version  = true
evidence_scope                   = recorded-observation/offline
held_out_test_read               = false
held_out_test_authorized         = false
gate_thresholds_v1_generated     = false
sim_observable_only              = false
real_finetune_candidate          = false
control_candidate                = false
real_control_allowed             = false
E05                              = not_entered
E06                              = not_entered
G6                               = not_entered
```

## Next experiment boundary

A new experiment may separate:

1. condition-semantic causality against the matched shuffled-condition null;
2. task-phase preservation using an independently frozen non-inferiority
   margin;
3. fixed-frame and lag probes as temporal-vision sensitivity diagnostics.

The B1.5 validation result may motivate that hypothesis, but it cannot be used
to rewrite the completed E04 threshold or unlock held-out access.

No Unity/AGX, real machine, Jetson, shadow, control, deployment, or closed-loop
claim is authorized.
