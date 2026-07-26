# SimVerify M5 Evidence Decision v3 — 2026-07-26

Decision: `revise_condition`

Experiment version: `B1.4_G5.1_E04`

Terminal for experiment version: `true`

Evidence scope: `recorded-observation/offline`

Closed-loop execution: `false`

Held-out test read: `false`

## Plain-language conclusion

The current model can use the supported next-sector condition when all four
recorded cameras are present, and it can carry that condition correctly across
two adjacent cycles after the condition-router lifecycle fix.

It is not robust enough to promote. Removing the stick-up camera or both stick
cameras can reverse the requested swing response. Fixed images also reverse
the response, and a one-tick image lag leaves the matched condition-response
envelope. Camera-role swaps barely affect the output, so explicit role
sensitivity is not established.

The data, observable annotation, and four-camera condition signal are usable.
The conditioned candidate must be revised before further robustness or
transfer testing. Therefore the final class is `revise_condition`, not
`reject`, `sim_observable_only`, or `real_finetune_candidate`.

## Verified Gate path

| Gate | Result | Consequence |
| --- | --- | --- |
| G0/G1/G2/M1/M2/G3 | pass, inherited and checksum reverified | observable-only foundation retained |
| G4 B1.4 | next-condition understanding established offline development | G5 core authorized |
| G5 v1 | fail | condition router lifecycle defect identified |
| G5.1 | two-cycle core continuity established development | E04 authorized |
| E04 | camera counterfactual robustness not established | stop |
| E05 | not entered | E04 did not authorize |
| E06 | not entered | E04 stop retained |
| G6 | not entered | no robust promotable candidate |
| held-out test | locked unread | no final test claim |

## Immutable M5 v3 package

Path:

```text
/data/pingfan/Excavator_real_stack_data/simverify_m5_decision_v3
```

Builder Git commit:
`e5e5aa9`.

| Artifact | SHA-256 |
| --- | --- |
| `decision.json` | `98f1332514fd5215bce32c494a46edd87bf66c9b36c7318056aad3aad3e66e34` |
| `m5_manifest.json` | `1dde132ea2cb9544a87ffa66cc1143402aa184b4e5e72d3fc056761e9cc85696` |
| `checksums.sha256` | `20adc8c9371fa47089bc2c82d4d462c53ff2731e6c8a34d15012b73cda6f6ce6` |

Independent `sha256sum -c checksums.sha256` verification passed.

The builder reverified:

- the prior M5 v2 package carrying G0 through G3 evidence;
- the passing B1.4 next-condition G4 package;
- the immutable failed G5 v1 package;
- the passing G5.1 router-lifecycle package and its exact G5 v1 linkage;
- the E04 package, its exact G5.1 linkage, and matching B1.4 checkpoint;
- 303 checksum-bound input files across those packages;
- zero held-out episode overlap and no closed-loop evidence.

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

A new experiment may start only after freezing one primary conditioned-visual
factor that addresses the observed stick-up dependence and missing camera-role
sensitivity. Condition representation, camera augmentation, and runtime
semantics may not be changed together.

This M5 result does not authorize E05/E06 continuation for the current
experiment, held-out test access, real fine-tuning, shadow, control, deployment,
or any closed-loop success claim.
