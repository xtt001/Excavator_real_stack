# SimVerify M5 v4 Decision Contract

Status: `method_frozen_before_implementation`

Experiment version: `B1.5_G4_G5.1_E04`

Evidence scope: `recorded-observation/offline`

Closed-loop execution: `false`

Held-out test read: `false`

## Purpose

M5 v4 closes the B1.5 train-only `video7` dropout revision as a distinct
experiment version. It binds the passing B1.5/B2.5 next-condition and
two-cycle evidence to the complete failed B1.5 E04 camera-counterfactual
result.

It does not rerun a model, change an E04 threshold, enter E05/E06/G6, read
held-out episodes, or promote a checkpoint.

## Immutable inputs

The builder must checksum-verify and SHA-bind:

1. prior M5 v3 package
   `/data/pingfan/Excavator_real_stack_data/simverify_m5_decision_v3`;
2. passing B1.5/B2.5 G4 package
   `/home/pingfan/Excavator_real_stack_artifacts/simverify_b1_5_next_condition_causal_validation_v1`;
3. immutable failed G5 v1 lifecycle evidence
   `/data/pingfan/Excavator_real_stack_data/simverify_g5_two_cycle_recorded_path_development_v1`;
4. passing B1.5/B2.5 G5.1 package
   `/home/pingfan/Excavator_real_stack_artifacts/simverify_b1_5_g5_1_two_cycle_router_lifecycle_development_v1`;
5. complete B1.5 E04 package
   `/home/pingfan/Excavator_real_stack_artifacts/simverify_b1_5_e04_camera_counterfactual_development_v1`.

The expected manifest, Gate, and checksum-file SHA-256 values are:

| Input | Manifest SHA-256 | Gate SHA-256 | Checksums SHA-256 |
| --- | --- | --- | --- |
| prior M5 v3 | `1dde132ea2cb9544a87ffa66cc1143402aa184b4e5e72d3fc056761e9cc85696` | `98f1332514fd5215bce32c494a46edd87bf66c9b36c7318056aad3aad3e66e34` | `20adc8c9371fa47089bc2c82d4d462c53ff2731e6c8a34d15012b73cda6f6ce6` |
| G4 B1.5/B2.5 | `9819ce8c56f109889b77c90f3fe4a06840dcd45d7cece3ea5d131588f7b1cbc6` | `0daafb5f5516364a491da78e7ee3d3a9ef019570f73cf9faf694c6de0f88ee14` | `39a7e9168523376d395d5749d0b491b448bd7f7f36bf3e1f804c2b975a6336a9` |
| failed G5 v1 | `8fb302ab070c80518a8b6c03a7a5558290002ba76ed67e7a725f1e3d397c4b62` | `d5a45c7d1ae94f2a6f4c6b0a5ec472ef9a8ce9f0ea102cfe352a7d6084d5a5c3` | `9ebeada6ef7341ca7c5581544be584fe2518f1773577a86bfb56b8de00fa4386` |
| G5.1 B1.5/B2.5 | `8977a26dfe7addc89d78f2c6c8bb304e459152bb3e2b97eae5a0778170c836d9` | `f5a8ad6c2ea8ba82c78c68dca660bd8b404714640ab41969e22fd7c3d623f8fc` | `a0bf1fd5659d414c2eab7c35c96d6dcf4eac9599ae8b6014c9cacdc0db79e7a2` |
| E04 B1.5 | `7f206f1954be525a240b5e4e0ed3ab89e9c31ea38aa3bd0f833b8e32f864afa6` | `8ff1340093185cdf2d0c070a896ff0828325531798f4517d1be3a7651aa5f004` | `ab4d1ee6f82f5ea24d44295821d76759f082c6793ebe2bb94b6cab8a68ae5a99` |

## Chain validation

The builder must reject the package unless all of the following hold:

- prior M5 v3 is terminal `revise_condition` for `B1.4_G5.1_E04`;
- G4 declares exactly candidate `B1.5` and null `B2.5` and passes;
- the supplied G5 v1 package is the frozen failed lifecycle predecessor;
- G5.1 declares the same B1.5/B2.5 pair, binds that G5 v1 manifest, and
  authorizes remaining G5 robustness;
- E04 declares candidate `B1.5`, binds that G5.1 manifest, has decision
  `e04_camera_counterfactual_robustness_not_established`, and does not
  authorize E05;
- the B1.5 checkpoint SHA is identical across G4, G5.1, and E04;
- the B2.5 checkpoint SHA is identical across G4 and G5.1;
- B1.5 checkpoint provenance binds the supplied prior M5 v3 manifest and
  checksum identities;
- no input manifest or Gate claims held-out access, closed-loop execution, or
  real-control permission;
- no E04 source episode overlaps held-out episodes `1`, `13`, `25`, or `33`.

## Frozen decision

The only authorized decision is `revise_condition`.

It is not `reject` because:

- the observable data and annotation chain remains valid;
- B1.5 exceeds the matched B2.5 semantic null in G4 and G5.1;
- full-camera two-cycle phase, event order, ready continuity, and router
  lifecycle pass;
- B1.5 removes the old negative `drop_video7` semantic direction in both
  eligible source episodes.

It is not `sim_observable_only` or `real_finetune_candidate` because:

- complete E04 passes only 5 of 12 variants;
- the frozen targeted `drop_video7` failure envelope is not met;
- E05, E06, and G6 were not entered;
- held-out test remains locked.

The decision package must preserve the distinction between:

1. condition semantics established under recorded observations;
2. improved but insufficient camera robustness;
3. no evidence of environmental response or closed-loop excavation.

## Outputs

```text
simverify_m5_decision_v4/
  decision.json
  m5_manifest.json
  checksums.sha256
```

`decision.json` must include the complete Gate path, candidate/null identities,
the passing and failing E04 variants, the per-source `drop_video7` semantic
margins, promotion locks, and the next admissible research action.

`m5_manifest.json` must include Git provenance, this contract SHA, every input
identity, and the B1.5/B2.5 checkpoint SHAs.

## Next admissible research action

After M5 v4 closes this experiment, a new experiment may freeze a revised
camera-robustness design. It must separate condition-semantic causality,
task-phase non-inferiority, and temporal-vision diagnostics. Any new numeric
margin must be frozen before another validation or held-out run.

The existing E04 result may be used to formulate the new hypothesis, but its
thresholds and decision must remain unchanged.
