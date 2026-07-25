# SimVerify M3 B1/B2 Smoke Evidence — 2026-07-25

Status: `passed_formal_training_authorized`

Evidence scope: `recorded-observation/offline`

Closed-loop execution: `false`

Held-out test read: `false`

Implementation commit:
`1e24fe843e980ef9e6f67e0659d5ed0bb797c919`

Both one-epoch runs were launched from a clean
`v2.0.0-simVerify` worktree at that commit. Their losses are smoke diagnostics,
not Gate operands.

## B1 smoke

Bundle:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_b1_conditioned_v1_seed0_smoke1
```

- status: `completed`;
- best epoch: `0`;
- validation loss: `75.589294`;
- dataset stats SHA-256:
  `2f600ef381c636b3723afb5a89ae660e1f57b148934c8a2b24930b45260d39aa`;
- resolved config SHA-256:
  `c4a8c71e913983195df718fa76bd79a8afd671fb740a9fba6bd7f2ac2acf0360`;
- run metadata SHA-256:
  `01b68262e4e42fd7942eee2f6cdad7a4b87f57ee7d67f750540c1ad7a215066e`;
- best checkpoint SHA-256:
  `03d09caef94523b7d5ded4829c24603cee1679d728dbd0522d914200d8192579`.

The checkpoint embeds B1, observed hindsight association, condition low-dim
input, G3 provenance, and sim-domain real/Jetson prohibition.

## B2 smoke

Bundle:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_b2_shuffled_condition_v1_seed0_smoke1
```

- status: `completed`;
- best epoch: `0`;
- validation loss: `75.589355`;
- dataset stats SHA-256:
  `2f600ef381c636b3723afb5a89ae660e1f57b148934c8a2b24930b45260d39aa`;
- resolved config SHA-256:
  `228989cb68bb25052657ced7ba4ebf2bdca888c7167789be9563118a64c63097`;
- run metadata SHA-256:
  `93c0e4686a6cbc3d48d7bea6f54ba7844c28d4f236b8fb1037844f46fe8886d4`;
- best checkpoint SHA-256:
  `2eff9dcb3cb85224cfd80a0c9fbe84bbfece2f5f2250c010254740b50ef6910b`.

The checkpoint embeds B2 and the deterministic train-valid-start mapping:

```text
rows       = 31519
changed    = 27465
unchanged  = 4054
mapping SHA-256 =
  cdc2c0fbb83f8d225f3b978a00b3778024b4a8f6f7c06a7a6bb6b28235c6e86f
```

All nine source token counts equal their shuffled token counts. Validation
shuffle is disabled.

## Reload smoke

Each best checkpoint was reloaded in FP32 with temporal aggregation and replayed
over 20 recorded episode-12 observations with four cameras, qpos, qvel, and
the recorded condition.

For both B1 and B2:

- temporal-aggregation output shape: `(20, 4)`;
- normalized raw chunk shape: `(20, 4)`;
- direct source-domain raw chunk shape: `(20, 4)`;
- all outputs finite.

The identical dataset-stats SHA proves B1 and B2 use the same train-only
normalization. Formal matched training is authorized. G4 remains pending until
formal checkpoints, B2 null replay, B1 interventions, and source-episode paired
bootstrap exist.
