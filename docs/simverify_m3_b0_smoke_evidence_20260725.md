# SimVerify M3/B0 Smoke Evidence — 2026-07-25

Status: `smoke2_passed_formal_B0_training_authorized`

Evidence scope: `recorded-observation/offline`

This evidence validates the B0 training and checkpoint path. It is not a G3
policy result, does not generate `gate_thresholds_v1.json`, and does not
authorize B1, held-out access, or closed-loop claims.

## Failed smoke1 retained

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_b0_unconditioned_v1_seed0_smoke1/
```

Smoke1 completed one epoch, but the checkpoint serializer stored only
task/seed/class. The run metadata had the required sim-domain prohibition, but
`policy_best.ckpt` did not. The checkpoint SHA-256 was
`cacbbe10f1af4d89304aa04ede25a34005ba91b0af54b3b990d1fcc21f0ce6f1`.
It is retained as non-promotable failed evidence.

## Passing smoke2

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_b0_unconditioned_v1_seed0_smoke2/
```

Git commit:
`9b0690696962e5cbbd2868e64678d1ad5d587374`

Git dirty at run start: false.

| Artifact | SHA-256 |
| --- | --- |
| `dataset_stats.pkl` | `7214c9bdf5df4f957e4c4e9ec3bdfe56dbc88b28209b23d8f3a7d7cc1cb95c68` |
| `resolved_config.yaml` | `07c54508acc42faf7b621e3e3f3efd37d5370135c88266f88e526d1dffe933f7` |
| `run_metadata.json` | `0ec3c5ca771bd1124c38e9a3b58e62aa932bcd786d3b8dc337ceb576ea4527a0` |
| `policy_best.ckpt` | `526f635eca90a375162946957bbd5fb2ce5bbd284f067ab2bc4c792e4c3f4c51` |

The run completed one epoch with best epoch zero. The validation loss
`81.60751342773438` is recorded only to prove the training loop completed; it
is not a Gate threshold or evidence that the cycle task is learned.

## Contract checks

- train episodes: 15;
- validation episodes: three;
- normalization episode ids exactly equal train ids;
- sample-valid mask: `conditions/valid_mask`;
- condition input: absent;
- qpos+qvel dimension: eight;
- camera role encoding: enabled;
- deadzone loss: enabled;
- model parameters: 83.87 million;
- source domain is real: false;
- held-out access: false;
- closed-loop execution: false.

The checkpoint itself now embeds:

```text
domain=sim
source_action_domain=actuator_speed_cmd
deployment_status=offline_evaluation_only
real_control_allowed=false
jetson_allowed=false
```

## Reload and bounded replay

`policy_best.ckpt` reloaded in the real-stack ACT adapter and replayed the
first 20 recorded observations of validation episode 12.

- policy action shape: `(20, 4)`;
- latest raw normalized chunk shape: `(20, 4)`;
- latest raw direct-unit chunk shape: `(20, 4)`;
- normalized/direct storage alias: false;
- inference precision: FP32;
- temporal aggregation: enabled;
- environment response generated: false.

This bounded reload is an import/trace smoke only. Formal G3 evidence requires
the completed 2000-epoch B0 checkpoint, complete accepted-cycle replay, repeat
noise runs, task-event metrics, and source-episode bootstrap.

