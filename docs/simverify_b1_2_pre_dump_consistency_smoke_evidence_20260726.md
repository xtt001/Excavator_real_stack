# SimVerify B1.2 pre-dump consistency smoke evidence — 2026-07-26

Status: `passed_formal_training_authorized`

Evidence scope: `recorded-observation/offline`

Closed-loop execution: `false`

Held-out test read: `false`

Implementation commit:
`10bd76c657476e125da3b70ab0ade0b2cb74a1e8`

The one-epoch smoke launched from a clean `v2.0.0-simVerify` worktree at that
commit. The branch was one commit ahead of upstream because both push attempts
failed when the configured SSH endpoint closed the connection. This does not
change the recorded code snapshot.

Bundle:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_b1_2_pre_dump_consistency_v1_seed0_smoke1
```

- status: `completed`;
- best epoch: `0`;
- validation loss: `75.58953857421875`;
- train auxiliary normalized-action L1 mean: `0.0036`;
- train mean eligible examples per four-sample batch: `3.25`;
- validation auxiliary loss and eligible count: `0`, as frozen;
- dataset stats SHA-256:
  `2f600ef381c636b3723afb5a89ae660e1f57b148934c8a2b24930b45260d39aa`;
- resolved config SHA-256:
  `9e1639570ab0c4e29c4212e37dbc612b9804315fd6f41e78c8e2ed2b521d73f1`;
- run metadata SHA-256:
  `871c78340cce03218545a4bc0377909be932383b1508cf6d37508e9c6f66b25b`;
- best checkpoint SHA-256:
  `50942a2f8f25ce5680253d6e0b14d6578ff021b1214b3c05fd1b4be0c8622db8`.

The resolved provenance records 31,519 valid train starts, 23,496 chunk-safe
pre-dump auxiliary pairs, and 8,023 crossing-or-post starts without a pair.
Every eligible next-sector token changes while exact left/center/right
marginals remain 6,528 / 9,504 / 7,464. The mapping SHA-256 is:

```text
a8e7e2b04b0249be5a8dfbdc6d57a1dda0e678d09b2aa644a039b24dc147310f
```

The smoke proves loader, deterministic zero-latent paired inference, gradient,
AMP, validation exclusion, checkpointing, and provenance paths execute
together. Its losses are diagnostics only and are not Gate operands.
