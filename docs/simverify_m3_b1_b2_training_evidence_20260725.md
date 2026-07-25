# SimVerify M3 B1/B2 Formal Training Evidence — 2026-07-25

Status: `formal_checkpoints_completed_condition_replay_pending`

Evidence scope: `recorded-observation/offline`

Closed-loop execution: `false`

Held-out test read: `false`

Both runs used clean Git commit
`5f70c5efec991d8df6b75e6415dd42739420a96c`, the same source-episode split,
the same 14-dimensional ACT architecture, seed, optimizer, normalization,
cameras, chunks, losses, and 2000 epochs.

## B1

Bundle:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_b1_conditioned_v1_seed0
```

- status: `completed`;
- best epoch: `1880`;
- best validation loss: `0.1814105063676834`;
- dataset stats SHA-256:
  `2f600ef381c636b3723afb5a89ae660e1f57b148934c8a2b24930b45260d39aa`;
- resolved config SHA-256:
  `ab7cb61ddb87c2f17854a5f17b8670f177ed06c943a279716da241f9f0396589`;
- run metadata SHA-256:
  `b0534ea3d0fe46ad32f6e587418f6e554011caba415ee951816b188754416ae5`;
- best checkpoint SHA-256:
  `50b73b956df35a54be179ba46cea1f44b953f18792ae538b46cd82226a76f08b`.

## B2

Bundle:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_b2_shuffled_condition_v1_seed0
```

- status: `completed`;
- best epoch: `1990`;
- best validation loss: `0.20184281468391418`;
- dataset stats SHA-256:
  `2f600ef381c636b3723afb5a89ae660e1f57b148934c8a2b24930b45260d39aa`;
- resolved config SHA-256:
  `f9408edea5c5d41ab73222f1d3443548961e80f84d4cc5676017d0cf609523ac`;
- run metadata SHA-256:
  `85c9878cf7b3b055f14ac68c85f4c3d0d77fe20074dccf389cf046516736ca66`;
- best checkpoint SHA-256:
  `d29428e8d3e366ff5fe154b9eff25e2f64a7add16fba5a7187d4262481488868`.

The B2 metadata and checkpoint both embed mapping SHA
`cdc2c0fbb83f8d225f3b978a00b3778024b4a8f6f7c06a7a6bb6b28235c6e86f`.
Both checkpoints retain sim-domain, real-control-forbidden, and
Jetson-forbidden semantics.

## Interpretation boundary

The B1 validation loss is lower than B2 by about `0.02043`, but this is not
evidence that condition is used. Training loss and teacher-forced validation
loss cannot distinguish genuine token response from capacity, optimization,
or correlated-observation effects.

G4 remains pending until:

- B1 and B2 are replayed on the exact same supported condition anchors;
- each intervention changes exactly one condition field;
- B1 same-checkpoint replay noise is measured;
- unsupported anchors remain outside the success denominator;
- source-episode paired bootstrap compares B1 against the B2 null.
