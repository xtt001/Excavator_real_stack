# SimVerify M3 G3 B0 Evidence — 2026-07-25

Decision: `pass_recorded_observation_baseline`

Authorized next ablations: `B1`, `B2`

Evidence scope: `recorded-observation/offline`

Closed-loop execution: `false`

Held-out test read: `false`

`gate_thresholds_v1.json` generated: `false`

## Frozen implementation

- replay implementation commit:
  `6200036172890bb02e418967673a5f88176b55e7`;
- G3 calibration implementation commit:
  `3ec356a596ec6660af34e996cca537b8528d7c45`;
- B0 checkpoint SHA-256:
  `46baef77b8f2af5887aa11deca8b42652a547268ab203ee31a4c192ef14fc055`;
- checkpoint contract: B0, condition absent, sim-domain,
  real-control forbidden, Jetson forbidden.

## Replay packages

The complete accepted B0 paths were replayed from 15 train source episodes and
3 validation source episodes. Validation was repeated three times with the
same FP32 checkpoint.

| Package | Episodes | Cycles | Manifest SHA-256 | Checksums SHA-256 |
| --- | ---: | ---: | --- | --- |
| train repeat 0 | 15 | 111 | `0e6f6c2d81501d13d74c749116099a12db1236f6e56e7e906855bceef72a6552` | `b0dde9118b9e533d264e76a7a4082c8b9e90cf8ba12ac6de230a7035e6b38199` |
| validation repeat 0 | 3 | 31 | `ef1961efd4fae9f527c20d64bcca96e73490ef0d8e1ef495a4a7618e40e2c78b` | `00c9541c5a574902467ccc8d80572fa3935ef1ac09c32c6e4d1d02283079a2d3` |
| validation repeat 1 | 3 | 31 | `e1abbd1ca3c0f6b5c745f46e2cb67e5356a0beee1b53a82bcb2730acf66ba1e6` | `fc345eb2a2a27047fd2c7ccfe542db64bb05d898f47ddbbc8166cf7a0435c1c9` |
| validation repeat 2 | 3 | 31 | `ea7cecd49f63208a385fc0948b9377ac86c63e7e5eef8c0d42515c5b1cef6652` | `bf7b37856e702c09e8014c2c4595a83a9fd3b9d3eac4bf92b85b6fad9bde14c9` |

Every listed checksum inventory was independently verified without failure.

## Repeat noise

Across 62 reference-versus-repeat cycle trace comparisons:

- normalized raw chunk maximum absolute delta:
  `8.280277252197266e-4`;
- direct source-domain raw chunk maximum absolute delta:
  `4.803687334060669e-4`;
- temporal aggregation maximum absolute delta:
  `9.688735008239746e-5`;
- auxiliary per-cycle MAE maximum delta:
  `1.7881393432617188e-6`;
- event coverage, event order, missing-phase, effective recall, opposite-axis,
  and unexpected-axis metric maximum deltas: all `0`;
- changed event ticks: `0`;
- changed missing-event rows: `0`.

The measured semantic repeat-noise floor is therefore zero. The non-zero
floating-point action delta is retained and not rounded away.

## Source-episode G3 calibration

The immutable calibration package is:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_g3_b0_calibration_v1
```

- manifest SHA-256:
  `a2873857da24f9532cffed296c8b9e7eff9f0021339815a4a69cc818059616b2`;
- checksums SHA-256:
  `89b9f9742f3a8ec84c2ae9ae71b9cbf24b5669c98f75aacb04c6d25ec4e4a0ff`;
- checksum verification: `4 / 4`, no failures;
- bootstrap unit: source episode;
- bootstrap repetitions: `100000`;
- bootstrap seed: `20260725`.

The validation policy-versus-expert event-coverage delta was paired by cycle
and averaged within each source episode. Its three source-episode means were:

```text
[0.022222222222222216, 0.0, 0.0]
```

The bootstrap mean interval was:

```text
p02.5 = 0.0
p50   = 0.007407407407407405
p97.5 = 0.022222222222222216
```

The minimum allowed delta was zero minus measured semantic repeat noise, also
zero. The lower interval endpoint therefore passed at the boundary. Validation
event-order violation was `0`, equal to the expert envelope plus measured
semantic repeat noise.

Validation policy event coverage was:

```text
minimum = 0.8333333333333334
p02.5   = 0.8333333333333334
p50     = 1.0
p97.5   = 1.0
maximum = 1.0
```

Six cycles missed the generated `dig_entry_proxy`; no policy cycle violated
event order. These misses remain in the evidence. They were not deleted or
converted into a hand-selected tolerance.

## Diagnostics that are not Gate thresholds

Validation per-cycle diagnostics include:

- deadzone-effective recall mean: `0.9358378140359519`;
- opposite-direction rate mean: `0.020506501132892253`;
- unexpected-effective-axis rate mean: `0.21374606801167947`;
- auxiliary action MAE mean: `0.13989606667910853`.

The validation unexpected-axis rate reached `0.5314861460957179` on the
single-cycle source episode 20. It remains visible. Direction metrics and MAE
do not receive invented pass percentages in G3; action direction is already
part of the generated event signatures, while final cross-model direction
limits require B2 null and B1 paired evidence.

## What G3 does and does not establish

G3 establishes that the unconditioned B0 observation path can reproduce the
major observable recorded-path action-event semantics well enough to make a
condition ablation interpretable. It authorizes only the matched B1 and B2
experiments.

It does not establish:

- simulation closed-loop task success;
- digging, payload, soil, or environment response;
- condition use;
- real-camera or real-machine transfer;
- deployment or control candidacy.

The held-out test remains locked. Global `gate_thresholds_v1.json` cannot be
generated until B2 shuffled-condition null and B1 paired effects exist.
