# SimVerify B1.2 pre-dump consistency evidence — 2026-07-26

Decision: `revise_condition`

Candidate decision: `condition_understanding_not_established`

Evidence scope: `recorded-observation/offline`

Closed-loop execution: `false`

Held-out test read: `false`

B1.2 changed one primary factor from frozen B1. It retained the recorded
condition and expert action supervision, then added a deterministic
zero-latent paired consistency loss only for chunk-safe pre-dump train starts.
The paired branch changed only the next-sector token and received no expert
action target.

No held-out episode, Unity/AGX execution, PACT import, Jetson, real hardware,
or real-control default was used.

## Implementation and training provenance

- implementation commit:
  `10bd76c657476e125da3b70ab0ade0b2cb74a1e8`;
- clean training/evaluation snapshot:
  `6ebd39a56292c47cad5706084757e5a78479cc90`;
- training bundle:
  `/data/pingfan/Excavator_real_stack_data/simverify_b1_2_pre_dump_consistency_v1_seed0`;
- status: `completed`;
- best epoch: `1990`;
- best validation loss, diagnostic only: `0.24565671384334564`;
- `dataset_stats.pkl` SHA-256:
  `2f600ef381c636b3723afb5a89ae660e1f57b148934c8a2b24930b45260d39aa`;
- `resolved_config.yaml` SHA-256:
  `6e7c0a5d405edbd0a78dea54690bc0daac54859e244c537b6b58b59f179234a5`;
- `run_metadata.json` SHA-256:
  `4b50af2e7abbb145e6d2801aa0c00fea532ba29a7727cb2daa10c3ca778c6b4e`;
- `policy_best.ckpt` SHA-256:
  `279c28be3d2167e6ed45b4d13af215d671e0c7e74947b6b8f0337e7bb61362f5`.

The train manifest contains 31,519 valid starts, 23,496 eligible pre-dump
pairs, and 8,023 crossing-or-post starts without an auxiliary pair. Every
eligible next-sector token changed, exact left/center/right marginals remained
6,528 / 9,504 / 7,464, and the mapping SHA-256 is:

```text
a8e7e2b04b0249be5a8dfbdc6d57a1dda0e678d09b2aa644a039b24dc147310f
```

Validation had zero auxiliary examples and zero auxiliary loss. Checkpoint
selection therefore used the unchanged B1 validation objective.

## Replay provenance

Each requested replay retained 124 anchors: 45 supported and 79 unsupported.
Unsupported anchors were excluded from all success denominators.

| Replay | Manifest SHA-256 | Checksums SHA-256 |
| --- | --- | --- |
| requested repeat 0 | `03233e64355d697e6f69caa92c397cd40657c977f0d3d2600b320072dd04f8b3` | `5f32d634bd9d1474541bed99c724dbb49f8cd7c03e18d7d777f9a81c0d66e334` |
| requested repeat 1 | `70ed65a2afe45e5a6c61930c3c149ef9be634cc97770898ef044d622762337f9` | `ee8e1fd2253a71c3883ce2a3d0e570487e97ba4e484e94509a5ec8ba33fb6195` |
| requested repeat 2 | `cf2c4f61370b94ec4614cd50f9fa9afd02921eb127ba4797cb5bd7e02bb87d3e` | `6b132d000fe27aa44f6940549ffb97fd53d220ce8e7a22c051bcd02983661b23` |
| identical-token mask | `4ff38fbca0ee7dfde3946ef81ed563f27580e821b706661d9b468b1ba26d72a1` | `e314c4a38494df3d5846a8b4bc05c45efa6b52cee9b6f5d2862d40abd6da5d38` |

All checksum inventories passed. The 45 supported masked anchors delivered
identical base and target tokens; maximum action effect and maximum per-tick
effect were both exactly zero.

Every replay trace separately stores the normalized raw policy chunk, direct
source-domain raw policy chunk, temporal-aggregation action, and future
runtime-safe action.

## Fixed-observation causal Gate

The immutable Gate artifact is:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_b1_2_condition_causal_v2_validation_v1
```

- bootstrap unit: source episode;
- bootstrap repetitions: `100000`;
- manifest SHA-256:
  `35d5ef11c57d6d6f58d091786d677eea5bd0b4dfe6c3d4167e73b7c85115fe02`;
- checksums SHA-256:
  `5eb651ec43c8beb29fda38d81735504ff18b1fea59a4e8de93e28b80c02934ff`.

| Criterion | current sector | next sector |
| --- | --- | --- |
| action effect > identical-token mask | pass | pass |
| signed semantic margin > B2 | pass | fail |
| identity beats all five semantic permutations | fail, 3/5 | fail, 3/5 |
| phase-specific effect > mask | pass | pass |
| phase specificity > B2 | fail | fail |
| task envelope preservation | pass | pass |
| factor decision | **fail** | **fail** |

The next-sector signed-semantic comparison was negative in all three source
episodes versus B2. Its paired-bootstrap mean was `-0.008929`, with 2.5%
quantile `-0.022982`. This is not a marginal miss.

## B1 / B1.1 / B1.2 comparison

Values are means over source-episode aggregates, not frame-weighted means.

| Factor / metric | B1 | B1.1 | B1.2 |
| --- | ---: | ---: | ---: |
| current action effect | 0.030324 | 0.029173 | 0.021055 |
| current signed semantic margin | 0.011494 | 0.007250 | 0.009079 |
| current phase specificity | 0.010145 | 0.006097 | 0.006858 |
| current semantic permutations rejected | 5/5 | 3/5 | 3/5 |
| next action effect | 0.028491 | 0.021576 | 0.003686 |
| next signed semantic margin | 0.062237 | 0.034670 | 0.005483 |
| next phase specificity | -0.002953 | 0.000652 | 0.001746 |
| next semantic permutations rejected | 3/5 | 3/5 | 3/5 |

B1.2 produced the best mean next-sector phase specificity of the three
experiments, but reduced next-sector action effect by 87% and signed semantic
margin by 91% relative to B1. Current-sector evidence also remained below B1.

## Interpretation

The intervention did what its loss requested: it made the pre-dump policy more
invariant to the next token and moved the mean next-sector effect toward the
post-dump phase. It did not preserve enough meaningful next-sector response.
Changing the token still changes the action above numerical noise, but the
change is too weak and does not reliably match the token's sector meaning.

Across B1, B1.1, and B1.2, the shared low-dimensional condition path has shown
a timing-versus-semantics tradeoff:

```text
B1:   stronger meaning, wrong next-token timing
B1.1: slightly better timing, weaker meaning
B1.2: best timing, next-token meaning nearly erased
```

This supports revising condition representation or routing, not tuning the
B1.2 coefficient after seeing validation results. A further experiment would
need a newly frozen one-factor contract; it is not authorized by this result
alone.

## Plain-language conclusion

B1.2 taught the model to mostly ignore the future target before dumping, which
fixed the timing direction. The model then mostly ignored that future target
everywhere, so it still could not reliably act according to the requested
next sector. The model reacts to condition bits, but the evidence does not show
that it understands their meaning at the correct time.

Therefore B1.2 is rejected, the overall status remains `revise_condition`, and
held-out test, G5, deployment, and any real-machine claim remain locked.
