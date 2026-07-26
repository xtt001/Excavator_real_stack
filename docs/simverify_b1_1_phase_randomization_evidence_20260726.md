# SimVerify B1.1 phase-randomization evidence — 2026-07-26

Decision: `revise_condition`

Candidate decision: `condition_understanding_not_established`

Evidence scope: `recorded-observation/offline`

The frozen B1.1 experiment changed one primary factor from B1: deterministic
train-only next-sector randomization for sampled ACT chunks that lie completely
before observable dump-end. Architecture, source episodes, split, optimizer,
loss, seed, training length, validation input, and checkpoint deployment
prohibitions were unchanged.

No held-out episode, Unity/AGX execution, PACT import, Jetson, real hardware, or
closed-loop environment was used.

## Implementation and train provenance

- implementation Git commit:
  `531de9ca88152921b65b215a0524da97dddf0a87`;
- training bundle:
  `/data/pingfan/Excavator_real_stack_data/simverify_b1_1_next_phase_randomized_v1_seed0`;
- status: `completed`;
- best epoch: `1990`;
- best validation loss, diagnostic only: `0.20196416974067688`;
- `policy_best.ckpt` SHA-256:
  `392f5aec11bb25c6e2b8c21b9e65f27f623360ac940311adba34d68cf13950c1`;
- `run_metadata.json` SHA-256:
  `92fd7d3b744dc2eb749e346edfe3cc3ebac1b80b5d045a74b3d7cfc56412208c`;
- embedded baseline: `B1.1`;
- embedded domain: `sim`;
- embedded real-control and Jetson permission: `false`.

The train-only mapping recorded:

| Field | Value |
| --- | ---: |
| train source episodes | 15 |
| accepted train cycles | 111 |
| valid train starts | 31,519 |
| chunk-safe pre-dump randomized starts | 23,496 |
| crossing or post-dump preserved starts | 8,023 |
| changed eligible fraction | 1.0 |
| mapping SHA-256 | `a8e7e2b04b0249be5a8dfbdc6d57a1dda0e678d09b2aa644a039b24dc147310f` |

Exact next-sector marginals were preserved:

| Sector | Source | Randomized |
| --- | ---: | ---: |
| left | 6,528 | 6,528 |
| center | 9,504 | 9,504 |
| right | 7,464 | 7,464 |

Current-sector bits, normalization statistics, all validation conditions, and
all crossing/post-dump train conditions remained unchanged.

## Replay provenance

Each requested replay retained 124 anchors: 45 supported and 79 unsupported.
Unsupported anchors were not included in success denominators.

| Replay | Manifest SHA-256 | Checksums SHA-256 |
| --- | --- | --- |
| requested repeat 0 | `aeaf32ab21a5e60f780e55fb0f4b160d37730a6fa634870b6b8acd056158ffc2` | `af56ebb552c71e1ce342936703d3afd4e9d4960a9b8b9809c38d20907629019f` |
| requested repeat 1 | `17e420c502459abbc930703bf176d4f2a7a8346064386ed05682f6d23de294c9` | `8bbbf0aac6fc1436b650b67def005a54f329f2f3aeb6b2909cfcf031bffbc522` |
| requested repeat 2 | `1cb12f5f327faa85d0d8df39eee348241ada6129f1a8d860c51be0189dadbef0` | `b342e37fb8a4140c5ce89e0930cdc77480e88d4a3dddc0ed68340cddcfa564cd` |
| identical-token mask | `445ee4b01db2ff51674c107a019af4d4413eaefdf5a14e6205d77a2c35efe63d` | `ac9f9b35e38dc5a162fe0e7d326e7d8f77389f788405516a84c464e80e1783ad` |

Every checksum inventory passed. For all 45 supported masked anchors, delivered
base and target tokens were identical, maximum action effect was exactly zero,
and maximum per-tick effect was exactly zero.

## Fixed-observation causal result

The Gate used 100,000 source-episode bootstrap repetitions. Its immutable
artifact is:

`/data/pingfan/Excavator_real_stack_data/simverify_b1_1_condition_causal_v2_validation_v1`

- manifest SHA-256:
  `3740a3c69bc4fb4d934a6f2ac41cff2435b43cc754a5b3bb52dfb6ed2e017cfd`;
- checksums SHA-256:
  `3d975116a6825e9dc8dad2116a560438a703ec7e08dca6b7bd5afc1747a48de2`.

| Criterion | current sector | next sector |
| --- | --- | --- |
| action effect > identical-token mask | pass | pass |
| signed semantic margin > B2 | pass | fail |
| identity beats all five semantic permutations | fail, 3/5 | fail, 3/5 |
| phase specificity | fail | fail |
| task envelope preservation | pass | pass |
| factor decision | **fail** | **fail** |

### Comparison with frozen B1

Values below are means over source-episode aggregates, not frame-weighted
means.

| Factor / metric | B1 | B1.1 |
| --- | ---: | ---: |
| current action effect | 0.030324 | 0.029173 |
| current signed semantic margin | 0.011494 | 0.007250 |
| current phase specificity | 0.010145 | 0.006097 |
| current semantic permutations rejected | 5/5 | 3/5 |
| next action effect | 0.028491 | 0.021576 |
| next signed semantic margin | 0.062237 | 0.034670 |
| next phase specificity | -0.002953 | 0.000652 |
| next semantic permutations rejected | 3/5 | 3/5 |

The intended timing signal moved in the desired direction on average:
next-sector phase specificity changed from `-0.002953` to `0.000652`. Episode
12 changed from `-0.011604` to `0.000825`, and episode 34 improved from
`-0.022368` to `-0.008491`. It was still negative in episode 34, so its
source-episode bootstrap lower bound remained below zero.

This partial timing improvement came with weaker next-sector action and
semantic margins. It also regressed the previously passing current-sector
semantic and phase-vs-B2 evidence. Therefore B1.1 is not a revision candidate
and does not replace B1.

## Interpretation and next bounded experiment

Hard pairing a wrong next token with the recorded pre-dump action was too
blunt for the shared six-dimensional conditioning path. The causal result is
consistent with current and next fields being entangled in the shared
low-dimensional representation: modifying the next-field association also
changed current-field evidence. This mechanism is an inference, not directly
observed internal state.

The next candidate should not repeat hard label randomization. A narrower
B1.2 experiment would keep the true condition on the supervised branch and add
only a pre-dump paired consistency term:

1. run the same chunk-safe pre-dump observation once with the true next token
   and once with a deranged next token;
2. apply expert-action supervision only to the true-token branch;
3. require the two predicted pre-dump action chunks to agree;
4. leave crossing/post-dump samples and current-sector supervision unchanged.

This tests pre-dump invariance without teaching recorded actions under a false
condition. It must be frozen as a new one-factor contract with a train-derived
loss scale before implementation.

## Plain-language conclusion

B1.1 made the model's next-sector response somewhat better timed, but it also
made the meaning weaker and damaged the previously good current-sector
evidence. The fix is therefore rejected. The overall research decision remains
`revise_condition`; held-out and later gates remain locked.
