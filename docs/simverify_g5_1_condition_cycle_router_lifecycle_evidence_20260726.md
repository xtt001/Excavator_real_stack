# SimVerify G5.1 Condition-Cycle Router Lifecycle Evidence

Date: 2026-07-26

Status: `g5_core_two_cycle_condition_continuity_established_development`

Evidence scope: `recorded-observation/offline teacher-forced development`

Closed-loop execution: `false`

Held-out test read: `false`

## Causal question

G5 core v1 preserved ACT temporal aggregation across the shared ready boundary,
but the monotonic observable condition router was never told that a new cycle
had started. It therefore stayed on the `next` route through nearly all of the
second cycle, and B1.4 failed the second-cycle next-sector semantic Gate.

G5.1 changes one runtime factor: immediately before the shared ready
observation, reset only the condition-cycle router. ACT step count, temporal
aggregation chunks, cached actions, visual history, timestamps, checkpoint
weights, and normalization remain unchanged.

The frozen v1 failure artifact remains immutable and is a verified input to the
G5.1 artifact:

- manifest SHA256:
  `8fb302ab070c80518a8b6c03a7a5558290002ba76ed67e7a725f1e3d397c4b62`;
- Gate SHA256:
  `d5a45c7d1ae94f2a6f4c6b0a5ec472ef9a8ce9f0ea102cfe352a7d6084d5a5c3`;
- checksum-package SHA256:
  `9ebeada6ef7341ca7c5581544be584fe2518f1773577a86bfb56b8de00fa4386`;
- verified files: `93`.

## Implementation verification

Local implementation commit:
`fcc58536d206fa78e9ad7d2d9e13c0ea44aabe2b`.

Focused verification:

- `44 passed`, including G5 replay, observable router, condition/delta stitch,
  transition stitch, and policy action-source tests;
- `ruff check` passed on the new G5 builder, CLI, and focused tests;
- `git diff --check` passed;
- a real B1.4 checkpoint smoke on validation episode 12, cycles 5 -> 6,
  produced:
  - boundary route index `0`;
  - second-cycle route-0 ticks `154`;
  - second-cycle route-2 ticks `28`;
  - condition-cycle router reset count `1`;
  - policy step after replay `557`;
  - temporal aggregation storage still present.

## Formal development artifact

Path:

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_g5_1_two_cycle_router_lifecycle_development_v1
```

Artifact identity:

- Git commit:
  `fcc58536d206fa78e9ad7d2d9e13c0ea44aabe2b`;
- Git dirty: `false`;
- manifest SHA256:
  `abc80a128960c9f719d00d1720fc4aa63811b3b6569707600e5219d44c7d4d8f`;
- Gate SHA256:
  `8e3575025110ecb39178185bfd5d0957240bfd9467eca3302bb63360c5cd8357`;
- checksum-package SHA256:
  `f4f7a07a5b1cc5749f5e723404557e582f6b0ddaa49e66c65cba820c9b9cf02c`;
- checksum verification: `94/94` files passed;
- traces: `21 pairs x 2 baselines x 2 condition modes = 84`.

The support registry contains 13 supported changed-target train pairs and four
supported changed-target validation pairs. The train-derived minimum is one
supported pair per contributing source episode. Validation episode 12
contributes one and episode 34 contributes three. Unsupported counterfactuals
remain in the artifact but do not enter semantic success denominators.

## Gate result by source episode

| Metric | Episode 12 | Episode 34 | Gate |
| --- | ---: | ---: | --- |
| two-cycle phase coverage mean | 0.969697 | 0.983333 | >= 0.85 |
| event-order valid rate | 1.0 | 1.0 | >= 1.0 |
| ready-boundary discontinuity q95 | 0.025583 | 0.024192 | <= 0.151286 |
| minimum second-cycle route-0 ticks | 94 | 110 | >= 1 |
| minimum second-cycle route-2 ticks | 28 | 27 | >= 1 |
| shared-boundary route-0 rate | 1.0 | 1.0 | >= 1.0 |
| B1.4 switch action effect mean | 0.002456 | 0.002287 | > 0 |
| B1.4 route-2 semantic margin | 0.015225 | 0.023286 | > 0 |
| B2.4 route-2 semantic margin | 0.008093 | 0.000037 | null reference |
| B1.4 minus B2.4 semantic margin | 0.007132 | 0.023249 | > 0 |

Every condition-cycle trace reset the router exactly once. Expert validation
continuity and every G5.1 criterion passed.

## Decision and limits

The v1 failure was a real runtime lifecycle defect: ACT temporal memory and the
condition router require different reset ownership. After separating them,
B1.4 responds in the requested second-cycle direction and exceeds the matched
shuffled-condition null in both eligible validation source episodes.

This establishes only a development result on recorded observations. It does
not show physical response, closed-loop digging, real-domain transfer, shadow
safety, or deployability. Validation has already participated in method
development, and the supported semantic comparison contains only four changed
pairs. Held-out test remains locked.

The result authorizes the remaining frozen G5 robustness operands: E04 camera
counterfactuals, E05 state-hold replay, and E06 delay/latest-wins replay. Those
must be completed before any held-out-test threshold package or G6 runtime
equivalence decision.
