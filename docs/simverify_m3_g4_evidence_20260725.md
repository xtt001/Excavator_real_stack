# SimVerify M3 G4 Condition Evidence — 2026-07-25

Status: `terminal_revise_condition`

Evidence scope: `recorded-observation/offline`

Closed-loop execution: `false`

Held-out test read: `false`

Global `gate_thresholds_v1.json` generated: `false`

## Immutable evidence package

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_g4_condition_calibration_v1/
```

- manifest SHA-256:
  `884f4796698e3f7519f80bd45fc21b99ad3ea7a38fbd7e47c0184dd4e4409679`
- checksum inventory SHA-256:
  `dedf0b4fb056c6f429c9090471d7840088cb9c1a14dcde6bd74a8a2b3a87ec0e`
- independent checksum verification: pass
- implementation Git SHA:
  `d0f345bab0b150773df08540b2b555a2400a7278`
- bootstrap unit: source episode
- bootstrap repetitions: `100000`
- bootstrap seed: `20260725`

The package binds the G3 manifest, M0 and M2 manifests, the immutable M0
threshold-method contract, all three B1 replay manifests/checksum inventories,
the B2 replay manifest/checksum inventory, and both checkpoint SHA-256 values.

## Gate-method audit

The M0 threshold contract remains unchanged and is bound by SHA. G4 records two
method corrections before any held-out read:

1. a paired `B1-B2` confidence interval must not add the absolute B2 null a
   second time;
2. `condition_ignored_rate` repeat uncertainty must be measured in rate units,
   not action-magnitude units.

The corrected action/direction test is:

```text
paired_bootstrap_CI95(B1 - B2).lower
  > B1_same_checkpoint_repeat_metric_noise_q97.5
```

The corrected ignored-rate test is:

```text
paired_bootstrap_CI95(B1_rate - B2_rate).upper
  < -B1_same_checkpoint_repeat_rate_noise_q97.5
```

These are algebraic and dimensional corrections, not thresholds selected from
the observed outcome. B2 remains the null operand in the paired contrast.

## Support

| Factor | Supported anchors | Source episodes | Minimum passed |
| --- | ---: | --- | --- |
| current sector | 23 | 12, 34 | yes |
| next sector | 22 | 12, 20, 34 | yes |

The other 79 anchors remain visible as unsupported counterfactuals and are not
counted as successes or failures.

## Result by criterion

| Criterion | Current | Next |
| --- | --- | --- |
| B1 action effect above paired B2 null plus repeat noise | pass | pass |
| target-direction advantage over B2 | pass | fail |
| response latency inside expert plus repeat-jitter envelope | pass | pass |
| phase coverage/order preservation | pass | pass |
| lower condition-ignored rate than B2 | fail | fail |

### Action effect

Current-sector source-episode B1-minus-B2 effects were:

```text
episode 12: 0.0140120105
episode 34: 0.0259939409
```

The paired-bootstrap lower endpoint was `0.0140120105`; the repeat-noise
margin was `0.0000003064`.

Next-sector source-episode effects were:

```text
episode 12: 0.0157420408
episode 20: 0.0322513562
episode 34: 0.0068286473
```

The paired-bootstrap lower endpoint was `0.0068286473`; the repeat-noise
margin was `0.0000014624`.

These results show a stable token-dependent action change beyond the shuffled
null. They do not prove physical task success.

### Direction

Current-sector direction passed: the paired B1-minus-B2 lower endpoint was
`0.4666666667`.

Next-sector direction did not pass. B1 was direction-correct on all 22
supported anchors, but B2 direction accuracy by source episode was:

```text
episode 12: 0.875
episode 20: 1.000
episode 34: 1.000
```

Consequently the paired lower endpoint was `0.0`, not strictly above the zero
repeat-noise margin. The observed direction metric cannot distinguish the
next-sector response from the shuffled-condition null.

### Condition ignored

The measured action-effect noise floor was `0.0000014615`. At that floor,
both B1 and B2 had zero ignored-anchor rate for current and next sectors. This
does not mean B1 ignored condition. It means shuffled-condition B2 also reacts
to token swaps and therefore does not instantiate an ignored-condition null
for this criterion.

### Repeat stability

- future-runtime-safe/temporal-aggregation maximum repeat delta:
  `0.0001110882`
- raw direct chunk maximum repeat delta: `0.0006095823`
- raw normalized chunk maximum repeat delta: `0.0010507479`
- same-token validation consistency source-episode values:
  `0.9999944955`, `0.9999918147`, `0.9999934346`
- generated validation lower bound: `0.9999918957`

Raw normalized policy chunks, direct-unit chunks, temporal aggregation
actions, and future runtime-safe actions remain separate artifacts.

## Decision

G4 returns `revise_condition`.

The source data did not fail an arbitrary target percentage. The supported
condition interventions produced repeat-stable action effects, and task phase
semantics were preserved. The blocking evidence is narrower: next-sector
target direction is not identifiable against the current B2 null, and B2 is
not an ignored-condition control under the ignored-rate definition.

Therefore:

- G5 is not entered;
- G6 is not entered;
- held-out episodes `1, 13, 25, 33` remain unread;
- no global `gate_thresholds_v1.json` is frozen;
- no sim, real, shadow, or control candidate is promoted;
- the independent Transformer condition-token experiment is not authorized,
  because B1 did not exhibit absolute `condition_ignored`; the failure is
  null/next-direction identifiability.
