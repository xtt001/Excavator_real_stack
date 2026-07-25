# SimVerify M3 B1/B2 Condition Ablation Contract

Status: `implementation_and_smokes_passed_formal_training_authorized`

Evidence scope: `recorded-observation/offline`

G3 authorizes B1 and B2 as a matched causal ablation. It does not authorize
held-out access or a closed-loop claim.

## Condition representation

Both models use `cycle_condition_v1` as six low-dimensional inputs appended
after source-domain qpos and qvel:

```text
current-sector one-hot[3] + next-sector one-hot[3]
```

The ACT state dimension is therefore 14. No Transformer condition token is
added in this experiment. That alternative remains forbidden unless the
low-dimensional B1 result is `condition_ignored`.

The structure follows the frozen current-plus-next lookahead decision and the
read-only PACT reference at commit
`9bcb29212b59cc3f788ed6c5046677de26c1ee3b`. No PACT package or ACT adapter is
imported into Real Stack.

## Matched factors

B1 and B2 share:

- the exact B0 source-episode split;
- accepted-row sampling and 20-tick chunks;
- train-only normalization;
- four camera roles;
- qpos, qvel, and six-dimensional condition input;
- ACT architecture, optimizer, epoch count, seed, AMP settings, and losses;
- sim-domain and real/Jetson prohibition.

Their only primary difference is the association between a valid train start
and its condition:

- B1: observed hindsight `cycle_condition_v1`;
- B2: deterministic global permutation over train valid starts.

Validation always receives the observed condition. B2 therefore estimates the
null where the condition channel exists but its training association with the
recorded path has been destroyed.

## Shuffle contract

The B2 permutation:

- uses seed `20260725`;
- covers train valid starts only;
- never uses validation or held-out rows;
- preserves the exact six-vector marginal counts;
- records row count, changed/unchanged count, token counts, and mapping SHA;
- fails if any vector is not two one-hot[3] fields.

The shuffled mapping is generated before DataLoader workers start and is
immutable within a run.

The formal-data import smoke produced:

- valid train starts: `31519`;
- shuffled association mapping SHA-256:
  `cdc2c0fbb83f8d225f3b978a00b3778024b4a8f6f7c06a7a6bb6b28235c6e86f`;
- changed train associations: `27465 / 31519` (`0.8713791681208161`);
- unchanged-by-chance associations: `4054 / 31519`;
- source and shuffled counts equal for all nine current-to-next tokens;
- validation shuffle: disabled.

## Training authorization

Formal training remains blocked until focused tests prove:

1. B1 emits 14-dimensional normalized proprioception;
2. B2 train token marginals are exactly preserved;
3. B2 validation is not shuffled;
4. the same shuffle seed reproduces the same mapping SHA;
5. B1/B2 configs differ only in declared experiment identity, output path, and
   condition-label association;
6. checkpoints embed the correct B1/B2 and sim-domain contracts.

These implementation checks have passed, authorizing only the one-epoch
import/training smokes. Formal 2000-epoch training begins only after both smoke
bundles pass reload and recorded-observation inference.
