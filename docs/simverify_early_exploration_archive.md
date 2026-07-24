# SimVerify early-exploration archive

## Status

This branch is a **historical, non-mainline research archive** based on
`db720c1fb3dcaf5edd7b43d65b58f2219aa48163`.

It exists to preserve the smallest reproducible parts of early SimVerify
exploration without mixing them into `fs/perf_optimize` or presenting them as the
current V2 design. Nothing in this branch is a deployment default, a production
data contract, or evidence of closed-loop task success.

The current SimVerify design source of truth is maintained separately in the
PACT repository on branch `v2.0.0-simVerify`.

## Disposition

| Exploration | Decision | Reason |
| --- | --- | --- |
| Fixed bucket-tip FK v0.2 | Archived | Demonstrates that simulator normalized `qpos` can recover one fixed geometric point. Real sensor alignment and grid extrinsics remain uncalibrated. |
| Observation-only 3x1 dig-sector annotation | Archived | Demonstrates a hindsight `actual_dig_sector` proposal from swing `qpos` and two eye cameras. It does not recover a historical command. |
| Semantic cycle crop | Historical reproducer only | Preserved because earlier E58/E59 reports reference the transformation. It is not part of the current conditioned-cycle design. |
| 3x2 goal-grid annotation pilot | Superseded and omitted | The current first-stage target space is lateral 3x1, not near/far 3x2. |
| Dynamic lowest-measurement-point FK v0.1 | Superseded and omitted | Its target was not a fixed physical point and changed with bucket pose. |
| Preliminary 3x1 pilot generator | Superseded and omitted | The later cross-sensor annotator contains the retained semantics and rejection rules. |
| Local `AGENTS.md` reasoning injection and backup | Omitted | Local agent configuration is not product or research code. |

## Preserved capability boundaries

The archived results support only the following statements:

1. A fixed simulator bucket-tip point can be approximated from simulator
   normalized `qpos` inside the calibrated pose domain.
2. Historical real observations can provide a coarse achieved left/center/right
   sector when swing `qpos` and both eye cameras agree.
3. Historical trajectories without a recorded target cannot provide
   command-following supervision.
4. Simulator privilege may audit a frozen observation-only rule, but it must not
   silently become a training label or policy input.

They do not establish:

- real-machine metric FK;
- physical sandbox thirds;
- soil-contact truth;
- conditioned-policy instruction following;
- closed-loop single- or multi-cycle success.

## Git contents

The branch intentionally keeps:

- focused collection, fitting, audit, annotation and validation code;
- focused unit tests;
- compact reports, manifests, summaries and diagnostic figures;
- the selected small fixed-tip FK artifact;
- the generated 3x1 annotation sidecar used by the report.

Large intermediate samples, rejected model variants and superseded pilot outputs
are not promoted into this branch. The complete pre-cleanup working-tree snapshot
remains recoverable from Git stash commit
`f6e05ecdfdb09374a9cdbc1c8d857e4cf01242f6` with message
`archive: simverify early exploration full snapshot 2026-07-24`.

Do not drop that stash until this archive branch has been reviewed and, if
desired, pushed to a remote.

## Promotion rule

Files from this branch must be cherry-picked by capability, never merged as a
whole. Promotion requires a current design need, a named owner, updated semantics
and fresh validation against the target dataset. Historical passing tests alone
are not sufficient.

## Validation

Run from the repository root:

```bash
git diff --check
PYTHONPATH=testbed pytest -q \
  testbed/tests/test_semantic_cycle_crop.py \
  testbed/tests/test_collect_fixed_tip_fk_samples.py \
  testbed/tests/test_fit_fixed_tip_fk.py \
  testbed/tests/test_audit_fixed_tip_fk_real_transfer.py \
  testbed/tests/test_dig_sector_annotation.py
```
