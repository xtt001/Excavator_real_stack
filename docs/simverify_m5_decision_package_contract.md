# SimVerify M5 decision-package contract

Status: frozen before implementation.

Evidence scope: `recorded-observation/offline`

## Purpose

The M5 package closes the current frozen SimVerify experiment version by
binding its terminal decision to the immutable M0 through G4 evidence chain.
It does not run a new model experiment, read held-out episodes, enter G5/G6,
or promote a checkpoint.

## Required inputs

- immutable M0 export and checksum inventory;
- passing M1 import-smoke report;
- immutable M2 evaluator contract and authorization report;
- passing G3 B0 calibration package;
- fixed-observation causal v2 packages for B1, rejected B1.1, and rejected
  B1.2.

Every package checksum inventory and every replay package referenced by G3 or
G4 must verify. M0 and M2 manifest SHA values must match the values embedded in
the replay packages.

## Frozen decision rule

The dependency chain is:

```text
G0 -> G1 -> G2 -> M1 -> M2 -> G3 -> G4 -> G5 -> G6 -> M5
```

For the current version:

- G0/G1/G2, M1, M2, and G3 passed;
- B1, B1.1, and B1.2 did not establish both current- and next-sector causal
  understanding under the frozen G4 criteria;
- therefore G4 returns `revise_condition`;
- G5, G6, held-out test, and deployment remain not entered and locked;
- M5 terminal decision is exactly `revise_condition`.

The package must reject any input claiming held-out access, closed-loop
execution, real-control permission, a control candidate, or a generated global
threshold file after the failed G4 path.

## Outputs

```text
simverify_m5_decision_v2/
  decision.json
  m5_manifest.json
  checksums.sha256
```

`decision.json` records the complete Gate path, terminal reason, promotion
locks, and compact B1/B1.1/B1.2 source-episode summaries.
`m5_manifest.json` records Git provenance and every input package identity.

## Proof boundary

The package proves only that the current recorded-observation/offline
experiment has a reproducible terminal evidence decision. It does not prove
simulator closed-loop completion, real-machine performance, real-domain
transfer, or deployability.
