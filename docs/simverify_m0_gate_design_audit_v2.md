# SimVerify M0 Gate Design Audit v2

Status: `predeclared_before_v2_rerun`

Evidence scope: `recorded-observation/offline`

This audit replaces the unsupported numerical choices in the failed M0
annotation Gate with decision-relevant operands. It does not read held-out
annotations, authorize training, claim closed-loop execution, or change the
observable-only research boundary.

## 1. Audit inputs and lock

The failed v1 run remains immutable:

- Git commit:
  `7f7e5d9e3bfa2fc9e3b1c127d64749d61b789082`;
- source snapshot SHA-256:
  `0beb933f5e03a8cc15442251769ad8fb4fca7579ae333fc5924e158a0fe2b61b`;
- split SHA-256:
  `d0df42f6a65623daeaef0b986d453532a4f49806cadb2cf56c50c2dd88105175`;
- failed event-selector Gate SHA-256:
  `1f306ae51b3cbaa08737cda5346a3a1bc20806d356f8e2e68ed325b32de11962`;
- split: 16 train, 4 validation, and locked held-out episodes
  `[1, 13, 25, 33]`;
- held-out observable labels, features, conditions, transition outcomes, and
  oracle fields remain unread.

The v2 rerun must use the same split seed, source bytes, ResNet-18 SHA, camera
contract, source-domain state/action contract, and bootstrap seed. It writes a
new immutable output directory and never overwrites the failed v1 temporary
artifact.

## 2. Gate design rules

Every Gate must satisfy all of the following:

1. **Direct consequence**: failure must name a concrete contract violation,
   not merely a large or small statistic.
2. **Matched unit**: episode-level claims use source-episode resampling; no
   frame-level pseudo-replication.
3. **Matched null**: a null must destroy the relationship being claimed while
   preserving the relevant source-episode structure.
4. **Decision sensitivity**: parameter uncertainty is a Gate only when it can
   change a label, consume a cluster, break event order, or remove required
   condition support.
5. **Review is allowed**: G2 permits uncertain events to enter review/exclude.
   A few uncertain points do not automatically invalidate the complete
   labeler.
6. **No held-out tuning**: all thresholds and Gate methods are frozen from
   train/validation before held-out annotation or evaluation.
7. **Capability boundary**: passing M0 proves only that a reproducible
   recorded-observation package can be built. It does not prove policy
   learning, closed-loop success, sim-to-real transfer, or control safety.

The formal v2 run uses 1024 bootstrap/null replicates. This is the smallest
power of two above 1000, so a p02.5 tail contains about 25 samples instead of
the roughly 6 samples provided by the failed 256-replicate run. The fixed seed
is retained for reproducibility.

## 3. M0 Gate catalogue and design reasons

### M0-PROV-01 — target and source identity

**Reason**

An annotation or export is not reproducible if the branch, Git SHA, dirty
state, source VDS backing files, camera model, or feature checkpoint changes
during the run.

**Operand and pass rule**

- required branch and clean Git state;
- frozen baseline tag identity;
- complete source-chain and inventory SHA-256 before observation reads;
- unchanged source-chain SHA-256 before finalization;
- frozen ResNet-18 SHA-256.

Any mismatch fails immediately. There is no percentage threshold.

**Cannot prove**

That the source behavior is good or that its labels are correct.

### M0-SPLIT-01 — episode isolation and held-out lock

**Reason**

Cycles and adjacent frames from one source episode are correlated. Splitting
them across train/validation/test would leak background, controller epoch, and
trajectory state.

**Operand and pass rule**

- complete source episode is the split unit;
- train, validation, and held-out sets are disjoint and exhaustive;
- held-out annotation/feature/condition/transition access count is zero;
- held-out may undergo only parameter-free export and checksum QC.

**Cannot prove**

Cross-session or real-domain generalization.

### M0-NUM-01 — numeric estimator computability

**Reason**

If a source-episode resample cannot fit the observable dump clusters, the
numeric candidate method depends on a lucky episode composition.

**Operand and pass rule**

Every requested source-episode bootstrap refit must complete. v1's
`failure_rate <= 0.01` is retired because it silently conditions later
statistics on successful refits and its `0.01` was not derived from the
contract.

**Cannot prove**

Physical dumping, payload release, or soil contact.

### M0-NUM-02 — dump-boundary identifiability

**Reason**

The fitted boundary is usable only if its source-episode uncertainty remains
between the two adjacent observable cluster centers.

**Operand and pass rule**

For the lower and upper 95% bootstrap bounds:

```text
left-center p97.5
  < boundary p02.5
  <= boundary p97.5
  < right-center p02.5
```

This directly prevents boundary uncertainty from reaching or inverting a
cluster. The v1 rule `boundary CI width < 25% of cluster gap` is diagnostic
only: 25% was an engineering constant, and a wider CI can still lie cleanly
between the centers.

**Cannot prove**

That a sample near the boundary has a unique label; those samples enter the
review band.

### M0-EVT-01 — visual-role identifiability

**Reason**

Frozen eye/stick features must carry information about their numeric-anchor
event roles. Otherwise “visual confirmation” is only a decorative sidecar.

**Operand and pass rule**

For eye and stick separately:

```text
source-episode bootstrap p02.5 balanced accuracy
  > episode-mapping permutation-null p95 balanced accuracy
```

Balanced accuracy is used because event counts differ by phase. Eye is
evaluated only on its required dig/dump roles; stick is evaluated on all
required local phases.

**Cannot prove**

That the visual role equals physical contact or that exact event frames are
unique.

### M0-EVT-02 — interval-confirmation reliability

**Reason**

The policy dataset needs reliable cycle intervals and condition transitions,
not an identical visual argmin/argmax pixel row in every bootstrap refit.
G2 explicitly allows uncertain events to be reviewed or excluded.

**Operand and pass rule**

For every base-confirmed event, record its source-episode bootstrap interval
confirmation frequency. For a cycle, take the minimum over its six required
events. Choose:

```text
the maximum cycle-minimum confirmation frequency
that preserves all 9 current->next transitions
in both train and validation
```

The selected frequency must also have a one-sided 95% Wilson lower bound
strictly above 0.5. The `0.5` boundary means that the empirical source-episode
population supports the interval more often than it rejects it; the actual
threshold is maximized from the calibration data rather than fixed at 0.5.

On the immutable failed-run operands, this method selects:

```text
224 / 256 = 0.875
one-sided 95% Wilson lower bound ~= 0.837
```

At that threshold the read-only audit retains:

| split | cycles | adjacent pairs | nonzero transitions |
| --- | ---: | ---: | ---: |
| train | 222 | 150 | 9 / 9 |
| validation | 50 | 38 | 9 / 9 |

The formal 1024-replicate v2 rerun must recompute these values; the 256-run
numbers are evidence for the design, not frozen v2 outputs.

**Cannot prove**

That every retained transition has enough examples for every later
counterfactual. `condition_support_index.json` still filters unsupported
anchors.

### M0-EVT-03 — representative ownership

**Reason**

READY is an observable interval/envelope. Long, flat envelopes can contain
many equally valid low-motion frames, so an exact visual point can move while
the interval remains valid.

**Operand and pass rule**

- numeric qpos/qvel/action candidate logic owns event type, interval, and the
  deterministic representative source row;
- eye/stick evidence confirms the interval;
- visual-selected point, signed offset, and point-reselection CI remain
  diagnostics in the sidecar;
- downstream `source_steps` use the numeric observable anchor.

There is no exact-point Gate. The v1 `0.99` point-reselection and
offset-endpoint-width Gates are retired as promotion operands.

**Cannot prove**

Visually localized contact timing. No physical contact claim is made.

### M0-SECTOR-01 — sector estimator computability

**Reason**

Every retained source-episode refit must still contain three identifiable
swing-qpos clusters after the interval-confirmation mask.

**Operand and pass rule**

All requested masked sector refits must complete. Any failure stops M0.

### M0-SECTOR-02 — sector-boundary identifiability

**Reason**

The two 3x1 boundaries must stay between their adjacent center uncertainty
bands; otherwise resampling can consume or invert a sector.

**Operand and pass rule**

Apply the same center-CI separation inequality as M0-NUM-02 to both sector
boundaries. Boundary CI half-width becomes the review margin. Samples inside
the review band are excluded rather than forced into a sector.

The failed run's masked-bootstrap boundary CIs remained between adjacent
center CIs, even though one old width/gap ratio exceeded 25%; this is why the
decision-relevant rule replaces the ratio.

### M0-SECTOR-03 — eye/qpos sector cross-confirmation

**Reason**

The hindsight condition must not be derived only from swing qpos. Eye-pair
features provide the required independent observable evidence.

**Operand and pass rule**

- qpos and eye label must agree for an accepted cycle;
- source-episode bootstrap p02.5 eye balanced accuracy must exceed the
  episode-mapping permutation-null p95;
- ambiguous, boundary, or disagreeing rows enter review/exclude.

**Cannot prove**

Real-camera generalization or sim-to-real geometric equivalence.

### M0-COV-01 — retained condition support

**Reason**

A labeler can be statistically stable yet delete the transitions needed to
train or select a conditioned model.

**Operand and pass rule**

- train and validation each retain all 9 current->next transitions after the
  interval mask and eye/qpos fusion;
- counts are reported per transition;
- retained adjacent cycles have no current/next condition continuity error;
- later token counterfactuals additionally require leave-source-episode-out
  nearest-neighbor support;
- held-out counts remain `locked_unread`.

If final fusion removes a transition that the provisional calibration
preserved, M0 fails rather than lowering the reliability threshold after the
fact.

### M0-EXPORT-01 — sim-time 20 Hz integrity

**Reason**

Training is invalid if images, state, action, and condition come from different
source rows or if wall-clock time is mistaken for simulator time.

**Operand and pass rule**

- target time is `step_id * dt`;
- image/qpos/qvel/action/condition use one source row;
- action offset is zero;
- source indices are monotonic and selection error is finite;
- every source episode is processed;
- transition-preservation QC reports all active segments and explicitly lists
  missed segments;
- no sustained segment of at least one 20 Hz period may be silently missed.

Shorter missed segments remain reported because a 20 Hz target grid cannot
guarantee representation of every sub-period pulse.

### M0-PRIV-01 — privilege isolation

**Reason**

The annotation and exported policy inputs must remain available on a future
real machine.

**Operand and pass rule**

Every output episode and manifest is scanned for forbidden paths/fields.
Oracle artifacts are written only after main checksums and are absent from the
main artifact graph. Any violation fails M0.

### M0-AUTH-01 — M1 authorization

**Reason**

No individual statistical Gate is sufficient. M1 may consume a package only
after the complete contract is materialized and hash-bound.

**Operand and pass rule**

M1 is authorized only when M0-PROV, SPLIT, NUM, EVT, SECTOR, COV, EXPORT, and
PRIV all pass; the final dataset manifest and checksum inventory exist; and
the immutable output rename succeeds. The in-package authorization report
records passed Gate preconditions conditionally; authorization becomes
effective only after the manifest/checksum write and immutable rename.

Passing M0-AUTH-01 still leaves `training_authorized=false`.

## 4. Retired v1 operands

| v1 operand | v2 disposition | Reason |
| --- | --- | --- |
| phase coverage p02.5 > wrong-prototype coverage-null p95 | diagnostic | The null permuted prototype identity but preserved the real numeric interval and event-specific change rule; it did not isolate temporal visual confirmation. |
| every point confirmation/reselection >= 0.99 | replaced | `0.99` was not generated from the support requirement and conflated interval validity with exact-frame uniqueness. |
| stable-point fraction >= aggregate validation coverage p02.5 | removed | It compared the fraction of items passing an extreme per-item threshold with an aggregate mean coverage statistic; the operands estimate different quantities. |
| offset-endpoint CI width < 25% of median interval | diagnostic | Search-envelope parameter width is not itself a label error; interval consensus measures the downstream effect. |
| numeric/sector boundary CI width < 25% of cluster gap | replaced | Center-CI separation directly tests whether boundary uncertainty can change cluster identity. |
| bootstrap failure rate <= 1% | replaced | Build-critical refits must all be computable; successful-only summaries otherwise hide failure conditioning. |

## 5. Rerun and promotion sequence

1. Run focused tests and `git diff --check`.
2. Freeze this audit, Gate v2 code, schema, and Git SHA in a small commit.
3. Build `/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v2`
   from source bytes with 1024 source-episode replicates.
4. Verify every M0 Gate and artifact checksum without reading held-out
   annotations or evaluation fields.
5. If any Gate fails, stop with `revise_annotation_contract` and preserve the
   failed temporary artifact.
6. If and only if the complete immutable M0 package exists, run the bounded M1
   import smoke on one train and one validation episode.
7. M1 proves importability only. It does not authorize training or any
   closed-loop claim.
