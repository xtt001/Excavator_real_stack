# SimVerify M0 Observable Export Contract

## Status and scope

This document is the implementation companion to
`docs/v2_0_0_simverify_research_plan.md`. It defines M0 data-contract
materialization and the preconditions for a later M1 import smoke.

Current execution status:

```text
draft_m0_annotation_gate_failed
```

The M0 package is not frozen. M1 is neither implemented nor authorized by
this revision. After removing the invalid `ready_end -> dump_start`
dig-entry fallback, the unchanged 256-sample source-episode bootstrap produced
sector boundary CI-width/minimum-cluster-gap ratios of `0.397689` and
`0.353175`, above the frozen `0.25` limit. The current implementation therefore
stops before visual extraction, export materialization, M1, or training.

The evidence scope is always:

```text
recorded-observation/offline
```

M0 and M1 do not authorize training, simulator execution, real-machine
control, checkpoint transfer, or any closed-loop success claim.

## Source lock

The M0 builder consumes the 24 source episodes in:

```text
/data/pingfan/excavator_testbed_data/
  yulong_v2_2_pro_full_task_four_camera_jpeg_20260717_cycle_clean_v1/
  clean_all_vds/
```

Each small VDS wrapper and every resolved backing HDF5 file are content-hashed.
The raw source file SHA-256 is required; size and mtime alone are not accepted
as source identity.  Source HDF5 files are opened read-only and are checked for
size/mtime changes after materialization.

The source numeric contract is:

- `qpos`: source-native normalized positions in
  `swing, boom, stick, bucket` order;
- `qvel`: source-native recorded speeds in the same order; no SI unit is
  inferred because the embedded metadata does not provide one;
- `action`: post-deadzone/post-response-profile `actuator_speed_cmd` in the
  same axis order;
- time: `timestamps/step_id * metadata.dt`, with `dt=0.02 s`.

No sim-to-real numerical mapping is performed.

The action contract also records and verifies, for every source episode, the
embedded joystick axis map/inversion, scale, symmetric limit, deadzone, and
response-profile enabled/attack/release/recenter/exponent/use-measured-dt
fields. Only this allowlisted action-generation subset is copied into
provenance; the full source config is not a policy input.

## Camera and pixel transform

The physical-role mapping is frozen as:

```text
eye_left  -> video4
eye_right -> video5
stick_down -> video6
stick_up   -> video7
```

The policy order is:

```text
video4, video5, video6, video7
```

All source JPEGs decode to 512x288.  M0 applies:

```text
JPEG decode -> BGR-to-RGB -> no crop
  -> linear resize 512x288 to 384x216 -> JPEG storage
```

This aligns pixel dimensions and color semantics with the current Real Stack
policy input.  It does not claim equal intrinsics, extrinsics, field of view,
projection, or scene distribution.

## Sim-time 50 Hz to 20 Hz selection

For target tick `k`, the exporter selects the first complete source row whose
sim time is not earlier than `k / 20`.

The selected source index is shared by:

- all four images;
- qpos;
- qvel;
- action;
- condition.

The action-label offset is exactly zero.  `timestamps/step_ns` is neither used
nor exported.  Each output episode records target tick, source row, source
step, source sim time, target sim time, and non-negative selection error.

Full-data QC recomputes per-axis non-zero action-sign segments using the
source deadzone.  It reports preservation, missed-segment duration, and onset
delay, including a separate assertion for segments lasting at least 50 ms.

## Privilege isolation

The exporter is allowlist-based.  It never copies source datasets and then
tries to remove privilege afterwards.

Policy-bound HDF5 data may contain only:

```text
observations/qpos
observations/qvel
observations/encoded_images/video4
observations/encoded_images/video5
observations/encoded_images/video6
observations/encoded_images/video7
action
timestamps/step_id
timestamps/sim_time_s
conditions/cycle_condition_v1
conditions/cycle_id
conditions/valid_mask
diagnostics/source_observation_index
diagnostics/source_action_index
diagnostics/source_step_id
diagnostics/source_sim_time_s
diagnostics/target_tick
diagnostics/target_sim_time_s
diagnostics/selection_error_s
```

The main export rejects virtual datasets, external links, unknown groups, and
unknown dataset attributes.  In particular, it excludes:

- `observations/env_state`;
- `rewards`;
- all source `v2/**` labels and tokens;
- bucket mass, exact tip coordinates, terrain grids, contact state, planner
  state, oracle fields, and future actions.

An optional `oracle_audit/` is generated only after observable thresholds and
annotations are frozen.  It has independent checksums and is not referenced by
the importer or the main package checksum list.  Removing it cannot change
training inputs, main evaluation inputs, or M1 results.

## Observable annotation

Numeric candidates are generated from qpos, qvel, and action. The intended
frozen local ImageNet ResNet-18 stage then provides independent visual
confirmation:

- eye-pair features cross-confirm left/center/right at this cycle's dig entry
  and the immediately adjacent next cycle's dig entry;
- eye/stick features confirm that numeric event candidates remain inside the
  corresponding train/validation visual support envelope;
- qpos and eye evidence must agree for an automatic sector label;
- disagreement, boundary samples, missing visual evidence, or invalid event
  order enter the review queue.

The last candidate in an episode has no observable next dig entry and is
therefore review-only; ready-end pixels are not used to invent a hindsight
next-sector token. Cross-event nearest-centroid accuracy and an
episode-mapping permutation null are retained as visual identifiability
evidence. Event acceptance itself is a support-envelope check because numeric
signals, not a visual event classifier, generate the candidate and event
order.

The numeric labeler currently emits candidate intervals, numeric
representative source rows, and reason codes for:

- `ready_start`;
- `dig_entry_proxy`;
- `carry_transition_proxy`;
- `dump_start_proxy`;
- `dump_end_proxy`;
- `ready_end`.

`ready` is an empirical observable envelope and may contain non-zero qvel.
These proxies do not claim soil contact, payload, or removed-volume truth.

The interval-wide selector is a separate, fail-closed stage:

- it extracts one-row visual halos around every numeric half-open interval;
- `annotation_feature_input_manifest_v1.json` freezes the deduplicated source
  rows used for extraction, including per-episode half-open row ranges, the
  SHA-256 of the ordered little-endian int64 row list, eye/stick
  dimension/dtype/normalization, decode chunk size, configured inference batch
  size, and complete extractor provenance;
- train numeric representatives fit episode-balanced eye/stick prototypes;
- validation numeric representatives freeze own-prototype support and
  event-specific change/stability envelopes;
- validation interval matches freeze signed p02.5/p97.5 offset bounds;
- top-1 and cross-event margin are retained as identifiability diagnostics,
  not Gate operands, while event acceptance remains an own-prototype
  support-envelope check;
- `ready` minimizes two-sided stick motion, dig/dump-start use entering
  change, carry uses centered change, and dump-end uses exiting change;
- offset eligibility is applied before candidate ranking;
- a shared ready gap is selected once and becomes both `ready_end(i)` and
  `ready_start(i+1)`;
- confirmed ready boundaries rebuild `source_steps`, after which the complete
  event order is checked again.

Eye and stick metrics remain separate in the applied sidecar. Eye is required
for dig/dump global-scene confirmation and later sector cross-checking; stick
is required for every local phase. Evidence conflict or missing required halo,
support, change, offset, or event order produces `ambiguous`.

Per-event confidence is an empirical support score, not a probability. It is
the minimum of required-role support percentile, event change/stability
percentile, signed-offset centrality, and add-one-smoothed source-episode
reselection support. An accepted event therefore cannot have zero confidence.

Point-event stability is evaluated over the complete source-episode outer
bootstrap. A point-confirmed event passes the stability mask only when:

- its confirmation frequency is at least `0.99`;
- its reselection-within-point-tolerance frequency is at least `0.99`; and
- the p02.5-to-p97.5 width of its selected source-row distribution is no wider
  than that phase's frozen signed-offset span.

An event that fails this point rule becomes `ambiguous`; this does not by
itself fail the complete labeler. For each event phase, the aggregate fraction
of point events that pass must be no lower than the p02.5 lower bound of that
phase's validation-coverage outer bootstrap. The selector Gate also requires
the validation-coverage bootstrap lower bound to exceed the episode-mapping
permutation-null p95, offset-endpoint stability, and an outer-bootstrap failure
rate no greater than `0.01`. Cross-event top-1 accuracy and margin remain
reported diagnostics only.

Current-sector evidence has a role-local order rule. It requires a confirmed
`dig_entry_proxy`, requires any available `ready_start` not to follow that dig,
and requires any available carry/dump/ready-end event not to precede that dig.
Missing unrelated ready or dump evidence can still make the complete cycle
review-only, but it does not erase an otherwise valid current-sector
observation. Complete six-event order remains required for accepted cycle
`source_steps` and condition materialization.

The sector bootstrap is an outer source-episode bootstrap conditional on the
separately frozen numeric candidate intervals. Every replicate refits the
episode-balanced event prototypes, validation support/change/offset
envelopes, shared event representatives, event order, and then the train
sector clusters. Resampling already-selected qpos rows is not sufficient.
After point-event stability is assessed, the point-stability mask is frozen
and reapplied to every retained event in each selector-successful replicate.
The role-local current-sector order is recomputed after that complete mask, so
an unstable non-dig event cannot irreversibly discard or incorrectly retain a
dig row. Source-episode draw multiplicity is preserved. The sector clusters
and boundary distribution are then recomputed from the resulting stable,
order-valid dig rows. The pre-mask sector summary remains in the selector
artifact as a diagnostic, while
`annotation_event_selected_sector_bootstrap_v1.json` records the masked
distribution and hashes both the selector core and the point-selection/
stability-mask artifact. That stability-masked distribution is the operand
used by the sector Gate.
Numeric-candidate threshold bootstrap remains the independent preceding Gate.

Repeated bucket-release pulses are merged only while every intervening swing
sample remains inside the train-fitted dump cluster. No release-gap duration
threshold is fitted: episode bootstrap did not identify a stable gap boundary,
and leaving the observable dump cluster is already the cycle separator.

Historical commands remain:

```text
unknown_not_recorded
```

Accepted policy conditions are explicitly:

```text
hindsight_outcome
```

## Condition schema

`cycle_condition_v1` is a float32 vector of length six:

```text
[current_left, current_center, current_right,
 next_left, next_center, next_right]
```

Each valid half is one-hot.  It is not normalized, is constant within a cycle,
and changes only at a shared observable ready boundary.  Invalid/review rows
have `valid_mask=0`, `cycle_id=-1`, and an all-zero vector; they are not model
samples.  There is no `next_valid` model input.

The first conditioned implementation, if M3 and M4 are later authorized,
concatenates this named field after qpos and qvel and uses the existing robot
state projection.  A separate Transformer condition token is not part of M0.

## Episode split and held-out lock

The split unit is the complete source episode.  Assignment is deterministic,
SHA-ranked, and stratified by the recorded controller epoch.  Train,
validation, and held-out test are disjoint.

Numeric and visual annotation thresholds use train data and validation
calibration only. Held-out episodes may undergo the parameter-free export,
checksum, and structural 20 Hz QC needed for M0, but their observable labels,
features, conditions, oracle fields, and transition outcomes remain unread.
Those fields stay locked until a finite `gate_thresholds_v1.json` and its SHA
exist.

The split artifact reports episode IDs for all splits. Train and validation
report:

- episode IDs;
- accepted cycle count;
- sector counts;
- all current-to-next transition counts;
- adjacent two-cycle pairs;
- all three-cycle combinations;
- continuity errors.

The held-out entry is explicitly `locked_unread` with null counts and is
excluded from the M0/M1 success denominator. Missing combinations remain
explicit and are never silently included in a success denominator.

## Gate-threshold method freeze

M0 writes `gate_thresholds_contract_v1.json`, not a fabricated numeric
`gate_thresholds_v1.json`.

All G3/G4/G5 model metrics have:

```text
status=deferred
value=null
```

Each entry names the required B0/B1/B2 validation artifacts, estimator,
direction, and episode-level paired-bootstrap method.  The held-out test stays
unauthorized.  Only after the required model replay artifacts exist may a
separate, finite, SHA-bound `gate_thresholds_v1.json` be produced and the test
lock opened.

## M0 artifact layout

```text
sim_observable_cycle_v1/
  episodes/episode_*.hdf5
  dataset_manifest.json
  source_episode_manifest.json
  export_field_contract.json
  camera_mapping.json
  state_action_contract.json
  cycle_condition_v1.schema.json
  annotation_thresholds_v1.json
  annotation_feature_input_manifest_v1.json
  annotation_feature_prototypes_v1.npz
  annotation_event_selector_prototypes_v1.npz
  annotation_event_selector_v1.json
  annotation_event_selected_sector_bootstrap_v1.json
  annotation_event_selector_gate_report.json
  annotation_event_selections_pre_gate_v1.json
  annotation_event_selections_v1.json
  annotation_manifest.json
  cycle_annotations.jsonl
  review_queue.jsonl
  split_groups.json
  transition_inventory.json
  condition_support_index.json
  resample_20hz_qc.json
  privilege_scan_contract_v1.json
  privilege_scan_report.json
  gate_thresholds_contract_v1.json
  checksums.sha256
  oracle_audit/
```

Large HDF5 and feature artifacts stay outside Git.

## M1 import smoke

M1 opens one train and one validation episode from the materialized package
and hashes only those bounded inputs plus their manifest, split, and sidecar.
It does not import PACT, start AGX/Unity, call ACT, read source paths, or read
`oracle_audit/`.

The smoke fails closed on:

- checksum mismatch;
- missing or reordered camera;
- transform ID pending or mismatched;
- wrong image shape/color;
- non-finite or wrong-shaped state/action;
- use of wall-clock time or a non-zero action offset;
- observation/action source-index mismatch;
- condition schema, one-hot, mask, cycle-constancy, or sidecar mismatch;
- split leakage;
- virtual/external HDF5 dependency;
- any privilege path.

Passing M1 proves only that the recorded-observation package can be consumed by
the Real Stack import boundary.
