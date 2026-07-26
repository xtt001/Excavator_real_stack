# SimVerify G5.1 Condition-Cycle Router Lifecycle Amendment

Status: `method_frozen_before_g5_1_run`

Evidence scope: `recorded-observation/offline teacher-forced development`

Closed-loop execution: `false`

Held-out test read: `false`

The immutable G5 core v1 result remains
`g5_core_two_cycle_condition_continuity_not_established`. Its expert
continuity, B1.4 event coverage/order, ready-boundary action continuity, and
second-cycle route activation passed. Its next-sector semantic margin failed.
The G5.1 builder must verify that artifact's checksum package and record its
manifest and Gate hashes before generating any replacement evidence.

## Diagnosed lifecycle defect

In v1 the causal condition router ended the first cycle at route 2 and was not
notified that the shared ready boundary began a new condition cycle. It
therefore remained at route 2 for essentially the whole second cycle
(`second_cycle_route2_tick_count >= 190`).

This is a lifecycle ownership defect, not a training-factor change. Temporal
aggregation should remain continuous across adjacent cycles, while the
monotonic current -> neutral -> next condition router must start a new causal
route at each observable ready boundary.

## Single runtime change

`ACTAdapter.reset_condition_cycle()` resets only:

- `ObservablePhaseRouter.route` to current (`0`);
- its consecutive-transition dwell counter;
- the last route diagnostic.

It must preserve:

- ACT step index;
- all temporal-aggregation action chunks;
- cached actions and temporal weights;
- visual history;
- temporal timestamps;
- factorized-action state;
- checkpoint weights and normalization.

The G5.1 replay calls this method exactly once immediately before the shared
ready-boundary observation. The full policy is still reset only once at the
start of the two-cycle trace.

Both `switched` and `unchanged` traces receive the same lifecycle reset. Their
only difference remains the delivered condition.

## Counterfactual support correction

G5 v1 evaluated all 13 changed-target pairs, but `unchanged` asks the second
cycle to retain the first cycle's next target. Only 4/13 of those second-cycle
counterfactuals pass the frozen M2 next-sector support rule.

V1 remains unchanged. G5.1 uses the same rule already required by E03:
unsupported counterfactuals remain visible but cannot enter semantic success
denominators.

Train two-cycle pairs generate the minimum support count per contributing source
episode. The observed train minimum is one supported changed-target pair.
Validation source episodes 12 and 34 must each meet that train-derived minimum.

The immutable v1 traces were also rescored on this supported subset and still
failed: B1.4 semantic margin was negative in both source episodes and below
B2.4. Therefore support filtering alone does not explain away the v1 failure.

## Additional lifecycle Gate

In addition to the unchanged G5 core criteria, G5.1 requires:

- route index is 0 at every shared ready boundary;
- the second cycle contains nonzero route-0 and route-2 ticks;
- the route-2 semantic Gate uses only supported changed-target pairs;
- B1.4 supported semantic margin is positive in every eligible source episode;
- B1.4 exceeds B2.4 in every eligible source episode.

Passing authorizes only the remaining recorded-observation G5 robustness
operands. It does not authorize held-out test, closed-loop claims, real
fine-tuning, shadow, control, or deployment.
