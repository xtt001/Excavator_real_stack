# [Execution target G29/H1: Counterfactual command-response feasibility]

Date: 2026-07-12
Decision: **the proposed near-synchronous response assumption is feasible as an
offline upper-bound test, but it does not remove the observed G28 deadlocks**

## [Execution target G29/H1: Hypothesis and boundary]

The test asks a deliberately stronger question than the field system can
currently guarantee:

> If an already-effective target command receives a same-direction response on
> the next tick, would the state-hold deadlock disappear?

The simulator does not rewrite actions, simulate hydraulics, modify source
HDF5, or treat teleoperation qvel heuristics as fault ground truth.  The
empirical response profile is fitted from the formal 19 train episodes only;
the five validation episodes are used only for replay.  Held-out 105..109 is
forbidden and unevaluated.

The focused owner is
`testbed/testbed/policies/counterfactual_response.py`; the CLI is
`testbed/testbed/cli/simulate_counterfactual_response.py`.

## [Execution target G29/H1: Response profile from existing data]

The train-fold response sidecar contains 180 effective command-onset events.
The one-tick response probability is low for most directions, while the
20-tick probability is high:

| axis/direction | events | response at 1 tick | median first response | response by 20 ticks |
| --- | ---: | ---: | ---: | ---: |
| boom+ | 33 | 6.1% | 8 ticks | 93.9% |
| boom- | 26 | 53.8% | 1 tick | 100.0% |
| bucket+ | 44 | 4.5% | 4 ticks | 100.0% |
| bucket- | 39 | 0.0% | 8 ticks | 94.9% |
| swing+ | 19 | 0.0% | 8 ticks | 100.0% |
| swing- | 19 | 0.0% | 4 ticks | 100.0% |

Thus “almost simultaneous” is a useful optimistic stress assumption, not a
description supported by the current response latency distribution.

## [Execution target G29/H1: State-hold counterfactual result]

Validation state-hold has 48 anchors.  For each anchor, the simulator counts
whether the target axis/direction ever crosses its direct mechanical deadzone.
An optimistic plant response can only help when that count is nonzero.

| trace | observed recovered | observed deadlocked | deadlocks with target command present | deadlocks attributable to missing target command | optimistic response gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| G28 raw | 40 | 8 | 0 | 8 | 0 |
| G28 raw + assist | 45 | 3 | 0 | 3 | 0 |
| H2 raw + assist | 45 | 3 | 0 | 3 | 0 |

Every G28 assist deadlock has zero target-effective ticks.  The traces instead
contain a non-target effective command in two anchors (`bucket-`/`bucket+`)
while the target is `boom+`; the remaining deadlocks contain no effective
target command.  Therefore even an instantaneous, perfectly reliable plant
response cannot turn these traces into target progress.  The bottleneck is
command selection/phase intent, not response latency.

An oracle that injects the known target direction would recover all 48 anchors,
but that is not deployable: it uses the expert anchor label and is an upper
bound for a future intent/goal module, not a proposed runtime fallback.

## [Execution target G29/H1: Artifacts and verification]

One-tick optimistic report:

`/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g29_counterfactual_response_sim/counterfactual_response_report.json`

20-tick empirical comparison:

`/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g29_counterfactual_response_sim_h20/counterfactual_response_report.json`

Both runs preserve the same conclusion.  Source sidecar manifest SHA-256 is
`7d367faf43604d3584774609a083f12e9cd38cdd75524f153c72f95d570f72a5`; formal
split SHA-256 is
`09fe85bdab539ca2a12b5b4613f507ea009706cb38077b46e168f5171da59a3d`.

Focused tests: `3 passed`; changed-owner Ruff and `git diff --check` pass.

## [Execution target G29/H1: Feasibility conclusion]

Yes, the requirement can be simulated offline, and the simulation is useful:
it falsifies the idea that faster response alone fixes the current deadlock.
The next useful offline experiment is not a stronger retry governor.  It is a
target-independent goal/intent proposal test that asks whether the model can
choose the missing target axis/direction before any response monitor is
consulted.  The response monitor should remain a confidence/diagnostic layer
until policy-on command/feedback pairs exist.
