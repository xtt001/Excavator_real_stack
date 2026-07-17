# G49 train-reference task-sequence compatibility

Date: 2026-07-16

## Decision question

Do the saved N0--N5 outputs form a task-shaped action sequence supported by the
expert training cohort, even when they do not reproduce one held-out expert
trajectory tick by tick?

This is not a task-success test. It is a teacher-forced empirical compatibility
test designed to avoid treating one validation demonstration as the unique
correct action sequence.

## Frozen inputs

- training reference: 120 expert episodes from
  `g48_new_trainval_view_v1`;
- calibration and model evaluation: the chronological 20 validation episodes;
- model outputs: existing `n0_open_loop_val20` through
  `n5_open_loop_val20` action artifacts; no inference was rerun;
- event contract: `single_demo_intent_events_v2_h40_train120_val20`;
- action domain: direct policy output;
- asymmetric deadzone source:
  `n5_state_hold_raw_val20_h20/resolved_direct_output_deadzone.json`;
- sealed test data read: false.

Formal report:

`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/task_sequence_compatibility_v1_n0_n5_train120_val20/task_sequence_compatibility_report.json`

Report SHA-256:
`e80421f4999e07a83e3fe7564973cebd33637756ff68ff7f88ab5f80131dbb5a`.

## Sequence semantics

The evaluator converts each trajectory into grouped per-axis semantic direction
changes after asymmetric deadzone thresholding.

- Leading recording idle is ignored. A policy may activate at step 0 without a
  sequence penalty.
- Exact amplitude and timing are ignored.
- Returning to idle and re-entering the same direction is collapsed, so
  deadzone chatter is not counted as a new task phase.
- A token is emitted for the first effective direction of an axis and whenever
  that axis later changes sign.
- Joint directions starting on the same tick remain one set-valued token.
- Generalized edit similarity uses Jaccard substitution cost and compares each
  validation trajectory with its closest one of 120 training expert motifs.

The 20 validation expert trajectories provide the calibration distribution.
Two counterfactual controls verify that the metric is sensitive to sequence:
the same expert motifs in reverse order, and motifs collapsed to only their
first event.

Directions appearing in at least 90% of training episodes form the descriptive
core repertoire. In this dataset all eight axis/direction labels satisfy that
criterion. Core coverage is reported independently from ordering similarity so
a short but well-ordered subset cannot masquerade as a complete task sequence.

## Calibration

| Trajectory cohort | Nearest-train mean | Median | Q25 | Semantic events median |
| --- | ---: | ---: | ---: | ---: |
| Held-out validation experts | 0.824 | 0.833 | 0.769 | 11 |
| Reversed validation experts | 0.462 | 0.462 | 0.423 | 11 |
| First-event-only collapse | 0.125 | 0.125 | 0.125 | 1 |

The separation shows that this is not merely a first-action or direction-set
membership score. It rewards expert-like progression order and penalizes a
controller that only starts once.

## N0--N5 results

| Model | Sequence mean | Median | Core coverage mean | Full 8-direction episodes | Direction-histogram similarity | Exact train-bigram mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N0 | 0.750 | 0.778 | 0.750 | 0/20 | 0.744 | 0.992 |
| N1 | 0.792 | 0.792 | 0.900 | 8/20 | 0.914 | 0.935 |
| N2 | 0.859 | 0.894 | 0.838 | 2/20 | 0.876 | 0.972 |
| N3 | 0.820 | 0.789 | 0.806 | 1/20 | 0.818 | 0.950 |
| N4 | 0.792 | 0.800 | 0.781 | 1/20 | 0.807 | 0.943 |
| N5 | 0.768 | 0.792 | 0.938 | 15/20 | 0.951 | 0.873 |

The models are all substantially closer to the training task sequences than
the reversed and first-event-only controls. The correct cohort-level conclusion
is therefore not “the models fail whenever one validation action differs.”
They have learned recognizable task-shaped sequences.

The remaining limitation is structured:

- N2 best preserves the ordering of the phases it emits, but it covers
  `stick+` in only 12/20 and `stick-` in only 6/20 episodes. Its high nearest
  sequence score is therefore not complete-task proof.
- N5 best preserves the full task repertoire: it covers all eight core
  directions in 15/20 episodes and its aggregate direction histogram is even
  closer to training than the held-out expert cohort. Its weaker bigram and
  nearest-sequence scores show more ordering/grouping variation.
- N1 is the strongest compromise in this batch: good sequence compatibility,
  0.900 mean core coverage, and a training-like direction histogram, but it
  still omits `stick-` in 11/20 episodes.
- N0 and N4 systematically under-cover both stick directions. N3 also
  under-covers `stick+`.

There is no single scalar winner under this evidence. N2 is the strongest
order-preservation model, N5 the strongest repertoire-completeness model, and
N1 the most balanced of the two properties.

## Why individual failures no longer dominate

Episode 10125 is low against the training cohort for every model, but the held-
out expert itself also scores only 0.333 and contains only four semantic
direction changes versus a training median of roughly ten. This episode remains
visible in the row artifact, but it does not veto the other 19 episodes or
define task correctness. Cohort medians, quartiles, repertoire coverage, and
controls govern the interpretation.

## Capability boundary

This run directly measures only the sequence of thresholded model proposals
while observations continue along recorded expert trajectories. It does not
show that the model would reach those later observations under its own actions,
that commands move the machine, that the sequence is safe, or that the task is
completed on held-out terrain.

The next promotion-relevant test must preserve these sequence metrics but add
self-generated or policy-on state progression. Until then, the permitted claim
is: **existing models learned a substantial expert-supported task grammar, with
different gaps in ordering and phase coverage; none is yet proven to execute
the complete real task closed loop.**
