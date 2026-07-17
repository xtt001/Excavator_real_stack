# [Execution target G48/H1: This slice prepares training but does not train]

This slice makes the next ACT comparison reproducible without starting a GPU
job.  It does not modify a source HDF5, instantiate an ACT model, create a
checkpoint, inspect a model on the frozen new-style test block, or change any
field/runtime configuration.

Target lock:

- worktree: `/home/pingfan/Excavator_real_stack_e52_deadlock_eval`
- branch: `codex/e52-offline-deadlock-gate`
- base HEAD: `0fab67eda7e449b70622b65afc7ada01142f56e5`
- policy action domain: direct model output with identity scale
  `[1.0, 1.0, 1.0, 1.0]`
- permanently forbidden legacy selection set: old dataset `episode_105..109`
- new-style frozen test block: source `episode_156..175`, not linked into any
  training view

# [Execution target G48/H2: Frozen train and validation views]

The current ACT loader accepts one `dataset_dir` and integer episode IDs.  Old
and new sources reuse some integer IDs, so directly concatenating IDs would be
ambiguous.  The preparation tool creates absolute symbolic links with new
composite IDs; it never copies or rewrites HDF5 payloads.  Every link records
the source path, source episode ID, role, and source SHA-256.

New-only view:

- path: `/data/pingfan/Excavator_real_stack_data/g48_new_trainval_view_v1`
- train: 120 new-style episodes
- validation: 20 chronological new-style episodes
- test: zero linked episodes
- manifest SHA-256:
  `93d16983e70b0a614908cdf232d8f5879946b5e6b6175ae7dee6dfc5199e0348`
- split SHA-256:
  `4b796da6520e1469ec4cd0d1d94af7a9886c1fde8e1a16b123ea36a474aa3ebc`

Old+new view:

- path:
  `/data/pingfan/Excavator_real_stack_data/g48_old_new_trainval_composite_v1`
- train: 120 new-style + 20 old-style episodes
- validation: the same 20 new-style episodes used by the new-only view
- old validation episodes are not added, so model selection remains aligned
  with the current field style
- test: zero linked episodes
- manifest SHA-256:
  `ffa811434ef8abf34932283a041a707ec9982fd63d6a41a382116565deb4d5c9`
- split SHA-256:
  `7528a11f43c943e2ba9e7abfbd25a02d06140594d08262a1a0dec96b6677663b`

The data specs are:

- `testbed/testbed/configs/data_g48_new_trainval_view_v1.yaml`
- `testbed/testbed/configs/data_g48_old_new_trainval_composite_v1.yaml`

The builder is `python -m testbed.cli.prepare_training_composite`.  It refuses
to overwrite an existing view.  `--verify <view> --verify-hashes` rechecks every
link and source hash without training.

# [Execution target G48/H3: Four attributable experiment configs]

All configs use `video4/video5`, qpos only, 20-step ACT chunks, 2000 epochs, the
same new-style validation block, and no state-hold transition resampling.  The
latter is deliberately disabled so the comparison isolates data and target
semantics instead of changing two mechanisms at once.

| Stage | Train data | Objective | Purpose | Preflight |
|---|---|---|---|---|
| A | new 120 | ordinary continuous ACT | new-style reference | ready |
| B | new 120 + old 20 | ordinary continuous ACT | measure the value of compatible old data | ready |
| C | new 120 + old 20 | effective-action semantics plus raw continuous tie-breaker | measure deadzone-aware supervision | ready |
| D | new 120, initialized from C | same effective-action objective, LR `2e-6` | reinforce current field style after mixed pretraining | pending C checkpoint |

Configs:

- `testbed/testbed/configs/act_real_gmsl_eye2_g48_a_new_continuous_2000.yaml`
- `testbed/testbed/configs/act_real_gmsl_eye2_g48_b_old_new_continuous_2000.yaml`
- `testbed/testbed/configs/act_real_gmsl_eye2_g48_c_old_new_effective_action_2000.yaml`
- `testbed/testbed/configs/act_real_gmsl_eye2_g48_d_new_effective_action_finetune_2000.yaml`

C and D implement the agreed training semantics:

- expert commands inside the directional mechanical deadzone become neutral
  targets;
- each axis receives neutral/positive/negative auxiliary supervision;
- active magnitude, the first four transition ticks, and persistent effective
  commands receive explicit weight;
- raw continuous action remains with weight `0.35`, preserving legitimate
  multi-axis operator style;
- the auxiliary class is not a runtime gate and never replaces the continuous
  ACT command.

D names C's future `policy_best.ckpt` as an explicit dependency.  Its preflight
status must remain `pending_dependency` until C genuinely finishes; no empty or
latest checkpoint is silently substituted.

# [Execution target G48/H4: Read-only preflight evidence]

The preflight implementation is split by responsibility:

- `testbed/testbed/data/training_composite.py` owns the immutable symlink view
  and provenance manifest;
- `testbed/testbed/policies/act/training_preflight.py` owns read-only ACT config,
  split, HDF5 input, camera, deadzone-domain, output-directory, and dependency
  checks;
- the CLI wrappers only parse arguments and print/write reports.

The combined real-data report is:

`/data/pingfan/Excavator_real_stack_data/runs/g48_act_training_preparation_20260714/training_preflight_report.json`

Results:

- A: ready, 120 train / 20 validation;
- B: ready, 140 train / 20 validation;
- C: ready, 140 train / 20 validation;
- D: pending only because C's best checkpoint does not exist yet;
- all four pin the policy action-scale contract to identity;
- no manifest contains a test role or the forbidden source IDs `105..109`;
- every selected HDF5 has finite four-axis storage already covered by G47 QC,
  qpos/qvel/action dimensions of four, and both requested encoded cameras;
- checkpoint output directories are absent or empty.

An additional read-only `load_data` smoke used the actual frozen splits and
decoded one batch from A and C:

- A batch: tuple contract, images `(2,2,3,216,384)`, qpos `(2,4)`, action
  `(2,1043,4)`;
- C batch: mapping contract with the same observation/action shapes plus
  `effective_action_phase`, `effective_action_valid`, transition, persistence,
  and loss-weight tensors;
- both reused the saved split; normalization used train IDs only.

No call to `Runner.train()` or `train_policy()` was made.

Repository verification after the real-data smoke:

- focused composite/preflight tests: `3 passed`;
- complete repository suite with both repository roots on `PYTHONPATH`:
  `491 passed, 4 subtests passed`;
- Ruff on all new Python owners/tests: passed;
- `git diff --check`: passed.

# [Execution target G48/H5: Deferred commands and selection order]

These commands are prepared but were not executed:

```bash
cd /home/pingfan/Excavator_real_stack_e52_deadlock_eval/testbed
PYTHONPATH=/home/pingfan/Excavator_real_stack_e52_deadlock_eval/testbed \
  python -m testbed.cli.train \
  --config testbed/configs/act_real_gmsl_eye2_g48_a_new_continuous_2000.yaml
```

Replace only the config basename for B and C.  D must not start until C has a
completed `run_metadata.json` and a verified `policy_best.ckpt`; D uses model
initialization, not optimizer resume.

The comparison order after training is fixed:

1. Select each checkpoint using only the common new-style validation block.
2. Compare A versus B to isolate the value of old compatible demonstrations.
3. Compare B versus C to isolate deadzone-aware target semantics.
4. Compare C versus D to isolate new-style fine-tuning.
5. Metrics must include open-loop error, per-episode first effective main-axis
   startup, every transition, per-axis deadzone hit, wrong/opposite direction,
   multi-axis support, release/tail, recursive state-hold, and raw/assist views.
6. Freeze candidate, config, checkpoint, thresholds, and evaluation code.
7. Only then evaluate the new-style test block once.  The legacy held-out
   `105..109` remains separate and untouched until the same final freeze.

# [Execution target G48/H6: Rollback is data-only and immediate]

This slice changes no source data and no runtime behavior.  Rollback consists
of ignoring the four configs and deleting the two generated symlink-view
directories plus the G48 report directory.  Source HDF5 files, the G47 split,
old checkpoints, runtime policy configs, and field defaults remain unchanged.
