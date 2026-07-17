# [Execution target G28/H1: Goal/progress/effect-conditioned ACT]

Date: 2026-07-12
Status: validation-rejected; reference remains raw ACT + mechanical assist

## [Execution target G28/H1: Objective and boundary]

Build a genuinely multi-task ACT proposal without repeating E52 suppression,
hard factorized projection, previous-command phase cues, or an uncalibrated
retry governor. The continuous ACT proposal remains the first and only action
source. The new branch adds:

```text
current image + qpos
        |
        +--> continuous ACT proposal chunk --------------------> sent action
        |
        +--> observation-context goal/progress forecast
        |
        +--> proposal-conditioned future-effect forecast (diagnostic)
```

The future qpos labels are training-only. The auxiliary heads never overwrite,
attenuate, clip, or gate the action. Identity action scale is enforced by the
offline evaluator; joystick scaling remains outside the model contract.

## [Execution target G28/H1: Data and split lock]

- Dataset: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/real_teleop_v1_episodes_72_104_20hz`
- Train IDs (19): `97,78,100,85,86,75,102,80,104,76,99,83,73,93,87,98,90,82,79`
- Validation IDs (5): `94,91,84,74,92`
- Split: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_hold_liveness_20260712/h1_direct_relabel_formal/ckpt/train_val_split.yaml`
- Split SHA-256: `09fe85bdab539ca2a12b5b4613f507ea009706cb38077b46e168f5171da59a3d`
- Held-out IDs `105..109`: forbidden until validation gates pass; this slice did not evaluate them.
- Future horizons: 4, 8, 20 ticks; stick remains structurally unsupported and masked.
- Future-delta normalization is computed from the 19 train IDs only. No source HDF5 is modified.

## [Execution target G28/H1: Implementation owners]

- `testbed/testbed/policies/act/goal_effect.py`: target construction, train-fold scale, goal/effect head, masked losses.
- `testbed/testbed/policies/act/detr/models/detr_vae.py`: observation-only context and zero-initialized action residual; proposal-conditioned effect head.
- `testbed/testbed/data/dataset.py`: causal qpos/qvel-derived labels and train-only normalization.
- `testbed/testbed/policies/act/adapter.py` and `trainer.py`: auxiliary loss wiring, checkpoint expansion, compact inference diagnostics.
- `testbed/testbed/cli/evaluate_complete_offline.py`: one auditable report joining all offline gates.
- Config: `testbed/testbed/configs/act_real_gmsl_eye2_goal_effect_v1.yaml`.

## [Execution target G28/H1: Training record]

- Initialization: H2 best checkpoint, model-only non-strict expansion for the new heads.
- Epochs: 200; batch 4; CUDA bf16 autocast; formal train/validation split fixed above.
- Best checkpoint: epoch 70; validation loss `0.21756469458341599`.
- Artifact root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g28_goal_effect_act_formal/ckpt`.
- `policy_best.ckpt` SHA-256: `ce56b43e50da4ce2ce1e5ab269469ea7d027b632e57a449e371fe54600ec90f6`.
- `dataset_stats.pkl` SHA-256: `9a337e6ecfbd4cd15da0a2dffcccf3b6844b677fe9337f388f2bf9c7d2e65282`.
- `resolved_config.yaml` SHA-256: `cec232e6ca719a104ef513adb3a0eb4da644e0ace9ccb80b486469692e4bd7aa`.

## [Execution target G28/H1: Acceptance and recovery]

The locked validation requirement for a new candidate is at least `46/48`
assist-recovered anchors, zero teacher-forcing-hidden deadlocks, no induced
deadlock against H2, and no release/tail/wrong-motion regression. If this
fails, the candidate cannot consume held-out data and cannot replace the raw
ACT + assist reference. Record the exact callback in the result document and
backtrack without threshold tuning.
