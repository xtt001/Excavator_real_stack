# E30 Window-Conditioned Motion Intent Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the E30 eye2 training path that teaches ACT when to move through runtime deadzones and when to stay below them.

**Architecture:** Add a focused data helper that derives action-aligned deadzone intent masks from expert actions plus handoff labels. Extend ACT dataset batches only when configured, keep legacy four-tuple batches unchanged, and make ACT loss consume explicit masks instead of inferring episode phase from local chunks.

**Tech Stack:** Python, HDF5, NumPy, PyTorch, PyTest, repo-local `PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed`.

---

## File Map

- Create `testbed/testbed/data/deadzone_intent_labels.py`: derive `move_mask`, `stop_mask`, `wrong_mask`, and masked action-stat selectors from actions, thresholds, and optional handoff masks.
- Modify `testbed/testbed/data/dataset.py`: optionally compute masked action normalization stats and return extended ACT batches with E30 masks.
- Modify `testbed/testbed/policies/act/trainer.py`: accept both legacy four-tuples and extended dict / tuple batches.
- Modify `testbed/testbed/policies/act/adapter.py`: accept optional deadzone intent masks and action loss masks in `forward_loss`; add the E30 window-conditioned loss terms without removing existing `deadzone_loss`.
- Create `testbed/tests/test_deadzone_intent_labels.py`: unit tests for mask derivation and masked stats.
- Modify `testbed/tests/test_realworld_v1.py` or create focused dataset tests: verify extended batch alignment from a tiny HDF5 fixture.
- Modify `testbed/tests/test_act_deadzone_loss.py`: verify action-loss masking and E30 same-dir / stop / wrong losses.
- Create E30 smoke config under `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/`.

## Execution Status

- Implemented `testbed.data.deadzone_intent_labels` and focused label/stat tests.
- Extended `EpisodicDataset` with optional E30 dict batches while preserving legacy four-tuple batches when disabled.
- Added mask-aware action normalization through `deadzone_intent.use_action_loss_mask_for_stats`.
- Added `deadzone_intent.require_action_loss_in_chunk` to keep pure automation-tail chunks out of ordinary action-imitation sampling.
- Extended `ACTTrainer._forward` to pass E30 masks to `ACTAdapter.forward_loss`.
- Added `window_deadzone_loss` terms in `ACTAdapter` and preserved old `deadzone_loss` behavior for existing configs.
- Verified targeted tests: `32 passed, 1 warning` for dataset, label, ACT deadzone/loss, handoff label, phase-gate, and visual-domain tests.
- E30-eye2 smoke completed with non-zero window deadzone loss terms and produced `policy_latest.ckpt`, `policy_best.ckpt`, `dataset_stats.pkl`, `resolved_config.yaml`, and `run_metadata.json`.
- Full E30-eye2 training completed from `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e30_window_deadzone_intent_eye2.yaml`: best epoch 1155, best val loss 0.12504172697663307.
- Full E30-eye2 replay and runtime-scaled gates completed. Result: reject as candidate because replay MAE/RMSE, startup intent, extra/wrong, and tail stop stability all regress versus E22b, despite start40 becoming fully quiet.

## Task 1: Deadzone Intent Label Helper

**Files:**
- Create: `testbed/testbed/data/deadzone_intent_labels.py`
- Test: `testbed/tests/test_deadzone_intent_labels.py`

- [ ] **Step 1: Write failing mask derivation tests**

Add tests that assert:

```python
thresholds = {
    "swing": {"pos": 0.5, "neg": 0.5},
    "boom": {"pos": 0.4, "neg": 0.4},
    "stick": {"pos": 0.3, "neg": 0.3},
    "bucket": {"pos": 0.2, "neg": 0.2},
}
actions = np.asarray([
    [0.60, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],
    [0.00, -0.50, 0.00, 0.00],
], dtype=np.float32)
labels = compute_deadzone_intent_labels(actions=actions, thresholds=thresholds)
assert labels.move_mask.shape == (3, 4, 2)
assert labels.move_mask[0, 0, 0]
assert labels.move_mask[2, 1, 1]
assert labels.stop_mask.tolist() == [False, True, False]
assert labels.wrong_mask[0, 0, 0] == False
assert labels.wrong_mask[0, 0, 1] == True
assert labels.wrong_mask[1].all()
```

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m pytest testbed/tests/test_deadzone_intent_labels.py -q
```

Expected: fail because `testbed.data.deadzone_intent_labels` does not exist.

- [ ] **Step 3: Implement helper**

Implement a dataclass:

```python
@dataclass(frozen=True)
class DeadzoneIntentLabels:
    move_mask: np.ndarray
    stop_mask: np.ndarray
    wrong_mask: np.ndarray
    action_loss_mask: np.ndarray
```

and functions:

```python
compute_deadzone_intent_labels(actions, thresholds, action_loss_mask=None, tail_idle_mask=None, owner_automation=None)
masked_action_stats(actions, action_loss_mask)
```

- [ ] **Step 4: Verify green**

Run the same pytest command. Expected: pass.

## Task 2: Dataset Extended Batch Contract

**Files:**
- Modify: `testbed/testbed/data/dataset.py`
- Test: `testbed/tests/test_realworld_v1.py`

- [ ] **Step 1: Write failing dataset test**

Create a tiny HDF5 fixture with `action`, `observations/qpos`, `observations/qvel`, encoded or raw `video4/video5`, and `handoff/action_loss_mask`, `handoff/tail_idle_mask`, `handoff/owner_automation`. Assert that enabling E30 masks returns an extended sample containing:

```python
sample["action"].shape == (target_len, 4)
sample["deadzone_move_mask"].shape == (target_len, 4, 2)
sample["deadzone_stop_mask"].shape == (target_len,)
sample["deadzone_wrong_mask"].shape == (target_len, 4, 2)
sample["action_loss_mask"].shape == (target_len,)
sample["is_pad"].shape == (target_len,)
```

Also assert legacy dataset construction still returns the original four-tuple.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m pytest testbed/tests/test_realworld_v1.py -q -k "deadzone_intent_mask or action_loss_mask"
```

Expected: fail because `EpisodicDataset` has no E30 mask option.

- [ ] **Step 3: Implement dataset options**

Add optional constructor/config arguments:

```python
deadzone_intent: dict[str, Any] | None = None
```

When enabled, read thresholds, handoff masks, derive labels for the sampled chunk, pad them to `target_len`, and return a mapping. Keep the legacy four-tuple path unchanged when disabled.

- [ ] **Step 4: Verify green**

Run the same pytest command. Expected: pass.

## Task 3: Mask-Aware Action Stats

**Files:**
- Modify: `testbed/testbed/data/dataset.py`
- Test: `testbed/tests/test_deadzone_intent_labels.py`

- [ ] **Step 1: Write failing stats test**

Assert that stats computed with `action_loss_mask == true` ignore automation tail actions:

```python
actions = np.asarray([[1, 0, 0, 0], [1, 0, 0, 0], [-9, 0, 0, 0]], dtype=np.float32)
mask = np.asarray([1, 1, 0], dtype=bool)
mean, std = masked_action_stats(actions, mask)
assert mean[0] == pytest.approx(1.0)
assert std[0] >= 0.01
```

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m pytest testbed/tests/test_deadzone_intent_labels.py -q
```

Expected: fail until masked stats are wired into data stats.

- [ ] **Step 3: Implement mask-aware stats option**

Add an optional `action_stat_mask_path` / `use_action_loss_mask_for_stats` option used by E30 configs. Preserve old stats behavior for existing configs.

- [ ] **Step 4: Verify green**

Run the same pytest command and the dataset mask tests.

## Task 4: ACT Trainer And Adapter Loss

**Files:**
- Modify: `testbed/testbed/policies/act/trainer.py`
- Modify: `testbed/testbed/policies/act/adapter.py`
- Test: `testbed/tests/test_act_deadzone_loss.py`

- [ ] **Step 1: Write failing adapter tests**

Add tests that call `forward_loss` or `_window_deadzone_loss_terms` with explicit masks:

```python
expert = torch.tensor([[[0.60, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
policy = torch.tensor([[[0.40, 0.0, 0.0, 0.0], [0.0, 0.45, 0.0, 0.0]]])
move_mask = torch.zeros((1, 2, 4, 2), dtype=torch.bool)
move_mask[0, 0, 0, 0] = True
stop_mask = torch.tensor([[False, True]])
wrong_mask = torch.ones((1, 2, 4, 2), dtype=torch.bool)
wrong_mask[0, 0, 0, 0] = False
```

Assert same-dir loss is positive for the first step, stop loss is positive for the second step, and wrong loss is zero when no wrong crossing occurs.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m pytest testbed/tests/test_act_deadzone_loss.py -q
```

Expected: fail because the E30 loss path does not exist.

- [ ] **Step 3: Implement trainer / adapter support**

Trainer must detect mapping batches and pass:

```python
deadzone_move_mask
deadzone_stop_mask
deadzone_wrong_mask
action_loss_mask
```

Adapter must apply `action_loss_mask` to L1 action imitation and add E30 loss terms when configured.

- [ ] **Step 4: Verify green**

Run:

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m pytest testbed/tests/test_act_deadzone_loss.py testbed/tests/test_realworld_v1.py -q -k "deadzone_loss or deadzone_intent_mask or action_loss_mask"
```

Expected: pass.

## Task 5: E30 Eye2 Smoke Config

**Files:**
- Create: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e30_window_deadzone_intent_eye2_smoke.yaml`
- Create: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e30_window_deadzone_intent_eye2_smoke/train_val_split.yaml`

- [ ] **Step 1: Create config**

Use E16 settings, but set:

```yaml
task:
  dataset_dir: /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/handoff_eligibility_20hz_dwell10
train:
  num_epochs: 1
  ckpt_dir: /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e30_window_deadzone_intent_eye2_smoke
  split_path: /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e30_window_deadzone_intent_eye2_smoke/train_val_split.yaml
  deadzone_intent:
    enabled: true
    threshold_json: /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/deadzone_policy_raw_for_runtime_scale.json
    use_handoff_masks: true
    use_action_loss_mask_for_l1: true
    use_action_loss_mask_for_stats: true
```

- [ ] **Step 2: Run smoke**

Run:

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m testbed.cli.train --config /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e30_window_deadzone_intent_eye2_smoke.yaml
```

Expected: completed run with `policy_latest.ckpt`, `resolved_config.yaml`, `run_metadata.json`, and logged E30 loss terms.

- [ ] **Step 3: Update experiment ledger**

Record smoke status and any blocker in:

```text
docs/superpowers/plans/2026-07-08-policy-gate-experiments.md
```

## Verification Commands

Run before launching a full E30 training:

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m pytest \
  testbed/tests/test_deadzone_intent_labels.py \
  testbed/tests/test_act_deadzone_loss.py \
  testbed/tests/test_realworld_v1.py \
  -q -k "deadzone_intent or deadzone_loss or action_loss_mask"

git diff --check
```
