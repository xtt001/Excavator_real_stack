# CUDA 训练交接说明（2026-06-10）

本文档用于把当前已修复、已 QC 的 20Hz 训练数据迁移到 CUDA 电脑，并启动 ACT 训练。

## 当前结论

可以开始迁移到 CUDA 电脑训练。当前推荐训练入口是：

```bash
tb-train --config testbed/testbed/configs/act_real_20hz_v1.yaml
```

当前可训练数据不是原始 50Hz HDF5，而是修复后的 20Hz 数据集：

```text
/media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1
```

训练配置会读取：

```text
/media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1/qc_full/train_ready_manifest.json
```

该 manifest 只把 train-ready episode 交给训练 loader。失败 episode 不会参与训练。

## 外置硬盘迁移

可以直接把外置硬盘插到 CUDA 电脑，但要满足以下条件：

1. CUDA 电脑能看到数据目录。
2. 挂载路径最好保持为：

```text
/media/mundane/EXTERNAL_USB
```

如果 CUDA 电脑上的挂载路径不同，有两种处理方式。

方式 A：建立同名软链接，保持配置不变：

```bash
sudo mkdir -p /media/mundane
sudo ln -s /实际挂载路径/EXTERNAL_USB /media/mundane/EXTERNAL_USB
```

方式 B：复制一份训练配置，修改其中两个路径：

```yaml
task:
  dataset_dir: /你的挂载路径/real_teleop_v1_repaired_20hz_v1
  train_ready_manifest_path: /你的挂载路径/real_teleop_v1_repaired_20hz_v1/qc_full/train_ready_manifest.json
```

如果外置硬盘较慢，可以把 20Hz 数据目录复制到 CUDA 电脑本地 NVMe，再改配置路径。不要复制原始 50Hz 数据作为训练输入。

## 当前数据标注

当前 20Hz QC 结果：

```text
PASS: 29
WARN: 2
FAIL: 9
train-ready: 31
```

严格 PASS：

```text
episode_26, episode_29, episode_30, episode_32, episode_33,
episode_34, episode_35, episode_36, episode_38, episode_39,
episode_40, episode_41, episode_42, episode_43, episode_48,
episode_49, episode_50, episode_51, episode_53, episode_54,
episode_55, episode_56, episode_57, episode_58, episode_59,
episode_61, episode_62, episode_63, episode_65
```

WARN 但仍 train-ready：

```text
episode_47, episode_60
```

这两条的原因是 `bucket_semantic_review`，当前判定为“下挖深度偏浅，需要人工复核，但不直接丢弃”。它们会进入训练 manifest。

FAIL，已从训练 manifest 排除：

```text
episode_27, episode_28, episode_31, episode_37, episode_44,
episode_45, episode_46, episode_52, episode_64
```

失败原因摘要：

```text
episode_27: raw_qpos_branch_jump；虽然 FPV gap 已 mask，但 swing raw qpos 有约 2π 分支跳变
episode_28: episode_length_outlier + episode_total_steps_outlier
episode_31: qpos_jump
episode_37: qpos_jump
episode_44: qpos_jump + raw_qpos_branch_jump + bucket_reference_outlier
episode_45: raw_qpos_branch_jump + episode_total_steps_outlier
episode_46: episode_length_outlier + episode_total_steps_outlier
episode_52: bucket_semantic_outlier
episode_64: bucket_semantic_outlier
```

带 INFO 的 episode 不等于坏数据。当前 INFO 主要表示：

- `usable_with_gap_mask`：FPV 大 gap 被 `train_exclude_mask` 覆盖，训练 sampler 不会采跨 gap 的 chunk/window。
- `bucket_reference_semantic_keep`：原本触发了 bucket reference warn，但 bucket 语义复判为可保留。

## 关键数据文件

训练数据：

```text
/media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1/episode_*.hdf5
```

训练 manifest：

```text
/media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1/qc_full/train_ready_manifest.json
```

QC 总结：

```text
/media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1/qc_full/training_qc_summary.json
/media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1/qc_full/training_qc_episodes.csv
```

可视化总报告：

```text
/media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1/visual_review_current/index.html
```

## CUDA 电脑环境准备

在 CUDA 电脑上获取代码：

```bash
git clone git@github.com:xtt001/Excavator_real_stack.git
cd Excavator_real_stack
git checkout tx/data-cleaning-training
git pull
```

创建环境：

```bash
conda env create -f environment.yml
conda activate excavator-real-stack
```

如果 CUDA 机器上 PyTorch 不是 CUDA 版本，需要按该机器的 CUDA 版本重装 PyTorch。检查：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

必须看到 `cuda available True` 再开始正式训练。

## 训练前检查

确认数据路径可访问：

```bash
ls /media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1/episode_26.hdf5
ls /media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1/qc_full/train_ready_manifest.json
```

确认 manifest 是当前版本：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("/media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1/qc_full/train_ready_manifest.json")
m = json.loads(p.read_text())
print("train_ready", len(m["train_ready_episode_ids"]))
print("warn", m["warn_episode_ids"])
print("failed", m["failed_episode_ids"])
PY
```

期望输出：

```text
train_ready 31
warn ['episode_47', 'episode_60']
failed ['episode_27', 'episode_28', 'episode_31', 'episode_37', 'episode_44', 'episode_45', 'episode_46', 'episode_52', 'episode_64']
```

先跑 1 epoch smoke：

```bash
tb-train \
  --config testbed/testbed/configs/act_real_20hz_v1.yaml \
  --epochs 1 \
  --ckpt-dir runs/ckpts/smoke_real_excavation_act_20hz_v1
```

如果 smoke 通过，再跑正式训练：

```bash
tb-train \
  --config testbed/testbed/configs/act_real_20hz_v1.yaml \
  --ckpt-dir runs/ckpts/real_excavation_act_20hz_v1
```

## 当前训练配置语义

当前配置文件：

```text
testbed/testbed/configs/act_real_20hz_v1.yaml
```

关键设置：

```yaml
task:
  dataset_dir: /media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1
  train_ready_manifest_path: /media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_20hz_v1/qc_full/train_ready_manifest.json
  episode_len:
  camera_names:
    - fpv

policy:
  class: ACT
  device: cuda
  low_dim_keys:
    - qpos
  act_params:
    chunk_size: 20

train:
  batch_size: 4
  amp: true
  ckpt_dir: runs/ckpts/real_excavation_act_20hz_v1
```

注意：

- `episode_len:` 为空，loader 会自动使用当前数据集中最大长度。
- `chunk_size: 20` 表示 20Hz 下约 1 秒 action horizon。
- 当前 low-dimensional observation 只使用 `qpos`。如果要加 `qvel`，需要作为单独实验，不要直接覆盖当前基线配置。
- 20Hz 数据已经是 action pre-aligned，训练 loader 不会再套旧的 real `t0-1` 偏移。
- 对带 `train_exclude_mask` 的 episode，训练 sampler 会跳过跨 FPV gap 的 chunk/window。

## 不要做的事

不要直接用以下目录训练：

```text
/media/mundane/EXTERNAL_USB/real_teleop_v1
/media/mundane/EXTERNAL_USB/real_teleop_v1_repaired_bucket_v1
```

原因：

- `real_teleop_v1` 是原始 50Hz 数据，包含历史 bucket qpos 分支问题和不同步风险。
- `real_teleop_v1_repaired_bucket_v1` 是修复后的 50Hz 数据，但当前训练策略明确采用 20Hz 语义。

不要手工把 FAIL episode 加回训练，除非先重新跑 QC 并更新 manifest。

不要删除或覆盖原始数据。后续若需要重建 20Hz 数据，应输出到新目录或确认当前目录可以被覆盖。

## 训练产物交接

训练完成后，至少保留并回传：

```text
runs/ckpts/real_excavation_act_20hz_v1/policy_best.ckpt
runs/ckpts/real_excavation_act_20hz_v1/policy_latest.ckpt
runs/ckpts/real_excavation_act_20hz_v1/dataset_stats.pkl
runs/ckpts/real_excavation_act_20hz_v1/resolved_config.yaml
runs/ckpts/real_excavation_act_20hz_v1/run_metadata.json
```

后续部署或 policy shadow/control 需要 checkpoint、dataset stats 和 resolved config 一起使用。

## 快速判断是否路径正确

如果训练启动时报 `train_ready_manifest_path does not exist`，说明硬盘挂载路径和配置不一致。

如果训练启动时报 `cuda available False` 或模型在 CPU 上跑，说明 CUDA/PyTorch 环境不对。

如果训练 loader 报 HDF5 路径缺失，优先检查：

```bash
find /media -path '*real_teleop_v1_repaired_20hz_v1' -type d
```

找到实际路径后，用软链接或修改配置修正。
