# SimVerify Expert-Habit Definition Audit Evidence — 2026-07-27

## 1. Decision

```text
decision=accept
evidence_scope=recorded-observation/offline_definition_audit
scenario_freeze_authorized=false
training_authorized=false
held_out_test_authorized=false
held_out_observation_read_count=0
```

`accept` 只表示任务合同、observable dig-ready detector、自然习惯 inventory、condition
support 和候选场景生成规则可以交给用户审阅。它不表示 ACT 已训练、condition 已被模型
利用、AGX 已闭环完成挖掘、物理效果成立或可迁移真机。

正式不可变产物：

```text
/data/pingfan/Excavator_real_stack_data/
  simverify_habit_cycle_definition_v3/
```

构建 Git commit：`505aec120bb6e3ef9f5ca54ce9fdf1687c4237f6`

`audit_manifest.json` SHA-256：
`edc609beedb4956a03218657b4434b129945c7959d8f25293a0b0071da94f7a7`

`checksums.sha256` SHA-256：
`84fcb51656002728010bd3d63ff48f423661cc74f638acfa0f1c9358892e92b5`

校验结果：`7/7` 文件通过，失败 `0`。

## 2. Why the audit was iterated

不可变 v1 首先得到 `revise_boundary`。原因不是数据缺少 ready，而是第一版把 qpos dwell
同时当作宽松候选和最终确认器：为了消除 4 段更早的目标区穿越，它选择 32 source steps
（约 0.64 秒），同时只保留约 28% train ready reference。

v2 按原计划恢复两级 detector：

1. 数值层以 ready candidate recall 为目标；
2. 冻结视觉层拒绝更早穿越并确认 ready。

v2 在 validation 上达到 36/36 reference match、零过早确认，但实现错误地把 4 个 train
拟合误差也计入 validation Gate，因此仍为 `revise_boundary`。

v3 没有删除这些困难样本，也没有改变 detector 参数；它只按预注册 split 语义修正 Gate：
train 误差进入 review inventory，validation 负责否证。v1/v2 产物均保留，未覆盖。

## 3. Dig-ready boundary evidence

数值候选 dwell 为 `1` source step（约 `0.02 s`）。这不是单独的 ready 判定；它只保证
宽松候选不漏掉 train reference，最终确认仍要求冻结 eye+stick 视觉证据。

| Check | Result |
| --- | ---: |
| train reference | 131 |
| train reference match | 127 |
| train early confirmation review | 4 |
| validation reference | 36 |
| validation reference match | 36 |
| validation early confirmation | 0 |
| validation match rate | 100% |
| train-derived Wilson lower threshold | 92.41% |
| validation Wilson lower | 90.36% |

validation point rate `100%` 高于 train-derived lower threshold `92.41%`。Wilson lower 只作为
样本不确定性报告，不替代 point-rate Gate。

视觉审计：

| Task | Validation N | Accuracy | Balanced accuracy | Result |
| --- | ---: | ---: | ---: | --- |
| ready sector, eye pair | 31 | 96.77% | 96.97% | pass |
| ready vs dump, eye+stick | 62 | 100% | 100% | pass |
| right ready vs dump, eye+stick | 26 | 100% | 100% | pass |

right ready 与固定卸料 corridor 在本次 source-episode 隔离 validation 上可分。该结果仅适用
当前仿真画面和固定卸料区，不证明真实相机域可分。

## 4. Natural expert-habit inventory

这些计数来自重新计算的 observable numeric transition；旧 583-cycle 结论没有直接复用。
sector bootstrap review margin 使边界样本保持 unknown，因此计数低于旧宽松切片是预期的。

| Split | stay | step_left | step_right | nonadjacent diagnostic |
| --- | ---: | ---: | ---: | ---: |
| train | 62 | 28 | 33 | 17 |
| validation | 18 | 7 | 7 | 5 |

train source-episode 支持：

- `stay`: 15 episodes；
- `adjacent`: 14 episodes；
- 两类均出现于 `pre_fix_candidate` 和 `post_fix_candidate` controller epoch。

validation 中 `stay` 来自 3 episodes，`adjacent` 来自 4 episodes，也跨两个 epoch。train
同区连续 run 共 42 段，中位 `1` cycle、p95 `3`、最大 `5`；validation 共 11 段，中位
`1`、p95 `3`、最大 `4`。因此“连续 stay”真实存在，但长 run 不是默认常态。

非相邻 jump 保留诊断，不拆步、不进入候选场景和首轮主分母。

## 5. Condition support and null risk

reference-matched 条目：

| Item | Count |
| --- | ---: |
| train entries | 110 |
| train with supported alternative | 108 |
| validation entries | 31 |
| validation with supported alternative | 31 |
| all supported alternatives | 191 / 199 |

替代支持要求相同 current sector、不同合法目标、不同 source episode、train-only neighbor，
且距离不超过 train 同意图最近邻 p95。其余标记 `coverage_gap`，未来不计成成功或失败。

validation hindsight target 的猜测风险：

- global frequency prior accuracy: `35.48%`；
- current-sector prior accuracy: `58.06%`；
- 1 秒历史加 eye feature accuracy: `67.74%`，balanced accuracy `69.36%`；
- shuffled-target null p95 balanced accuracy: `48.09%`。

这说明 observation 和操作习惯本身能猜中一部分目标。以后判断 ACT 是否“看懂 condition”
必须在支持内 anchor 上做 B1 正确 condition 对 B2 matched shuffled condition 的配对实验；
不能用总体 replay accuracy 或自然序列命中率代替。

## 6. Candidate fixed scenarios

候选总数 `66`：

- `repeat_same`: 12 candidates，覆盖 10 source episodes；
- `move_adjacent`: 54 candidates，覆盖 13 source episodes。

排序后的前三个 `repeat_same`：

```text
repeat_same:episode_30:cycle_7_9
repeat_same:episode_27:cycle_24_26
repeat_same:episode_6:cycle_16_18
```

排序后的前三个 `move_adjacent`：

```text
move_adjacent:episode_29:cycle_25_25
move_adjacent:episode_16:cycle_20_20
move_adjacent:episode_16:cycle_21_21
```

这只是候选排序，不是冻结清单。用户需要确认：

1. `repeat_same` 是否优先选择 3-cycle run，同时保留若干更常见的 2-cycle run；
2. `move_adjacent` 是否按自然频率全保留，还是仅选择少量人工可读 smoke 场景；
3. 4 个 train early-confirm review 样本是否从未来训练边界中排除或进入显式 review split。

在这些选择确认并形成新 frozen scenario manifest 前，不启动 B0/B1/B2 训练。

## 7. Artifact hashes

```text
source_snapshot_manifest.json
  95f21b5c5ad43f4a8ca084a09890ed1263367ec04ec39a95b69d86fd50ec4734
habit_transition_inventory_v1.json
  d6e8703f84fc0446b35703c31450cbaea92155cd326e43cf85e636885228cd69
dig_ready_boundary_audit_v1.json
  81efc603b5291a73d48e3c0cbeeacce8763596adc51c479f1ec02e9c8f7dacf9
habit_condition_support_v1.json
  555671f050599ff0b32e5744f64b1ded642b253cd5298820e794c1639935c0c4
expert_habit_scenario_candidates_v1.json
  b43aaa4b142947aabfd449860085bc1cc628bc0f49bc76503bb4a4658d2098ed
definition_falsification_decision_v1.json
  9a86b7b0112413d5cacf4a847c72641a541a77107d6b3c11c52767417c47b79f
```
