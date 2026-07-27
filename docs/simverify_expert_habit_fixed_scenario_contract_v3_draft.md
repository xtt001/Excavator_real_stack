# SimVerify Expert-Habit Fixed Scenario Contract v3（审阅草案）

状态：`definition_audit_implementation_ready`

证据范围：`recorded-observation/offline_definition_audit`

训练授权：`false`

场景冻结授权：`false`

## 1. 本轮到底构建什么

本轮不训练规划模型，也不要求 left/center/right 的所有排列组合。数据中的专业人员仿真
示教只被称为 `sim_expert_demonstration_habit`，用于提炼两类自然习惯：

- `stay`：下一铲仍在当前 sector；
- `adjacent`：下一铲移动到相邻 sector，脚本表示为 `step_left` 或 `step_right`。

固定场景脚本在卸料完成后提交目标。ACT 是执行器：负责完成挖掘、卸料、返回和进入指定
dig-ready；它不选择下一目标，不读 terrain privilege，也不在线优化地形。该命名不声明
真机域习惯、物理效果或闭环能力已经验证。

## 2. Cycle 和目标提交合同

一个 cycle 是：

```text
当前 sector 的因果可确认 dig-ready
  -> 离开初始 ready
  -> dig
  -> carry
  -> dump_start
  -> dump_end
  -> 下一目标 sector 的因果可确认 dig-ready
```

目标只允许在 `dump_end` 后原子提交。此时本轮历史完整，return 尚未开始。脚本输出相对
意图，监督器结合当前 sector 得到绝对目标：

| current | stay | step_left | step_right |
| --- | --- | --- | --- |
| left | left | fail closed | center |
| center | center | left | right |
| right | right | center | fail closed |

`left↔right` 是 `nonadjacent_jump`，只进入 inventory 诊断，不进入首轮训练或主成功分母，
也不得拆成数据中不存在的两步轨迹。

同区循环不能因目标 sector 与起点相同而立即成功。终点检测必须在依次观察
`leave_initial_ready -> dig -> carry -> dump_start -> dump_end` 后才武装。完成同时要求
事件顺序完整且终点 dig-ready 与 committed target 一致。

每次执行分别记录：

```text
scripted_target_sector
hindsight_expert_target_sector
realized_target_sector
observable_cycle_completed
physical_effect_validated
```

历史数据没有 recorded command，因此 `command=unknown_not_recorded`，专家目标只能标为
`hindsight_observable_next_dig_entry`。未来固定脚本目标才可标为
`scripted_fixed_scenario`。二者不得混写。

## 3. Dig-ready 的可证伪定义

离线 reference 与 runtime 确认严格分离：

1. 离线 reference 可以利用下一次 observable dig-entry，定位它之前最后一个目标工作区
   连续区间；它只是真值参照。
2. runtime 候选从 `dump_end` 后向前扫描，只读 committed target、当前/历史
   image/qpos/qvel/action。
3. 数值 dwell 不是最终判定器；train 上选择真实 ready 候选召回率最大时延迟最短的
   dwell，保留更早的目标区穿越交给下一层视觉证据拒绝。
4. 冻结 eye+stick ResNet-18 特征用 train 的 ready reference、dump 和错误前向候选校准；
   runtime 取第一个被判为 ready 的前向候选。
5. train 的拟合误差保留为 review inventory；validation 才用于检查过早确认、漏确认、
   sector 可分性、ready/dump 可分性，并单独检查 right dig-ready 与固定卸料 corridor。

二分类 ready/dump Gate 使用数学 chance null `0.5` 和 validation Wilson lower bound。
shuffled label 在完全可分的二簇上会因标签整体翻转产生 `p95=1.0`，因此只保留为诊断，
不作为二分类通过门槛；三分类 sector 仍使用 shuffled-label null。

任何 runtime confirm 是否落入 reference interval，只用于审计 detector，不能作为
runtime 输入。门槛来自 train operating point、episode 隔离 validation 和 shuffled-label
null；held-out observation 保持锁定。

## 4. 专家习惯审计

审计从只读 source HDF5 重新复算，不继承旧 cycle 数量作为结论。输出每个 split、
source episode 和 controller epoch 的自然频率，并统计连续 `stay` run length。

组合覆盖不是 Gate。真正的停止条件是：

- `stay` 或 `adjacent` 只由单个 source episode 支持；
- 数据具有多个 controller epoch，但习惯只出现在其中一个；
- 相邻 return 动作没有可观察支持；
- dig-ready 的因果 detector 在 validation 上产生过早确认或明显失配。

频率不均匀是待保留的专家习惯，不做均匀重采样。

## 5. 观察充分性和 Condition 支持

在 `dump_end` 决策点比较瞬时观测与 0.5/1/2 秒因果历史，分别报告：

- dig-ready 与 dump-end phase 可辨识性；
- hindsight target / stay-adjacent 习惯可辨识性；
- 随后 0.5 秒专家 action 的最近邻 MAE 与有效方向一致性。

外部 lifecycle state 和 scripted target 始终是 runtime 合同的一部分。即使历史能猜中专家
频率，也不得把这种猜测当作 command。

反事实 condition 只在相同 current sector、不同合法相邻目标、离开 source episode 且
状态距离不超过 train 同意图最近邻 p95 时成立。无替代支持标为 `coverage_gap`，不计成功
或失败。审计同时报告：

- 全局频率先验；
- current-sector 先验；
- shuffled-target null；
- 状态历史预测。

这些先验越强，越说明未来必须用 B1 对 B2 的支持内配对实验，而不是说明模型看懂了
condition。

## 6. 候选固定场景

候选只来自 train source episodes：

- `repeat_same`：数据中真实、连续、同 sector 且至少两个 `stay` cycle 的 run；
- `move_adjacent`：数据中真实单步 `step_left` 或 `step_right`；
- 非相邻 jump 不进入候选。

所有候选均保留，不擅自挑固定数量。它们按 family 的 source-episode 支持、cycle 数、
边界置信度和 return action 支持排序，记录 episode、cycle、source row、VDS SHA、
相对意图、绝对目标、适用支持范围和不支持替代目标。用户确认后才能另建 frozen
scenario manifest；本轮产物始终保持 `user_review_required_not_frozen`。

## 7. 用户确认后的训练合同（本轮不执行）

第一轮只训练一个现有 ACT 执行器：

- dump 前 condition 分支不生效；
- `dump_end` 后脚本原子提交 next sector，condition 分支负责 return 和目标 dig-ready；
- 20 Hz、20-step chunk 保持不变；
- chunk 跨 cycle 时，loss mask 在当前 cycle 半开区间结束，禁止监督下一 cycle target；
- hindsight label 可以训练，但评测不得伪装成 recorded command；
- B0 无 condition、B1 正确支持内 condition、B2 matched shuffled condition；
- backbone、chunk horizon、temporal aggregation 和辅助 loss 均不同时改变。

若 B1 只能产生局部方向差异、不能到达目标，才单独预注册 endpoint/progress auxiliary
loss。该实验不得同时更改结构。

## 8. 测试与证据阶梯

单元和数据合同必须覆盖相对映射、边界越界、同区重新武装、事件顺序、缺失目标、
非相邻诊断、跨 cycle loss mask、episode split、无 privilege、无未来 runtime 输入、
command/outcome 分离和 provenance/SHA。

证据必须分层报告：

```text
offline replay
< nearest-neighbor stitching
< AGX closed loop
< physical digging effect
< real-machine evidence
```

低层证据不能替代高层。本轮最多给出
`accept / revise_boundary / revise_observation / collect_interventional_data` 的定义决策，
不会产生策略能力或闭环成功结论。

## 9. 本轮正式产物

```text
habit_transition_inventory_v1.json
dig_ready_boundary_audit_v1.json
habit_condition_support_v1.json
expert_habit_scenario_candidates_v1.json
definition_falsification_decision_v1.json
audit_manifest.json
checksums.sha256
```

每个产物记录 Git SHA、dirty state、source snapshot SHA、schema、参数、输入 episode 范围、
输出 SHA 和 `held_out_observation_read_count=0`。源 HDF5 不修改；输出目录不可覆盖。
