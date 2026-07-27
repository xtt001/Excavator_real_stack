# SimVerify Observable Ready-to-Ready Cycle Contract v2（审阅草案）

状态：`draft_for_user_review`

证据范围：`definition_only_no_new_evidence`

训练授权：`false`

闭环成功声明：`false`

本文只定义任务、数据和运行时语义，不修改既有 M0/M1/M2/M3/M5 历史证据，不授权新训练、
held-out test、真机控制或部署。用户确认本文合理性后，才能把它冻结为新实验的上位合同。

## 1. 需要解决的理解偏差

已有研究计划已经把 cycle 写成 `ready_i -> ready_i+1`，但没有把以下边界放进同一个闭合
合同：

1. 原始录制 episode 的边界；
2. 可观察 cycle 的共享 ready 边界；
3. condition 的生效和切换边界；
4. ACT 训练 action chunk 的监督边界；
5. temporal aggregation 中旧 cycle 预测的失效边界；
6. 评测可以声明的能力边界。

如果这些边界不分开，容易把“模型能预测约 1 秒的局部专家动作”误解为“模型已经被直接
训练为完成一个完整挖掘循环”。本文禁止这种混用。

## 2. 第一阶段任务空间

第一阶段只使用横向离散 3x1 区域：

```text
left | center | right
```

这里的 sector 是下铲区域类别，不是精确三维铲尖目标，也不表示近/中/远。sector 数值边界
继续由 train/validation 的 dig-entry swing qpos 分布和 eye-pair 视觉特征交叉生成；不读取
held-out test，不继承仿真 planner 的 privileged cell。

“精准分区执行”在本阶段只表示：

> 循环结束时进入 condition 指定的 left/center/right 离散区域。

它不表示在某个 sector 内到达任意连续坐标。

## 3. 规范 cycle 定义

设：

- `B_i`：第 `i` 个可观察 ready 共享边界；
- `S_i ∈ {left, center, right}`：边界 `B_i` 对应的本轮下铲区域；
- `C_i`：第 `i` 个完整 cycle。

则：

```text
C_i = [B_i, B_i+1)

ready above S_i at B_i
  -> 在 S_i 完成本轮下铲和挖掘动作
  -> 提升、运送到固定卸料区并卸料
  -> 回转并接近 S_i+1
  -> 在 B_i+1 首次因果确认 ready above S_i+1
```

因此：

```text
current_sector(C_i)    = S_i
next_ready_sector(C_i) = S_i+1
```

例如：

```text
C_0: left ready  -> dig left   -> dump -> center ready
C_1: center ready -> dig center -> dump -> right ready
```

对应：

```text
C_0 condition metadata = left  -> center
C_1 condition metadata = center -> right
```

### 3.1 cycle 是任务管理原子，不是开环动作原子

一个 cycle 通常远长于一个 ACT action chunk。第一阶段仍保持：

```text
policy frequency = 20 Hz
action chunk horizon = 20 ticks ≈ 1 s
runtime execution = 每个 tick 只执行当前聚合得到的一步动作
```

模型在一个 cycle 内会被多次调用。本文不允许一次开环输出整个十几秒 cycle，也不把两个
cycle 合并成一个超长 action target。

### 3.2 转场归属

从卸料区回转到 `S_i+1` 上空的动作属于 `C_i`，不是两个 cycle 之间无人负责的空白，也不
属于 `C_i+1` 的开头。

`C_i+1` 从已经确认到达 `S_i+1` ready 后才开始，负责在 `S_i+1` 下铲，并最终到达
`S_i+2` ready。

### 3.3 同区域转移

`left -> left`、`center -> center`、`right -> right` 都是合法 cycle。即使 sector 没有改变，
也必须经历可观察的下铲、提升、卸料和新的 ready 边界，不能仅凭 sector 相同把相邻 cycle
合并。

### 3.4 完成、失败和停止

只有确认目标 `ready above S_i+1` 才能把 `C_i` 标为 `completed` 并启动下一 cycle。

以下情况结束的是一次失败执行尝试，不是成功的 cycle 边界：

- 超过 train/validation 生成的最大支持时长仍未到达目标 ready；
- 观测、动作或 condition 失效；
- 进入无数据支持或安全监督器禁止的状态；
- sector/ready 证据持续冲突；
- 外部监督器触发停止。

失败时不得把当前姿态伪标成 `B_i+1`，不得递增 `cycle_id`，也不得自动复用或切换到下一
condition。恢复、人工接管和 gohome 均在 ACT cycle 之外处理。

## 4. “ready above sector”的操作性定义

第一阶段不能使用真实铲尖世界坐标、精确土面、bucket mass、接触状态或 terrain grid。
因此 `ready above S_i` 定义为一个可观察准备包络，而不是几何真值：

1. **区域一致**：swing qpos 的 sector 与 eye-pair 视觉 sector 对 `S_i` 一致；
2. **局部构型一致**：boom/stick/bucket 的 qpos、qvel、action 候选与 stick-pair 冻结视觉
   特征都落入 train/validation 生成的 ready 支持包络；
3. **前序阶段一致**：除 episode 起始边界外，必须已经观察到前一 cycle 的 dump release，
   并离开卸料姿态；
4. **稳定但不静止**：允许非零缓慢 qvel，不要求所有关节完全停止；
5. **因果确认**：只使用当前及历史观测持续确认，不读取未来观测；
6. **置信度有效**：qpos/视觉冲突、区域边界样本或 ready 支持不足时标为 ambiguous，不
   强行生成边界。

ready 的所有数值阈值、dwell tick 数和置信度阈值只能从 train/validation 分布及
source-episode bootstrap 生成。本文不预设人工常数。

### 4.1 边界确认时刻

若 ready 需要连续 `K` 个 20 Hz tick 才能确认，`B_i` 定义为第 `K` 个 tick 到来后首次
可以因果确认 ready 的时刻，而不是回填到窗口的第一个 tick。

这样离线标注时刻和未来 runtime 可实现时刻具有相同的信息边界。允许额外保存 ready
候选区间的起点作为诊断，但它不能作为运行时已经知道的 condition 切换时刻。

`K` 本身仍须由 train/validation 稳定性生成，本文不冻结其数值。

现有 M0 v3 ready selector 使用区间两侧的离线视觉 halo 做确认，其代表帧不能直接视为
已经满足本节的 runtime-causal 边界。若本文获批，必须先审计能否只用当前及历史窗口复现
稳定边界；不能把既有离线代表帧原样改名为因果检测结果。

## 5. 共享边界和行归属

所有 cycle 范围使用半开区间：

```text
C_i rows   = [B_i, B_i+1)
C_i+1 rows = [B_i+1, B_i+2)
```

所以：

- `B_i+1` 是 `C_i` 的终点事件；
- `B_i+1` 这一观测行和从该行开始执行的动作归 `C_i+1`；
- `C_i` 的最后一个受监督动作位于 `B_i+1 - 1`；
- 相邻 accepted cycle 不重叠、不留一行空洞；
- sidecar 的 `source_steps` 和 `target_steps_20hz` 都必须明确为 `[start, end)`。

终点事件属于前一 cycle 的成功判据，但边界行属于下一 cycle 的训练数据。这两个说法不
冲突：前一行执行的动作把系统带到边界观测，边界观测随后启动下一任务。

## 6. sector 标签与 runtime 目标的来源必须分开

### 6.1 历史仿真数据

现有历史数据没有记录执行前 command，因此：

```text
command.current_sector    = unknown_not_recorded
command.next_ready_sector = unknown_not_recorded
condition_source          = hindsight_outcome
```

离线标注时：

- `S_i` 由 `B_i` 后实际发生的 `dig_entry_proxy` 所在 sector 确认；
- `S_i+1` 由下一 accepted cycle 的实际 `dig_entry_proxy` 所在 sector 确认；
- 最后一个没有后继 dig-entry 的 partial cycle 不能编造 `next_ready_sector`；
- hindsight 标签只能用于离线技术验证，不能伪装成当时真实下达的规划指令。

### 6.2 未来 runtime

runtime 中 `S_i+1` 必须来自显式外部规划/任务指令。ready detector 只负责确认机器是否
到达该目标，不得从未来实际下铲结果反推目标。

本阶段 planner 可以简单给出预设的 left/center/right 序列；不要求实现地形规划器。

## 7. condition 的语义和能力预期

保留数据 schema：

```text
cycle_condition_v1 =
  current_sector one-hot [3]
  + next_ready_sector one-hot [3]
```

但必须区分“cycle 元数据”和“当前数据能够识别的因果命令”：

- `current_sector` 描述 cycle 的起始前提：机器已在该 sector ready，随后在这里下铲；
- 在 hindsight 数据中，`current_sector` 已被当前图像和 qpos 强约束，不应被当作可任意
  swap 的反事实命令；
- `next_ready_sector` 是本轮循环的可控终点，是第一阶段需要验证的主要因果 condition；
- 第一阶段不要求 next sector 改变 dump 前的挖掘动作；
- next-sector condition 可以只在 observable `dump_end_proxy` 后影响 return/approach
  动作，从而避免破坏本轮下铲和卸料阶段。

因此近期合理的模型能力预期是：

> 从已经满足 `current_sector` ready 前提的状态开始，完成当前一铲，并在卸料后按
> `next_ready_sector` 选择 left/center/right 回转方向和终点。

不是：

> 从任意姿态接收任意 current/next pair，就能自行恢复、规划并完成任务。

condition 在语义 cycle 内保持不变。内部 phase router 可以控制 next-sector 分支何时对
动作生效，但不能把同一 cycle 改写成另一个 condition。

## 8. ACT 训练窗口的 cycle 边界

设 action chunk 长度 `H=20`，训练观测时刻为 `t0 ∈ [B_i, B_i+1)`。第 `j` 个 action
target 的 cycle 有效掩码定义为：

```text
cycle_action_valid_mask[j] = (t0 + j < B_i+1),  j = 0 ... H-1
```

训练 action loss 使用：

```text
effective_action_loss_mask
  = non_padding_mask
  ∩ existing_action_loss_mask
  ∩ cycle_action_valid_mask
```

规则如下：

1. condition 取 `C_i` 的 condition，并在该样本内固定；
2. 到达 `B_i+1` 之前的 action target 正常监督；
3. `B_i+1` 及之后属于新 cycle 的 action target 必须 pad/mask；
4. 不允许用旧 condition 监督新 cycle 的未来动作；
5. 不应简单删除所有靠近边界的 `t0`，因为这样会丢失 runtime 最需要的终点附近纠偏状态；
6. mask 后至少一个 action tick 有效的 `t0` 才能作为训练样本；
7. 统计必须报告每种转移的有效 `t0` 数、有效 action tick 数和短掩码窗口比例，不能只
   报 cycle 数量。

这里的 mask 解决的是 condition 跨 cycle 污染，不是 task-success loss。是否增加终点
sector、ready 或 progress 辅助 loss，必须在这份数据语义冻结后作为新的单因素实验定义。

## 9. runtime condition 切换和 temporal aggregation

在 `B_i+1` 被因果确认时，runtime 必须原子完成：

1. `cycle_id: i -> i+1`；
2. condition 从 `[S_i, S_i+1]` 更新为 `[S_i+1, S_i+2]`；
3. phase router 重新从 current phase 开始；
4. 日志记录边界观测、确认依据、旧/新 condition 和切换 tick；
5. 若计划序列已经按要求完成，则在该 ready 边界正常停止 cycle policy，由外部监督器决定
   是否调用 gohome；
6. 若任务要求继续但缺少 `S_i+2` 指令，则 fail closed，不复用旧目标。

### 9.1 本草案建议：aggregation 按 cycle 隔离

旧 cycle 产生的 raw chunks 带有旧 condition。若在 `B_i+1` 后继续参与 temporal
aggregation，它们会与新 condition 的 chunk 混合，导致 condition 语义不再原子。

本草案建议：

- 不重置全局 policy tick 和模型权重；
- 不把整个 runtime 当作重新启动；
- 但在共享 ready 边界后，旧 `cycle_id` 的 chunk 不再进入 action aggregation；
- 新 cycle 的第一个 runtime-safe action 只由新 `cycle_id` 的有效预测产生；
- 边界动作连续性由 ready 状态和单独的安全变化率约束验证，不靠混入旧 condition 的动作
  来获得。

这会修订当前 G5.1“跨边界保留全部 temporal aggregation chunks”的历史实验语义，因此
目前只是待审阅定义，尚未冻结或实现。历史 G5.1 结果保持不变。

## 10. episode 头尾和不完整 cycle

以下数据不能作为完整 conditioned cycle：

- 录制开始时已经处于下铲、运送或卸料中，没有可信 `B_i`；
- 录制结束前没有确认 `B_i+1`；
- 缺少下一实际 dig-entry，无法形成 hindsight `S_i+1`；
- 任一关键事件顺序不完整；
- ready 或 sector 证据冲突；
- 相邻 cycle 的共享边界不一致；
- `next_ready_sector(C_i) != current_sector(C_i+1)`。

这些片段可以保留用于结构 QC 或其他明确的局部实验，但：

- `valid_mask=0`；
- `cycle_id=-1`；
- condition 全零；
- 不进入完整 cycle、transition 或连续 cycle 的成功分母。

## 11. 能力声明分级

| 级别 | 必须证明什么 | 不能据此声称什么 |
| --- | --- | --- |
| L0 局部模仿 | 约 1 秒 action chunk 在 recorded observation 上可预测 | 完整 cycle |
| L1 单循环阶段 | 下铲、提升、卸料、return、ready 等可观察阶段完整且有序 | 按 condition 到目标 |
| L2 单循环落区 | 正确 condition 相对 shuffled null 能使 cycle 结束于指定 next sector | 连续多循环 |
| L3 连续循环 | 多个共享 ready 边界均正确切换并完成指定 sector 序列 | 铲到土或真实生产效果 |
| L4 仿真物理任务 | 在仿真闭环中同时证明可观察序列和独立物理效果 | 真机泛化 |
| L5 真机候选 | 真实数据域离线、shadow 和安全链路逐级通过 | 自动部署或生产控制 |

补充限制：

- teacher-forced replay、MAE、raw chunk 相似只能支持 L0/L1 的部分证据；
- 最近邻/transition stitching 只能支持数据流形内的离线 progress 证据；
- AGX probe 才能产生 sim closed-loop diagnostic，但不能自动证明真实挖土成功；
- bucket mass 等 privilege 只能做物理隔离的 post-hoc oracle audit，不能改变 observable
  主结果；
- 没有真实 command 的 hindsight 数据不能单独证明任意反事实 condition 泛化。

## 12. 本定义下的近期验收问题

新实验必须逐项回答：

1. 在不提供 next-sector condition 时，模型能否完成可观察的单 cycle 阶段？
2. 提供 `next=left/center/right` 后，dump 前动作是否保持任务阶段，dump 后动作是否产生
   正确且可重复的方向差异？
3. cycle 是否在指定 sector 的 observable ready 首次确认处结束？
4. 正确 condition 是否稳定优于 matched shuffled-condition null？
5. `left->left` 等同区转移是否仍能形成独立完整 cycle？
6. 在 `left->center->right` 等连续序列中，边界行、condition、phase router 和 action
   aggregation 是否按相同 `cycle_id` 原子切换？
7. 一旦离开 train/validation 支持，评测是否明确停止而不是最近邻强行续接？

只有 L2 通过后才讨论 L3；L3 通过仍不等于仿真挖土成功或真机候选。

## 13. 与当前实现/历史合同的差异

| 项目 | 当前状态 | 本草案 |
| --- | --- | --- |
| cycle 语义 | 研究计划已有 ready-to-ready 方向 | 给出半开区间、共享边界和行归属 |
| ready 确认 | M0 v3 使用双侧离线视觉 halo 选择代表帧 | runtime 边界必须只用当前及历史观测确认 |
| condition sidecar | accepted cycle 内固定，边界处切换 | 保留 |
| policy 因果目标 | B1.4/B1.5 主要验证 next sector | 明确 next sector 是近期主要可控命令 |
| action chunk | 可以从有效行起始并跨相邻 accepted cycle | 跨边界 action target 必须 mask |
| 边界附近 t0 | 可进入训练，但后续动作可能来自新 cycle | 保留 t0，仅 mask 新 cycle targets |
| runtime phase router | shared ready 后重置 route | 保留 |
| temporal aggregation | G5.1 跨边界保留旧 chunks | 建议按 cycle_id 排除旧 chunks |
| 成功口径 | 离线 Gate 分散在多个合同 | 用 L0-L5 明确声明边界 |

## 14. 请用户确认的六个定义决策

在实现或训练前，需要明确接受或修改以下六项：

1. **任务边界**：一个 cycle 是 `当前下铲区 ready -> 下一下铲区 ready`。
2. **“上空”含义**：近期只表示 sector 对齐加 observable ready 包络，不表示精确三维点。
3. **共享边界归属**：使用 `[B_i, B_i+1)`，边界观测行归下一 cycle。
4. **condition 能力**：保留 current+next 元数据，但近期主要因果命令是 `next sector`。
5. **训练跨界规则**：保留边界附近观测，mask 掉属于下一 cycle 的 action targets。
6. **aggregation 规则**：边界后旧 cycle chunks 失效，新 cycle 不混入旧 condition 预测。

任何一项修改都应先更新本文，再设计数据重物化、训练或闭环实验。
