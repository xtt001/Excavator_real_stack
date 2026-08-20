---
type: experiment-execution-sequence-design
project: Excavator Real Stack
version: v2.0.1-real-transition-sequence-v2
status: implemented-offline-verified-field-prep-published-not-field-validated
created: 2026-08-14
updated: 2026-08-14
scope_authority: sequence-and-recording-profile
field_execution_authority: false
final_conclusion: docs/v2_0_1_real_transition_final_conclusion.md
implementation: testbed/testbed/tasks/real_transition.py
field_preparation: docs/v2_0_1_real_transition_field_preparation_and_calibration_runbook.md
---

# v2.0.1 Real Transition 实验执行序列设计

## 0. 结论

本轮不再使用固定 P0/P1 作为正式数采顺序。正式方案在录制前根据 seed 生成并冻结 24 条不同序列，共 96 个 cycle。四种原子 transition、3/4/5 三种 run 长度、初始侧、首目标和配对先后顺序都有显式平衡。

这套设计降低了模型把 cycle 位置、固定上下文或地形新旧当成目标提示的机会。它不能单独证明模型没有利用地形捷径。最终仍需 B0/B1/B2、同一 observation 下的目标干预和独立 source block 验证。

## 1. 要控制的捷径

固定序列会形成下面的相关性：

```text
地形和凹坑随 cycle 推进变化
cycle 位置又固定对应某个目标
模型可以根据地形或执行阶段猜目标
```

地形本身仍是必要 observation。ACT 需要根据土面决定如何挖、如何抬臂和何时卸料。实验只需要切断“看见某类地形就能确定目标 A/B”的稳定对应关系。

因此，目标分配遵守三条规则：

1. 序列生成器只接收 seed 和标识符，不读取现场画面、凹坑、实际落点或操作员选择；
2. `sequence_manifest.json` 在录制前冻结，现场不能根据地形换目标；
3. 目标在长度、位置、初始侧、原子 transition 和配对先后顺序上做对消。

## 2. 执行能力与首次数采范围分开

底层执行合同接受任意正整数长度的有限 A/B 目标序列，不把能力写死在 3–5 cycle。当前首次数采 profile 只授权 3、4、5 cycle，原因是这三个长度足以检查短序列衔接、长度变化和五铲连续性，同时还能在首批 96 cycle 内保持四类原子 transition 平衡。

后续如果要录制 6 次以上或自由组合，应发布新的 collection profile、平衡规则和配额，并增加相应的 manifest 授权校验。事件状态机、run package 和按长度计算的超时不需要回到固定四铲实现。

## 3. 冻结后的数量

完整目标包如下：

| 项目 | 数量 |
|---|---:|
| source block | 6 |
| 连续 run | 24 |
| cycle | 96 |
| 3 / 4 / 5 cycle run | 各 8 条 |
| train / validation / locked-test run | 16 / 4 / 4 |
| train / validation / locked-test cycle | 64 / 16 / 16 |
| `AA/AB/BA/BB` | 各 24 个 |
| train 中每种原子 transition | 16 个 |
| validation 中每种原子 transition | 4 个 |
| locked test 中每种原子 transition | 4 个 |

最小闭环包由前四个 block 构成，共 16 条 run、64 个 cycle：

- 两个 train block，共 32 cycle；
- 一个 validation block，共 16 cycle；
- 一个 locked-test block，共 16 cycle；
- 四种原子 transition 各 16 个。

后两个 train expansion block 再增加 32 cycle，四种原子 transition 各增加 8 个。64 和 96 都是首轮工程配额，不代表统计充分性或真机可部署结论。

## 4. Block 结构

| block 类型 | run 长度 | cycle 合计 | 用途 |
|---|---|---:|---|
| `train_short` | 3, 3, 4, 5 | 15 | 最小包或 train expansion |
| `train_long` | 3, 4, 5, 5 | 17 | 与 short 配成 32-cycle 平衡组 |
| `evaluation` | 3, 4, 4, 5 | 16 | validation 或 locked test |

每个 block 固定满足：

- A 起点两条，B 起点两条；
- 第一个 cycle 的 `AA/AB/BA/BB` 各一条；
- 前三个 cycle 位置上，目标 A/B 都是 2:2；
- 四条 run 分成两个 matched-start pair。每个 pair 共享初始侧，首目标分别为 A 和 B；
- pair 内两条 run 相邻执行，`matched_start_pair_member_rank` 明确记录谁先谁后。

两个 `train_short/train_long` block 配成一组后，四种原子 transition 各 8 个。validation 和 locked-test 各自也是四种原子 transition 各 4 个。

## 5. 配对顺序与地形新旧

只保证 pair 内有 A/B 两个目标还不够。如果 A 目标总是先执行，第一条 run 较新的土面仍会成为 A 的提示。

代码把 block 分为三个 pair-order balance group：

- 最小包中的两个 train block；
- 两个 train expansion block；
- validation 与 locked-test 两个 evaluation block。

对每一种初始侧，每组都有一个 block 按 A 目标优先，另一个按 B 目标优先。顺序在录制前冻结。这样“pair 中第一条/第二条”不能稳定预测目标。

现场仍要为每条 run 记录 `workface_reset_id` 和 `workface_action`。pair 应使用预先选定、尽量可比的工作条带或恢复方式。不能因为第一条执行后地形不理想，就临时交换第二条的目标或 sequence。不可比、不可达或不安全时终止并保留失败记录。

## 6. 单条序列约束

首批 profile 中的每条正式序列都满足：

- 长度为 3、4 或 5 cycle；
- 目标列表同时包含 A 和 B；
- 同时包含至少一个 stay 和一个 switch；
- session 内不重复；
- 旧版 P0/P1 不进入正式序列池，只能作为单独诊断样例。

这些是首批数据覆盖约束，不是 ACT 能力上限。它们避免某条 run 只反复执行一种目标或一种运动模式。

## 7. 运行时约束

运行时按 manifest 的实际长度完成，不再假设四个 cycle：

- 3/4/5 cycle 的 run stop 默认分别为 180/240/300 s；
- 录制上限为 15000 step，对应 50 Hz 下最长 300 s；
- 每个目标仍需在 cycle 开始前提交；
- `initial-ready` 和 `target-ready` 必须同时记录铲斗离土、操作员确认和稳定窗证据；
- realized side 由当前 swing qpos 自动计算并记录，不能人工输入；
- realized side 与 scripted target 不一致时立即 abort，不消费下一目标；
- `workface_reset_id` 和 `workface_action` 缺失时，run 在创建文件前被拒绝；
- 计划 sequence、pair 和 member rank 由 runtime 写入 run package，现场输入不能覆盖这些字段。

时间门槛来自既有历史数据和旧计划。上机前仍需根据当天液压、土面和操作节奏复核。放宽门槛必须发布新的 resolved config，不能改写已经冻结的 run。

## 8. 当前离线证据

截至 2026-08-14：

- real-transition 定向测试通过：14 passed，另有 5 个 subtest passed；
- 3、4、5 cycle package 均完成事件、HDF5 对齐、封存和 checksum round trip；
- 旧版 v1 P0 package 仍可只读验证，但不能用于新录制 runtime；
- 500 个不同 seed 的 manifest 生成审计全部通过；
- seed `20260814` 的样例 manifest 为 24 条唯一序列、96 cycle、四种原子 transition 各 24 个，并通过 pair-order 对消校验。

全量 `testbed/tests` 回归结果为 355 passed、1 skipped、9 个 subtest passed、7 failed。7 个失败位于 ACT deadzone、deadzone eval、go-home 和 runtime-gate 的既有测试路径，不在本次 real-transition 修改范围内，也与修改前记录的失败项一致。因此，本次 real-transition 定向链路通过，但仓库全量测试仍不是全绿状态。

这些结果只证明代码合同和离线生成器自洽。当前电脑没有连接现场 Jetson，也没有执行真实录制、操作员流程、传感器检查或真机动作。

## 9. 录制后必须做的捷径审计

sequence 平衡只是第一层控制。录制后还要按完整 source block 检查：

- `target_side` 与 cycle index、run rank、pair member rank、时间顺序、`workface_reset_id` 的关联；
- 初始画面或土面特征能否在不读取 condition 时预测目标；
- B0、B1、B2 在相同 split 和采样规则下的差异；
- 同一 observation 改变 `target_side_code` 后，返回方向是否产生符合目标的变化；
- 四种原子 transition 和 3/4/5 连续组合是否在独立 block 上成立。

如果仅凭 cycle index、pair rank 或地形特征就能高置信预测目标，这批数据不能支持 condition 因果结论。应先调整配对、工作条带和序列 profile，再决定是否补录。

## 10. 仍未完成的链路

本次实现覆盖 sequence manifest、split manifest、专家连续录制状态机、事件合同和 run package。以下部分仍未实现：

- raw run 到 ready-to-ready cycle 的离线 annotation/materializer；
- `real_transition_condition_v1` 训练数组；
- B0/B1/B2 训练和评估入口；
- N5 condition warm-start；
- conditioned-policy 真机 goal router 和分级 Gate。

因此，当前代码可以离线生成和检查新的录制计划，也具备专家录制支架。它还不能训练或执行最终的 Conditioned ACT，本文也不是上机授权。

真机同步、只读传感器检查、home/A/B 稳定窗采集和合同冻结按《[v2.0.1 真机测试与标定准备手册](v2_0_1_real_transition_field_preparation_and_calibration_runbook.md)》执行。该手册只授权准备阶段，不改变本文的正式录制证据边界。
