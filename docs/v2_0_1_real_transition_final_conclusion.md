---
type: final-scope-decision
project: Excavator Real Stack
version: v2.0.1-real-transition-scope-v2
status: accepted-recording-sequence-field-prep-code-aligned-not-field-validated
created: 2026-08-14
updated: 2026-08-14
discussion_record: docs/v2_0_1_real_transition_design_decision_log.md
previous_implementation_plan: docs/v2_0_1_real_transition_data_definition_and_recording_plan.md
sequence_design: docs/v2_0_1_real_transition_experiment_sequence_design.md
field_preparation: docs/v2_0_1_real_transition_field_preparation_and_calibration_runbook.md
scope_authority: true
field_execution_authority: false
---

# v2.0.1 Real Transition 最终结论

## 0. 最终决定

v2.0.1 真机阶段只回答一个问题：

> 在固定真机 context 和 A/B 两个 ready 区域下，Conditioned ACT 能否学会 `AA`、`AB`、`BA`、`BB` 四种完整原子 cycle，并根据外部逐-cycle 目标连续组合执行 3–5 个 cycle。

本轮不在真机上探索细网格、精确落点、感知规划、路径规划或混合决策控制闭环。这些长期方向先在仿真中建立系统证据，以后通过稳定接口接入真机。

## 1. A/B 与 ready

A 和 B 是以当天标定 home swing 方向为中线的两个已示教 ready 区域。它们是区域，不是固定角度或单一 qpos 点。

`Ready_i` 表示机器已处于第 `i` 铲可以自然开始的稳定姿态和作业状态。一个原子 cycle 定义为：

```text
Ready_i
→ 上层脚本提交本 cycle 目标 A/B
→ 挖掘
→ 运土
→ 卸料
→ 返回目标区域
→ Ready_(i+1)
```

目标必须在本 cycle 首个操作意图前提交。程序目标和实际到达结果分开记录，不得根据实际落点回填目标。

## 2. 四种原子 cycle

| 编号 | 名称 | 物理语义 |
|---:|---|---|
| 1 | `AA` | 从 A ready 开始挖，完成后回到 A ready |
| 2 | `AB` | 从 A ready 开始挖，完成后回到 B ready |
| 3 | `BA` | 从 B ready 开始挖，完成后回到 A ready |
| 4 | `BB` | 从 B ready 开始挖，完成后回到 B ready |

`stay` 和 `switch` 可以用于对称统计，不是原子 cycle 的权威名称。训练、数据覆盖和验收均按 `AA/AB/BA/BB` 四类分开计算。

## 3. 连续组合与本次数采范围

执行合同接受任意正整数长度的有限 A/B 目标序列，不把能力写死在 3–5 cycle。首次数采 profile 只录制 3、4、5 个连续 cycle。一条 run 只在开始时设定初始侧和重置一次 policy。中间不 go-home、不人工摆位、不重置 policy 或 temporal aggregation。

组合必须保持状态连续：前一 cycle 的 realized target 是后一 cycle 的 current side。例如：

```text
长度 3：A → A → B → A       = AA, AB, BA
长度 4：B → B → A → A → B   = BB, BA, AA, AB
长度 5：A → B → B → A → A → B = AB, BB, BA, AA, AB
```

上述只是语义例子，不是固定脚本。数采和验收使用多条合法组合，确保模型根据当前观测和本 cycle 目标执行，不依赖固定序列。以后扩展到 6 次以上或更长自由组合时，应发布新的 collection profile、数据配额和 manifest 授权校验；底层事件状态机与 run package 无需恢复成固定长度。

旧版 `P0=A→B→B→A→A` 和 `P1=B→A→A→B→B` 可保留为长度 4 的诊断样例。它们不再是唯一录制顺序，也不能代替 3–5 cycle 组合能力的验收。

## 4. 职责边界

- 上层脚本负责设定初始侧，并在每个 cycle 开始前提交下一 ready 目标 A/B。
- ACT 只完成已提交的 ready-to-ready cycle，不预测下一目标，不决定任务顺序。
- ready、goal lifecycle、安全停止和结果记录属于 runtime 合同，不允许由 ACT 隐式猜测。
- 人工示教时，师傅服从本 cycle 的 A/B 目标，其余摇杆幅值、时序和关节协调保持自然操作。

## 5. 验收边界

### 5.1 原子能力

- `AA/AB/BA/BB` 四种原子 cycle 分别在独立来源上完成；
- 每个 cycle 都包含可接受的挖掘、运土、卸料和返回；
- realized target 与 scripted target 一致，终点位于当天已示教支持范围内；
- 原有单铲能力没有明显退化。

### 5.2 Condition 能力

- 从相近 A ready 分别提交 A、B，结果应分别落到 A、B；
- 从相近 B ready 分别提交 A、B，结果应分别落到 A、B；
- 目标干预造成的差异必须大于同目标重复执行的自然波动；
- 固定序列成功不能单独证明 condition 被使用。

### 5.3 组合能力

- 3、4、5 cycle 长度均有独立组合验证；
- 验证序列不只使用 P0/P1；
- 中间不人工摆位、不重置 policy；
- 连续执行中的额外落点误差、物理效果退化和失败原因分开报告；
- 任一 cycle 失败时停止，不提前消费下一目标。

## 6. 本轮明确不做的内容

- 不在真机重复仿真已完成的细网格 condition 研究；
- 不训练精确地面坐标或固定点位控制；
- 不让 ACT 做上层顺序规划；
- 不在真机接入感知地图、路径规划或 ACT/传统控制路由；
- 不声称任意位置、跨场地泛化或无人值守；
- 不用本轮结果代替后续仿真系统级闭环研究。

## 7. 数据与实现要求

- 数采必须分开统计四种原子 cycle 和 3/4/5 三种组合长度；
- 数据划分继续按完整 source block 隔离，不在 cycle 级随机泄漏相邻数据；
- 不需穷举所有长度 3–5 的序列，但必须避免四种原子 cycle 或某一长度只出现在一条固定模板中；
- 新 sequence manifest 必须显式记录初始侧、每个 target、长度、原子 cycle 列表和 source split；
- 原始 run 连续录制，训练数据在离线复核后切成 ready-to-ready cycle；
- 正式 sequence 只由 seed 和标识符生成，在录制前冻结，不读取现场地形或实际落点；
- 每个 matched-start pair 的 A/B 首目标和执行先后都必须对消，现场记录 `workface_reset_id` 与处理方式；
- 当前代码中的完整目标 profile 固定为 24 条唯一 run、96 cycle。3/4/5 cycle run 各 8 条，train/validation/locked-test 为 64/16/16 cycle，`AA/AB/BA/BB` 各 24 个；现场 session 的 seed 和具体 sequence manifest 仍需在录制前生成并冻结；
- 前四个 block 构成 64-cycle 最小闭环包，四种原子 transition 各 16 个。后两个 train block 追加 32 cycle；
- 64/96 是首轮工程配额，不代表统计充分性。详细约束见《[实验执行序列设计](v2_0_1_real_transition_experiment_sequence_design.md)》。

## 8. 当前代码与最终结论的对齐状态

| 能力 | 当前 `fs/v2.0.1` | 对最终任务的意义 |
|---|---|---|
| A/B home-side contract | 已实现只读采集、生成与校验 | 可作为新任务基础，仍需当天真机标定 |
| 专家 raw run 录制、task event、HDF5 对齐和封存 | 已实现 | 主体可复用 |
| sequence manifest | 已升级为 seeded 多序列 v2 | 24 条唯一序列、96 cycle 和平衡约束已离线验证 |
| 专家录制 runtime | 已按 manifest 长度执行 | 支持可变 cycle 数、动态超时和 realized-target mismatch fail-closed |
| 旧版 P0/P1 v1 | 保留只读验证 | 不允许启动新的现场录制 |
| 离线 cycle annotation/materializer | 已实现 | 可生成带 condition、valid mask 和 provenance 的 20 Hz cycle |
| Conditioned ACT 训练与 B0/B1/B2 对照 | 未实现 | 不能训练目标模型 |
| 真机 conditioned-policy goal router/lifecycle | 未实现 | 不能闭环执行四种原子 cycle |
| 真机验证 | 未进行 | 不能声称现场可用 |

因此，sequence、专家录制和离线 cycle 数据链已与最终任务对齐；训练对照和
conditioned-policy runtime 仍未完成。每次正式录制仍需按《[v2.0.1 主从端现场启动与命令手册](v2_0_1_host_slave_start_commands.md)》通过当天硬件和相机门禁。
